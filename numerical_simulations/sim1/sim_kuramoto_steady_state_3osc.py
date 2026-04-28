# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:46:47 2024

@author: H.Yokoyama
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append(current_path)

import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 20 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 2.0
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300 
#%%
from my_modules import my_timeseries_graph_plot as my_tp

from copy import deepcopy
from joblib import Parallel, delayed
from scipy.linalg import expm
from lingam import var_lingam

import lingam
import numpy as np
import networkx as nx
#%%
def func_kuramoto(x, K, omega):
    Nosc = x.shape[0]
    dxdt = np.zeros(x.shape)
    for i in range(Nosc):
        phase_dynamics = 0
        for j in range(Nosc):
            if i!=j:
                phase_diff = np.mod(x[j]-x[i], 2*np.pi)
                phase_dynamics += K[i,j] * np.sin(phase_diff)
        
        dxdt[i] = omega[i] + phase_dynamics/Nosc
    return dxdt

                
def get_jacobian_matrix(phs_diff_eq, K, omega):
    # convert vector to matrix 
    Nosc = len(omega)
    if (np.ndim(phs_diff_eq)==1) & (len(phs_diff_eq)==int(Nosc*(Nosc-1))):
        phs_diff_mat = np.ones((Nosc, Nosc)) - np.eye(Nosc)
        phs_diff_mat[phs_diff_mat==1] = phs_diff_eq
        
    elif np.ndim(phs_diff_eq)==2:
        phs_diff_mat = phs_diff_eq
        
    
    cnt  = 0
    J    = np.zeros((Nosc,Nosc))
    for i in range(Nosc):
        for j in range(Nosc):
            if i==j:
                diff   = phs_diff_mat[i,:]
                J[i,j] += -K[i,:]@np.cos(diff)
            else:
                diff    = phs_diff_mat[i,j]
                J[i,j] += K[i,j]*np.cos(diff)
            
            cnt += 1
    return J/Nosc


def runge_kutta(h, func, theta_now, K, omega):
    k1=func(theta_now, K, omega)#omega+K*np.sin(theta_now[::-1]-theta_now)
    
    theta4k2=theta_now+(h/2)*k1
    k2=func(theta4k2, K, omega)#omega+K*np.sin(theta4k2[::-1]-theta4k2)
    
    theta4k3=theta_now+(h/2)*k2
    k3=func(theta4k3, K, omega)#omega+K*np.sin(theta4k3[::-1]-theta4k3)
    
    theta4k4=theta_now+h*k3
    k4=func(theta4k4, K, omega)#omega+K*np.sin(theta4k4[::-1]-theta4k4)
    
    theta_next=theta_now+(h/6)*(k1+2*k2+2*k3+k4)
    theta_next=np.mod(theta_next, 2*np.pi)
    
    return theta_next


def euler_maruyama(h, func, theta_now, 
                   K, omega, 
                   noise_scale=0.001):
    dt = h
    p  = noise_scale
    dw = np.random.randn(theta_now.shape[0])
    
    theta      = theta_now + func(theta_now, K, omega) * dt
    theta_next = theta + p * np.sqrt(dt) * dw
    theta_next = np.mod(theta_next, 2*np.pi)
    return theta_next


def solve_kuramoto(solver, func_kuramoto, theta0, Nt, Nosc, kappa, omega):
    theta       = np.zeros((Nt, Nosc))
    theta[0, :] = theta0

    for t in range(1, Nt):
        theta_now  = theta[t-1, :] 
        theta_next = solver(h, func_kuramoto, theta_now, kappa, omega)
        theta[t, :] = theta_next.reshape(1, Nosc)
    
    return theta, theta0

def stability(func_jaco, theta_eq, K, omega):
    jaco = func_jaco(theta_eq, K, omega)
    
    vals   = np.linalg.eig(jaco)
    eigval = vals[0]
    eigvec = vals[1]
    
    label  = ['Stable focus', 
              'Stable node',
              'Saddle',
              'Unstable fixed point (Saddle-Node)',
              'Center (Hopf)',
              'Unstable node',
              'Unstable focus']
    
    # [0:'Stable focus', 1:'Stable node',   2:'Saddle', 3:'Transcritical (Saddle-Node)', 
    #  4:'Center (Hopf)',5:'Unstable node', 6:'Unstable focus']
    
    
    determinant = np.linalg.det(jaco)
    trace = np.matrix.trace(jaco)
    
    ##############
    if all(np.imag(eigval)==0):
        if all(np.real(eigval)>0):
            # 5:'Unstable node';
            eqpt_idx = 5;
        elif all(np.real(eigval)<0):
            # 1:'Stable node';
            eqpt_idx = 1;
        else:
            if np.isclose(determinant, 0, rtol=1e-08, atol=1e-12):
                # nature = "Transcritical (Saddle-Node)"
                eqpt_idx = 3
            elif determinant < 0:
                # nature = "Saddle"
                eqpt_idx = 2
    else:
        if all(np.real(eigval)<0):
            # 0:'Stable focus';
            eqpt_idx = 0
        elif any(np.real(eigval)>0):
            # 6:'Unstable focus';
            eqpt_idx = 6;
        elif all(np.isclose(np.real(eigval), 0, rtol=1e-08, atol=1e-12)):
            # 4:'Center (Hopf)';
            eqpt_idx = 4;
    
    lyap = np.exp(abs(np.real(eigval)).max())
    
    return label[eqpt_idx], lyap, eigval

def vis_directed_graph(K, vmin, vmax, seed, pos=None, node_size_weight=None, cbar = True, ax=None):
    import matplotlib as mpl
    if ax == None:
        ax = plt.gca()
    # plt.figure(figsize=(6, 5))
    im_ratio = 5 / 6
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]
    
    G      = nx.from_numpy_array(K, create_using=nx.MultiDiGraph())
    G.edges(data=True)
    
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
    
    labels = {i : i for i in G.nodes()}          
    
    node_sizes  = [1000  for i in range(len(G))]
    if node_size_weight is not None:
        node_sizes  = node_size_weight * node_sizes 
        
    M           = G.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1] = 1
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue', ax=ax)
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.09',
                                         arrowsize=10, edge_color=edge_colors,
                                         width=4,
                                         edge_vmin=vmin, edge_vmax=vmax, ax=ax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w', ax=ax)
    plt.axis('equal')
    # set alpha value for each edge
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
        
    for i in range(M):
        if vmin < 0:
            c_idx = int((weight[i]/vmax + 1)/2 * cm.N)
        elif vmin>=0:
            c_idx = int((edge_alphas[i] * cm.N))
            
        rgb = np.array(cm(c_idx))[0:3]
        # edges[i].set_alpha(edge_alphas[i])
        edges[i].set_color(rgb)
    
    pc = mpl.collections.PatchCollection(edges, cmap=cm)
    pc.set_array(edge_colors)
    pc.set_clim(vmin=vmin, vmax=vmax)
    
    ax.set_axis_off()
    
    if cbar:
        plt.colorbar(pc, ax=ax, label='coupling strength (a.u.)', 
                     fraction=0.05*im_ratio, pad=0.035)
    
    return edges, pos


def solve_steady_state_kuramoto(omega, kappa, dt, noise_scale=np.sqrt(0.5), Ndelta=50):
    Nosc = len(omega)
    
    prob_phi = np.zeros((Ndelta, Nosc, Nosc))
    phi_st   = np.zeros((Nosc, Nosc))
    
    for i in range(Nosc):
        
        for j in range(i, Nosc):
            if i != j:
                phi_plot, prob_fk = solve_fokker_planck_2osc(omega[i], omega[j], 
                                                             kappa[i,j], kappa[j,i],  
                                                             dt, noise_scale, Nosc, 
                                                             N=Ndelta)
                prob_phi[:,i,j] = prob_fk[:,0]
                prob_phi[:,j,i] = prob_fk[:,1]
                
                phi_st[i,j] = phi_plot[prob_fk[:,0].argmax()]
                phi_st[j,i] = phi_plot[prob_fk[:,1].argmax()]
                
    return phi_st, phi_plot, prob_phi

def check_estimated_equilibrium_points(theta, phi_diff_eq, omega, kappa, dt, 
                                       title, phi_diff_fk, prob_fk,
                                       Ndelta=50, noise_scale=np.sqrt(.5)):
    Nosc = len(omega)
    
    fig = plt.figure(constrained_layout = False, figsize=(12, 3));
    plt.subplots_adjust(wspace=0.5, top=0.7);
    
    gs  = fig.add_gridspec(1, Nosc)
    
    fig.suptitle(title, fontsize=20)
    
    cnt = 0
    for i in range(Nosc):
        
        for j in range(i, Nosc):
            if i != j:

                plt.subplot(gs[0,cnt])
                phi_diff_emp = np.mod(theta[:,i]-theta[:,j], 2*np.pi)
                n_plt, bins_plt, patches = plt.hist(phi_diff_emp, 
                                                    bins=18, 
                                                    range=[0, 2*np.pi], 
                                                    color='whitesmoke', 
                                                    edgecolor='black',
                                                    density=True, 
                                                    label='emprical')
                
                plt.xlabel('$\\theta_{%d} - \\theta_{%d} $'%(i,j))
                
                plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
                
                #### plot equilibrium ponits
                eq_point = phi_diff_eq[j,i]
                plt.plot([eq_point, eq_point], [0, 1.1], 
                         c='r', linestyle='--', 
                         label='equilibrium point')
                
                #### plot steady state density of phase difference
                plt.plot(phi_diff_fk, prob_fk[:,j,i], c='b', label='analytical')
                
                plt.ylim(0, 1)
                if (i==Nosc-2) & (j==Nosc-1):
                    plt.legend(bbox_to_anchor=(1.05, 1), 
                               loc='upper left', 
                               borderaxespad=0)
                elif cnt==0:
                    plt.ylabel('density')
                
                cnt += 1
    return fig

          
def solve_fokker_planck_2osc(omega1, omega2, k1, k2, dt, noise_scale, Nosc, N=100):
    _, _      = kappa.shape
    t         = np.arange(0, 1000+dt, dt)
    Nt        = len(t)
    phi_diff  = np.linspace(0, 2*np.pi, N)
    D         = (noise_scale*np.sqrt(dt))
    
    G21 = omega2 - omega1 - ((k2 + k1)/Nosc) * np.sin(phi_diff) # d(θ2-θ1)/dt  
    G12 = omega1 - omega2 - ((k1 + k2)/Nosc) * np.sin(phi_diff) # d(θ1-θ2)/dt 
    
    P = np.zeros((Nt, N, 2))
    
    delta    = (2*np.pi-0)/N
    P[0,:,0] = np.ones(N)
    P[0,:,1] = np.ones(N)
    for t in range(1, Nt):
        P21_now = P[t-1,:,0]
        P12_now = P[t-1,:,1]
        
        dP21    = np.zeros(N)
        dP12    = np.zeros(N)
        for i in range(1, N-1):
            term1   = P21_now[i]* ((G21[i+1] - G21[i-1])/(2*delta)) # numerical approximation with center difference
            term2   = G21[i] * ((P21_now[i+1] - P21_now[i-1])/(2*delta)) # numerical approximation with center difference
            term3   = D * ((P21_now[i+1] - 2*P21_now[i] + P21_now[i-1])/delta**2) # numerical approximation of second derivative
            dP21[i] = -term1 - term2 + term3 # dP(θ2-θ1)/dt
            
            term1   = P12_now[i]* ((G12[i+1] - G12[i-1])/(2*delta)) # numerical approximation with center difference
            term2   = G12[i] * ((P12_now[i+1] - P12_now[i-1])/(2*delta)) # numerical approximation with center difference
            term3   = D * ((P12_now[i+1] - 2*P12_now[i] + P12_now[i-1])/delta**2) # numerical approximation of second derivative
            dP12[i] = -term1 - term2 + term3 # dP(θ1-θ2)/dt
        
        P[t,:,0] = P21_now + dP21*dt # numerical integral with Euler method
        P[t,:,1] = P12_now + dP12*dt # numerical integral with Euler method
        
        error = abs(P[t,:,0] - P21_now).mean() + abs(P[t,:,1] - P12_now).mean() 
        if error <= 1E-9:#1E-8: #1E-6:
            P = P[:t+1,:,:]
            break
    
    
    # normalization
    prob = np.array([P[-1,:,i]/(P[-1,:,i].sum() * delta) for i in range(2)]).T
    return phi_diff, prob
#%%
fig_dir = current_path + '/figures/jVARLiNGAM_steady_state/'
if os.path.exists(fig_dir)==False:  # Make the directory for data saving
    os.makedirs(fig_dir)
    
# anglular velocity
np.random.seed(100)

time        = 90#100# measurement time
h           = 1/100 #0.01    # micro time
Nt          = int(time/h)

Nosc        = 3
# K           = np.array([[.0, .2, .1],
#                         [1., .0, .0],
#                         [1., 1., .0]])

K           = np.array([[.0,  .3, .1],
                        [1.,  .0, .0],
                        [.9,  1., .0]])


causal_orders = []
order_param   = []
stab_label    = []
lyap_val      = []
eigval        = []
theta_save    = []
TE_save       = []
TEC_save      = []
B_est         = []
eq_points     = []
x_save        = []

prob_all      = []

scale         = np.hstack((0.1, np.arange(0.2, 4.2, .2)))
omega         = 2*np.pi*np.random.normal(loc=10, scale=np.sqrt(5E-3), size=Nosc) ##

cnt = 0
for gain in scale:
    #%%
    print(gain)
    kappa   = gain * K 

    # phi_st, phi_plot, prob_phi = solve_steady_state_kuramoto(omega, kappa, h, 
    #                                                          noise_scale=np.sqrt(0.5), 
    #                                                          Ndelta=50)
    
    phi_st, phi_plot, prob_phi = solve_steady_state_kuramoto(omega, kappa, h, 
                                                             noise_scale=1, 
                                                             Ndelta=36)
    
    # x_ini      = np.random.normal(loc=np.pi, scale=np.sqrt(1E-2), size=Nosc) 
    x_ini      = np.random.uniform(low=0, high=np.pi, size=Nosc) 
    # results    = solve_kuramoto(runge_kutta, func_kuramoto, x_ini,
    #                             Nt, Nosc, kappa, omega)
    results    = solve_kuramoto(euler_maruyama, func_kuramoto, x_ini,
                                Nt, Nosc, kappa, omega)
    
    theta    = results[0]
    #%% ### Calculate Kuramoto order parameters

    theta_mean = np.mod(np.angle(np.mean(np.exp(1j*theta),axis=1)), 2*np.pi)
    theta_diff = np.mod(theta_mean[:,np.newaxis]-theta, 2*np.pi)
    r = abs(np.mean(np.exp(1j*theta_diff),axis=1)).mean()
    
    
    # theta_mean = np.mod(np.angle(np.mean(np.exp(1j*theta),axis=1)), 2*np.pi)
    # theta_diff = np.mod(theta - theta_mean[:,np.newaxis], 2*np.pi)
    # r_local    = np.zeros(Nosc)
    # d_local    = K.sum(axis=1)
    # for n in range(Nosc):
    #     r_local[n] = K[n,:]@abs(np.mean(np.exp(1j*theta_diff), axis=0))
        
    # r = r_local.sum()/d_local.sum()
    
    #%% #### Apply Jacobian-informed VARLiNGAM
    J  = get_jacobian_matrix(phi_st, kappa, omega)
    M  = expm(J*h)
    
    VARLiNGAM = var_lingam.VARLiNGAM(lags=1,
                                     prune=False,
                                     ar_coefs=M[np.newaxis,:,:], 
                                     lingam_model=lingam.ICALiNGAM()
                                     )

    VARLiNGAM.fit(np.unwrap(theta.T).T)
    # B_est_ = VARLiNGAM.adjacency_matrices_
    # B0_est = B_est_[0]
    # B_est  = B_est_[1]
    
    # TE     = np.zeros((2, Nosc, Nosc))
    # for i in range(Nosc):
    #     for j in range(Nosc):

    #         TE[0,i,j] = VARLiNGAM.estimate_total_effect(np.unwrap(theta.T).T, 
    #                                                     i, j, from_lag=0)
    #         TE[1,i,j] = VARLiNGAM.estimate_total_effect(np.unwrap(theta.T).T, 
    #                                                     i, j, from_lag=1)
    
    TE     = np.zeros((2, Nosc, Nosc))
    causal_order = np.array(VARLiNGAM.causal_order_)
    for i in range(Nosc):# from index (source)
        order_i = np.where(causal_order==i)[0][0]
        for j in range(Nosc): # to index (reference)
            order_j = np.where(causal_order==j)[0][0]
            
            if order_i<order_j:
                TE[0,i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=0)
                TE[1,i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=1)
    
    TEC = abs(TE).sum(axis=2)
    Adj_mat = VARLiNGAM.adjacency_matrices_
    #%% #### Conduct Stability analysis
    out = stability(get_jacobian_matrix, phi_st, kappa, omega)
    label = out[0]
    lyap  = out[1]
    eigs  = out[2]
        
        
    
    order_param.append(r)
    stab_label.append(np.array(label))
    lyap_val.append(np.array(lyap))
    eigval.append(eigs)
    prob_all.append(prob_phi)
    
    B_est.append(np.array(Adj_mat))
    TE_save.append(np.array(TE))
    TEC_save.append(TEC)
    eq_points.append(phi_st)
    x_save.append(theta)
    
    
    causal_orders.append(causal_order)
    cnt += 1
    
    print('K = %.2f : %s'%(gain, label))
#%%
stab_label=np.array(stab_label)
lyap_val = np.array(lyap_val)
prob_all = np.array(prob_all)

B_est    = np.array(B_est)
TE_save  = np.array(TE_save)
TEC_save = np.array(TEC_save)


#%%
eig_max = np.zeros(scale.shape)
for i, eig in enumerate(eigval):
    eig_max[i] = np.real(eig[abs(np.real(eig)).argmax()])
#%%

Klist = np.linspace(0,scale.max(),1000)


# ### gaussian case
# SD    = np.sqrt(((omega - omega.mean())**2).sum()/(Nosc))
# Kc    = (np.sqrt(8/np.pi)*SD)/abs(np.linalg.eig(K)[0].max())
# r     = np.sqrt(1-Kc/Klist)
# r[Kc>Klist]=0

# eig_max = np.real(np.linalg.eig(K)[0]).max()
# g0      = 1/(np.pi)
# g0_dev2 = -2/(np.pi)

############
g0      = 1/(np.pi)
d_mean  = K.sum(axis=1).mean()
Kc      = 2/(g0 * np.pi * d_mean)

# g0      = 1/(np.pi)
# d_mean  = K.sum(axis=1).sum()
# d2_mean = (K.sum(axis=1)**2).sum()
# Lambda  = d2_mean/d_mean
# Kc      = 2/(g0 * np.pi * Lambda)

# alpha   = np.sqrt(-16/(g0_dev2*np.pi*Kc**3))
# r       = alpha*np.sqrt(1-Kc/Klist)
# r[Kc>Klist]=0

plt.figure(figsize=(4, 3))
plt.scatter(scale, np.array(order_param), 
            zorder=3, c='c', edgecolors='k', alpha=0.7, 
            label='emprical')
plt.plot(scale, np.array(order_param), 
         zorder=2, c='c')
# plt.plot(Klist, r, 'r',zorder=0, 
#           label='analytical')
plt.plot([Kc,Kc], [0.2, 1.], 'k--',
         zorder=1, 
         label='$K_c$')
plt.xlabel('Coupling strength $K$')
plt.ylabel('Order parameter $r$')
plt.legend(bbox_to_anchor=(1.05, 1), 
           loc='upper left', 
           borderaxespad=0)
plt.ylim(.25, 1.)
plt.savefig(fig_dir + 'kuramoto_order.png', bbox_inches="tight")
plt.savefig(fig_dir + 'kuramoto_order.svg', bbox_inches="tight")
plt.show()

#%% visualize time-series graph
# vmin = -3.1
# vmax = +3.1

vmin = -1
vmax = +1
cnt = 0

plt.rcParams["font.size"] = 24 # 全体のフォントサイズが変更されます。
for B, causal_order, TEC in zip(B_est, causal_orders, TEC_save):
    B0   = B[0]
    Btau = B[1]
    ##%%
    fig_path = fig_dir + '/scale_%4.2f/'%scale[cnt]
    
    print(fig_path)
    if os.path.exists(fig_path)==False:  # Make the directory for data saving
        os.makedirs(fig_path)
    ######### visualize graph structure (time-series graph)
    # fig, axes = plt.subplots(1,1, figsize=(8,10))
    
    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(2,2)
    # plt.subplots_adjust(wspace=0.3, top=0.75)
    plt.subplots_adjust(wspace=.6, hspace=.75, top=0.8)
    
    
    ##### Plot time-series graph
    val_matrix = np.zeros((Nosc,Nosc,2))
    graph      = np.zeros((Nosc,Nosc,2),dtype='<U3')
    
    
    val_matrix[:,:,0] = B0[np.ix_(causal_order,causal_order)].T
    val_matrix[:,:,1] = Btau[np.ix_(causal_order,causal_order)].T

    graph[val_matrix!=0] = '-->'
    
    ax = plt.subplot(gs[0:2, 0])
    my_tp.plot_time_series_graph(graph = graph, 
                                 val_matrix=val_matrix,
                                 var_names=np.arange(Nosc)[causal_order],#var_names=ch_names,# 
                                 vmax_edges=vmax,
                                 vmin_edges=vmin,
                                 link_colorbar_label='causal effect\n($j$-VARLiNGAM)',
                                 alpha=.6,
                                 node_size=.05,
                                 curved_radius=0.35,
                                 arrow_linewidth=10,
                                 label_fontsize=28,
                                 tick_label_size=18,
                                 edge_ticks=1,
                                 fig_ax=(fig, ax)
                                 )
    ########## total effect centrality###########
    fig.suptitle('$\\kappa = %3.1f$ \n total effect centrality'%(scale[cnt]))
    ##### total effect centrality (instantaneous)
    plt.subplot(gs[0, 1])
    TEC0 = abs(TEC[0,:])
    TEC1 = abs(TEC[1,:])
    
    
    plt.bar(np.arange(Nosc), TEC0/TEC0.max())
    plt.plot([-0.5, Nosc+0.5],[0,0],'k--')
    plt.xlim(-0.5, Nosc-0.5)
    plt.ylim(-.1, 1.2)
    plt.xlabel('# Node')
    plt.ylabel('normalized TEC (a.u.)')
    plt.xticks(ticks=np.arange(Nosc))
    plt.title('instantaneous')

    ##### total effect centrality (lagged)
    plt.subplot(gs[1, 1])
    plt.bar(np.arange(Nosc), TEC1/TEC1.max())
    plt.plot([-0.5, Nosc+0.5],[0,0],'k--')
    plt.xlim(-0.5, Nosc-0.5)
    plt.ylim(-.1, 1.2)
    plt.ylabel('normalized TEC (a.u.)')
    plt.xlabel('# Node')
    plt.xticks(ticks=np.arange(Nosc))
    plt.title('lagged')
    
    plt.savefig(fig_path + 'causal_effect_centrality.png', bbox_inches="tight")
    plt.savefig(fig_path + 'causal_effect_centrality.svg', bbox_inches="tight")
    plt.show()
    
    cnt += 1

#%% Visualize estimation results
cnt  = 0
seed = 20
pos  = []
for B, TEC, x, phi_diff_eq, prob_phi in zip(B_est, TEC_save, x_save, eq_points, prob_all):
    #%%
    kappa = scale[cnt] * K
    

    ##%% ######################################################################
    fig_path = fig_dir + '/scale_%4.2f/'%scale[cnt]
    if os.path.exists(fig_path)==False:  # Make the directory for data saving
        os.makedirs(fig_path)
    
    B0   = B[0]
    Btau = B[1]

    theta = x
    
    title = '   $\\kappa = %3.1f$\n%s'%(scale[cnt], stab_label[cnt])
    fig   = check_estimated_equilibrium_points(theta, phi_diff_eq, 
                                               omega, kappa, h, 
                                               title, phi_plot, prob_phi)
    plt.savefig(fig_path + 'equilibrium_points.png', bbox_inches="tight")
    plt.savefig(fig_path + 'equilibrium_points.svg', bbox_inches="tight")
    plt.show()
    ##%% ######################################################################
    
    #%%
    #### plot heatmap of connectivity matrix
    fig, axes = plt.subplots(figsize=(20, 9), 
                             ncols=4, nrows=2,
                             gridspec_kw={'width_ratios': [2, 2, 2, .001],
                                          'height_ratios': [.75, 1]})
    # fig.suptitle(title)
    
    plt.subplots_adjust(wspace=.6, hspace=.1, top=0.8)
    
    if len(pos)==0:
        _, pos = vis_directed_graph(kappa.T, vmin, vmax, 
                                    seed, cbar=False, ax=axes[0,0])
    else:
        vis_directed_graph(kappa.T, vmin, vmax, 
                           seed, pos = pos, cbar=False, ax=axes[0,0])
        
    axes[0,0].set_title('adjacency matrix \n $A_{ij}$ ')
    
    
    # ax1 = fig.add_subplot(gs[0, 1])
    vis_directed_graph(B0.T, vmin, vmax, seed, 
                       pos = pos, cbar=False, ax=axes[0,1])
    axes[0,1].set_title("instantaneous effect\n $B_0$")

    
    # ax2 = fig.add_subplot(gs[0, 2])
    vis_directed_graph(Btau.T, vmin, vmax, 
                       seed, pos = pos, cbar=False, ax=axes[0,2])
    axes[0,2].set_title("lagged effect\n $B_{\\tau}$")
    ###########################################################################
    ##### Plot estimated solution (B0) 
    im = axes[1,0].imshow(K, cmap='RdBu_r', interpolation='none',
                        vmin=vmin, vmax=vmax)
    # axes[1,0].set_title("true graph\n $A_{ij}$")
    # axes[1,0].tick_params(labelsize=18)
    # Add minor tick grid
    axes[1,0].set_xticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,0].set_xticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,0].set_yticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,0].grid(which='minor', axis='both')
    axes[1,0].set_xlabel('source (parent)')
    axes[1,0].set_ylabel('target (child)')
    
    ##### Plot estimated solution (B0) 
    im = axes[1,1].imshow(B0, cmap='RdBu_r', interpolation='none',
                        vmin=vmin, vmax=vmax)
    # axes[1,1].set_title("instantaneous effect\n $B_0$")
    # axes[1,1].tick_params(labelsize=18)
    # Add minor tick grid
    axes[1,1].set_xticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,1].set_yticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,1].grid(which='minor', axis='both')
    axes[1,1].set_xlabel('source (parent)')
    axes[1,1].set_ylabel('target (child)')
    ##### Plot estimated solution (Bt)
    im = axes[1,2].imshow(Btau, cmap='RdBu_r', interpolation='none',
                        vmin=vmin, vmax=vmax)
    # axes[1,2].set_title("lagged effect\n $B_{\\tau}$")
    # Add minor tick grid
    axes[1,2].set_xticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,2].set_yticks(np.arange(-.5, Nosc, 1), minor=True)
    axes[1,2].grid(which='minor', axis='both')
    axes[1,2].set_xlabel('source (parent)')
    axes[1,2].set_ylabel('target (child)')
    
    
    axes[0,-1].set_axis_off()
    axes[1,-1].set_axis_off()
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=-0.05)
    cbar.ax.set_ylabel('coupling strength (a.u.)')
    # fig.subplots_adjust(top=0.75)
    plt.savefig(fig_path + 'estimated_graph.png', bbox_inches="tight")
    plt.savefig(fig_path + 'estimated_graph.svg', bbox_inches="tight")
    plt.show()
    
    
    #%%
    cnt += 1
