# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:37:43 2024

@author: H.Yokoyama
"""


from IPython import get_ipython
# get_ipython().magic('reset -sf')
# get_ipython().magic('clear')

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
import mne
import numpy as np
import logging
import copy
import joblib
import itertools
import networkx as nx
import lingam
from lingam import var_lingam
# from my_modules.coupling_oscillator_vbem import CouplingOscillator
from my_modules.coupling_oscillator_bayes import CouplingOscillator
from joblib import Parallel, delayed
from scipy import optimize as op
from scipy.linalg import expm
from copy import deepcopy
from my_modules import my_timeseries_graph_plot as my_tp
#%%
########### function of kuramoto oscillator 

def func_coupled_oscillator(theta, K1, K2, omega):
    ndims = np.ndim(K1)
    Nosc  = theta.shape[0]
    coupling_func = np.zeros(Nosc)
    for i in range(Nosc):
        for j in range(Nosc):
            if i != j:
                diff = theta[j] - theta[i]
                if ndims==2:
                    coupling_func[i] += K1[i,j] * np.cos(diff) + K2[i,j] * np.sin(diff)
                elif ndims == 3:
                    P = K1.shape[2] 
                    for p in range(P):
                        m = p + 1
                        coupling_func[i] += K1[i,j,p] * np.cos(m*diff) + K2[i,j,p] * np.sin(m*diff)
                    
    phase_dynamics = omega + coupling_func
    return phase_dynamics

def runge_kutta(h, func, theta_now, 
                K1, K2, omega):
    k1=func(theta_now, K1, K2, omega)
    
    theta4k2=theta_now+(h/2)*k1
    k2=func(theta4k2, K1, K2, omega)
    
    theta4k3=theta_now+(h/2)*k2
    k3=func(theta4k3, K1, K2, omega)
    
    theta4k4=theta_now+h*k3
    k4=func(theta4k4, K1, K2, omega)
    
    theta_next=theta_now+(h/6)*(k1+2*k2+2*k3+k4)
    theta_next=np.mod(theta_next, 2*np.pi) 
    
    return theta_next

def euler_maruyama(h, func, theta_now, 
                   K1, K2, omega, 
                   noise_scale=0.001):
    dt = h
    p  = noise_scale
    dw = np.random.randn(theta_now.shape[0])
    
    theta      = theta_now + func(theta_now, K1, K2, omega) * dt
    theta_next = theta + p * np.sqrt(dt) * dw
    theta_next = np.mod(theta_next, 2*np.pi)
    return theta_next

def solve_oscillator(theta_ini, h, K1, K2, omega, Nt, Nosc):
    fs          = 1/h
    dtheta      = np.zeros((Nt, Nosc))
    theta       = np.zeros((Nt, Nosc))
    time        = np.arange(0, Nt)/fs
    
    theta[0, :] = theta_ini
    phase_dynamics       = np.zeros((Nt, Nosc))
    phase_dynamics[0, :] = func_coupled_oscillator(theta[0, :], K1, K2, omega)#[0,:])
    for t in range(1, Nt):
        theta_now  = theta[t-1, :] 
        # theta_next = runge_kutta(h, func_coupled_oscillator, theta_now, K1, K2, omega)
        theta_next = euler_maruyama(h, func_coupled_oscillator, theta_now, K1, K2, omega)

        theta[t, :]          = theta_next.reshape(1, Nosc)
        phase_dynamics[t, :] = func_coupled_oscillator(theta[t, :], K1, K2, omega)#[t, :])
        

        for i in range(Nosc):
            theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
            
            dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
    
    return theta, dtheta, time


def get_jacobian_matrix(theta_diff, K1, K2, omega):#K1, K2, Nosc, P):
    # K1 : cosine coefficient matrix in p-th Fourier series with size of (Nosc,Nosc,P)
    # K2 : cosine coefficient matrix in p-th Fourier series with size of (Nosc,Nosc,P)
    Nosc,_,P = K1.shape 
    
    J    = np.zeros((Nosc,Nosc))
    for i in range(Nosc):
        for j in range(Nosc):
            for k in range(P):
                p = k + 1
                if i==j:
                    diff   = p*theta_diff[i,:]#(theta - theta[i])
                    J[i,j] += np.sum(p*(K1[i,:,k]*np.sin(diff) - K2[i,:,k]*np.cos(diff)))
                else:
                    diff    = p*theta_diff[i,j]#(theta[j] - theta[i])
                    J[i,j] += p*(-K1[i,j,k] * np.sin(diff) + K2[i,j,k] * np.cos(diff))
    return J

def solve_fokker_planck_2osc(omega1, omega2, a1, a2, b1, b2, dt, noise_scale, Nosc, N=100):
    Np        = len(a1)
    # N         = 100
    # t         = np.arange(0, 100+dt, dt)
    t         = np.arange(0, 1000+dt, dt)
    Nt        = len(t)
    phi_diff  = np.linspace(0, 2*np.pi, N)
    phi_diff_ = np.linspace(2*np.pi, 0, N)
    D         = (noise_scale*np.sqrt(dt))
    
    g_diff1   = 0
    g_diff2   = 0
    
    g_diff1_  = 0
    g_diff2_  = 0
    for n in range(Np):
        p    = n + 1                    
        phi  = p * phi_diff
        phi_ = p * phi_diff_
        
        # g_diff1  +=  a1[n] * np.cos(phi)  + b1[n] * np.sin(phi)   
        # g_diff2  +=  a2[n] * np.cos(-phi) + b2[n] * np.sin(-phi)   
        
        # g_diff1_ +=  a1[n] * np.cos(-phi) + b1[n] * np.sin(-phi)   
        # g_diff2_ +=  a2[n] * np.cos(phi)  + b2[n] * np.sin(phi) 
        
        g_diff1  +=  a1[n] * np.cos(phi)  + b1[n] * np.sin(phi)   
        g_diff2  +=  a2[n] * np.cos(phi_) + b2[n] * np.sin(phi_)   
        
        g_diff1_ +=  a1[n] * np.cos(phi_) + b1[n] * np.sin(phi_)   
        g_diff2_ +=  a2[n] * np.cos(phi)  + b2[n] * np.sin(phi)
        
    G21 = (omega2 + g_diff2)  - (omega1 + g_diff1)  # d(θ2-θ1)/dt  
    G12 = (omega1 + g_diff1_) - (omega2 + g_diff2_) # d(θ1-θ2)/dt  
    
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

def solve_steady_state(omega, a, b, dt,
                       noise_scale=np.sqrt(0.5), Ndelta=50):
    Nosc = len(omega)
    
    prob_phi = np.zeros((Ndelta, Nosc, Nosc))
    phi_st   = np.zeros((Nosc, Nosc))
    
    ij_list = []
    
    a1_list = []
    a2_list = []
    b1_list = []
    b2_list = []
    omega1_list = []
    omega2_list = []
    
    
    for i in range(Nosc):
        for j in range(i,Nosc):
            ij_list.append([i,j])
            
            a1_list.append(a[i,j,:])
            b1_list.append(b[i,j,:])
            omega1_list.append(omega[i])
            
            a2_list.append(a[j,i,:])
            b2_list.append(b[j,i,:])
            omega2_list.append(omega[j])
            
    out = Parallel(n_jobs=10, verbose=6)(
                     delayed(solve_fokker_planck_2osc)
                     (omega1, omega2, a1, a2, b1, b2,  
                      dt,noise_scale, Nosc, N=Ndelta)
                     for omega1, omega2, a1, a2, b1, b2 in 
                       zip(omega1_list, omega2_list, 
                           a1_list, a2_list, b1_list, b2_list))
    
    for ij, res in zip(ij_list, out):
        i = ij[0]
        j = ij[1]
        
        phi_plot = res[0]
        prob_fk  = res[1]
        
        prob_phi[:,i,j] = prob_fk[:,0]
        prob_phi[:,j,i] = prob_fk[:,1]
        
        phi_st[i,j] = phi_plot[prob_fk[:,0].argmax()]
        phi_st[j,i] = phi_plot[prob_fk[:,1].argmax()]
    
    return phi_st, phi_plot, prob_phi
# def solve_steady_state(omega, a, b, dt,
#                        noise_scale=np.sqrt(0.5), Ndelta=50):
#     Nosc = len(omega)
    
#     prob_phi = np.zeros((Ndelta, Nosc, Nosc))
#     phi_st   = np.zeros((Nosc, Nosc))
    
#     for i in range(Nosc):
        
#         for j in range(i, Nosc):
#             if i != j:
#                 omega1 = omega[i]
#                 a1     = a[i,j,:] 
#                 b1     = b[i,j,:]
                
#                 omega2 = omega[j]
#                 a2     = a[j,i,:]
#                 b2     = b[j,i,:]
                
#                 phi_plot, prob_fk = solve_fokker_planck_2osc(omega1, omega2, 
#                                                              a1, a2, b1, b2, dt, 
#                                                              noise_scale, Nosc, N=Ndelta)
#                 prob_phi[:,i,j] = prob_fk[:,0]
#                 prob_phi[:,j,i] = prob_fk[:,1]
                
#                 phi_st[i,j] = phi_plot[prob_fk[:,0].argmax()]
#                 phi_st[j,i] = phi_plot[prob_fk[:,1].argmax()]
                
#     return phi_st, phi_plot, prob_phi

#%% graph visualization
def vis_directed_graph(K, vmin, vmax, seed, node_names, pos=None, node_size_weight=None):
    import matplotlib as mpl
    
    plt.figure(figsize=(6, 5))
    im_ratio = 5 / 6
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]#np.array(sorted(weight[weight!=0]))#
    
    G      = nx.from_numpy_array(K, create_using=nx.MultiDiGraph())
    G.edges(data=True)
    labels = {i : node for i, node in enumerate(node_names)}          
    G = nx.relabel_nodes(G, labels)
    
    if pos is None:
        pos = nx.spring_layout(G, seed=seed)
    
    node_sizes  = [300  for i in range(len(G))]
    if node_size_weight is not None:
        node_sizes  = node_size_weight * node_sizes 
        
    M           = G.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1] = 1
    edge_alphas[edge_alphas<=0.2] = 0
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.3',
                                         arrowsize=20, edge_color=edge_colors,
                                         alpha=edge_alphas,
                                         width=1.5,
                                         edge_vmin=vmin, edge_vmax=vmax)
    
    # nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color = 'w')
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
    ax = plt.gca()
    ax.set_axis_off()
    
    plt.colorbar(pc, ax=ax, label='coupling strength (a.u.)', 
                 fraction=0.05*im_ratio, pad=0.035)
    
    return edges, pos

def plot_phase_interaction(theta, dtheta, omega, K1, K2, Nosc, ch_names, ylims=[8, 24]):
    P = K1.shape[2]
    
    phi_delta_plot = np.linspace(0, 2*np.pi, 30)
    
    fig = plt.figure(constrained_layout = False, figsize=(24, 24));
    plt.subplots_adjust(wspace=0.9, hspace=0.9);
    gs  = fig.add_gridspec(Nosc, Nosc)
    
    cnt = 0
    for ref in range(Nosc):
        dphi = deepcopy(dtheta[:,ref])
        for osc in range(Nosc):
            if osc != ref:
                prc_model = omega[ref] * np.ones(phi_delta_plot.shape)
                
                phi_delta_obs = np.mod(deepcopy(theta[:, osc]) - deepcopy(theta[:, ref]), 2*np.pi)
                
                
                for k in range(P):
                    p   = k + 1
                    cos = np.cos(p*phi_delta_plot)
                    sin = np.sin(p*phi_delta_plot)
                    prc_model += K1[ref,osc,k] * cos + K2[ref,osc,k] * sin
                    
                plt.subplot(gs[ref, osc])
                plt.plot(phi_delta_plot, prc_model, c='r', linewidth = 3, zorder=2, label='true')
                plt.scatter(phi_delta_obs, dphi, c = 'gray', marker = '.', alpha=0.2, zorder=1, label='sample')
                
                plt.xlabel('$\\theta_{%s} - \\theta_{%s} $'%(ch_names[osc], ch_names[ref]))
                plt.ylabel('$d \\theta_{%s} / dt $'%(ch_names[ref]))
                plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
                
                ylims   = np.array([omega[ref]- 30, omega[ref] + 30]) 
                plt.ylim(ylims)
            elif (osc==0) & (ref==0):
                ax = plt.subplot(gs[ref, osc])
                plt.text(-2.5, 0, '$\\theta_{parent}$ - $\\theta_{target}$', fontsize=16)
                plt.xlim(-1.2, 2)
                plt.ylim(-.1, .5)
                ax.set_axis_off()
                
            cnt += 1
            
def check_estimated_equilibrium_points(theta, phi_diff_eq, omega, dt, 
                                       title, phi_diff_fk, prob_fk):
    Nosc = len(omega)
    
    fig = plt.figure(constrained_layout = False, figsize=(24, 24));
    plt.subplots_adjust(wspace=0.5);
    gs  = fig.add_gridspec(Nosc, Nosc)
    
    fig.suptitle(title)
    
    cnt = 0
    for i in range(Nosc):
        
        for j in range(i, Nosc):
            if i != j:

                plt.subplot(gs[i,j])
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
                
                plt.ylim(0, 1.15)
                if (i==Nosc-2) & (j==Nosc-1):
                    plt.legend(bbox_to_anchor=(1.05, 1), 
                               loc='upper left', 
                               borderaxespad=0)
                elif cnt==0:
                    plt.ylabel('density')
                
                cnt += 1
    return fig

def get_total_effect(VARLiNGAM, causal_order, Nosc, i,j, 
                     from_lag=0, data=None):
    
    order_i = np.where(causal_order==i)[0][0]
    order_j = np.where(causal_order==j)[0][0]
    
    if order_i<order_j:
        if data is None:
            TE = VARLiNGAM.estimate_total_effect2(Nosc, i, j, 
                                                  from_lag=from_lag)
        else:
            TE = VARLiNGAM.estimate_total_effect(data, i, j, 
                                                 from_lag=from_lag)
    else:
        TE = 0
    
    return TE#ins, TEtau
#%%

prepro_data_dir = current_path + '/save_data/preprocess/' 


name     = []
ext      = []
for file in os.listdir(prepro_data_dir):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])


#%% [0] load preprocessed observational data and estimate model
data_save_dir = current_path + '/save_data/est_model/' 

sbjID       = 'chb15_06'
band        = 'theta'
fname       = 'bayse_est_%s'%sbjID
fullpath    = data_save_dir + fname + '.npy'
datadict    = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()

theta       = datadict['theta'].T
dtheta      = datadict['dtheta'].T
ch_names    = datadict['ch_names']
pos         = datadict['ch_pos']
fs          = datadict['fs']
h           = 1/fs
time        = datadict['time']

P           = datadict['Popt']
dtheta_pred = datadict['dtheta_pred']
omega_est   = datadict['omega_est']
K1est       = datadict['K1est']
K2est       = datadict['K2est']
Kest        = datadict['Kest']

Nt, Nosc    = theta.shape    
#%%
########################################################
########### visualize dynamical property ################
########################################################
fig_dir = current_path + '/figures/dysvar_%s/'%band
if os.path.exists(fig_dir)==False:  # Make the directory for data saving
    os.makedirs(fig_dir)
    

##### visualize estimated phase interaction function
seed = 20  # Seed random number generators for reproducibility
vmin = -1.1
vmax = +1.1
vis_directed_graph(Kest.T, 0, 0.8*Kest.max(), 100, ch_names, pos=pos)
plt.title('Network structure')
plt.savefig(fig_dir + 'network_graph_bayes.png', bbox_inches="tight")
plt.savefig(fig_dir + 'network_graph_bayes.svg', bbox_inches="tight")

plt.show()
#%%
# plot_phase_interaction(theta, dtheta, omega_est, K1est, K2est, Nosc, ch_names)
# plt.savefig(fig_dir + 'phase_interaction_bayes.png', bbox_inches="tight") 
# plt.savefig(fig_dir + 'phase_interaction_bayes.svg', bbox_inches="tight")
# plt.show()
#%% [1] Step.1 : Find the steady-state solutions
np.random.seed(100)

# phi_st, phi_plot, prob_phi = solve_steady_state(omega_est, K1est, K2est, h,
#                                                 noise_scale=1,#3,
#                                                 Ndelta=60)
phi_st, phi_plot, prob_phi = solve_steady_state(omega_est, K1est, K2est, h,
                                                noise_scale=1,
                                                Ndelta=36)


check_estimated_equilibrium_points(theta, phi_st, omega_est, h, 
                                    '', phi_plot, prob_phi)            
plt.savefig(fig_dir + 'equilibrium_points.png', bbox_inches="tight")
plt.savefig(fig_dir + 'equilibrium_points.svg', bbox_inches="tight")
plt.show()
#%% [2] Step.2 : Determine Jacobian matrix with the equilibrium points

np.random.seed(100)

J = get_jacobian_matrix(phi_st, K1est, K2est, omega_est)
eigval, eigvec = np.linalg.eig(J)
print(eigval)

# x0  = np.random.uniform(0, 2*np.pi, Nosc)  
x0   = np.random.uniform(0, np.pi, Nosc)  
Nsim = int(60*fs)
theta_est, dtheta, time = solve_oscillator(x0, h, 
                                           K1est, K2est, omega_est, 
                                           Nt, Nosc)
#%% [3] Step.3 : Calculate Linearized matrix
M   = expm(J*h)
#%% [4] Step.4 : Estimate instantaneous and lagged effects B0, B with LiNGAM
VARLiNGAM = var_lingam.VARLiNGAM(lags=1,
                                  prune=False,
                                  ar_coefs=M[np.newaxis,:,:], 
                                  lingam_model=lingam.ICALiNGAM())

VARLiNGAM.fit(np.unwrap(theta_est.T).T)
B_est_ = VARLiNGAM.adjacency_matrices_

B0_est = B_est_[0]
B_est  = B_est_[1]
#%% [5] Step 5 : Asess causal effect and centrality
TEins = np.zeros((Nosc, Nosc))
TEtau = np.zeros((Nosc, Nosc))

causal_order = np.array(VARLiNGAM.causal_order_)


for i in range(Nosc):# from index (source)
    list_j = np.arange(0,Nosc)
    
    out = Parallel(n_jobs=-1, verbose=6)(
        delayed(get_total_effect)
        (VARLiNGAM, causal_order, Nosc, i,j, from_lag=0) 
            for j in list_j)
            
    TEins[i,:] = np.array([res for res in out])
    
    
for i in range(Nosc):# from index (source)
    list_j = np.arange(0,Nosc)
    
    out = Parallel(n_jobs=-1, verbose=6)(
        delayed(get_total_effect)
        (VARLiNGAM, causal_order, Nosc, i,j, from_lag=1) 
            for j in list_j)
            
    TEtau[i,:] = np.array([res for res in out])
    
    print('%s (%02d/%02d): Total effect estimation completed!'%(ch_names[i], i+1, Nosc))
            
TECins = abs(TEins).sum(axis=1)
TECtau = abs(TEtau).sum(axis=1)

# for i in range(Nosc):# from index (source)
#     order_i = np.where(causal_order==i)[0][0]
#     for j in range(Nosc): # to index (reference)
#         order_j = np.where(causal_order==j)[0][0]
        
#         if order_i<order_j:
#             # TEins[i,j] = VARLiNGAM.estimate_total_effect(np.unwrap(theta_est.T).T, i, j, from_lag=0)
#             # TEtau[i,j] = VARLiNGAM.estimate_total_effect(np.unwrap(theta_est.T).T, i, j, from_lag=1)
            
#             TEins[i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=0)
#             TEtau[i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=1)
            
# TECins = abs(TEins).sum(axis=1)
# TECtau = abs(TEtau).sum(axis=1)

#%% Visualize estimation results 
#### plot heatmap of connectivity matrix
fig, axes = plt.subplots(figsize=(7, 3), ncols=2)
##### Plot estimated solution (B0) 
im = axes[0].imshow(B0_est, cmap='RdBu_r', interpolation='none',
                    vmin=vmin, vmax=vmax)
axes[0].set_title("instantaneous effect\n $B_0$", fontsize=13)
axes[0].tick_params(labelsize=13)
# Add minor tick grid
axes[0].set_xticks(np.arange(0, Nosc, 1), minor=False)
axes[0].set_yticks(np.arange(0, Nosc, 1), minor=False)
axes[0].set_xticklabels(ch_names, minor=False, rotation=90)
axes[0].set_yticklabels(ch_names, minor=False)
axes[0].grid(which='minor', axis='both')
axes[0].set_xlabel('source (parent)')
axes[0].set_ylabel('target (child)')
##### Plot estimated solution (Bt)
im = axes[1].imshow(B_est, cmap='RdBu_r', interpolation='none',
                    vmin=vmin, vmax=vmax)
axes[1].set_title("lagged effect\n $B_{\\tau}$", fontsize=13)
axes[1].set_yticklabels([])    # Remove yticks
axes[1].tick_params(labelsize=13)
# Add minor tick grid
axes[1].set_xticks(np.arange(0, Nosc, 1), minor=False)
axes[1].set_yticks(np.arange(0, Nosc, 1), minor=False)
axes[1].set_xticklabels(ch_names, minor=False, rotation=90)
axes[1].set_yticklabels(ch_names, minor=False)
axes[1].grid(which='minor', axis='both')
axes[1].set_xlabel('source (parent)')
##### Adjust space between subplots
fig.subplots_adjust(wspace=0.1)
##### Add Colorbar (with abit of hard-coding)
im_ratio = 3 / 10
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
cbar.ax.tick_params(labelsize=13)

# if save_name is not None:
#     fig.savefig(save_name, bbox_inches='tight')

plt.savefig(fig_dir + 'estimated_graph_matrix.png', bbox_inches="tight")
plt.savefig(fig_dir + 'estimated_graph_matrix.svg', bbox_inches="tight")
plt.show()



#%%
##### visualize graph structures
vis_directed_graph(B0_est.T, vmin, vmax, seed, ch_names, pos = pos)
plt.title("instantaneous effect\n $B_0$")
plt.savefig(fig_dir + 'network_graph_inst.png', bbox_inches="tight")
plt.savefig(fig_dir + 'network_graph_inst.svg', bbox_inches="tight")
plt.show()

# node_size_weight = abs(IEC)/abs(IEC.max())
vis_directed_graph(B_est.T, vmin, vmax, seed, 
                   ch_names, 
                   pos = pos)
plt.title("lagged effect\n $B_{\\tau}$")
plt.savefig(fig_dir + 'network_graph_lagged.png', bbox_inches="tight")
plt.savefig(fig_dir + 'network_graph_lagged.svg', bbox_inches="tight")
plt.show()
#%%
fig, axes = plt.subplots(1,1, figsize=(20,30))


val_matrix = np.zeros((Nosc,Nosc,2))
graph      = np.zeros((Nosc,Nosc,2),dtype='<U3')


val_matrix[:,:,0] = B0_est[np.ix_(causal_order,causal_order)].T
val_matrix[:,:,1] = B_est[np.ix_(causal_order,causal_order)].T

# val_matrix[:,:,0] = B0_est.T
# val_matrix[:,:,1] = B_est.T
graph[val_matrix!=0] = '-->'


my_tp.plot_time_series_graph(graph = graph, 
                             val_matrix=val_matrix,
                             var_names=np.array(ch_names)[causal_order],#var_names=ch_names,# 
                             vmax_edges=vmax,
                             vmin_edges=vmin,
                             link_colorbar_label='causal effect\n(j-VARLiNGAM)',
                             alpha=.6,
                             node_size=.035,
                             curved_radius=0.35,
                             arrow_linewidth=10,
                             label_fontsize=40,
                             tick_label_size=34,
                             fig_ax=(fig, axes)
                             )
plt.savefig(fig_dir + 'timeseries_network_graph.png', bbox_inches="tight")
plt.savefig(fig_dir + 'timeseries_network_graph.svg', bbox_inches="tight")
plt.show()


#%% plot results on causal centrality analysis
fig = plt.figure(figsize=(6, 8))
gs  = fig.add_gridspec(2,1)
plt.subplots_adjust(hspace=.6)

TECins = TECins/TECins.max()
TECtau = TECtau/TECtau.max()

########## instantaneous effect ###########
##### total effect centrality (instantaneous)
plt.subplot(gs[0, 0])
plt.bar(np.arange(Nosc), abs(TECins))
plt.plot([-0.5, Nosc+0.5],[0,0],'k--')
plt.xlim(-0.5, Nosc-0.5)
# plt.ylim(-.5, 15)
plt.xlabel('# Node')
plt.ylabel('normalized TEC (a.u.)')
plt.xticks(ticks=np.arange(Nosc), labels=ch_names, rotation=90)
plt.title('instantaneous effect')

##### total effect centrality (lagged)
plt.subplot(gs[1, 0])
plt.bar(np.arange(Nosc), abs(TECtau))
plt.plot([-0.5, Nosc+0.5],[0,0],'k--')
plt.xlim(-0.5, Nosc-0.5)
plt.ylabel('normalized TEC (a.u.)')
plt.xlabel('# Node')
plt.xticks(ticks=np.arange(Nosc), labels=ch_names, rotation=90)
plt.title('lagged effect')

plt.savefig(fig_dir + 'causal_effect_centrality.png', bbox_inches="tight")
plt.savefig(fig_dir + 'causal_effect_centrality.svg', bbox_inches="tight")
plt.show()

    
#%%
data_save_dir = current_path + '/save_data/causal_model/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
    
#%% save results
save_dict                 = {} 
save_dict['fs']           = fs
save_dict['ch_names']     = ch_names
save_dict['ch_pos']       = pos
save_dict['time']         = time
save_dict['theta']        = theta
save_dict['dtheta']       = dtheta

save_dict['B0_est']       = B0_est
save_dict['B_est']        = B_est
save_dict['TEins']        = TEins
save_dict['TEtau']        = TEtau
save_dict['M']            = M
save_dict['Jaco_mat']     = J
save_dict['phi_solution'] = phi_st
save_dict['causal_order'] = causal_order

save_dict['dtheta_pred'] = dtheta_pred
save_dict['omega_est']   = omega_est
save_dict['K1est']       = K1est
save_dict['K2est']       = K2est
save_dict['Kest']        = Kest
save_dict['Popt']        = P
# save_dict['elbo_save']   = elbo
#%%

save_name   = 'causal_' + fname.split('bayse_est_')[1]
fullpath_save   = data_save_dir + save_name 
np.save(fullpath_save, save_dict)