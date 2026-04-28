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
plt.rcParams["font.size"]        = 16 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 2.0
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300 
#%%
from my_modules import my_timeseries_graph_plot as my_tp

from copy import deepcopy
from scipy.linalg import expm


from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import Models, LinearMediation, Prediction
from tigramite.causal_effects import CausalEffects
from joblib import Parallel, delayed

import numpy as np
import networkx as nx

from my_modules import my_timeseries_graph_plot as my_tp
from my_modules.causal_effect_estimation_for_tigramite import *
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

#%%
fig_dir = current_path + '/figures/pcmci/'
if os.path.exists(fig_dir)==False:  # Make the directory for data saving
    os.makedirs(fig_dir)
    
# anglular velocity
np.random.seed(100)

time        = 90#100# measurement time
h           = 1/100 #0.01    # micro time
Nt          = int(time/h)

Nosc        = 3
K           = np.array([[.0,  .3, .1],
                        [1.,  .0, .0],
                        [.9,  1., .0]])

causal_orders   = []
order_param     = []
stab_label      = []
lyap_val        = []
eigval          = []
theta_save      = []
TE_save         = []
TEC_save        = []
graph_save      = []
link_coeff_save = []
eq_points       = []
x_save          = []

prob_all        = []

scale         = np.hstack((0.1, np.arange(0.2, 4.2, .2)))
omega         = 2*np.pi*np.random.normal(loc=10, scale=np.sqrt(5E-3), size=Nosc) ##

cnt = 0
for gain in scale:
    #%%
    print(gain)
    kappa   = gain * K 


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
    
    #%% #### Apply pcmci
    np.random.seed(200)
    
    ch_names = np.arange(0,Nosc).astype(str)
    
    pcmci = PCMCI(
            dataframe=pp.DataFrame(np.cos(theta), var_names = ch_names),
            cond_ind_test=ParCorr(),
            verbosity=2
            )

    results = pcmci.run_pcmci(tau_max=1, tau_min=1, pc_alpha=0.1)
    #%%
    graph       = results["graph"]
    link_coeff  = results["val_matrix"]
    tau_max_    = 1
    # lagged_data = create_lagged_data(theta, tau_max = tau_max_)
    # p = Nosc
    # ### convert from array to dataframe
    # dataframe   = pd.DataFrame(lagged_data, columns = np.array(range(p * (tau_max_ + 1))))

    # Gnx_lag, amat_lag = convert_tigramite_to_dowhy(graph, link_coeff)

    # causal_model = put_graph_into_dowhy(amat_lag)
    # lagged_label = make_lagged_time_label(amat_lag, ch_names, 1, tau_max_)
    # #%%
    
    # trgt_label_list = [ch_trg + ' at t' for ch_trg in ch_names]

    # TEtau = np.zeros((Nosc,Nosc))
    # for i, ch_ref in enumerate(ch_names):

    #     varname_interven = ch_ref + ' at t-1'
    #     out = Parallel(n_jobs=-1, verbose=6)(
    #                      delayed(calc_total_effect)
    #                           (dataframe, 
    #                            causal_model, 
    #                            lagged_label, 
    #                            varname_interven, 
    #                            varname_target,
    #                            seed=200) 
    #                               for varname_target in trgt_label_list)
    #     TEtau[i,:] = np.array(out)

    # TECtau = abs(TEtau).sum(axis=1)
    parents = pcmci.return_parents_dict(graph=graph, 
                                        val_matrix=link_coeff)

    med = LinearMediation(dataframe=pp.DataFrame(np.cos(theta), var_names = ch_names))
    med.fit_model(all_parents=parents, tau_max=tau_max_)
    ace    = med.get_all_ace(lag_mode='all_lags')
    TECtau = ace/ace.max()
    
    
    TECtau = TECtau/TECtau.max()
    #%% #### Conduct Stability analysis
    order_param.append(r)
    
    graph_save.append(graph)
    link_coeff_save.append(link_coeff)
    # TE_save.append(np.array(TEtau))
    TEC_save.append(TECtau)
    x_save.append(theta)
    
    cnt += 1
#%%
# TE_save  = np.array(TE_save)
TEC_save = np.array(TEC_save)

#%%
Klist = np.linspace(0,scale.max(),1000)

# ### gaussian case
# SD    = np.sqrt(((omega - omega.mean())**2).sum()/(Nosc))
# Kc    = (np.sqrt(8/np.pi)*SD)/abs(np.linalg.eig(K)[0].max())
# r     = np.sqrt(1-Kc/Klist)
# r[Kc>Klist]=0

eig_max = np.real(np.linalg.eig(K)[0]).max()
g0      = 1/(np.pi)
g0_dev2 = -2/(np.pi)
d_mean  = K.sum(axis=1).mean()
d2_mean = (K.sum(axis=1)**2).mean()
Kc      = 2/(g0 * np.pi * d_mean)

plt.figure(figsize=(4, 3))
plt.scatter(scale, np.array(order_param), 
            zorder=3, c='c', edgecolors='k', alpha=0.7, 
            label='emprical')
plt.plot(scale, np.array(order_param), 
         zorder=2, c='c')

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
for graph, link_coeff, TEC in zip(graph_save, link_coeff_save, TEC_save):
    # B0   = B[0]
    # Btau = B[1]
    
    B0         = link_coeff[:,:,0].T
    Btau       = link_coeff[:,:,1].T
    val_matrix = link_coeff
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
    ax = plt.subplot(gs[0:2, 0])
    my_tp.plot_time_series_graph(graph = graph, 
                                 val_matrix=val_matrix,
                                 var_names=np.arange(Nosc),
                                 vmax_edges=vmax,
                                 vmin_edges=vmin,
                                 link_colorbar_label='causal effect\n(PCMCI)',
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
    TEC0 = np.zeros(TEC.shape)
    TEC1 = TEC
    
    
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
for graph, link_coeff, TEC in zip(graph_save, link_coeff_save, TEC_save):
    print(cnt)
    fig_path = fig_dir + '/scale_%4.2f/'%scale[cnt]
    
    # B0   = B[0]
    # Btau = B[1]
    B0         = link_coeff[:,:,0].T
    Btau       = link_coeff[:,:,1].T
    ##%%
    kappa =  K
    
    ##%%
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
    
    
    ##%%
    cnt += 1
