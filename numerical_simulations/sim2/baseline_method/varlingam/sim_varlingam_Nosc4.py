# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 08:22:09 2025

@author: H.Yokoyama
"""


from IPython import get_ipython
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
plt.rcParams["font.size"]        = 16 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 2.0
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300 
#%%
import numpy as np
import logging
import copy
import joblib
import itertools
import networkx as nx
import lingam
from lingam import var_lingam

from copy import deepcopy
from my_modules import my_timeseries_graph_plot as my_tp
#%%
########### function of phase oscillator 

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
                   noise_scale=0.001):#0.005):
    dt = h
    p  = noise_scale
    dw = np.random.randn(theta_now.shape[0])
    
    theta      = theta_now + func(theta_now, K1, K2, omega) * dt
    theta_next = theta + p * np.sqrt(dt) * dw
    theta_next = np.mod(theta_next, 2*np.pi)
    return theta_next

def solve_oscillator(theta_ini, h, K1, K2, omega, Nt, Nosc, noise_scale=0.005):
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
        theta_next = euler_maruyama(h, func_coupled_oscillator, 
                                    theta_now, K1, K2, omega,noise_scale=noise_scale)

        theta[t, :]          = theta_next.reshape(1, Nosc)
        phase_dynamics[t, :] = func_coupled_oscillator(theta[t, :], K1, K2, omega)#[t, :])
        

        for i in range(Nosc):
            theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
            
            dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
    
    return theta, dtheta, time


#%% graph visualization
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

#%% [0] generate synthetic data and estimate model
###############
time        = 90#60
fs          = 100 
h           = 1/fs
Nt          = int(time/h) + 1 
Nosc        = 4

K1          = np.array([[ 0, .0,  .0, .0],
                        [.0,  0,  .0, .0],
                        [.0, .1,  .0, .0],
                        [.0, .1,  .1, .0]])

K2          = np.array([[ 0, .1,  .0, .0],
                        [.2,  0,  .0, .0],
                        [.0, .3,  .0, .0],
                        [.0, .3,  .3, .0]])

K1 = np.concatenate(( 0.4*K1[:,:,np.newaxis],
                      0.0*K1[:,:,np.newaxis]),axis=2)
K2 = np.concatenate(( 1.0*K2[:,:,np.newaxis],  
                      0.6*K2[:,:,np.newaxis]),axis=2)

omega  = 2*np.pi* np.array([9.9, 10.505, 10.69, 10.839])


Kexact = np.sum(np.sqrt(K1**2 + K2**2), axis=2)
P      = K1.shape[2]

np.random.seed(250)
x0     = np.random.uniform(0, np.pi, Nosc)





theta, dtheta, time = solve_oscillator(x0, h, 
                                       K1, K2, omega, 
                                       Nt, Nosc)

dtheta = dtheta[1:,:]
theta  = theta[:-1,:]
time   = time[:-1]
Nt     = Nt - 1 

#%%
########################################################
########### visualize dynamical property ################
########################################################
fig_dir = current_path + '/figures/Nosc4/'
if os.path.exists(fig_dir)==False:  # Make the directory for data saving
    os.makedirs(fig_dir)
    
##### visualize exact phase interaction function
seed = 20  # Seed random number generators for reproducibility
vmin = -1
vmax = +1
_, pos = vis_directed_graph(Kexact.T, vmin, vmax, seed)
plt.title('Network structure')
plt.savefig(fig_dir + 'network_graph_exact.png', bbox_inches="tight")
plt.savefig(fig_dir + 'network_graph_exact.svg', bbox_inches="tight")

plt.show()

#%% Apply VARLiNGAM
VARLiNGAM = var_lingam.VARLiNGAM(lags=1,
                                  prune=False,
                                  lingam_model=lingam.ICALiNGAM(),)

# VARLiNGAM.fit(theta)
VARLiNGAM.fit(np.cos(theta))
B_est_ = VARLiNGAM.adjacency_matrices_

B0_est = B_est_[0]
B_est  = B_est_[1]

#%% [5] Step 5 : Asess causal effect and centrality
TEins = np.zeros((Nosc, Nosc))
TEtau = np.zeros((Nosc, Nosc))
causal_order = np.array(VARLiNGAM.causal_order_)

for i in range(Nosc):# from index (source)
    order_i = np.where(causal_order==i)[0][0]
    for j in range(Nosc): # to index (reference)
        order_j = np.where(causal_order==j)[0][0]
        
        if order_i<order_j:
            TEins[i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=0)
            TEtau[i,j] = VARLiNGAM.estimate_total_effect2(Nosc, i, j, from_lag=1)
        
TECins = abs(TEins).sum(axis=1)
TECtau = abs(TEtau).sum(axis=1)

#%% Visualize estimation results
#### plot heatmap of connectivity matrix
fig, axes = plt.subplots(figsize=(20, 9), 
                         ncols=4, nrows=2,
                         gridspec_kw={'width_ratios': [2, 2, 2, .001],
                                      'height_ratios': [1, 1]})
# fig.suptitle(title)

plt.subplots_adjust(wspace=.6, hspace=.1, top=0.8)

if len(pos)==0:
    _, pos = vis_directed_graph(Kexact.T, vmin, vmax, 
                                seed, cbar=False, ax=axes[0,0])
else:
    vis_directed_graph(Kexact.T, vmin, vmax, 
                       seed, pos = pos, cbar=False, ax=axes[0,0])
    
axes[0,0].set_title('dynamical structure \n $K$ ')



vis_directed_graph(B0_est.T , vmin, vmax, seed, 
                   pos = pos, cbar=False, ax=axes[0,1])
axes[0,1].set_title("instantaneous effect\n $B_0$")


# ax2 = fig.add_subplot(gs[0, 2])
vis_directed_graph(B_est.T, vmin, vmax, 
                   seed, pos = pos, cbar=False, ax=axes[0,2])
axes[0,2].set_title("lagged effect\n $B_{\\tau}$")
###########################################################################
##### Plot estimated solution (B0) 
im = axes[1,0].imshow(Kexact, cmap='RdBu_r', interpolation='none',
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
im = axes[1,1].imshow(B0_est, cmap='RdBu_r', interpolation='none',
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
im = axes[1,2].imshow(B_est, cmap='RdBu_r', interpolation='none',
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
plt.savefig(fig_dir + 'estimated_graph.png', bbox_inches="tight")
plt.savefig(fig_dir + 'estimated_graph.svg', bbox_inches="tight")
plt.show()



#%%
######### visualize graph structure (time-series graph)
# fig, axes = plt.subplots(1,1, figsize=(8,10))

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2,2)
# plt.subplots_adjust(wspace=0.3, top=0.75)
plt.subplots_adjust(wspace=.6, hspace=.75, top=0.8)


##### Plot time-series graph
val_matrix = np.zeros((Nosc,Nosc,2))
graph      = np.zeros((Nosc,Nosc,2),dtype='<U3')


val_matrix[:,:,0] = B0_est[np.ix_(causal_order,causal_order)].T
val_matrix[:,:,1] = B_est[np.ix_(causal_order,causal_order)].T

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


##### total effect centrality (instantaneous)
plt.subplot(gs[0, 1])
TEC0 = TECins
TEC1 = TECtau


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

plt.savefig(fig_dir + 'causal_effect_centrality.png', bbox_inches="tight")
plt.savefig(fig_dir + 'causal_effect_centrality.svg', bbox_inches="tight")
plt.show()
