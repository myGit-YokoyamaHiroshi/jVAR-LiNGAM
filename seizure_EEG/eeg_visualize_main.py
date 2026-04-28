# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:57:10 2026

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


#%% graph visualization
def vis_directed_graph(K, vmin, vmax, seed, node_names, pos=None, node_size_weight=None, figsize=None):
    import matplotlib as mpl
    
    if figsize==None:
        figsize=(6, 5)
    
    plt.figure(figsize=figsize)
    im_ratio = 5 / 6
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]#np.array(sorted(weight[weight!=0]))#
    
    G      = nx.from_numpy_array(K, create_using=nx.MultiDiGraph())
    G.edges(data=True)
    labels = {i : node for i, node in enumerate(node_names)}          
    G = nx.relabel_nodes(G, labels)
    
    if pos is None:
        # pos = nx.spring_layout(G, seed=seed)
        
        plt.figure(figsize=(12, 5))
        im_ratio = 5 / 6
        
        for layer, nodes in enumerate(nx.topological_generations(G)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer
                
        pos = nx.multipartite_layout(G, subset_key="layer")
    
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
data_save_dir = current_path + '/save_data/causal_model/' 

sbjID       = 'chb15_06'
band        = 'theta'
fname       = 'causal_%s'%sbjID
fullpath    = data_save_dir + fname + '.npy'
datadict    = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()


fs          = datadict['fs']
ch_names    = datadict['ch_names']
pos         = datadict['ch_pos']
time        = datadict['time']
theta       = datadict['theta']
dtheta      = datadict['dtheta']

P           = datadict['Popt']
dtheta_pred = datadict['dtheta_pred']
omega_est   = datadict['omega_est']
K1est       = datadict['K1est']
K2est       = datadict['K2est']
Kest        = datadict['Kest']



B0_est       = datadict['B0_est']
B_est        = datadict['B_est']
TEins        = datadict['TEins']
TEtau        = datadict['TEtau']
TECins       = abs(TEins).sum(axis=1)
TECtau       = abs(TEtau).sum(axis=1)

M            = datadict['M'] 
J            = datadict['Jaco_mat']
phi_st       = datadict['phi_solution']
causal_order = datadict['causal_order']

h            = 1/fs
Nt, Nosc     = theta.shape    

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


_, pos_ordering = vis_directed_graph(B0_est.T, vmin, vmax, seed, 
                   ch_names)
plt.title("instantaneous effect\n $B_0$")
plt.savefig(fig_dir + 'topological_order.png', bbox_inches="tight")
plt.savefig(fig_dir + 'topological_order.svg', bbox_inches="tight")
plt.show()

B0_est_ = deepcopy(B0_est)
B0_est_[abs(B0_est_)<0.6] = 0    

vis_directed_graph(B0_est_.T, vmin, vmax, seed, 
                   ch_names, pos=pos_ordering, figsize=(12, 5))
plt.title("instantaneous effect\n $B_0$")
plt.savefig(fig_dir + 'topological_sort_2.png', bbox_inches="tight")
plt.savefig(fig_dir + 'topological_sort_2.svg', bbox_inches="tight")
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

    
