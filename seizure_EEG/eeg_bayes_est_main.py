# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:37:43 2024

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
plt.rcParams["font.size"]        = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 2.0
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300 
#%%
import joblib
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
from scipy import optimize as op
from scipy.linalg import expm
from copy import deepcopy
#%%
############################################    
####### bayesian estimation
def est_model(theta, T, Nosc, P, h, prec_param):#, check_iter=0):
    model    = CouplingOscillator(theta, P, T, h, prec_param)
    model.fit_model()
    
    return model

def calc_95ci(yPred, std_yPred):
    ci_upper = yPred + 2*std_yPred
    ci_lower = yPred - 2*std_yPred
    
    ci = np.concatenate((ci_upper[:,:,np.newaxis], ci_lower[:,:,np.newaxis]), axis=2)
    
    return ci
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
            
#%%
data_save_dir = current_path + '/save_data/est_model/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
    
prepro_data_dir = current_path + '/save_data/preprocess/' 


name     = []
ext      = []
for file in os.listdir(prepro_data_dir):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
#%% [0] load preprocessed observational data and estimate model

band = 'theta' #  'alpha' #  

for fname, extention in zip(name, ext):
    #%%
    fullpath     = prepro_data_dir + fname + extention 
    datadict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    onset_idx  = datadict['onset_idx']
    offset_idx = datadict['offset_idx']
    theta      = datadict['phi_' + band]
    time       = datadict['time']
    fs         = datadict['fs']
    ch_names   = datadict['ch_names']
    h          = 1/fs
    
    theta_wp   = np.unwrap(theta)
    dtheta     = (theta_wp[:, 1:] - theta_wp[:, :-1] )/h
    theta      = theta[:,:-1]
    
    
    theta      = theta[:,  (onset_idx+int(-30*fs)):(offset_idx+1)]
    dtheta     = dtheta[:, (onset_idx+int(-30*fs)):(offset_idx+1)]
    time       = time[(onset_idx+int(-30*fs)):(offset_idx+1)]
    
    
    # theta      = theta[:,  (onset_idx+int(-10*fs)):(offset_idx+1)]
    # dtheta     = dtheta[:, (onset_idx+int(-10*fs)):(offset_idx+1)]
    # time       = time[(onset_idx+int(-10*fs)):(offset_idx+1)]
    
    # theta      = theta[:,  (onset_idx+int(-10*fs)):(onset_idx+int(60*fs))]
    # dtheta     = dtheta[:, (onset_idx+int(-10*fs)):(onset_idx+int(60*fs))]
    # time       = time[(onset_idx+int(-10*fs)):(onset_idx+int(60*fs))]
    

    
    Nosc, Nt   = theta.shape
    #%% get EEG channel locations
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos_dict      = standard_montage._get_ch_pos()
    
    pos = {}
    for ch in ch_names: 
        if 'Z' in ch:
            ch_ = ch[0] + 'z'
        else:
            ch_ = ch
        
        pos_ch  = ch_pos_dict[ch_][:2]
        pos[ch] = pos_ch
        
        plt.scatter(pos_ch[0], pos_ch[1])
        plt.text(pos_ch[0], pos_ch[1], ch)
    plt.show()
    
    #%%
    
    Popt     = 3
    Nt       = theta.shape[1]
    
    Tstep    = 1
    noise_param = 1E-1#1E-2 # covariance of process noise
    prec_param  = 1/noise_param # precision parameter, cov(process noise) = 1/prec_param
    
    model    = est_model(theta.T, Tstep, Nosc, Popt, h, prec_param) 
    # #%% #############################
    # P_candi  = np.arange(2,9,1)
    
    # processed = joblib.Parallel(n_jobs=7, verbose=5)(
    #                 joblib.delayed(est_model)(theta.T, Tstep, Nosc, p, h, prec_param) 
    #                     for p in P_candi)
    # ##############
    # criterion = np.zeros(P_candi.shape)
    # for i, result in enumerate(processed):
    #     k            = result.Kb0.shape[0]
    #     n            = result.y_hat.shape[0]
    #     loglike      = result.loglike[-1]#result.loglike.sum()
        
    #     criterion[i] =  loglike#/n
    
    # idx_Popt = np.where(criterion == criterion.max())[0]
    #     # criterion[i] =  (-2 * loglike + (k * np.log(n)))/n
    
    # # idx_Popt = np.where(criterion == criterion.min())[0]
    
    
    # #############
    # Popt  = P_candi[idx_Popt]
    # model = processed[idx_Popt[0]]
    
    # plt.plot(P_candi, criterion)
    # plt.scatter(Popt, criterion[idx_Popt], c='r')
    # plt.xlabel('Model order $P$')
    # plt.ylabel('log likelihood')
    # # plt.ylabel('BIC')
    # plt.xticks(np.arange(0, 14, 2)) 
    # plt.show()
    #%%
    ############# store estimation results
    
    omega_est   = model.omega[-1,:]
    K1est       = model.coeff_cos
    K2est       = model.coeff_sin
    Kest        = np.sum(np.sqrt(K1est**2 + K2est**2), axis=2)
    
    dtheta_pred = model.y_hat
    
    ll_save     = model.loglike
    Popt        = model.P
    # std_yPred   = model.std_yPred
    
    print('optimal order P = %d'%Popt)
    del model
    
    
    #%% save data 
    save_dict                = {} 
    save_dict['fs']          = fs
    save_dict['ch_names']    = ch_names
    save_dict['time']        = time
    save_dict['theta']       = theta
    save_dict['dtheta']      = dtheta
    save_dict['ch_pos']      = pos
    
    save_dict['Popt']        = Popt
    save_dict['dtheta_pred'] = dtheta_pred
    save_dict['omega_est']   = omega_est
    save_dict['K1est']       = K1est
    save_dict['K2est']       = K2est
    save_dict['Kest']        = Kest
    # save_dict['elbo_save']   = elbo
    #%%
    save_name   = 'bayse_est_' + fname.split('preprocess_')[1]
    fullpath_save   = data_save_dir + save_name 
    np.save(fullpath_save, save_dict)
    
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
plot_phase_interaction(theta.T, dtheta.T, omega_est, K1est, K2est, Nosc, ch_names)
plt.savefig(fig_dir + 'phase_interaction_bayes.png', bbox_inches="tight") 
plt.savefig(fig_dir + 'phase_interaction_bayes.svg', bbox_inches="tight")
plt.show()
