# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:55:19 2025

@author: H.Yokoyama
"""

import numpy as np
import pandas as pd

import dowhy.gcm as gcm
from dowhy.gcm._noise import *

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from joblib import Parallel, delayed

# from statsmodels.tsa.api import VAR
from copy import deepcopy

#%%
def generate_data(causal_order, Nnode, noise, Nt, amat, confounder=False, Z=None):
    data_sim = np.zeros((Nt, Nnode))
    
    for n in causal_order:
        data_sim[:,n] += noise[:,n]
        
        if amat[n,:].sum() != 0:
            data_sim[:,n] += amat[n,:] @ data_sim.T
            
        if confounder == True:
            data_sim[:,n] += Z[:,n]    
    
    return data_sim


def draw_data(data, link_coeffs, Nsample):
    import networkx as nx
    from scipy.stats import uniform, norm
    
    links = link_coeffs[:,:,0]
    links = links - np.diag(np.diag(links))
    amat  = links.T
    
    Gnx   = nx.from_numpy_array(links, create_using=nx.DiGraph())    
    causal_model = gcm.StructuralCausalModel(Gnx)
    for i, n in enumerate(Gnx.nodes):
        if amat[i,:].sum()==0:
            causal_model.set_causal_mechanism(n, 
                                              gcm.EmpiricalDistribution())
        else:
            causal_model.set_causal_mechanism(n, 
                                              gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor(), 
                                                                      noise_model=gcm.ScipyDistribution(uniform)))
    gcm.fit(causal_model, data)
    result = gcm.draw_samples(causal_model, num_samples=Nsample)
    
    return result, causal_model

def create_lagged_data(data,tau_max,**kwargs):
    T = data.shape[0]
    p = data.shape[1]
    if "mask_vector" in kwargs.keys():
        mask_vector = kwargs["mask_vector"]
    else:
        mask_vector = np.full(T, False)

    num_samples = np.sum(mask_vector == False)

    lagged_data = np.full((num_samples,p * (tau_max + 1)),np.nan)

    # each column is 1,...,p, 1 at lag - 1, 2 at lag - 1,..., p at lag -1,...,1 at lag -tau_max,...,p at lag -tau_max
    for i in range(p):
        for tau in range(tau_max + 1):
            data_temp = np.full((num_samples),np.nan)
            index_vec = np.full((num_samples), False)
            index_vec[range(tau, num_samples)] = True
            # only choose the positions that are NOT masked, i.e., mask_vector = FALSE
            index_vec[mask_vector] = False
            data_temp[index_vec] = data[range(T - tau),i]

            lagged_data[:,i +  tau * p] = data_temp
    return lagged_data

def make_lagged_time_label(amat, var_name, tau_min, tau_max):
    lag_val         = np.arange(tau_min-1, tau_max+1)
    var_names_lag   = []
    Nnode = amat.shape[0]
    Nvar  = Nnode/(tau_max + 1)
    cnt = 0
    lag = 0
    for i in range(Nnode):
        if lag_val[lag] == 0:
            str_name = var_name[cnt] + " at t" 
        else:
            str_name = var_name[cnt] + " at t-" + str(lag_val[lag]) 
        var_names_lag.append(str_name)
        cnt += 1
        if np.mod(cnt, Nvar)==0:
            cnt = 0
            lag += 1
    
    return var_names_lag

def convert_tigramite_to_dowhy(tigramite_dag, tigramite_coeff):
    import networkx as nx
    # convert the DAG returned by pcmci to the format of networkx used in dowhy
    p       = tigramite_dag.shape[0]
    tau_max = tigramite_dag.shape[2] - 1
    amat    = np.full((p * (tau_max + 1), p * (tau_max + 1)),-1.0)
    G       = nx.DiGraph()
    for i in range(p * (tau_max + 1)):
        G.add_node(i)
    for i in range(p):
        for j in range(p):
            tau = 0
            if tigramite_dag[i, j, 0] == '-->' and tigramite_dag[j, i, 0] == '<--':
                # i --> j
                amat[i,j] = tigramite_coeff[i,j,tau]
                G.add_edge(i, j)
            if tigramite_dag[i, j, 0] == '<--' and tigramite_dag[j, i, 0] == '-->':
                # j  --> i
                amat[j,i] = tigramite_coeff[j,i,tau]
                G.add_edge(j, i)
            if tigramite_dag[i, j, 0] == '' and tigramite_dag[j, i, 0] == '':
                amat[i,j] = 0

            # time-invariance
            for delta_tau in range(1,tau_max + 1):
                amat[i + delta_tau*p, j + delta_tau * p] = amat[i, j]
                amat[j + delta_tau * p, i + delta_tau * p] = amat[j, i]
                if amat[i,j] > 0:
                    G.add_edge(i + delta_tau * p, j + delta_tau * p)
                if amat[j,i] > 0:
                    G.add_edge(j + delta_tau * p, j + delta_tau * p)

            for tau in range(1,tau_max + 1):
                amat[j, i + tau*p] = 0 # arrow of time
                if tigramite_dag[i,j,tau] == '-->':
                    # i-tau --> j
                    amat[i + tau * p,j] = tigramite_coeff[i,j,tau]
                    G.add_edge(i + tau*p, j)
                if tigramite_dag[i,j,tau] == '':
                    amat[i + tau * p,j] = 0

                # time-invariance
                for delta_tau in range(1,tau_max - tau + 1):
                    if delta_tau > 0:
                        amat[i + (tau + delta_tau) * p, j + delta_tau * p] = amat[i + tau * p, j]
                        amat[j + delta_tau * p, i + (tau + delta_tau) * p] = amat[j, i + tau * p]
                        if amat[i + tau*p,j] > 0:
                            coeff = amat[i + tau*p,j]
                            G.add_edge(i + (tau + delta_tau) * p, j + delta_tau * p, weight=coeff)
                        if amat[j, i + tau * p] > 0:
                            coeff = amat[j, i + tau * p]
                            G.add_edge(j + delta_tau * p, i + (tau + delta_tau) * p, weight=coeff)

    return G,amat


def put_graph_into_dowhy(amat):
    import networkx as nx
    from scipy.stats import uniform, norm
    
    links = amat.T 
    Gnx   = nx.from_numpy_array(links, create_using=nx.DiGraph())    
    causal_model = gcm.StructuralCausalModel(Gnx)
    
    # gcm.auto.assign_causal_mechanisms(causal_model, data)
    for i, n in enumerate(Gnx.nodes):
        if amat[i,:].sum()==0:
            causal_model.set_causal_mechanism(n, 
                                              gcm.EmpiricalDistribution())
        else:
            causal_model.set_causal_mechanism(n, 
                                              gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor(), 
                                                                      noise_model=gcm.ScipyDistribution(uniform)))
    return causal_model




def calc_total_effect(dataframe, causal_model, 
                      var_names_lag, 
                      varname_interven, 
                      varname_target,
                      seed=None):
    
    if seed != None:
        np.random.seed(None)
        
    ##### fit the generative model
    gcm.auto.assign_causal_mechanisms(causal_model, dataframe)
    gcm.fit(causal_model, dataframe)

    ### get interventional samples
    result  = [varname_interven in var for var in var_names_lag]
    idx     = np.where(np.array(result)==True)[0]
    target  = np.where(np.array(var_names_lag)==varname_target)[0]
    
    do1    = gcm.interventional_samples(causal_model,
                                        {num: lambda x: 1 for num in idx},
                                        num_samples_to_draw=10000)

    do0    = gcm.interventional_samples(causal_model,
                                        {num: lambda x: 0 for num in idx},
                                        num_samples_to_draw=10000)
    
    #### assess the averaged total effect
    effect = do1[target].values.mean() - do0[target].values.mean()
    return effect

