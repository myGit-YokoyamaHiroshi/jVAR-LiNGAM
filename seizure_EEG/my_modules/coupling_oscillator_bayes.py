# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:02:42 2025

@author: H.Yokoyama
"""


from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np


class CouplingOscillator:
    def __init__(self, x, P, T, h, prec_param):
        self.x          = x
        self.P          = P
        self.T          = T
        self.h          = h
        self.prec_param = prec_param

##############################################################################
    def fit_model(self):
        x          = self.x
        P          = self.P
        T          = self.T
        h          = self.h
        prec_param = self.prec_param
        
        Nt, Nosc   = x.shape
        
        self.Nosc  = Nosc
        
        Kb0        = np.eye(Nosc*(Nosc*2*P+1)*T)

        
        mu_beta0 = np.zeros(Nosc*(Nosc*2*P+1)*T)
        L        = np.nan * np.ones(Nt-T)
        
        beta     = np.zeros((Nt-T, Nosc*Nosc, 2*P))
        OMEGA    = np.zeros((Nt-T, Nosc))
            
        S        = np.zeros((Nosc, Nosc, Nt-T))
        
        theta_hat   = np.nan * np.ones((Nt-T, Nosc))
        #%%
        cnt = 1
        
        Total_Epoch = int((Nt-T)/T)
        
        for i in range(T, Nt, T):#(0, Nt-T-1, T):
            #%%
            if np.mod(i, 100)==0:
                print('Epoch: (%d / %d), index: %d'%(cnt, Total_Epoch, i))
            #########################################################################
            self.make_fourier_features(x, i)            
            Dx      = self.make_Dx()#(x_train, T, Nosc, 2*P)
            self.Dx = Dx
            
            #### Update step : Update prior distribution (update model parameter) 
            mu_beta, Kb, sigma, loglike = self.update_coeff(mu_beta0, Kb0, 1/prec_param)
        
            mu_beta0     = deepcopy(mu_beta)
            Kb0          = deepcopy(Kb)
            S[:,:,i-T]   = deepcopy(sigma)
            L[i-T]       = deepcopy(loglike)
            
            # print([i-T])
            
            tmp_y = (Dx @ mu_beta).reshape((T, Nosc), order='C')
            if i == T:
                y_hat = deepcopy(tmp_y)
            else:
                y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
            
            tmp_beta = mu_beta.reshape((T*Nosc, Nosc*2*P+1))#B_hat.reshape((T*Nosc, Nosc*P+1))#
            for p in range(2*P):
                idx = np.arange(0, Nosc, 1) + p*Nosc
                beta[i-T:i, :, p]     = tmp_beta[:,idx].reshape((T, Nosc*Nosc))
                
            OMEGA[i-T:i, :]    = tmp_beta[:,-1].reshape((T, Nosc))
            cnt += 1
        
        
        coeff_cos = np.zeros((Nosc, Nosc, P))
        coeff_sin = np.zeros((Nosc, Nosc, P))
        
        for p in range(P):
            coeff_cos[:,:,p] = beta[-1, :, 2*p].reshape(Nosc, Nosc)
            coeff_sin[:,:,p] = beta[-1, :, 2*p+1].reshape(Nosc, Nosc)
        
        self.beta      = beta
        self.omega     = OMEGA
        self.coeff_cos = coeff_cos
        self.coeff_sin = coeff_sin
        
        self.loglike   = L
        self.y_hat     = y_hat
        self.S         = S 
        self.Kb0       = Kb0
        
        # return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0
##############################################################################    
    def make_fourier_features(self, x, idx):
        Nosc   = self.Nosc
        T      = self.T
        P      = self.P
        h      = self.h
        
        #########################################################################
        phase_diff = np.zeros((Nosc, Nosc))
        theta      = x[idx-1, :]
        for i in range(Nosc):
            for j in range(Nosc):
                phase_diff[i,j] = np.mod(theta[j] - theta[i], 2*np.pi)
        
        for p in range(1, P+1):
            tmp_sin   = np.sin(p * phase_diff)
            tmp_cos   = np.cos(p * phase_diff)
            tmp_cos   = tmp_cos - np.eye(Nosc)
            
            p_fourier = np.concatenate((tmp_cos, tmp_sin), axis=1)
            
            if p ==1:
                x_train = p_fourier
            elif p>1:
                x_train = np.concatenate((x_train, p_fourier), axis=1)
            
        x_train = np.concatenate((x_train, np.ones((Nosc, 1))), axis=1)
          
        #########################################################################
        y_train = np.zeros((1, Nosc))
        for n in range(Nosc):
            theta_unwrap = np.unwrap(deepcopy(x[idx-1:idx+1, n]))
            y_train[:, n] = (theta_unwrap[1] - theta_unwrap[0])/h

        y_train = y_train.reshape(-1, order='C')
        #########################################################################
        self.x_train = x_train
        self.y_train = y_train
##############################################################################
    
    def make_Dx(self):#(X, T, N, P):
        X    = self.x_train
        T    = self.T
        N    = self.Nosc
        P    = 2*self.P
        
        Dx   = np.zeros((N*T,  N*(N*P+1)*T), dtype=float)
        
        cnt = 0;
        
        if X.shape[0] == 1:
            order_index = np.sort(repmat(np.arange(0, T), 1, N))
            order_index = order_index[0,:]
        else:
            order_index = np.arange(0, X.shape[0])
            
        for i in order_index:#range(1, (X.shape[0])*(X.shape[1]-1) ):
            tmp_x = deepcopy(X[i, :]).reshape(-1)
            
            idx = np.arange(cnt*(N*P+1), (cnt+1)*(N*P+1), 1)
                
            Dx[cnt, idx] = tmp_x
            cnt += 1
            
        return Dx
    
##############################################################################
    def update_coeff(self, mu_beta0, Kb0, sigma0):#(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
        Y  = self.y_train
        Dx = self.Dx
        T  = self.T
        N  = self.Nosc
        
        def mylogdet(S):
            L       = np.linalg.cholesky(S)
            logdetS = 2*np.sum(np.log(np.diag(L)))
            
            return logdetS
    
        def inv_use_cholensky(M):
            L     = np.linalg.cholesky(M)
            L_inv = np.linalg.inv(L)
            M_inv = np.dot(L_inv.T, L_inv)
            
            return M_inv
        
        ############# Estimate posterior distribution #############################
        sigma     = sigma0 * np.eye(N*T) + Dx @ Kb0 @ Dx.T # (1/sigma0) * np.eye(N*T) + Dx @ Kb0 @ Dx.T # 
        sigma_inv = inv_use_cholensky(sigma)
        
        KbDx = Kb0 @ Dx.T
        DxKb = Dx @ Kb0
        
        Yv    = Y.reshape(-1) - np.dot(Dx, mu_beta0).reshape(-1)
        #### update covariance
        Kb      = Kb0 - KbDx @ sigma_inv @ DxKb 
        #### update mean
        mu_beta = mu_beta0 + KbDx @ sigma_inv @ Yv 
        ############# Calculate log-likelihood ####################################
        
        Ndim    = Y.shape[0]
        loglike = 0.5 * (-Ndim * np.log(2*np.pi) - mylogdet(sigma) - Yv @ sigma_inv @ Yv)
        
        ###########################################################################
        return mu_beta, Kb, sigma, loglike



    