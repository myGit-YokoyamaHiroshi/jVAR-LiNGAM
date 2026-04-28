# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:35:23 2023

@author: H.Yokoyama
"""
import sys
import numpy as np
from scipy.special import digamma, gammaln

class CouplingOscillator:
    def __init__(self, Nosc, P, h, max_iter=500, varbose=False):
        self.P        = P
        self.h        = h
        self.max_iter = max_iter
        self.ELBO     = -sys.float_info.max
        
        #### initial setting for gamma priors
        self.a0       = 1E-6
        self.b0       = 1E-6
        self.c0       = 1E-6
        self.d0       = 1E-6
        
        self.varbose  = varbose
    
    def fit_model(self, dtheta, theta, T):
        P         = self.P
        Nt, Nosc  = theta.shape
        self.Nt   = Nt
        self.Nosc = Nosc
        self.dim  = Nosc*(Nosc*2*P+1)
        ########################################################################
        ########## batch regression ###########################################
        ########################################################################
        if T==Nt: ## fitting with all sample (batch)
            Dx  = self.make_design_matrix(theta)
            Y   = dtheta.reshape(-1,order='f')
            
            mu_beta0 = np.zeros(self.dim)
            K_beta0  = np.eye(self.dim)
            
            yPred, mu_new, K_new, elbo_save = self.vb_regression(Y, Dx, mu_beta0, K_beta0)
            
            #### store estimated value (before reshaping)
            self.mu_beta = mu_new
            self.Kbeta   = K_new
            
            #### reshape estimated value
            yPred    = yPred.reshape(dtheta.shape, order='f') 
            coeff_vb = mu_new.reshape(Nosc, 2*P*Nosc+1)
            
            Kest_vb  = coeff_vb[:, 1:]
            K1est_vb = Kest_vb[:, :P*Nosc].reshape((Nosc,Nosc,P), order='f')
            K2est_vb = Kest_vb[:, P*Nosc:].reshape((Nosc,Nosc,P), order='f')
            
            #### store estimated value with reshaped format
            self.K1Pred      = K1est_vb # cosine coefficient for P-th Fourier series
            self.K2Pred      = K2est_vb #   sine coefficient for P-th Fourier series
            self.omegaPred   = coeff_vb[:,0] # natural frequency
            self.yPred       = yPred# estimated phase velosity
            self.elbo_save   = np.array(elbo_save)
            
            self.sigma_obs   = (self.d0/self.c0) * self.h
            self.sigma_param = self.b0/self.a0
            
        ########################################################################
        ########## sequential regression ###################################### 
        ########################################################################
        elif T<Nt: ## fitting using sliding window T with step size=1 (sequential fitting)
            dt          = self.h    
            Nepc        = int(Nt/T)
            idx_tmp     = np.arange(0, Nt+T, T)
            idx_epc     = np.vstack((idx_tmp[:-1], idx_tmp[1:])).T
            #### initial setting of the multivariate normal distribution (prior for model parameters)
            mu_beta0    = np.zeros(self.dim)
            K_beta0     = np.eye(self.dim)
            #### parameter initialization
            yPred       = np.zeros((Nepc, T, Nosc))
            K1Pred      = np.zeros((Nepc, Nosc, Nosc,P))
            K2Pred      = np.zeros((Nepc, Nosc, Nosc,P))
            omegaPred   = np.zeros((Nepc, Nosc))
            
            elbo_save   = np.zeros(Nepc)
            sigma_obs   = np.zeros(Nepc)
            sigma_param = np.zeros(Nepc)
            coeff_vb    = np.zeros((Nepc, Nosc, Nosc*2*P+1))
            
            for i, idx in enumerate(idx_epc):
                Y   = dtheta[idx[0]:idx[1],:].reshape(-1,order='f')
                X   = theta[idx[0]:idx[1],:]
                
                Dx  = self.make_design_matrix(X)
                res = self.vb_regression(Y, Dx, mu_beta0, K_beta0)
                #### store result 
                yPred[i,:,:]    = res[0].reshape((T, Nosc),order='f')
                mu_beta0          = res[1]
                K_beta0           = res[2]
                elbo_save[i]    = res[3][-1]
                
                sigma_obs[i]    = self.d0/self.c0
                sigma_param[i]  = self.b0/self.a0
                
                coeff_vb = mu_beta0.reshape(Nosc, 2*P*Nosc+1)
                omega_vb = coeff_vb[:,0]
                Kest_vb  = coeff_vb[:, 1:]
                
                K1Pred[i,:,:,:] = Kest_vb[:, :P*Nosc].reshape((Nosc,Nosc,P), order='f')
                K2Pred[i,:,:,:] = Kest_vb[:, P*Nosc:].reshape((Nosc,Nosc,P), order='f')
                omegaPred[i,:]  = omega_vb
                
                print('(Epoch: %d / %d) cov_par = %.6f, cov_obs = %.6f'%(i+1, Nepc, self.b0/self.a0, (self.d0/self.c0)*dt))
            
            # yPred       = np.zeros(dtheta.shape)
            # K1Pred      = np.zeros((Nt - T + 1, Nosc, Nosc,P))
            # K2Pred      = np.zeros((Nt - T + 1, Nosc, Nosc,P))
            # omegaPred   = np.zeros((Nt - T + 1, Nosc))
            
            # elbo_save   = np.zeros(Nt - T + 1)
            # sigma_obs   = np.zeros(Nt - T + 1)
            # sigma_param = np.zeros(Nt - T + 1)
            # coeff_vb    = np.zeros((Nt, Nosc, Nosc*2*P+1))
            
            # for t in range(T, Nt+1):
            #     Y   = dtheta[t-T:t,:].reshape(-1,order='f')
            #     X   = theta[t-T:t,:]
            #     Dx  = self.make_design_matrix(X)
            #     res = self.vb_regression(Y, Dx, mu_beta0, K_beta0)
            #     #### store result 
            #     yPred[t-T:t,:]    = res[0].reshape((T, Nosc),order='f')
            #     mu_beta0          = res[1]
            #     K_beta0           = res[2]
            #     elbo_save[t-T]    = res[3][-1]
                
            #     sigma_obs[t-T]    = self.d0/self.c0
            #     sigma_param[t-T]  = self.b0/self.a0
                
            #     coeff_vb = mu_beta0.reshape(Nosc, 2*P*Nosc+1)
            #     omega_vb = coeff_vb[:,0]
            #     Kest_vb  = coeff_vb[:, 1:]
                
            #     K1Pred[t-T,:,:,:] = Kest_vb[:, :P*Nosc].reshape((Nosc,Nosc,P), order='f')
            #     K2Pred[t-T,:,:,:] = Kest_vb[:, P*Nosc:].reshape((Nosc,Nosc,P), order='f')
            #     omegaPred[t-T,:]  = omega_vb
                
            #     if np.mod(t+1, 10)==0:
            #         print('(iter: %d / %d) cov_par = %.6f, cov_obs = %.6f'%(t-T+1, Nt-T, self.b0/self.a0, (self.d0/self.c0)*dt))
            #### store estimated value with reshaped format
            self.K1Pred      = K1Pred    # cosine coefficient for P-th Fourier series
            self.K2Pred      = K2Pred    #   sine coefficient for P-th Fourier series
            self.omegaPred   = omegaPred # natural frequency
            self.yPred       = yPred     # estimated phase velosity
            self.sigma_obs   = sigma_obs * dt
            self.sigma_param = sigma_param
            self.elbo_save   = elbo_save
            
    ############ self-defined function
    def make_feature_matrix(self, theta):
        Nosc       = self.Nosc
        P          = self.P
        theta_diff = (theta[:, np.newaxis]-theta[np.newaxis,:]).T
        sin_mat    = np.zeros((Nosc,Nosc,P))
        cos_mat    = np.zeros((Nosc,Nosc,P))
        
        for p in range(1,P+1):
            tmp_cos  = np.cos(p*theta_diff)
            diag_val = np.diag(np.diag(tmp_cos))
            cos_mat[:,:,p-1] = tmp_cos - diag_val
            
            sin_mat[:,:,p-1] = np.sin(p*theta_diff)
            
        return cos_mat, sin_mat
    #######################
    def make_design_matrix(self, theta_all):
        Nosc       = self.Nosc
        P          = self.P
        T          = theta_all.shape[0]
        
        cos_mat = np.zeros((T, Nosc,Nosc,P))
        sin_mat = np.zeros((T, Nosc,Nosc,P))
        
        for t in range(T):
            res  = self.make_feature_matrix(theta_all[t,:])
            cos_mat[t,:,:,:] = res[0]
            sin_mat[t,:,:,:] = res[1]
            
        Dx      = np.zeros((Nosc*(Nosc*2*P+1), Nosc*T))
        for i in range(Nosc):
            cos_i = np.array([cos_mat[t,i,:,:].reshape(-1, order='f') for t in range(T)]).T
            sin_i = np.array([sin_mat[t,i,:,:].reshape(-1, order='f') for t in range(T)]).T
            
            idx_h = np.arange(i*(T), (i+1)*(T), 1)
            idx_v = np.arange(i*(Nosc*2*P+1), (i+1)*(Nosc*2*P+1), 1)
            
            Dx[np.ix_(idx_v, idx_h)] = np.vstack((np.ones((1,T)), cos_i, sin_i))
        
        return Dx

    #### variational bayesian regression (VBEM) 
    def vb_regression(self, Y, Dx, mu_beta, Kb):
        # print(Dx.shape)
        
        dt        = self.h
        Llast     = self.ELBO
        maxiter   = self.max_iter
        a         = self.a0
        b         = self.b0
        c         = self.c0
        d         = self.d0
        ############# Estimate posterior distribution #############################
        XX      = Dx @ Dx.T
        Xy      = Dx @ Y.T
        
        Nbeta     = len(mu_beta)
        Ny        = len(Y)
        
        R         = dt
        elbo_save = [] 
        
        for i in range(maxiter):
            A       = a/b
            B       = c/d        
            #### update covariance
            aI      = (A*np.eye(Nbeta))
            
            # K_inv   = aI + (B/R)*XX
            # K_new   = np.linalg.solve(K_inv, np.eye(Nbeta))
            K_new   = np.linalg.inv(aI + (B/R)*XX) 
            #### update mean
            mu_new  = K_new @ (mu_beta@aI + (B/R)*Xy) 
            #### update gamma prior
            yPred   = Dx.T @ mu_new
            err     = Y - yPred
            
            a       = a + Nbeta/2
            b       = b + (1/2) * (np.trace(K_new))
            c       = c + Ny/2
            d       = d + (1/(2*R)) * np.sum(err**2) + (1/2) * np.trace(Dx.T@K_new@Dx)
            ###########################################################################
            #### Calculate evidence lower bound
            ###### Likelihood terms
            logPy    =    Ny/2 * (digamma(c) - np.log(d)) -    Ny/2 * np.log(2*np.pi) - 1/(2*R) * (c/d) * (np.sum(err**2) + np.trace(Dx.T@K_new@Dx))
            logPbeta = Nbeta/2 * (digamma(a) - np.log(b)) - Nbeta/2 * np.log(2*np.pi) - 1/2 * (a/b) * (np.trace(K_new))
            logPalpa = self.a0 * np.log(self.b0) - gammaln(self.a0) + (self.a0-1) * (digamma(a)-np.log(b)) - self.b0 * (a/b)
            logPtau  = self.c0 * np.log(self.d0) - gammaln(self.c0) + (self.c0-1) * (digamma(c)-np.log(d)) - self.d0 * (c/d) 
            ###### Shannon entropy terms
            _, logdetK = np.linalg.slogdet(K_new) 
            Hbeta   = 1/2 * logdetK + Nbeta/2 * (1 + np.log(2*np.pi))
            Halpa   = a - np.log(b) + gammaln(a) + (1-a)*digamma(a) 
            Htau    = c - np.log(d) + gammaln(c) + (1-c)*digamma(c)
            ###### 
            ELBO    = (logPy + logPbeta + logPalpa + logPtau + Hbeta + Halpa + Htau)/self.Nt
            ###########################################################################
            
            elbo_save.append(ELBO)
            
            if self.varbose == True:
                
                if np.mod(i+1, 10)==0:
                    print('iter: %d/%d -- ELBO = %.2f'%(i+1, maxiter, ELBO))
                    
            
            if abs(Llast - ELBO)/abs(Llast) < 1E-6:
                break
            Llast   = ELBO
            mu_beta = mu_new
            Kb      = K_new
        
        self.a0   = a
        self.b0   = b
        self.c0   = c
        self.d0   = d
        self.ELBO = ELBO
        
        return yPred, mu_new, K_new, elbo_save