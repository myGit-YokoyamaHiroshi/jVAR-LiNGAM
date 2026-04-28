# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:37:46 2024

@author: H.Yokoyama
"""

fdir = __file__ 
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os

current_path = os.path.dirname(__file__)
os.chdir(current_path)


current_path = os.getcwd()
param_path   = current_path + '/save_data/' 
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)
    
import matplotlib.pylab as plt
from matplotlib import font_manager
import matplotlib
if os.name == 'posix': # for linux
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
    matplotlib.rc('font', family="Arial")

plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)
from mne.io import read_raw_edf
from joblib import Parallel, delayed
from scipy import signal as sig
import numpy as np
import scipy as sci
import mne
#%% check the directory and get the file name automatically
name     = []
ext      = []
file_dir = current_path + '/raw_data/' 
for file in os.listdir(file_dir):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
#%%
fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

data_save_dir = current_path + '/save_data/preprocess/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
#%%

remove_ch = [ 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6']
# del mat
fs_dwn = 100
for fname, extention in zip(name, ext):
    if extention == '.edf':
        #%%
        
        f_edf    = file_dir + fname  + extention
        raw      = read_raw_edf(f_edf)
        fs       = raw.info['sfreq']
        
        #%%
        if fname == 'chb15_06':
            onset_time  = 272# unit: sec
            offset_time = 397# unit: sec
            
            onset_idx   = int(onset_time*fs_dwn)
            offset_idx  = int(offset_time*fs_dwn)
        #%%
        
        ch_names = []
        for ch in raw.ch_names:
            if ('Ref' in ch) | ('FP' in ch) | ('FT' in ch) | ('--' in ch) | ('-0' in ch) | ('FP1-F7' in ch) | ('FP2-F8' in ch) | ('P7-O1' in ch):
            # if ('Ref' in ch) | ('FP' in ch) | ('FT' in ch) | ('T7-' in ch) | ('T8-' in ch) | ('--' in ch) | ('-0' in ch) | ('FP1-F7' in ch) | ('FP2-F8' in ch) | ('P7-O1' in ch):
                raw.drop_channels(ch)
            else:
                rename = ch.split('-')[0]
                ch_names.append(rename)
            # rename = ch.split('-')[0]
            # # ch_names.append(rename)
            # if rename == '':
            #     raw.drop_channels(ch)
            # elif rename == 'Ref':
            #     rename = ch.split('-')[0]
            #     ch_names.append(rename)
            # else:
            #     ch_names.append(rename)
            
        freqs     = (60, 120)
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        
        filt_data = mne.filter.notch_filter(x=raw.get_data(), 
                                            Fs=fs, 
                                            freqs=freqs, 
                                            picks=eeg_picks)
        
        raw = mne.io.RawArray(filt_data, raw.info)
        
        if fs != fs_dwn:
            raw.resample(fs_dwn)
            fs = fs_dwn
            
        time = raw.times
        Nch  = len(ch_names)
        Nt   = len(time)
        
        #%% apply bandstop 
        filt_data   = raw.get_data() * 1E+6 # V to uV, size: ch x Nt
        
        eeg_raw = deepcopy(filt_data)
        #%%
        ### show time-series of theta oscillation during seizure
        raw.plot(start=240, duration=180, 
                 scalings=80E-6, 
                 show_scrollbars=False, 
                 show_scalebars=False, 
                 lowpass=8, highpass=4);
        
        plt.show()
        
        ### show power spectrum density in ictal state
        fig, ax = plt.subplots(); 
        raw.plot_psd(tmin=onset_time, tmax=offset_time, fmax=50, ax=ax); 
        ax.set_ylim(-30, 30); 
        ax.set_xticks(np.arange(0,60,1));
        ax.set_title('ictal'); 
        ax.set_xlim(0,15)
        plt.show();
        
        ### show power spectrum density in interictal state (0~100s)
        fig, ax = plt.subplots();
        raw.plot_psd(tmin=0, tmax=100, fmax=50, ax=ax); 
        ax.set_ylim(-30, 30); 
        ax.set_xticks(np.arange(0,60,1));
        ax.set_title('interictal'); 
        ax.set_xlim(0,15)
        plt.show();
        #%% Extract theta phase 
        trans_width = .1   # Width of transition from pass band to stop band, Hz
        numtaps     = 6001   # Size of the FIR filter.
        band        = [6, 7]
        # band        = [5.8, 6.3]
        # band        = [5, 6]
        b           = sig.firwin(numtaps, cutoff = band, 
                                  fs=fs, 
                                  width = trans_width, 
                                  window = "hanning",
                                  pass_zero = 'bandpass')
        a          = 1            
        eeg_theta  = sig.filtfilt(b, a, filt_data -  filt_data.mean(axis=0))
        
        phi_theta  = np.zeros(eeg_theta.shape)
        for ch in range(Nch):
            tmp             = deepcopy(eeg_theta[ch,:])
            phi             = np.mod(np.unwrap(np.angle(sig.hilbert(tmp))), 2*np.pi)
            phi_theta[ch,:] = phi
        #%% Extract alpha phase 
        trans_width = .1   # Width of transition from pass band to stop band, Hz
        numtaps     = 6001   # Size of the FIR filter.
        # band        = [55, 65]
        band        = [10, 12]
        b           = sig.firwin(numtaps, cutoff = band, 
                                  fs=fs, 
                                  width = trans_width, 
                                  window = "hanning",
                                  pass_zero = 'bandpass')
        a          = 1            
        eeg_alpha  = sig.filtfilt(b, a, filt_data -  filt_data.mean(axis=0))
        
        phi_alpha  = np.zeros(eeg_alpha.shape) 
        for ch in range(Nch):
            tmp             = deepcopy(eeg_alpha[ch,:])
            phi             = np.mod(np.unwrap(np.angle(sig.hilbert(tmp))), 2*np.pi)
            phi_alpha[ch,:] = phi
            
        #%%
        
        #%% save data 
        save_dict                = {} 
        save_dict['fs']          = fs
        save_dict['ch_names']    = ch_names
        save_dict['ch_names_bp'] = raw.ch_names
        save_dict['time']        = time
        save_dict['eeg_raw']     = eeg_raw
        save_dict['eeg_theta']   = eeg_theta
        save_dict['eeg_alpha']   = eeg_alpha
        save_dict['phi_theta']   = phi_theta
        save_dict['phi_alpha']   = phi_alpha
        save_dict['onset_time']  = onset_time
        save_dict['offset_time'] = offset_time
        save_dict['onset_idx']   = onset_idx
        save_dict['offset_idx']  = offset_idx
        
        raw.close()
        del raw
        #%%
        save_name   = 'preprocess_' + fname
        fullpath_save   = data_save_dir + save_name 
        np.save(fullpath_save, save_dict)
        
        
        