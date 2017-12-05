# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:56:07 2017

 - Compute timne-frequency decomposition using morlet wavelet for each subject and condition
 - Save values 
 - plot distribution

@author: claire
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op


import mne
from mne import io

from mne import Epochs, create_info
from mne.parallel import parallel_func

from mne.time_frequency import tfr_morlet, psd_welch, psd_multitaper

exclude = [7]


event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}




test=[]
df=[]
df_temp=[]
conditions = ['stim', 'imag']
stimuli = ['face', 'house']

coord=[(0,0), (0,1), (0,2), (0,3)]



#freq_range = [['delta', 2, 4], ['theta', 4, 6],['low_alpha', 6, 8], ['alpha', 10, 12], ['beta', 15, 20],['low_gamma', 30, 45], ['gamma', 70,100]]

chan_range= ['POz', 'PO4', 'PO8', 'PO3', 'PO7']

for subject_id in range(1, 26):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG/Preproc')
        
    print '-----Now Processing %s -------' %subject   
    
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)

# select channels
   # epochs.pick_channels(chan_range)
# -----------------------------------
# Power spectrum density
# -----------------------------------
    fig, ax = plt.subplots(nrows=4, ncols=1)
    n=0
    for cond in event_id:
        epochs[cond].plot_psd_topomap(ch_type='eeg', normalize=True, axes=coord[n])
        ax[n].set_title('toto') #('%s' + cond %subject)
        n=n+1


















