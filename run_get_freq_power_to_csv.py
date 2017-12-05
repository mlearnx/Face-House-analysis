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
import pandas as pd
import seaborn as sns

import mne
from mne import io

from mne import Epochs, create_info
from mne.parallel import parallel_func

from mne.time_frequency import tfr_morlet, psd_welch


exclude = [7]


event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}




test=[]
df=[]
df_temp=[]
conditions = ['stim', 'imag']
stimuli = ['face', 'house']
freq_range = [['delta', 2, 4], ['theta', 4, 6],['low_alpha', 6, 8], ['alpha', 10, 12], ['beta', 15, 20],['low_gamma', 30, 45], ['gamma', 70,100]]

chan_range= ['POz', 'PO4', 'PO8', 'PO3', 'PO7']

for subject_id in range(1, 26):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG/Preproc')
    dir_save= os.path.join(data_path, 'CSP-Timefreq')

    if not op.exists(dir_save):
        os.makedirs(dir_save)
        
    print '-----Now Processing %s -------' %subject   
    
    fname_in = os.path.join(data_path, '%s-clean-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)

# select channels
    epochs.pick_channels(chan_range)
# -----------------------------------
# Power spectrum density
# -----------------------------------

    for cond in conditions:
        for stim in stimuli:
            for freq, fmin, fmax in freq_range:
                    eve_id = cond + '/' + stim
                    cond_epochs= epochs[eve_id].copy().crop(0, 1.5)
                    
                    cond_psd = psd_welch(cond_epochs, fmin=fmin, fmax=fmax)
                    
                    test.append([subject, cond, stim, freq, np.median(cond_psd[0]) ])
    
    
df=pd.DataFrame(test, columns=['Subject', 'Modality', 'Stim',  'Freq', 'Median'])
df=df.set_index('Subject')

df.to_csv('power_density_all_freq_stim_type_occ_elec.csv')


#
## -----------------------------------
## Plotting in Python
## -----------------------------------
#
#sns.set(style="whitegrid", color_codes=True)
#
#
#alpha = df[df['Freq'] == 'alpha']
#gamma = df[df['Freq'] == 'gamma']
#
#plt.figure()
#sns.violinplot(x=alpha['Modality'],y=alpha['Median'], inner=None )
#sns.swarmplot(x=alpha['Modality'],y=alpha['Median'], color='w', alpha=.5)
#plt.title('10-12 Hz Median Alpha')
#
#plt.figure()
#sns.violinplot(x=gamma['Modality'],y=gamma['Median'], inner=None )
#sns.swarmplot(x=gamma['Modality'],y=gamma['Median'], color='w', alpha=.5)
#plt.title('70-100 Hz Median Gamma')
#
#
## plot the difference of power between the conditions
#
#
#alpha_stim = df[(df.Modality == 'stim') & (df.Freq == 'alpha')]
#alpha_im = df[(df.Modality == 'imag') & (df.Freq == 'alpha')]
#alpha_diff = alpha_stim.Median-alpha_im.Median
#
#gamma_stim = df[(df.Modality == 'stim') & (df.Freq == 'gamma')]
#gamma_im = df[(df.Modality == 'imag') & (df.Freq == 'gamma')]
#gamma_diff = gamma_stim.Median-gamma_im.Median
#
#
#
#
#plt.figure()
#sns.violinplot(x=alpha_diff ,y=gamma['Median'], inner=None )
#sns.swarmplot(x=gamma['Modality'],y=gamma['Median'], color='w', alpha=.5)
#plt.title('70-100 Hz Median Gamma')
#
#
#






