# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:41:46 2017

@author: claire
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op

import mne
from mne import io

from mne import Epochs, create_info
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from mne.parallel import parallel_func

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from mne.time_frequency import tfr_morlet

cond1, cond2 = 'stim', 'imag'

exclude = [7]

subject_id=12

event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}

for subject_id in range(12,13):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG/Preproc')
    dir_save= os.path.join(data_path, 'CSP-Timefreq')

    if not op.exists(dir_save):
        os.makedirs(dir_save)
        
    print '-----Now Processing %s -------' %subject   
    
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    
    mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
    mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)

    event_id= {'stim':100, 'imag': 200}
    
    #Time-frequency parameters
    fmin= 1.5
    fmax = 100
    freqs = np.exp(np.linspace(log(fmin), log(fmax), 65))
    n_cycles = np.concatenate((np.linspace(1, 8, 47), np.ones(18)*8)) # formule eeglab, allows both low and high freq
    zero_mean = True  #  correct morlet wavelet to be of mean zero
    
    # Assemble list of frequency range tuples
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples
    
    
    channel_ind = mne.pick_channels(epochs.info['ch_names'], ['PO8'])    
    
    epochs_power = list()
    for condition in [epochs[k] for k in event_id]:
        this_tfr = tfr_morlet(condition, freqs, n_cycles=n_cycles,
                               average=False, zero_mean=zero_mean,
                              return_itc=False)
        this_tfr.apply_baseline(mode='ratio', baseline=(None, 0))
        this_power = this_tfr.data[:, channel_ind, :, :]  # we only have one channel.
        epochs_power.append(this_power)

    


        
    # Assemble the classifier using scikit-learn pipeline
    clf = make_pipeline(CSP(n_components=4, reg= 'oas', log=True, norm_trace=False),
                        LinearDiscriminantAnalysis())
    n_splits = 5  # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    
    # Classification & 
    tmin, tmax = -0.5, 1
    n_freqs = 65 #8  # how many frequency bins to use
      # how many complete cycles: used to define window size

    # Instantiate label encoder
    le = LabelEncoder()
    
    
    
    ###############################################################################
    # Loop through frequencies, apply classifier and save scores
    
    # init scores
    freq_scores = np.zeros((n_freqs - 1,))
    
    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in enumerate(freq_ranges):
                    
            y = le.fit_transform(epochs.events[:, 2])
        
            X = epochs.get_data()
        
            # Save mean scores over folds for each frequency and time window
            freq_scores[freq] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                        scoring='roc_auc', cv=cv,
                                                        n_jobs=1), axis=0)
    a_vs_b = '%s_vs_%s'%(cond1, cond2)
    fname_csp = os.path.join(dir_save, '%s-csp-freq-%s.mat' %(subject, a_vs_b))
    
    from scipy.io import savemat
    savemat(fname_csp, {'scores':freq_scores, 'freqs': freqs })
    