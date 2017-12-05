# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:07:21 2017

@author: claire
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
import os
from mne.parallel import parallel_func

from sklearn.model_selection import StratifiedKFold


def run_time_decoding(subject_id, cond1, cond2, event_id):
    
    subject = "S%02d" %subject_id
    
    print subject, cond1, cond2    


    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/')
    
    raw_fname = data_path + subject + '-raw.fif'
    event_fname = data_path + subject +'-eve.fif'
    tmin, tmax = -0.5, 1.5

    raw = mne.io.read_raw_fif(raw_fname, preload=True)

    events = mne.read_events(event_fname)

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True, eog=True,
                           exclude='bads')
    
    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0.), preload=True,
                        decim=4)
    
    epochs.pick_types(eeg=True, exclude='bads')

    #only look at occipital channels
#    select_chans = [u'Iz', u'Oz', u'O1', u'O2', u'O3', u'PO7', u'PO8', u'POz', u'PO1', u'PO3', u'PO2', u'PO4']
    select_chans = [ u'PO7', u'PO8']
    #select_chans = [ u'Cz', u'FPz']

    ch_names=[ch_name.replace('', '') for ch_name in select_chans]
    epochs.pick_types(eeg=True).pick_channels(ch_names)
    
    # average group of 4 trials
    
    data_cond1 =  epochs['imag/face'].get_data()
    data_cond2 = epochs['imag/house'].get_data()
    
    mean_cond1=[]
    ind_trial = 0
    while ind_trial<= len(data_cond1)-5:
        mean_cond1.append(mean(data_cond1[ind_trial:(ind_trial+4)], 0))
        print ind_trial
        ind_trial+=5
    
    mean_cond2=[]
    ind_trial = 0
    while ind_trial<= len(data_cond2)-5:
        mean_cond2.append(mean(data_cond2[ind_trial:(ind_trial+4)], 0))
        print ind_trial
        ind_trial+=5
    
    X=[]
    # create variable for decoding
    X = mean_cond1 + mean_cond2
    X=np.array(X)
    y = np.array([0] * len(mean_cond1) + [1] * len(mean_cond2))     
    # fit and time decoder
    #X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
    #y = epochs.events[:, 2]  # target: Audio left or right
    
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    cv.get_n_splits(X, y)

    clf = make_pipeline(StandardScaler(), LogisticRegression())

    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')

    #scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)

    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)


    # save scores
    a_vs_b = '%s_vs_%s' %(cond1,cond2)
    print 'a_vs_b = %s' %a_vs_b    
    
    fname_td= os.path.join(data_path, '%s-td-auc-%s_ave_4_trials_po7_po8.mat' %(subject, a_vs_b))
    print 'Saving %s' %fname_td
    from scipy.io import savemat
    savemat(fname_td, {'scores': scores,
                       'times':epochs.times })
    
    
    
parallel, run_func, _=parallel_func(run_time_decoding, n_jobs=1)
#parallel(run_func(subject_id, 'stim-face', 'stim-house', {'stim/face':101, 'stim/house':102})    
#        for subject_id in [1,2,3,4,5,6,8,9,10,11])
parallel(run_func(subject_id, 'imag-face', 'imag-house', {'imag/face':201, 'imag/house':202})    
        for subject_id in [1,2,3,4,5,6,8,9,10,11])

    
    
    
    
    
    
    
    