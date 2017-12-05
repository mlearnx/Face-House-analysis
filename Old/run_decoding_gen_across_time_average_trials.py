# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:24:19 2017

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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

cond= 'stim'
ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

exclude = [7]
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    #epochs.interpolate_bads()
    #all_epochs.append(epochs)
    
    #epochs = epochs=mne.concatenate_epochs(all_epochs)
    
    epochs=epochs[cond]
    
    epochs.crop(tmin=-0.2, tmax=1.5)
    
    # average group of 4 trials
    
    data_cond1 =  epochs[cond+'/face'].get_data()
    data_cond2 = epochs[cond+'/house'].get_data()
    
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
     
    clf = make_pipeline(StandardScaler(), LogisticRegression())
    
    # define the Temporal Generalization object
    time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')
    
    cv = StratifiedKFold(n_splits=3, shuffle=False)
    cv.get_n_splits(X, y)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=1)
    
    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)
    
    
     # save scores
    fname_td= os.path.join(data_path, '%s_gen_across_time_%s_ave_4_trials.mat' %(subject, cond))
    print 'Saving %s' %fname_td
    from scipy.io import savemat
    savemat(fname_td, {'scores': scores,
                       'times':epochs.times })
                       
                       
    
    
    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label='score')
    ax.axhline(.5, color='k', linestyle='--', label='chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('AUC')
    ax.legend()
    ax.axvline(.0, color='k', linestyle='-')
    ax.set_title('Decoding EEG sensors over time - %s subject %s' %(cond,subject))
    plt.show()
    
    plt.savefig(ana_path + ' gen_across_time_%s_%s_ave_4_trials.pdf' %(cond,subject),  bbox_to_inches='tight')
    
    # Plot the full matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
                   extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal Generalization - %s subject %s' %(cond, subject))
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)
    plt.show()
    plt.savefig(ana_path + ' gen_across_time_matrix_%s_%s_ave_4_trials.pdf' %(cond,subject),  bbox_to_inches='tight')
