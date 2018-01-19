"""
=================
Sliding estimator
=================

A sliding estimator fits a logistic legression model for every time point.
In this example, we contrast the condition 'famous' against 'scrambled'
using this approach. The end result is an averaging effect across sensors.
The contrast across different sensors are combined into a single plot.

Results script: :ref:`sphx_glr_auto_examples_statistics_plot_sliding_estimator.py`
"""  # noqa: E501

###############################################################################
# Let us first import the libraries

import os

import numpy as np
from scipy.io import savemat

import mne

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import permutation_test_score
from mne.decoding import (GeneralizingEstimator, LinearModel, cross_val_multiscore)

print(__doc__)

###############################################################################
# Then we write a function to do time decoding on one subject

ana_path = '/home/claire/DATA/Data_Face_House_new_proc'
results_path = os.path.join(ana_path, 'Analysis', 'Sliding_Estimator')


exclude =[7]

condition1 = 'stim/face'
condition2 = 'stim/house'

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    else:

        print("processing subject: %s (%s vs %s)"
              % (subject_id, condition1, condition2))
    
        subject = "S%02d" % subject_id
        data_path = os.path.join(ana_path, subject,'EEG', 'New_Preproc' )
        epochs = mne.read_epochs(os.path.join(data_path,
                                 '%s-causal-highpass-2Hz-epo.fif' %subject))
    
        # We define the epochs and the labels
        epochs =epochs[condition1, condition2]
        epochs.apply_baseline()
    
        # Let us restrict ourselves to the MEG channels, and also decimate to
        # make it faster (although we might miss some detail / alias)
        epochs.pick_types(eeg=True).decimate(2, verbose='error')
        #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
        #mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
        
         # average group of 4 trials
        
        data_cond1 =  epochs[condition1].get_data()
        data_cond2 = epochs[condition2].get_data()
        
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
    
        # Use AUC because chance level is same regardless of the class balance
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
        time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')
            #
        scores = cross_val_multiscore(time_gen, X=X, y=y, cv=StratifiedKFold())
    
        #cv=StratifiedKFold()
        
        #scores, permutation_scores, pvalue = permutation_test_score(estimator = se, X=X, y=y, groups=None, scoring = None, 
        #    cv=3, n_permutations=100)
        
        #print("********* %s Classification score %s (pvalue : %s) ***********" % (subject, scores, pvalue))
        
        
        
        # let's save the scores now
        cond1 = condition1.replace('/', '-')
        cond2 = condition2.replace('/', '-')
        a_vs_b = '%s_vs_%s' % (cond1,cond2)
        fname_td = os.path.join(results_path, '%s-causal-highpass-2Hz-temp-gene-auc-%s-ave4trials.mat'
                                % (subject, a_vs_b))
        savemat(fname_td, {'scores': scores, 'times': epochs.times})#, 'perm_scores': permutation_scores, 'pval' : pvalue})


# Here we go parallel inside the :class:`mne.decoding.SlidingEstimator`
# so we don't dispatch manually to multiple jobs.

