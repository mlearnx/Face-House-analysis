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

# Define train and test conditions
train1 = 'stim/face'
train2 = 'stim/house'
test1 = 'imag/face'
test2 ='imag/house'

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    else:

        print("processing subject: %s (%s-%s vs %s-%s)"
              % (subject_id, train1, train2,  test1, test2))
    
        subject = "S%02d" % subject_id
        data_path = os.path.join(ana_path, subject,'EEG', 'New_Preproc' )
        epochs = mne.read_epochs(os.path.join(data_path,
                                 '%s-causal-highpass-2Hz-epo.fif' %subject))
    
        # We define the epochs and the labels
        #epochs =epochs[condition1, condition2]
        epochs.apply_baseline()
    
        # Let us restrict ourselves to the MEG channels, and also decimate to
        # make it faster (although we might miss some detail / alias)
        epochs.pick_types(eeg=True).decimate(2, verbose='error')
        #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
        #mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
       
        #--------------------------- 
        # average group of 4 trials
        #---------------------------
        data_train1 =  epochs[train1].get_data()
        data_train2 = epochs[train2].get_data()
        
        data_test1 =  epochs[test1].get_data()
        data_test2 = epochs[test2].get_data()
        
        mean_train1=[]
        ind_trial = 0
        while ind_trial<= len(data_train1)-5:
            mean_train1.append(mean(data_train1[ind_trial:(ind_trial+4)], 0))
            print ind_trial
            ind_trial+=5
        
        mean_train2=[]
        ind_trial = 0
        while ind_trial<= len(data_train2)-5:
            mean_train2.append(mean(data_train2[ind_trial:(ind_trial+4)], 0))
            print ind_trial
            ind_trial+=5
            
        mean_test1=[]
        ind_trial = 0
        while ind_trial<= len(data_test1)-5:
            mean_test1.append(mean(data_test1[ind_trial:(ind_trial+4)], 0))
            print ind_trial
            ind_trial+=5
        
        mean_test2=[]
        ind_trial = 0
        while ind_trial<= len(data_test2)-5:
            mean_test2.append(mean(data_test2[ind_trial:(ind_trial+4)], 0))
            print ind_trial
            ind_trial+=5
        
        
        #--------------------------- 
        # define decoding pipeline and run
        #---------------------------        
        
        # Use AUC because chance level is same regardless of the class balance
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
        time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')
            #
           
        # We will train the classifier on all stim face vs house trials
        # and test on all images face vs house trials.
        
        #le = LabelEncoder()
        
        # train on stim
        time_gen.fit(X=np.array(mean_train1 + mean_train2),
                     y = np.array([0] * len(mean_train1) + [1] * len(mean_train2)) )
        
        # score on imagery
        scores=time_gen.score(X=np.array(mean_test1 + mean_test2),
                     y = np.array([0] * len(mean_test1) + [1] * len(mean_test2)) )

        
        
        # let's save the scores now
        
        
        fname_td = os.path.join(results_path, '%s-causal-highpass-2Hz-temp-gene-across-conditions-stim_vs_imag-ave4trials.mat'
                                % (subject))
        savemat(fname_td, {'scores': scores, 'times': epochs.times})#, 'perm_scores': permutation_scores, 'pval' : pvalue})


# Here we go parallel inside the :class:`mne.decoding.SlidingEstimator`
# so we don't dispatch manually to multiple jobs.

