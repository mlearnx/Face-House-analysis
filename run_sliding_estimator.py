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
from mne.decoding import SlidingEstimator, cross_val_multiscore

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


###############################################################################
# Then we write a function to do time decoding on one subject

ana_path = '/home/claire/DATA/Data_Face_House_new_proc'
results_path = os.path.join(ana_path, 'Analysis', 'Sliding_Estimator')

def run_time_decoding(subject_id, condition1, condition2):
    print("processing subject: %s (%s vs %s)"
          % (subject_id, condition1, condition2))

    subject = "S%02d" % subject_id
    data_path = os.path.join(ana_path, subject,'EEG', 'New_Preproc' )
    epochs = mne.read_epochs(os.path.join(data_path,
                             '%s-causal-highpass-2Hz-epo.fif' %subject))

    # We define the epochs and the labels
    epochs = mne.concatenate_epochs([epochs[condition1],
                                    epochs[condition2]])
    epochs.apply_baseline()

    # Let us restrict ourselves to the MEG channels, and also decimate to
    # make it faster (although we might miss some detail / alias)
    epochs.pick_types(eeg=True).decimate(2, verbose='error')
    mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
    mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)

    # Get the data and labels
    X = epochs.get_data()
    # fit and time decoder
    le=LabelEncoder()
    y = le.fit_transform(epochs.events[:, 2])  # target: Audio left or right

    # Use AUC because chance level is same regardless of the class balance
    se = SlidingEstimator(
        make_pipeline(StandardScaler(), LogisticRegression()),
        scoring='roc_auc', n_jobs=1)
    scores = cross_val_multiscore(se, X=X, y=y, cv=StratifiedKFold())

    # let's save the scores now
    cond1 = condition1.replace('/', '-')
    cond2 = condition2.replace('/', '-')
    a_vs_b = '%s_vs_%s' % (cond1,cond2)
    fname_td = os.path.join(results_path, '%s-causal-highpass-2Hz-td-auc-%s.mat'
                            % (subject, a_vs_b))
    savemat(fname_td, {'scores': scores, 'times': epochs.times})


# Here we go parallel inside the :class:`mne.decoding.SlidingEstimator`
# so we don't dispatch manually to multiple jobs.


exclude =[7]


#for subject_id in range(1,26):
#    if subject_id in exclude:
#        continue
#    else:
#        run_time_decoding(subject_id, 'stim/face', 'stim/house')
#        
#for subject_id in range(1,26):
#    if subject_id in exclude:
#        continue
#    else:
#        run_time_decoding(subject_id, 'imag/face', 'imag/house')



for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    else:
        run_time_decoding(subject_id, 'stim', 'imag')
