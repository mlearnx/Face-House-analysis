"""
============================================================================
Decoding in time-frequency space data using the Common Spatial Pattern (CSP)
============================================================================

The time-frequency decomposition is estimated by iterating over raw data that
has been band-passed at different frequencies. This is used to compute a
covariance matrix over each epoch or a rolling time-window and extract the CSP
filtered signals. A linear discriminant classifier is then applied to these
signals.
"""
# Authors: Laura Gwilliams <laura.gwilliams@nyu.edu>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Alex Barachant <alexandre.barachant@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import os
import mne
from mne import io

from mne import Epochs, create_info
from mne.io import read_raw_edf
from mne.decoding import CSP
from mne.time_frequency import AverageTFR
from mne.parallel import parallel_func

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder



def run_csp_timefreq_decoding(subject_id, cond1, cond2, event_id):

    subject = "S%02d" %subject_id
    data_path= '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/No_Low_pass/'
    events_path = '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/'
    
    print '-----Now Processing %s -------' %subject

    raw = mne.io.read_raw_fif(data_path+subject+'-raw.fif', preload=True)
    events =  mne.read_events(events_path + subject + '-eve.fif')
    # Extract information from the raw file
    sfreq = raw.info['sfreq']
    # Assemble the classifier using scikit-learn pipeline
    clf = make_pipeline(CSP(n_components=4, reg='oas', log=True),
                        LinearDiscriminantAnalysis())
    n_splits = 5 # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Classification & Time-frequency parameters
    tmin, tmax = -0.5, 1.5
    n_cycles = 12.  # how many complete cycles: used to define window size
    min_freq = 2
    max_freq = 30 #25.
    n_freqs = 10 #8  # how many frequency bins to use

    # Assemble list of frequency range tuples
    freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples
    
    # Infer window spacing from the max freq and number of cycles to avoid gaps
    window_spacing = (n_cycles / np.max(freqs) / 2.)
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)
    
    # Instantiate label encoder
    le = LabelEncoder()
    
    ###############################################################################
    # Loop through frequencies, apply classifier and save scores
    
    # init scores
    freq_scores = np.zeros((n_freqs - 1,))
    
    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in enumerate(freq_ranges):
    
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
    
        # Apply band-pass filter to isolate the specified frequencies
        raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    
        # Extract epochs from filtered data, padded by window size
        epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                        proj=False, baseline=None, preload=True)
        epochs.drop_bad()
        #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
        #mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
        
        y = le.fit_transform(epochs.events[:, 2])
    
        X = epochs.get_data()
    
        # Save mean scores over folds for each frequency and time window
        freq_scores[freq] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                    scoring='roc_auc', cv=cv,
                                                    n_jobs=1), axis=0)
    a_vs_b = '%s_vs_%s'%(cond1, cond2)
    fname_csp = os.path.join(data_path, '%s-csp-freq-%s.mat' %(subject, a_vs_b))
    
    from scipy.io import savemat
    savemat(fname_csp, {'scores':freq_scores, 'freqs': freqs })
    

    ###############################################################################
    # Loop through frequencies and time, apply classifier and save scores
    
    # init scores
    tf_scores = np.zeros((n_freqs - 1, n_windows))
    
    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in enumerate(freq_ranges):
    
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
    
        # Apply band-pass filter to isolate the specified frequencies
        raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    
        # Extract epochs from filtered data, padded by window size
        epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                        proj=False, baseline=None, preload=True)
        epochs.drop_bad()
        #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
        #mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
        y = le.fit_transform(epochs.events[:, 2])
    
        # Roll covariance, csp and lda over time
        for t, w_time in enumerate(centered_w_times):
    
            # Center the min and max of the window
            w_tmin = w_time - w_size / 2.
            w_tmax = w_time + w_size / 2.
    
            # Crop data into time-window of interest
            X = epochs.copy().crop(w_tmin, w_tmax).get_data()
    
            # Save mean scores over folds for each frequency and time window
            tf_scores[freq, t] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                         scoring='roc_auc', cv=cv,
                                                         n_jobs=1), axis=0)
    a_vs_b = '%s_vs_%s'%(cond1, cond2)
    fname_csp_tfr = os.path.join(data_path, '%s-csp-time-freq-%s.mat' %(subject, a_vs_b))
    
    from scipy.io import savemat
    savemat(fname_csp_tfr, {'scores':tf_scores, 'freqs': freqs, 'sfreq': sfreq, 'centered_w_times': centered_w_times })


parallel, run_func, _=parallel_func(run_csp_timefreq_decoding, n_jobs=6)
parallel(run_func(subject_id, 'imag-face', 'imag-house', {'imag/face':201, 'imag/house':202})    
        for subject_id in [1,2,3,4,5,6,8,9,10,11])
#parallel(run_func(subject_id, 'imagery', 'perception', {'imag/face':201, 'imag/house':202, 'stim/face': 101, 'stim/house':102})    
#        for subject_id in [1,2,3,4,5,6,8,9,10,11])
