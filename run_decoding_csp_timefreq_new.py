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
import os
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs, find_events, create_info
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

###############################################################################
# Set parameters and read data
cond1, cond2 = 'stim', 'imag'

exclude = [7]

subject_id=1

event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}

for subject_id in range(12,13):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG/Preproc')
    dir_save= os.path.join(data_path, 'CSP-Timefreq')

    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
        
    print '-----Now Processing %s -------' %subject   
    
    fname_in = os.path.join(data_path, '%s-clean-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    
    #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
    #mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
    #event_id= {'stim':100, 'imag': 200}

    event_id = {'imag/face', 'imag/house'}
    epochs=epochs['imag/face', 'imag/house']

    # Assemble the classifier using scikit-learn pipeline
    clf = make_pipeline(CSP(n_components=4, reg='oas', log=True, norm_trace=False),
                        LinearDiscriminantAnalysis())
    n_splits = 5  # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    # Classification & Time-frequency parameters
    tmin, tmax = -.200, 1.5
    n_cycles = 10.  # how many complete cycles: used to define window size
    min_freq = 2.
    max_freq = 25.
    n_freqs = 8  # how many frequency bins to use
    sfreq = epochs.info['sfreq']
    
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
        epochs_filter = epochs.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    
       
        y = le.fit_transform(epochs_filter.events[:, 2])
    
        X = epochs_filter.get_data()
    
        # Save mean scores over folds for each frequency and time window
        freq_scores[freq] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                    scoring='roc_auc', cv=cv,
                                                    n_jobs=1), axis=0)
    
    ###############################################################################
    # Plot frequency results
    
    class_balance = np.mean(y == y[0])
    class_balance = max(class_balance, 1. - class_balance)

    plt.figure()
    plt.bar(left=freqs[:-1], height=freq_scores, width=np.diff(freqs)[0],
            align='edge', edgecolor='black')
    plt.xticks(freqs)
    plt.ylim([0, 1])
    plt.axhline(class_balance, color='k', linestyle='--',
                label='chance level')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Decoding Scores')
    plt.title('Frequency Decoding Scores')
    
    ###############################################################################
    # Loop through frequencies and time, apply classifier and save scores
    
    # init scores
    tf_scores = np.zeros((n_freqs - 1, n_windows))
    
    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in enumerate(freq_ranges):
    
        # Infer window size based on the frequency being used
        w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds
    
        # Apply band-pass filter to isolate the specified frequencies
        epochs_filter = epochs.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')
    
        
        y = le.fit_transform(epochs.events[:, 2])
    
        # Roll covariance, csp and lda over time
        for t, w_time in enumerate(centered_w_times):
    
            # Center the min and max of the window
            w_tmin = w_time - w_size / 2.
            w_tmax = w_time + w_size / 2.
    
            # Crop data into time-window of interest
            X = epochs_filter.copy().crop(w_tmin, w_tmax).get_data()
    
            # Save mean scores over folds for each frequency and time window
            tf_scores[freq, t] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                         scoring='roc_auc', cv=cv,
                                                         n_jobs=1), axis=0)
    
    ###############################################################################
    # Plot time-frequency results
    plt.figure()
    # Set up time frequency object
    av_tfr = AverageTFR(create_info(['freq'], sfreq), tf_scores[np.newaxis, :],
                        centered_w_times, freqs[1:], 1)
    
    chance = class_balance  # set chance level to white in the plot
    av_tfr.plot([0], vmin=chance, title="Time-Frequency Decoding Scores",
                cmap=plt.cm.Reds)
