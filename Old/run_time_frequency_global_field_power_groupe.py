"""
===========================================================
Explore event-related dynamics for specific frequency bands
===========================================================

The objective is to show you how to explore spectrally localized
effects. For this purpose we adapt the method described in [1]_ and use it on
the somato dataset. The idea is to track the band-limited temporal evolution
of spatial patterns by using the Global Field Power (GFP).

We first bandpass filter the signals and then apply a Hilbert transform. To
reveal oscillatory activity the evoked response is then subtracted from every
single trial. Finally, we rectify the signals prior to averaging across trials
by taking the magniude of the Hilbert.
Then the GFP is computed as described in [2]_, using the sum of the squares
but without normalization by the rank.
Baselining is subsequently applied to make the GFPs comparable between
frequencies.
The procedure is then repeated for each frequency band of interest and
all GFPs are visualized. To estimate uncertainty, non-parametric confidence
intervals are computed as described in [3]_ across channels.

The advantage of this method over summarizing the Space x Time x Frequency
output of a Morlet Wavelet in frequency bands is relative speed and, more
importantly, the clear-cut comparability of the spectral decomposition (the
same type of filter is used across all bands).

References
----------

.. [1] Hari R. and Salmelin R. Human cortical oscillations: a neuromagnetic
       view through the skull (1997). Trends in Neuroscience 20 (1),
       pp. 44-49.
.. [2] Engemann D. and Gramfort A. (2015) Automated model selection in
       covariance estimation and spatial whitening of MEG and EEG signals,
       vol. 108, 328-342, NeuroImage.
.. [3] Efron B. and Hastie T. Computer Age Statistical Inference (2016).
       Cambrdige University Press, Chapter 11.2.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.parallel import parallel_func

import os


print(__doc__)

ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

rng = np.random.RandomState(42)

def get_gfp_ci(average, n_bootstraps=2000):
    """get confidence intervals from non-parametric bootstrap"""
    indices = np.arange(len(average.ch_names), dtype=int)
    gfps_bs = np.empty((n_bootstraps, len(average.times)))
    for iteration in range(n_bootstraps):
        bs_indices = rng.choice(indices, replace=True, size=len(indices))
        gfps_bs[iteration] = np.sum(average.data[bs_indices] ** 2, 0)
    gfps_bs = mne.baseline.rescale(gfps_bs, average.times, baseline=(None, 0))
    ci_low, ci_up = np.percentile(gfps_bs, (2.5, 97.5), axis=0)
    return ci_low, ci_up


def run_time_freq_gfp(subject_id, cond, event_id):

    # Set parameters
    subject = "S%02d" %subject_id
    data_path= '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/No_Low_pass/'
    events_path = '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/'
    
    raw_fname = data_path +  subject + '-raw.fif'
    event_fname = events_path + subject +'-eve.fif'
    #event_id = {'imag/face':201, 'imag/house': 202}#imag/face':201, 'imag/house': 202}
    # let's explore some frequency bands
#    iter_freqs = [
#        ('Theta', 4, 7),
#        ('Alpha', 8, 12),
#        ('Beta', 13, 25),
#        ('Gamma', 30, 45), 
#            ]
    
    iter_freqs = [
        ('Delta', 1, 3),
       
            ]
    ###############################################################################
    # We create average power time courses for each frequency band
    
    # set epoching parameters
    tmin, tmax = -0.5, 1.5
    baseline = None
    
    # get the header to extract events
    raw = mne.io.read_raw_fif(raw_fname, preload=False)
    events = mne.read_events(event_fname)
    
    frequency_map = list()
    
    for band, fmin, fmax in iter_freqs:
        # (re)load the data to save memory
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.pick_types(eeg=True, eog=True)  # we just look at gradiometers
    
        # bandpass filter and compute Hilbert
        raw.filter(fmin, fmax, n_jobs=6,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   fir_design='firwin')
        raw.apply_hilbert(n_jobs=5, envelope=False)
    
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                             preload=True)
        epochs.drop_bad()
        # remove evoked response and get analytic signal (envelope)
        epochs.subtract_evoked()  # for this we need to construct new epochs.
        epochs = mne.EpochsArray(
            data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
        # now average and move on
        frequency_map.append(([band, fmin, fmax], epochs.average()))
    
    for ((freq_name, fmin, fmax), average) in frequency_map:
        times = average.times * 1e3
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
        ci_low, ci_up = get_gfp_ci(average)
        
        this_freq = '%s'%(freq_name)
        fname_gfp_freq = os.path.join(data_path, '%s-gfp-%s-%s.mat' %(subject, cond, this_freq))
    
        from scipy.io import savemat
        savemat(fname_gfp_freq, {'times':times, 'freq': freq_name, 'gfp':gfp, 'ci_low': ci_low, 'ci_up': ci_up })
        
    
    
    

parallel, run_func, _=parallel_func(run_time_freq_gfp, n_jobs=1)
parallel(run_func(subject_id, 'perception', {'stim/face':101, 'stim/house': 102})    
        for subject_id in [1, 2, 3, 4, 5,6,8, 9, 10, 11]) 
parallel(run_func(subject_id, 'imagery', {'imag/face':201, 'imag/house': 202})    
        for subject_id in [1, 2, 3, 4, 5,6,8, 9, 10, 11]) 