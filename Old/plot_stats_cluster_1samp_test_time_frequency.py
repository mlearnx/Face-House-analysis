"""
.. _tut_stats_cluster_sensor_1samp_tfr:

===============================================================
Non-parametric 1 sample cluster statistic on single trial power
===============================================================

This script shows how to estimate significant clusters
in time-frequency power estimates. It uses a non-parametric
statistical procedure based on permutations and cluster
level statistics.

The procedure consists in:

  - extracting epochs
  - compute single trial power estimates
  - baseline line correct the power estimates (power ratios)
  - compute stats to see if ratio deviates from 1.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test
from mne.datasets import sample
import os
print(__doc__)

###############################################################################
# Set parameters
# --------------
exclude = [7]
all_cond1= list()
all_cond2=list()

cond1=['stim/face', 'stim/house']
cond2=['imag/face', 'imag/house']

# We start by exploring the frequence content of our epochs.
for subject_id in range(1,11):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    # Take only one channel
    ch_name = 'POz'
    epochs.pick_channels([ch_name])
    
    evoked = epochs.average()
    
    
    
    
    # Factor to down-sample the temporal dimension of the TFR computed by
    # tfr_morlet. Decimation occurs after frequency decomposition and can
    # be used to reduce memory usage (and possibly computational time of downstream
    # operations such as nonparametric statistics) if you don't need high
    # spectrotemporal resolution.
    decim = 5
    frequencies = np.arange(8, 30, 2)  # define frequencies of interest
    sfreq = raw.info['sfreq']  # sampling in Hz
    tfr_epochs = tfr_morlet(epochs, frequencies, n_cycles=4., decim=decim,
                            average=False, return_itc=False, n_jobs=1)
    
    # Baseline power
    tfr_epochs.apply_baseline(mode='logratio', baseline=(-.100, 0))
    
    # Crop in time to keep only what is between 0 and 400 ms
    evoked.crop(0., 0.5)
    tfr_epochs.crop(0., 0.5)
    
    epochs_power = tfr_epochs.data[:, 0, :, :]  # take the 1 channel

###############################################################################
# Compute statistic
# -----------------
threshold = 2.5
T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_1samp_test(epochs_power, n_permutations=100,
                                   threshold=threshold, tail=0)

###############################################################################
# View time-frequency plots
# -------------------------

evoked_data = evoked.data
times = 1e3 * evoked.times

plt.figure()
plt.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)

# Create new stats image with only significant clusters
T_obs_plot = np.nan * np.ones_like(T_obs)
for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
        T_obs_plot[c] = T_obs[c]

vmax = np.max(np.abs(T_obs))
vmin = -vmax
plt.subplot(2, 1, 1)
plt.imshow(T_obs, cmap=plt.cm.gray,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
           extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
           aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power (%s)' % ch_name)

ax2 = plt.subplot(2, 1, 2)
evoked.plot(axes=[ax2])
plt.show()
