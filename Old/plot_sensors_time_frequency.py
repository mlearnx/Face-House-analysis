"""
.. _tut_sensors_time_frequency:

=============================================
Frequency and time-frequency sensors analysis
=============================================

The objective is to show you how to explore the spectral content
of your data (frequency and time-frequency). Here we'll work on Epochs.

We will use the somatosensory dataset that contains so
called event related synchronizations (ERS) / desynchronizations (ERD) in
the beta band.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper

#
###############################################################################
# Frequency analysis
# ------------------
exclude = [7]
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,11):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs = epochs=mne.concatenate_epochs(all_epochs)

vmin=0.2
vmax=0.2
epochs['stim/house'].plot_psd_topomap(normalize=True, vmin=vmin, vmax=vmax)
epochs['imag/face'].plot_psd_topomap(normalize=True, vmin=vmin, vmax=vmax)



# define frequencies of interest (log-spaced)
freqs = np.logspace(*np.log10([2, 20]), num=10)
n_cycles = freqs / 2.  # different number of cycle per frequency
epochs_imagery=epochs['imag']
epochs_percept = epochs['stim']
power_imag, itc_imag = tfr_morlet(epochs_imagery, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=6)
power_stim, itc_stim = tfr_morlet(epochs_percept, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=6)

power_imag.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power - Imagery')
    #power.plot([82], baseline=(-0.5, 0), mode='logratio', title=power.ch_names[82])

tmin = 0.05
tmax= 0.5

vmin=0.2
vmax=0.2

fig, axis = plt.subplots(3, 2, figsize=(7, 4))
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=4, fmax=7,
                   baseline=(-0.5, 0), mode='logratio', vmax=0.1, axes=axis[0, 0],
                   title='Theta Imagery', show=False)
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', vmax=0.04 , axes=axis[1, 0],
                   title='Alpha Imagery', show=False)
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio',   vmax=0.04 ,axes=axis[2,0],
                   title='Beta Imagery',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=4, fmax=7,
                   baseline=(-0.5, 0), mode='logratio', vmax=0.1, axes=axis[0,1],
                   title='Theta Percept',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio',vmax=0.04 ,axes=axis[1,1],
                   title='Alpha Percept',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio',  vmax=0.04 , axes=axis[2,1],
                   title='Beta Percept', show=False)
mne.viz.tight_layout()
plt.suptitle('Topographies Subject %s'%subject_id)
plt.show()


fig, axis = plt.subplots(3, 2, figsize=(7, 4))
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=4, fmax=7,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0, 0],
                   title='Theta Imagery', show=False)
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1, 0],
                   title='Alpha Imagery', show=False)
power_imag.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[2,0],
                   title='Beta Imagery',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=4, fmax=7,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0,1],
                   title='Theta Percept',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1,1],
                   title='Alpha Percept',  show=False)
power_stim.plot_topomap(ch_type='eeg', tmin=tmin, tmax=tmax, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[2,1],
                   title='Beta Percept', show=False)
mne.viz.tight_layout()
plt.suptitle('Topographies Subject %s'%subject_id)
plt.show()
help(power.plot_topomap)

    #epochs['stim/face'].plot_psd(fmin=2., fmax=30.)
    
    ###############################################################################
    # Now let's take a look at the spatial distributions of the PSD.
    #epochs.plot_psd_topomap(ch_type='eeg', normalize=True)
    
    ###############################################################################
    # Alternatively, you can also create PSDs from Epochs objects with functions
    # that start with ``psd_`` such as
    # :func:`mne.time_frequency.psd_multitaper` and
    # :func:`mne.time_frequency.psd_welch`.
    
#    f, ax = plt.subplots()
#    psds, freqs = psd_multitaper(epochs, fmin=2, fmax=30, n_jobs=6)
#    psds = 10 * np.log10(psds)
#    psds_mean = psds.mean(0).mean(0)
#    psds_std = psds.mean(0).std(0)
#    
#    ax.plot(freqs, psds_mean, color='k')
#    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
#                    color='k', alpha=.5)
#    ax.set(title='Multitaper PSD (electrodes) - Imagery', xlabel='Frequency',
#           ylabel='Power Spectral Density (dB)')
#    plt.show()

###############################################################################
# Time-frequency analysis: power and intertrial coherence
# -------------------------------------------------------
#
# We now compute time-frequency representations (TFRs) from our Epochs.
# We'll look at power and intertrial coherence (ITC).
#
# To this we'll use the function :func:`mne.time_frequency.tfr_morlet`
# but you can also use :func:`mne.time_frequency.tfr_multitaper`
# or :func:`mne.time_frequency.tfr_stockwell`.
    
    
    
    ###############################################################################
    # Inspect power
    # -------------
    #
    # .. note::
    #     The generated figures are interactive. In the topo you can click
    #     on an image to visualize the data for one censor.
    #     You can also select a portion in the time-frequency plane to
    #     obtain a topomap for a certain time-frequency region.
    

###############################################################################
# Inspect ITC
# -----------
#itc.plot_topo(title='Inter-Trial coherence Imagery', vmin=0., vmax=1., cmap='Reds')

###############################################################################
# .. note::
#     Baseline correction can be applied to power or done in plots
#     To illustrate the baseline correction in plots the next line is
#     commented power.apply_baseline(baseline=(-0.5, 0), mode='logratio')

###############################################################################
# Exercise
# --------
#
#    - Visualize the intertrial coherence values as topomaps as done with
#      power.
