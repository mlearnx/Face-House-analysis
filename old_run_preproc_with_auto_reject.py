# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:09:36 2017

@author: claire
"""



from __future__ import print_function

import mne
from mne import io
import os
import os.path as op
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

import numpy as np

from mne.utils import check_random_state  # noqa
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults) 
                        
from functools import partial  # noqa      

import matplotlib.pyplot as plt  # noqa                  
                        
from autoreject import get_rejection_threshold  # noqa

from autoreject import plot_epochs  # noqa



print(__doc__)

#--------------------------
# Import Data
#--------------------------

overwrite = True

subject = 'S01'
data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG')
dir_save= os.path.join(data_path, 'Preproc')

epo_fname = subject + '-epo.fif'
ica_fname = subject + '-ica.fif'

if not op.exists(dir_save):
    os.makedirs(dir_save)


if op.exists(os.path.join(dir_save, epo_fname)) and not overwrite:
    print(epo_fname + ' already exists')
print(subject)


#---------------------------------------
# Import Data
#---------------------------------------
raw = mne.io.read_raw_edf(os.path.join(data_path, subject + '_task.bdf'),  stim_channel=-1, misc=['EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'],   preload=True)
raw.rename_channels(mapping={'E1H1\t//EXG1 HE ': 'EOG L', 'E2H2\t//EXG2 HE ': 'EOG R', 'E3LE\t//EXG3 LE ': 'EOG V L', 'E5M2\t//EXG5 M2 ': 'M2', 'E4M1\t//EXG4 M1 ': 'M1' })
raw.set_channel_types(mapping={'EOG L': 'eog', 'EOG R': 'eog', 'EOG V L': 'eog', 'M1':'misc', 'M2': 'misc'})


events = mne.find_events(raw, verbose=True)

mne.write_events(os.path.join(dir_save , subject + '-eve.fif'), events)


# get eletrodes loc
montage= mne.channels.read_montage('standard_1020', path = '/home/claire/Appli/mne-python/mne/channels/data/montages/')
raw.set_montage(montage)

#raw.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    

raw.set_eeg_reference(['M1', 'M2'])


raw.filter(0.5, None,  fir_window='hamming', fir_design='firwin',  n_jobs=1)

event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}


#--------------------------
# Set up Autoreject
#--------------------------

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

picks=mne.pick_types(raw.info, eeg=True, stim=False, eog=False, exclude = 'bads')

# create epochs
raw.info['projs'] = list()  # remove proj, don't proj while interpolating

tmin = -0.5
tmax = 1.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject=None,
                    verbose=False, detrend=0, picks=picks,  preload=True)

epochs.decimate(decim =4) # decim from 1024 to 256

# set up function to compute sensor-level threshold
thresh_func = partial(compute_thresholds, picks=picks, method='bayesian_optimization')

#---------------------------------------
# Run autoreject
#---------------------------------------

epochs.ch_names
ar = LocalAutoRejectCV( picks=picks,thresh_func=thresh_func)
epochs_clean=ar.fit_transform(epochs)

from autoreject import get_rejection_threshold
reject = get_rejection_threshold(epochs)
reject


#---------------------------------------
# Check autocorrect
#---------------------------------------

# plot epochs

plot_epochs(epochs, bad_epochs_idx=ar.bad_epochs_idx,n_channels=64, 
            fix_log=ar.fix_log, scalings=dict(eeg=40e-6),
            title='')



# plot evoked
evoked=epochs['stim'].average(picks=picks)
evoked_clean=epochs_clean['stim'].average(picks=picks)

set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = -100, 100
evoked.pick_types(eeg=True, exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False, spatial_colors=True)
axes[0].set_title('Before autoreject')
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim, spatial_colors=True)
axes[1].set_title('After autoreject')

plt.tight_layout()

#---------------------------------------
# re-reference to average reference
#---------------------------------------

epochs_clean.set_eeg_reference( ref_channels='average', projection = True)
epochs_clean.apply_proj()


#ica = mne.preprocessing.read_ica(os.path.join(data_path, ica_fname))
#ica.apply(epochs_clean)

#---------------------------------------
# Compute ICA andf remove EOG artefacts
#---------------------------------------


n_components=25
ica = ICA(n_components=n_components, method='fastica')
ica.fit(epochs_clean)
ica.plot_components()


picks=mne.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    


n_max_eog = 3  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw)  # get single EOG trials
eog_average =eog_epochs.average()

eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation


ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course


ica.plot_properties(eog_epochs, picks=eog_inds, psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})

print(ica.labels_)

# show before/after ica removal plot
ica.plot_overlay(eog_average, exclude=eog_inds, show=True)
# red -> before, black -> after. Yes! We remove quite a lot!

 #to definitely register this component as a bad one to be removed
# there is the ``ica.exclude`` attribute, a simple Python list
ica.exclude.extend(eog_inds)

ica.apply(epochs_clean)

#-----------------------------
# Save epoch data and ICA dec
#-----------------------------

ica.save(os.path.join(data_path, subject + '-ica.fif'))


mne.Epochs.save(epochs_clean, os.path.join(dir_save, subject + '-epo.fif'))

del(epochs, epochs_clean, ica)


