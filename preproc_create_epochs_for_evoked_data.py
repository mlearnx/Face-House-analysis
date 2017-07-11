# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:44:05 2017

@author: claire
"""



#-------------------------------------------------------------------------
# Epoch data for evoked activity analysis:
# - low pass filter raw file
# - epoch
# - apply ica to reject eog activity
# - reject bad epochs
#--------------------------------------------------------------------------


import mne
import utils
from mne import io
import os
import os.path as op
from matplotlib import pyplot as plt
from mne.preprocessing import ICA

print(__doc__)

overwrite = False

subject = 'S01'
data_path= '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/'


dir_evoked = data_path + 'Lowpass/'
lowpass_fname = subject + 'lowpass-epo.fif'

if not op.exists(dir_evoked):
    os.makedirs(dir_evoked)

# load raw file and apply ica to remove EOG IC
raw = mne.io.read_raw_fif(subject + '-raw.fif', preload=True)
events= mne.read_events(subject+'-eve.fif')

ica = mne.preprocessing.read_ica(subject + '-ica.fif')
ica.apply(raw)

# high pass filter

raw.filter(0.1,45, fir_window='hamming', fir_design='firwin',  n_jobs=6)



# specify epochs length
tmin = -0.5
tmax = 1.8

epochs = mne.Epochs(raw, events, event_id= [101, 102, 201, 202], tmin=tmin, tmax=tmax,  baseline = None) # 

epochs.plot(n_epochs=2, events=events)

epochs.drop_bad()

epochs.event_id = {'stim/face' : 101, 'stim/house': 102, 'imag/face': 201, 'imag/house': 202}

mne.Epochs.save(epochs, data_path + 'S01-epo.fif')

# average epochs and get Evoked datasets
evokeds = [epochs[cond].average() for cond in ['stim/face', 'stim/house', 'imag/face', 'imag/house' ]]

# save evoked data to disk
mne.write_evokeds('S01_face_house-ave.fif', evokeds)


#-----------------------------------------
# visualize evoked data
#-----------------------------------------

picks = mne.pick_types(evokeds[0].info, eeg=True, eog=False)
evokeds[0].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[1].info, eeg=True, eog=False)
evokeds[1].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[2].info, eeg=True, eog=False)
evokeds[2].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[3].info, eeg=True, eog=False)
evokeds[3].plot(spatial_colors=True, gfp=True, picks=picks)

