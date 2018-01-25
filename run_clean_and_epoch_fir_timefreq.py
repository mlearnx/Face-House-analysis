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
from mne.preprocessing import ICA, create_eog_epochs
from autoreject import get_rejection_threshold
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults,  plot_epochs ) 
                        
from functools import partial  # noqa      

print(__doc__)

#--------------------------
# Import Data
#--------------------------

overwrite = True


ana_path = '/home/claire/DATA/Data_Face_House_new_proc'


baseline = (None, 0)


exclude = [7, 12]

subject_id =4

#data = 'task' # 'clue'
data = 'clue' # 'clue'


if 'task'in data:
    event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}
elif 'clue'in data:
    event_id = {'square' :10, 'diam':20}



subject = 'S%02d' %subject_id
data_path = os.path.join(ana_path, subject , 'EEG', 'New_Preproc')

#raw_fname_ica = os.path.join(data_path,'%s-noncausal-highpass-1Hz-raw.fif' %subject)

#raw_fname_erp = os.path.join(data_path,'%s-causal-highpass-2Hz-raw.fif' %subject)

raw_fname_fir = os.path.join(data_path,'%s-%s-fir-highpass-1Hz-raw.fif' %(subject, data))

                        

print(subject)

raw =io.read_raw_fif(raw_fname_fir, preload=True)

events = mne.read_events(op.join(data_path, '%s-%s-eve.fif' % (subject, data)))

raw.set_eeg_reference(projection=True)

#    raw_ica.pick_types(misc=False, eeg=True, eog=True)
#    raw_ica.plot(n_channels=64, scalings={"eeg": 45e-6}, events=events,
#         event_color={101: "green", 102:'green', 201: "blue", 202:'blue'})
#
#    if raw_ica.annotations is not None:
#        onset = np.floor(raw_ica.annotations.onset * raw_ica.info["sfreq"])
#        duration = np.ceil(raw_ica.annotations.duration * raw_ica.info["sfreq"])
#        segments = column_stack([onset, duration]).astype(int)
#        np.savetxt(os.path.join(data_path, '%s_raw_bad_segments.csv' %subject), segments, delimiter=",", comments="",
#                   header="onset,duration", fmt="%d")
#
#    with open(os.path.join(data_path, '%s_raw_bad_channels.csv' %subject), "w") as f:
#        f.write(",".join(raw_ica.info["bads"]))
#        f.write("\n")


# Epoch the data
print('Epoching')

tmin = -1
tmax = 1.9
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject=None,
                    verbose=False, detrend=0,  preload=True)
                    

print('Decimate')
epochs.decimate(decim =4) # decim from 1024 to 256

print('run autoreject')
   
   
reject = get_rejection_threshold(epochs)
reject

rej= {'eeg': reject['eeg']}

epochs.drop_bad(reject=rej)
print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))

if epochs.drop_log_stats() > 20:
    print('----- More than 20 % of epochs rejected ---- check manually !!! --')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), reject=None,
                    verbose=False, detrend=0,  preload=True)
                    

    print('Decimate')
    epochs.decimate(decim =4) # decim from 1024 to 256


    epochs.plot(n_channels=64, scalings={'eeg': 45e-6})

print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))

   #---------------------------------------
   # Compute ICA andf remove EOG artefacts
   #---------------------------------------
print('Run ICA')


n_components=25
ica = ICA(n_components=n_components, method='fastica')
ica.fit(epochs)
ica.plot_components()

picks=mne.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    



n_max_eog = 3  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw)  # get single EOG trials
eog_average =eog_epochs.average()

eog_inds, scores = ica.find_bads_eog(eog_epochs, ch_name='EOG V L')  # find via correlation


ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).
ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course


ica.plot_properties(eog_epochs, picks=[0,1,2], psd_args={'fmax': 35.},
                    image_args={'sigma': 1.})

print(ica.labels_)

# show before/after ica removal plot
ica.plot_overlay(eog_average, exclude=eog_inds, show=True)
# red -> before, black -> after. Yes! We remove quite a lot!

 #to definitely register this component as a bad one to be removed
# there is the ``ica.exclude`` attribute, a simple Python list

ica.exclude.extend(eog_inds)

print('Apply ICA on  epochs')
ica.apply(epochs)

ica.save(os.path.join(data_path, subject + '-%s-ica.fif' %data))


mne.Epochs.save(epochs, os.path.join(data_path, subject + '-%s-fir-highpass-1Hz-epo.fif' %data))

del(epochs, reject, rej, ica, eog_inds)
plt.close('all')