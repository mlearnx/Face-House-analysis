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

event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}

baseline = (None, 0)


exclude = [7]

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join(ana_path, subject , 'EEG', 'New_Preproc')

    raw_fname_ica = os.path.join(data_path,'%s-noncausal-highpass-1Hz-raw.fif' %subject)
    
    raw_fname_erp = os.path.join(data_path,'%s-causal-highpass-2Hz-raw.fif' %subject)
                            
    
    print(subject)

    raw_ica =io.read_raw_fif(raw_fname_ica, preload=True)
    raw_erp =io.read_raw_fif(raw_fname_erp, preload=True)

    events = mne.read_events(op.join(data_path, '%s-eve.fif' % subject))
    
    raw_ica.set_eeg_reference(projection=True)
    
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
    print('  Epoching')
    
    tmin = -0.5
    tmax = 1.5
    epochs_ica = mne.Epochs(raw_ica, events, event_id, tmin, tmax,
                        baseline=(None, 0), reject=None,
                        verbose=False, detrend=0,  preload=True)
                        
    epochs_erp = mne.Epochs(raw_erp, events, event_id, tmin, tmax,
                        baseline=(None, 0), reject=None,
                        verbose=False, detrend=0,  preload=True)
    
    print('Decimate')
    epochs_ica.decimate(decim =4) # decim from 1024 to 256
    epochs_erp.decimate(decim=4)
    
    print('run autoreject')
       
       
    reject = get_rejection_threshold(epochs_ica)
    reject
    
    
    
    rej= {'eeg': reject['eeg']}
    epochs_ica.drop_bad(reject=rej)
    print('  Dropped %0.1f%% of epochs' % (epochs_ica.drop_log_stats(),))

    reject_erp = get_rejection_threshold(epochs_erp)
    reject_erp
    rej_erp= {'eeg': reject_erp['eeg']}
    epochs_erp.drop_bad(reject=rej_erp)
    print('  Dropped %0.1f%% of epochs' % (epochs_erp.drop_log_stats(),))
    
    
   #---------------------------------------
   # Compute ICA andf remove EOG artefacts
   #---------------------------------------
    print('Run ICA')
    
    raw_ica.interpolate_bads()
    
    n_components=25
    ica = ICA(n_components=n_components, method='fastica')
    ica.fit(epochs_ica)
    ica.plot_components()
    
    picks=mne.pick_types(raw_ica.info, eeg=True, eog=True, exclude='bads')    
    
    
    
    n_max_eog = 3  # here we bet on finding the vertical EOG components
    eog_epochs = create_eog_epochs(raw_ica)  # get single EOG trials
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
    
    
    
    print('Apply ICA on ERP epochs')
    ica.apply(epochs_erp)

    ica.save(os.path.join(data_path, subject + '-ica.fif'))


    mne.Epochs.save(epochs_erp, os.path.join(data_path, subject + '-causal-highpass-2Hz-epo.fif'))

    del(epochs, epochs_clean, ica)