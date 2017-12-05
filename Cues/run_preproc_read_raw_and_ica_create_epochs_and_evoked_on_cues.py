# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:28:20 2017

@author: claire
"""

from __future__ import print_function

import mne
import utils
from mne import io
import os
import os.path as op
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
from mne.epochs import combine_event_ids

print(__doc__)

overwrite = False

sq_face = 'S02', 'S04', 'S06', 'S08', 'S10'
sq_house = 'S01', 'S03', 'S05', 'S07', 'S09', 'S11'


subjects = 'S02', 'S04', 'S06', 'S08', 'S10', 'S01', 'S03', 'S05', 'S07', 'S09', 'S11'#S02

for subject in subjects:
    
    data_path= '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/'
    save_path =  '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/Ana_Cues/'
    #--------------------------------------------------------------------------------------
    # create raw.fif file : import chanloc, reref, rej bad electrodes, high pass filter
    
    # use ICA to remove EOG:
    # - import raw
    # - high pass at 1hz
    # - epoch data
    # - create evoked data
    #------------------------------
    raw_fname = subject + '-raw.fif'
    
    
    if not op.exists(save_path):
        os.makedirs(save_path)
        
    # Checks if preprocessing has already been done
    #if op.exists(save_path + raw_fname) and not overwrite:
     #   print(raw_fname + ' already exists')
      #  print(subject)
        
    
    # get and save events
    raw, events = import_bdf(data_path, subject)
    
     # rename event
    if subject in sq_house:
       events[events==10] = 300 # square = house
       events[events==20] = 303 # diam face
    if subject in sq_face:
        events[events==10] = 404 # square face
        events[events==20]= 400 # diam house
    
    # save events
    mne.write_events(save_path + subject  + '-eve.fif', events)
    
    raw.filter(1.,40, fir_window='hamming', fir_design='firwin',  n_jobs=6)
    
    # plot
    #raw.plot(events=events, duration =10, n_channels =64)
    
    raw.save(save_path + raw_fname)
    
    
    # Epoch data
   
    # specify epochs length
    tmin = -0.5
    tmax = 0.5
    
    
    if subject in sq_house:
        event_id = {'sq/house' :300, 'diam/face':303}
        
    elif subject in sq_face:
        event_id = {'sq/face' :404, 'diam/house':400}
    else:
        print ('--- WARNING subject is not in any category ---')
    
    
    
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,  baseline = None) # 
    
        
    epochs.decimate(decim =4) # decim from 1024 to 256
    
   
    mne.Epochs.save(epochs, save_path + subject+ '-epo.fif')
    
    






