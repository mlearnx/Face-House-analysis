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

import numpy as np

from mne.utils import check_random_state  # noqa
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults) 
                        
from functools import partial  # noqa      

                        




print(__doc__)

#--------------------------
# Import Data
#--------------------------

overwrite = True


ana_path = '/home/claire/DATA/Data_Face_House_new_proc'

exclude = [7,12]

dataset =['task'] #'task',

for data in dataset:
    for subject_id in range(1,26):
        if subject_id in exclude:
            continue
        subject = 'S%02d' %subject_id
        data_path = os.path.join(ana_path, subject , 'EEG')
        fname_in = os.path.join(data_path, '%s_%s.bdf' %(subject, data))
    
        dir_save= os.path.join(data_path, 'New_Preproc')
        #raw_fname_out_1 = os.path.join(dir_save,'%s-%s-noncausal-highpass-1Hz-raw.fif' %(subject,data ))
        
        #raw_fname_out_2 = os.path.join(dir_save,'%s-causal-highpass-2Hz-raw.fif' %subject)
        
        raw_fname_out_3 = os.path.join(dir_save,'%s-%s-fir-highpass-1Hz-raw.fif' %(subject, data))
    
                                
        if not op.exists(dir_save):
            os.makedir(dir_save)
        
        
        
        print(subject)
    
    
        #---------------------------------------
        # Import Data from Task
        #---------------------------------------
        raw = mne.io.read_raw_edf(fname_in,  stim_channel=-1, misc=['EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'],   preload=True)
        raw.rename_channels(mapping={'E1H1\t//EXG1 HE ': 'EOG L', 'E2H2\t//EXG2 HE ': 'EOG R', 'E3LE\t//EXG3 LE ': 'EOG V L', 'E5M2\t//EXG5 M2 ': 'M2', 'E4M1\t//EXG4 M1 ': 'M1' })
        raw.set_channel_types(mapping={'EOG L': 'eog', 'EOG R': 'eog', 'EOG V L': 'eog', 'M1':'misc', 'M2': 'misc'})
        
        
        
        events = mne.find_events(raw, verbose=True)
        mne.write_events(os.path.join(dir_save , subject + '-%s-eve.fif' %data), events)
        
        
        # get eletrodes loc
        montage= mne.channels.read_montage('standard_1020', path = '/home/claire/Appli/mne-python/mne/channels/data/montages/')
        raw.set_montage(montage)
        
        #raw.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    
        
        raw.set_eeg_reference(ref_channels='average')
        
        # High-pass EOG to get reasonable thresholds in autoreject
        picks_eog = mne.pick_types(raw.info, eeg=False, eog=True)
                
            
        #raw_one = raw.filter(1.,40, method='iir',  phase= 'zero-double', iir_params=None, n_jobs=1) # 4th order Butterworth filter non causal
        #raw_two = raw.filter(2.,40, method='iir',  phase= 'minimum', iir_params=None, n_jobs=1) # 4th order Butterworth filter causal
        
        raw_fir = raw.filter(1.,40, fir_design='firwin')
        
        #raw_out_1 = raw_fname_out_1
        #raw_out_2 = raw_fname_out_2
        
        #raw_one.save(raw_fname_out_1, overwrite=True)
        #raw_two.save(raw_fname_out_2, overwrite = True)

        raw_fir.save(raw_fname_out_3, overwrite = True)