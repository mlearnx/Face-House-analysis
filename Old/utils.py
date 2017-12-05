# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:03:23 2017

@author: claire
"""

# set of functions
import mne
from mne import io


def import_bdf(data_path, subject):
    # function to import bdf file and do basic preprocessing :
# - chanloc and chan info
# - ref to mastoids

    # import data
    raw = mne.io.read_raw_edf(data_path + subject + '_task.bdf',  stim_channel=-1, misc=['EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'],   preload=True)
    raw.rename_channels(mapping={'E1H1\t//EXG1 HE ': 'EOG L', 'E2H2\t//EXG2 HE ': 'EOG R', 'E3LE\t//EXG3 LE ': 'EOG V L', 'E5M2\t//EXG5 M2 ': 'M2', 'E4M1\t//EXG4 M1 ': 'M1' })
    raw.set_channel_types(mapping={'EOG L': 'eog', 'EOG R': 'eog', 'EOG V L': 'eog'})
    
    raw, _ =mne.io.set_eeg_reference(raw, ref_channels=['M1', 'M2'])
    raw.info['bads'] = ['M1', 'M2']
    
    # get eletrodes loc
    montage= mne.channels.read_montage('standard_1020', path = '/home/claire/Appli/mne-python/mne/channels/data/montages/')
    raw.set_montage(montage)
    raw.interpolate_bads(reset_bads=False) # 
    events = mne.find_events(raw, verbose=True)

    raw.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    

    return raw, events
    


