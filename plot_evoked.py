# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:43:47 2017

@author: claire
"""

import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import testing
from mne.channels.layout import _find_topomap_coords
from mne.stats.cluster_level import spatio_temporal_cluster_1samp_test as tst
from mne.defaults import DEFAULTS
from mne.channels.layout import find_layout



ana_path = '/home/claire/DATA/Data_Face_House_new_proc/'
data_folder= '/EEG/Preproc'
montage= mne.channels.read_montage('standard_1020', path = '/home/claire/Appli/mne-python/mne/channels/data/montages/')

event_id = {'stim/face':101, 'stim/house':102, 'imag/face':201, 'imag/house':202 }
exclude = [7, 11]


for subject_id in range(1,25):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join(ana_path + subject + data_folder)
    fname_in = os.path.join(data_path, '%s-clean-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
   #epochs.filter(None, 45)
    epochs.interpolate_bads()

    epochs.pick_channels(['Cz'])#, 'PO7', 'PO3', 'PO4', 'Cz', 'Fz', 'FPz', 'POz'])

    evoked = epochs["stim/face"].average()
    

    layout = find_layout(evoked.info)
    pos = layout.pos.copy()
    
    #f = plt.figure()
    #f.set_size_inches((10, 10))
    
    evokeds = {cond:list(epochs[cond].iter_evoked()) for cond in event_id}
    ylims = (evoked.data.min() * DEFAULTS["scalings"]["eeg"],
             evoked.data.max() * DEFAULTS["scalings"]["eeg"])
    ylims = (-30, 40)
    ymax = np.min(np.abs(np.array(ylims)))
    for pick, (pos_, ch_name) in enumerate(zip(pos, evoked.ch_names)):
#        mne.viz.plot_compare_evokeds(evokeds, picks=pick, 
#                             ylim=dict(eeg=ylims),
#                             show=False,
#                             show_sensors=False,
#                            show_legend=False,
#                             title='');
       
    
    #ax_l = plt.axes([0, 0] + list(pos[0, 2:]))
        mne.viz.plot_compare_evokeds(evokeds, ylim=dict(eeg=ylims), title='%s' %subject, show_sensors=False,
                         picks=0, ci=.95,
                                 show=False)
    
   