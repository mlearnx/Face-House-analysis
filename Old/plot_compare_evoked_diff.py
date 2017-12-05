# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:01:51 2017

@author: claire

Plot the ERP difference between conditions for each subject


"""

import os
import os.path as op
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import mne
from mne.parallel import parallel_func
from autoreject import (set_matplotlib_defaults) 

from mne.stats import _bootstrap_ci

ana_path = '/home/claire/DATA/Data_Face_House_new_proc/'
data_folder= '/EEG/Preproc'


#evokeds=list()
#
#for subject_id in [1,2,3,4,5,6,8,9,10,11]:
#    subject = 'S%02d' %subject_id
#    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
#    fname_in = op.join(data_path, '%s-ave.fif' %subject)
#    evokeds.append(mne.read_evokeds(fname_in))
#
#
#
#times= np.arange(0.1,1, 0.05)
#
#
#for idx, evoked in enumerate(evokeds):
#    for cond in [1,3]:
#        comm = evoked[cond].comment
#        evoked[cond].plot_joint(title='Subject %s %s' % (idx + 1, comm))

event_id = {'stim/face':101, 'stim/house':102, 'imag/face':201, 'imag/house':202 }
exclude = [7, 11]
all_epochs=list()

chan_range= ['PO8']


# def figure params

set_matplotlib_defaults(plt)
#plt.tight_layout()
fig1, ax1 = plt.subplots(nrows=2, ncols=2)
fig2, ax2 = plt.subplots(nrows=2, ncols=2)
fig3, ax3 = plt.subplots(nrows=2, ncols=2)
fig4, ax4 = plt.subplots(nrows=2, ncols=2)
fig5, ax5 = plt.subplots(nrows=2, ncols=2)
fig6, ax6 = plt.subplots(nrows=2, ncols=2)

nplot_x=0
nplot_y =0

count=0
nsuj=0
coord=[(0,0), (0,1), (1,0), (1,1)]

for subject_id in range(1,25):
    if subject_id in exclude:
        continue
    nsuj=nsuj+1
    if nsuj in range(1,5):
        ax=ax1
    elif nsuj in range(5,9):
        ax=ax2
    elif nsuj in range(9,13):
        ax=ax3
    elif nsuj in range(13,17):
        ax=ax4
    elif nsuj in range(17,21):
        ax=ax5
    else:
        ax=ax6
#    
#    if count>1:
#        nplot_y =1
#       
#    
#    if nplot_x > 1: # change figure
#        nplot_x=0
#        nplot_y=0
#        count=0
        
    if count>3:
        count=0
        
    subject = 'S%02d' %subject_id
    data_path = os.path.join(ana_path + subject + data_folder)
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
   #epochs.filter(None, 45)
    epochs.interpolate_bads()
    #epochs.pick_channels(chan_range)

    # get evoked
    
    # average epochs and get Evoked datasets
    #evokeds = [epochs[cond].average() for cond in ['stim/face', 'stim/house', 'imag/face', 'imag/house' ]]
    
    
    #all_evokeds = dict((cond, epochs[cond].average()) for cond in event_id)
    #print(all_evokeds['stim/face'])
    
    #channel_ind = mne.pick_channels(epochs.info['ch_names'], ['PO8'])    

    
    stim_face, stim_house = epochs['stim/face'].average(), epochs['stim/house'].average()
    imag_face, imag_house = epochs['imag/face'].average(), epochs['imag/house'].average()
    
    
    # plot perception menus imagery 
    diff_evoked = mne.combine_evoked([stim_face, -stim_house ], weights= 'equal')
    diff_evoked.plot(spatial_colors=True, axes=ax[coord[count]], titles = '%s ' %subject)
    

    
    count=count+1
    
    
    
    
    
    diff_evoked.plot_joint(title = '%s ERP diff Perception Face-House' %subject))
    
    
    
    
    mne.viz.plot_epochs_image(epochs['imag/face'], picks=channel_ind)
    # plot 95% CI :
    # - 
    plt.plot()
    diff_evoked.plot(axes=ax[0,0])
    
    
    diff_evoked.plot()    
    
    
    diff=np.sum(diff_evoked.data ** 2, axis=0)
    diff = mne.baseline.rescale(diff, times, baseline=(None, 0))
    ax[0,0].set_title('%s' %subject)
    
    times= diff_evoked.times * 1e3
    ci_low, ci_up = _bootstrap_ci(diff_evoked.data, random_state=0,
                                  stat_fun=median)
    ci_low = mne.baseline.rescale(ci_low, diff_evoked.times, baseline=(None, 0))
    ci_up = mne.baseline.rescale(ci_up, diff_evoked.times, baseline=(None, 0))
    ax[0,0].fill_between(times, diff + ci_up, diff - ci_low, color='red', alpha=0.3)
    
    
    
    
    # plot stim face menus stim house
    #mne.combine_evoked([stim_face,-stim_house], weights= 'equal').plot_joint(title='%s ERP diff Stim Face minus Stim House'%subject )
    
    #plot imag face menus imag house
    #mne.combine_evoked([imag_face,-imag_house], weights= 'equal').plot_joint(title='%s ERP diff Imag Face minus Imag House'%subject)

    
    
   











