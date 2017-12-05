# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:01:51 2017

@author: claire
"""

import os
import os.path as op
import numpy as np

import mne
from mne.parallel import parallel_func

ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'



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

# get evoked

# average epochs and get Evoked datasets
evokeds = [epochs[cond].average() for cond in ['stim/face', 'stim/house', 'imag/face', 'imag/house' ]]

times= np.arange(0.1,1, 0.05)

for i in range(0,len(evokeds)):
    evokeds[i].plot_topomap(times, average=0.05)
    plt.suptitle('%s' %evokeds[i].comment)


for i in range(0,len(evokeds)):

    ts_args = dict(gfp=True)
    topomap_args = dict(sensors=False)
    evokeds[i].plot_joint(title='%s' %evokeds[i].comment, times=[.130, .200, .250, .350],
                            ts_args=ts_args, topomap_args=topomap_args)



evoked_dict = dict()
for i in range(0,len(evokeds)):
    evoked_dict['%s' %evokeds[i].comment] = evokeds[i]
print(evoked_dict)

colors = dict(stim="Crimson", imag="CornFlowerBlue")
linestyles = dict(face='-', house='--')
pick = evoked_dict['stim/face'].ch_names.index('Oz')

mne.viz.plot_compare_evokeds(evoked_dict, picks=pick,
                             colors=colors, linestyles=linestyles)













