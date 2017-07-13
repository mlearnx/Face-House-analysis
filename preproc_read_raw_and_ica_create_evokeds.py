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


print(__doc__)

overwrite = False

subject = 'S08'
data_path= '/home/claire/DATA/Data_Face_House/' + subject + '/EEG/'

#--------------------------------------------------------------------------------------
# create raw.fif file : import chanloc, reref, rej bad electrodes, high pass filter

# use ICA to remove EOG:
# - import raw
# - high pass at 1hz
# - run ica
#------------------------------
raw_fname = subject + '-raw.fif'

# Checks if preprocessing has already been done
if op.exists(data_path + raw_fname) and not overwrite:
    print(raw_fname + ' already exists')
print(subject)

# get and save events
raw, events = import_bdf(data_path, subject)

# save events
mne.write_events(data_path + subject  + '-eve.fif', events)

raw.filter(1.,40, fir_window='hamming', fir_design='firwin',  n_jobs=6)
# plot
raw.plot(events=events, duration =10, n_channels =64)

raw.save(data_path + raw_fname)

decim = 4  # we need sufficient statistics, not all time points -> saves time

picks=mne.pick_types(raw.info, eeg=True, eog=True, exclude='bads')    
eog_average = create_eog_epochs(raw, picks=picks).average()

#eog_average = create_eog_epochs(raw, picks=picks).average(picks=picks[:-2])
#eog_average.average()
print('We found %i EOG events' % eog_average.nave)
eog_average.plot_joint()
# Initialize the ICA estimator and fit it on the data
ica = mne.preprocessing.ICA(n_components=25, method='extended-infomax')
ica.fit(raw, decim=decim)
# Find components highy correlated with the eog events

n_max_eog = 3  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw, picks=picks)  # get single EOG trials
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

# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving

# uncomment this for reading and writing
ica.save(data_path+ subject + '-ica.fif')
# ica = read_ica('my-ica.fif')

#----------------------------------------------------------------------------#
# create epochs and evoked data
#----------------------------------------------------------------------------#


dir_evoked = data_path + 'Evoked_Lowpass/'
lowpass_fname = dir_evoked + subject + '-epo.fif'

if not op.exists(dir_evoked):
    os.makedirs(dir_evoked)


if op.exists(data_path + lowpass_fname) and not overwrite:
    print(lowpass_fname + ' already exists')
print(subject)


# load raw file and apply ica to remove EOG IC
raw, events = utils.import_bdf(data_path, subject)
#events= mne.read_events(data_path + subject+'-eve.fif')
bad_chan = mne.io.read_raw_fif(data_path +subject + '-raw.fif')
raw.info['bads'] = bad_chan.info['bads']
# high pass filter
raw.filter(0.1,45, fir_window='hamming', fir_design='firwin',  n_jobs=6)
# reject eog using ica
ica = mne.preprocessing.read_ica(data_path+ subject + '-ica.fif')
ica.apply(raw)


# specify epochs length
tmin = -0.5
tmax = 1.5

epochs = mne.Epochs(raw, events, event_id= [101, 102, 201, 202], tmin=tmin, tmax=tmax,  baseline = None) # 

# inspect epochs !!
epochs.plot(n_epochs=5, events=events, n_channels=64)
epochs.drop_bad()

epochs.event_id = {'stim/face' : 101, 'stim/house': 102, 'imag/face': 201, 'imag/house': 202}

epochs.decimate(decim =4) # decim from 1024 to 256

mne.Epochs.save(epochs, dir_evoked + subject+ '-epo.fif')

# average epochs and get Evoked datasets
evokeds = [epochs[cond].average() for cond in ['stim/face', 'stim/house', 'imag/face', 'imag/house' ]]

# save evoked data to disk
mne.write_evokeds(dir_evoked + subject+'-ave.fif', evokeds)




picks = mne.pick_types(evokeds[0].info, eeg=True, eog=False)
evokeds[0].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[1].info, eeg=True, eog=False)
evokeds[1].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[2].info, eeg=True, eog=False)
evokeds[2].plot(spatial_colors=True, gfp=True, picks=picks)

picks = mne.pick_types(evokeds[3].info, eeg=True, eog=False)
evokeds[3].plot(spatial_colors=True, gfp=True, picks=picks)
















# ------------------------------
# Decoding Face from House 
# ------------------------------

import matplotlib.pyplot as plt
import mne
from mne.decoding import GeneralizationAcrossTime 
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
print(__doc__)



data_path= '/home/claire/DATA/Data_Face_House/S01/'

epochs = mne.read_epochs(data_path + 'S01-epo.fif', preload= True)

decim = 4

epochs.decimate(decim)
#------------------------- Percept Veh vs Percept Animals -----------------------------------

event_id=['imag/face','stim/house']
epochs_list = [epochs[k] for k in event_id]
#
epochs_clas = mne.epochs.concatenate_epochs(epochs_list)
#
## We define the epochs and the labels
n_cond1 = len(epochs_clas[event_id[0]])
n_cond2 = len(epochs_clas[event_id[1]])
y = np.r_[np.ones((n_cond1, )), np.zeros((n_cond2, ))]


#------------------Start of Decoding Script --------------


cv = StratifiedKFold(y=y)  # do a stratified cross-validation

# define the GeneralizationAcrossTime object
train_times = {'start': -0.1, 'stop':2}

gat = GeneralizationAcrossTime(predict_mode='cross-validation', train_times= train_times, n_jobs=6,cv=cv, scorer=roc_auc_score)

# fit and score
print("Fitting")
gat.fit(epochs_clas, y=y)
print("Scoring")
gat.score(epochs_clas)


# let's visualize now
gat.plot()
gat.plot_diagonal()







# ------------------------------
# Time Decoding Generalization 
# ------------------------------


data_path= '/home/claire/DATA/Data_Face_House/S01/'

epochs = mne.read_epochs(data_path + 'S01-epo.fif', preload= True)

# epoch starts 1 second after trial starts to avoid evoked effects
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

decim = 4

epochs.decimate(decim)

epochs_train.decimate(decim)

clf = make_pipeline(StandardScaler(), LogisticRegression())
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=6)

# Fit classifiers 
# experimental condition y indicates stim category
# get all images then train on veh then test on stim
# veh vs animal during images
time_gen.fit(X=epochs['stim'].get_data(), 
             y=epochs['stim'].events[:,2] == 102)


# Score on the epochs where the stimulus was presented to the right.
# veh vs animals during stimuli
scores = time_gen.score(X=epochs['imag'].get_data(),
                        y=epochs['imag'].events[:, 2] ==202)


# Plot
fig, ax = plt.subplots(1)
im = ax.matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower',
                extent=epochs.times[[0, -1, 0, -1]])
ax.axhline(0., color='k')
ax.axvline(0., color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Generalization across time and condition')
plt.colorbar(im, ax=ax)
plt.show()


#------------------------------------------------------------------------------------
# Train the classifier on Veh vs Image on Perception data and test on Imagery data
#------------------------------------------------------------------------------------


data_path= '/home/claire/DATA/Data_Face_House/S01/'

epochs = mne.read_epochs(data_path + 'S01-epo.fif', preload= True)

# epoch starts 1 second after trial starts to avoid evoked effects
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

decim = 4

epochs.decimate(decim)

epochs_train.decimate(decim)


# define events of interest
triggers = epochs.events[:, 2]

imagery_vs_percept = np.in1d(triggers, (101, 102)).astype(int)  # create vectors of 0 and 1 for imagery and perception 



train_times = {'start': 0, 'stop':1.50}

gat = GeneralizationAcrossTime(predict_mode='mean-prediction', n_jobs=6)

# for perception data, which one are vehicules ?

face_vs_house_percept = (triggers[np.in1d(triggers, (101, 102))] == 101).astype(int)

#face_vs_house_imag = (triggers[np.in1d(triggers, (101, 102, 201, 202))] == 101).astype(int)


# To make scikit-learn happy, we converted the bool array to integers
# in the same line. This results in an array of zeros and ones:
print("The unique classes' labels are: %s" % np.unique(face_vs_house_percept))

gat.fit(epochs[('stim/face', 'stim/house')], y=face_vs_house_percept)


# for imagery data, which one are faces ?

face_vs_house_imagery = (triggers[np.in1d(triggers, (201, 202))] == 201).astype(int)

print("The unique classes' labels are: %s" % np.unique(face_vs_house_imagery))


gat.score(epochs[('imag/face', 'imag/house')], y=face_vs_house_imagery)
gat.plot(title="Temporal Generalization (face vs house): perception to imagery")





