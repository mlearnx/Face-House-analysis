# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:24:19 2017

@author: claire
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
import os
from sklearn.preprocessing import LabelEncoder

cond= 'diam'
ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/Ana_Cues/'

exclude = []
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,12):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject + '/EEG/Ana_Cues/')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    #epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs = mne.concatenate_epochs(all_epochs)

epochs=epochs[cond]

epochs.crop(tmin=-0.2, tmax=0.5)

# fit and time decoder
le=LabelEncoder()

X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
y = le.fit_transform(epochs.events[:, 2])  # target: Audio left or right

clf = make_pipeline(StandardScaler(), LogisticRegression())

# define the Temporal Generalization object
time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(time_gen, X, y, cv=5, n_jobs=6)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)


class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)
# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label='score')
ax.axhline(class_balance, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding EEG sensors over time - %s subject %s' %(cond,subject))
plt.show()

plt.savefig(ana_path + ' gen_across_time_%s_%s.pdf' %(cond,subject),  bbox_to_inches='tight')

# Plot the full matrix
fig, ax = plt.subplots(1, 1)
im = ax.imshow(scores, interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal Generalization - %s subject %s' %(cond, subject))
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im, ax=ax)
plt.show()
plt.savefig(ana_path + ' gen_across_time_matrix_%s_%s.pdf' %(cond,subject),  bbox_to_inches='tight')
