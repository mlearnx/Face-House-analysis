# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:51:22 2017

@author: claire
"""
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import GeneralizingEstimator
from sklearn.preprocessing import LabelEncoder

print(__doc__)


ana_path = ana_path = '/home/claire/DATA/Data_Face_House_new_proc'


exclude = [7]  # Excluded subjects

all_epochs = list()

for subject_id in range(1, 12):
    if subject_id in exclude:
        continue
    subject = "S%02d" % subject_id
    print("processing subject: %s" % subject)
    data_path =  os.path.join(ana_path, subject , 'EEG', 'New_Preproc')
    
    epochs = mne.read_epochs(os.path.join(data_path, '%s-causal-highpass-2Hz-epo.fif' %subject), preload=True)
    epochs.interpolate_bads(reset_bads=True)

                                
    all_epochs.append(epochs)
    
epochs=mne.concatenate_epochs(all_epochs)

decim =2
epochs.decimate(decim)


# We will train the classifier on all stim face vs house trials
# and test on all images face vs house trials.
clf = make_pipeline(StandardScaler(), LogisticRegression())
time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=6)

le = LabelEncoder()

# train on stim
time_gen.fit(X=epochs['stim'].get_data(),
             y = le.fit_transform(epochs['stim'].events[:,2]) )

# score on imagery
scores=time_gen.score(X=epochs['imag'].get_data(),
             y = le.fit_transform(epochs['imag'].events[:,2]))


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
