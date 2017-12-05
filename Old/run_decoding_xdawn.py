# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:07:21 2017

@author: claire
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from mne import io, pick_types, read_events, Epochs
from mne.datasets import sample
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from mne.viz import tight_layout

import os
import mne
from sklearn.preprocessing import LabelEncoder


exclude = []
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,12):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject + '/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    epochs.interpolate_bads()
#    all_epochs.append(epochs)
    
#epochs = mne.concatenate_epochs(all_epochs)

epochs.pick_types(eeg=True)

# Create classification pipeline
clf = make_pipeline(Xdawn(n_components=3, reg='oas'),
                    Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1'))


le = LabelEncoder()
labels=le.fit_transform(epochs.events[:,2])


# Cross validator
cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=42)


# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv:
    clf.fit(epochs[train], labels[train])
    preds[test] = clf.predict(epochs[test])

# Classification report
target_names = ['stim/face', 'stim/house', 'imag/face', 'imag/house']
report = classification_report(labels, preds, target_names=target_names)
print(report)

# Normalized confusion matrix
cm = confusion_matrix(labels, preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()














