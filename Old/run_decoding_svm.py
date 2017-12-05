# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:07:58 2017

@author: claire
"""

from sklearn.svm import SVC  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP  # noqa
import numpy as np
import matplotlib.pyplot as plt
import os
from mne import Epochs, pick_types, find_events
import mne
from sklearn.preprocessing import LabelEncoder



cond = ['stim', 'imag']

exclude = [7]
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,11):#,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs = mne.concatenate_epochs(all_epochs)

mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    

epochs_data = epochs.get_data()

from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa

cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
clf = Pipeline([('CSP', CSP), ('SVC', SVC)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

print(scores.mean())  # should match results above


# average group of 4 trials
    data_cond1 =  epochs['stim'].get_data()
    data_cond2 = epochs['imag'].get_data()
    
    mean_cond1=[]
    ind_trial = 0
    while ind_trial<= len(data_cond1)-5:
        mean_cond1.append(mean(data_cond1[ind_trial:(ind_trial+4)], 0))
        print ind_trial
        ind_trial+=5
    
    mean_cond2=[]
    ind_trial = 0
    while ind_trial<= len(data_cond2)-5:
        mean_cond2.append(mean(data_cond2[ind_trial:(ind_trial+4)], 0))
        print ind_trial
        ind_trial+=5
    
    X=[]
    # create variable for decoding
    X = mean_cond1 + mean_cond2
    epochs_data=np.array(X)
    labels = np.array([0] * len(mean_cond1) + [1] * len(mean_cond2))     


# Instantiate label encoder
le = LabelEncoder()
labels = le.fit_transform(epochs.events[:, 2])

n_components = 3  # pick some components
svc = SVC(C=1, kernel='linear')
csp = CSP(n_components=n_components, norm_trace=False)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))