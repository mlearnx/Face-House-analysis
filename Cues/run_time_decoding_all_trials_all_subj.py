# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:07:21 2017

@author: claire
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import mne
from mne.decoding import (SlidingEstimator,GeneralizingEstimator,cross_val_multiscore, LinearModel, get_coef)
from mne.decoding import (LinearModel, get_coef)

import os
from mne.parallel import parallel_func

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC




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
    epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs = mne.concatenate_epochs(all_epochs)

#mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
#mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)

    #only look at occipital channels
#    select_chans = [u'Iz', u'Oz', u'O1', u'O2', u'O3', u'PO7', u'PO8', u'POz', u'PO1', u'PO3', u'PO2', u'PO4']
    #select_chans = [ u'PO7', u'PO8']
    #select_chans = [ u'Cz', u'FPz']

    #ch_names=[ch_name.replace('', '') for ch_name in select_chans]
   # epochs.pick_types(eeg=True).pick_channels(ch_names)
    
    # average group of 4 trials
#data_cond1 =  epochs['imag/face'].get_data()
#data_cond2 = epochs['imag/house'].get_data()
#
#mean_cond1=[]
#ind_trial = 0
#while ind_trial<= len(data_cond1)-5:
#    mean_cond1.append(mean(data_cond1[ind_trial:(ind_trial+4)], 0))
#    print ind_trial
#    ind_trial+=5
#
#mean_cond2=[]
#ind_trial = 0
#while ind_trial<= len(data_cond2)-5:
#    mean_cond2.append(mean(data_cond2[ind_trial:(ind_trial+4)], 0))
#    print ind_trial
#    ind_trial+=5
#
#X=[]
## create variable for decoding
#X = mean_cond1 + mean_cond2
#X=np.array(X)
#y = np.array([0] * len(mean_cond1) + [1] * len(mean_cond2))     


#----------------------------------#
# Time decoding
#----------------------------------#

epochs=epochs['sq']


# fit and time decoder
X = epochs.get_data()  # MEG signals: n_epochs, n_channels, n_times
le = LabelEncoder()
y = le.fit_transform(epochs.events[:, 2])

clf = make_pipeline(StandardScaler(), LogisticRegression())

time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)
# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)


class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(class_balance, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel
#clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
#time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
#time_decod.fit(X, y)
#
#coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
#evoked = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
#evoked.plot_joint(times=np.arange(0., .500, .100), title='patterns')

#----------------------#
# With statistics
#----------------------#





#cv=StratifiedKFold(n_splits=3, shuffle=False)
#cv.get_n_splits(X, y)


clf = make_pipeline(StandardScaler(), LogisticRegression())

time_decod = SlidingEstimator(clf, n_jobs=1, scoring='accuracy')

#from sklearn.svm import SVC
svc = SVC(C=1, kernel='linear')

cv = StratifiedKFold(3)



score, permutation_scores, pvalue = permutation_test_score(
    svc, X, y, scoring="accuracy", cv=3, n_permutations=100, n_jobs=1)












#scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)

scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

class_balance = np.mean(y == y[0])
class_balance = max(class_balance, 1. - class_balance)

    # Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label='score')
ax.axhline(class_balance, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

    
from mne.stats import permutation_t_test
n_permutations = 50000
T0, p_values, H0 = permutation_t_test(scores, n_permutations, n_jobs=1)




















