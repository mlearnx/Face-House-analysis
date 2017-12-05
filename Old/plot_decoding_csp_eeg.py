"""
===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]_. The EEGBCI
dataset is documented in [2]_. The data set is available at PhysioNet [3]_.

References
----------

.. [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
       of the abnormal components in the clinical EEG. Electroencephalography
       and Clinical Neurophysiology, 79(6):440--447, December 1991.
.. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
       Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
       (BCI) System. IEEE TBME 51(6):1034-1043.
.. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
       Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
       PhysioToolkit, and PhysioNet: Components of a New Research Resource for
       Complex Physiologic Signals. Circulation 101(23):e215-e220.
"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import os
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.decoding import CSP
from sklearn.preprocessing import LabelEncoder
import mne
print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.

cond = ['stim', 'imag']

exclude = [7]
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,12):
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
    
epochs_train = epochs[cond].copy().crop(tmin=0.2, tmax=1.5)

# Instantiate label encoder
le = LabelEncoder()
labels = le.fit_transform(epochs_train.events[:, 2])

###############################################################################
# Classification with linear discrimant analysis

from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa

# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs[cond].get_data()
epochs_data_train = epochs_train.get_data()

# Use scikit-learn Pipeline with cross_val_score function
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=6)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

evoked = epochs[cond].average()
evoked.data = csp.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])

layout = read_layout('biosemi')
evoked.plot_topomap(times=[0, 1, 2, 3, 4, 5], ch_type='eeg', layout=layout,
                    scale_time=1, time_format='%i', scale=1,
                    unit='Patterns (AU)', size=1.5)

###############################################################################
# Look at performance over time

sfreq = epochs.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show()
