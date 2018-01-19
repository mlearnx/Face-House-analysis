"""
===============================
Decoding (ML) across time (MEG)
===============================

A sliding estimator fits a logistic regression model for every time point.
In this example, we contrast the condition `'famous'` vs `'scrambled'`
and `'famous'` vs `'unfamiliar'` using this approach. The end result is an
averaging effect across sensors. The contrast across different sensors are
combined into a single plot.

Analysis script: :ref:`sphx_glr_auto_scripts_10-sliding_estimator.py`
"""

###############################################################################
# Let us first import the necessary libraries

import os
import os.path as op
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import sem

sys.path.append(op.join('..', '..', 'processing'))
from autoreject import set_matplotlib_defaults

###############################################################################

data_path = '/home/claire/DATA/Data_Face_House_new_proc/Analysis/Sliding_Estimator'

exclude=[7]

# Now we loop over subjects to load the scores
 # stim/face vs stim/house decodes imag/face vs imag/house
scores = []
for subject_id in range(12, 26):
    if subject_id in exclude:
        continue
    subject = "S%02d" % subject_id

    # Load the scores for the subject
    
    fname_td = os.path.join(data_path, '%s-causal-highpass-2Hz-temp-gene-across-conditions-stim_vs_imag-ave4trials.mat'
                            % (subject))
    mat = loadmat(fname_td)
    scores.append(mat['scores'][0])

# ... and average them
times = mat['times'][0]
mean_scores, sem_scores = dict(), dict()

mean_scores = np.mean(scores, axis=0)
sem_scores = sem(scores)

###############################################################################
set_matplotlib_defaults(plt)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(times, np.diag(mean_scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding EEG sensors over time - Group average 4 trials')
plt.show()

# Plot the full matrix
fig, ax = plt.subplots(1, 1)
im = ax.imshow(mean_scores[a_vs_b], interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal Generalization - %s ' % a_vs_b)
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im, ax=ax)
plt.show()


fig.savefig(op.join('/home/claire/DATA/Data_Face_House_new_proc/Analysis/Figures', 'temp-generalization-matrix-causal-highpass-2Hz-%s-ave4trials.pdf' %a_vs_b), bbox_to_inches='tight')
