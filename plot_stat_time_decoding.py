# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:15:53 2018

@author: claire
"""

from autoreject import set_matplotlib_defaults
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import sem
import os


data_path = '/home/claire/DATA/Data_Face_House_new_proc/Analysis/Sliding_Estimator'

exclude=[7]

# loop over subjects to load the scores
a_vs_bs = ['stim_vs_imag']
scores = {'stim_vs_imag': list()}
for subject_id in range(1, 26):
    if subject_id in exclude:
        continue
    subject = "S%02d" % subject_id

    # Load the scores for the subject
    for a_vs_b in a_vs_bs:
        fname_td = os.path.join(data_path, '%s-causal-highpass-2Hz-td-auc-%s.mat'
                                % (subject, a_vs_b))
        mat = loadmat(fname_td)
        scores[a_vs_b].append(mat['scores'][0])

# ... and average them
times = mat['times'][0]
mean_scores, sem_scores = dict(), dict()
for a_vs_b in a_vs_bs:
    mean_scores[a_vs_b] = np.mean(scores[a_vs_b], axis=0)
    sem_scores[a_vs_b] = sem(scores[a_vs_b])




# load stat matrix

fname_stats = os.path.join(data_path, 'stats-causal-highpass-2Hz-td-auc-%s.mat'
                                % a_vs_b)
stats = loadmat(fname_stats)

p_values= stats['p_vals']

stat_times = [index for index in range(len(p_values[0])) if p_values[0][index]<=0.001]


# plot with stats
set_matplotlib_defaults(plt)
color = 'b'#
c=color
fig, ax = plt.subplots(1, figsize=(3.5, 2.5))
ax.plot(times, mean_scores[a_vs_b], c, label=a_vs_b.replace('_', ' '))
ax.set(xlabel='Time (s)', ylabel='Area under curve (AUC)')
ax.fill_between(times, mean_scores[a_vs_b] - sem_scores[a_vs_b],
                mean_scores[a_vs_b] + sem_scores[a_vs_b],
                color=c, alpha=0.33, edgecolor='none')
ax.axhline(0.5, color='k', linestyle='--', label='Chance level')
ax.axvline(0.0, color='k', linestyle='--')
ax.legend()

# plot significant time range
ymin, ymax = ax.get_ylim()
ax.fill_betweenx((ymin, ymax), times[stat_times[0]], times[stat_times[-1]],
                             color='orange', alpha=0.3)

ax.legend(loc='upper right')
title = 'Decoding %s (p < 0.001)'%a_vs_b
ax.set(title=title)

# plot figure
fig.tight_layout()
plt.show()


fig.savefig(os.path.join('/home/claire/DATA/Data_Face_House_new_proc/Analysis/Figures', 'stat_time_decoding_highpass-2Hz-%s.pdf' %a_vs_b), bbox_to_inches='tight')


