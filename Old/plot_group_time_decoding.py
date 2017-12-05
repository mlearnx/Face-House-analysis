# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:06:17 2017

@author: claire
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import sem

ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

a_vs_bs = ['stim-face_vs_stim-house_ave_4_trials', 'imag-face_vs_imag-house_ave_4_trials']
scores={'stim-face_vs_stim-house_ave_4_trials':list(),'imag-face_vs_imag-house_ave_4_trials': list()}

for subject_id in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/')
    
    # load score for each subject
    for a_vs_b in a_vs_bs:
        fname_td = os.path.join(data_path, '%s-td-auc-%s.mat' %(subject, a_vs_b))
        mat = loadmat(fname_td)
        scores[a_vs_b].append(mat['scores'][0])
# average subjects

times = mat['times'][0]
mean_scores, sem_scores = dict(), dict()

for a_vs_b in a_vs_bs:
    mean_scores[a_vs_b] = np.mean(scores[a_vs_b], axis = 0)
    sem_scores[a_vs_b] = sem(scores[a_vs_b])
    
    
# Plot mean AUC across subjects

colors = ['b', 'g']    

for c, a_vs_b in zip(colors, a_vs_bs):
    plt.plot(times, mean_scores[a_vs_b], c, label=a_vs_b.replace('_', ' '))
    plt.xlabel('Time (s)')
    plt.ylabel('Area under curve (AUC)')
    plt.fill_between(times, mean_scores[a_vs_b] - sem_scores[a_vs_b],
                     mean_scores[a_vs_b] + sem_scores[a_vs_b],
                     color=c, alpha=0.2)
plt.axhline(0.5, color='k', linestyle='--', label='Chance level')
plt.axvline(0.0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(ana_path + 'time_decoding_average_4_trials.pdf', bbox_to_inches='tight')





# plot individual subjects

fig, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(12, 8))
axes = axes.ravel()
for idx in range(10):
    axes[idx].axhline(0.5, color='k', linestyle='--', label='Chance level')
    axes[idx].axvline(0.0, color='k', linestyle='--')
    for a_vs_b in a_vs_bs:
        axes[idx].plot(times, scores[a_vs_b][idx], label=a_vs_b)
        axes[idx].set_title('sub%03d' % (idx + 1))

axes[-1].axis('off')
axes[-2].legend(bbox_to_anchor=(2.35, 0.5), loc='center right', fontsize=12)
fig.text(0.5, 0, 'Time (s)', ha='center', fontsize=16)
fig.text(0.01, 0.5, 'Area under curve (AUC)', va='center',
         rotation='vertical', fontsize=16)
plt.subplots_adjust(bottom=0.06, left=0.06, right=0.98, top=0.95)
plt.show()