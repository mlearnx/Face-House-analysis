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

a_vs_bs = ['gen_across_time_stim_ave_4_trials']
scores={'gen_across_time_stim_ave_4_trials':list()}

exclude = [7]
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,12):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass/')
    
    # load score for each subject
    for a_vs_b in a_vs_bs:
        fname_td = os.path.join(data_path, '%s_%s.mat' %(subject, a_vs_b))
        mat = loadmat(fname_td)
        scores[a_vs_b].append(mat['scores'][0])
# average subjects

times = mat['times'][0]
mean_scores, sem_scores = dict(), dict()

for a_vs_b in a_vs_bs:
    mean_scores[a_vs_b] = np.mean(scores[a_vs_b], axis = 0)
    sem_scores[a_vs_b] = sem(scores[a_vs_b])
    
    
#  # Plot the diagonal across subjects

colors = ['b', 'g']    

for c, a_vs_b in zip(colors, a_vs_bs):
    plt.plot(times, np.diag(mean_scores[a_vs_b]), label=a_vs_b.replace('_', ' '))
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
plt.savefig(ana_path + 'group_gen_across_time_decoding_average_4_trials.pdf', bbox_to_inches='tight')



# Plot the full matrix
fig, ax = plt.subplots(1, 1)
im = ax.imshow(mean_scores[a_vs_b], interpolation='lanczos', origin='lower', cmap='RdBu_r',
               extent=epochs.times[[0, -1, 0, -1]], vmin=0., vmax=1.)
ax.set_xlabel('Testing Time (s)')
ax.set_ylabel('Training Time (s)')
ax.set_title('Temporal Generalization - %s subject %s' %(cond, subject))
ax.axvline(0, color='k')
ax.axhline(0, color='k')
plt.colorbar(im, ax=ax)
plt.show()
plt.savefig(ana_path + ' gen_across_time_matrix_%s_%s_ave_4_trials.pdf' %(cond,subject),  bbox_to_inches='tight')
















