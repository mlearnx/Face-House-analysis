# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:23:16 2017

@author: claire
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
print(__doc__)


import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import sem





ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

condition = 'imagery'#, 'imag-face_vs_imag-house'
er_freqrange ={'imagery':list()} #,'imag-face_vs_imag-house':list()

gfp_all = {'Delta': list(), 'Theta':list(), 'Alpha':list(), 'Beta': list(), 'Gamma':list()}
times_all = {'Delta': list(),'Theta':list(), 'Alpha':list(), 'Beta': list(), 'Gamma':list()}
ci_low = {'Delta': list(),'Theta':list(), 'Alpha':list(), 'Beta': list(), 'Gamma':list()}
ci_up = {'Delta': list(),'Theta':list(), 'Alpha':list(), 'Beta': list(), 'Gamma':list()}

iter_freqs = [ 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']


for band in iter_freqs:
    
    for subject_id in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
        subject = 'S%02d' %subject_id
        data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/No_Low_pass')
        
        gfp_fname = os.path.join(data_path, '%s-gfp-%s-%s.mat'% (subject, condition, band))
        mat = loadmat(gfp_fname)
        gfp_all[band].append(mat['gfp'])
        times_all[band].append(mat['times'])
        ci_up[band].append(mat['ci_up'])
        ci_low[band].append(mat['ci_low'])

# average across subjects

mean_gfp, mean_ci_up, mean_ci_low = dict(), dict(), dict()
for band in iter_freqs:
    mean_gfp[band] = np.mean(gfp_all[band], axis = 0)
    mean_ci_up[band]=np.mean(ci_up[band], axis = 0)
    mean_ci_low[band]=np.mean(ci_low[band], axis = 0)


# Plot group results
freqs = [('Delta', 1, 3), 
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45), 
            ]

fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.cm.viridis((0.08, 0.2, 0.35, 0.75, 0.95))
for ((band, fmin, fmax)), color, ax in zip(
        freqs, colors, axes.ravel()[::-1]):
    times = times_all[band][0]
    ax.plot(times, mean_gfp[band], label=band, color=color, linewidth=2.5)
    ax.plot(times, np.zeros_like(times), linestyle='--', color='red',
            linewidth=1)
    ax.fill_between(times[0], mean_gfp[band][0] + mean_ci_up[band][0], mean_gfp[band][0] - mean_ci_low[band][0], color=color,
                    alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('Mean GFP Group')
    ax.annotate('%s (%d-%dHz)' % (band, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')
    ax.set_xlim(-500, 1500)

axes.ravel()[-1].set_xlabel('Time [ms]')
#axes.ravel()[-1].set_ylim([-2, 3])
#ax.set_title('Group average event related dynamic in frequency bands %s' % (condition) )

plt.savefig(ana_path + 'group-tf-gfp-%s.pdf' %condition, bbox_to_inches='tight')