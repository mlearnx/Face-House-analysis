# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:47:22 2017

@author: claire
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import sem

from mne import create_info
from mne.time_frequency import AverageTFR

ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

a_vs_bs = ['imag-face_vs_imag-house'] #, 'imag-face_vs_imag-house'
tf_scores={'imag-face_vs_imag-house':list()} #,'imag-face_vs_imag-house':list()


for subject_id in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/No_Low_pass')
    
    # load time-freq score for each subject
    for a_vs_b in a_vs_bs:
        fname_tf = os.path.join(data_path, '%s-csp-time-freq-%s.mat' %(subject, a_vs_b))
        mat = loadmat(fname_tf)
        tf_scores[a_vs_b].append(mat['scores'])

# average scores across subjects
freqs = mat['freqs'][0]
mean_tf_scores, sem_tf_scores = dict(), dict()
for a_vs_b in a_vs_bs:
    mean_tf_scores[a_vs_b] = np.mean(tf_scores[a_vs_b], axis = 0)
    sem_tf_scores[a_vs_b] = sem(tf_scores[a_vs_b])

sfreq= mat['sfreq']
centered_w_times = [-0.3, -0.1,  0.1,  0.3,  0.5,  0.7,  0.9,  1.1,  1.3]
                
# Plot time-frequency results across subject
for  a_vs_b in  a_vs_bs:
    # Set up time frequency object
    av_tfr = AverageTFR(create_info(['freq'], sfreq), mean_tf_scores[a_vs_b][np.newaxis, :],
                        centered_w_times, freqs[1:], 1)
    
    chance = 0.5#np.mean(y)  # set chance level to white in the plot
    av_tfr.plot([0], vmin=chance, title="Time-Frequency Decoding Scores %s" %a_vs_b,
                cmap=plt.cm.Reds)

plt.savefig(ana_path + 'csp_time_freq_decoding_%s.pdf' %a_vs_b, bbox_to_inches='tight')

###############################################################################
# Plot frequency results

f_scores={'imag-face_vs_imag-house':list()} #,'imag-face_vs_imag-house':list()

for subject_id in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/No_Low_pass')
    
    # load freq score for each subject
    for a_vs_b in a_vs_bs:
        fname_f = os.path.join(data_path, '%s-csp-freq-%s.mat' %(subject, a_vs_b))
        mat = loadmat(fname_f)
        f_scores[a_vs_b].append(mat['scores'])

# average scores across subjects
freqs = mat['freqs'][0]
mean_f_scores, sem_f_scores = dict(), dict()
for a_vs_b in a_vs_bs:
    mean_f_scores[a_vs_b] = np.mean(f_scores[a_vs_b], axis = 0)
    sem_f_scores[a_vs_b] = sem(f_scores[a_vs_b])

for  a_vs_b in  a_vs_bs:
    plt.bar(left=freqs[:-1], height=mean_f_scores[a_vs_b][0], width=np.diff(freqs)[0],
            align='edge', edgecolor='black')
    plt.xticks(freqs)
    plt.ylim([0, 1])
    plt.axhline(0.50, color='k', linestyle='--',
                label='chance level')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Decoding Scores')
    plt.title('Frequency Decoding Scores %s' % a_vs_b)

plt.savefig(ana_path + 'csp_freq_decoding_%s.pdf' %a_vs_b, bbox_to_inches='tight')
    