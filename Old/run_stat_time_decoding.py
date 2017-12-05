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
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.stats import ttest_1samp_no_p
from scipy.io import savemat


# STATISTICS ##################################################################


def _stat_fun(x, sigma=0, method='relative'):
    """Aux. function of stats"""
    t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
    t_values[np.isnan(t_values)] = 0
    return t_values


def stats(X, connectivity=None, n_jobs=-1):
    """Cluster statistics to control for multiple comparisons.

    Parameters
    ----------
    X : array, shape (n_samples, n_space, n_times)
        The data, chance is assumed to be 0.
    connectivity : None | array, shape (n_space, n_times)
        The connectivity matrix to apply cluster correction. If None uses
        neighboring cells of X.
    n_jobs : int
        The number of parallel processors.
    """
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, out_type='mask', stat_fun=_stat_fun, n_permutations=1000,
        n_jobs=n_jobs, connectivity=connectivity)
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
    return np.squeeze(p_values_).T

############################################################################


ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

fname = 'csp-freq-stim-face_vs_stim-house.mat'
scores=list()
for subject_id in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/No_Low_pass')
    
    # load score for each subject
    fname_td = os.path.join(data_path, subject + '-' + fname)
    mat = loadmat(fname_td)
    scores.append(mat['scores'][0])

times = mat['times'][0]

alpha = 0.05
chance = 0.5



# Compute stats: is decoding different from theoretical chance level (using
# permutations across subjects)
p_values = stats(np.array(scores) - chance)

# Save stats results
print('save')

savemat(ana_path+ 'stat_'+ fname , {'scores': scores, 'p_vals': p_values, 
                       'times': times, 'alpha': alpha, 'chance': chance })














    
    