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


data_path = '/home/claire/DATA/Data_Face_House_new_proc/Analysis/Sliding_Estimator'

exclude = [7, 13, 19]
a_vs_b = 'imag-face_vs_imag-house'
scores = {'imag-face_vs_imag-house': list()}

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    
    
    fname_td = os.path.join(data_path, '%s-causal-highpass-2Hz-td-auc-%s.mat'
                            % (subject, a_vs_b))
    mat = loadmat(fname_td)
    scores[a_vs_b].append(mat['scores'][0])


alpha = 0.05
chance = 0.5

times = mat['times']

# Compute stats: is decoding different from theoretical chance level (using
# permutations across subjects)
p_values = stats(np.array(scores[a_vs_b]) - chance)

# Save stats results
print('save')
fname = os.path.join(data_path, 'stats-causal-highpass-2Hz-td-auc-%s.mat'
                            % (a_vs_b))

savemat(fname, {'scores': scores, 'p_vals': p_values, 
                       'times': times, 'alpha': alpha, 'chance': chance })




    
    