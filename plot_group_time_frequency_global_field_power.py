# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:23:16 2017

@author: claire
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
print(__doc__)




###############################################################################
    # Now we can compute the Global Field Power
    #
    # Then we prepare a bootstrapping function to estimate confidence intervals
    
rng = np.random.RandomState(42)

def get_gfp_ci(average, n_bootstraps=2000):
    """get confidence intervals from non-parametric bootstrap"""
    indices = np.arange(len(average.ch_names), dtype=int)
    gfps_bs = np.empty((n_bootstraps, len(average.times)))
    for iteration in range(n_bootstraps):
        bs_indices = rng.choice(indices, replace=True, size=len(indices))
        gfps_bs[iteration] = np.sum(average.data[bs_indices] ** 2, 0)
    gfps_bs = mne.baseline.rescale(gfps_bs, average.times, baseline=(None, 0))
    ci_low, ci_up = np.percentile(gfps_bs, (2.5, 97.5), axis=0)
    return ci_low, ci_up

##############################################################################
# Now we can track the emergence of spatial patterns compared to baseline
# for each frequency band
#


fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.cm.viridis((0.1, 0.35, 0.75, 0.95))
for ((freq_name, fmin, fmax), average), color, ax in zip(
        frequency_map, colors, axes.ravel()[::-1]):
    times = average.times * 1e3
    gfp = np.sum(average.data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
    ax.plot(times, np.zeros_like(times), linestyle='--', color='red',
            linewidth=1)
    ci_low, ci_up = get_gfp_ci(average)
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color,
                    alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('GFP')
    ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')
    ax.set_xlim(-1050, 3050)

axes.ravel()[-1].set_xlabel('Time [ms]')
plt.title('Event related dynamic in frequecy bands %s  %s' % (subject, condition) )

plt.savefig(ana_path + 'csp_freq_decoding_%s.pdf' %a_vs_b, bbox_to_inches='tight')