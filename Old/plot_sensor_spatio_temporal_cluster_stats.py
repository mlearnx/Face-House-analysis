"""
============================================================
Non-parametric spatio-temporal statistics on EEG sensor data
============================================================

Run a non-parametric spatio-temporal cluster stats on EEG sensors
on the contrast faces vs. scrambled.
"""

import os.path as op
import numpy as np
from scipy import stats
from scipy import spatial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.viz import plot_topomap


##############################################################################
# Read all the data
ana_path = '/home/claire/DATA/Data_Face_House/Group Analysis/'

exclude = [7]  # Excluded subjects

contrasts = list()

for subject_id in range(1, 11):
    if subject_id in exclude:
        continue
    subject = "S%02d" % subject_id
    print("processing subject: %s" % subject)
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    
    cond2 = mne.read_evokeds(op.join(data_path, '%s-ave.fif'
                                        % subject),
                                condition=['imag/house', 'stim/house'])
    cond1 = mne.read_evokeds(op.join(data_path, '%s-ave.fif'
                                        % subject),
                                condition=['imag/face', 'stim/face'])
    contrast = mne.combine_evoked([cond1[0],cond1[1], cond2[0], cond2[1]], weights=[0.5,0.5,-0.5, -0.5])
    #contrast = mne.combine_evoked([cond1[0], cond2[0]], weights=[0.5,0.5,-0.5, -0.5])
                            
    #contrast = mne.combine_evoked([cond1[0], cond2[0]], weights=[1,-1])
    contrast.apply_baseline((-0.5, 0.0))
    contrasts.append(contrast)

contrast = mne.combine_evoked(contrasts, 'equal')

contrast.interpolate_bads(reset_bads=True)

##############################################################################
# Assemble the data and run the cluster stats on channel data

data = np.array([c.data for c in contrasts])

n_permutations = 1000  # number of permutations to run

# set family-wise p-value
p_accept = 0.01

connectivity = None
tail = 0.  # for two sided test

# set cluster threshold
ppf = stats.t.ppf
p_thresh = p_accept / (1 + (tail == 0))
n_samples = len(data)
threshold = -ppf(p_thresh, n_samples - 1)
if np.sign(tail) < 0:
    threshold = -threshold

# Make a triangulation between EEG channels locations to
# use as connectivity for cluster level stat
# XXX : make a mne.channels.make_eeg_connectivity function
connectivity, ch_names = mne.channels.find_ch_connectivity(contrast.info, 'eeg')


data = np.transpose(data, (0, 2, 1))  # transpose for clustering

cluster_stats = permutation_cluster_1samp_test(
    data, threshold=threshold, n_jobs=2, verbose=True, tail=1,
    connectivity=connectivity, out_type='indices',
    check_disjoint=True)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

print("Good clusters: %s" % good_cluster_inds)

##############################################################################
# Visualize the spatio-temporal clusters

times = contrast.times * 1e3
colors = 'r', 'steelblue'
linestyles = '-', '--'

pos = mne.find_layout(contrast.info).pos

T_obs_max = 5.
T_obs_min = -T_obs_max

# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for T0 stat
    T_obs_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at significant sensors
    signals = data[..., ch_inds].mean(axis=-1)
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    title = 'Cluster #{0}'.format(i_clu + 1)
    fig.suptitle(title, fontsize=14)

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(T_obs_map, pos, mask=mask, axes=ax_topo,
                            vmin=T_obs_min, vmax=T_obs_max,
                            show=False)

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel('Averaged T-map ({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]
    ))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    for signal, name, col, ls in zip(signals, ['Contrast'], colors,
                                     linestyles):
        ax_signals.plot(times, signal, color=col, linestyle=ls, label=name)

    # add information
    ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
    ax_signals.set_xlim([times[0], times[-1]])
    ax_signals.set_xlabel('time [ms]')
    ax_signals.set_ylabel('evoked [uV]')

    # plot significant time range
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
    ax_signals.legend(loc='lower right')
    ax_signals.set_ylim(ymin, ymax)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()
   # plt.savefig(ana_path+ 'spatiotemporal_stats_cluster-%02d-stim_face_vs_imag_face.pdf' % i_clu)
