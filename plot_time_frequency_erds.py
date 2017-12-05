"""
===============================
Compute and visualize ERDS maps
===============================

This example calculates and displays ERDS maps of event-related EEG data. ERDS
(sometimes also written as ERD/ERS) is short for event-related
desynchronization (ERD) and event-related synchronization (ERS) [1]_.
Conceptually, ERD corresponds to a decrease in power in a specific frequency
band relative to a baseline. Similarly, ERS corresponds to an increase in
power. An ERDS map is a time/frequency representation of ERD/ERS over a range
of frequencies [2]_. ERDS maps are also known as ERSP (event-related spectral
perturbation) [3]_.

We use a public EEG BCI data set containing two different motor imagery tasks
available at PhysioNet. The two tasks are imagined hand and feet movement. Our
goal is to generate ERDS maps for each of the two tasks.

First, we load the data and create epochs of 5s length. The data sets contain
multiple channels, but we will only consider the three channels C3, Cz, and C4.
We compute maps containing frequencies ranging from 2 to 35Hz. We map ERD to
red color and ERS to blue color, which is the convention in many ERDS
publications. Note that we do not perform any significance tests on the map
values, but instead we display the whole time/frequency maps.

References
----------

.. [1] G. Pfurtscheller, F. H. Lopes da Silva. Event-related EEG/MEG
       synchronization and desynchronization: basic principles. Clinical
       Neurophysiology 110(11), 1842-1857, 1999.
.. [2] B. Graimann, J. E. Huggins, S. P. Levine, G. Pfurtscheller.
       Visualization of significant ERD/ERS patterns in multichannel EEG and
       ECoG data. Clinical Neurophysiology 113(1), 43-47, 2002.
.. [3] S. Makeig. Auditory event-related dynamics of the EEG spectrum and
       effects of exposure to tones. Electroencephalography and Clinical
       Neurophysiology 86(4), 283-293, 1993.
"""
# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
import os

def center_cmap(cmap, vmin, vmax):
    """Center given colormap (ranging from vmin to vmax) at value 0.

    Note that eventually this could also be achieved by re-normalizing a given
    colormap by subclassing matplotlib.colors.Normalize as described here:
    https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    """  # noqa: E501
    vzero = abs(vmin) / (vmax - vmin)
    index_old = np.linspace(0, 1, cmap.N)
    index_new = np.hstack([np.linspace(0, vzero, cmap.N // 2, endpoint=False),
                           np.linspace(vzero, 1, cmap.N // 2)])
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}
    for old, new in zip(index_old, index_new):
        r, g, b, a = cmap(old)
        cdict["red"].append((new, r, r))
        cdict["green"].append((new, g, g))
        cdict["blue"].append((new, b, b))
        cdict["alpha"].append((new, a, a))
    return LinearSegmentedColormap("erds", cdict)


# load and preprocess data ####################################################
subject_id=12

subject = 'S%02d' %subject_id
data_path=  os.path.join('/home/claire/DATA/Data_Face_House_new_proc', subject, 'EEG/Preproc')

fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
epochs=mne.read_epochs(fname_in)


picks = mne.pick_channels(epochs.info['ch_names'], ['PO8', 'PO7', 'POz'])

mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)

event_ids= {'stim':100, 'imag': 200}

# compute ERDS maps ###########################################################
fmin= 1.5
fmax = 100
freqs = np.exp(np.linspace(log(fmin), log(fmax), 65))
n_cycles = np.concatenate((np.linspace(1, 8, 47), np.ones(18)*8)) # formule eeglab, allows both low and high freq
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white

tmin, tmax=-0.2, 1.5

for event in event_ids:
    power = tfr_multitaper(epochs[event], freqs=freqs, n_cycles=n_cycles,
                           use_fft=True, return_itc=False, decim=2)
    power.crop(tmin, tmax)

    fig, ax = plt.subplots(1, 4, figsize=(12, 4),
                           gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for i in range(3):
        power.plot([i], baseline=[-1, 0], mode="percent", vmin=vmin, vmax=vmax,
                   cmap=(cmap, False), axes=ax[i], colorbar=False, show=False)
        ax[i].set_title(epochs.ch_names[i], fontsize=10)
        ax[i].axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if i > 0:
            ax[i].set_ylabel("")
            ax[i].set_yticklabels("")
    fig.colorbar(ax[0].collections[0], cax=ax[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
