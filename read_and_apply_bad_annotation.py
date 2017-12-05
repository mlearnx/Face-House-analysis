# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:04:02 2017

@author: claire
"""

# read in bad segments

bad_segments = np.loadtxt(os.path.join(data_path, '%s_raw_bad_segments.csv' %subject), skiprows=1, delimiter=",")
bad_segments /= raw_ica.info["sfreq"]
raw_ica.annotations = mne.Annotations(*bad_segments.T, "bad")


# read in bad electrodes

with open(os.path.join(data_path, '%s_raw_bad_channels.csv' %subject)) as f:
    raw_ica.info["bads"] = f.read().strip().split(",")
