# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:17:12 2017

@author: claire
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:09:36 2017

@author: claire
"""

from __future__ import print_function

import mne
from mne import io
import os
import os.path as op
from mne.preprocessing import ICA, create_eog_epochs
from autoreject import get_rejection_threshold
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults) 
print(__doc__)

#--------------------------
# Import Data
#--------------------------

overwrite = True


ana_path = '/home/claire/DATA/Data_Face_House_new_proc'

event_id = {'stim/face' :101, 'stim/house':102, 'imag/face':201, 'imag/house':202}

baseline = (None, 0)


exclude = [7]

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join(ana_path, subject , 'EEG', 'New_Preproc')

    raw_fname_ica = os.path.join(data_path,'%s-noncausal-highpass-1Hz-raw.fif' %subject)
    ica_name = os.path.join(data_path,'%s-ica.fif' %subject)
    
                            
    
    print(subject)

    raw =io.read_raw_fif(raw_fname_ica)
    
    
    print('  Fitting ICA')
    
    n_components=25
    
    ica = ICA(method='fastica', random_state=42, n_components=n_components)
    picks = mne.pick_types(raw.info, eeg=True, eog=False,
                           stim=False, exclude='bads')
    ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12),
            decim=11)
    print('  Fit %d components (explaining at least %0.1f%% of the variance)'
          % (ica.n_components_, 100 * n_components))
    
    

    

