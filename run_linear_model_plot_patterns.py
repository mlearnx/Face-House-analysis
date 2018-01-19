"""
Get the topographical patternes associated with LinearModel


"""  # noqa: E501

###############################################################################
# Let us first import the libraries

import os

import numpy as np
from scipy.io import savemat

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore
from mne import io, EvokedArray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from mne.decoding import Vectorizer, get_coef
from sklearn.model_selection import permutation_test_score
from mne.decoding import LinearModel

print(__doc__)

###############################################################################
# Then we write a function to do time decoding on one subject

ana_path = '/home/claire/DATA/Data_Face_House_new_proc'
results_path = os.path.join(ana_path, 'Analysis', 'Sliding_Estimator')

condition1 = 'imag/face'
condition2 ='imag/house'

all_coefs = list()

all_evokeds = list()

exclude =[7]

for subject_id in range(1,26):
    if subject_id in exclude:
        continue
    else:

    
        print("processing subject: %s (%s vs %s)"
              % (subject_id, condition1, condition2))
    
        subject = "S%02d" % subject_id
        data_path = os.path.join(ana_path, subject,'EEG', 'New_Preproc' )
        epochs = mne.read_epochs(os.path.join(data_path,
                                 '%s-causal-highpass-2Hz-epo.fif' %subject))
    
        epochs.interpolate_bads()
        # We define the epochs and the labels
        epochs =epochs[condition1, condition2]
        epochs.apply_baseline()
    
        # Let us restrict ourselves to the MEG channels, and also decimate to
        # make it faster (although we might miss some detail / alias)
        epochs.pick_types(eeg=True).decimate(2, verbose='error')
        #mne.epochs.combine_event_ids(epochs, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
       # mne.epochs.combine_event_ids(epochs, ['imag/face', 'imag/house'], {'imag':200}, copy=False)
        
        # Get the data and labels
        X = epochs.get_data()
        # fit and time decoder
        le=LabelEncoder()
        y = le.fit_transform(epochs.events[:, 2])  # target: Audio left or right
    
        # Use AUC because chance level is same regardless of the class balance
        clf = make_pipeline(
            Vectorizer(), 
            StandardScaler(), 
            LinearModel(LogisticRegression()))
            
        
        time_decod= SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')
        time_decod.fit(X,y)
        
        
        # The `inverse_transform` parameter will call this method on any estimator
        # contained in the pipeline, in reverse order.
        coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
        
        
        all_coefs.append(coef)
        
        

        
all_coefs_ave = np.mean(all_coefs, axis=0)




test = EvokedArray(all_coefs_ave, epochs.info, tmin=epochs.times[0])


#use 170
fig =test.plot_joint(times=[0.,.050,0.100, 0.15, 0.25],title='Group EEG pattern %s %s' % (condition1,condition2))

fig.savefig(os.path.join('/home/claire/DATA/Data_Face_House_new_proc/Analysis/Figures', 'group_pattern-imag-face_imag-house.pdf'), bbox_to_inches='tight')

        





#for subject_id in range(1,26):
#    if subject_id in exclude:
#        continue
#    else:
#        run_time_decoding(subject_id, 'stim/face', 'stim/house')
#        
#for subject_id in range(1,26):
#    if subject_id in exclude:
#        continue
#    else:
#        run_time_decoding(subject_id, 'imag/face', 'imag/house')



