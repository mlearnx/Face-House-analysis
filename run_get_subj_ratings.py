# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:23:26 2017

@author: claire

Do:
    - Read CSV from subject's trial file
    - Get catch trial accuracy and reaction time


"""

import numpy as np
import os
import pandas as pd


ana_path = '/home/claire/DATA/Data_Face_House_new_proc'
exclude = [7]  # Excluded subjects
all_data = pd.DataFrame()

# get all catch trials data

for subject_id in range(12, 26):
    if subject_id in exclude:
        continue
    elif subject_id == 12:
        n_skip = 0
    else:
         n_skip=80
        
        
    subject = "S%02d" % subject_id
    print("processing subject: %s" % subject)
    data_path =  os.path.join(ana_path, subject)
    fname = os.path.join(data_path, '%s.csv' %subject)
    
    data = pd.read_csv(fname, sep=',', header='infer' ,  skiprows = n_skip )
    
    
    all_data= all_data.append(data[data['scale1'] != '--'])



