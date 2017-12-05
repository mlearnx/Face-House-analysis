# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:03:25 2017

@author: claire
"""

import mne
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from braindecode.datautil.signal_target import SignalAndTarget



exclude = [7]
all_epochs=list()
# We start by exploring the frequence content of our epochs.
for subject_id in range(1,6):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs_train = epochs=mne.concatenate_epochs(all_epochs)

mne.epochs.combine_event_ids(epochs_train, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
mne.epochs.combine_event_ids(epochs_train, ['imag/face', 'imag/house'], {'imag':200}, copy=False)


# Load Test subject
all_epochs=list()
for subject_id in range(7, 11):
    if subject_id in exclude:
        continue
    subject = 'S%02d' %subject_id
    data_path = os.path.join('/home/claire/DATA/Data_Face_House/' + subject +'/EEG/Evoked_Lowpass')
    fname_in = os.path.join(data_path, '%s-epo.fif' %subject)
    epochs=mne.read_epochs(fname_in)
    epochs.interpolate_bads()
    all_epochs.append(epochs)
    
epochs_test = epochs=mne.concatenate_epochs(all_epochs)

mne.epochs.combine_event_ids(epochs_test, ['stim/face', 'stim/house'], {'stim':100}, copy=False)    
mne.epochs.combine_event_ids(epochs_test, ['imag/face', 'imag/house'], {'imag':200}, copy=False)



# convert to braindecode format
le = LabelEncoder()
train_X=(epochs_train.get_data()*1e6).astype(np.float32)
train_y= le.fit_transform(epochs_train.events[:,2]).astype(np.int64)
test_X=(epochs_test.get_data()*1e6).astype(np.float32)
test_y= le.fit_transform(epochs_test.events[:,2]).astype(np.int64)

train_set = SignalAndTarget(train_X, y=train_y)
test_set = SignalAndTarget(test_X, y=test_y)

# create model
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from braindecode.torch_ext.util import set_random_seeds

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = False
set_random_seeds(seed=20170629, cuda=cuda)
n_classes = 2
in_chans = train_set.X.shape[1]
# final_conv_length = auto ensures we only get a single output in the time dimension
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto').create_network()
if cuda:
    model.cuda()
    

from torch import optim

optimizer = optim.Adam(model.parameters())


# Training loop

from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from numpy.random import RandomState
rng = RandomState((2017,6,30))
for i_epoch in range(30):
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                            batch_size=30)
    # Set model to training mode
    model.train()
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X
        batch_X = train_set.X[i_trials][:,:,:,None]
        batch_y = train_set.y[i_trials]
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        # Compute outputs of the network
        outputs = model(net_in)
        # Compute the loss
        loss = F.nll_loss(outputs, net_target)
        # Do the backpropagation
        loss.backward()
        # Update parameters with the optimizer
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    for setname, dataset in (('Train', train_set), ('Test', test_set)):
        # Here, we will use the entire dataset at once, which is still possible
        # for such smaller datasets. Otherwise we would have to use batches.
        net_in = np_to_var(dataset.X[:,:,:,None])
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(dataset.y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in)
        loss = F.nll_loss(outputs, net_target)
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))




















