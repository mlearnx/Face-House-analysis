# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:05:28 2017

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



from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 150
n_classes = 2
in_chans = train_set.X.shape[1]
# final_conv_length determines the size of the receptive field of the ConvNet
model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=input_time_length,
                        final_conv_length=12).create_network()
to_dense_prediction_model(model)

if cuda:
    model.cuda()

from torch import optim

optimizer = optim.Adam(model.parameters())


from braindecode.torch_ext.util import np_to_var
# determine output size
test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
if cuda:
    test_input = test_input.cuda()
out = model(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
print("{:d} predictions per input/trial".format(n_preds_per_input))

from braindecode.datautil.iterators import CropsFromTrialsIterator
iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
                                  n_preds_per_input=n_preds_per_input)

# Training Loop

from braindecode.torch_ext.util import np_to_var, var_to_np
import torch.nn.functional as F
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_for_set
rng = RandomState((2017,6,30))
for i_epoch in range(20):
    # Set model to training mode
    model.train()
    for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
        net_in = np_to_var(batch_X)
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(batch_y)
        if cuda:
            net_target = net_target.cuda()
        # Remove gradients of last backward pass from all parameters
        optimizer.zero_grad()
        outputs = model(net_in)
        # Mean predictions across trial
        # Note that this will give identical gradients to computing
        # a per-prediction loss (at least for the combination of log softmax activation
        # and negative log likelihood loss which we are using here)
        outputs = th.mean(outputs, dim=2)[:,:,0]
        loss = F.nll_loss(outputs, net_target)
        loss.backward()
        optimizer.step()

    # Print some statistics each epoch
    model.eval()
    print("Epoch {:d}".format(i_epoch))
    for setname, dataset in (('Train', train_set),('Test', test_set)):
        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2)[:,:,0]
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        print("{:6s} Loss: {:.5f}".format(setname, loss))
        # Assign the predictions to the trials
        preds_per_trial = compute_preds_per_trial_for_set(all_preds,
                                                          input_time_length,
                                                          dataset)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == dataset.y)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))











