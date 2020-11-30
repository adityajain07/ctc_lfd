#!/usr/bin/env python
# coding: utf-8

# In[22]:


'''
Author: Aditya Jain
Date  : 16th November, 2020
About : This code contains a RNN model with CTC loss function for primitive segmentation in a video
'''
# import comet_ml at the top of your file
# from comet_ml import Experiment
# # Create an experiment with your api key:
# experiment = Experiment(
#     api_key="epeaAhyRcHSkn92H4kusmbX8k",
#     project_name="ctc-lfd",
#     workspace="adityajain07",
#     log_code="True"
# )
# experiment.set_code()

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
from six.moves.urllib.request import urlretrieve
from tensorflow.python.keras.utils.generic_utils import Progbar
import pickle
import matplotlib.pyplot as plt
from CTCModel import CTCModel as CTCModel
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Input
from keras.optimizers import Adam
from numpy import zeros
import h5py
from keras.preprocessing import sequence
import numpy as np
from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise

# #### Loading IAM Dataset
dataset_path = "/home/aditya/Dropbox/LearningfromDemons/ctc_data/iam_lines.h5"

with h5py.File(dataset_path, "r") as f:
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]
    x_test  = f['x_test'][:]
    y_test  = f['y_test'][:]

x_train = x_train[:1000]

y_train = y_train[:1000]
# y_train = y_train[:, :5]
x_test  = x_test[:100]
y_test  = y_test[:100]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


nb_labels = 80 # number of labels (10, this is digits)
batch_size = 32 # size of the batch that are considered
padding_value = 255 # value for padding input observations
nb_epochs = 10 # number of training epochs
nb_train = len(x_train)
nb_test = len(x_test)
nb_features = x_train.shape[1]

print(nb_train, nb_test, nb_features)


# create list of input lengths
x_train_len = np.asarray([x_train.shape[2] for i in range(nb_train)])
x_test_len = np.asarray([x_train.shape[2] for i in range(nb_test)])
y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
y_test_len = np.asarray([len(y_test[i]) for i in range(nb_test)])


def create_network(shape, nb_labels, padding_value):

    # Define the network architecture
    # input_data = Input(name='input', shape=(None, nb_features)) # nb_features = image height
    input_data = Input(name='input', shape=shape)
    # masking = Masking(mask_value=padding_value)(input_data)
    noise = GaussianNoise(0.01)(input_data)
    blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(noise)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
    # blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)

    dense = TimeDistributed(Dense(nb_labels + 1, name="dense"))(blstm)
    outrnn = Activation('softmax', name='softmax')(dense)

    network = CTCModel([input_data], [outrnn])
    network.compile(Adam(lr=0.0001))

    return network


print("Network parameters: ", nb_features, nb_labels, padding_value)

# define a recurrent network using CTCModel
network = create_network(x_train.shape[1:], nb_labels, padding_value)


print(network.summary())

# CTC training
network.fit(x=[x_train, y_train, x_train_len, y_train_len], y=np.zeros(nb_train), batch_size=batch_size, epochs=nb_epochs)


# Evaluation: loss, label error rate and sequence error rate are requested
eval = network.evaluate(x=[x_test, y_test, x_test_len, y_test_len],                            batch_size=batch_size, metrics=['loss', 'ler', 'ser'])


# predict label sequences
pred = network.predict([x_test, x_test_len], batch_size=batch_size, max_value=padding_value)
for i in range(10):  # print the 10 first predictions
    print("Prediction :", [j for j in pred[i] if j!=-1], " -- Label : ", y_test[i]) # 


