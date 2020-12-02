#!/usr/bin/env python
# coding: utf-8

# In[129]:


from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense, Input, Reshape, TimeDistributed, Lambda, LSTM, Bidirectional, Conv2D, MaxPooling2D, Flatten
import tensorflow.keras.backend as K

from helper_func.lenet import lenet
from helper_func.misc import slide_window
from helper_func.ctc import ctc_decode

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# #### Importing Data

# In[131]:


import h5py
import matplotlib.pyplot as plt
from helper_func.misc import slide_window

dataset_path = "/home/aditya/Dropbox/LearningfromDemons/ctc_data/iam_lines.h5"
no_classes   = 80

with h5py.File(dataset_path, "r") as f:
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]
    x_test  = f['x_test'][:]
    y_test  = f['y_test'][:]
    

x_train = x_train[:1000]
y_train = y_train[:1000]
x_test  = x_test[:1000]
y_test  = y_test[:1000]
    
y_train = to_categorical(y_train, no_classes)
y_test  = to_categorical(y_train, no_classes)

print(x_train.shape)
print(y_train.shape)


# #### Defining LeNet Keras

# In[132]:


input_shape = (28,28,1)


cnn_model = keras.Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64)
    ]
)


# In[138]:


#### Doing Here

input_shape = (28,952)
image_height, image_width = input_shape
window_width = image_height
window_stride = window_width/4
no_classes    = 80
output_length = 97
num_windows   = int((image_width - window_width) / window_stride) + 1

image_input  = Input(shape=input_shape, name="image")
y_true       = Input(shape=(output_length,), name="y_true")
input_length = Input(shape=(1,), name="input_length")
label_length = Input(shape=(1,), name="label_length")

image_reshape = Reshape((image_height, image_width, 1))(image_input)
image_patches = Lambda(slide_window, arguments=
                       {"window_width": window_width, "window_stride": window_stride})(image_reshape)


convnet_outputs = TimeDistributed(cnn_model)(image_patches)
blstm           = LSTM(128, name="lstm1", return_sequences=True)(convnet_outputs)
softmax_output  = Dense(no_classes, activation="softmax", name="softmax_out")(blstm)


input_length_processed = Lambda(lambda x, num_windows=None: x * num_windows, 
                                arguments={"num_windows": num_windows})(input_length)

ctc_loss_output        = Lambda(lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]), 
                         name="ctc_loss")([y_true, softmax_output, input_length_processed, label_length])
ctc_decoded_output     = Lambda(lambda x: ctc_decode(x[0], x[1], output_length), name="ctc_decoded")(
        [softmax_output, input_length_processed])
    
model                  = KerasModel(inputs=[image_input, y_true, input_length, label_length], 
                                    outputs=[ctc_loss_output, ctc_decoded_output])

model.compile(optimizer="adam")

print(model.summary())


# In[140]:


import numpy as np

x_train     = np.asarray(x_train)
y_train     = np.asarray(y_train)
x_train_len = np.asarray([133 for i in range(1000)])
y_train_len = np.asarray([97 for i in range(1000)])

print(np.shape(x_train_len), np.shape(y_train_len))

model.fit([x_train, y_train, x_train_len, y_train_len])


# In[128]:


model.evaluate(x_test, y_test)


# In[ ]:




