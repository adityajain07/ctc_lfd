#!/usr/bin/env python
# coding: utf-8

# In[82]:


from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense, Input, Reshape, TimeDistributed, Lambda, LSTM, Bidirectional, Conv2D, MaxPooling2D, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model as KerasModel

from helper_func.lenet import lenet
from helper_func.misc import slide_window
from helper_func.ctc import ctc_decode


# #### Importing Data

# In[71]:


import h5py
import matplotlib.pyplot as plt
from helper_func.misc import slide_window

dataset_path = "/home/aditya/Dropbox/LearningfromDemons/ctc_data/iam_lines.h5"

with h5py.File(dataset_path, "r") as f:
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]
    x_test  = f['x_test'][:]
    y_test  = f['y_test'][:]
    
print(x_train.shape)
print(y_train.shape)


# #### Defining LeNet Keras

# In[79]:


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


# In[84]:


#### Doing Here

input_shape = (28,952)
image_height, image_width = input_shape
window_width = image_height
window_stride = window_width/2
no_classes    = 10
output_length = 97
num_windows   = int((image_width - window_width) / window_stride) + 1

image_input  = Input(shape=input_shape, name="image")
y_true       = Input(shape=(output_length,), name="y_true")
input_length = Input(shape=(1,), name="input_length")
label_length = Input(shape=(1,), name="label_length")

# input_data    = Input(name="input", shape=input_shape)
image_reshape = Reshape((image_height, image_width, 1))(image_input)
image_patches = Lambda(slide_window, arguments=
                       {"window_width": window_width, "window_stride": window_stride})(image_reshape)


convnet_outputs = TimeDistributed(cnn_model)(image_patches)
blstm           = LSTM(128, name="lstm1", return_sequences=True)(convnet_outputs)
softmax_output  = Dense(no_classes, activation="softmax")(blstm)

# model      = keras.Model(inputs=input_data, outputs=output)  

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


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


# In[27]:





# In[68]:


# data_pt = x_train[:10]
# print(data_pt.shape)
#
# ans = model.predict(data_pt)
# print(ans.shape)


# In[ ]:




