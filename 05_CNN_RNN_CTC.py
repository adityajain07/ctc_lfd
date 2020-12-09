#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Author       : Aditya Jain
Date Started : This notebook was created on 2nd December, 2020
About        : Implementing CNN+RNN+CTC
'''
# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key:
experiment = Experiment(
    api_key="epeaAhyRcHSkn92H4kusmbX8k",
    project_name="ctc-lfd",
    workspace="adityajain07",
    log_code="True"
)
experiment.set_code()


# In[1]:


from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense, Input, Reshape, TimeDistributed, Lambda, LSTM, Bidirectional, Conv2D, MaxPooling2D, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model 
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# #### Importing MIME Data

# In[2]:


data_read  = pickle.load(open("/home/aditya/Dropbox/LearningfromDemons/ctc_data/MIME_full.pickle","rb"))

image_data = data_read['data_image']
labels     = data_read['data_label']
prim_map   = data_read['primitive_map']
label_map  = data_read['label_map']

labels  = pad_sequences(labels, padding='post', value = 0)  # making sure all labels are of equal length

print(image_data.shape)
print(labels.shape)
print(prim_map)
print(label_map)

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=43)  
# note: passing a value to random_state produces the exact split every time

print("Training Data: ", x_train.shape, y_train.shape)
print("Testing Data: ", x_test.shape, y_test.shape)

no_classes    = len(prim_map)+1      # one extra label bcz of padding
max_label_len = labels.shape[-1]

training_pts  = int(x_train.shape[0])
test_pts      = int(x_test.shape[0])

print("Total classes of primitives: ", no_classes)
print("Max label length: ", max_label_len)


# In[3]:


print("Total training points: ", training_pts)
print("Total test points: ", test_pts)


# #### Model Architecture

# In[43]:


#### Doing Here

image_shape = x_train.shape[1:]        # the image shape
no_channels = 1                        # no of channels in the image, 3 in case of RGB
print(image_shape)

# no_classes        = 80
# max_label_len = 4
print(type(image_shape[0]))

# architecture is defined below

inputs     = Input(shape=image_shape)
reshape1   = Reshape((image_shape[0], image_shape[1], 1))(inputs)
conv_1     = Conv2D(32, (3,3), activation = 'relu', padding='same')(reshape1)
max_pool1  = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2     = Conv2D(64, (3,3), activation = 'relu', padding='same')(max_pool1)
max_pool2  = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3     = Conv2D(64, (3,3), activation = 'relu', padding='same')(max_pool2)
max_pool3  = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4     = Conv2D(64, (3,3), activation = 'relu', padding='same')(max_pool3)
max_pool4  = MaxPooling2D(pool_size=(2, 2))(conv_4)
squeezed   = Lambda(lambda x: K.squeeze(x, 1))(max_pool4)
# reshape    = Reshape(target_shape=(int(image_shape[0]/8), int(image_shape[1]/8*64)))(max_pool3)
# dense1     = Dense(64)(reshape)                                                  # this dense helps reduce no of params
blstm1     = Bidirectional(LSTM(64, return_sequences=True))(squeezed)
outputs    = Dense(no_classes+1, activation="softmax")(blstm1)


model_arch = Model(inputs, outputs)           # for viz the model architecture
model_arch.summary()


# #### Loss Function

# In[44]:


labels       = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 

loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')


# In[45]:


train_input_length = np.asarray([squeezed.shape[1] for i in range(training_pts)])              # the number of timesteps that go as input to LSTM layer
train_label_length = np.asarray([max_label_len for i in range(training_pts)])

test_input_length = np.asarray([squeezed.shape[1] for i in range(test_pts)])
test_label_length = np.asarray([max_label_len for i in range(test_pts)])


# #### Defining Callbacks for the training process

# In[77]:


class AccuracyCallback(keras.callbacks.Callback):
    '''
    The callback calculates the accuracy on training and test data at the end of every epoch
    
    Arguments:
    
    '''
    def __init__(self, pred_model, x_train, y_train, x_test, y_test):
        super(AccuracyCallback, self).__init__()
        self.train_acc = 0
        self.val_acc  = 0
        self.x_train   = x_train
        self.y_train   = y_train
        self.x_test    = x_test
        self.y_test    = y_test
        self.weights   = None
        self.pred_model= pred_model
    
    def on_epoch_end(self, epoch, logs=None):
        print("End of epoch number: ", epoch)
#         print(self.x_train.shape, self.y_train.shape)
#         print(self.x_test.shape, self.y_test.shape)
        
        self.model.save_weights('callback_model.hdf5')
        self.pred_model.load_weights('callback_model.hdf5')
        
        self.train_accuracy()
        self.val_accuracy()
        
    def train_accuracy(self):
        '''calculates accuracy on train data'''
        train_pred = self.pred_model.predict(x_train)
        decode_pred = K.get_value(K.ctc_decode(train_pred, input_length=np.ones(train_pred.shape[0])*train_pred.shape[1],
                         greedy=True)[0][0])
        
        train_points = self.x_train.shape[0]
        count       = 0
        
        # removing all extra label or -1's induced by CTC
        for i in range(train_points):   
            pred_label = []  # the final label
            
            x = decode_pred[i]
            for item in x:
                if item!=-1:
                    pred_label.append(item)
                
            pred_label = np.asarray(pred_label)            
            if np.array_equal(pred_label,y_train[i]):
                count += 1
                print("Correct Predictions on Train: ", pred_label, y_train[i])
        
        self.train_acc = count/train_points*100
        print("The training accuracy is: ", self.train_acc)
        
    
    def val_accuracy(self):
        '''calculates accuracy on test data'''
        test_pred = self.pred_model.predict(x_test)
        decode_pred = K.get_value(K.ctc_decode(test_pred, input_length=np.ones(test_pred.shape[0])*test_pred.shape[1],
                         greedy=True)[0][0])
        
        test_points = self.x_test.shape[0]
        count       = 0
        
        # removing all extra label or -1's induced by CTC
        for i in range(test_points):   
            pred_label = []  # the final label
            
            x = decode_pred[i]
            for item in x:
                if item!=-1:
                    pred_label.append(item)
                
            pred_label = np.asarray(pred_label)            
            if np.array_equal(pred_label,y_test[i]):
                count += 1
                print("Correct Predictions on Test: ", pred_label, y_test[i])
        
        self.val_acc = count/test_points*100
        print("The validation accuracy is: ", self.val_acc)


# #### Training

# In[78]:


model.fit(x=[x_train, y_train, train_input_length, train_label_length], y=np.zeros(training_pts), epochs=20,
         validation_data = ([x_test, y_test, test_input_length, test_label_length], [np.zeros(test_pts)]),
         callbacks=[AccuracyCallback(model_arch, x_train, y_train, x_test, y_test)],
         batch_size=32)


# #### Accuracy Calculation
# After the final epoch is trained, it calculates the training and test accuracy

# In[ ]:


accuracy = model_accuracy(model, input_data, labels)


# #### Inference

# In[86]:


## Saving the model

model.save_weights('first_run.hdf5')


# In[38]:


# model.save_weights('first_run.hdf5')
model_arch.load_weights('first_run.hdf5')
 
# predict outputs on validation images
test_points = 5

# Inference data
infer_data    = x_train[:test_points]
infer_label   = y_train[:test_points]

prediction  = model_arch.predict(infer_data)

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
 


# In[50]:


for i in range(test_points):
    print("True label: ", infer_label[i])
    
    pred_lab = []
    x = out[i]
    for i in x:
        if i!=-1:
            pred_lab.append(i)
            
    print("Predicted label: ", np.asarray(pred_lab))
    print(type(infer_label[i]))
    print('\n')
    


# In[74]:


a = np.array([1, 2, 4])
b = np.array([1, 2, 4])

np.array_equal(a,b)


# In[63]:


print(a.any())


# In[ ]:




