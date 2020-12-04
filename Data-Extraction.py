#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
Author       : Aditya Jain
Date Started : 3rd December, 2020
About        : This notebook generates training data from MIME dataset
'''


# In[6]:


import cv2
import matplotlib.pyplot as plt

# In[7]:


video_path = '/home/aditya/Dropbox/MIME_small/Push/3720Jul27/hd_kinect_rgb.mp4'


vidcap      = cv2.VideoCapture(video_path)     
fps         = vidcap.get(cv2.CAP_PROP_FPS)           #  FPS of the video      
frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)   #  total frame count
total_sec   = frame_count/fps                        #  total video length in seconds

# vidcap.set(cv2.CAP_PROP_POS_MSEC,1)
# flag, image = vidcap.read()
#
# # plt.figure()
# cv2.imshow('Original Image', image)
#
# scaled_down_image = cv2.resize(image, (80, 30))
# cv2.imshow('Scaled Down', scaled_down_image)
#
# gray_image = cv2.cvtColor(scaled_down_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray scale", gray_image)
#
# normal_gray_image = gray_image.astype("float32")/255
# cv2.imshow("Normal gray image", normal_gray_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# print('Hello')
import numpy as np
sec         = 0
n_frames    = 10
TIME_SEC    = total_sec/n_frames

final_img = []
while sec < total_sec:
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # setting which frame to get


    success, image = vidcap.read()

    if success:
        img_shape = image.shape
        image = cv2.resize(image, (int(img_shape[1]/8), int(img_shape[0]/8)))         # resize by a factor of 4
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                     # convert to grayscale
        image = image.astype("float32")/255                                 # normalizing the image from 0 to 1
        # cv2.imshow('Image', image)
        if sec==0:
            final_img = image
        else:
            final_img = np.hstack((final_img, image))

        print(image.shape)
    sec += TIME_SEC

cv2.imshow("final concatenated image", final_img)
# cv2.imwrite("concatenated1.jpg", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Hello')


