import cv2
import numpy as np
import tensorflow.keras.backend as K

class Inference:
	def __init__(self, video_path, n_frames, down_f, d_shape, model):
		''' This class hosts all functions needed to predict inference on a test image'''
		self.video_path = video_path
		self.input_data = None
		self.n_frames   = n_frames
		self.down_f		= down_f
		self.d_shape	= d_shape
		self.model	    = model

	def prep_data(self):
		'''prepares the data on which the model will do the prediction'''
		vidcap	 = cv2.VideoCapture(self.video_path)	
		fps		 = vidcap.get(cv2.CAP_PROP_FPS)		#  FPS of the video	  
		frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)#  total frame count
		total_sec   = frame_count/fps
		sec		 = 0
		time_sec	= total_sec/self.n_frames					 # the video will be sampled after every time_sec
		final_img   = []
		
		while sec < total_sec:		
			vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)	# setting which frame to get
		
			success, image = vidcap.read()

			if success:
				img_shape = image.shape
				image = cv2.resize(image, (int(img_shape[1]/self.down_f), int(img_shape[0]/self.down_f)))		 # resize by a factor of 4
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)										 # convert to grayscale
				image = image.astype("float32")/255													 # normalizing the image from 0 to 1
				if sec==0:
					final_img = image
				else:
					final_img = np.hstack((final_img, image))

		
			sec += time_sec

	
		self.input_data = cv2.resize(final_img, (self.d_shape[1], self.d_shape[0]))

		self.input_data = np.reshape(self.input_data, (1, self.d_shape[0], self.d_shape[1]))
		return self.input_data

	def predict(self):
		'''predicts the output based on input image'''
		prediction = self.model.predict(self.prep_data())

		output	 = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
						 greedy=True)[0][0])

		pred_lab = []
		x = output[0]
	
		for i in x:
			if i!=-1:
				pred_lab.append(i)
			
		# print("Predicted label: ", np.asarray(pred_lab))
		return x, np.asarray(pred_lab)
