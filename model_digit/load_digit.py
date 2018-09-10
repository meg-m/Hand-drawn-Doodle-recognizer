import keras.models
import numpy as np
import tensorflow as tf

from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow


def init(): 
	json_file = open('model_digit.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	
	#load weights into new model
	loaded_model.load_weights("model_digit.h5")
	print("Loaded Model from disk")

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.get_default_graph()

	return loaded_model,graph