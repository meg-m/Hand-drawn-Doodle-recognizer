import argparse
import base64
import keras.models	#for importing our keras model
import matplotlib.pyplot as plt
import numpy as np	#for matrix math
import os	#for reading operating system data
import pickle
import re	#for regular expressions, saves time dealing with string data
import sys 	#system level operations (like loading files)

from flask import Flask, render_template,request, jsonify
from keras.models import model_from_yaml
from numpy import array
from os import walk, getcwd
from PIL import Image
from random import *
from scipy.misc import imsave, imread, imresize	#scientific computing library for saving/reading & resizing images
from sklearn.model_selection import train_test_split

#initalize our flask app
app = Flask(__name__)

#global vars for easy reusability
global model_image, model_digit, model_guess, model_alphabet
global graph_image, graph_digit, graph_guess, graph_alphabet

#tell our app where our saved model is

sys.path.append(os.path.abspath("./model_image"))
from load_image import * 
#initialize these variables
model_image, graph_image = init()

sys.path.append(os.path.abspath("./model_guess"))
from load_guess import * 
#initialize these variables
model_guess, graph_guess = init()

sys.path.append(os.path.abspath("./model_digit"))
from load_digit import * 
#initialize these variables
model_digit, graph_digit = init()

#used for alphabet tab
sys.path.append(os.path.abspath("./model_alphabet"))
model_graph, graph_alphabet = init()


#path to data folder where all the npy files are present
# this is for guess image tab
mypath = "data/"
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
	print(filenames)
	if filenames != '.DS_Store':       ## mac junk
		txt_name_list.extend(filenames)
		break

x_train = []
x_test = []
y_train = []
y_test = []
xtotal = []
ytotal = []
slice_train = int(120000/len(txt_name_list))  ###Setting value to be 120000 for the final dataset
i = 0
seed = np.random.randint(1, 10e6)

for txt_name in txt_name_list:
	txt_path = mypath + txt_name
	x = np.load(txt_path)
	x = x.astype('float32') / 255.    ##scale images
	y = [i] * len(x) 

	np.random.seed(seed)
	np.random.shuffle(x)
	np.random.seed(seed)
	np.random.shuffle(y)
	x = x[:slice_train]
	y = y[:slice_train]
	if i != 0: 
		xtotal = np.concatenate((x,xtotal), axis=0)
		ytotal = np.concatenate((y,ytotal), axis=0)
	else:
		xtotal = x
		ytotal = y
	i += 1
x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, random_state=42) 


def load_model(bin_dir):
    ''' Load model from .yaml and the weights from .h5
        Arguments: bin_dir: The directory of the bin (normally bin/)
        Returns: Loaded model from file
    '''

    # load YAML and create model
    yaml_file = open('%s/model_alphabet.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model_alphabet.h5' % bin_dir)
    return model

## TODO- Merge below 3 methods to 1
#decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output_image.png','wb') as output:
        output.write(base64.b64decode(imgstr))

def convertDigitImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output_digit.png','wb') as output:
        output.write(base64.b64decode(imgstr))

def convertAlphaImage(imgData):
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output_alphabet.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

@app.route('/home/')
def home():
	return render_template("index.html")

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/digit/',methods=['GET','POST'])
def digit():
	print("inside digit route")
	return render_template("index_digit.html")

@app.route('/alpha/',methods=['GET','POST'])
def alpha():
	return render_template("index_alphabet.html")

@app.route('/image/',methods=['GET','POST'])
def image():
	return render_template("index_image.html")

@app.route('/guess/',methods=['GET','POST'])
def guess():
	num = randint(1, 1000)    # Pick a random number between 1 and 1001.
	##Visualize a quickdraw file
	print("y_train=",y_train[num])
	imgstr = x_train[num].reshape(28,28)
	imgstr = 1 - imgstr
	img = Image.fromarray(imgstr*255)
	bw_img = img.convert('RGB')
	bw_img.save("static/imageToGuess.png") #or- absolute path to static folder
	
	#render out pre-built HTML file right on the index page
	return  render_template("index_guess.html", num=num)


@app.route('/guessResult/',methods=['GET','POST'])
def guessResult():

	inputText = request.args.get('a')
	num = request.args.get('b')
	arr = ['apple','bat','cloud','crown','face','flower', 'hand', 'house','icecream','moon','star','sun','tree','tshirt','umbrella']
	
	word = arr[y_train[int(num)]]
	print("word = ",word)	
	if(inputText == word):
		return render_template("guess-result.html", correct= 1, word= word)
	else:
		return render_template("guess-result.html", correct= 0, word= word)


@app.route('/show/',methods=['GET','POST'])
def show():
	return str(y_train[num])


@app.route('/predict-image/',methods=['GET','POST'])
def predictImage():

	imgData = request.get_data()
	convertImage(imgData)
	
	x = imread('output_image.png',mode='L')	#read the image into memory
	x = np.invert(x)	#compute a bit-wise inversion so black becomes white and vice versa
	x = imresize(x,(28,28))	#make it the right size
	x = x.reshape(1,28,28,1)	#convert to a 4D tensor to feed into our model
	x = x.astype('float32')
	
	#perform the prediction
	with graph_image.as_default():
		out = model_image.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

@app.route('/predict-alpha/', methods=['GET','POST'])
def predictAlpha():
    convertAlphaImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output_alphabet.png', mode='L')
    x = np.invert(x)	#compute a bit-wise inversion so black becomes white and vice versa
    imsave('resized.png', x)	# Visualize new array
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)	# reshape image data for use in neural network
    x = x.astype('float32')	# Convert type to float32
    x /= 255	# Normalize to prevent issues with model

    # Predict from model
    with graph_alphabet.as_default():
        out = model_alphabet.predict(x)	
        print (out)
        print(np.argmax(out,axis=1))

    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))+1]),
                'confidence': str(max(out[0]) * 100)[:6]}
    return jsonify(response)



@app.route('/predict-digit/',methods=['GET','POST'])
def predictDigit():
	imgData = request.get_data()
	convertDigitImage(imgData)	#encode it into a suitable format
	
	x = imread('output_digit.png',mode='L')	#read the image into memory
	x = np.invert(x)	#compute a bit-wise inversion so black becomes white and vice versa
	x = imresize(x,(28,28))	#make it the right size
	x = x.reshape(1,28,28,1)	#convert to a 4D tensor to feed into our model
	
	with graph_digit.as_default():
		#perform the prediction
		out = model_digit.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print ("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	


@app.route('/predict-guess/',methods=['GET','POST'])
def predictGuess():
	imgData = request.get_data()
	convertImage(imgData)		#encode it into a suitable format	
	
	x = imread('output_guess.png',mode='L')	#read the image into memory	
	x = np.invert(x)	#compute a bit-wise inversion so black becomes white and vice versa	
	x = imresize(x,(28,28))	#make it the right size	
	x = x.reshape(1,28,28,1)	#convert to a 4D tensor to feed into our model
		
	with graph_guess.as_default():
		#perform the prediction
		out = model_guess.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	


if __name__ == "__main__":
	
	model_alphabet = load_model('bin')
	mapping = pickle.load(open('bin/mapping.p', 'rb'))

	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	
