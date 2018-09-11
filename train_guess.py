from __future__ import print_function

import base64
import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from keras.layers import Dense, Dropout, Flatten, Input, UpSampling2D  
from keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPooling2D
from keras.models import Sequential, Model
from keras import backend as K
from os import walk, getcwd
from random import *
from sklearn.model_selection import train_test_split


batch_size = 512
num_classes = 15
epochs = 20

# input image dimensions: 28x28 pixel images. 
img_rows, img_cols = 28, 28

#Path where all the npy quickdraw files are present
mypath = "data/"
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    print(filenames)
    txt_name_list.extend(filenames)

x_train = []
x_test = []
y_train = []
y_test = []
xtotal = []
ytotal = []
slice_train = int(120000/len(txt_name_list))  ###Setting value to be 120000 for the final dataset
print("slice train = ", )
i = 0
seed = np.random.randint(1, 10e6)


##Creates test/train split with quickdraw data
for txt_name in txt_name_list:
    txt_path = mypath + txt_name
    x = np.load(txt_path)
    x = x.astype('float32') / 255.    ##scale images
    print("length of x=", len(x) )
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
x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.1, random_state=42) 


#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert to numpy
x_train = np.array(x_train); y_train = np.array(y_train)  # convert to numpy arrays
x_test = np.array(x_test); y_test = np.array(y_test)  # convert to numpy arrays

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN Model
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(15, activation='softmax'))

cnn.compile(loss = keras.losses.categorical_crossentropy,
            optimizer = keras.optimizers.Adadelta(),
            metrics = ['accuracy'])

# Train our CNN
history = cnn.fit(x_train, y_train,
        batch_size = batch_size,
        epochs = epochs ,
        verbose = 1,
        validation_data = (x_test, y_test))

#Save the model
model_json = cnn.to_json()    # serialize model to JSON
with open("model_guess.json", "w") as json_file:
    json_file.write(model_json)                                
cnn.save('model_guess.h5')  # creates  HDF5 file
    
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

