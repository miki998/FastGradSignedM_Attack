import os

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

AGE_CLASS = {'(0,2)':0,'(4,6)':1 
             ,'(8,13)':2,'(15,20)':3
             ,'(25,32)':4,'(38,43)':5
             ,'(48,53)':6,'(60,100)':7}

#load images
X_train, y_train = [], []
X_test, y_test = [], []

for image in os.listdir('train'):
    y_train.append(image.split('_')[1].split('.')[0])
    X_train.append(cv2.imread('train/'+image))

for image in os.listdir('test'):
    y_test.append(image.split('_')[1].split('.')[0])
    X_test.append(cv2.imread('test/'+image))
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

num_classes = len(AGE_CLASS)
input_shape = (227,227,3)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))
#loading previously trained weights
model.load_weights('age.h5')

def get_conf(prediction):
    #return confidence of underage or adult and return underage:0 or adult:1
    und_conf,ad_conf = np.sum(prediction[0,:3])+prediction[0,3]/2,prediction[0,3]/2+np.sum(prediction[0,4:])
    return max(und_conf,ad_conf),np.argmax([und_conf,ad_conf])

if __name__=='__main__':
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	loss_object = tf.keras.losses.CategoricalCrossentropy()

	#generate adversarial of first image
	nb_image = 0
	img = X_test[nb_image]
	label = np.zeros(len(AGE_CLASS)) ; label[int(y_test[nb_image])] = 1.
	img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
	img = img.astype(np.float32)
	tens = tf.convert_to_tensor(img)
	prediction = model(tens)
	loss = loss_object(label,prediction)

	gradient = tf.gradients(loss,tens)
	signed_grad = tf.sign(gradient)
	array = signed_grad.eval(session=sess)
	eps = 9

	adv_x = img + eps*array.reshape(1,array.shape[2],array.shape[3],array.shape[4])
	prediction = model.predict(adv_x)
	conf,adulthod = get_conf(prediction)
	cv2.imwrite('adversarial_img/adversarial1.jpg',adv_x.astype(int).reshape(227,227,3))
	print('first image generated')

	#generate adversarial of second image
	number = 26
	img = X_test[number]
	prediction = model.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))
	label = np.zeros(len(AGE_CLASS)) ; label[int(y_test[number])] = 1.
	conf,adulthod = get_conf(prediction)

	img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
	img = img.astype(np.float32)
	tens = tf.convert_to_tensor(img)
	prediction = model(tens)
	loss = loss_object(label,prediction)

	gradient = tf.gradients(loss,tens)
	signed_grad = tf.sign(gradient)
	array = signed_grad.eval(session=sess)

	eps = 10

	adv_x = img + eps*array.reshape(1,array.shape[2],array.shape[3],array.shape[4])
	prediction = model.predict(adv_x)
	conf,adulthod = get_conf(prediction)
	cv2.imwrite('adversarial_img/adversarial2.jpg',adv_x.astype(int).reshape(227,227,3))
	print('second image generated')

