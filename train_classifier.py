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

GENDER_CLASS = {'m':0,'f':1}

def get_filepath_andlabel(s_from_datatxt):
    #we only get files from faces folder

    properties = s_from_datatxt.split()
    path_image = 'faces/'+properties[0]+'/'+'coarse_tilt_aligned_face.'+properties[2]+'.'+properties[1]
    path_landmarks = 'faces/'+properties[0]+'/'+'landmarks.'+properties[2]+'.'+properties[1]
    encode_class = ''.join([properties[3],properties[4]])
    if encode_class not in AGE_CLASS: return None,None,None,None
    age_label = AGE_CLASS[encode_class]
    
    if properties[5] not in GENDER_CLASS: return None,None,None,None
    gender_label = GENDER_CLASS[properties[5]]

    return path_image,path_landmarks,age_label,gender_label

def get_cropped(path_image, path_landmark):
    img = cv2.imread(PATH+path_image)
    with open(PATH+'faces/30601258@N03/landmarks.8.8747478879_afab76198a_o.txt') as g:
        L = g.readlines()
    getXY = [L[2:][i].split(',')[4:] for i in range(len(L[2:]))]
    getXY = np.array(getXY).astype(float)
    Xmin,Ymin = min(getXY[:,0]),min(getXY[:,1])
    Xmax,Ymax = max(getXY[:,0]),max(getXY[:,1])
    return img[int(Ymin):int(Ymax),int(Xmin):int(Xmax)]

if __name__=='__main__':
	PATH = 'data/'

	path_image = []
	path_landmarks = []
	age_labels = []
	gender_labels = []
	with open(PATH+'fold_frontal_0_data.txt') as f:
		lines = f.readlines()

	for i in range(len(lines[2:])):
		path1,path2,age_label,gender_label = get_filepath_andlabel(lines[i])
		if not path1: continue
		path_image.append(path1); age_labels.append(age_label); gender_labels.append(gender_label)
		path_landmarks.append(path2)


	path_image = np.array(path_image)
	path_landmarks = np.array(path_landmarks)
	age_labels = np.array(age_labels)
	gender_labels = np.array(gender_labels)

	labs = [0,0,0,0,0,0,0,0]
	# getting the cropped images and labels
	age_labels = list(age_labels)
	gender_labels = list(gender_labels)

	X = [get_cropped(path1,path2) for path1,path2 in tqdm(zip(path_image,path_landmarks))]
	Xall = []
	age_all = []
	gen_all = []
	for i,img in enumerate(X,start=0):
		try: 
		        if labs[age_labels[i]] < 100: 
		        	img = cv2.resize(img,(227,227),interpolation=cv2.INTER_AREA)
		        	Xall.append(img)
		        	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			        #Xall.append(gray.reshape(gray.shape[0],gray.shape[1],1))
			        age_all.append(age_labels[i]); gen_all.append(gender_labels[i])
			        labs[age_labels[i]] = labs[age_labels[i]] + 1
		except: 
			print('continue')

	Xall = np.array(Xall)
	age_all = np.array(age_all)
	gen_all = np.array(gen_all)

	# here we first deal with age
	ratio = 0.8
	indexes = random.sample(list(range(Xall.shape[0])),int(ratio*Xall.shape[0]))
	excluded = np.delete(np.array(range(Xall.shape[0])), indexes)
	X_train,y_tr = Xall[indexes],age_all[indexes]
	X_test,y_te = Xall[excluded],age_all[excluded]


	num_classes = len(AGE_CLASS) #here we do age first
	y_train, y_test = [], []

	for i in range(len(y_tr)):
		tmp = np.zeros(num_classes)
		tmp[y_tr[i]] = 1
		y_train.append(tmp)
    
	for i in range(len(y_te)):
		tmp = np.zeros(num_classes)
		tmp[y_te[i]] = 1
		y_test.append(tmp)

	y_train = np.array(y_train)
	y_test = np.array(y_test)

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

	model_check = ModelCheckpoint('age.h5', monitor='val_loss')
	batch_size = 64
	epochs = 25
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

	hist = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test)
         ,shuffle=True,callbacks=[model_check])
	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

