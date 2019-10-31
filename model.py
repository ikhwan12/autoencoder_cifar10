# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:29:39 2019

@author: ikhwa
"""
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization

# define cnn model
def cnn_model(input_img=None, encode = None):
	#using encoder as feature extraction and flatten for classifier
	if encode != None :
		pool = MaxPooling2D(pool_size=(2, 2))(encode)
		conv = Dropout(0.5)(pool)
		flat = Flatten()(conv)
	else :
		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
		conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
		conv1 = MaxPooling2D((2, 2))(conv1)
		conv1 = Dropout(0.2)(conv1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
		conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
		conv2 = MaxPooling2D((2, 2))(conv2)
		conv2 = Dropout(0.3)(conv2)
		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
		conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
		conv3 = MaxPooling2D((2, 2))(conv3)
		conv3 = Dropout(0.4)(conv3)
		flat = Flatten()(conv3)
	den = Dense(128, activation='relu')(flat)
	out = Dense(10, activation='softmax')(den)
	return out

# define encoder for autoencoder
def encoder(input_img):
	
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #32 x 32 x 32
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2))(conv1) #16 x 16 x 32
	conv1 = Dropout(0.2)(conv1)
	
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1) #16 x 16 x 64
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2) #8 x 8 x 64
	conv2 = Dropout(0.3)(conv2)
	
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2) #8 x 8 x 128
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Dropout(0.4)(conv3)
	
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #8 x 8 x 256
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	return conv4

# define decoder for autoencoder
def decoder(conv4):
	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4) #8 x 8 x 256
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5) #8 x 8 x 128
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	conv6 = UpSampling2D((2,2))(conv6) 
	
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #16 x 16 x 64
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7) # 16 x 16 x 32
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	conv8 = UpSampling2D((2,2))(conv8) # 32 x 32 x 32
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv8) # 32 x 32 x 3
	return decoded