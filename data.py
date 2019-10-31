# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:38:40 2019

@author: ikhwa
"""
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np

def cifar10_extract(target_label=0, test_size=0.2):
	(x_train, t_train), (x_test, t_test) = cifar10.load_data()
	t_target_train = t_train==target_label
	t_target_train = t_target_train.reshape(t_target_train.size)
	x_target_train = x_train[t_target_train]
	t_target_test = t_test==target_label
	t_target_test = t_target_test.reshape(t_target_test.size)
	x_target_test = x_test[t_target_test]
	
	t_target_train = target_label*np.ones((len(x_target_train),1),dtype=np.uint8)
	t_target_test = target_label*np.ones((len(x_target_test),1),dtype=np.int32)
	x_data = np.concatenate([x_target_train, x_target_test],axis=0)
	y_data = np.concatenate([t_target_train, t_target_test],axis=0)
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
	return x_train, np.uint8(y_train), x_test, y_test

def normalize(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm