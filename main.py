# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:47:41 2019

@author: ikhwa
"""

import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import load_model
#from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import os
from model import encoder, decoder, cnn_model
from data import cifar10_extract, normalize
from utils import shuffle, save_report
from visualize import summarize_diagnostics, show_intermediate_activation
from sklearn.metrics import classification_report
import argparse
import json

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck']

def load_data(categorical=True, norm=True):
	flag = 0
	for label in labels:
		target_label = labels.index(label)
		if flag == 0 :
			if label in ['bird', 'deer', 'truck']:
				trainX, trainy, testX, testy = cifar10_extract(target_label, test_size=0.5)
			else :
				trainX, trainy, testX, testy = cifar10_extract(target_label, test_size=0.2)
			flag+=1
		else :
			if label in ['bird', 'deer', 'truck']:
				temp0, temp1, temp2, temp3 = cifar10_extract(target_label, test_size=0.5)
			else :
				temp0, temp1, temp2, temp3 = cifar10_extract(target_label, test_size=0.2)
			trainX = np.concatenate([trainX,temp0],axis=0)
			trainy = np.concatenate([trainy,temp1],axis=0)
			testX = np.concatenate([testX,temp2],axis=0)
			testy = np.concatenate([testy,temp3],axis=0)
	trainX, trainy = shuffle(trainX, trainy)
	testX, testy = shuffle(testX, testy)
	if categorical == True :
		# one hot encode target values
		trainy = to_categorical(trainy)
		testy = to_categorical(testy)
	if norm == True :
		#Normalization
		trainX, testX = normalize(trainX, testX)
	return trainX, trainy, testX, testy

def test(lb, ub, model_path='cnn_autoencoder.h5'):
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	# mean-std normalization
	x_train, x_test = normalize(x_train, x_test)
	model = load_model(model_path)
	model.summary()
	indices = np.argmax(model.predict(x_test[lb:ub]),1)
	print('Predicted : '+str([labels[x] for x in indices]))
	print('Ground Truth : '+str([labels[x] for x in y_test[lb:ub,0]]))
	#show_imgs(x_test[lb:ub])

def evaluate(model_path='cnn_autoencoder.h5', num_classes=10):
	trainX, trainy, testX, testy = load_data()
	model = load_model(model_path)
	test_eval = model.evaluate(testX, testy, verbose=1)
	print('Test loss:', test_eval[0])
	print('Test accuracy:', test_eval[1])

	predicted_classes = model.predict(testX)
	predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
	class_labels = np.argmax(np.round(testy), axis=1)
	#print('test class : ',predicted_classes[0])
	predicted_classes = np.reshape(predicted_classes,(len(testy),1))
	class_labels = np.reshape(class_labels, (len(testy),1))

	print(predicted_classes.shape, ' ----- ', class_labels.shape)

	correct = np.where(predicted_classes==class_labels)[0]

	print("Found %d correct labels" % len(correct))
	for i, correct in enumerate(correct[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(testX[correct].reshape(32,32,3), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predicted_classes[correct], testy[correct]))
		plt.tight_layout()

	incorrect = np.where(predicted_classes!=class_labels)[0]
	print("Found %d incorrect labels" % len(incorrect))
	for i, incorrect in enumerate(incorrect[:9]):
		plt.subplot(3,3,i+1)
		plt.imshow(testX[incorrect].reshape(32,32,3), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], testy[incorrect]))
		plt.tight_layout()

	target_names = ["Class {}".format(i) for i in range(num_classes)]
	report = classification_report(class_labels, predicted_classes, target_names=target_names, output_dict=True)
	save_report(report)
	print(report)

class Classification(object):
	def __init__(self, trainX, trainy, testX, testy, batch_size=64, epochs=100, size=(32,32), chan=3,lr=0.001):
		self.trainX = trainX
		self.trainy = trainy
		self.testX = testX
		self.testy = testy
		self.batch_size = batch_size
		self.epochs = epochs
		self.inChannel = chan
		self.lr = lr
		self.x, self.y = size
		self.input_img = Input(shape = (self.x, self.y, self.inChannel))

	def feature_extraction(self, model_path = None) :
		autoencoder = Model(self.input_img, decoder(encoder(self.input_img)))
		autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop(lr=self.lr))
		autoencoder.summary()
		if model_path == None :
			save_dir = 'checkpoint/'
			if not os.path.isdir(save_dir):
				os.makedirs(save_dir)
			#es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
			#chkpt = save_dir	 + 'autoencoder_weights.{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'
			#cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
			# create data generator
			datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
			# prepare iterator
			it_train = datagen.flow(self.trainX, self.trainX, batch_size=64)
			# fit model
			steps = int(self.trainX.shape[0] / 64)
			autoencoder_train = autoencoder.fit_generator(it_train,
												steps_per_epoch=steps,
												epochs=self.epochs,
												verbose=1,
												validation_data=(self.testX, self.testX),
												shuffle=True)
			autoencoder.save('autoencoder.h5')
			with open('autoencoder_hist.json','w') as f:
				json.dump(autoencoder_train.history,f)
		else :
			autoencoder_train = load_model(model_path)
		return autoencoder, autoencoder_train

	def classifier(self, autoencoder=None):
		encode = encoder(self.input_img)
		model = Model(self.input_img,cnn_model(encode=encode)) if autoencoder != None else Model(self.input_img,cnn_model(self.input_img))
		#model.initializer.
		if autoencoder == None :
			print('Train CNN Only')
			model.compile(loss=keras.losses.categorical_crossentropy,
						  optimizer=keras.optimizers.Adam(lr=self.lr),
						  metrics=['accuracy'])
			model.summary()
			# create data generator
			datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
			# prepare iterator
			it_train = datagen.flow(self.trainX, self.trainy, batch_size=64)
			# fit model
			steps = int(self.trainX.shape[0] / 64)
			model_train = model.fit_generator(it_train,
									   steps_per_epoch=steps,
									   epochs=100,
									   verbose=1,
									   validation_data=(self.testX, self.testy))
			with open('cnn_hist.json','w') as f:
				json.dump(model_train.history,f)
			model.save('cnn.h5')
			return model_train
		else :
			for l1,l2 in zip(model.layers[0:19],autoencoder.layers[0:19]):
				l1.set_weights(l2.get_weights())
			for layer in model.layers[0:19]:
				layer.trainable = True
			model.compile(loss=keras.losses.categorical_crossentropy,
						  optimizer=keras.optimizers.Adam(lr=self.lr),
						  metrics=['accuracy'])
			model.summary()
			# create data generator
			datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
			# prepare iterator
			it_train = datagen.flow(self.trainX, self.trainy, batch_size=64)
			# fit model
			steps = int(self.trainX.shape[0] / 64)
			model_train = model.fit_generator(it_train,
									   steps_per_epoch=steps,
									   epochs=100,
									   verbose=1,
									   validation_data=(self.testX, self.testy))
			with open('cnn_autoencoder_hist.json','w') as f:
				json.dump(model_train.history,f)
			model.save('cnn_autoencoder.h5')
			return model_train

if __name__ == "__main__":
	parse = argparse.ArgumentParser()
	parse.add_argument('--train', dest='train',action='store_true',help='Train mode')
	parse.add_argument('--cnn_only', dest='cnn_only',action='store_true',help='Train mode')
	parse.add_argument('--test', dest='test',action='store_true',help='Test mode')
	parse.add_argument('--evaluate', dest='eval',action='store_true',help='Show evaluation result')
	parse.add_argument('--imshow', dest='imshow',action='store_true',help='show the intermediate activation layer')
	parse.add_argument('--clf', dest='clf',action='store_true',help='set true for classification problem')
	parse.add_argument('-p',dest='model_path',help='autoencoder model path for training or full model path for testing')
	parse.add_argument('-l',dest='lb',default=0,type=int,help='lower index for cifar-10 dataset testing')
	parse.add_argument('-u',dest='ub',default=16,type=int,help='upper index for cifar-10 dataset testing')
	parse.add_argument('-n',dest='fname',default='intermediate_activation',help='directory name to save intermediate activateion images')

	args = parse.parse_args()

	if args.train :
		print('-'*30)
		print('Train Mode')
		print('-'*30)
		trainX, trainy, testX, testy = load_data()
		#train(trainX, trainy, testX, testy)

		run = Classification(trainX=trainX,
						 trainy=trainy,
						 testX=testX,
						 testy=testy)
		if args.cnn_only:
			hist = run.classifier()
			summarize_diagnostics(hist, model='cnn')
		else :
			if args.model_path :
				encode = run.feature_extraction(model_path=args.model_path)
			else :
				encoder_w, hst = run.feature_extraction()
				summarize_diagnostics(hst, model='autoencoder')
			hist = run.classifier(autoencoder=encoder_w)
			summarize_diagnostics(hist, model='cnn_autoencoder')

	elif args.test :
		print('-'*30)
		print('Test Mode')
		print('-'*30)
		if args.model_path :
			test(lb=args.lb, ub=args.ub, model_path=args.model_path)
		else :
			test(lb=args.lb, ub=args.ub)
	elif args.eval :
		print('-'*30)
		print('Evaluation Result')
		print('-'*30)
		if args.model_path :
			evaluate(model_path=args.model_path)
		else :
			evaluate()
	elif args.imshow:
		print('-'*50)
		print('Generate the intermediate activation layer')
		print('-'*50)
		if args.model_path == '':
			print('Please enter the model path')
		else :
			trainX, trainy, testX, testy = load_data()
			model = load_model(args.model_path)
			if args.clf:
				show_intermediate_activation(testX[0:1], args.model_path, args.fname, classification=True)
			else :
				show_intermediate_activation(testX[0:1], args.model_path, args.fname)
	else :
		print('Error : Please enter the correct argument.')
