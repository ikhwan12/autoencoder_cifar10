# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:42:37 2019

@author: ikhwa
"""
import sys
from PIL import Image
from matplotlib import pyplot as plt
import os
from keras.models import load_model
from keras import models
import numpy as np

def summarize_diagnostics(history, model='autoencoder'):
	# plot diagnostic learning curves
	img_dir = 'imgs/'
	if not os.path.isdir(img_dir):
		os.makedirs(img_dir)
	if model == 'autoencoder' :
		# plot loss
		plt.subplot(211)
		plt.title('Mean Squared Error Loss')
		plt.plot(history.history['loss'], color='blue', label='train')
		plt.plot(history.history['val_loss'], color='orange', label='test')
		# save plot to file
		filename = sys.argv[0].split('/')[-1]
		plt.savefig(img_dir + filename + '_plot_'+ model+'.png')
		plt.close()
	else :
		# plot loss
		plt.subplot(211)
		plt.title('Cross Entropy Loss')
		plt.plot(history.history['loss'], color='blue', label='train')
		plt.plot(history.history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(212)
		plt.title('Classification Accuracy')
		plt.plot(history.history['acc'], color='blue', label='train')
		plt.plot(history.history['val_acc'], color='orange', label='test')
		# save plot to file
		filename = sys.argv[0].split('/')[-1]
		plt.savefig(img_dir + filename + '_plot_'+ model+'.png')
		plt.close()

def show_imgs(X):
	plt.figure(1)
	k = 0
	for i in range(0,4):
		for j in range(0,4):
			plt.subplot2grid((4,4),(i,j))
			plt.imshow(Image.fromarray(X[k]))
			k = k+1
	plt.show()
	
def show_intermediate_activation(x, model_path, save_dir = 'intermediate_layer', layer_name=None, classification=False):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	model = load_model(model_path)
	if classification:
		layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None][1:24]
	else :
		out = model.predict(x)
		out = np.reshape(out, (32,32,3))
		layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None][1:]
		plt.imsave('{}/output'.format(save_dir),out)
	# Extracts the outputs of the top 19 layers
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
	activations = activation_model.predict(x) 
	for i in range(len(activations)):
		layer_activation = activations[i]
		plt.imsave('{}/layer_{}'.format(save_dir,i),layer_activation[0, :, :, 2])
	x = np.reshape(x, (32,32,3))
	plt.imsave('{}/input'.format(save_dir),x)