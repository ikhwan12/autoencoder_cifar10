# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:41:34 2019

@author: ikhwa
"""
import numpy as np
import pandas as pd

def print_names_and_shapes(activations: dict):
	for layer_name, layer_activations in activations.items():
		print(layer_name)
		print(layer_activations.shape)
		print('')
	print('-' * 80)

def print_names_and_values(activations: dict):
	for layer_name, layer_activations in activations.items():
		print(layer_name)
		print(layer_activations)
		print('')
	print('-' * 80)

def shuffle(a, b):
	assert len(a) == len(b)
	shuffled_a = np.empty(a.shape, dtype=a.dtype)
	shuffled_b = np.empty(b.shape, dtype=b.dtype)
	permutation = np.random.permutation(len(a))
	for old_index, new_index in enumerate(permutation):
		shuffled_a[new_index] = a[old_index]
		shuffled_b[new_index] = b[old_index]
	return shuffled_a, shuffled_b

def save_report(report, fname='classification_report'):
	df = pd.DataFrame(report).transpose()
	df.to_csv(fname+'.csv')