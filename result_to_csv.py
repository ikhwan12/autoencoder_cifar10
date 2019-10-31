# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:28:00 2019

@author: ikhwa
"""

import json
import codecs
import pandas as pd
import argparse

def loadHist(path):
	with codecs.open(path, 'r', encoding='utf-8') as file:
		n = json.loads(file.read())
		return n
	
if __name__ == "__main__":
	parse = argparse.ArgumentParser()
	parse.add_argument('--json', dest='json_path',default='models/cnn_autoencoder_hist.json',help='json file path')
	parse.add_argument('--acc', dest='acc_path',default='accuracy',help='output name file for accuracy result')
	parse.add_argument('--loss', dest='loss_path',default='loss',help='output name file for loss result')
	args = parse.parse_args()
	
	json_path = args.json_path
	dic = loadHist(json_path)
	df = pd.DataFrame.from_dict(dic)
	accuracy = df[['acc', 'val_acc']]
	loss = df[['loss', 'val_loss']]
	accuracy.to_csv('{}.csv'.format(args.acc_path))
	loss.to_csv('{}.csv'.format(args.loss_path))