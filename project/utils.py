import os
import pickle
import itertools
import datetime
import re

import numpy as np

import torch.nn as nn
import torch.optim as optim

import config

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import _LRScheduler
from criterion import LSR
#from pytorch_loss import LabelSmoothSoftmaxCEV1, LabelSmoothSoftmaxCEV2, LabelSmoothSoftmaxCEV3


def save_object(filename, obj):

	"""
		Args:
			filename (string): Name that the saved file should take
			obj (object): Object to be saved
	"""
	
	if filename is None:
		print("Please provide filename.")
	
	f = open(config.DATA_DIR+filename, 'wb')
	pickle.dump(obj, f)
	f.close()

def load_object(filename):

	"""
		Args:
			filename (string): Name file to be loaded

		Returns: loaded object

	"""
	
	f = open(config.DATA_DIR+filename, 'rb')
	loaded = pickle.load(f)
	f.close()
		
	return loaded

def get_criterion(kind):

	if kind == 'CE':
		return nn.CrossEntropyLoss()
	elif kind == 'NLL':
		return nn.NLLLoss()
	elif kind == 'MSE':
		print("Using MSE loss")
		return nn.MSELoss()
	elif kind == 'LSR': # Crossentropy with label smoothing
		print("Using LSR criterion")
		#return LSR
		return LabelSmoothSoftmaxCEV2()

def get_optimizer(kind, model):

	net_params = split_weights(model) if config.NO_BIAS_DECAY else model.parameters()

	if kind == 'SGD':
		return optim.SGD(net_params, 
					lr=config.optim_params[kind]['lr'], 
					momentum=config.optim_params[kind]['momentum'],
					nesterov=config.optim_params[kind]['nesterov'],
					weight_decay=config.optim_params[kind]['weight_decay'])

	elif kind == 'Adam':
		return optim.Adam(net_params, lr=config.optim_params[kind]['lr'], weight_decay=config.optim_params[kind]['weight_decay'])
	

def get_scheduler(optimizer, config, iter_per_epoch=None):

	if config.WARM_UP:
		print("Learning rate warmup on")
		warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.WARM_EPOCH)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.MILESTONES, gamma=config.GAMMA)
	else:
		warmup_scheduler = None
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.MILESTONES, gamma=config.GAMMA)

	return warmup_scheduler, lr_scheduler

def get_model(confg):
	"""
	Load specified model.
	"""
	if config.MODEL == 'sanity':
		from models.sanity import Sanity
		model = Sanity()
	elif config.MODEL == 'resnet18':
		from models.resnet import resnet18
		model = resnet18()
	elif config.MODEL == 'efficientnet-b4':
		model = EfficientNet.from_name('efficientnet-b4')
		model._fc = nn.Linear(in_features=1792, out_features=len(config.CLASS_NAMES))
	elif config.MODEL == 'jibanulnet':
		from models.JibanulNet import JibanulNet
		model = JibanulNet()
	elif config.MODEL == 'efficientnet-b5':
		model = EfficientNet.from_name('efficientnet-b5')
		model._fc = nn.Linear(in_features=2048, out_features=len(config.CLASS_NAMES))
	else:
		print("Invalid model name", config.MODEL)
	print(config.MODEL, "loaded")

	model.to(config.DEVICE)
	print("Model loaded to", config.DEVICE)
	return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion Matrix")
	else:
		print('Confusion matrix, without normalization')

	#print(cm)
	fig = plt.figure(figsize=[14,14])
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	return fig

def most_recent_folder(net_weights, fmt):
	"""
		return most recent created folder under net_weights
		if no none-empty folder were found, return empty folder
	"""
	# get subfolders in net_weights
	folders = os.listdir(net_weights)

	# filter out empty folders
	folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
	if len(folders) == 0:
		return ''

	# sort folders by folder created time
	folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
	return folders[-1]

def most_recent_weights(weights_folder):
	"""
		return most recent created weights file
		if folder is empty return empty string
	"""
	weight_files = os.listdir(weights_folder)
	if len(weights_folder) == 0:
		return ''

	regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

	# sort files by epoch
	weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

	return weight_files[-1]

def last_epoch(weights_folder):
	weight_file = most_recent_weights(weights_folder)
	if not weight_file:
	   raise Exception('no recent weights were found')
	resume_epoch = int(weight_file.split('-')[2])

	return resume_epoch

def save_report(targets, predictions, class_names, report_dir, epoch):

	with open(os.path.join(report_dir, 'epoch_' +str(epoch)+'_report.txt'), 'w') as f:

		print(classification_report(targets, predictions, target_names=class_names), file=f)

def init_weights2(net):
	"""the weights of conv layer and fully connected layers 
	are both initilized with Xavier algorithm, In particular,
	we set the parameters to random values uniformly drawn from [-a, a]
	where a = sqrt(6 * (din + dout)), for batch normalization 
	layers, y=1, b=0, all bias initialized to 0.
	"""
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
			
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
		
		elif isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	return net

def split_weights(net):
	"""split network weights into to categlories,
	one are weights in conv layer and linear layer,
	others are other learnable paramters(conv bias, 
	bn weights, bn bias, linear bias)
	Args:
		net: network architecture
	
	Returns:
		a dictionary of params splite into to categlories
	"""

	decay = []
	no_decay = []

	for m in net.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			decay.append(m.weight)

			if m.bias is not None:
				no_decay.append(m.bias)
		
		else: 
			if hasattr(m, 'weight'):
				no_decay.append(m.weight)
			if hasattr(m, 'bias'):
				no_decay.append(m.bias)
		
	assert len(list(net.parameters())) == len(decay) + len(no_decay)

	return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

class WarmUpLR(_LRScheduler):
	"""warmup_training learning rate scheduler
	Args:
		optimizer: optimzier(e.g. SGD)
		total_iters: totoal_iters of warmup phase
	"""
	def __init__(self, optimizer, total_iters, last_epoch=-1):

		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		"""we will use the first m batches, and set the learning
		rate to base_lr * m / total_iters
		
		Formula can be found on section 3.1 of "Bag of Tricks for Image Classification with Convolutional Neural Networks" by Tong He, et al
		"""
		#print("last_epoch var in WarmUpLR", self.last_epoch)
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)
