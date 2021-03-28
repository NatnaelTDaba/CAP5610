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

def get_optimizer(kind, model):

	if kind == 'SGD':
		return optim.SGD(model.parameters(), 
						lr=config.optim_params[kind]['lr'], 
						momentum=config.optim_params[kind]['momentum'])

def get_scheduler(optimizer):

    return optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

def get_model(confg):
    """
    Load specified model.
    """
    if config.MODEL == 'sanity':
        from models.sanity import Sanity
        model = Sanity()
    if config.MODEL == 'resnet18':
        from models.resnet import resnet18
        model = resnet18()
        
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
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch