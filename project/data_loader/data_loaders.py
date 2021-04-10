import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

import sys
import numpy as np
sys.path.append('../')

from .datasets import DiabeticRetinopathyDataset


def get_loader(config, test=False):
    """
    Args: 
        config (coinfguration object): Contains different parameters for training 

    """

    # If test loader is desired, create and return DataLoader object for test images.
    if test:
        dataset = DiabeticRetinopathyDataset(config.TEST_IMG_PATH, transforms=config.transforms['test'], test=True)
        test_loader = DataLoader(dataset, batch_size=config.loader_params['bs'], shuffle=config.loader_params['shuffle']['test'])

        return test_loader

    # Create training and validation datasets from our custom dataset class DiabeticRetinopathyDataset.
    train_dataset = DiabeticRetinopathyDataset(config.IMG_PATH, config.CSV_PATH, transforms=config.transforms['train'])
    val_dataset = DiabeticRetinopathyDataset(config.IMG_PATH, config.CSV_PATH, transforms=config.transforms['val'])

    # Create random samplers for our training and validation datasets.
    targets = train_dataset.csv['diagnosis'].values
    # Generate a unique set of training and valiadtion indices for sampling. 
    # Stratified sampling is used so that the validation set has the same class
    # distribtuion as the training set.
    train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state=42, shuffle=True, stratify=targets)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create DataLoader object for both sets 
    train_loader = DataLoader(train_dataset, batch_size=config.loader_params['bs'], sampler=train_sampler, num_workers=config.loader_params['workers'])
    val_loader = DataLoader(val_dataset, batch_size=config.loader_params['bs'], sampler=valid_sampler, num_workers=config.loader_params['workers'])

    return (train_loader, val_loader)

