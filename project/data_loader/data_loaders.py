import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

import sys
import numpy as np
sys.path.append('../')

import config
from .datasets import DiabeticRetinopathyDataset


def get_loader(args, test=False):

    if test:
        dataset = DiabeticRetinopathyDataset(config.TEST_IMG_PATH, transforms=config.transforms['test'], test=True)
        test_loader = DataLoader(dataset, batch_size=args.b, shuffle=False)

        return test_loader

    
    train_dataset = DiabeticRetinopathyDataset(config.IMG_PATH, config.CSV_PATH, transforms=config.transforms['train'])
    val_dataset = DiabeticRetinopathyDataset(config.IMG_PATH, config.CSV_PATH, transforms=config.transforms['val'])

    targets = train_dataset.csv['level'].values
    train_idx, valid_idx= train_test_split(np.arange(len(targets)), test_size=0.2, random_state=42, shuffle=True, stratify=targets)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.b, sampler=valid_sampler, num_workers=config.NUM_WORKERS)

    return (train_loader, val_loader)

