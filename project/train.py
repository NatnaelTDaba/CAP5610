import argparse
import sys

from data_loader import get_loader
import config
from models import Baseline

from utils import *

#from trainer import Trainer

def main(config):
    
    train_loader, val_loader = get_loader(config)
    print("Data loaded")

    model = Baseline()
    print("Network loaded")
    print(model)

    model = model.to(config.DEVICE)

    criterion = get_criterion(config.LOSS)
    optimizer = get_optimizer(config.OPTIM, model)
    print("criterion and optimizer defined")


if __name__ == '__main__':

    main(config)

    




    



