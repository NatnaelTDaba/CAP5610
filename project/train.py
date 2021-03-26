import argparse
import sys

from data_loader import get_loader
import config
from models import Sanity

from utils import get_criterion, get_optimizer

from trainer import Trainer

def main(config):
    
    print("Loading data ...")
    train_loader, valid_loader = get_loader(config)
    single_batch = [next(iter(train_loader))]*50
    model = Sanity()
    model = model.to(config.DEVICE)

    criterion = get_criterion(config.LOSS)

    optimizer = get_optimizer(config.OPTIM, model)

    trainer = Trainer(model, criterion, optimizer, 
                      config=config, 
                      train_loader=single_batch, 
                      valid_loader=single_batch)

    trainer.train()
    
if __name__ == '__main__':

    main(config)

    




    



