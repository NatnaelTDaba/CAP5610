import argparse
import sys

from data_loader import get_loader
import config
from models import Sanity

from utils import get_criterion, get_optimizer, get_model, get_scheduler

from trainer import Trainer

def main(config):
    
    print("Loading data ...")
    train_loader, valid_loader = get_loader(config)
    print("Done!")
    print("Generating multiples of single batch")
    single_batch = [next(iter(train_loader))]*2
    print("Done!")
    print("Loading model")
    model = get_model(config)
    print("Done!")
    criterion = get_criterion(config.LOSS)

    optimizer = get_optimizer(config.OPTIM, model)

    lr_scheduler = get_scheduler(optimizer)

    trainer = Trainer(model, criterion, optimizer, 
                      config=config, 
                      train_loader=single_batch, 
                      valid_loader=single_batch,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    
if __name__ == '__main__':
    
    main(config)

    




    



