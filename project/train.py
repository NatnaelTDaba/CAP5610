import argparse
import sys

from data_loader import get_loader, get_loader1
import config
from models import Sanity

from utils import get_criterion, get_optimizer, get_model, get_scheduler, init_weights2

from trainer import Trainer

def main(config):
    
    train_loader, valid_loader = get_loader1(config)
    print("Done!")    
    if config.SANITY:
        print("Sanity mode...training on single batch")
        train_loader = valid_loader = [next(iter(train_loader))]*10
    model = get_model(config)
    model = init_weights2(model)
    print("Done!")
    print(model)
    criterion = get_criterion(config.LOSS)

    optimizer = get_optimizer(config.OPTIM, model)

    warmup_scheduler, lr_scheduler = get_scheduler(optimizer, config, len(train_loader))

    trainer = Trainer(model, criterion, optimizer, 
                      config=config, 
                      train_loader=train_loader, 
                      valid_loader=valid_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_scheduler=warmup_scheduler)

    trainer.train()
    
if __name__ == '__main__':
    
    main(config)

    




    



