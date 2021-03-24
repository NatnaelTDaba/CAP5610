import argparse
import sys
from data_loader.data_loaders import *

parser = argparse.ArgumentParser()

parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')

args = parser.parse_args()

train_loader, val_loader = get_loader(args)


