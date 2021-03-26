import torch
from torchvision import transforms

DATA_DIR = './data/'
CSV_PATH = DATA_DIR+'trainLabels.csv'
IMG_PATH = DATA_DIR+'train/'
TEST_IMG_PATH = DATA_DIR+'test/'

NEW_SIZE = (224, 224)

transforms = {
                'train': transforms.Compose([
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor()]),

                'val': transforms.Compose([
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor()
                    ]),

                'test': transforms.Compose([
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor()])
            }

loader_params = {
                    'bs': 128,
                    'shuffle': {'train': True, 'test': False},
                    'workers': 4
                }

OPTIM = 'SGD'

optim_params = {
                    OPTIM: {'lr':0.001, 'momentum': 0.9}
                }

 

LOSS = 'CE' # type of criterion; CE: cross entropy loss, 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = 1000

