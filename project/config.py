import torch
from torchvision import transforms

from datetime import datetime



WORK_DIR = '/home/abhijit/nat/CAP5610/project/'
DATA_DIR = WORK_DIR+'data/'
PLOTS_DIR = WORK_DIR+'plots/'
CHECKPOINT_PATH = WORK_DIR+'checkpoint/'
CSV_PATH = DATA_DIR+'train.csv'
TRAIN_IMG_PATH = DATA_DIR+'train/'
VAL_IMG_PATH = DATA_DIR+'val/'
TEST_IMG_PATH = DATA_DIR+'test/'
LOG_DIR = WORK_DIR+'runs'
REPORTS_DIR = WORK_DIR+'reports/'


NEW_SIZE = (224, 224)

# mean and std of RGB channels computed over training set
MEAN = (0.4152, 0.2218, 0.0740) 
STD = (0.2659, 0.1433, 0.0786) 

transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(size=NEW_SIZE, scale=(1 / 1.15, 1.15), ratio=(0.7561, 1.3225)),
                    transforms.RandomAffine(degrees=(-180,180), translate=(40 / 224, 40 / 224), scale=None, shear=None),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)]),

                'val': transforms.Compose([
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)]),

                'test': transforms.Compose([
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)])
            }

loader_params = {
                    'bs': 128,
                    'shuffle': {'train': True, 'val': False, 'test': False},
                    'workers': 0
                }

OPTIM = 'SGD'

optim_params = {
                    OPTIM: {'lr':1e-3, 'momentum': 0.9, 'nesterov':True, 'weight_decay':5e-4}
                }

MODEL = 'resnet18' 

LOSS = 'CE' # type of criterion; CE: cross entropy loss, 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = 1000

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'

TIME_NOW = datetime.now().strftime(DATE_FORMAT)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe","Proliferative DR"]

RESUME = False

STEP_SIZE = 20

GAMMA = 0.1

SANITY = False

BALANCED = True

WEIGHT_INIT = False