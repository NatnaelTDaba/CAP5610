import torch
from torchvision import transforms

from datetime import datetime


WORK_DIR = '/home/abhijit/nat/CAP5610/project/'
DATA_DIR = WORK_DIR+'data/'
PLOTS_DIR = WORK_DIR+'runs/plots/'
CHECKPOINT_PATH = WORK_DIR+'checkpoint/'
CSV_PATH = DATA_DIR+'trainLabels.csv'
IMG_PATH = DATA_DIR+'train/'
TEST_IMG_PATH = DATA_DIR+'test/'
LOG_DIR = WORK_DIR+'runs'

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

MODEL = 'sanity' 

LOSS = 'CE' # type of criterion; CE: cross entropy loss, 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = 1000

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'

TIME_NOW = datetime.now().strftime(DATE_FORMAT)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe","Proliferative DR"]

RESUME = False

STEP_SIZE = 20

GAMMA = 0.1