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

# available models
#   -resnet18
#   -efficientnet-b4
#   -sanity
#   -jibanulnet
#   -efficientnet-b6
MODEL = 'efficientnet-b5' 

if MODEL == 'jibanulnet' or MODEL == 'resnet18':
    NEW_SIZE = (224, 224)
elif MODEL == 'efficientnet-b4':
    NEW_SIZE = (380, 380)
elif MODEL == 'efficientnet-b5':
    NEW_SIZE = (456, 456)

# mean and std of RGB channels computed over training set
MEAN = (0.4152, 0.2218, 0.0740) 
STD = (0.2659, 0.1433, 0.0786) 

transforms = {
                'train': transforms.Compose([
                    #transforms.RandomResizedCrop(size=NEW_SIZE, scale=(1 / 1.15, 1.15), ratio=(0.8, 1.2)),
                    #transforms.RandomAffine(degrees=(-180,180), translate=(20 / NEW_SIZE[0], 20 / NEW_SIZE[0]), scale=(0.2, 0.4), shear=45),
                    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize(NEW_SIZE),
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
                    'bs': 7,
                    'shuffle': {'train': True, 'val': False, 'test': False},
                    'workers': 0
                }
# available optimizers
# SGD
# Adam
OPTIM = 'Adam'

optim_params = {
                    OPTIM: {'lr':3e-4, 'momentum': 0.9, 'nesterov':True, 'weight_decay':5e-3}
                }


# Loss types
# CE: Categorical Cross Entropy loss
# MSE: Mean Squared Error loss
# LSR: Categorical Cross Entropy loss with label smoothing applied to target distribution
LOSS = 'CE' # type of criterion; CE: cross entropy loss, 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on", DEVICE)

EPOCH = 1000

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'

TIME_NOW = datetime.now().strftime(DATE_FORMAT)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

RESUME = False

STEP_SIZE = 20

GAMMA = 0.1

SANITY = False

BALANCED = True

WEIGHT_INIT = True

NO_BIAS_DECAY = True

WARM_UP = True

WARM_EPOCH = 15

# MILESTONES = [40, 60, 80, 90]
MILESTONES = [40, 60, 80]