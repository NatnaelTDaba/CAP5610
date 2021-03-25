import torch
from torchvision import transforms

DATA_DIR = './data/'
CSV_PATH = DATA_DIR+'trainLabels.csv'
IMG_PATH = DATA_DIR+'train/'
TEST_IMG_PATH = DATA_DIR+'test/'

NEW_SIZE = (224, 224)

NUM_WORKERS = 1

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
                'bs': 10,
                'shuffle': {'train': True, 'test': False}
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
