from models import resnet
from data_loader import DiabeticRetinopathyDataset
from torch.utils.data import DataLoader
import pandas as pd
import time
import os
from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

resnet18_model = resnet.resnet18()
print(resnet18_model)

import torch
import config 

resnet18_model.to('cuda:0')
resnet18_model.load_state_dict(torch.load('checkpoint/resnet18/Saturday_27_March_2021_03h_56m_16s/resnet18-6-best.pth'))

start = time.time()
resnet18_model.eval()

img_path = '/home/abhijit/nat/CAP5610/project/data/test/'
csv_path = '/home/abhijit/nat/CAP5610/project/data/retinopathy_solution.csv'
test_set = DiabeticRetinopathyDataset(img_path, csv_path, config.transforms['test'])
print("test set loaded")
print("length",len(test_set))

test_loader = DataLoader(test_set, batch_size=config.loader_params['bs'], shuffle=config.loader_params['shuffle']['test'], num_workers=8)
start = time.time()
resnet18_model.eval()

correct = 0.0
all_predictions = []
all_targets = []
total = len(test_loader)
for batch_idx, (images, labels) in enumerate(test_loader):

	images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
	outputs = resnet18_model(images)
	
	_, preds = outputs.max(1)
	correct += preds.eq(labels).sum()

	all_predictions.extend(preds.cpu().tolist())
	all_targets.extend(labels.cpu().tolist())

	print(batch_idx+1,"/", total)

finish = time.time()
print("Took", finish-start, "sec")
print("Accuracy", correct.float()/len(test_set))
matrix = confusion_matrix(all_targets, all_predictions)

fig = plot_confusion_matrix(matrix, config.CLASS_NAMES, normalize=False)
fig.savefig(os.path.join('./', 'test_confusion_matrix.png'), bbox_inches='tight')
test = pd.read_csv('./data/retinopathy_solution.csv')
my_submission = pd.DataFrame({'image': test.image, 'level': all_predictions})
my_submission.to_csv('submission.csv', index=False) 
