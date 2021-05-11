from utils import get_model
import config
import sys
import os
from data_loader import get_loader1
import numpy as np
import torch
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
torch.cuda.empty_cache()
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-ensemble', action='store_true', default=False, help='ensemble models')
parser.add_argument('-move_thr', action='store_true', default=False, help='move threshold')
parser.add_argument('-test_on_val', action='store_true', default=False, help='test model on validation set')
parser.add_argument('-test_augm', action='store_true', default=False, help='use data augmenation during test')

args = parser.parse_args()


exptA = 'resnet18/Tuesday_13_April_2021_23h_06m_53s/resnet18-17-best.pth' # Baseline
exptB = 'resnet18/Thursday_15_April_2021_18h_39m_32s/resnet18-58-best.pth' # Baseline + Balanced
exptC = 'resnet18/Saturday_17_April_2021_03h_56m_46s/resnet18-21-best.pth' # Baseline + Balanced + Finetuned
exptD = 'resnet18/Sunday_18_April_2021_13h_30m_18s/resnet18-29-best.pth' # Baseline + Balanced + Finetuned - Data augmentation:
exptE = 'efficientnet-b4/Sunday_18_April_2021_21h_59m_38s/efficientnet-b4-130-best.pth' #  Baseline + Balanced + Finetuned - Data augmentation (model: efficientnet-b4)
exptF = 'efficientnet-b4/Wednesday_21_April_2021_17h_58m_28s/efficientnet-b4-61-best.pth' # + Data augmentation (only vertical and horizontal flips) Note: check config file 
exptG = 'efficientnet-b4/Sunday_25_April_2021_00h_31m_05s/efficientnet-b4-95-best.pth' # + Data augmentation (only vertical and horizotal flips) + labelSmoothing
exptH = 'jibanulnet/Monday_26_April_2021_14h_48m_38s/jibanulnet-61-best.pth' # + Data augmentation (only vertical and horizontal flips)
exptI = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-44-best.pth'
exptJ = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-40-best.pth'
exptK = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-58-best.pth'
exptL = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-56-best.pth'
exptM = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-76-best.pth'
exptN = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-78-best.pth'
exptO = 'efficientnet-b5/Wednesday_28_April_2021_16h_46m_45s/efficientnet-b5-48-best.pth'

ckpt_path = './checkpoint/'+exptK

test_time_augmentation = args.test_augm
move_threshold = args.move_thr
test_on_val = args.test_on_val
ensemble = args.ensemble

config.MODEL = ckpt_path.split('/')[2]

if config.MODEL == 'resnet18' or config.MODEL == 'jibanulnet':
    print('ResNet-18 or JibanulNet detected...changing image size to (224,224)')
    config.NEW_SIZE = (224,224)
    
elif config.MODEL == 'efficientnet-b4':
    config.NEW_SIZE = (380,380)

if ckpt_path.split('/')[3] == 'Tuesday_13_April_2021_23h_06m_53s':
        config.transforms['test'] = transforms.Compose([
                    transforms.Resize(config.NEW_SIZE),
                    transforms.ToTensor()])
else:
    config.transforms['test'] = transforms.Compose([
                transforms.Resize(config.NEW_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(config.MEAN, config.STD)])

device = config.DEVICE


model = get_model(config)

model.load_state_dict(torch.load(ckpt_path))

config.loader_params['bs'] = 2
if test_time_augmentation:
    config.transforms['test'] = config.transforms['train']
if test_on_val:
    test_loader = get_loader1(config, test=False)[1]
else:
    test_loader = get_loader1(config, test=True)

class_priors = torch.Tensor([0.4930010242403551, 0.101058381700239, 0.2727893479003073, 0.05257767156025948, 0.0805735745988392]).reshape(1,-1).to(device)
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def test(model, loader):

    model.eval()

    valid_loss = 0.0
    correct = 0.0

    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, labels) in enumerate(loader):

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if move_threshold:
            outputs = outputs/class_priors
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        all_predictions.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        print("Batch",batch_idx+1,'/',len(loader))
    matrix = confusion_matrix(all_targets, all_predictions)
    print("Evaluating model", ckpt_path)
    print(classification_report(all_targets, all_predictions, target_names=class_names, digits=4))
    print("accuracy", accuracy_score(all_targets, all_predictions))

    return matrix

def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)

def test_ensemble(models, loaders):

    model_preds = []
    ensemble_preds = []
    count = 1
    for model, loader in zip(models, loaders):

        model.eval()

        all_predictions = []
        all_targets = []

        for batch_idx, (images, labels) in enumerate(loader):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if move_threshold:
                outputs = outputs/class_priors

            _, preds = outputs.max(1)

            all_predictions.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            print("Batch",batch_idx+1,'/',len(loader))
        model_preds.append(all_predictions)
        print("Model",count, "done!")
        count += 1
    model_preds = np.array(model_preds)

    for i in range(model_preds.shape[1]):
        sample_preds = model_preds[:,i]
        labels, counts = np.unique(sample_preds, return_counts=True)
        max_count_idx = np.argmax(counts)
        ensemble_preds.append(labels[max_count_idx])
    
    matrix = confusion_matrix(all_targets, ensemble_preds)
    print("Evaluating model", ckpt_path)
    print(classification_report(all_targets, ensemble_preds, target_names=class_names, digits=4))
    print("accuracy", accuracy_score(all_targets, ensemble_preds))

    return matrix

def get_ensemble_models(ckpt_paths):

    
    models = []
    loaders = []
    for path in ckpt_paths:
        config.MODEL = path.split('/')[0]

        if config.MODEL == 'resnet18' or config.MODEL == 'jibanulnet':
            print('ResNet-18 or JibanulNet detected...changing image size to (224,224)')
            config.NEW_SIZE = (224,224)
    
        elif config.MODEL == 'efficientnet-b4':
            config.NEW_SIZE = (380,380)

        elif config.MODEL == 'efficientnet-b5':
            config.NEW_SIZE = (456,456)


        if path.split('/')[0] == 'Tuesday_13_April_2021_23h_06m_53s':
            config.transforms['test'] = transforms.Compose([
                    transforms.Resize(config.NEW_SIZE),
                    transforms.ToTensor()])
        elif not test_time_augmentation:
            config.transforms['test'] = transforms.Compose([
                transforms.Resize(config.NEW_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(config.MEAN, config.STD)])
        
        model = get_model(config)
        model.load_state_dict(torch.load(os.path.join('./checkpoint/', path)))
        models.append(model)
        loaders.append(get_loader1(config, test=True))
        

    return models, loaders

if ensemble:
    
    ckpt_paths = [exptF, exptK, exptL]
    models, loaders = get_ensemble_models(ckpt_paths)
    conf_matrix = test_ensemble(models, loaders)
else:
    conf_matrix = test(model, test_loader)  
    
print("Quadratic weighted kappa", quadratic_weighted_kappa(conf_matrix))

