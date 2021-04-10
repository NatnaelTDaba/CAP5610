import os 

import torch

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

import pandas as pd

class DiabeticRetinopathyDataset(Dataset):
    """Diabetic Retinopathy Dataset."""

    def __init__(self, img_path, csv_path=None, transforms=None, test=False):
        """
        Args:
            csv_path (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # This loads the image_name-label pair for training or just the image_name for test data
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
        self.root_dir = img_path
        self.test = test

    def __getitem__(self, idx):
        """
        Args:
            idx: Index to fetch the ith image and it's corresponding label.

        Returns: Image and it's corresponding label.
        """

        img_path = os.path.join(self.root_dir, self.csv.iloc[idx, 0]+'.png')
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        if self.test:
            return image

        label = self.csv.iloc[idx, 1]

        return image, label

    def __len__(self):
        """
        Args:

        Returns: Size of the dataset.
        """

        return len(self.csv)

    