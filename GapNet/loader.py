from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class HumanProteinDataset(Dataset):
    """Human Protein dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.colors = ['red','green','blue','yellow']
        self.flags = cv2.IMREAD_GRAYSCALE

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])

        image = [cv2.imread(img_name+'_'+color+'.png', self.flags).astype(np.float32)/255 for color in self.colors]
        image = torch.from_numpy(np.stack(image, axis=-1)).permute(2, 0, 1)
        classes = list(map(int, self.annotations.iloc[idx]["Target"].split()))
        label = np.zeros(28)
        label[classes] = 1
        label = torch.LongTensor(label)
        sample = {'image' : image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
