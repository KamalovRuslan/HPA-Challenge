from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
        self.kernel = np.ones((5,5),np.uint8)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.annotations.iloc[idx, 0])

        channels = {color : cv2.imread(img_name+'_'+color+'.png', self.flags).astype(np.float32)/255 for color in self.colors}
        classes = list(map(int, self.annotations.iloc[idx]["Target"].split()))
        label = np.zeros(28)
        label[classes] = 1
        label = torch.ByteTensor(label)

        blur = cv2.GaussianBlur(channels['green'],(5,5),0)
        _, th = cv2.threshold(channels['green'] * 255.,127,255,cv2.THRESH_BINARY)
        dilation = cv2.dilate(th, self.kernel, iterations=1)
        target_map = cv2.resize(dilation, (32, 32))
        target_map /= 255.
        sample = {'red'    : torch.from_numpy(channels['red']).unsqueeze(0),
                  'green'  : torch.from_numpy(channels['green']).unsqueeze(0),
                  'blue'   : torch.from_numpy(channels['blue']).unsqueeze(0),
                  'yellow' : torch.from_numpy(channels['yellow']).unsqueeze(0),
                  'target_map' : torch.from_numpy(target_map).unsqueeze(0).expand(28, 32, 32),
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
