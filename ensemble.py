# Imports
import os
# Pytorch imports
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math
from functools import partial
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
import time
from numpy.random import seed
from os import path
import copy

from PIL import Image
from timeit import default_timer as timer

import sys
import logging

import cnn_interpretability.utils as utils

randseed = 10
seed(10)

img_dir = "NiFTiFiles/"
mask=None
# Assigning weight paths
model1_weights = 'model_weights/classifier1_weights.pth'
model2_weights = 'model_weights/classifier2_weights.pth'
model3_weights = 'model_weights/classifier3_weights.pth'

def modelToGpu(model):
  if torch.cuda.is_available():
      model = model.cuda()
      cuda_device = torch.cuda.current_device()
      print('Moved network to GPU')
  else:
      cuda_device = -1
      print('GPU not available')

  model = model.to('cuda')

# Function to normalise the outputs from the networks
def normaliseTensor(output):
  output[output < 0] = 0
  output -= output.min() 
  output /= output.max()
  output[torch.isnan(output)]=0
  return output

def set_filenames():
    # Set filenames
    filenames = filter(lambda filename: filename.endswith('nii'), os.listdir(img_dir))
    filenames = [os.path.join(img_dir, filename) for filename in filenames]
    return filenames

def set_labels(filenames):
    # Set labels
    labels = map(lambda filename: 3 if filename.startswith('NiFTiFiles/LM') else (2 if filename.startswith('NiFTiFiles/EM') else (1 if filename.startswith('NiFTiFiles/AD') else 0)), filenames)
    labels_arr = []
    for x in labels:
        labels_arr.append(x)
    labels = np.array(labels_arr)[:, None]
    return labels
    


# 3D DenseNet implementation taken from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/densenet.py
# The hyper-paremeters have been moved around from the original to fit this dataset

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=3,
                 conv1_t_stride=2,
                 conv1_t_stride2=2,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, conv1_t_size, conv1_t_size),
                                    stride=(conv1_t_stride, conv1_t_stride2, conv1_t_stride2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)

        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         block_config=(6, 12, 24, 16),
                         **kwargs)

    return model

"""
New Dataset class to read all images.
Implementation taken from https://github.com/jrieke/cnn-interpretability/blob/master/utils.py
It was modified to suit the specific dataset
"""
class ADNIDataset(Dataset):
    """
    PyTorch dataset that consists of MRI images and labels.
    
    Args:
        filenames (iterable of strings): The filenames fo the MRI images.
        labels (iterable): The labels for the images.
        mask (array): If not None (default), images are masked by multiplying with this array.
        transform: Any transformations to apply to the images.
    """

    def __init__(self, filenames, labels, mask=None, transform=None):
        self.filenames = filenames
        self.labels = torch.LongTensor(labels)
        self.mask = mask
        self.transform = transform

        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self, idx):
        """Return the image as a numpy array and the label."""
        label = self.labels[idx]

        struct_arr = utils.load_nifti(self.filenames[idx], mask=self.mask)
        #struct_arr = utils.resize_image(struct_arr, (55,55,55), 1)
        # TDOO: Try normalizing each image to mean 0 and std 1 here.
        #struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)
        struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)  # prevent 0 division by adding small factor
        struct_arr = struct_arr[None]  # add (empty) channel dimension
        struct_arr = torch.FloatTensor(struct_arr)

        if self.transform is not None:
            struct_arr = self.transform(struct_arr)

        return struct_arr, label

    def image_shape(self):
        """The shape of the MRI images."""
        return utils.load_nifti(self.filenames[0], mask=mask).shape

    def fit_normalization(self, num_sample=None, show_progress=False):
        """
        Calculate the voxel-wise mean and std across the dataset for normalization.
        
        Args:
            num_sample (int or None): If None (default), calculate the values across the complete dataset, 
                                      otherwise sample a number of images.
            show_progress (bool): Show a progress bar during the calculation."
        """
            
        if num_sample is None:
            num_sample = len(self)

        image_shape = self.image_shape()
        all_struct_arr = np.zeros((num_sample, image_shape[0], image_shape[1], image_shape[2]))

        sampled_filenames = np.random.choice(self.filenames, num_sample, replace=False)
        if show_progress:
            sampled_filenames = tqdm_notebook(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            struct_arr = utils.load_nifti(filename, mask=mask)
            all_struct_arr[i] = struct_arr

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx):
        """Return the raw image at index idx (i.e. not normalized, no color channel, no transform."""
        return utils.load_nifti(self.filenames[idx], mask=self.mask)


if __name__ == "__main__":

    # Create model objects
    model1 = generate_model(121, num_classes=4, drop_rate=0.5, growth_rate=32,
    conv1_t_size=3, conv1_t_stride=2, conv1_t_stride2=2)
    
    model2 = generate_model(121, num_classes=4, drop_rate=0.5, 
    growth_rate=22, conv1_t_size=7, conv1_t_stride=2, conv1_t_stride2=2)
    
    model3 = generate_model(121, num_classes=4, drop_rate=0.5, 
    growth_rate=28, conv1_t_size=7, conv1_t_stride=2, conv1_t_stride2=2)

    # Set Directories, filenames and labels from the dataset
    filenames = set_filenames()
    labels = set_labels(filenames)

    # Set training and validation files
    x_train, x_test, y_train, y_test = train_test_split(filenames, 
    labels, train_size=0.8, 
    test_size=0.2,random_state=randseed, 
    stratify=labels)
    
    # Create validation set loader
    val_dataset = ADNIDataset(x_test, y_test, mask=mask)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # Pass the model to the GPU
    modelToGpu(model1)
    modelToGpu(model2)
    modelToGpu(model3)
    # Load weights and eval function
    model1.load_state_dict(torch.load(model1_weights))
    model1.eval() # Set batch and dropout to evaluation mode
    model2.load_state_dict(torch.load(model2_weights))
    model2.eval()
    model3.load_state_dict(torch.load(model3_weights))
    model3.eval()
    print('Models loaded and evaluated')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')
            outputs_model3 = model3(images)
            outputs_model2 = model2(images)
            outputs_model1 = model1(images)
            outputs = (normaliseTensor(outputs_model3) + normaliseTensor(outputs_model2) + normaliseTensor(outputs_model1))/3
            _, predicted = torch.max(outputs.data, 1)
            total += labels[0].size(0)
            correct += (predicted == labels[0]).sum().item()

    print('Accuracy of the network on the validation images: %d%%' % (
        100 * correct / total))
    print(correct)
    print(total)
    
   
    

