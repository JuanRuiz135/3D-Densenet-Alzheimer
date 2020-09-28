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
# Data science and other tools
from collections import OrderedDict
import math
from functools import partial
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import time
from numpy.random import seed
from os import path
import copy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from PIL import Image
from timeit import default_timer as timer
import sys
import logging

import cnn_interpretability.utils as utils

randseed = 10
seed(10)

img_dir = "NiFTiFiles/"
mask=None

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
                                    kernel_size=(conv1_t_size, 3, 3),
                                    stride=(conv1_t_stride, 2, 2),
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
        print('Model Loaded')

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


class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        cn = []
        ad = []
        emci = []
        lmci = []
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.25, random_state=19)
        X = torch.randn(self.class_vector.size(0),4).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        indices = np.hstack([train_index, test_index])

        for i in indices:
          if y[indices[i]] == 0:
            cn.append(indices[i])
          elif y[indices[i]] == 1:
            ad.append(indices[i])
          elif y[indices[i]] == 2:
            emci.append(indices[i])  
          else:
            lmci.append(indices[i])
        

        new_indices = []
        for i in range(s.get_n_splits(X, y)):
          new_indices.append(cn[i])
          new_indices.append(cn[i])
          new_indices.append(ad[i])
          new_indices.append(ad[i])
          new_indices.append(emci[i])
          new_indices.append(emci[i])
          new_indices.append(lmci[i])
          new_indices.append(lmci[i])

        return new_indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def train_net(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=5):
  since = time.time() # time training starts

  val_acc_history = [] # values of val acc
  loss_history = [] # values of val loss

  best_model_weights = copy.deepcopy(model.state_dict()) # saves best weights for the model
  best_acc = 0.0 

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train() # set model to train mode
        dataset_loader = train_dataloader # set data loader to train_loader
      else:
        model.eval() # set model to evaluate mode
        dataset_loader = val_dataloader #set data loader to val_loader

      running_loss = 0.0
      running_corrects = 0.0

      # Iterate over data
      for i, data in enumerate(dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to('cuda'), data[1].to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
          # Get model outputs and calculate loss
          outputs = model(inputs)
          labels = labels.view(-1)
          #print(labels) #print input per epoch
          loss = criterion(outputs, labels)
          _, preds = torch.max(outputs, 1)
          if phase == 'train':
            loss.backward()
            optimizer.step()
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      scheduler.step(running_loss)
      epoch_loss = running_loss / len(dataset_loader.dataset)
      epoch_acc = running_corrects.double() / len(dataset_loader.dataset)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_weights = copy.deepcopy(model.state_dict())
      if phase == 'val':
        val_acc_history.append(epoch_acc)
        loss_history.append(epoch_loss)

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:.2f}%'.format(best_acc*100))

  # load best model weights
  model.load_state_dict(best_model_weights)
  return model, val_acc_history, loss_history


if __name__ == "__main__":

    # Create model object
    model = generate_model(121, num_classes=4, drop_rate=0.5, growth_rate=28)

    # Set Directories, filenames and labels from the dataset
    filenames = set_filenames()
    labels = set_labels(filenames)

    # Set training and validation files
    x_train, x_test, y_train, y_test = train_test_split(filenames, 
    labels, train_size=0.8, 
    test_size=0.2,random_state=randseed, 
    stratify=labels)
    
    train_dataset = ADNIDataset(x_train, y_train, mask=mask)
    val_dataset = ADNIDataset(x_test, y_test, mask=mask)

    class_vector = train_dataset.labels

    sampler = StratifiedSampler(class_vector=class_vector, batch_size=4) 

    # create data loaders for the network training
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available(), sampler=sampler) 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # Set parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=40, eps=1e-6)
    model = model.to('cuda')
    

    EPOCHS = 100
    model_ft, acc_hist, loss_hist = train_net(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)

    # Save the model
    weights_path = './classifier3_weights.pth'
    torch.save(model_ft.state_dict(), weights_path)
    
   
    

