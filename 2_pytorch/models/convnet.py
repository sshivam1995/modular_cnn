import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.channels, self.height, self.width = im_size
        self.filter_count = hidden_dim
        self.kernel_size = kernel_size
        pad = (self.kernel_size-1)//2
        stride = 1
        self.conv1 = nn.Conv2d(self.channels, self.filter_count, kernel_size=self.kernel_size, stride=stride, padding=pad)
        self.relu1 = nn.ReLU(nn.Linear(self.channels * self.height * self.width, self.filter_count))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(self.filter_count * (self.height // 2) * (self.width // 2), n_classes)
        # self.network = nn.Sequential(self.conv1, self.relu1, self.max_pool, self.fc1)
        self.network = nn.Sequential(self.conv1, self.relu1, self.max_pool)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        # images_vectorized = images.view(images.size(0), -1)
        out = self.network(images)
        scores = self.fc1(out.view(images.size(0), -1))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

