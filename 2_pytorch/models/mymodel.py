import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        channels, height, width = im_size
        c0 = 32
        p0 = 0.1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=c0, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c0, out_channels=c0*2, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p0))
        c0 *= 2
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=c0, out_channels=c0*2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c0*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c0*2, out_channels=c0*2, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p0))
        c0 *= 2
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=c0, out_channels=c0*2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c0*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c0*2, out_channels=c0*2, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=p0))

        self.conv_blocks = nn.Sequential(self.conv1, self.conv2, self.conv3)

        self.affine = nn.Sequential(
            nn.Linear(c0*2*int(height*width/(8*8)), c0*int(height*width/(8*8))),
            nn.ReLU(inplace=True),
            nn.Linear(c0 * int(height * width / (8 * 8)), int(c0 / 2) * int(height * width / (8 * 8))),
            nn.ReLU(inplace=True),
            nn.Linear(int(c0 / 2) * int(height * width / (8 * 8)), int(c0 / 2) * int(height * width / (8 * 8))),
            nn.ReLU(inplace=True),
            nn.Linear(int(c0/2)*int(height*width/(8*8)), int(c0/4)*int(height*width/(8*8))),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=p0),
            nn.Linear(int(c0 / 4) * int(height * width / (8 * 8)), int(c0 / 4) * int(height * width / (8 * 8))),
            nn.ReLU(inplace=True),
            nn.Linear(int(c0/4)*int(height*width/(8*8)), n_classes)
        )

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################
        out = self.conv_blocks(images)
        scores = self.affine(out.view(images.size(0), -1))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
