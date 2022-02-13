Name: Shashwat Shivam
Email: sshivam6@gatech.edu
Best accuracy on eval ai - 84%

Architecture:
(conv-bn-relu-conv-relu-maxpool-dropout)*3 - (linear-relu)*5 - linear

Data augmentation - for training data added randomcrop and randomhorizontalflip for increasing accuracy. 

Optimizer - used Adam

Filter size - used 3x3 filters for convolution

Wanted to schedule lr as a function of epoch (adaptive lr; large at start, small after 20 epoch) but couldn't figure it out
