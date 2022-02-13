#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 7 \
    --hidden-dim 20 \
    --epochs 15 \
    --weight-decay 0.1 \
    --momentum 0.9 \
    --batch-size 64 \
    --lr 1e-3 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
