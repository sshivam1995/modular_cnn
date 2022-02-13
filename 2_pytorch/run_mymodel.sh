#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 3 \
    --hidden-dim 10 \
    --epochs 15 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --batch-size 64 \
    --lr 1e-3 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
