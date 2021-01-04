'''
adopted from pytorch.org
(Classifying names with a character-level RNN-Sean Robertson)
'''
from __future__ import print_function
import os
import sys
import time
import math
import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTM
from train import train, test

# import neptune

''' argument setting '''
parser = argparse.ArgumentParser()
parser.add_argument('--logInterval', type=int, default=100, help='')

parser.add_argument('--trainPath', type=str,
                    default='./data/train.txt', help='')
parser.add_argument('--testPath', type=str,
                    default='./data/valid.txt', help='')
parser.add_argument('--validPath', type=str,
                    default='./data/valid.txt', help='')

parser.add_argument('--max_epochs', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--hidden_size1', type=int, default=200, help='')
parser.add_argument('--hidden_size2', type=int, default=400, help='')

parser.add_argument('--num_layers', type=int, default=2, help='')

# parser.add_argument('--hidden_size2', type=int, default=64, help='')
parser.add_argument('--output_size', type=int, default=785, help='')
parser.add_argument('--embedding_size', type=int, default=100, help='')


parser.add_argument('--patience', type=int, default=5, help='')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate (no default)')
parser.add_argument('--device', type=str, metavar='cpu', default='cpu')

parser.add_argument('--saveModel', type=str, default='bestmodel', help='')
parser.add_argument('--wvModel', type=str, default='./pre_trainned/mid.model', help='')
parser.add_argument('--class_embedding_path', type=str, default='./data/class_embedding.csv', help='')


args = parser.parse_args()

if __name__ == "__main__":   
    # prepare data
    batch_size = args.batch_size
    device = torch.device(args.device)

    # set up model
    print('Set model')
    model = LSTM(args).to(device)

    # define loss
    print('Set loss and Optimizer')
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = 'optim.' + args.optim
    optimizer = eval(optimizer)(model.parameters(), lr=args.lr)

    print('Train start')
    best_loss = -1.0
    bad_counter = 0

    eNum = 0

    loss= test(args, model, args.validPath, criterion)
    print("0 Epoch loss : " + str(loss))

    for ei in range(args.max_epochs):
        print('Epoch: ' + str(ei+1))
        eNum += 1

        # train
        train(args, model, args.trainPath, criterion, optimizer)

        # valid test
        loss= test(args, model, args.validPath, criterion)
        
        print('valid loss : {}'.format(loss.tolist()))

        if loss < best_loss or best_loss < 0:
            print('find best')
            best_loss = loss
            bad_counter = 0
            torch.save(model.state_dict(), args.saveModel)
        else:
            bad_counter += 1

        if bad_counter > args.patience:
            print('Early Stopping')
            break

    print('-----------------test-----------------')
    
    loss= test(args, model, args.testPath, criterion)
    print("Test loss : " + str(loss))
