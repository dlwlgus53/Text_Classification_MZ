import time
import csv
import pdb

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data import TextIterator

from sklearn.metrics import recall_score, f1_score, precision_score

def train(args, model, train_path, criterion, optimizer):
    
    model = model.train()
    optimizer.zero_grad()
    
    train_iter = TextIterator(train_path, args)
    
    iloop = 0
    all_loss = 0
    current_loss = 0

    for x_data, y_data, original_len, attention_mask in train_iter:
        output_ = model(x_data)
        output = output_[[0 for _ in range(args.batch_size)],original_len-1, :]
        loss = criterion(output.squeeze(), y_data.squeeze())
        loss.backward()
        optimizer.step()
        all_loss += loss.item()
        current_loss +=loss.item()
        
        if (iloop+1) % args.logInterval == 0:
            print('%d %.4f' % (iloop+1, current_loss / args.logInterval))
            current_loss = 0

        iloop += 1
        
    return 



def test(args, model, test_path, criterion):
    loss = 0
    model = model.eval()

    with torch.no_grad():
        iloop = 0
        test_iter = TextIterator(test_path, args)

        for x_data, y_data, original_len, attention_mask in test_iter: # model in here
            output_= model(x_data)
            
            output = output_[[0 for _ in range(args.batch_size)],original_len-1, :]
            _loss = criterion(output.squeeze(), y_data.squeeze())
            loss += _loss
            iloop += 1

        loss /= iloop

    
    return loss
