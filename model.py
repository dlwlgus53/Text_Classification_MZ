'''
dopted from pytorch.org
(Classifying names with a character-level RNN-Sean Robertson)
'''
import pdb
import torch
import torch.nn as nn

'''
    input_size = args.embedding_size
    hidden_size = args.hidden_size
    output_size = args.output_size
'''

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.embedding_size = args.embedding_size
        self.batch_size = args.batch_size
        self.hidden_size1 = args.hidden_size1
        self.hidden_size2 = args.hidden_size2
        self.device = args.device
        self.output_size = args.output_size
        self.hidden = self.initHidden()
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size1, self.hidden_size2) # linear 1
        self.fc2 = nn.Linear(self.hidden_size2, self.output_size) # linear 3
        self.relu = nn.ReLU()

    def forward(self, input):
        # B: batch, W: window_len
        output, __ = self.rnn(input, self.hidden) #B, W, H
        output = self.relu(self.fc1(output)) # B, W, M1
        output = self.fc2(output) # B, W, 2 (output)
        return output

    def initHidden(self):
        hidden = torch.zeros(2, 1, self.batch_size, self.hidden_size1, 
            requires_grad=True).to(self.device)

        return (hidden[0], hidden[1])