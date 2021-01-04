  
import numpy as np
import torch
import pandas as pd
import gensim
import warnings
import pdb
from konlpy.tag import  Mecab
# from gensim.models.wrappers import FastText
# from gensim.models import FastText 
from gensim.models import Word2Vec

from tqdm import tqdm
import time
from fasttext import util

warnings.simplefilter("ignore", UserWarning)


class TextIterator:
    def __init__(self, filename, args):
        self.start_index = 0
        self.touch_end = False
        self.batch_size = args.batch_size
        self.device=args.device
        path = filename
        label_embedding_path = args.class_embedding_path
        print("load model")
        wvmodel = Word2Vec.load(args.wvModel)
       
        
        data = pd.read_csv(path, header = None, sep = '\t')
        label_embedding = pd.read_csv(label_embedding_path).set_index('key')
        
        labels, sentences = data[0], data[1]
        
        self.labels = np.array(label_embedding.loc[labels]['value'].tolist())
        tokenized_texts = self.tokenize(sentences)
        
        
        # padding in model
        # padded_texts = self.padding(sentences)
                
        # embedding
        print("Embedding")
        self.embedded_texts = []
        for sent in tqdm(tokenized_texts):
            embedded_sent = []
            for word in sent:
                if word in  wvmodel.wv.vocab:
                    embedded_sent.append((wvmodel.wv[word]))
                    
                else:
                    embedded_sent.append((np.zeros(100)) )

            self.embedded_texts.append(embedded_sent)

        sentences = sentences.fillna('0')
        self.original_len = [len(el) for el in sentences]
            
        
    ##################################INIT###################################
    def tokenize(self,sentences): 
        tokenizer =  Mecab().morphs
        print("tokenize start")
        tokenized_texts = []
        for sent in tqdm(sentences):
            try:
                tokenized_texts.append(tokenizer(sent))
            except:
                tokenized_texts.append('')
        print("tokenize end")
        return tokenized_texts

        

    def __iter__(self):
        return self

    def reset(self):
        for fp in self.fps:
            fp.seek(0)

    def __next__(self): # passing x_data, y_data, index_data, rindex_data

        if self.touch_end:
            self.start_index = 0
            self.touch_end = False
            raise StopIteration


        if self.start_index + self.batch_size > len(self.labels):
            self.start_index = len(self.labels)- self.batch_size
            self.touch_end = True

        original_len = self.original_len[self.start_index:self.start_index + self.batch_size]
        x_data = self.embedded_texts[self.start_index:self.start_index + self.batch_size]
        
        x_data = \
            torch.tensor(self.make_padding(x_data,original_len))\
            .type(torch.float32).to(self.device)
            
        y_data =\
            torch.tensor(self.labels[self.start_index:self.start_index + self.batch_size])\
            .type(torch.LongTensor).to(self.device)
            
        attention_mask =\
            torch.tensor(self.make_attention(original_len))\
            .type(torch.LongTensor).to(self.device)

        self.start_index += self.batch_size

        '''
            x_data      : [32, 15, 22] (Batch, Window, Factors)
            y_data      : [32, 1] 0, 1
            index_data  : batch size 32
            rindex_data : batch size 32
        '''
        return x_data, y_data, np.array(original_len), attention_mask

    def make_attention(self, original_len):
        attention_masks = []
        max_len = max(original_len)

        for length in original_len:
            seq_mask = [1 for _ in range(length)]
            for j in range(max_len - length):
                seq_mask.append(0)
            attention_masks.append(seq_mask)

        return attention_masks
    
    def make_padding(self, x_data,original_len):
        max_len =max(original_len)
        for i in range(len(x_data)):
            nedded = max_len - len(x_data[i])
            for j in range(nedded):
                x_data[i].append([0 for k in range(100)])
        
        return x_data
        
    
if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='')
    args = parser.parse_args()
    
    filename = './data/dev.txt'
    train_iter = FSIterator(filename, args)
    for x_data, y_data, original_len, attention_mask in train_iter: # TODO for debugging
        # x_data : B x len x Embedding size
        # y_data : B
        # original_len : B (list)
        # attention_mask : B x len
        pass