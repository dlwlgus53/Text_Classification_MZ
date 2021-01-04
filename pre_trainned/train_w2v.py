from gensim.models import Word2Vec
import pandas as pd
from konlpy.tag import  Mecab
from tqdm import tqdm
import pdb
data = pd.read_csv("../data/train.txt",header = None, sep = '\t')
tokenizer =  Mecab().morphs
sentences = data[1]
print("tokenize start")
tokenized_texts = []
for sent in tqdm(sentences):
    try:
        tokenized_texts.append(tokenizer(sent))
    except:
        pass
print("tokenize end")
pdb.set_trace()

model = Word2Vec(tokenized_texts, size = 200, window = 3, min_count = 1, workers = 4)
print("train model")
model.save("mid.model")
model.intersect_word2vec_format('./ko.bin')
model.save("w2v.model")
# model = Word2Vec.load('word2vec.model')



