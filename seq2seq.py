import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import random
import math
import time

#from torch.utils.data import DataLoader
#from torch.nn.utils.rnn import pad_sequence

#from collections import Counter
#from torchtext.datasets import Multi30k
#from torchtext.vocab import vocab
#from torchtext.data import get_tokenizer

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# en_core_seb_sm: 영어 토큰화 전용 모델, de_core_news_sm: 독일어 토큰화 전용 모델
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
# 토큰화 함수를 만드는데, input 문장의 토큰화 이후에 순서를 뒤집는 작업이 추가(논문 상 input 문장의 순서를 뒤집었을 때, 더 좋은 성능을 보임)
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# 토큰화된 문장을 파이토치 텐서로 받아주는 Field 함수를 정의함 (독일어-> 영어)
SRC = Field(tokenize = tokenize_de, init_token = '<sos>', eos_token='<eos>', lower=True) # 독일어(source)
TRG = Field(tokenize = tokenize_en, init_token = '<sos>', eos_token='<eos>', lower=True) #영어(Target)

# 학습데이터 가져오기
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
print(vars(train_data.examples[0]))

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = [('src',SRC, 'trg',TRG)])

train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

import torchtext
print(torchtext.__version__)