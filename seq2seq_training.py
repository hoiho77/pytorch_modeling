import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import torchtext
print(torchtext.__version__) # 0.10.0
print(torch.__version__) # 1.9.0+cpu
import spacy
import random
import math
import time
from seq2seq import *

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

# 단어들을 벡터에 정수로 부여하기 위한 vocabulary 생성하기
# vocab > field의 요소에 대해 모든 가능한 값과 이에 대응되는 숫자표현을 정의
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
# TRG.vocab['bus'] -- vocab 확인하기

# 만든 vocabulary로 데이터셋을 숫자 벡터로 매핑
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
# 미니배치 안의 문장들을 모두 같은 길이로 맞춰주는 기능이 존재함 (미니 배치 안에서 가장 긴 문장과 길이 맞춤 ([1,1,1,...,1,1]))
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size= BATCH_SIZE, device=DEVICE)
# gpu 할당 시 출력되지 않음
print(next(iter(train_iterator)).src)
print(next(iter(train_iterator)).src.shape)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
# list(model.modules()) 모델 구조확인

optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
clip = 5 # 몇으로 잡아야하는지 확인이 필요함

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model, iterator, optimizer, criterion, clip):
    # clip 주로 RNN계열에서 gradient vanishing이나 gradient exploding이 많이 발생하는데, gradient exploding을 방지하여 학습의 안정화를 도모하기 위해 사용하는 방법이다.
    # gradient가 일정 threshold를 넘어가면 clipping을 해준다. clipping은 gradient의 L2norm(norm이지만 보통 L2 norm사용)으로 나눠주는 방식으로 하게된다.
    # threshold의 경우 gradient가 가질 수 있는 최대 L2norm을 뜻하고 이는 하이퍼파라미터로 사용자가 설정해주어야 한다.
    # clipping이 없으면 gradient가 너무 뛰어서 global minimum에 도달하지 못하고 너무 엉뚱한 방향으로 향하게 되지만,
    # clipping을 하게 되면 gradient vector가 방향은 유지하되 적은 값만큼 이동하여 도달하려고 하는 곳으로 안정적으로 내려가게 된다.
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

#train(Seq2Seq, train_iterator, optimizer, criterion, clip)

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')