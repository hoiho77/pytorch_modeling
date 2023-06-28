import torch
import torch.nn as nn
import random
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, DROPOUT): # input 사이즈, 임베딩 사이즈, 히든 사이즈, 레이어수, 드롭아웃
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim) # input_dim
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=DROPOUT)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell

    # 이 코드에서 hidden, cell 부분만 리턴하는데, 이 hidden, cell을 Decoder의 input으로 사용하기 위해서입니다.
    # 코드에서 n directions 라고 적힌 부분은, 단방향 LSTM의 경우 1, 양방향일 경우 2의 값을 갖습니다(단방향을 가정).

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
         super().__init__()

         self.hid_dim = hid_dim
         self.n_layers = n_layers
         self.embedding = nn.Embedding(output_dim, emb_dim) #output_dim

         self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
         self.dropout = nn.Dropout(dropout)

         self.fc_out = nn.Linear(hid_dim, output_dim)
         self.output_dim = output_dim

    def forward(self, input, hidden, cell):
        # input = [batch size]  * 첫 번째 step의 input은 <sos>가 됩니다.
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batchsize]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))
        #* 최종적으로는 fully-connected layer를 통해서 정해둔 임베딩 크기에 맞게 prediction 벡터를 리턴합니다.

        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__ (self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing >> 모델 학습 시에는 디코더의 입력값으로 이전 시점의 디코더 출력 단어가 아닌 실제 정답 단어를 입력해 줘야 한다
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens (첫번째 토큰의 경우 <sos>로 initialize)
        input = trg[0, :]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden,cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not --> 랜덤으로 적용 (모두 적용하는 것이 아님)
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token >> input을 수정함
            input = trg[t] if teacher_force else top1

        return outputs


