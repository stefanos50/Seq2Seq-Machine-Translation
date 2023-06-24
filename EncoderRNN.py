from torch import nn
import torch
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedded_size, layers_count,p):
        super(EncoderRNN, self).__init__()

        # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
        self.input_size = input_size
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.layers_count = layers_count
        self.p = p

        self.dropout = nn.Dropout(p=self.p)
        self.embedding = nn.Embedding(input_size, self.embedded_size,padding_idx=2)
        self.rnn = nn.LSTM(self.embedded_size, self.hidden_size, self.layers_count,dropout=self.p,bias=True)
        #torch.nn.init.xavier_uniform(self.embedding.weight)

        #for name, param in self.rnn.named_parameters():
            #if 'weight' in name:
                #nn.init.xavier_uniform(param)

    def forward(self, src):
        #embedded = self.embedding(src).view(1, 1, -1)
        #outputs, hidden = self.gru(embedded)
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden,cell) = self.rnn(embedded)
        return hidden, cell

