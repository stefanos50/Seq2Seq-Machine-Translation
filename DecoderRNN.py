import torch
from torch import nn
class DecoderRNN(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, embedded_size, layers_count,p):
        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers_count = layers_count
        self.p = p

        self.dropout = nn.Dropout(self.p)
        self.embedding = nn.Embedding(self.input_size,embedded_size,padding_idx=2)
        self.rnn = nn.LSTM(self.embedded_size,self.hidden_size,self.layers_count,dropout=self.p,bias=True)
        self.fc = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        #torch.nn.init.xavier_uniform(self.embedding.weight)
        #torch.nn.init.xavier_uniform(self.fc.weight)
        #for name, param in self.rnn.named_parameters():
           # if 'weight' in name:
                #nn.init.xavier_uniform(param)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        embedding = self.dropout(self.embedding(input))

        outs,(hidden,cell) = self.rnn(embedding,(hidden,cell))

        pred = self.fc(outs)

        preds = pred.squeeze(0)
        #preds = self.softmax(preds)

        return preds,hidden,cell
        #input = input.view(1, -1)
        #embedded = nn.functional.relu(self.embedding(input))
        #output, hidden = self.gru(embedded, hidden)
        #prediction = self.softmax(self.out(output[0]))

        #return prediction, hidden