import math 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from allennlp.modules.elmo import Elmo, batch_to_ids



class DiscriminatorModel(nn.Module):
    def __init__(self, ntoken, ninp, nlayers=2, nhid = 1150, dropouti=0.1, dropoute=0.1, dropout=0.4, dropouth=0.3):
        #directly takes the embedding 
        super(DiscriminatorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.ninp = ninp
        self.ntoken = ntoken
        self.embeddings = nn.Embedding(ntoken, ninp)
        self.nhid = nhid
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropout = dropout
        self.dropouth = dropouth
        self.nlayers = nlayers
        
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid) for l in range(nlayers)]
        self.rnns = torch.nn.ModuleList(self.rnns)

		
        #classification to fake and real.
        self.real_fake_classify = [nn.Linear(nhid*3,1)]
        self.real_fake_classify = torch.nn.ModuleList(self.real_fake_classify)

        self.init_weights()
   
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
		
        for ff in self.real_fake_classify:
            torch.nn.init.xavier_normal_(ff.weight)
            ff.bias.data.fill_(0.01)

    def forward(self, input):

        input_emb = self.embeddings(input)
        input_emb = self.lockdrop(input_emb, self.dropouti)
		
        #forward through the decoder lstm 
        raw_output = input_emb
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
 
        output_lstm = self.lockdrop(raw_output, self.dropout)

	#get global average and max pooling over time steps
        m = nn.AdaptiveAvgPool1d(1)
        m2 = nn.AdaptiveMaxPool1d(1)

        lstm_out_avg = m(output_lstm.view(output_lstm.size(1),output_lstm.size(2),output_lstm.size(0)))
        lstm_out_max = m2(output_lstm.view(output_lstm.size(1),output_lstm.size(2),output_lstm.size(0)))

        lstm_out_avg = lstm_out_avg.view(lstm_out_avg.size(0),lstm_out_avg.size(1))
        lstm_out_max = lstm_out_max.view(lstm_out_avg.size(0),lstm_out_avg.size(1))
        last_hidden = new_h[0].view(lstm_out_avg.size(0),lstm_out_avg.size(1))
        lstm_out = torch.cat([lstm_out_avg,lstm_out_max, last_hidden],dim=1)

        for ff in self.real_fake_classify:       		
            lstm_out = ff(lstm_out)
			
        real_fake_out = lstm_out
        return real_fake_out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid).zero_(), weight.new(1, bsz, self.nhid).zero_()) for l in range(self.nlayers)]
