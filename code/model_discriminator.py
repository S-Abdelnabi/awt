import math 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from allennlp.modules.elmo import Elmo, batch_to_ids

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DiscriminatorModel(nn.Module):
    def __init__(self, ntoken, ninp, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        #directly takes the embedding 
        super(DiscriminatorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.ninp = ninp
        self.ntoken = ntoken
        self.transformer_drop = transformer_drop
        self.attention_heads = attention_heads
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, nlayers_encoder)		
        self.embeddings = nn.Embedding(ntoken, ninp)
		
        #classification to fake and real.
        self.real_fake_classify = [nn.Linear(ninp*2,1)]
        self.real_fake_classify = torch.nn.ModuleList(self.real_fake_classify)

        self.init_weights()
   
    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
		
        for ff in self.real_fake_classify:
            torch.nn.init.xavier_normal_(ff.weight)
            ff.bias.data.fill_(0.01)

    def forward(self, input):

        input_emb = self.embeddings(input) * math.sqrt(self.ninp)
        input_emb = self.pos_encoder(input_emb)
 
        discr_encoder_out = self.transformer_encoder(input_emb)

        last_state = discr_encoder_out[discr_encoder_out.size(0)-1,:,:]

        m = nn.AdaptiveAvgPool1d(1)
        disc_avg = m(discr_encoder_out.view(discr_encoder_out.size(1),discr_encoder_out.size(2),discr_encoder_out.size(0)))
        disc_avg = disc_avg.view(disc_avg.size(0),disc_avg.size(1))

        disc_enc_rep = torch.cat([disc_avg,last_state],dim=1)
        for ff in self.real_fake_classify:       		
            disc_enc_rep = ff(disc_enc_rep)
			
        real_fake_out = disc_enc_rep
        return real_fake_out
