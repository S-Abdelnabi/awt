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
    def __init__(self, d_model, dropout=0.2, max_len=5000):
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

class AutoencModel(nn.Module):
    """Container module with an encoder followed by a classification over vocab. The output is then passed to the message encoder."""

    def __init__(self, ntoken, ninp, nlayers_encoder=6, pos_drop = 0.2, transformer_drop=0.1, dropouti=0.15, dropoute=0.1, tie_weights=False, attention_heads=8):
        super(AutoencModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.idrop = nn.Dropout(dropouti)
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, pos_drop)
		
        self.transformer_drop = transformer_drop
        self.embeddings = nn.Embedding(ntoken, ninp)
        self.attention_heads = attention_heads
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_dec_encoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 

        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.sent_decoder = nn.TransformerDecoder(self.sent_dec_encoder_layer, nlayers_encoder)			

        self.tie_weights = tie_weights

        #decodes the last transformer encoder layer output to vocab.
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.embeddings.weight
        self.init_weights() 

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward_sent_encoder(self, input):
        emb = embedded_dropout(self.embeddings, input, dropout=self.dropoute if self.training else 0) * math.sqrt(self.ninp)
        emb = self.pos_encoder(emb)

        sent_encoder_out = self.sent_encoder(emb)
        m = nn.AdaptiveAvgPool1d(1)
        sent_embedding = m(sent_encoder_out.view(sent_encoder_out.size(1),sent_encoder_out.size(2),sent_encoder_out.size(0)))
        sent_embedding = sent_embedding.view(sent_embedding.size(0),sent_embedding.size(1))

        return sent_embedding, sent_encoder_out

    def forward_sent_decoder(self, sent_embedding, input_data, sent_encoder_out, gumbel_temp):
        device = input_data.device
        mask = self._generate_square_subsequent_mask(len(input_data)).to(device)
		
        input_data_emb = self.embeddings(input_data) * math.sqrt(self.ninp)


        input_decoder = torch.zeros([input_data.size(0),input_data.size(1),self.ninp]).float()
        input_decoder = input_decoder.to(device)
        input_decoder[1:input_data_emb.size(0),:,:] = input_data_emb[0:input_data_emb.size(0)-1,:,:]

        sent_embeddings_repeat = sent_embedding.view(1,sent_embedding.size(0),sent_embedding.size(1)).repeat(input_data.size(0),1,1)   
        input_decoder = input_decoder + sent_embeddings_repeat
        input_decoder = self.pos_encoder(input_decoder)

        sent_decoded = self.sent_decoder(input_decoder, memory=sent_encoder_out, tgt_mask=mask)
        sent_decoded_vocab = self.decoder(sent_decoded.view(sent_decoded.size(0)*sent_decoded.size(1), sent_decoded.size(2)))	
		
        return sent_decoded_vocab
		

    def forward_sent(self, input_dec, input_noise, gumbel_temp):
        sent_embedding, encoder_out = self.forward_sent_encoder(input_noise)
        sent_decoded_vocab_soft = self.forward_sent_decoder(sent_embedding, input_dec, encoder_out, gumbel_temp)
        return sent_decoded_vocab_soft
