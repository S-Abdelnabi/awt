import math 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout

def sample_gumbel(x):
    noise = torch.cuda.FloatTensor(x.size()).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise)

def gumbel_softmax_sample(x, tau=0.5):
    noise = sample_gumbel(x)
    y = (F.log_softmax(x,dim=-1) + noise) / tau
    #ysft = F.softmax(y)
    return y.view_as(x)

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

class TranslatorGeneratorModel(nn.Module):
    """Container module with an encoder followed by a classification over vocab. The output is then passed to the message encoder."""

    def __init__(self, ntoken, ninp, msg_len=64, msg_in_mlp_layers = 1, msg_in_mlp_nodes= [], nlayers_encoder=6, transformer_drop=0.1, dropouti=0.15, dropoute=0.1, tie_weights=False, shared_encoder=False, attention_heads=8, pretrained_model=None):
        super(TranslatorGeneratorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
		
        self.transformer_drop = transformer_drop
        self.embeddings = nn.Embedding(ntoken, ninp)
        self.attention_heads = attention_heads
        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.sent_dec_encoder_layer = nn.TransformerDecoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 

        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, nlayers_encoder)
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_encoder)
        self.sent_decoder = nn.TransformerDecoder(self.sent_dec_encoder_layer, nlayers_encoder)			

        self.tie_weights = tie_weights
        self.msg_in_mlp_layers = msg_in_mlp_layers

        #MLP for the input message
        if msg_in_mlp_layers == 1:
            self.msg_in_mlp = nn.Linear(msg_len, ninp)
        else:
            self.msg_in_mlp = [nn.Linear(msg_len if l == 0 else msg_in_mlp_nodes[l-1], msg_in_mlp_nodes[l] if l!=msg_in_mlp_layers-1 else ninp) for l in range(msg_in_mlp_layers)]
            self.msg_in_mlp = torch.nn.ModuleList(self.msg_in_mlp)

        #mlp for the message decoding. Takes the last token output
        self.msg_out_mlp = [nn.Linear(2*ninp, msg_len)]
        self.msg_out_mlp = torch.nn.ModuleList(self.msg_out_mlp)

        if shared_encoder:
            self.msg_decoder = self.sent_encoder

        #decodes the last transformer encoder layer output to vocab.
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.embeddings.weight
        self.init_weights()
        if pretrained_model:
            self.init_model(pretrained_model)  

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_model(self, pretrained_model):
        with torch.no_grad():
            self.embeddings.weight.data = copy.deepcopy(pretrained_model.embeddings.weight.data)
            self.sent_encoder = copy.deepcopy(pretrained_model.sent_encoder)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_msg_decode(self, input):

        input = input * math.sqrt(self.ninp)
        input = self.pos_encoder(input) 
        msg_decoder_out = self.msg_decoder(input)

        m = nn.AdaptiveAvgPool1d(1)
        m2 = nn.AdaptiveMaxPool1d(1)
		
        msg_dec_avg = m(msg_decoder_out.view(msg_decoder_out.size(1),msg_decoder_out.size(2),msg_decoder_out.size(0)))
        msg_dec_avg = msg_dec_avg.view(msg_dec_avg.size(0),msg_dec_avg.size(1))
		
        last_state = msg_decoder_out[msg_decoder_out.size(0)-1,:,:]
        msg_decoder_rep = torch.cat([msg_dec_avg,last_state],dim=1)
		
        for ff in self.msg_out_mlp:       		
            msg_decoder_rep = ff(msg_decoder_rep)

        decoded_msg_out = msg_decoder_rep


        return decoded_msg_out

    def forward_sent_encoder(self, input,msg_input, gumbel_temp, only_embedding=False):
        if only_embedding:
            emb = self.embeddings(input) 
            return emb

        emb = self.embeddings(input) * math.sqrt(self.ninp)
        emb = self.pos_encoder(emb)

        sent_encoder_out = self.sent_encoder(emb)
        m = nn.AdaptiveAvgPool1d(1)
        sent_embedding = m(sent_encoder_out.view(sent_encoder_out.size(1),sent_encoder_out.size(2),sent_encoder_out.size(0)))
        sent_embedding = sent_embedding.view(sent_embedding.size(0),sent_embedding.size(1))

        #get msg fc
        prev_msg_out = msg_input

        if self.msg_in_mlp_layers == 1:
            prev_msg_out = F.relu(self.msg_in_mlp(prev_msg_out)) 
        else: 
            for l, ff in enumerate(self.msg_in_mlp):
                prev_msg_out = F.relu(ff(prev_msg_out))
        msg_out = prev_msg_out

        #add the message to the sentence embedding 
        both_embeddings = sent_embedding.add(msg_out)
        return both_embeddings, sent_encoder_out

    def forward_sent_decoder(self, both_embeddings, input_data, sent_encoder_out, gumbel_temp):
        device = input_data.device
        mask = self._generate_square_subsequent_mask(len(input_data)).to(device)
		
        input_data_emb = self.embeddings(input_data)

        input_decoder = torch.zeros([input_data.size(0),input_data.size(1),self.ninp]).float()
        input_decoder = input_decoder.to(device)
        input_decoder[1:input_data_emb.size(0),:,:] = input_data_emb[0:input_data_emb.size(0)-1,:,:]

        both_embeddings_repeat = both_embeddings.view(1,both_embeddings.size(0),both_embeddings.size(1)).repeat(input_data.size(0),1,1)   
        input_decoder = input_decoder + both_embeddings_repeat
        input_decoder = self.pos_encoder(input_decoder)

        sent_decoded = self.sent_decoder(input_decoder, memory=sent_encoder_out, tgt_mask=mask)
        sent_decoded_vocab = self.decoder(sent_decoded.view(sent_decoded.size(0)*sent_decoded.size(1), sent_decoded.size(2)))	
        sent_decoded_vocab_hot = F.gumbel_softmax(F.log_softmax(sent_decoded_vocab,dim=-1), tau = gumbel_temp, hard=True)
        sent_decoded_vocab_hot_out =  sent_decoded_vocab_hot.view(input_decoder.size(0), input_decoder.size(1), sent_decoded_vocab_hot.size(1))

        sent_decoded_vocab_emb = torch.mm(sent_decoded_vocab_hot,self.embeddings.weight)
        sent_decoded_vocab_emb = sent_decoded_vocab_emb.view(input_decoder.size(0), input_decoder.size(1), input_decoder.size(2))
		
        sent_decoded_vocab_soft = gumbel_softmax_sample(sent_decoded_vocab, tau = gumbel_temp)
		
        return sent_decoded_vocab_emb, sent_decoded_vocab_hot_out, sent_decoded_vocab_soft
		

    def forward_sent(self, input,msg_input, gumbel_temp, only_embedding=False):
        if only_embedding:
            return self.forward_sent_encoder(input,msg_input, gumbel_temp, only_embedding=True)
        sent_msg_embedding, encoder_out = self.forward_sent_encoder(input,msg_input, gumbel_temp, only_embedding=False)
        sent_decoded_vocab_emb, sent_decoded_vocab_hot, sent_decoded_vocab_soft = self.forward_sent_decoder(sent_msg_embedding, input, encoder_out, gumbel_temp)
        return sent_decoded_vocab_emb, sent_decoded_vocab_hot, sent_decoded_vocab_soft

class TranslatorDiscriminatorModel(nn.Module):
    def __init__(self, ninp, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8, dropouti=0.1):
        #directly takes the embedding 
        super(TranslatorDiscriminatorModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.ninp = ninp
        self.transformer_drop = transformer_drop
        self.attention_heads = attention_heads
        self.pos_encoder = PositionalEncoding(ninp, transformer_drop)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=attention_heads, dropout=transformer_drop) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, nlayers_encoder)

        #classification to fake and real.
        self.real_fake_classify = [nn.Linear(ninp*2,1)]
        self.real_fake_classify = torch.nn.ModuleList(self.real_fake_classify)

        self.init_weights()

    def init_weights(self):
        for ff in self.real_fake_classify:
            torch.nn.init.xavier_normal_(ff.weight)
            ff.bias.data.fill_(0.01)

    def forward(self, input_emb):
        #print(input_emb)
        input_emb = input_emb * math.sqrt(self.ninp)
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
