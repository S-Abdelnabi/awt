import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import model_mt_autoenc_cce
import lang_model
from sentence_transformers import SentenceTransformer
from scipy.stats import binom_test

from sklearn.metrics import f1_score
from utils import batchify, repackage_hidden, get_batch_different, generate_msgs, get_batch_noise
from nltk.translate.meteor_score import meteor_score

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
					
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')

parser.add_argument('--given_msg', type=list, default=[],
                    help='test against this msg only')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

randomhash = ''.join(str(time.time()).split('.'))

parser.add_argument('--gen_path', type=str,  default=randomhash+'.pt',
                    help='path to the generator')
parser.add_argument('--disc_path', type=str,  default=randomhash+'.pt',
                    help='path to the discriminator')
parser.add_argument('--autoenc_attack_path', type=str,  default=randomhash+'.pt',
                    help='path to the adversary autoencoder')
#language loss
parser.add_argument('--use_lm_loss', type=int, default=0,
                    help='whether to use language model loss')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')

					
#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=int, default=0.5,
                    help='Gumbel softmax temprature')
parser.add_argument('--gumbel_hard', type=bool, default=True,
                    help='whether to use one hot encoding in the forward pass')
#lang model params.
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize_lm', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti_lm', type=float, default=0.15,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute_lm', type=float, default=0.05,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

parser.add_argument('--use_elmo', type=int, default=1,
                    help='whether to use language model loss')
					
#message arguments
parser.add_argument('--msg_len', type=int, default=8,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The number of messages encododed during training')
parser.add_argument('--repeat_cycle', type=int, default=2,
                    help='Number of sentences to average')
parser.add_argument('--msgs_segment', type=int, default=5,
                    help='Long message')

parser.add_argument('--bert_threshold', type=float, default=20,
                    help='Threshold on the bert distance')
					
parser.add_argument('--samples_num', type=int, default=10,
                    help='Decoder beam size')

#denoising
parser.add_argument('--sub_prob', type=float, default=0.1,
                    help='Probability of substituting words')
					
args = parser.parse_args()
args.tied_lm = True
np.random.seed(args.seed)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

train_batch_size = 20
eval_batch_size = 1
test_batch_size = 1
train_data = batchify(corpus.train, train_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

ntokens = len(corpus.dictionary)

print('Args:', args)

criterion1 =  nn.NLLLoss()
criterion2 =  nn.BCEWithLogitsLoss()
if args.use_lm_loss:
    criterion_lm = nn.CrossEntropyLoss() 

if args.use_lm_loss:
    with open(args.lm_ckpt, 'rb') as f:
        pretrained_lm, _,_ = torch.load(f,map_location='cpu')
        langModel = lang_model.RNNModel(args.model, ntokens, args.emsize_lm, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti_lm, args.dropoute_lm, args.wdrop, args.tied_lm, pretrained_lm)
    del pretrained_lm
    if args.cuda:
        langModel = langModel.cuda()
        criterion_lm = criterion_lm.cuda()

### generate random msgs ###
all_msgs = generate_msgs(args)
print(all_msgs)

if not args.given_msg == []:
    all_msgs = [int(i) for i in args.given_msg]
    all_msgs = np.asarray(all_msgs)
    all_msgs = all_msgs.reshape([1,args.msg_len])


def get_idx_from_logits(sequence,seq_len,bsz):
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx

	
def get_idx_from_logits_softmax(sequence,seq_len,bsz):
    soft = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    #print(sequence.size())	
    sequence = soft(sequence)
    sequence_idx = torch.argmax(sequence, dim=-1)
    return sequence_idx
### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') 
#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list


def autoenc_greedy(data, batch_size):
    sent_emb, encoder_out = model_autoenc_attack.forward_sent_encoder(data)
    sent_out_soft = model_autoenc_attack.forward_sent_decoder(sent_emb, data, encoder_out, args.gumbel_temp)
    sent_out = get_idx_from_logits_softmax(sent_out_soft,data.size(0),batch_size)
    sent_out = sent_out[0,:]
    sent_out = torch.cat( (sent_out, torch.zeros(1,dtype=torch.long).cuda()), axis=0)
    sent_out = sent_out.view(2,1)
	
    for j in range(1,data.size(0)):
        sent_out_soft =  model_autoenc_attack.forward_sent_decoder(sent_emb, sent_out, encoder_out, args.gumbel_temp)
        sent_out_new = get_idx_from_logits_softmax(sent_out_soft,j+1,batch_size)				
        sent_out = torch.cat( (sent_out[0:j,:].view(j,1),sent_out_new[j,:].view(1,1)),axis=0)
        sent_out = torch.cat( (sent_out, torch.zeros( (1,1),dtype=torch.long).cuda()), axis=0)
    return sent_out

def evaluate(data_source, out_file, batch_size=10, on_train=False):
    # Turn on evaluation mode which disables dropout.
    if args.use_lm_loss:
        langModel.eval()
    model_autoenc_attack.eval()

	
    total_loss_lm = 0
    ntokens = len(corpus.dictionary)
    tot_count = 0

    batch_count = 0
    meteor_tot = 0
    l2_distances = 0
	
    meteor_tot_noise = 0
    l2_distances_noise = 0
	
    bert_diff = 0

    for i in range(0, data_source.size(0) - args.bptt, args.bptt):
        data, data_noise, targets = get_batch_noise(data_source, i, args,ntokens, evaluation=True)
        if args.use_elmo:
            data_text = convert_idx_to_words(data)
        else:
            data_text = None
        if args.use_lm_loss: 
            hidden = langModel.init_hidden(batch_size)
        with torch.no_grad():
            data_adv_autoenc = autoenc_greedy(data_noise, batch_size)
			
            #language loss
            if args.use_lm_loss:
                lm_targets = data_adv_autoenc[1:data_adv_autoenc.size(0)]
                lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
                lm_inputs = data_adv_autoenc[0:data_adv_autoenc.size(0)-1]
                lm_out,hidden = langModel(lm_inputs,hidden, decode=True)
                lm_loss = criterion_lm(lm_out,lm_targets)
                total_loss_lm += lm_loss.data
                hidden = repackage_hidden(hidden)

            word_idx_adv = data_adv_autoenc
            output_text_adv = '' 
            orig_text = '' 
            noisy_text = '' 
            for k in range(0, data.size(0)):
                output_text_adv = output_text_adv + corpus.dictionary.idx2word[word_idx_adv[k,0]] + ' '
                orig_text = orig_text + corpus.dictionary.idx2word[data[k,0]] + ' '
                noisy_text = noisy_text + corpus.dictionary.idx2word[data_noise[k,0]] + ' '

            sentences = [output_text_adv, orig_text]
            sbert_embs = sbert_model.encode(sentences)
            meteor_adv = meteor_score([orig_text],output_text_adv)
            bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])

            sentences = [noisy_text, orig_text]
            sbert_embs = sbert_model.encode(sentences)
            meteor_noise = meteor_score([orig_text],noisy_text)
            bert_diff_noise = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
				
            l2_distances = l2_distances + bert_diff_adv
            l2_distances_noise = l2_distances_noise + bert_diff_noise
            meteor_tot = meteor_tot + meteor_adv
            meteor_tot_noise = meteor_tot_noise + meteor_noise
			
            out_file.write('****'+'\n')
            out_file.write(str(batch_count)+'\n')
            meteor_pair = meteor_adv
            out_file.write(orig_text+'\n')
            out_file.write(str(bert_diff_noise) +'\n')				
            out_file.write(str(meteor_noise)+'\n')
            out_file.write(noisy_text+'\n')

            out_file.write(str(bert_diff_adv) +'\n')				
            out_file.write(str(meteor_pair)+'\n')
            out_file.write(output_text_adv+'\n')

        batch_count = batch_count + 1
        f_metrics.write(str(meteor_pair) + ',' + str(bert_diff_adv) + ',' + str(lm_loss.item()) + '\n')

    if args.use_lm_loss: 
        total_loss_lm = total_loss_lm.item()  
    return total_loss_lm/batch_count, meteor_tot/batch_count, meteor_tot_noise/batch_count, l2_distances/batch_count, l2_distances_noise/batch_count
	
	
with open(args.autoenc_attack_path, 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)


if args.cuda:
    model_autoenc_attack.cuda()

f = open('val_out.txt','w')
f_metrics = open('val_out_metrics.txt','w')
val_lm_loss,val_meteor, val_meteor_noise, val_l2_sbert, val_l2_sbert_noise = evaluate(val_data, f, eval_batch_size)
print(val_lm_loss)
print(val_meteor)
print(val_meteor_noise)
print(val_l2_sbert)
print(val_l2_sbert_noise)

print('-' * 150)
print('| validation | lm loss {:5.2f} | meteor {:5.4f} | meteor noise {:5.4f} | SentBert dist. {:5.4f} | SentBert dist. noise {:5.4f}'.format(val_lm_loss, val_meteor, val_meteor_noise, val_l2_sbert, val_l2_sbert_noise))
print('-' * 150)
f.close()


# Run on test data.
f = open('test_out.txt','w')
f_metrics = open('test_out_metrics.txt','w')
test_lm_loss, test_meteor, test_meteor_noise, test_l2_sbert, test_l2_sbert_noise = evaluate(test_data, f, test_batch_size)

print(test_lm_loss)
print(test_meteor)
print(test_meteor_noise)
print(test_l2_sbert)
print(test_l2_sbert_noise)


print('=' * 150)
print('| test | lm loss {:5.2f} | meteor {:5.4f} | meteor noise {:5.4f} | SentBert dist. {:5.4f} | SentBert dist. noise {:5.4f}'.format(test_lm_loss, test_meteor, test_meteor_noise, test_l2_sbert, test_l2_sbert_noise))
print('=' * 150)
f.close()

