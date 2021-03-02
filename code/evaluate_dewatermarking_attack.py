import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import data_wm_pairs
import model_mt_autoenc_cce
import lang_model
from sentence_transformers import SentenceTransformer
from scipy.stats import binom_test

from sklearn.metrics import f1_score
from utils import batchify, repackage_hidden, get_batch_different, generate_msgs
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
corpus_dae = data_wm_pairs.Corpus(args.data)

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



def random_cut_sequence(sequence, limit=10):

    rand_cut_start = np.random.randint(low=0, high=limit)
    rand_cut_end = sequence.size(0) - np.random.randint(low=0, high=limit)

    sequence = sequence[rand_cut_start:rand_cut_end, :]
    new_seq_len = rand_cut_end -  rand_cut_start 
    #print(sequence.size())
    return sequence

def compare_msg_whole(msgs,msg_out):
    correct = np.count_nonzero(np.sum(np.equal(msgs.detach().cpu().numpy().astype(int),msg_out.detach().cpu().numpy().astype(int)),axis=1)==args.msg_len*args.msgs_segment)
    return correct
	
def compare_msg_bits(msgs,msg_out):
    correct = np.count_nonzero(np.equal(msgs.detach().cpu().numpy().astype(int),msg_out.detach().cpu().numpy().astype(int))==True)
    return correct
	
#def get_idx_from_logits(sequence,seq_len,bsz):
#    sequence = sequence.view(seq_len,bsz,sequence.size(1))
#    sequence_idx = torch.argmax(sequence, dim=-1)
#    return sequence_idx

	
def get_idx_from_logits(sequence,seq_len,bsz):
    m = nn.Softmax(dim=-1)
    sequence = sequence.view(seq_len,bsz,sequence.size(1))
    sequence = m(sequence)    
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


def noisy_sampling(sent_encoder_out,both_embeddings,data):
    candidates_emb = []
    candidates_one_hot = []
    candidates_soft_prob = []
    for i in range(0,args.samples_num):
        with torch.no_grad():
            sent_out_emb, sent_out_hot, sent_out_soft = model_gen.forward_sent_decoder(both_embeddings, data, sent_encoder_out, args.gumbel_temp)
            candidates_emb.append(sent_out_emb.detach())
            candidates_one_hot.append(sent_out_hot.detach())
            candidates_soft_prob.append(sent_out_soft.detach())			
    return candidates_emb,candidates_one_hot,candidates_soft_prob

def autoenc_greedy(data, batch_size):
    sent_emb, encoder_out = model_autoenc_attack.forward_sent_encoder(data)
    sent_out_soft = model_autoenc_attack.forward_sent_decoder(sent_emb, data, encoder_out, args.gumbel_temp)
    sent_out = get_idx_from_logits(sent_out_soft.detach(),data.size(0),batch_size)
    sent_out = sent_out[0,:]
    sent_out = torch.cat( (sent_out, torch.zeros(1,dtype=torch.long).cuda()), axis=0)
    sent_out = sent_out.view(2,1)
	
    for j in range(1,data.size(0)):
        sent_out_soft =  model_autoenc_attack.forward_sent_decoder(sent_emb, sent_out, encoder_out, args.gumbel_temp)
        sent_out_new = get_idx_from_logits(sent_out_soft.detach(),j+1,batch_size)				
        sent_out = torch.cat( (sent_out[0:j,:].view(j,1),sent_out_new[j,:].view(1,1)),axis=0)
        sent_out = torch.cat( (sent_out, torch.zeros( (1,1),dtype=torch.long).cuda()), axis=0)
    return sent_out.detach()

def evaluate(data_source, out_file, batch_size=10, on_train=False):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    model_disc.eval()
    langModel.eval()
    model_autoenc_attack.eval()
	
    total_loss_lm = 0
    ntokens = len(corpus.dictionary)
    tot_count = 0
    correct_msg_count = 0
    tot_count_bits = 0
    correct_msg_count_bits = 0
    sig = nn.Sigmoid()
    batch_count = 0
    meteor_tot = 0
    l2_distances = 0
    bert_diff = [0 for i in range(0,args.samples_num)]
    long_msg_count = 0
    long_msg = np.zeros([batch_size, args.msg_len*args.msgs_segment])
    long_msg_out = np.zeros([batch_size, args.msg_len*args.msgs_segment])
    long_msg_out = torch.from_numpy(long_msg_out).float().cuda()
    long_msg = torch.from_numpy(long_msg).float().cuda()
    p_value = []
    real_correct = 0
    fake_correct = 0 
    y_out = []
    y_label = []
    for i in range(0, data_source.size(0) - args.bptt, args.bptt):
        data, msgs, targets = get_batch_different(data_source, i, args, all_msgs, evaluation=True)
        long_msg[:,long_msg_count*args.msg_len:long_msg_count*args.msg_len+args.msg_len] = msgs	
        if args.use_lm_loss: 
            hidden = langModel.init_hidden(batch_size)
        with torch.no_grad():
            both_embeddings, sent_encoder_out = model_gen.forward_sent_encoder(data,msgs,args.gumbel_temp)
            candidates_emb,candidates_one_hot,candidates_soft_prob = noisy_sampling(sent_encoder_out,both_embeddings,data)
            data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
            real_out = model_disc(data_emb)
            label = torch.full( (data.size(1),1), real_label)
            if args.cuda:
                label = label.cuda()
            real_out_label = torch.round(sig(real_out))
            real_correct = real_correct + np.count_nonzero(np.equal(label.detach().cpu().numpy().astype(int),real_out_label.detach().cpu().numpy().astype(int))==True) 
            y_label.append(label.detach().cpu().numpy().astype(int)[0,0])
            y_out.append(real_out_label.detach().cpu().numpy().astype(int)[0,0])
			
            output_text_beams = []
            word_idx_beams = []
            word_idx_dae_beams = []
            
            lm_loss_beams = []			
            for beam in range(0,args.samples_num):
                hidden = langModel.init_hidden(batch_size)
                word_idx = get_idx_from_logits(candidates_soft_prob[beam],data.size(0),batch_size)
                output_text = '' 
                orig_text = '' 
                for k in range(0, data.size(0)):
                    output_text = output_text + corpus.dictionary.idx2word[word_idx[k,0]] + ' '
                    orig_text = orig_text + corpus.dictionary.idx2word[data[k,0]] + ' '
                if orig_text == output_text:
                    continue				
                output_text_beams.append(output_text)
                word_idx_beams.append(word_idx)
                word_idx_dae = torch.zeros_like(word_idx).cuda()
                for k in range(0, data.size(0)):
                    word_idx_dae[k] = corpus_dae.dictionary.word2idx[corpus.dictionary.idx2word[word_idx[k,0]]] 
                word_idx_dae_beams.append(word_idx_dae)
                lm_targets = word_idx[1:data.size(0)]
                lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
                lm_inputs = word_idx[0:data.size(0)-1]
                lm_out,hidden = langModel(lm_inputs,hidden, decode=True)
                lm_loss = criterion_lm(lm_out,lm_targets)
                lm_loss_beams.append(lm_loss.item())
			#get the best beam with non-zero diff

            if len(lm_loss_beams) > 0:
                beam_argsort = np.argsort(np.asarray(lm_loss_beams))			
                best_beam_idx = beam_argsort[0]
            else:
                hidden = langModel.init_hidden(batch_size)
                lm_targets = data[1:data.size(0)]
                lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
                lm_inputs = data[0:data.size(0)-1]
                lm_out,hidden = langModel(lm_inputs,hidden, decode=True)
                lm_loss = criterion_lm(lm_out,lm_targets)
                lm_loss_beams.append(lm_loss.item())
                output_text_beams.append(orig_text)
                word_idx_beams.append(data)
                word_idx_dae = torch.zeros_like(data).cuda()
                for k in range(0, data.size(0)):
                    word_idx_dae[k] = corpus_dae.dictionary.word2idx[corpus.dictionary.idx2word[data[k,0]]] 
                word_idx_dae_beams.append(word_idx_dae)
                best_beam_idx = 0	

            #calculate bert diff and meteor score for the selected sample 
            sentences = [output_text_beams[best_beam_idx], orig_text]
            sbert_embs = sbert_model.encode(sentences)
            bert_diff_selected = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
            meteor_score_selected = meteor_score([orig_text],output_text_beams[best_beam_idx])
			
            best_beam_data_dae = word_idx_dae_beams[best_beam_idx]
            data_adv_autoenc_dae = autoenc_greedy(best_beam_data_dae, batch_size)	
            data_adv_autoenc = torch.zeros_like(data_adv_autoenc_dae).cuda()
            for k in range(0, data.size(0)):
                data_adv_autoenc[k] = corpus.dictionary.word2idx[corpus_dae.dictionary.idx2word[data_adv_autoenc_dae[k,0]]]   
                
            data_dv_autoenc_emb = model_gen.forward_sent(data_adv_autoenc,msgs,args.gumbel_temp,only_embedding=True)
            msg_out = model_gen.forward_msg_decode(data_dv_autoenc_emb.detach())

            #get prediction (and the loss) of the discriminator on the fake sequence.
            fake_out = model_disc(data_dv_autoenc_emb)
            label.fill_(fake_label)
            fake_out_label = torch.round(sig(fake_out))
            fake_correct = fake_correct + np.count_nonzero(np.equal(label.detach().cpu().numpy().astype(int),fake_out_label.detach().cpu().numpy().astype(int))==True)
            y_label.append(label.detach().cpu().numpy().astype(int)[0,0])
            y_out.append(fake_out_label.detach().cpu().numpy().astype(int)[0,0])
			
            #language loss
            if args.use_lm_loss:	
                lm_targets = data_adv_autoenc[1:data_adv_autoenc.size(0)]
                lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
                lm_inputs = data_adv_autoenc[0:data_adv_autoenc.size(0)-1]
                lm_out,hidden = langModel(lm_inputs,hidden, decode=True)
                lm_loss = criterion_lm(lm_out,lm_targets)
                total_loss_lm += lm_loss.data
                hidden = repackage_hidden(hidden)

            if bert_diff_selected < args.bert_threshold:
                word_idx_adv = data_adv_autoenc
                output_text_adv = '' 
                for k in range(0, data.size(0)):
                    output_text_adv = output_text_adv + corpus.dictionary.idx2word[word_idx_adv[k,0]] + ' '
                sentences = [output_text_adv, orig_text]
                sbert_embs = sbert_model.encode(sentences)
                meteor_adv = meteor_score([orig_text],output_text_adv)
                bert_diff_adv = np.linalg.norm(sbert_embs[0]-sbert_embs[1])
				
                long_msg_out[:,long_msg_count*args.msg_len:long_msg_count*args.msg_len+args.msg_len] = msg_out			
                l2_distances = l2_distances + bert_diff_adv
                meteor_tot = meteor_tot + meteor_adv
                out_file.write('****'+'\n')
                out_file.write(str(batch_count)+'\n')
                meteor_pair = meteor_adv
                out_file.write(orig_text+'\n')

                out_file.write(str(bert_diff_adv) +'\n')				
                out_file.write(output_text_adv+'\n')

                out_file.write(str(bert_diff_selected) +'\n')
                out_file.write(output_text_beams[best_beam_idx]+'\n')

                #compute meteor score of the adv output
                out_file.write(str(compare_msg_bits(msgs,torch.round(sig(msg_out))))+'\n')
                out_file.write(np.array2string(msgs.detach().cpu().numpy().astype(int))+'\n')
                out_file.write(np.array2string(torch.round(sig(msg_out)).detach().cpu().numpy().astype(int))+'\n')
				
            else:
                meteor_pair = 1
                meteor_tot = meteor_tot + meteor_pair
                msg_out_random =  model_gen.forward_msg_decode(data_emb)
                long_msg_out[:,long_msg_count*args.msg_len:long_msg_count*args.msg_len+args.msg_len] = msg_out_random
        if batch_count != 0 and (batch_count+1)%args.msgs_segment==0:
            long_msg_count = 0
            tot_count = tot_count + 1
            tot_count_bits = tot_count_bits + long_msg.shape[0]*long_msg.shape[1]
            long_msg_out = torch.round(sig(long_msg_out))
            similar_bits = compare_msg_bits(long_msg,long_msg_out)
            all_bits = long_msg.shape[0]*long_msg.shape[1]
            correct_msg_count = correct_msg_count + compare_msg_whole(long_msg,long_msg_out)	
            correct_msg_count_bits = correct_msg_count_bits + similar_bits 
            p_value.append(binom_test(similar_bits, all_bits, 0.5))
        else:
            long_msg_count = long_msg_count + 1

        batch_count = batch_count + 1
        f_metrics.write(str(meteor_pair) + ',' + str(bert_diff[best_beam_idx]) + ',' + str(lm_loss.item()) + '\n')
    p_value_smaller = sum(i < 0.05 for i in p_value)
    Fscore = f1_score(y_label,y_out)

    if args.use_lm_loss: 
        total_loss_lm = total_loss_lm.item()  
    return total_loss_lm/batch_count, correct_msg_count/tot_count, correct_msg_count_bits/tot_count_bits, meteor_tot/batch_count, l2_distances/batch_count, fake_correct/batch_count, real_correct/batch_count, Fscore ,  np.mean(p_value), p_value_smaller/len(p_value)
	

	
	
	# Load the best saved model.
with open(args.gen_path, 'rb') as f:
    model_gen, _, _ , _= torch.load(f)
#print(model_gen)

with open(args.disc_path, 'rb') as f:
    model_disc, _, _ , _= torch.load(f)
#print(model_gen)

with open(args.autoenc_attack_path, 'rb') as f:
    model_autoenc_attack, _ , _= torch.load(f)
#print(model_gen)


if args.cuda:
    model_gen.cuda()
    model_disc.cuda()
    model_autoenc_attack.cuda()

f = open('val_out.txt','w')
f_metrics = open('val_out_metrics.txt','w')
val_lm_loss, val_correct_msg, val_correct_bits_msg,val_meteor, val_l2_sbert, val_correct_fake, val_correct_real, val_Fscore, val_pvalue, val_pvalue_inst = evaluate(val_data, f, eval_batch_size)

print('-' * 150)
print('| validation | lm loss {:5.2f} | msg accuracy {:5.2f} | msg bit accuracy {:5.2f} |  meteor {:5.4f} | SentBert dist. {:5.4f} | fake accuracy {:5.2f} | real accuracy {:5.2f} | F1 score {:.5f} | P-value {:.9f} | P-value inst {:5.2f}'.format(val_lm_loss,val_correct_msg*100, val_correct_bits_msg*100, val_meteor, val_l2_sbert, val_correct_fake*100,val_correct_real*100,val_Fscore,val_pvalue,val_pvalue_inst*100))
print('-' * 150)
f.close()


# Run on test data.
f = open('test_out.txt','w')
f_metrics = open('test_out_metrics.txt','w')
test_lm_loss, test_correct_msg, test_correct_bits_msg, test_meteor, test_l2_sbert, test_correct_fake, test_correct_real, test_Fscore, test_pvalue, test_pvalue_inst = evaluate(test_data, f, test_batch_size)

print('=' * 150)
print('| test | lm loss {:5.2f} | msg accuracy {:5.2f} | msg bit accuracy {:5.2f} | meteor {:5.4f} | SentBert dist. {:5.4f} | fake accuracy {:5.2f} | real accuracy {:5.2f} | F1 score {:.5f} | P-value {:.9f} | P-value inst {:5.2f}'.format(test_lm_loss, test_correct_msg*100, test_correct_bits_msg*100, test_meteor, test_l2_sbert, test_correct_fake*100,test_correct_real*100,test_Fscore,test_pvalue,test_pvalue_inst*100))
print('=' * 150)
f.close()

