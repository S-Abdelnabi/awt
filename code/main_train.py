import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys
import data
import model_mt_autoenc_cce
import lang_model

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden
from fb_semantic_encoder import BLSTMEncoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.00003,
                    help='initial learning rate')
parser.add_argument('--disc_lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')
parser.add_argument('--fixed_length', type=int, default=0,
                    help='whether to use a fixed input length (bptt value)')
parser.add_argument('--dropout_transformer', type=float, default=0.1,
                    help='dropout applied to transformer layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash,
                    help='path to save the final model')
parser.add_argument('--save_interval', type=int, default=20,
                    help='saving models regualrly')
					
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

#message arguments
parser.add_argument('--msg_len', type=int, default=64,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The total number of messages')
parser.add_argument('--msg_in_mlp_layers', type=int, default=1,
                    help='message encoding FC layers number')
parser.add_argument('--msg_in_mlp_nodes', type=list, default=[],
                    help='nodes in the MLP of the message')

#transformer arguments
parser.add_argument('--attn_heads', type=int, default=4,
                    help='The number of attention heads in the transformer')
parser.add_argument('--encoding_layers', type=int, default=3,
                    help='The number of encoding layers')
parser.add_argument('--shared_encoder', type=bool, default=True,
                    help='If the message encoder and language encoder will share weights')

#adv. transformer arguments
parser.add_argument('--adv_attn_heads', type=int, default=4,
                    help='The number of attention heads in the adversary transformer')
parser.add_argument('--adv_encoding_layers', type=int, default=3,
                    help='The number of encoding layers in the adversary transformer')

#gumbel softmax arguments
parser.add_argument('--gumbel_temp', type=int, default=0.5,
                    help='Gumbel softmax temprature')

#Adam optimizer arguments
parser.add_argument('--scheduler', type=int, default=1,
                    help='whether to schedule the lr according to the formula in: Attention is all you need')
parser.add_argument('--warm_up', type=int, default=6000,
                    help='number of linear warm up steps')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam beta1 parameter')
parser.add_argument('--beta2', type=float, default=0.98,
                    help='Adam beta2 parameter')
parser.add_argument('--eps', type=float, default=1e-9,
                    help='Adam eps parameter')
#GAN arguments
parser.add_argument('--msg_weight', type=float, default=25,
                    help='The factor multiplied with the message loss')

#fb InferSent semantic loss 
parser.add_argument('--use_semantic_loss', type=int, default=1,
                    help='whether to use semantic loss')
parser.add_argument('--glove_path', type=str, default='sent_encoder/GloVe/glove.840B.300d.txt',
                    help='path to glove embeddings')
parser.add_argument('--infersent_path', type=str, default='sent_encoder/infersent2.pkl',
                    help='path to the trained sentence semantic model')
parser.add_argument('--sem_weight', type=float, default=40,
                    help='The factor multiplied with the semantic loss')
					
#language loss
parser.add_argument('--use_lm_loss', type=int, default=1,
                    help='whether to use language model loss')
parser.add_argument('--lm_weight', type=float, default=1,
                    help='The factor multiplied with the lm loss')
parser.add_argument('--lm_ckpt', type=str, default='WT2_lm.pt',
                    help='path to the fine tuned language model')
					
#reconstruction loss
parser.add_argument('--use_reconst_loss', type=int, default=1,
                    help='whether to use language reconstruction loss')
parser.add_argument('--reconst_weight', type=float, default=1,
                    help='The factor multiplied with the reconstruct loss')

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
parser.add_argument('--dropouti_lm', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute_lm', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
#GAN arguments			
parser.add_argument('--discr_interval', type=int, default=1,
                    help='when to update the discriminator')
parser.add_argument('--autoenc_path', type=str, default='',
                    help='path of the autoencoder path to use as init to the generator, in case the model is pretrained as autoencoder only')
parser.add_argument('--gen_weight', type=float, default=2,
                    help='The factor multiplied with the gen loss')



args = parser.parse_args()
args.tied = True
args.tied_lm = True

log_file_loss_val = open('log_file_loss_val.txt','w') 
log_file_loss_train = open('log_file_loss_train.txt','w') 

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

def model_save(fn):
    with open(fn+'_gen.pt', 'wb') as f:
        torch.save([model_gen, criterion, criterion_reconst, optimizer_gen], f)
    with open(fn+'_disc.pt', 'wb') as f:
        torch.save([model_disc, criterion, criterion_reconst, optimizer_disc], f)

def model_load(fn):
    global model_gen, model_disc, criterion, criterion_reconst, optimizer_gen, optimizer_disc
    with open(fn+'_gen.pt', 'rb') as f:
        model_gen, criterion, criterion_reconst, optimizer_gen = torch.load(f,map_location='cpu')
    with open(fn+'_disc.pt', 'rb') as f:
        model_disc, criterion, criterion_reconst, optimizer_disc = torch.load(f,map_location='cpu')
		
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

criterion = None
criterion_reconst = None 
criterion_sem = None 
criterion_lm = None 

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

if args.autoenc_path != '':
    with open(args.autoenc_path,'rb') as f:
        autoenc_model, _, _ = torch.load(f)
else:
    autoenc_model = None 

## global variable for the number of steps ( batches) ##
step_num = 1
discr_step_num = 1

def learing_rate_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -0.8) * min(np.power(step_num, -0.5), step_num*np.power(warm_up, -1.5))
    return lr

def learing_rate_disc_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -1.1) * min(np.power(discr_step_num, -0.5), discr_step_num*np.power(warm_up, -1.5))
    return lr
	
if args.resume:
    all_msgs = np.loadtxt('msgs.txt')
    print('Resuming model ...')
    model_load(args.resume) 
    optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() if args.scheduler else args.lr
    optimizer_disc.param_groups[0]['lr'] = learing_rate_disc_scheduler() if args.scheduler else args.lr
else:
    ### generate random msgs ###
    all_msgs = generate_msgs(args)
    model_gen = model_mt_autoenc_cce.TranslatorGeneratorModel(ntokens, args.emsize, args.msg_len, args.msg_in_mlp_layers , args.msg_in_mlp_nodes, args.encoding_layers, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.shared_encoder, args.attn_heads,autoenc_model)
    model_disc = model_mt_autoenc_cce.TranslatorDiscriminatorModel(args.emsize, args.adv_encoding_layers, args.dropout_transformer, args.adv_attn_heads, args.dropouti)
    for p in model_disc.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


###
if not criterion:
    criterion =  nn.BCEWithLogitsLoss()
if args.use_semantic_loss and not criterion_sem:
    criterion_sem = nn.L1Loss()
if args.use_lm_loss and not criterion_lm:
    criterion_lm = nn.CrossEntropyLoss() 
if args.use_reconst_loss and not criterion_reconst:
    criterion_reconst = nn.CrossEntropyLoss()
###

### semantic model ###
if args.use_semantic_loss: 
    modelSentEncoder = BLSTMEncoder(word2idx, idx2word, args.glove_path)
    encoderState = torch.load(args.infersent_path,map_location='cpu')
    state = modelSentEncoder.state_dict()
    for k in encoderState:
        if k in state:
            state[k] = encoderState[k]
    modelSentEncoder.load_state_dict(state)

## language model ## 
if args.use_lm_loss:
    with open(args.lm_ckpt, 'rb') as f:
        pretrained_lm, _,_ = torch.load(f,map_location='cpu')
        langModel = lang_model.RNNModel(args.model, ntokens, args.emsize_lm, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti_lm, args.dropoute_lm, args.wdrop, args.tied_lm, pretrained_lm)
    del pretrained_lm



if args.cuda:
    model_gen = model_gen.cuda()
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()
    if args.use_lm_loss:
        criterion_lm = criterion_lm.cuda()
        langModel = langModel.cuda()
		
###
params = list(model_gen.parameters()) + list(criterion.parameters()) + list(criterion_reconst.parameters()) + list(model_disc.parameters())
params_gen = model_gen.parameters()
params_disc = model_disc.parameters()

total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

#convert word ids to text	
def convert_idx_to_words(idx):
    batch_list = []
    for i in range(0,idx.size(1)):
        sent_list = []
        for j in range(0,idx.size(0)):
            sent_list.append(corpus.dictionary.idx2word[idx[j,i]])
        batch_list.append(sent_list)
    return batch_list

### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()
    model_disc.eval()
    if args.use_semantic_loss: 
        modelSentEncoder.eval()
    if args.use_lm_loss:
        langModel.eval()
		
    total_loss_gen = 0
    total_loss_disc = 0
    total_loss_msg = 0
    total_loss_sem = 0
    total_loss_reconst = 0
    total_loss_lm = 0
	
    ntokens = len(corpus.dictionary)
    batches_count = 0
    for i in range(0, data_source.size(0) - args.bptt, args.bptt):
        if args.use_lm_loss: 
            hidden = langModel.init_hidden(batch_size)
        data, msgs, targets = get_batch_different(data_source, i, args,all_msgs, evaluation=True)
        #get a batch of fake (edited) sequence from the generator
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        msg_out = model_gen.forward_msg_decode(fake_data_emb)
        #get prediction (and the loss) of the discriminator on the real sequence. First gen the embeddings from the generator
        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
        real_out = model_disc(data_emb)
        label = torch.full( (data.size(1),1), real_label)
        if args.cuda:
            label = label.cuda()
        errD_real = criterion(real_out,label)
        #get prediction (and the loss) of the discriminator on the fake sequence.
        fake_out = model_disc(fake_data_emb.detach())
        label.fill_(fake_label)
        errD_fake = criterion(fake_out,label)
        errD = errD_real + errD_fake

        #generator loss
        label.fill_(real_label) 
        errG_disc = criterion(fake_out,label)

        #semantic loss
        if args.use_semantic_loss: 
            orig_sem_emb = modelSentEncoder.forward_encode_nopad(data)
            fake_sem_emb = modelSentEncoder.forward_encode_nopad(fake_one_hot,one_hot=True)
            sem_loss = criterion_sem(orig_sem_emb,fake_sem_emb)
            total_loss_sem += sem_loss.data

        #msg loss of the generator
        msg_loss = criterion(msg_out, msgs)

        #lm loss of the generator
        if args.use_lm_loss:
            lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
            lm_targets = torch.argmax(lm_targets,dim=-1)
            lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
            lm_inputs = fake_one_hot[0:fake_one_hot.size(0)-1]
            lm_out,hidden = langModel(lm_inputs,hidden, decode=True,one_hot=True)
            lm_loss = criterion_lm(lm_out,lm_targets)
            total_loss_lm += lm_loss.data
            hidden = repackage_hidden(hidden)

        #reconstruction loss 
        reconst_loss = criterion_reconst(fake_data_prob,data.view(-1))
        total_loss_reconst += reconst_loss.data

        total_loss_gen +=  errG_disc.data
        total_loss_disc +=  errD.data
        total_loss_msg += msg_loss.data
        batches_count = batches_count + 1
    if args.use_semantic_loss: 
        total_loss_sem = total_loss_sem.item()
    if args.use_lm_loss: 
        total_loss_lm = total_loss_lm.item() 		

    return total_loss_reconst.item()/batches_count, total_loss_gen.item() / batches_count, total_loss_msg.item() / batches_count, total_loss_sem / batches_count, total_loss_lm/batches_count, total_loss_disc.item() / batches_count         


def train():
    # Turn on training mode which enables dropout.
    total_loss_gen = 0
    total_loss_msg = 0
    total_loss_sem = 0
    total_loss_disc = 0
    total_loss_reconst = 0
    total_loss_lm = 0
    if args.use_lm_loss:
        hidden = langModel.init_hidden(args.batch_size)
		
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, msgs, targets = get_batch_different(train_data, i, args,all_msgs, seq_len=seq_len)
        else:
            seq_len = args.bptt
            data, msgs, targets = get_batch_different(train_data, i, args, all_msgs, seq_len=None)

        model_gen.train()
        model_disc.train()
        if args.use_semantic_loss: 
            modelSentEncoder.train()
            #set parameters trainable to false.
            for p in modelSentEncoder.parameters(): #reset requires_grad
                p.requires_grad = False #they are set to False below in the generator update

        if args.use_lm_loss:
            langModel.train()
            #set parameters trainable to false.
            for p in langModel.parameters(): #reset requires_grad
                p.requires_grad = False #they are set to False below in the generator update
            hidden = repackage_hidden(hidden)

        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        ####### Update lr #######
        if args.scheduler:
            optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() 
            optimizer_disc.param_groups[0]['lr'] = learing_rate_disc_scheduler() 
		

        ####### Update Disc Network ####### 
        # Maximize log (D(x) + log (1 - D(G(z))) #
        # Train with all-real batch #
        
        label = torch.full( (data.size(1),1), real_label)
        if args.cuda:
            label = label.cuda()

	#get the embeddings from the generator network of the real 
        data_emb = model_gen.forward_sent(data,msgs,args.gumbel_temp,only_embedding=True)
        real_out = model_disc(data_emb)

        errD_real = criterion(real_out,label)
        errD_real.backward()

        # Train with all-fake batch #
        # Generate batch of fake sequence #
        fake_data_emb, fake_one_hot, fake_data_prob = model_gen.forward_sent(data,msgs,args.gumbel_temp)
        msg_out = model_gen.forward_msg_decode(fake_data_emb)
        # Classify all batch with the discriminator #
        fake_out = model_disc(fake_data_emb.detach())
        label.fill_(fake_label)
        errD_fake = criterion(fake_out,label)
        errD_fake.backward()
        # add the gradients #
        errD = errD_real + errD_fake
        # update the discriminator #
        if batch % args.discr_interval == 0 and batch > 0:
            optimizer_disc.step()

        ####### Update Generator Network #######
        # Maximize log(D(G(z)))
        # For the generator loss the labels are real #
        label.fill_(real_label) 
        # Classify with the updated discriminator #
        fake_out2 = model_disc(fake_data_emb)
        errG_disc = criterion(fake_out2,label)
        errG_msg = criterion(msg_out,msgs)
        errG_reconst = criterion_reconst(fake_data_prob,data.view(-1))
        total_loss_reconst += errG_reconst.data

        if args.use_semantic_loss: 
            # Compute sentence embedding #
            orig_sent_emb = modelSentEncoder.forward_encode_nopad(data)
            fake_sent_emb = modelSentEncoder.forward_encode_nopad(fake_one_hot, one_hot=True)
            errG_sem = criterion_sem(orig_sent_emb,fake_sent_emb) 
            errG = args.gen_weight*errG_disc + args.msg_weight*errG_msg + args.sem_weight*errG_sem + args.reconst_weight*errG_reconst
            sem_losses.append(errG_sem.item())
            total_loss_sem += errG_sem.item()
        else:
            errG = args.gen_weight*errG_disc + args.msg_weight*errG_msg + args.reconst_weight*errG_reconst

        if args.use_lm_loss:
            lm_targets = fake_one_hot[1:fake_one_hot.size(0)]
            lm_targets = torch.argmax(lm_targets,dim=-1)
            lm_targets = lm_targets.view(lm_targets.size(0)*lm_targets.size(1),)
            lm_inputs = fake_one_hot[0:fake_one_hot.size(0)-1]
            lm_out,hidden = langModel(lm_inputs,hidden, decode=True,one_hot=True)
            lm_loss = criterion_lm(lm_out,lm_targets)
            errG = errG + args.lm_weight*lm_loss
            total_loss_lm += lm_loss.item()		

        errG.backward()
        # update the generator #
        optimizer_gen.step()

        # save losses #
        G_losses.append(errG_disc.item())
        msg_losses.append(errG_msg.item())
        D_losses.append(errD.item())

        total_loss_reconst += errG_reconst.data
        total_loss_gen += errG_disc.item()
        total_loss_msg += errG_msg.item()
        total_loss_disc += errD.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_gen = total_loss_gen / args.log_interval
            cur_loss_disc = total_loss_disc / args.log_interval
            cur_loss_msg = total_loss_msg / args.log_interval
            cur_loss_sem = total_loss_sem / args.log_interval
            cur_loss_reconst = total_loss_reconst / args.log_interval
            cur_loss_lm = total_loss_lm / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | gen lr {:05.5f} | disc lr {:05.5f} | ms/batch {:5.2f} | '
                    'gen loss {:5.2f} | disc loss {:5.2f} | msg loss {:5.2f} | sem loss {:5.2f} | reconst loss {:5.4f} | lm loss {:5.4f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer_gen.param_groups[0]['lr'], optimizer_disc.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss_gen, cur_loss_disc, cur_loss_msg, cur_loss_sem, cur_loss_reconst, cur_loss_lm))
            total_loss_gen = 0
            total_loss_msg = 0
            total_loss_disc = 0
            total_loss_sem = 0
            total_loss_reconst = 0
            total_loss_lm = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        global step_num, discr_step_num
        step_num += 1
        discr_step_num += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []
G_losses = []
D_losses = []
msg_losses = []
sem_losses = []
stored_loss = 100000000
stored_loss_msg = 100000000
stored_loss_text = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_gen = None
    optimizer_disc = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer_gen = torch.optim.SGD(params_gen, lr=args.lr, weight_decay=args.wdecay)
        optimizer_disc = torch.optim.SGD(params_disc, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer_gen = torch.optim.Adam(params_gen, lr=learing_rate_scheduler() if args.scheduler else args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)
        optimizer_disc = torch.optim.Adam(params_disc, lr=learing_rate_disc_scheduler() if args.scheduler else args.disc_lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer_gen.param_groups[0]:
            tmp = {}
            for prm in model_gen.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer_gen.state[prm]['ax'].clone()

            tmp_disc = {}
            for prm in model_disc.parameters():
                tmp_disc[prm] = prm.data.clone()
                prm.data = optimizer_disc.state[prm]['ax'].clone()

            val_loss_reconst2, val_loss_gen2, val_loss_msg2, val_loss_sem2, val_loss_lm2, val_loss_disc2 = evaluate(val_data, eval_batch_size)
            val_loss_gen_tot2 = val_loss_msg2 + val_loss_sem2 +  val_loss_reconst2 + val_loss_gen2 + val_loss_lm2 

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | val gen loss {:5.2f} | '
                'val disc loss {:5.2f} | val msg loss {:5.2f} | val sem loss {:5.2f} | val reconst loss {:5.2f} | val lm loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_gen2, val_loss_disc2, val_loss_msg2, val_loss_sem2, val_loss_reconst2, val_loss_lm2))
            print('-' * 89)
            log_file_loss_val.write(str(val_loss_gen2) + ', '+ str(val_loss_disc2) + ', '+ str(val_loss_msg2) + ', '+ str(val_loss_sem2) + ', '+ str(val_loss_reconst2) + ', '+ str(val_loss_lm2) + '\n')
            log_file_loss_val.flush()


            if val_loss_gen_tot2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss_gen_tot2

            for prm in model_gen.parameters():
                prm.data = tmp[prm].clone()

            for prm in model_disc.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss_reconst, val_loss_gen, val_loss_msg, val_loss_sem, val_loss_lm, val_loss_disc = evaluate(val_data, eval_batch_size)
            val_loss_gen_tot =  val_loss_msg + val_loss_sem + val_loss_reconst + val_loss_gen + val_loss_lm
            val_loss_text =   val_loss_sem  + val_loss_reconst + val_loss_lm
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | val gen loss {:5.2f} | '
                'val disc loss {:5.2f} | val msg loss {:5.2f} | val sem loss {:5.2f} | val reconst loss {:5.2f} | val lm loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_gen, val_loss_disc, val_loss_msg, val_loss_sem, val_loss_reconst, val_loss_lm))
            print('-' * 89)
            log_file_loss_val.write(str(val_loss_gen) + ', '+ str(val_loss_disc) + ', '+ str(val_loss_msg) + ', '+ str(val_loss_sem) + ', '+ str(val_loss_reconst) + ', '+ str(val_loss_lm) + '\n')
            log_file_loss_val.flush()

            if val_loss_gen_tot < stored_loss:
                model_save(args.save)
                print('Saving model (new best generator validation)')
                stored_loss = val_loss_gen_tot
            if val_loss_msg < stored_loss_msg:
                model_save(args.save+'_msg')
                print('Saving model (new best msg validation)')
                stored_loss_msg = val_loss_msg
            if val_loss_text < stored_loss_text:
                model_save(args.save+'_reconst')
                print('Saving model (new best reconstruct validation)')
                stored_loss_text = val_loss_text
            if epoch % args.save_interval == 0:
                model_save(args.save+'_interval')
                print('Saving model (intervals)')

            if args.optimizer == 'sgd' and 't0' not in optimizer_gen.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer_gen = torch.optim.ASGD(model_gen.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                optimizer_disc = torch.optim.ASGD(model_disc.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if args.optimizer == 'sgd' and epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer_gen.param_groups[0]['lr'] /= 10.
                optimizer_disc.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss_gen_tot)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_gen = model_gen.cuda()
    model_disc = model_disc.cuda()
    criterion = criterion.cuda()
    criterion_reconst = criterion_reconst.cuda()
    if args.use_semantic_loss:
        criterion_sem = criterion_sem.cuda()
        modelSentEncoder = modelSentEncoder.cuda()


# Run on test data.
test_loss_reconst, test_loss_gen, test_loss_msg, test_loss_sem, test_loss_lm, test_loss_disc = evaluate(test_data, test_batch_size)

print('-' * 89)
print('| End of training | test gen loss {:5.2f} | test disc loss {:5.2f} | test msg loss {:5.2f} | test sem loss {:5.2f} | test reconst loss {:5.2f} | test lm loss {:5.2f}'.format(test_loss_gen, test_loss_disc, test_loss_msg, test_loss_sem, test_loss_reconst, test_loss_lm))
print('-' * 89)

