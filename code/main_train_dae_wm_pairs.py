import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#import data
import data_wm_pairs
import model_denoise_autoenc_attack
import lang_model

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden, get_batch_no_msg

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

parser.add_argument('--pos_drop', type=float, default=0.2,
                    help='dropout applied to input layers (0 = no dropout)')
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
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

parser.add_argument('--msg_len', type=int, default=4,
                    help='The length of the binary message')
parser.add_argument('--msgs_num', type=int, default=3,
                    help='The total number of messages')
					
#transformer arguments
parser.add_argument('--attn_heads', type=int, default=4,
                    help='The number of attention heads in the transformer')
parser.add_argument('--encoding_layers', type=int, default=3,
                    help='The number of encoding layers')


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
					
					
#reconstruction loss
parser.add_argument('--use_reconst_loss', type=int, default=1,
                    help='whether to use language reconstruction loss')
parser.add_argument('--reconst_weight', type=float, default=1,
                    help='The factor multiplied with the reconstruct loss')
					



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
    with open(fn+'_dae_wm_pairs.pt', 'wb') as f:
        torch.save([model_gen, criterion_reconst, optimizer_gen], f)

def model_load(fn):
    global model_gen, criterion_reconst, optimizer_gen
    with open(fn+'_dae_wm_pairs.pt', 'rb') as f:
        model_gen, criterion_reconst, optimizer_gen = torch.load(f,map_location='cpu')
		
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
corpus = data_wm_pairs.Corpus(args.data)
print('Running on watermarked output of AWT_adv')	

eval_batch_size = 1
test_batch_size = 1

train_fake_data = batchify(corpus.train_fake, args.batch_size, args)
val_fake_data = batchify(corpus.valid_fake, eval_batch_size, args)
test_fake_data = batchify(corpus.test_fake, test_batch_size, args)

train_real_data = batchify(corpus.train_real, args.batch_size, args)
val_real_data = batchify(corpus.valid_real, eval_batch_size, args)
test_real_data = batchify(corpus.test_real, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

criterion_reconst = None 

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

## global variable for the number of steps ( batches) ##
step_num = 1

def learing_rate_scheduler():
    d_model = args.emsize
    warm_up = args.warm_up
    lr = np.power(d_model, -1.1) * min(np.power(step_num, -0.5), step_num*np.power(warm_up, -1.5))
    return lr
	
if args.resume:
    all_msgs = np.loadtxt('msgs.txt')
    print('Resuming model ...')
    model_load(args.resume) 
    optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() if args.scheduler else args.lr
else:
    ### generate random msgs ###
    all_msgs = generate_msgs(args)
    model_gen = model_denoise_autoenc_attack.AutoencModel(ntokens, args.emsize, args.encoding_layers, args.pos_drop, args.dropout_transformer, args.dropouti, args.dropoute, args.tied, args.attn_heads)



if args.use_reconst_loss and not criterion_reconst:
    criterion_reconst = nn.CrossEntropyLoss()


if args.cuda:
    model_gen = model_gen.cuda()
    criterion_reconst = criterion_reconst.cuda()
###
params = list(model_gen.parameters()) + list(criterion_reconst.parameters()) 
params_gen = model_gen.parameters()

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

def evaluate(data_source_real, data_source_fake, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model_gen.eval()

    total_loss_reconst = 0
	
    ntokens = len(corpus.dictionary)
    batches_count = 0
    for i in range(0, data_source_real.size(0) - args.bptt, args.bptt):
        data_real, targets_real = get_batch_no_msg(data_source_real, i, args, evaluation=True)
        data_fake, targets_fake = get_batch_no_msg(data_source_fake, i, args, evaluation=True)
        
        out_data_prob = model_gen.forward_sent(data_real,data_fake,args.gumbel_temp)

	#reconstruction loss 
        reconst_loss = criterion_reconst(out_data_prob,data_real.view(-1))
        total_loss_reconst += reconst_loss.data

        batches_count = batches_count + 1

    return total_loss_reconst.item()/batches_count


def train():
    # Turn on training mode which enables dropout.
    total_loss_reconst = 0
	
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0
    fake_batch_itr = 0
    while i < train_real_data.size(0) - 1 - 1:
        if not args.fixed_length:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data_real, targets_real = get_batch_no_msg(train_real_data, i, args, seq_len=seq_len)
            data_fake, targets_fake = get_batch_no_msg(train_fake_data, fake_batch_itr, args, seq_len=seq_len)
        else:
            seq_len = args.bptt
            data_real, targets_real = get_batch_no_msg(train_real_data, i, args, seq_len=None)
            data_fake, targets_fake = get_batch_no_msg(train_fake_data, fake_batch_itr, args, seq_len=None)
            
        model_gen.train()
        optimizer_gen.zero_grad()

        ####### Update lr #######
        if args.scheduler:
            optimizer_gen.param_groups[0]['lr'] = learing_rate_scheduler() 		

        out_data_prob = model_gen.forward_sent(data_real, data_fake, args.gumbel_temp)
	
        errG_reconst = criterion_reconst(out_data_prob,data_real.view(-1))
        total_loss_reconst += errG_reconst.data

        errG = args.reconst_weight*errG_reconst

        errG.backward()
        # update the generator #
        optimizer_gen.step()

        total_loss_reconst += errG_reconst.data
        log_file_loss_train.write(str(errG_reconst.data) + '\n')
        log_file_loss_train.flush()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss_reconst = total_loss_reconst / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | gen lr {:05.8f} | '
                    'reconst loss {:5.4f}'.format(
                epoch, batch, len(train_fake_data) // args.bptt, optimizer_gen.param_groups[0]['lr'], cur_loss_reconst))

            total_loss_reconst = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
        fake_batch_itr += seq_len
        if fake_batch_itr > train_fake_data.size(0)-1-1:
            fake_batch_itr = 0	
            
        global step_num
        step_num += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
stored_loss_msg = 100000000
stored_loss_text = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer_gen = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer_gen = torch.optim.SGD(params_gen, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer_gen = torch.optim.Adam(params_gen, lr=learing_rate_scheduler() if args.scheduler else args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer_gen.param_groups[0]:
            tmp = {}
            for prm in model_gen.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer_gen.state[prm]['ax'].clone()

            val_loss_reconst2 = evaluate(val_real_data, val_fake_data, eval_batch_size)
            val_loss_gen_tot2 = val_loss_reconst2  

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | val reconst loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_reconst2))
            print('-' * 89)
            log_file_loss_val.write(str(val_loss_reconst2) + '\n')
            log_file_loss_val.flush()


            if val_loss_gen_tot2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss_gen_tot2

            for prm in model_gen.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss_reconst = evaluate(val_real_data, val_fake_data, eval_batch_size)
            val_loss_gen_tot =  val_loss_reconst 
            val_loss_text =  val_loss_reconst
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'val reconst loss {:5.2f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss_reconst))
            print('-' * 89)
            log_file_loss_val.write(str(val_loss_reconst) + '\n')
            log_file_loss_val.flush()

            if val_loss_gen_tot < stored_loss:
                model_save(args.save)
                print('Saving model (new best generator validation)')
                stored_loss = val_loss_gen_tot

            if args.optimizer == 'sgd' and 't0' not in optimizer_gen.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer_gen = torch.optim.ASGD(model_gen.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
				
            if args.optimizer == 'sgd' and epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer_gen.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss_gen_tot)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
if args.cuda:
    model_gen = model_gen.cuda()
    criterion_reconst = criterion_reconst.cuda()


# Run on test data.
test_loss_reconst = evaluate(test_real_data, test_fake_data, test_batch_size)

print('-' * 89)
print('| End of training | test reconst loss {:5.2f}'.format(test_loss_reconst))
print('-' * 89)
