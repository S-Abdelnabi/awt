import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import data_disc_lm
import model_discriminator
import model_discriminator_lstm


from sklearn.metrics import f1_score

from utils import batchify, get_batch_different, generate_msgs, repackage_hidden, get_batch_no_msg

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
				
parser.add_argument('--bptt', type=int, default=80,
                    help='sequence length')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

randomhash = ''.join(str(time.time()).split('.'))

parser.add_argument('--disc_path', type=str,  default=randomhash+'.pt',
                    help='path to the classifier')
					
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


import os
import hashlib

print('Producing dataset...')
corpus = data_disc_lm.Corpus('data_classifier')

eval_batch_size = 1
test_batch_size = 1

val_fake_data = batchify(corpus.valid_fake, eval_batch_size, args)
test_fake_data = batchify(corpus.test_fake, test_batch_size, args)

val_real_data = batchify(corpus.valid_real, eval_batch_size, args)
test_real_data = batchify(corpus.test_real, test_batch_size, args)

ntokens = len(corpus.dictionary)
print(ntokens)
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

### conventions for real and fake labels during training ###
real_label = 1
fake_label = 0

def evaluate(data_source_real, data_source_fake, batch_size=10):
    # Turn on evaluation mode which disables dropout.
	
    model_disc.eval()	
    total_loss_disc = 0
    sig = nn.Sigmoid()
	
    ntokens = len(corpus.dictionary)
    batches_count = 0
    y_out = []
    y_label = []
    real_correct = 0
    fake_correct = 0
	
    for i in range(0, data_source_real.size(0) - args.bptt, args.bptt):
        data_real, targets_real = get_batch_no_msg(data_source_real, i, args, evaluation=True)
        data_fake, targets_fake = get_batch_no_msg(data_source_fake, i, args, evaluation=True)
		
        #run the real and fake to the disc model 
        real_out = model_disc.forward(data_real)
        fake_out = model_disc.forward(data_fake)
		
        label = torch.full( (data_real.size(1),1), real_label)
        if args.cuda:
            label = label.cuda()
        errD_real = criterion(real_out,label)
        real_out_label = torch.round(sig(real_out))
        real_correct = real_correct + np.count_nonzero(np.equal(label.detach().cpu().numpy().astype(int),real_out_label.detach().cpu().numpy().astype(int))==True) 
        y_label.append(label.detach().cpu().numpy().astype(int)[0,0])
        y_out.append(real_out_label.detach().cpu().numpy().astype(int)[0,0])
		#get prediction (and the loss) of the discriminator on the fake sequence.

        label.fill_(fake_label)
        errD_fake = criterion(fake_out,label)
        errD = errD_real + errD_fake
        fake_out_label = torch.round(sig(fake_out))
        fake_correct = fake_correct + np.count_nonzero(np.equal(label.detach().cpu().numpy().astype(int),fake_out_label.detach().cpu().numpy().astype(int))==True)
        y_label.append(label.detach().cpu().numpy().astype(int)[0,0])
        y_out.append(fake_out_label.detach().cpu().numpy().astype(int)[0,0])

		
        total_loss_disc +=  errD.data
        batches_count = batches_count + 1
    Fscore = f1_score(y_label,y_out)
    return total_loss_disc.item() / batches_count, fake_correct/batches_count, real_correct/batches_count, Fscore

	# Load the best saved model.
with open(args.disc_path, 'rb') as f:
    model_disc, _, _= torch.load(f)
criterion =  nn.BCEWithLogitsLoss()
#print(model_disc)


if args.cuda:
    model_disc.cuda()
    criterion.cuda()

val_disc_loss, val_correct_fake, val_correct_real, val_Fscore  = evaluate(val_real_data, val_fake_data, eval_batch_size)

print('-' * 150)
print('| validation | disc loss {:5.2f} | fake accuracy {:5.2f} | real accuracy {:5.2f} | F1 score {:.5f}'.format(val_disc_loss, val_correct_fake*100,val_correct_real*100,val_Fscore))
print('-' * 150)



test_disc_loss, test_correct_fake, test_correct_real, test_Fscore  = evaluate(test_real_data, test_fake_data, test_batch_size)

print('-' * 150)
print('| Test | disc loss {:5.2f} | fake accuracy {:5.2f} | real accuracy {:5.2f} | F1 score {:.5f}'.format(test_disc_loss, test_correct_fake*100,test_correct_real*100,test_Fscore))
print('-' * 150)
