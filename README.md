# Adversarial Watermarking Transformer (AWT) #
Code for the paper: Adversarial Watermarking Transformer: Towards Tracing Text Provenance with Data Hiding

## Abstract ## 
Recent advances in natural language generation have introduced powerful language models with high-quality output text. However, this raises concerns about the potential misuse of such models for malicious purposes. In this paper, we study natural language watermarking as a defense to help better mark and trace the provenance of text. We introduce the Adversarial Watermarking Transformer (AWT) with a jointly trained encoder-decoder and adversarial training that, given an input text and a binary message, generates an output text that is unobtrusively encoded with the given message. We further study different training and inference strategies to achieve minimal changes to the semantics and correctness of the input text. AWT is the first end-to-end model to hide data in text by automatically learning -without ground truth- word substitutions
along with their locations in order to encode the message. We empirically show that our model is effective in largely preserving text utility and decoding the watermark while hiding its presence against adversaries. Additionally, we demonstrate that our method is robust against a range of attacks.

- - -

## Enviroment ##

- Python 3.7.6
- PyTorch 1.2.0
- To set it up: 
```javascript
conda env create --name awt --file=environment.yml
```
- - -

## Requirements ##

- Model checkpt of InferSent:
	- get the model infersent2.pkl from: https://github.com/facebookresearch/InferSent, place it in 'encoder' directory, or change the argument 'infersent_path' in 'main_train.py' accordingly
  
	- Download GloVe following the instructions in: https://github.com/facebookresearch/InferSent, place it in 'encoder/GloVe' directory, or change the argument 'glove_path' in 'main_train.py' accordingly
  
- Model checkpt of AWD LSTM LM:
	- Download our trained checkpt
  
- Model checkpt of SBERT:
	- Follow instructions from: https://github.com/UKPLab/sentence-transformers
- - -

## Dataset ##

- You will need the WikiText-2 (WT2) dataset. Follow the instructions in: https://github.com/salesforce/awd-lstm-lm to download it

- - -

## Training AWT ##
- Phase 1 of training AWT
```javascript
python main_train.py --msg_len 4 --data data/wikitext-2 --batch_size 80  --epochs 200 --save WT2_mt_noft --optimizer adam --fixed_length 1 --bptt 80 --use_lm_loss 0 --use_semantic_loss 0  --discr_interval 1 --msg_weight 5 --gen_weight 1.5 --reconst_weight 1.5 --scheduler 1
```
- Phase 2 of training AWT
```javascript
python main_train.py --msg_len 4 --data data/wikitext-2 --batch_size 80  --epochs 200 --save WT2_mt_full --optimizer adam --fixed_length 0 --bptt 80  --discr_interval 3 --msg_weight 6 --gen_weight 1 --reconst_weight 2 --scheduler 1 --shared_encoder 1 --use_semantic_loss 1 --sem_weight 6 --resume WT2_mt_noft --use_lm_loss 1 --lm_weight 1.3
```
- - -

## Evaluating effectivness ##
- Needs the checkpoints in the current directory 

### Sampling ### 
- selecting best sample based on SBERT:
```javascript
python evaluate_sampling_bert.py --msg_len 4 --data data/wikitext-2 --bptt 80 --msgs_segment [sentences agg. number] --gen_path [model_gen] --disc_path [model_disc] --use_lm_loss 1 --seed 200 --samples_num [num_samples]
```
- selecting the best sample based on LM loss:
```javascript
python evaluate_sampling_lm.py --msg_len 4 --data data/wikitext-2 --bptt 80 --msgs_segment [sentences agg. number]  --gen_path [model_gen] --disc_path [model_disc] --use_lm_loss 1 --seed 200 --samples_num [num_samples]
```
### Selective Encoding ###
- threshold on the increase of the LM loss
- thresholds used in the paper: 0.45, 0.5, 0.53, 0.59, 0.7 (encodes from 75% to 95% of the sentences)
```javascript
python evaluate_selective_lm_threshold.py --msg_len 4 --data data/wikitext-2 --bptt 80 --msgs_segment [sentences agg. number]  --gen_path [model_gen] --disc_path [model_disc] --use_lm_loss 1 --seed 200 --lm_threshold [threshold] --samples_num 1
```

### Averaging ###
```javascript
python evaluate_avg.py --msg_len 4 --data data/wikitext-2 --bptt 80 --gen_path [model_gen] --disc_path [model_disc] --use_lm_loss 1 --seed 200 --samples_num [num_samples] --avg_cycle [number of sentences to avg]
```
- - -

## Evaluating Robustness ##

### DAE ###

#### Training ####
- To train the denosining-autoencoder as in the paper:
```javascript
python main_train_dae.py --data data/wikitext-2 --bptt 80 --pos_drop 0.1 --optimizer adam --save model1 --batch_size 64 --epochs 2000 --dropoute 0.05 --sub_prob 0.1
```
- sub_prob: prob. of substuiting words. 
- dropoute: embedding dropout prob. 

#### Evaluate ####
- Evaluate the DAE on its own on clean data
	- apply noise, denoise, then compare to the original text 
```javascript
python evaluate_denoise_autoenc.py --data data/wikitext-2 --bptt 80 --autoenc_attack_path [dae model name] --use_lm_loss 1 --seed 200 --sub_prob [sub. noise prob.]
```
#### Attack ####
- Run the attack.
	- First sample from AWT, then input to the DAE, then decode the msg 
```javascript
python evaluate_denoise_autoenc_attack_greedy.py --data data/wikitext-2 --bptt 80 --msg_len 4 --msgs_segment [sentences agg. number] --gen_path [awt model_gen]  --disc_path  [awt model_disc] --samples_num [num_samples] --autoenc_attack_path [dae model name] --use_lm_loss 1 --seed 200
```

### Random changes ###

### Re-watermarking ###
- To implement this attack you need to train a second AWT model with different seed


