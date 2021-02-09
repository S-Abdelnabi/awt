import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train_real = self.tokenize(os.path.join(path, 'train_real_lm_20samples.txt'))
        self.valid_real = self.tokenize(os.path.join(path, 'valid_real_lm_20samples.txt'))
        self.test_real = self.tokenize(os.path.join(path, 'test_real_lm_20samples.txt'))

        self.train_fake = self.tokenize(os.path.join(path, 'train_fake_lm_20samples.txt'))
        self.valid_fake = self.tokenize(os.path.join(path, 'valid_fake_lm_20samples.txt'))
        self.test_fake = self.tokenize(os.path.join(path, 'test_fake_lm_20samples.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.rstrip().split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.rstrip().split() 
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
