import re
import numpy as np
import collections
from statics import load_data_statis

def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(lines, token='char'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['unk'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.
    Defined in :numref:`sec_machine_translation`"""
    steps = num_steps // 2
    if len(line) > num_steps:
        return line[0:steps] + line[-steps:]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


# U, Z, O, B   ->   X
def load_data(num_steps):
    X_data = np.load('./data/X_data.npy')
    y_data = np.load('./data/y_data.npy')

    X = [re.sub('[UZOB]', 'X', line).strip() for line in X_data]
    X = tokenize(X)
    # print(len(X[0]))
    vocab = Vocab(X)
    lines = [vocab[l] for l in X]

    final_data = []
    for line in lines:

        line = truncate_pad(line, num_steps, 0)
        final_data.append(line)
    # print(np.sum(y_data))
    return np.array(final_data), np.array(y_data)


def load_data_PeNGaRoo(num_steps):

    X_train_data = np.load('../data/X_train.npy')
    y_train_data = np.load('../data/y_train.npy')
    X_test_data = np.load('../data/X_test.npy')
    y_test_data = np.load('../data/y_test.npy')
    X_train_data = [re.sub('[UZOB]', 'X', line).strip() for line in X_train_data]
    X_train_data = tokenize(X_train_data)

    vocab = Vocab(X_train_data)
    # print(vocab.token_to_idx.items())
    lines = [vocab[l] for l in X_train_data]

    final_data = []
    final_data_s = []
    for line in lines:
        # print(line)
        lins = load_data_statis(line)
        final_data_s.append(lins)
        line = truncate_pad(line, num_steps, 0)
        final_data.append(line)
    X_test_data = tokenize(X_test_data)
    test_lines = [vocab[l] for l in X_test_data]

    test_final_data = []
    test_final_data_s = []
    for line in test_lines:
        # print(line)
        lins = load_data_statis(line)
        test_final_data_s.append(lins)
        line = truncate_pad(line, num_steps, 0)
        test_final_data.append(line)


    # return final_data, np.array(y_train_data), test_final_data, y_test_data,vocab
    return np.array(final_data),np.array(final_data_s), np.array(y_train_data), test_final_data,test_final_data_s, y_test_data,vocab

#
load_data_PeNGaRoo(500)
