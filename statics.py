import re
import numpy as np
import collections


def count_corpus(tokens):
    """Count token frequencies.

    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)




def load_data_statis(X_train_data,is_train=False):
    d = [4, 20, 8, 2, 13, 7, 18, 6, 3, 1, 17, 11, 15, 14, 12, 10, 9, 5, 19, 16]







    corpus = count_corpus(X_train_data)
    l = []
    # print(corpus)
    for j in (d):
        s_l = corpus[j] / len(X_train_data)
        l.append(s_l)
    # else:
    #     statis_tr = []
    #     for i in X_train_data:
    #
    #         corpus = count_corpus(i)
    #         l = []
    #         # print(corpus)
    #         for j in (d):
    #             s_l = corpus[j] / len(i)
    #             l.append(s_l)
    #         statis_tr.append(l)


    return l
