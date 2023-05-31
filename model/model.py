import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, ):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.cv1 = nn.Conv1d(num_hiddens, num_hiddens * 2, (3,), (2,), 1)
        self.bn1 = nn.BatchNorm1d(num_hiddens * 2)
        self.attention2 = MultiHeadAttention(
            num_hiddens * 2, num_hiddens * 2, num_hiddens * 2, num_hiddens * 2, num_heads, dropout,
            use_bias)
        self.ffn2 = PositionWiseFFN(
            num_hiddens * 2, ffn_num_hiddens * 2, num_hiddens * 2)
        self.addnorm3 = AddNorm(num_hiddens * 2, dropout)
        self.addnorm4 = AddNorm(num_hiddens * 2, dropout)
        self.cv2 = nn.Conv1d(num_hiddens * 2, num_hiddens, (3,), (2,), 1)
        self.bn2 = nn.BatchNorm1d(num_hiddens)
        self.attention3 = MultiHeadAttention(
            num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, dropout,
            use_bias)
        self.ffn3 = PositionWiseFFN(
            num_hiddens, ffn_num_hiddens, num_hiddens)
        self.addnorm5 = AddNorm(num_hiddens, dropout)
        self.addnorm6 = AddNorm(num_hiddens, dropout)
        self.num_heads = num_heads
        self.relu = nn.ReLU()

    def forward(self, X):
        X1 = self.addnorm1(X, self.attention1(X, X, X))
        X2 = self.addnorm2(X1, self.ffn(X1))

        X3 = X2.permute(0, 2, 1)
        X3 = self.cv1(X3)
        X3 = self.bn1(X3)
        X3 = self.relu(X3)
        X3 = X3.permute(0, 2, 1)
        X4 = self.addnorm3(X3, self.attention2(X3, X3, X3))
        X5 = self.addnorm4(X4, self.ffn2(X4))

        X5 = X5.permute(0, 2, 1)
        X5 = self.cv2(X5)
        X5 = self.bn2(X5)
        X5 = self.relu(X5)
        X5 = X5.permute(0, 2, 1)
        X6 = self.addnorm5(X5, self.attention3(X5, X5, X5))
        X7 = self.addnorm6(X6, self.ffn3(X6))

        return X7


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, dropout, **kwargs):
        super(BiRNN, self).__init__(**kwargs)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True, dropout=dropout)
        self.attention = EncoderBlock(512, 512, 512, 512,
                                      512, 512, 1024,
                                      8, dropout)
        self.decoder1 = nn.Linear(2 * 512, 128)
        self.decoder2 = nn.Linear(128, 2)
        self.flaten = nn.Flatten()
        self.bn = nn.BatchNorm1d(128)
        self.net1 = nn.Sequential(nn.Conv1d(1, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
                                  nn.Dropout(0.2),
                                  nn.Conv1d(32, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
                                  nn.Dropout(0.2),
                                  nn.Conv1d(128, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, inputs, xs):
        ls = self.net1(xs)
        ls = ls.permute(0, 2, 1)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        encoding = encoding.reshape((encoding.shape[0], 4, -1))
        encoding = torch.cat((encoding, ls), dim=1)
        encoding = self.attention(encoding)
        encoding = self.flaten(encoding)
        outs = self.decoder1(encoding)
        outs = self.bn(outs)
        outs = self.decoder2(outs)

        return outs


if __name__ == '__main__':
    data = torch.ones(4, 500, dtype=torch.long)
    data2 = torch.randn((4, 20)).unsqueeze(1)
    print(data2.shape)
    net = BiRNN(21, 32, 512, 2, 0.5)
    print(net(data, data2).shape)
