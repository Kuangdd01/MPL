import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from torch import einsum

from .layer import DynamicLSTM


def finf(dtype):
    return torch.finfo(dtype).max


class MatRNN(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, emb_dim=300, feature_dim=2048):
        super(MatRNN, self).__init__()
        self.eps = 1e-5
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.linear_f = nn.Linear(feature_dim, emb_dim)

        self.rnn = DynamicLSTM(emb_dim, emb_dim, num_layers=1, bias=True, batch_first=True, dropout=0.,
                               bidirectional=True, only_use_last_hidden_state=False, rnn_type='LSTM')
        self.linear_rnn = nn.Linear(emb_dim * 2, emb_dim)
        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear_rnn.weight.data = torch.zeros(emb_dim, emb_dim * 2)
        self.linear_rnn.bias.data = torch.zeros(emb_dim)
        self.linear_p.weight.data = torch.eye(emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

    def encode_k(self, label, feature):
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        f_emb = self.linear_f(feature)
        k_emb += f_emb
        k_emb = self.dropout(k_emb)
        return k_emb

    def encode_p(self, caption_id, phrase_span_mask, length, k_emb):
        caption = self.wv(caption_id)
        rnn_feature, _ = self.rnn(caption, length)
        rnn_feature = self.linear_rnn(rnn_feature)
        hidden = caption + rnn_feature

        p_emb = self.cross_atten(span_mask=phrase_span_mask, hidden=hidden, k_emb=k_emb)

        p_emb = self.linear_p(p_emb) + self.eps * self.linear_mini(p_emb)
        p_emb = self.dropout(p_emb)
        return p_emb

    def forward(self, caption_id, phrase_span_mask, length, label, feature):
        k_emb = self.encode_k(label, feature)
        p_emb = self.encode_p(caption_id, phrase_span_mask, length, k_emb=k_emb)
        return p_emb, k_emb

    def cross_atten(self, span_mask, hidden, k_emb):
        b, q, s = span_mask.shape  # b q s
        _, _, d = hidden.shape  # b s d
        _, k, _ = k_emb.shape  # b k d

        att = einsum('b s d, b k d -> b s k', hidden, k_emb)
        att = att * 1.0 / np.sqrt(d)
        att = torch.softmax(att, dim=-1)  # b s k
        max_att = torch.amax(att, dim=-1)  # b s

        q_max_att = repeat(max_att, 'b s -> b q s', b=b, q=q, s=s)
        q_max_att = q_max_att.clone()
        q_max_att.masked_fill_(~span_mask.bool(), -finf(q_max_att.dtype))
        q_max_norm_att = torch.softmax(q_max_att, dim=-1)

        p_emb = einsum('b q s, b s d -> b q d', q_max_norm_att, hidden)
        return p_emb
