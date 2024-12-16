import torch
import torch.nn as nn
from .layer import DynamicLSTM, REDUCE


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class GatedRNN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.rnn = DynamicLSTM(emb_dim, emb_dim, num_layers=1, bias=True, batch_first=True, dropout=0.,
                               bidirectional=True, only_use_last_hidden_state=False, rnn_type='LSTM')
        self.linear_rnn = nn.Linear(emb_dim * 2, emb_dim)
        self.rnn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(emb_dim)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, length):
        # rnn
        h, _ = self.rnn(x, length)
        x = x + self.rnn_gate.tanh() * self.linear_rnn(h)
        # ff
        x = x + self.ff_gate.tanh() * self.ff(x)
        return x


class Dual(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, reduce_method='mean', emb_dim=300, feature_dim=2048):
        super(Dual, self).__init__()
        self.eps = 1e-5
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)
        # region
        self.linear_f = nn.Linear(feature_dim, emb_dim)
        self.region_gate = nn.Parameter(torch.tensor([0.]))
        # phrase
        self.gated_rnn = GatedRNN(emb_dim)
        self.linear_p = nn.Linear(emb_dim, emb_dim)
        # dropout
        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        # init
        self.linear_p.weight.data = torch.zeros(emb_dim, emb_dim)

        self.reduce_func = REDUCE[reduce_method]

    def encode_k(self, label, feature):
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        k_emb = k_emb + self.region_gate.tanh() * self.linear_f(feature)
        k_emb = self.dropout(k_emb)
        return k_emb

    def encode_p(self, caption_id, phrase_span_mask, length):
        caption = self.wv(caption_id)
        hidden = self.gated_rnn(caption, length)
        p_emb = self.reduce_func(phrase_span_mask, hidden)
        p_emb = p_emb + self.linear_p(p_emb)
        p_emb = self.dropout(p_emb)
        return p_emb

    def forward(self, caption_id, phrase_span_mask, length, label, feature):
        p_emb = self.encode_p(caption_id, phrase_span_mask, length)
        k_emb = self.encode_k(label, feature)
        return p_emb, k_emb
