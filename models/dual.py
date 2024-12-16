import torch
import torch.nn as nn
from .layer import DynamicLSTM, REDUCE


class Dual(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, reduce_method='mean', emb_dim=300, feature_dim=2048):
        super(Dual, self).__init__()
        self.eps = 1e-5
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.linear_f = nn.Linear(feature_dim, emb_dim)

        self.rnn = DynamicLSTM(emb_dim, emb_dim, num_layers=1, bias=True, batch_first=True, dropout=0.,
                               bidirectional=True, only_use_last_hidden_state=False, rnn_type='LSTM')
        self.linear_rnn = nn.Linear(emb_dim * 2, emb_dim)
        self.relu = nn.ReLU()
        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear_p.weight.data = torch.eye(emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

        self.reduce_func = REDUCE[reduce_method]

    def encode_k(self, label, feature):
        # 加入label
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        f_emb = self.linear_f(feature)
        k_emb += f_emb
        k_emb = self.dropout(k_emb)
        return k_emb

    def encode_p(self, caption_id, phrase_span_mask, length):
        caption = self.wv(caption_id)
        hidden, _ = self.rnn(caption, length)
        hidden = self.relu(self.linear_rnn(hidden))
        p_emb = self.reduce_func(phrase_span_mask, hidden)
        p_emb = self.linear_p(p_emb) + self.eps * self.linear_mini(p_emb)
        p_emb = self.dropout(p_emb)
        return p_emb

    def forward(self, caption_id, phrase_span_mask, length, label, feature):
        p_emb = self.encode_p(caption_id, phrase_span_mask, length)
        k_emb = self.encode_k(label, feature)
        return p_emb, k_emb
