import numpy as np
import torch
import torch.nn as nn


def finf(dtype):
    return torch.finfo(dtype).max


class MATnet(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, emb_dim=300, feature_dim=2048):
        super(MATnet, self).__init__()

        self.eps = 1e-6
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_f = nn.Linear(feature_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.cosSim = nn.CosineSimilarity(dim=-1, eps=self.eps)
        self.crossentropy = nn.CrossEntropyLoss()
        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear_p.weight.data = torch.eye(emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

    def _encode(self, query, label, feature):
        """
        :param query: query phrases [B, queries, words]
        :param label: object labels, predicted by the detector [B, objects]
        :param feature: object features, predicted by the detector [B, objects, feature_dim]

        :return: 	q_emb[B, queries, words, dim] for query word embedding;
                    k_emb[B, objects, dim] for object embedding
        """

        q_emb = self.wv(query)
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        f_emb = self.linear_f(feature)
        k_emb += f_emb
        # k_emb += f_emb + a_emb

        return q_emb, k_emb

    def encode_pk(self, query, label, feature):
        """
        :param query: query phrases [B, queries, words]
        :param label: object labels, predicted by the detector [B, objects]
        :param feature: object features, predicted by the detector [B, objects, feature_dim]

        :return:	p_emb[B, queries, dim] for phrase embedding
                    k_emb[B, objects, dim] for object embedding
        """
        eps = 1e-5

        q_emb, k_emb = self._encode(query, label, feature)  # [B, querys, Q, dim] & [B, K, dim]

        # q_emb [B, querys, Q, dim]
        scale = 1.0 / np.sqrt(k_emb.size(-1))
        att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)
        att = self.softmax(att.mul_(scale))  # [B, querys, Q, K]

        q_max_att = torch.max(att, dim=3).values  # [B, querys, Q]
        q_max_att.masked_fill_(query==0, -finf(q_max_att.dtype))
        q_max_norm_att = self.softmax(q_max_att)
        # attended
        q_max_norm_att = q_max_norm_att / 5
        p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb)  # [B, querys, dim]
        p_emb = self.linear_p(p_emb) + eps * self.linear_mini(p_emb)

        return p_emb, k_emb

    def forward(self, query, label, feature):
        """
        :param query: [B, all_query=32, Q=12] pad with 0
        :param label: [B, K=64] pad with 0
        :param feature: [B, K, feature_dim]
        :return:
        """
        p_emb, k_emb = self.encode_pk(query, label, feature)
        p_emb = self.dropout(p_emb)
        k_emb = self.dropout(k_emb)
        return p_emb, k_emb
