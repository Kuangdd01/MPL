import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import einsum

from .layer import DynamicLSTM, REDUCE


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):
    def __init__(self, query_dim=300, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Dual(nn.Module):
    def __init__(self, vectors, v_feature_dropout_prob=0.1, dropout_prob=0.1, scale=12, reduce_method='sum', emb_dim=300, feature_dim=2048):
        super(Dual, self).__init__()
        self.eps = 1e-5
        self.scale = scale
        self.wv = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.linear_f = nn.Linear(feature_dim, emb_dim)

        self.linear_rnn = Attention(emb_dim)
        self.linear_rnn.to_out.weight.data = torch.zeros_like(self.linear_rnn.to_out.weight)

        self.linear_p = nn.Linear(emb_dim, emb_dim)
        self.linear_mini = nn.Linear(emb_dim, emb_dim)

        self.v_dropout = nn.Dropout(v_feature_dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

        self.linear_p.weight.data = torch.eye(emb_dim)
        self.linear_f.weight.data = torch.zeros(emb_dim, feature_dim)

        self.reduce_func = REDUCE[reduce_method]

    def encode_k(self, label, feature):
        k_emb = self.wv(label)
        feature = self.v_dropout(feature)
        f_emb = self.linear_f(feature)
        k_emb += f_emb
        k_emb = self.dropout(k_emb)
        return k_emb

    def encode_p(self, caption_id, phrase_span_mask, length):
        caption = self.wv(caption_id)
        mask = len2mask(length, caption_id.shape).to(caption_id.device)
        hidden = self.linear_rnn(caption, mask=mask)
        hidden = caption + hidden
        p_emb = self.reduce_func(phrase_span_mask, hidden)
        p_emb = p_emb / self.scale
        p_emb = self.linear_p(p_emb) + self.eps * self.linear_mini(p_emb)
        p_emb = self.dropout(p_emb)
        return p_emb

    def forward(self, caption_id, phrase_span_mask, length, label, feature):
        p_emb = self.encode_p(caption_id, phrase_span_mask, length)
        k_emb = self.encode_k(label, feature)
        return p_emb, k_emb


def len2mask(length, shape):
    batch, max_length = shape
    mask = torch.lt(torch.arange(max_length, device=length.device).unsqueeze(0).expand(shape), length.unsqueeze(1))
    return mask
