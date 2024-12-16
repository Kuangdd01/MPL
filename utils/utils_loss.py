# -*- coding: utf-8 -*-
"""
"""
import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from torch import einsum
from einops import rearrange, repeat

from . import debugger as cutils

def calculate_entropy(tensor: torch.Tensor, devices=None):
    #[b,q,k]
    logp = torch.log(tensor + 1e-9)
    return (tensor * logp).sum() 

def exist(x):
    return x is not None


def default(val, d):
    return val if exist(val) else d


def feps(dtype):
    return torch.finfo(dtype).eps


def finf(dtype):
    return torch.finfo(dtype).max


def log_softmax(x, dim=-1):
    maxv = x.amax(dim=dim, keepdim=True)
    x = x - maxv
    x = x - torch.logsumexp(x, dim=dim, keepdim=True)
    return x

#all masked 1
def get_negative_mask(batchsize: int, neg: int = None, device='cpu'):
    """
    only 'neg' negative samples are masked as 1, others are 0.
    """
    if (not exist(neg)) or neg >= batchsize - 1:
        return torch.ones(batchsize, batchsize, dtype=torch.bool, device=device)
    identity = torch.eye(batchsize, batchsize, dtype=torch.bool, device=device)
    mask = identity.clone().detach()
    for st in range(1, neg + 1):
        mask += torch.roll(identity, shifts=-1 * st, dims=0)
    return mask

def get_negative_mask_(batchsize: int, neg: int = None, device='cpu'):
    msk = torch.eye(batchsize, device=device)
    msk = ~msk.bool()
    msk = msk.float()
    return msk


@torch.no_grad()
def top_k_mask(target, k, dim=-1):
    k = min(k, target.shape[dim])
    _, indices = torch.topk(target, k=k, dim=dim)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask.scatter_(dim, indices, 1)
    return mask
    

from flow.inner_similarity import converting
class WSLoss(nn.Module):
    def __init__(self, conf_ema_m=0.99, hard=True, temperature=0.07, base_temperature=1., 
                 no_contrastive=False, neg_num=None, args=None):
        super().__init__()
        self.q = 32
        self.k = 100
        self.conf_ema_m = conf_ema_m
        self.hard = hard
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.no_contrastive = no_contrastive
        self.neg_num = neg_num
        if exist(neg_num):
            assert no_contrastive is False

    
    def forward(self, output, batch_index, topk=None, 
                x_mask=None, update_conf=True, 
                origin_att=None, inner_mask=None,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True,
                pseudo_target=None,
                x_mask_plus=None):
        b, _, q_, k_ = output.shape
        self.q = q_
        self.k = k_
        cutils.breakpoint_if_find_debug_file()
        output = output / self.temperature
        if self.no_contrastive:
            output = einsum('i i q k -> i q k', output)
            logit = log_softmax(output, dim=-1)
        else:
            neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
            neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)  #sim score[b q (b2 k)]
            output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
            neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
            neg_masked_output = log_softmax(neg_masked_output, dim=-1) #[b, q, bk]
            neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
            if x_mask_plus is not None:
                neg_masked_output = neg_masked_output.masked_fill(~x_mask_plus, 0)
            logit = einsum('i i q k -> i q k', neg_masked_output)
            
            cutils.breakpoint_if_nan_or_inf(logit)
        # loss for CL
        if exist(x_mask):
            pseudo_target.masked_fill_(~x_mask, 0)
        if exist(topk):
            mask = top_k_mask(pseudo_target, topk)
            mask = mask & x_mask
            # do it again
            pseudo_target.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        # entropy = calculate_entropy(pseudo_target)
        loss = -(pseudo_target * logit).sum(-1)
        # loss += entropy
        if origin_att is not None:
            fake_loss = converting(inner_mask, origin_att, logit.clone(), logit.device)
            loss = (loss.sum() + 1 * fake_loss) / (phrase_mask.sum() + feps(loss.dtype))
        else:
            loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_target if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }
    def false_negative_forward(self, output, topk=None, 
                x_mask=None, 
                origin_att=None, inner_mask=None,
                return_atten=True,
                return_logit=True,
                return_target=True,
                return_score=True,
                pseudo_target=None):
        
        b, _, q_, k_ = output.shape
        self.q = q_
        self.k = k_
        cutils.breakpoint_if_find_debug_file()
        assert len(pseudo_target.shape) == 4
        
        output = output / self.temperature
        
        neg_mask = get_negative_mask(b, self.neg_num, device=output.device)
        neg_mask = repeat(neg_mask, 'b1 b2 -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)  #sim score[b q (b2 k)]
        output = rearrange(output, 'b1 b2 q k -> b1 q (b2 k)', b1=b, b2=b, q=self.q, k=self.k)
        neg_masked_output = output.masked_fill(~neg_mask, -finf(output.dtype))
        neg_masked_output = log_softmax(neg_masked_output, dim=-1) #[b, q, bk]
        neg_masked_output = rearrange(neg_masked_output, 'b1 q (b2 k) -> b1 b2 q k', b1=b, b2=b, q=self.q, k=self.k)
        batch_logit = pseudo_target * neg_masked_output
        logit = einsum('i i q k -> i q k', neg_masked_output)
        pseudo_tgt_diag = einsum('b b q k -> b q k', pseudo_target)
        pseudo_tgt_diag = pseudo_tgt_diag.softmax(-1)
        cutils.breakpoint_if_nan_or_inf(batch_logit)
        if exist(topk):
            mask = top_k_mask(pseudo_tgt_diag, topk)
            mask = mask & x_mask
            # do it again
            pseudo_tgt_diag.masked_fill_(~mask, 0)
        else:
            mask = x_mask 
        phrase_mask = mask[:, :, 0]
        cutils.breakpoint_if_nan_or_inf(pseudo_target)
        loss = -batch_logit
        loss = loss.sum() / (phrase_mask.sum() + feps(loss.dtype))
        loss = loss * self.base_temperature
        cutils.breakpoint_if_nan_or_inf(loss)
        # update confidence
        with torch.no_grad():
            conf_logit = logit
            if exist(x_mask):
                conf_logit.masked_fill_(~x_mask, -finf(conf_logit.dtype))
                conf = conf_logit.softmax(dim=-1)
                conf.masked_fill_(~x_mask, 0)
            else:
                conf = conf_logit.softmax(dim=-1)
            cutils.breakpoint_if_nan_or_inf(conf)
        return loss, {
            'loss': loss,
            'atten': output if return_atten else None,
            'logit': logit if return_logit else None,
            'target': pseudo_tgt_diag if return_target else None,
            'score': conf if return_score else None,
            'temperature': self.temperature
        }