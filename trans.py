import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LayerNorm(nn.Module):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g + x + self.b


class Conv1D(nn.Module):
    """Convolution one dimension."""

    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        # for 1x1 conv
        if rf == 1:
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    """Main class Attention"""

    def __init__(self, nx, n_ctx, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        # tensor2tensor
        assert n_state % 12 == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = 12
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # Mask
        w = w * self.b + -1e9 * (1 - self.b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state):  # 768 or 1024?
        super(MLP, self).__init__()
        nx = 768
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = nn.ReLU
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, scale=False):
        super(Block, self).__init__()
        nx = 768
        self.attn = Attention(nx, n_ctx, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class Transformer(nn.Module):
    """Transformer openAI"""

    def __init__(self, vocab=10_000, n_ctx=512):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, 768)
        self.drop = nn.Dropout(0.1)
        block = Block(n_ctx, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(12)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)

        h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class LMHead(nn.Module):
    """ Language Model """

    def __init__(self, model, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = 768
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight
        self.trunc_and_reshape = trunc_and_reshape

    def forward(self, h):
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) \
            if self.trunc_and_reshape else h
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class Model(nn.Module):
    def __init__(self, vocab=40000, n_ctx=512, return_prob=False):
        super(Model, self).__init__()
        self.transformer = Transformer(vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, trunc_and_reshape=False)
        self.return_prob = return_prob
        if self.return_prob:
            pos_emb_m = torch.zeros(1, 1, vocab)
            pos_emb_m[:, :, -n_ctx] = -1e12
            self.register_buffer('pos_emb_m', pos_emb_m)

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_prob:
            lm_logits = F.softmax(lm_logits + self.pos_emb_m, dim=-1)
        return lm_logits
