# -*- coding: utf-8 -*-
"""
    TabTransformer 的构造模块;
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# EmbeddingLayer 离散特征映射层
class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, category_counts, category_columns):
        super().__init__()
        self.category_columns = category_columns
        self.col_to_idx = {col: idx for idx, col in enumerate(category_columns)}
        self.embeddings = nn.ModuleDict({col: nn.Embedding(category_counts[col], embedding_dim)
                                         for col in category_columns})

    def forward(self, x):
        embeddings = [torch.unsqueeze(self.embeddings[col](x[:, self.col_to_idx[col]].long()), 1)
                      for col in self.category_columns]
        vectors = torch.cat(embeddings, dim=1)
        return vectors


# Residual 标准化实现
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# PreNorm 前置 layerNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# gelu 激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# FeedForward 前馈网络实现
class FeedForward(nn.Module):
    # scale 为中间这一层扩大多少倍；标准transformer 中 Feed Forward 512 变成4倍，然后下降到512
    def __init__(self, embedding_dim, scale=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * scale * 2, bias=True),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * scale, embedding_dim, bias=True)
        )

    def forward(self, x):
        return self.net(x)


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value
        v = score @ v
        return v, score


# MutilHeadAttention 多头自注意力实现
class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, d_model, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.inner_dim = d_model * n_head
        self.n_head = n_head
        self.to_qkv = nn.Linear(embedding_dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, embedding_dim)
        self.attention = ScaleDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. dot product with weight matrices
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        x, attention = self.attention(q, k, v, mask=None)
        # 添加dropout
        x = self.dropout(x)
        # 4. concat and pass to linear layer
        x = self.concat(x)
        # 5. concat 之后，进入一个线性层
        out = self.to_out(x)
        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


# Transformer 定义，实际只是 transformer encoder
class Transformer(nn.Module):
    def __init__(self,
                 category_count,
                 category_columns,
                 embedding_dim,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout,
                 ff_dropout):
        super().__init__()

        self.embeddings = EmbeddingLayer(
            embedding_dim,
            category_count,
            category_columns)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embedding_dim, MultiHeadAttention(
                    embedding_dim=embedding_dim,
                    d_model=dim_head,
                    n_head=heads,
                    dropout=attn_dropout)),
                PreNorm(embedding_dim, FeedForward(embedding_dim, dropout=ff_dropout))])
            )

    def forward(self, x, return_attn=False):
        x = self.embeddings(x)
        post_softmax_attentions = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attention = attn(x)
            post_softmax_attentions.append(post_softmax_attention)
            x = x + attn_out
            x = ff(x) + x

        # 判断是否返回注意力权重
        if not return_attn:
            return x
        else:
            return x, torch.stack(post_softmax_attentions)


# MLP  层
class MLP(nn.Module):
    def __init__(self, dims, dropout=0.015, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            is_last = ind >= (len(dims_pairs) - 1)
            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# TransformerEncoder 基于pytorch 实现的TransformerEncoderLayer 实现,
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, layers):
        """
        这个是官方TransformerEncoder 的标准实现，和上述的transformer 实现还是有一些区别的；
        官方的encoder 中需要输入x 的embedding 维度和 encoder_layer 中的维度一致；
        而tabTransformer 中对输入的x，构造 K， Q， V 时候，embedding_dim 进行了升高维度；
        根本是 head_dim 并不等于d_model/ n_head:
        内部自己实现的 TransformerEncoder 可以在实验中进行参考试错;
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation=F.gelu)
        self.layers = layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.layers)

    def forward(self, x):
        batch, seq_length, dim = x.shape
        # 先判断输入维度
        assert dim == self.d_model, f"x dim: {dim} is not equal d_model:{self.d_model}"
        x = self.transformer_encoder(x)
        return x


if __name__ == "__main__":
    d_model = 512
    n_head = 4
    dim_feedforward = 512 * 4
    x = torch.rand(4, 3, 512)
    # pytorch 官方 Transformer 实现， 必须三个维度:
    encoder = TransformerEncoder(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, layers=2)
    y = encoder(x)
    print(y)
