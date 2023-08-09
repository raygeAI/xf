# -*- coding: utf-8 -*-
"""
    引用module 中的模块构建TabTransformer 表格数据处理， 官方实现 https://github.com/lucidrains/tab-transformer-pytorch，对相关代码做了较大改造
"""
import torch
from torch import nn
from tab_transformer.module import Transformer, MLP


# TabTransformer 表格数据处理
class TabTransformer(nn.Module):
    """
        TabTransformer: Tabular Data Modeling Using Contextual Embeddings 依据论文实现
    """

    def __init__(
            self,
            category_count,
            category_columns,
            continuous_columns,
            embedding_dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden=(4, 2),
            mlp_act=nn.ReLU(),
            attn_dropout=0,
            ff_dropout=0.0015,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, category_count.values())), "number of each category must be positive"
        assert len(category_count) + len(continuous_columns) > 0, "input shape must not be null"
        self.device = device

        # 离散值列数
        self.num_categories = len(category_count)
        # 连续值列数
        self.num_continuous = len(continuous_columns)

        if self.num_continuous > 0:
            self.norm = nn.LayerNorm(self.num_continuous)

        # transformer
        self.transformer = Transformer(
            category_count=category_count,
            category_columns=category_columns,
            embedding_dim=embedding_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        input_size = (embedding_dim * self.num_categories) + self.num_continuous
        l = input_size // (self.num_continuous + self.num_categories)
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden))
        all_dimensions = [input_size, * hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

        # 统计模型参数
        self.param_stat()

    def param_stat(self):
        n_params = sum(p.numel() for p in self.parameters())
        print("number of model parameters: %.4fM" % (n_params / 1e6,))

    def forward(self, x: torch.Tensor, return_attn: bool=False):
        x_category = x["category_data"].to(self.device)
        x_continuous = x["num_data"].to(self.device)
        # 对离散值维度进行断言
        assert x_category.shape[-1] == self.num_categories, \
            f'you must pass in {self.num_categories} values for your categories input'
        # 对连续值维度进行断言判断
        assert x_continuous.shape[1] == self.num_continuous, \
            f'you must pass in {self.num_continuous} values for your continuous input'

        # 存放所有数据承接
        x_all_features = []
        if self.num_categories > 0:
            x, attention_weight = self.transformer(x_category, return_attn=True)
            # 依据框架图进行押平
            flat_category_feature = x.flatten(1)
            x_all_features.append(flat_category_feature)

        if self.num_continuous > 0:
            # 在输入之前对连续数值进行std 标准化
            normed_continuous = self.norm(x_continuous)
            x_all_features.append(normed_continuous)

        x = torch.cat(x_all_features, dim=-1)
        # 这里是一个多层 mlp;
        out = self.mlp(x)

        if not return_attn:
            return out

        return out, attention_weight


# 算法处理
if __name__ == '__main__':
    import torch
    torch.manual_seed(109)
    model = TabTransformer(
        category_count={"a": 10, "b": 5, "c": 6, "d": 5, "e": 8},
        category_columns=["a", "b", "c", "d", "e"],
        continuous_columns=["a", "b", "c"],  # number of continuous values
        embedding_dim=32,  # dimension, paper set at 32
        dim_out=1,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        dim_head=16,
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_categ = torch.randint(0, 5, (2, 5))  # category values, from 0 - max
    x_continuous = torch.randn(2, 3)  # assume continuous values are already normalized individually
    x = {
        "num_data": x_continuous,
        "category_data": x_categ
    }
    model = model.to(device)
    pred = model(x)  # (1, 1)
    print(pred)
