# -*- coding: utf-8 -*-
"""
    TabularDataSet 输入表格训练数据集合定义
"""
import random
from typing import List
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


# TabularDataSet 表格数据输入定义
class TabularDataSet(Dataset):
    """
    TabularDataSet 表格数据集合 dataSet
    """

    def __init__(
            self,
            dataframe: DataFrame,
            category_columns: List[str],
            num_columns: List[str],
            output_columns: List[str],
            is_test:bool = False
    ):
        super().__init__()
        self.data = dataframe
        self.category_data = dataframe[category_columns].values.astype(np.int64)
        self.num_data = dataframe[num_columns].values.astype(np.float32)
        if not is_test:
            self.labels = dataframe[output_columns].values.astype(np.float32)
        else:
            self.labels = np.empty((self.data.shape[0], len(output_columns)))
        print(f"data set size: {self.data.shape[0]}")

    def __getitem__(self, index):
        return {
            "category_data": self.category_data[index],
            "num_data": self.num_data[index],
            "labels": self.labels[index]
        }

    # 数据集合的长度
    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    df = pd.DataFrame({
        "num1": [random.randint(0, 100) for i in range(5)],
        "num2": np.random.random(5),
        "cate1": [random.randint(0, 100) for i in range(5)],
        "cate2": [random.randint(0, 100) for i in range(5)]
    })
    dataset = TabularDataSet(dataframe=df, category_columns=["cate1", "cate2"], num_columns=["num1", "num2"])
