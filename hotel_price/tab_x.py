# -*- coding: utf-8 -*-
"""
在xf 酒店价格预测问题上，只有3w 数据，较难达到收敛，容易陷入局部最小，过拟合。需要太多参数优化
"""
from typing import Tuple, Dict
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from pandas import DataFrame

from tab_transformer.tab_transformer import TabTransformer
from tab_transformer.data_set import TabularDataSet
from data_util import get_categorical_count
from data_util import serialization, deserialization
from data_util import read_csv
from data_util import categorical_encoder
from data_util import standard_scaler
from data_util import inverse_transform
from data_util import target_encoding
from util import SetSeed
from metric import cal_error
from log import logger


class TabXConfig:
    FILE = "data/train/train.csv"
    TEST_FILE = "data/train/test.csv"
    MODEL_FILE = "model/tab_transformer/model_file/tab_transformer.pt"
    CATEGORY_STAT_FILE = "model/tab_transformer/encoder/categorical_stat"
    CATEGORY_ENCODER_FILE = "model/tab_transformer/encoder/category_encoder.pkl"
    SERIALIZE_FILE = "model/tab_transformer/encoder/encoder_map.pkl"
    TARGET_ENCODING_SERIALIZE_FILE = "model/tab_transformer/encoder/target_encoding_map.pkl"
    CONTINUOUS_MAP_FILE = "model/tab_transformer/encoder/continuous_map.pkl"
    PREDICT_CSV_FILE = "data/valid/{}_tab_transformer_{}.csv".format(time.strftime("%Y_%m_%d"),
                                                                          time.strftime("%H_%M_%S"))
    TEST_RESULT_FILE = "data/test/{}_test_tab_transformer_{}.csv".format(time.strftime("%Y_%m_%d"),
                                                                              time.strftime("%H_%M_%S"))
    FRAC = 0.8
    SEED = 29
    LEARNING_RATE = 0.00003
    WEIGHT_DECAY = 0.3
    NUM_EPOCH = 800
    BATCH_SIZE = 512 + 256
    CONTINUOUS_COLUMNS = [
        'days',
        'months',
        'reviews_per_month',
        'calculated_host_listings_count',
        'availability',
        'availability_month',
        'availability_week',
        'minimum_nights',
        'number_of_reviews',
        'reviews_per_month_count',
    ]
    TARGET_ENCODING_MEAN_COLUMNS = [
        'neighbourhood_group_target_mean',
        'room_type_target_mean',
        'region_1_id_target_mean',
        'region_2_id_target_mean',
        # 'region_3_id_target_mean',
        'last_review_year_target_mean',
    ]

    CATEGORY_COLUMNS = [
        'host_id',
        'neighbourhood_group',
        'neighbourhood',
        'room_type',
        'region_1_id',
        'region_2_id',
        # 'region_3_id',
        'last_review_year',
        # 'last_review_isnull',
    ]
    TARGET_ENCODER_COLUMNS = [
        'neighbourhood_group',
        # 'neighbourhood_group_counts',
        'room_type',
        # 'room_type_counts',
        'region_1_id',
        # 'region_1_counts',
        'region_2_id',
        # 'region_2_counts',
        'region_3_id',
        # 'region_3_counts',
        'last_review_year',
    ]
    OUTPUT_COLUMNS = ["target"]
    # 设置运行设备
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


local_conf = TabXConfig()

tab_transformer_config = {
    "category_columns": local_conf.CATEGORY_COLUMNS,
    "continuous_columns": local_conf.CONTINUOUS_COLUMNS + local_conf.TARGET_ENCODING_MEAN_COLUMNS,
    "embedding_dim": 32,
    "depth": 4,
    "heads": 2,
    "dim_head": 16,
    "dim_out": len(local_conf.OUTPUT_COLUMNS),
    "mlp_hidden": (4, 2),
    "mlp_act": nn.ReLU()
}


# set_categorical_count 对类别特征进行统计
def set_categorical_count(dataframe: DataFrame) -> Dict[str, int]:
    category_stat = get_categorical_count(dataframe, category_columns=local_conf.CATEGORY_COLUMNS)
    # 序列化这个
    serialization(category_stat, local_conf.CATEGORY_STAT_FILE)
    global tab_transformer_config
    tab_transformer_config["category_count"] = category_stat
    return category_stat


# feature_engineering 特征工程
def feature_engineering(dataframe: DataFrame) -> DataFrame:
    if "target" in dataframe.columns.tolist():
        dataframe["target"] = np.log1p(dataframe["target"])
    dataframe["reviews_per_month"] = dataframe["reviews_per_month"].fillna(0)
    max_date = "2019-07-08"
    min_date = "2011-03-28"
    # 计算多久没评论
    dataframe["now"] = pd.to_datetime(max_date)
    dataframe["last_review"] = dataframe["last_review"].fillna(min_date)
    dataframe['last_review_year'] = pd.to_datetime(dataframe['last_review']).dt.year
    dataframe['last_review_isnull'] = dataframe['last_review'].isnull()
    dataframe["days"] = (dataframe["now"] - pd.to_datetime(dataframe["last_review"])).map(lambda x: x.days)
    dataframe["months"] = dataframe["days"] / 30.0
    dataframe['availability_month'] = dataframe['availability'] // 30
    dataframe['availability_week'] = dataframe['availability'] // 7
    dataframe['reviews_per_month_count'] = dataframe['reviews_per_month'] * dataframe['calculated_host_listings_count']
    for t in local_conf.CATEGORY_COLUMNS:
        dataframe[t] = dataframe[t].astype(str)
    dataframe = categorical_encoder(dataframe, local_conf.CATEGORY_COLUMNS, local_conf.CATEGORY_ENCODER_FILE)
    dataframe = standard_scaler(dataframe, local_conf.CONTINUOUS_COLUMNS, local_conf.SERIALIZE_FILE)
    return dataframe


# create_dataframe 创建dataframe
def create_dataframe(dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
    dataframe = feature_engineering(dataframe)
    set_categorical_count(dataframe)
    train_data = dataframe.sample(frac=local_conf.FRAC, random_state=local_conf.SEED)
    test_data = dataframe.drop(train_data.index)
    train_data = target_encoding(train_data, local_conf.TARGET_ENCODER_COLUMNS,
                                 local_conf.TARGET_ENCODING_SERIALIZE_FILE, is_train=True)
    test_data = target_encoding(test_data, local_conf.TARGET_ENCODER_COLUMNS, local_conf.TARGET_ENCODING_SERIALIZE_FILE,
                                is_train=False)
    return train_data, test_data


def create_dataset(data, is_test: bool = False):
    data_set = TabularDataSet(data, local_conf.CATEGORY_COLUMNS,
                              local_conf.CONTINUOUS_COLUMNS + local_conf.TARGET_ENCODING_MEAN_COLUMNS,
                              local_conf.OUTPUT_COLUMNS, is_test=is_test)
    return data_set


# model_instance
def model_instance(is_train: bool = True) -> TabTransformer:
    if not is_train:
        category_stat = deserialization(local_conf.CATEGORY_STAT_FILE)
        global tab_transformer_config
        tab_transformer_config["category_count"] = category_stat
    # 初始化模型
    model = TabTransformer(
        category_count=tab_transformer_config["category_count"],
        category_columns=tab_transformer_config["category_columns"],
        continuous_columns=tab_transformer_config["continuous_columns"],
        embedding_dim=tab_transformer_config["embedding_dim"],
        depth=tab_transformer_config["depth"],
        heads=tab_transformer_config["heads"],
        dim_head=tab_transformer_config["dim_head"],
        dim_out=tab_transformer_config["dim_out"],
        mlp_hidden=tab_transformer_config["mlp_hidden"],
        mlp_act=tab_transformer_config["mlp_act"],
    )
    if not is_train:
        model.load_state_dict(torch.load(local_conf.MODEL_FILE, map_location=local_conf.DEVICE))
    # 复制模型到 GPU(如果可用)
    model.to(local_conf.DEVICE)
    return model


# fit 训练数据
def fit(train_data: DataFrame):
    model = model_instance()
    # 定义损失函数
    criterion = nn.L1Loss()
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=local_conf.LEARNING_RATE,
        betas=(0.90, 0.95),
        weight_decay=local_conf.WEIGHT_DECAY)
    # 初始化数据集合
    train_set = create_dataset(train_data)
    # shuffle 为 true 或者false 需要实验查看，某种程度是否可以忽略日期
    train_loader = DataLoader(train_set, batch_size=local_conf.BATCH_SIZE, shuffle=True, drop_last=False)
    # 循环训练
    for epoch in range(local_conf.NUM_EPOCH):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # 将数据移动到 GPU（如果可用）
            labels = data["labels"].to(local_conf.DEVICE)
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
            # 反向传播并优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算损失
            running_loss += loss.item()

        # 输出损失信息
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, local_conf.NUM_EPOCH, running_loss / len(train_loader)))
    torch.save(model.state_dict(), local_conf.MODEL_FILE)


# valid 校验数据
def valid(valid_data: DataFrame):
    model = model_instance(is_train=False)
    # 开始推理（使用验证数据集校验模型）
    model.eval()
    # 初始化验证数据集
    val_set = create_dataset(valid_data)
    val_loader = DataLoader(val_set, batch_size=local_conf.BATCH_SIZE, shuffle=False, drop_last=False)
    category_encoder = deserialization(local_conf.CATEGORY_ENCODER_FILE)
    valid_data = inverse_transform(valid_data, local_conf.CATEGORY_COLUMNS, category_encoder)
    # 设置损失函数
    criterion = nn.L1Loss()
    outputs = []
    with torch.no_grad():
        total_loss = 0.0
        for i, data in enumerate(val_loader):
            # 将数据移动到 GPU（如果可用）
            labels = data["labels"].to(local_conf.DEVICE)
            # 前向传播
            output = model(data)
            loss = criterion(output, labels)
            # 计算损失
            total_loss += loss.item()
            output = torch.squeeze(output).cpu().numpy().tolist()
            outputs += output
        print('Validation Loss: {:.8f}'.format(total_loss / len(val_loader)))
    result = pd.DataFrame({
        "host_id": valid_data["host_id"],
        "predict_result": np.exp(outputs) - 1,
        "target": np.exp(valid_data["target"]) - 1,
    })
    result.to_csv(local_conf.PREDICT_CSV_FILE, index=False)
    test_rmse, test_mae, test_smape = cal_error(result["target"], result["predict_result"])
    logger.info(
        "test_rmse_error: {}, test_mae_error: {}, test_smape_error: {}".format(test_rmse, test_mae, test_smape))


def test(test_filepath: str):
    model = model_instance(is_train=False)
    model.eval()
    test_dataframe = read_csv(test_filepath)
    test_dataframe = feature_engineering(test_dataframe)
    test_dataframe = target_encoding(test_dataframe, local_conf.TARGET_ENCODER_COLUMNS,
                                     local_conf.TARGET_ENCODING_SERIALIZE_FILE, is_train=False)
    test_set = create_dataset(test_dataframe, is_test=True)
    data_loader = DataLoader(test_set, batch_size=local_conf.BATCH_SIZE, shuffle=False, drop_last=False)
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # 前向传播
            output = model(data)
            # 计算损失
            output = torch.squeeze(output).cpu().numpy().tolist()
            outputs += output
    result = pd.DataFrame({
        "id": test_dataframe["id"],
        "target": np.exp(outputs) - 1,
    })
    result.to_csv(local_conf.TEST_RESULT_FILE, index=False)


# main 入口函数
def main():
    SetSeed(local_conf.SEED, random, np, torch=torch, paddle=None)
    dataframe = read_csv(local_conf.FILE)
    train_data, valid_data = create_dataframe(dataframe)
    fit(train_data)
    valid(valid_data)
    test(local_conf.TEST_FILE)


if __name__ == "__main__":
    main()
