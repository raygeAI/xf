# -*- coding: utf-8 -*-
"""
lightGBM xf 酒店价格挑战赛模型
"""
import time
from typing import List, Optional, Tuple
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pandas import DataFrame
import lightgbm
from lightgbm import Dataset
from data_util import read_csv
from data_util import set_categorical
from data_util import target_encoding, category_count
from metric import cal_error
from log import logger

# 创建模型，训练模型
lightgbm_config = {
    "num_leaves": 250,
    "max_depth": 500,
    "objective": "regression",
    "metric": ["mae"],
    "subsample": 0.75,
    "colsample_bytree": 0.65,
    "learning_rate": 0.01,
    # goss 梯度单边采样,梯度采样策略
    # "boosting_type": "goss",
    "boosting": "gbdt",
    "boosting_type": "gbdt",
    "num_iterations": 500,
    "train_metric": True,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "verbose": 1,
    "feature_fraction": 0.75,
    "min_data_in_leaf": 25,
    "max_bin": 800,
    "random_state": 2023,
    "force_row_wise": True,
}

drop_columns = [
    "id",
    "last_review",
    "max"
]

TARGET = "target"

# 类型枚举特征
categorical_feature = [
    'host_id',
    'neighbourhood_group',
    'neighbourhood',
    'room_type',
    'region_1_id',
    'region_2_id',
    'region_3_id',
    'neighbourhood_group_room_type',
]

# target_encoder_columns
target_encoder_columns = [
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


# 训练配置文件
class TrainConfig:
    FRAC = 0.8
    SEED = 2023
    N_SPLITS = 5
    TRAIN_FILE = "./data/train/train.csv"
    TEST_FILE = "./data/train/test.csv"
    LIGHTGBM_MODEL_FILE = "model/lightgbm/model_file/lgb.txt"
    PREDICT_CSV_FILE = "./data/valid/{}_lightGBM_{}.csv".format(time.strftime("%Y_%m_%d"), time.strftime("%H_%M_%S"))
    TEST_RESULT_FILE = "./data/test/{}_test_lightGBM_{}.csv".format(time.strftime("%Y_%m_%d"),
                                                                    time.strftime("%H_%M_%S"))
    TARGET_ENCODING_SERIALIZE_FILE = "model/lightgbm/encoder/target_encoding_map.pkl"
    CATEGORY_COUNT_SERIALIZE_FILE = "model/lightgbm/encoder/category_count_map.pkl"
    SERIALIZE_FILE = "model/lightgbm/encoder/encoder_map.pkl"


# local_conf 初始化配置文件
local_conf = TrainConfig()


# feature_engineering 对数据进行特征工程
def feature_engineering(dataframe: DataFrame, is_train: bool = True) -> DataFrame:
    # 将目标进行标准化
    max_date = "2019-07-08"
    min_date = "2011-03-28"
    # 计算多久没评论
    dataframe["max"] = pd.to_datetime(max_date)
    dataframe["last_review"] = dataframe["last_review"].fillna(min_date)
    dataframe['last_review_year'] = pd.to_datetime(dataframe['last_review']).dt.year
    dataframe['last_review_isnull'] = dataframe['last_review'].isnull()
    dataframe["days"] = (dataframe["max"] - pd.to_datetime(dataframe["last_review"])).map(lambda x: x.days)
    dataframe["months"] = dataframe["days"] / 30
    dataframe['reviews_per_month_count'] = dataframe['reviews_per_month'] * dataframe['calculated_host_listings_count']
    dataframe['availability_month'] = dataframe['availability'] // 30
    dataframe['availability_week'] = dataframe['availability'] // 7
    dataframe["neighbourhood_group_room_type"] = dataframe["neighbourhood_group"] * 10 + dataframe["room_type"]
    dataframe = set_categorical(dataframe, categorical_feature)
    if is_train:
        dataframe["target"] = np.log1p(dataframe["target"])
    dataframe = dataframe.drop(drop_columns, axis=1)
    return dataframe


# category_encoding 编码
def category_encoding(dataframe: DataFrame, is_train: bool) -> DataFrame:
    dataframe = target_encoding(dataframe, target_encoder_columns,
                                local_conf.TARGET_ENCODING_SERIALIZE_FILE, is_train)
    dataframe = category_count(dataframe, target_encoder_columns,
                               local_conf.CATEGORY_COUNT_SERIALIZE_FILE, is_train)
    return dataframe


# create_dataframe 创建dataframe
def create_kf_set(filepath: str) -> List[Tuple[Dataset, Dataset, DataFrame]]:
    dataframe = read_csv(filepath)
    dataframe = feature_engineering(dataframe)
    kf = KFold(n_splits=local_conf.N_SPLITS, shuffle=True, random_state=local_conf.SEED)
    datasets = []
    for train_index, test_index in kf.split(dataframe):
        train_data = dataframe.iloc[train_index]
        test_data = dataframe.iloc[test_index]
        train_data = category_encoding(train_data.copy(), is_train=True)
        test_data = category_encoding(test_data.copy(), is_train=False)
        train_set = create_dataset(train_data)
        test_set = create_dataset(test_data)
        datasets.append((train_set, test_set, test_data))
    return datasets


# create_dataset 创建训练数据集合校验数据集
def create_dataset(dataframe: DataFrame, is_train: bool = True) -> Dataset:
    data_target = dataframe["target"]
    columns = dataframe.columns.tolist()
    if TARGET in columns and is_train:
        dataframe = dataframe.drop(["target"], axis=1)
    dataset = Dataset(data=dataframe, label=data_target, categorical_feature=categorical_feature)
    return dataset


# train 训练入口函数
def train(data_set: Dataset, valid_set: Optional[List[Dataset]]):
    model = lightgbm.train(
        params=lightgbm_config,
        train_set=data_set,
        valid_sets=valid_set,
        verbose_eval=1,
    )
    print("...feature importance...", pd.DataFrame({
        'column': model.feature_name(),
        'importance': model.feature_importance(),
    }).sort_values(by='importance'))
    model.save_model(local_conf.LIGHTGBM_MODEL_FILE)


# validate 预测接口
def validate(valid_dataframe: DataFrame):
    model = lightgbm.Booster(model_file=local_conf.LIGHTGBM_MODEL_FILE)
    target = valid_dataframe[TARGET]
    valid_dataframe.drop([TARGET], axis=1, inplace=True)
    predict_result = model.predict(valid_dataframe, data_has_header=True)
    result = pd.DataFrame({
        "host_id": valid_dataframe["host_id"],
        "predict_result": np.exp(predict_result) - 1,
        "target": np.exp(target.values) - 1
    })
    test_rmse, test_mae, test_smape = cal_error(result["target"], result["predict_result"])
    logger.info(
        "test_rmse_error: {},  test_mae_error: {} , test_smape_error: {}".format(test_rmse, test_mae, test_smape))
    result.to_csv(local_conf.PREDICT_CSV_FILE, index=False)
    return predict_result, test_mae


def test(test_file: str):
    dataframe = read_csv(test_file)
    test_data = feature_engineering(dataframe, is_train=False)
    test_data = category_encoding(test_data, is_train=False)
    model = lightgbm.Booster(model_file=local_conf.LIGHTGBM_MODEL_FILE)
    predict_result = model.predict(test_data, data_has_header=True)
    result = pd.DataFrame({
        "id": dataframe["id"],
        "target": np.exp(predict_result) - 1
    })
    result.to_csv(local_conf.TEST_RESULT_FILE, index=False)


def main():
    kf_set = create_kf_set(local_conf.TRAIN_FILE)
    time = 0
    mae = 0
    for train_dataset, test_dataset, valid_dataframe in kf_set:
        train(train_dataset, [train_dataset, test_dataset])
        _,test_mae = validate(valid_dataframe)
        print(".....................time......................", time, test_mae)
        time += 1
        mae += test_mae
    print(".....................mae......................", mae, mae / local_conf.N_SPLITS)
    test(local_conf.TEST_FILE)


# main 函数入口
if __name__ == "__main__":
    main()
