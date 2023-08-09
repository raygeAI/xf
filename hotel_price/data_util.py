# -*- coding: utf-8 -*-
"""
    公共的数据特征处理工具封装，不依赖具体类型的模型
"""
import os
import pickle
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OrdinalEncoder, StandardScaler
from config.config import CATEGORY_ENCODER_FILE, MIN_MAX_SCALER_FILE, ROBUST_SCALER_FILE, STANDARD_SCALER_FILE
from log import logger

# 表头数据(添加新特征，需要修改表头)
header_names = [
    'id',
    'host_id',
    'neighbourhood_group',
    'neighbourhood',
    'room_type',
    'minimum_nights',
    'number_of_reviews',
    'last_review',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability',
    'region_1_id',
    'region_2_id',
    'region_3_id',
    'target'
]

# 时间衍生特征
period_features = [
    "dayofweek",
    "days_in_month",
    "dayofyear",
    "weekday",
    "weekofyear",
    "month",
    "is_month_start",
    "is_month_end",
    "quarter",
    "is_quarter_start",
    "is_quarter_end",
    "year",
    "is_year_start",
    "is_year_end",
    "is_work_day",
    "holiday_name"
]


# time_period_features 挖掘时间周期特性
def time_period_features(dataframe: DataFrame, dt_column_name: str):
    # in day level
    dataframe['dayofweek'] = dataframe[dt_column_name].dt.dayofweek
    dataframe['days_in_month'] = dataframe[dt_column_name].dt.days_in_month
    dataframe['dayofyear'] = dataframe[dt_column_name].dt.dayofyear
    # week levels
    dataframe['weekday'] = dataframe[dt_column_name].dt.weekday
    dataframe['weekofyear'] = dataframe[dt_column_name].dt.isocalendar().week
    dataframe['weekofyear'] = dataframe['weekofyear'].astype(int)
    # month level
    dataframe['month'] = dataframe[dt_column_name].dt.month
    dataframe['is_month_start'] = dataframe[dt_column_name].dt.is_month_start
    dataframe['is_month_start'] = dataframe['is_month_start'].astype(int)
    dataframe['is_month_end'] = dataframe[dt_column_name].dt.is_month_end
    dataframe['is_month_end'] = dataframe['is_month_end'].astype(int)

    # quarter level
    dataframe['quarter'] = dataframe[dt_column_name].dt.quarter
    dataframe['is_quarter_start'] = dataframe[dt_column_name].dt.is_quarter_start
    dataframe['is_quarter_start'] = dataframe['is_quarter_start'].astype(int)
    dataframe['is_quarter_end'] = dataframe[dt_column_name].dt.is_quarter_end
    dataframe['is_quarter_end'] = dataframe['is_quarter_end'].astype(int)

    # year level
    dataframe['year'] = dataframe[dt_column_name].dt.year
    dataframe['is_year_start'] = dataframe[dt_column_name].dt.is_year_start
    dataframe['is_year_start'] = dataframe['is_year_start'].astype(int)
    dataframe['is_year_end'] = dataframe[dt_column_name].dt.is_year_end
    dataframe['is_year_end'] = dataframe['is_year_end'].astype(int)
    return dataframe


# read_tsv读取 tsv 数据
def read_csv(path: str) -> DataFrame:
    data = pd.read_csv(path, sep=",")
    return data


# data_detect 数据特征统计分布
def data_detect(dataframe: DataFrame, categories_columns: List[str]):
    na_stat = dataframe.isnull().sum()
    logger.info("na_stat: {} \n".format(na_stat))
    category_stat = {}
    # 遍历列数据
    columns = dataframe.columns
    for c in categories_columns:
        if c in columns:
            stat_series = dataframe[c].value_counts(dropna=False).to_dict()
            category_stat[c] = stat_series
    for k, v in category_stat.items():
        logger.info("k: {} \n , value: {}\n".format(k, v))
    return category_stat


# set_categorical 设置类别特征
def set_categorical(dataframe: DataFrame, category_columns: List[str]) -> DataFrame:
    for c in category_columns:
        dataframe[c] = dataframe[c].astype("category")
    return dataframe


# get_categorical_count 去重计数
def get_categorical_count(dataframe: DataFrame, category_columns: List[str] = None) -> Dict[str, int]:
    category_stat = {}
    for c in category_columns:
        category_stat[c] = len(dataframe[c].unique())
    return category_stat


# min_max_scaler  极大极小值标准化, 设置序列化目录
def min_max_scaler(dataframe: DataFrame, columns: List[str],
                   serialization_file: str = MIN_MAX_SCALER_FILE) -> DataFrame:
    scaler = MinMaxScaler().fit(dataframe[columns])
    serialization(scaler, serialization_file)
    dataframe[columns] = scaler.transform(dataframe[columns])
    return dataframe


# standard_scaler
def standard_scaler(dataframe: DataFrame, columns: List[str],
                    serialization_file: str = STANDARD_SCALER_FILE) -> DataFrame:
    scaler = StandardScaler().fit(dataframe[columns])
    serialization(scaler, serialization_file)
    dataframe[columns] = scaler.transform(dataframe[columns])
    return dataframe


# categorical_encoder 枚举类型特征编码
def categorical_encoder(dataframe: DataFrame, columns: List[str],
                        serialization_file: str = CATEGORY_ENCODER_FILE) -> DataFrame:
    """
    :param serialization_file: 文件序列话位置
    :param dataframe: 需要转换的dataframe
    :param columns: 需要转换的列(是枚举型)
    :return: 返回转化后的 dataframe
    """
    encoder = OrdinalEncoder()
    encoder.fit(dataframe[columns])
    serialization(encoder, serialization_file)
    dataframe[columns] = encoder.transform(dataframe[columns])
    return dataframe


# num_encoder 对数值特征进行
def robust_scaler(dataframe: DataFrame, columns: List[str], serialization_file: str = ROBUST_SCALER_FILE) -> DataFrame:
    scaler = RobustScaler().fit(dataframe[columns])
    # 保存scaler, 预测的时候需要进行编码
    serialization(scaler, serialization_file)
    dataframe[columns] = scaler.transform(dataframe[columns])
    return dataframe


# inverse_transform 逆变换
def inverse_transform(dataframe: DataFrame, columns: List[str], encoder) -> DataFrame:
    dataframe[columns] = encoder.inverse_transform(dataframe[columns])
    return dataframe


# serialization 序列化文件
def serialization(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


# deserialization 反序列化获取对象
def deserialization(file: str):
    with open(file, 'rb') as f:
        encoder = pickle.load(f)
    return encoder


# set_business_label
def set_business_label(dataframe: DataFrame) -> DataFrame:
    business_label = deserialization("model/hotel_label/hotel_label.pkl")
    dataframe["first_label_code"] = dataframe["hotel_id"].apply(
        lambda x: business_label[x][0] if x in business_label else "无")
    dataframe["second_label_code"] = dataframe["hotel_id"].apply(
        lambda x: business_label[x][1] if x in business_label else "无")
    return dataframe


# fill_group_na 对数据进行分组填充
def fill_group_na(dataframe: DataFrame, group_by_column: str, column: str, fill_method: str) -> DataFrame:
    # 均值进行填充
    if fill_method == "mean":
        dataframe[column] = dataframe.groupby(group_by_column)[column].apply(lambda x: x.fillna(x.mean()))
    elif fill_method == "median":
        dataframe[column] = dataframe.groupby(group_by_column)[column].apply(lambda x: x.fillna(x.median()))
    # mode 众数填充
    elif fill_method == "mode":
        dataframe[column] = dataframe.groupby(group_by_column)[column].apply(lambda x: x.fillna(x.mode()))
    # 对具体某一列固定值进行填充
    elif fill_method == "zero":
        if dataframe[column].dtype is np.dtype('str'):
            dataframe[column] = dataframe[column].fillna(value="0")
        else:
            dataframe[column] = dataframe[column].fillna(value=0)
    # pad 使用前一个数据进行填充
    elif fill_method == "pad":
        dataframe[column] = dataframe.groupby(group_by_column)[column].apply(lambda x: x.fillna(method="pad"))
    # 使用后一个数据进行填充
    elif fill_method == "bfill":
        dataframe[column] = dataframe.groupby(group_by_column)[column].apply(lambda x: x.fillna(method="bfill"))
    else:
        raise ValueError("fill_method must be one of  mean, median,mode,zero, pad, bfill")
    return dataframe


# filter_short_series 过滤掉较短的时间序列
def filter_short_series(dataframe: DataFrame, min_target_len: int) -> DataFrame:
    logger.info(f"hotel_id  cnt is {len(dataframe['hotel_id'].unique())}")
    hotel_id_stat = dataframe["hotel_id"].value_counts(ascending=True)
    hotel_id_stat = hotel_id_stat[hotel_id_stat >= min_target_len]
    hotel_list = list(hotel_id_stat.index)
    dataframe = dataframe[dataframe["hotel_id"].isin(hotel_list)]
    logger.info(f"after filter short series hotel_id cnt  is {len(dataframe['hotel_id'].unique())}")
    return dataframe


# filter_close_down_hotel 过滤掉停业的店铺
def filter_close_down_hotel(dataframe: DataFrame) -> DataFrame:
    # 假设最新的营业日还在营业的店，是没有停业店
    last = dataframe.iloc[-1]
    biz_day = last["bizday"]
    on_business_data = dataframe[dataframe["bizday"] == biz_day]
    hotel_ids = on_business_data["hotel_id"].unique().tolist()
    return dataframe[dataframe["hotel_id"].isin(hotel_ids)]


# target_encoding
def target_encoding(
        dataframe: DataFrame,
        target_encoding_columns: List[str],
        serialize_file: str, is_train: bool = True) -> DataFrame:
    mean_dicts = {}
    if is_train:
        for c in target_encoding_columns:
            c_mean = dataframe.groupby([c])["target"].mean()
            mean_dicts[c] = c_mean.to_dict()
        serialization(mean_dicts, serialize_file)
    # 对数据进行变换
    for c in target_encoding_columns:
        if not is_train:
            mean_dicts = deserialization(serialize_file)
        dataframe[c + "_target_mean"] = dataframe[c].map(mean_dicts[c])
    return dataframe


# category_count
def category_count(
        dataframe: DataFrame,
        target_encoding_columns: List[str],
        serialize_file: str, is_train: bool = True) -> DataFrame:
    category_count_dicts = {}
    if is_train:
        for c in target_encoding_columns:
            c_count = dataframe[c].value_counts()
            category_count_dicts[c] = c_count.to_dict()
        serialization(category_count_dicts, serialize_file)
    # 对数据进行变换
    for c in target_encoding_columns:
        if not is_train:
            category_count_dicts = deserialization(serialize_file)
        dataframe[c + "_count"] = dataframe[c].map(category_count_dicts[c])
    return dataframe


# GroupScaler 分组标准化, 这里传入数组
class GroupScaler:
    def __init__(self, id_column: str, serialize_file: str, scaler_type: str = "standard",
                 use_serialize_encoder: bool = False):
        self.id_column = id_column
        self.scaler_type = scaler_type
        self.serialize_file = self.set_encoder_serialization_file(serialize_file)
        self.use_serialize_encoder = use_serialize_encoder
        self.encoder_map = self.set_encoder_map()

    # adapt_scaler 适配scaler , 待添加其他算法
    def adapt_scaler(self):
        if self.scaler_type == "min_max":
            scaler = MinMaxScaler()
        elif self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "robust":
            scaler = RobustScaler()
        return scaler

    # 设置 encoder 的路径位置
    def set_encoder_serialization_file(self, serialize_file: str) -> str:
        parent_dir = os.path.dirname(serialize_file)
        filename = os.path.split(serialize_file)[-1]
        parent_dir = f"{parent_dir}/{self.scaler_type}/"
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        filepath = os.path.join(parent_dir, filename)
        return filepath

    def set_encoder_map(self):
        if self.use_serialize_encoder:
            encoder_map = deserialization(self.serialize_file)
        else:
            encoder_map = {}
        return encoder_map

    # serialize 序列化 encoder_map
    def serialize(self):
        with open(self.serialize_file, 'wb') as f:
            pickle.dump(self.encoder_map, f)

    # transform 对数据进行分组转换
    def transform(self, dataframe: DataFrame, to_transform_columns: List[str]):
        ids = dataframe[self.id_column].unique().tolist()
        dataframes = []
        for id in ids:
            id_dataframe = dataframe[dataframe[self.id_column] == id]
            if not self.use_serialize_encoder:
                scaler = self.adapt_scaler().fit(id_dataframe[to_transform_columns])
                self.encoder_map[id] = scaler
            else:
                # https://zhuanlan.zhihu.com/p/41202576 建议好好研究一下这个问题
                # 先 copy 一份然后进行更改
                scaler = self.encoder_map[id]
            id_dataframe = id_dataframe.copy()
            id_dataframe[to_transform_columns] = scaler.transform(id_dataframe[to_transform_columns])
            dataframes.append(id_dataframe)

        # 如果不是序列化，将scaler 序列化,
        if not self.use_serialize_encoder:
            self.serialize()
        # 将dataframe 拼接起来返回
        return pd.concat(dataframes, axis=0)

    # 将数据进行逆变换回来
    def inverse_transform(self, dataframe: DataFrame, to_transform_columns: List[str]) -> DataFrame:
        ids = dataframe[self.id_column].unique().tolist()
        # 如果encoder_map 为空，需要重新加载
        if not self.encoder_map:
            self.encoder_map = deserialization(self.serialize_file)
        dataframes = []
        for id in ids:
            if id in self.encoder_map:
                scaler = self.encoder_map[id]
            else:
                raise ValueError(f"unknown id {id}")
            id_dataframe = dataframe[dataframe[self.id_column] == id]
            # 这里不变更原来 dataframe
            id_dataframe = id_dataframe.copy()
            id_dataframe[to_transform_columns] = scaler.inverse_transform(id_dataframe[to_transform_columns])
            dataframes.append(id_dataframe)
        return pd.concat(dataframes, axis=0)

    # 动态传入encoder_map
    def transform_encoder(self, dataframe: DataFrame, to_transform_columns: List[str],
                          encoder_map: Dict[Any, Any]) -> DataFrame:
        ids = dataframe[self.id_column].unique().tolist()
        dataframes = []
        for id in ids:
            if id in self.encoder_map:
                scaler = encoder_map[id]
            else:
                raise ValueError(f"unknown id {id}")
            id_dataframe = dataframe[dataframe[self.id_column] == id]
            # 这里不变更原来 dataframe
            id_dataframe = id_dataframe.copy()
            id_dataframe[to_transform_columns] = scaler.transform(id_dataframe[to_transform_columns])
            dataframes.append(id_dataframe)
        return pd.concat(dataframes, axis=0)

    # 动态传入encoder_map 进行热加载
    def transform_decoder(self, dataframe: DataFrame, to_transform_columns: List[str],
                          encoder_map: Dict[Any, Any]) -> DataFrame:
        ids = dataframe[self.id_column].unique().tolist()
        dataframes = []
        for id in ids:
            if id in self.encoder_map:
                scaler = encoder_map[id]
            else:
                raise ValueError(f"unknown id {id}")
            id_dataframe = dataframe[dataframe[self.id_column] == id]
            # 这里不变更原来 dataframe
            id_dataframe = id_dataframe.copy()
            id_dataframe[to_transform_columns] = scaler.inverse_transform(id_dataframe[to_transform_columns])
            dataframes.append(id_dataframe)
        return pd.concat(dataframes, axis=0)
