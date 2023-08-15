# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation

EPOCHS = 3
SEED = 29
FRAC = 0.90
extra_tokens = ["{open}", "{close}", "{set_up}", "{set_down}", "{set}", "{set_min}", "{set_max}", "{NUM}"]
PREDICT_CSV_FILE = "data/test/{}_sentence_transformer_{}.csv".format(time.strftime("%Y_%m_%d"),
                                                                     time.strftime("%H_%M_%S"))
VALID_CSV_FILE = "data/valid/{}_text_rule_{}.csv".format(time.strftime("%Y_%m_%d"), time.strftime("%H_%M_%S"))
DISTANCE_THRESHOLD = 0.80


# TextSimilarity 文本相似度匹配问题；给文本抽取规则
class TextSimilarity:
    def __init__(
            self,
            train_data_file: str = "data/train.csv",
            rule_data_file: str = "data/规则数据.xlsx",
            pretrain_model_file: str = "BAAI/bge-large-zh",
            finetune_model_path="model/model_file/finetune",
            finetune_model_file: str = "finetune_model_file.pth"):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        # 初始化工具
        self.train_examples = self._create_dataset(train_data_file, rule_data_file)
        self.pretrain_model_file = pretrain_model_file
        self.extra_tokens = extra_tokens
        self.finetune_model_path = finetune_model_path
        self.finetune_model_file = finetune_model_file
        # 开启模型训练
        self.fit()

    # stratified_sampling 分层抽样，要求对每个类别的样本个数至少大于1个
    @staticmethod
    def stratified_sampling(dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
        stratified_sample = StratifiedShuffleSplit(n_splits=1, test_size=1 - FRAC, random_state=SEED)
        for train_index, test_index in stratified_sample.split(dataframe, dataframe['规则id']):
            # 训练集合
            train_set = dataframe.loc[train_index]
            # 保证测试集
            test_set = dataframe.loc[test_index]
        return train_set, test_set

    # _create_dataset 创建数据集合
    def _create_dataset(self, train_data_file: str, rule_data_file: str) -> List[InputExample]:
        train_data = pd.read_csv(train_data_file)
        train_data['规则id'] = train_data['规则id'].astype(str)
        rules = pd.read_excel(rule_data_file)
        rules['规则id'] = rules['规则id'].astype(str)
        self.rules = rules
        train_data = pd.merge(train_data, rules, left_on='规则id', right_on='规则id', how='inner')
        # print(train_data["规则id"].value_counts())
        # # 测试集合
        # train_data_sample = train_data.sample(frac=FRAC, random_state=SEED)
        # # 验证集合
        # evaluate_data_sample = train_data[~train_data.index.isin(train_data_sample.index)]
        train_data_sample, evaluate_data_sample = self.stratified_sampling(train_data)
        self.evaluate_data_sample = evaluate_data_sample
        train_examples = []
        for index, row in train_data_sample.iterrows():
            neg_rules = rules['规则表达式'].sample(5).values
            pos_rule = row['规则表达式']

            for text in neg_rules:
                if text == row['规则表达式']:
                    continue
                # 构造一些负样本
                train_examples.append(InputExample(texts=[row.text, text], label=self.label_smoothing(0.0)))

            if pos_rule is not np.nan:
                train_examples.append(InputExample(texts=[row.text, pos_rule], label=self.label_smoothing(1.0)))
        return train_examples

    # label_smoothing 计算标签平滑
    @staticmethod
    def label_smoothing(label) -> float:
        if label == 0.0:
            return 0.05
        if label == 1.00:
            return 0.95
        return label

    def create_model(self) -> SentenceTransformer:
        word_embedding_model = models.Transformer(self.pretrain_model_file, max_seq_length=512, do_lower_case=True)
        # 数据集太小，不足以调整token的参数
        # word_embedding_model.tokenizer.add_tokens(self.extra_tokens, special_tokens=True)
        # word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_tokens=False, # 设置为True ，取得0.74 的最佳结果
            pooling_mode_mean_sqrt_len_tokens=True,
        )
        # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=512,
        #                            activation_function=nn.Tanh())
        model = SentenceTransformer(modules=[
            word_embedding_model,
            pooling_model,
            # dense_model
        ])
        return model

    # fit 训练模型
    def fit(self) -> SentenceTransformer:
        # model = SentenceTransformer(self.pretrain_model_file)
        model = self.create_model()
        print("......model architecture......:\n", model)
        train_loss = losses.CosineSimilarityLoss(model, loss_fct=nn.MSELoss())
        train_dataloader = DataLoader(self.train_examples[:], shuffle=True, batch_size=16, )
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS, warmup_steps=10)
        model.save(self.finetune_model_path, self.finetune_model_file)
        return model

    # load 加载预训练好的模型
    def load(self) -> SentenceTransformer:
        return SentenceTransformer(self.finetune_model_path)

    # encode 即将文本编码映射为向量
    def encode(self, sentences: Union[str, List[str]]):
        model = self.load()
        return model.encode(sentences)

    # 生成测试文件
    def generate_text_rule_file(self, test_file: str):
        test_data = pd.read_csv(test_file)
        test_embedding = self.encode(list(test_data['text']))
        test_embedding = normalize(test_embedding)
        rules_embedding = self.encode(list(self.rules['规则表达式']))
        rules_embedding = normalize(rules_embedding)
        dist = 1 - cdist(test_embedding, rules_embedding, metric='cosine')

        test_rule = []
        for idx in range(len(dist)):
            if dist[idx].max() < DISTANCE_THRESHOLD:
                test_rule.append(None)
            else:
                test_rule.append(self.rules['规则id'].iloc[dist[idx].argmax()])

        test_data['规则id'] = test_rule
        test_data.to_csv(PREDICT_CSV_FILE, index=False)

    # evaluate 评估模型效果
    def evaluate(self) -> float:
        text_embeddings = self.encode(list(self.evaluate_data_sample["text"]))
        text_embeddings = normalize(text_embeddings)
        rule_embeddings = self.encode(list(self.rules["规则表达式"]))
        rule_embeddings = normalize(rule_embeddings)
        # cosine 值计算距离，为(-1,1) 原本含义是计算两个向量距离，当为0时候 垂直，-1 时候相反，1 时候重合(最相似)，
        dist = 1 - cdist(text_embeddings, rule_embeddings, metric='cosine')
        evaluate_rule = []
        for idx in range(len(dist)):
            # [0,2] 越小越相似；
            if dist[idx].max() < DISTANCE_THRESHOLD:
                evaluate_rule.append(-1)
            else:
                evaluate_rule.append(self.rules['规则id'].iloc[dist[idx].argmax()])
        self.evaluate_data_sample["predict_rule_id"] = evaluate_rule
        # 写到csv 文件
        self.evaluate_data_sample.to_csv(VALID_CSV_FILE, index=False)
        score = f1_score(
            self.evaluate_data_sample["规则id"].astype(int),
            self.evaluate_data_sample["predict_rule_id"].astype(int), average="micro")
        f1_scores = self.cal_f1_score()
        return score

    # cal_f1_score 计算 f1-score 赛题中定义的方式
    def cal_f1_score(self) -> float:
        precision = 0.0
        recall = 0.0
        for index, row in self.evaluate_data_sample.iterrows():
            if row["规则id"] == row["predict_rule_id"]:
                precision += 1
            if row["predict_rule_id"] == -1:
                recall += 1
        # 召回率
        recall = precision / (len(self.evaluate_data_sample) - recall)
        # 准确率
        precision = precision / len(self.evaluate_data_sample)
        f1_scores = 2 * recall * precision / (recall + precision)
        print("manual f1 score", f1_scores)
        return f1_scores


if __name__ == "__main__":
    text_similarity = TextSimilarity()
    score = text_similarity.evaluate()
    print("f1-score", score)
    text_similarity.generate_text_rule_file("data/test_a.csv")
