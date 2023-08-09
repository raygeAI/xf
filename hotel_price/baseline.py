# 读取数据集并进行标签缩放
import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train/train.csv')
test_df = pd.read_csv('./data/train/test.csv')
train_df['target'] = np.log1p(train_df['target'])

from sklearn.model_selection import cross_val_predict, cross_validate
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 训练集特征工程
train_df['last_review_isnull'] = train_df['last_review'].isnull()
train_df['last_review_year'] = pd.to_datetime(train_df['last_review']).dt.year
train_df['last_review_year_target_mean'] = train_df['last_review_year'].map(
    train_df.groupby(['last_review_year'])['target'].mean())
train_df['last_review_year_count'] = train_df['last_review_year'].map(train_df['last_review_year'].value_counts())

train_df['neighbourhood_group_mean'] = train_df['neighbourhood_group'].map(
    train_df.groupby(['neighbourhood_group'])['target'].mean())
train_df['neighbourhood_group_counts'] = train_df['neighbourhood_group'].map(
    train_df['neighbourhood_group'].value_counts())

train_df['room_type_mean'] = train_df['room_type'].map(train_df.groupby(['room_type'])['target'].mean())
train_df['room_type_counts'] = train_df['room_type'].map(train_df['room_type'].value_counts())

train_df['region_1_id_mean'] = train_df['region_1_id'].map(train_df.groupby(['region_1_id'])['target'].mean())
train_df['region_1_counts'] = train_df['region_1_id'].map(train_df['region_1_id'].value_counts())

train_df['region_2_id_mean'] = train_df['region_2_id'].map(train_df.groupby(['region_2_id'])['target'].mean())
train_df['region_2_counts'] = train_df['region_2_id'].map(train_df['region_2_id'].value_counts())

train_df['region_3_id_mean'] = train_df['region_3_id'].map(train_df.groupby(['region_3_id'])['target'].mean())
train_df['region_3_counts'] = train_df['region_3_id'].map(train_df['region_3_id'].value_counts())

train_df['availability_month'] = train_df['availability'] // 30
train_df['availability_week'] = train_df['availability'] // 7

train_df['reviews_per_month'] = train_df["reviews_per_month"].fillna(0)
train_df['reviews_per_month_count'] = train_df['reviews_per_month'] * train_df['calculated_host_listings_count']
train_df['room_type_calculated_host_listings_count'] = train_df['room_type'].map(
    train_df.groupby(['room_type'])['calculated_host_listings_count'].sum())

train_df['room_type_calculated_host_listings_mean'] = train_df['room_type'].map(
    train_df.groupby(['room_type'])['calculated_host_listings_count'].mean())

# 测试集数据增强
test_df['last_review_isnull'] = test_df['last_review'].isnull()
test_df['last_review_year'] = pd.to_datetime(test_df['last_review']).dt.year
test_df['last_review_year_target_mean'] = test_df['last_review_year'].map(
    train_df.groupby(['last_review_year'])['target'].mean())
test_df['last_review_year_count'] = test_df['last_review_year'].map(train_df['last_review_year'].value_counts())

test_df['neighbourhood_group_mean'] = test_df['neighbourhood_group'].map(
    train_df.groupby(['neighbourhood_group'])['target'].mean())
test_df['neighbourhood_group_counts'] = test_df['neighbourhood_group'].map(
    train_df['neighbourhood_group'].value_counts())

test_df['room_type_mean'] = test_df['room_type'].map(train_df.groupby(['room_type'])['target'].mean())
test_df['room_type_counts'] = test_df['room_type'].map(train_df['room_type'].value_counts())

test_df['region_1_id_mean'] = test_df['region_1_id'].map(train_df.groupby(['region_1_id'])['target'].mean())
test_df['region_1_counts'] = test_df['region_1_id'].map(train_df['region_1_id'].value_counts())

test_df['region_2_id_mean'] = test_df['region_2_id'].map(train_df.groupby(['region_2_id'])['target'].mean())
test_df['region_2_counts'] = test_df['region_2_id'].map(train_df['region_2_id'].value_counts())

test_df['region_3_id_mean'] = test_df['region_3_id'].map(train_df.groupby(['region_3_id'])['target'].mean())
test_df['region_3_counts'] = test_df['region_3_id'].map(train_df['region_3_id'].value_counts())

test_df['availability_month'] = test_df['availability'] // 30
test_df['availability_week'] = test_df['availability'] // 7

test_df['reviews_per_month'] = test_df["reviews_per_month"].fillna(0)
test_df['reviews_per_month_count'] = test_df['reviews_per_month'] * test_df['calculated_host_listings_count']
test_df['room_type_calculated_host_listings_count'] = test_df['room_type'].map(
    train_df.groupby(['room_type'])['calculated_host_listings_count'].sum())

test_df['room_type_calculated_host_listings_mean'] = test_df['room_type'].map(
    train_df.groupby(['room_type'])['calculated_host_listings_count'].mean())

min_date = "2011-03-28"
max_date = "2019-07-08"
# 计算多久没评论
max_date = pd.to_datetime(max_date)
train_df["last_review"] = train_df["last_review"].fillna("2011-03-28")
test_df["last_review"] = test_df["last_review"].fillna("2011-03-28")

train_df["days"] = (max_date - pd.to_datetime(train_df["last_review"])).map(lambda x: x.days)
test_df["days"] = (max_date - pd.to_datetime(test_df["last_review"])).map(lambda x: x.days)

train_df["month"] = (max_date - pd.to_datetime(train_df["last_review"])).map(lambda x: x.days) / 30
test_df["month"] = (max_date - pd.to_datetime(test_df["last_review"])).map(lambda x: x.days) / 30

train_df["neighbourhood_group_room_type"] = train_df["neighbourhood_group"] * 10 + train_df["room_type"]
test_df["neighbourhood_group_room_type"] = test_df["neighbourhood_group"] * 10 + test_df["room_type"]

drop_columns = ['id', 'target', 'last_review', ]

cv = 6
# 交叉验证训练模型
cat_val = cross_validate(
    CatBoostRegressor(
        max_depth=8,
        objective="MAE",
        eval_metric="MAE",
        iterations=1200,
        random_state=2023,
        learning_rate=0.0195,
        task_type="GPU",
    ),
    train_df.drop(drop_columns, axis=1),
    train_df['target'],
    return_estimator=True,
    cv=cv,
)
print("catboost_val", cat_val["test_score"])

lgb_val = cross_validate(
    LGBMRegressor(
        verbose=0,
        n_estimators=1000,
        force_row_wise=True,
        subsample=0.8,
        colsample_bytree=0.9,
        metric='mae',
        learning_rate=0.0195,
        random_state=2023
    ),
    train_df.drop(drop_columns, axis=1),
    train_df['target'],
    return_estimator=True,
    cv=cv
)
print("lgb_val", lgb_val["test_score"])

xgb_val = cross_validate(
    XGBRegressor(
        n_estimators=1000,
        # reg_lambda=0.02,
        # reg_alpha=0.02,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=2023,
        learning_rate=0.02),
    train_df.drop(drop_columns, axis=1),
    train_df['target'],
    return_estimator=True,
    cv=cv
)
print("xgb_val", xgb_val["test_score"])

scores = 0
for score in cat_val["test_score"] + lgb_val["test_score"] + xgb_val["test_score"]:
    scores += score

scores = scores / (cv * 3)
print("scores", scores)

# 模型预测
pred = np.zeros(len(test_df))
for clf in cat_val['estimator'] + lgb_val['estimator'] + xgb_val['estimator']:
    pred += clf.predict(test_df.drop(['id', 'last_review',
                                      # 'region_3_id',
                                      # 'neighbourhood'
                                      ], axis=1))

pred /= (cv * 3)
pred = np.exp(pred) - 1
pd.DataFrame({'id': range(30000, 40000), 'target': pred}).to_csv('baseline.csv', index=False)

# 0.38 ,太容易发生过拟合
