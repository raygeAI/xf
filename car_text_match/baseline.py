# coding: utf-8
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation

train_data = pd.read_csv('./data/train.csv')
train_data['规则id'] = train_data['规则id'].astype(str)

test_data = pd.read_csv('./data/test_a.csv')

rules = pd.read_excel('./data/规则数据.xlsx')
rules['规则id'] = rules['规则id'].astype(str)

train_data = pd.merge(train_data, rules, left_on='规则id', right_on='规则id', how='left')
train_data = train_data.sample(frac=1.0)

# 构造训练数据

train_examples = []
for row in train_data.iterrows():
    neg_rules = rules['规则表达式'].sample(5).values
    pos_rule = row[1]['规则表达式']

    for text in neg_rules:
        if text == row[1]['规则表达式']:
            continue
        train_examples.append(InputExample(texts=[row[1].text, text], label=0.0))

    if pos_rule is not np.nan:
        train_examples.append(InputExample(texts=[row[1].text, pos_rule], label=1.0))

print(len(train_examples))
# model_file = 'moka-ai/m3e-base'
model_file = "BAAI/bge-large-zh"
# https://huggingface.co/moka-ai/m3e-small
# model = SentenceTransformer(model_file)

# model_file = "model/model_file/m3e-small"
word_embedding_model = models.Transformer(model_file, max_seq_length=512)

tokens = ["{open}", "{close}", "{set_up}", "{set_down}", "{set}", "{set_min}", "{set_max}", "{NUM}"]
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_max_tokens=True)
print(pooling_model.get_pooling_mode_str())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                           activation_function=nn.Tanh())
# model = SentenceTransformer(
#     # model_name_or_path=model_file,
#     modules=[word_embedding_model, pooling_model, dense_model])
# print(model)

model = SentenceTransformer(model_file)
print("after", model)
train_dataloader = DataLoader(train_examples[:], shuffle=True, batch_size=16)
# val_dataloader = DataLoader(train_examples[-6000:], shuffle=True, batch_size=16)


train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=10)

test_feat = model.encode(list(test_data['text']))
test_feat = normalize(test_feat)

rules_feat = model.encode(list(rules['规则表达式']))
rules_feat = normalize(rules_feat)

dist = 1 - cdist(test_feat, rules_feat, metric='cosine')

test_rule = []
for idx in range(len(dist)):
    if dist[idx].max() < 0.8:
        test_rule.append(None)
    else:
        test_rule.append(rules['规则id'].iloc[dist[idx].argmax()])

test_data['规则id'] = test_rule

test_data.to_csv('./data/submit_2.csv', index=None)

# from sentence_transformers import SentenceTransformer
# sentences = ["样例数据-1", "样例数据-2"]
# model = SentenceTransformer('BAAI/bge-large-zh')
# embeddings_1 = model.encode(sentences, normalize_embeddings=True)
# embeddings_2 = model.encode(sentences, normalize_embeddings=True)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)
