# 导入系统库
import abc
import sys
from functools import partial
import argparse
import os
import random
import time
# 导入python的其他库
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from scipy.special import expit
# 导入Paddle库
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import inference

#导入PaddleNLP相关的库
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.downloader import get_path_from_url
from visualdl import LogWriter
from utils.data import convert_pairwise_example
from utils.data import convert_example_recall_infer
from scipy.special import softmax
from scipy import spatial

# 构建读取函数，读取原始数据
def read(src_path, is_predict=False):
    data=pd.read_csv(src_path,sep='\t')
    for index, row in tqdm(data.iterrows()):
        query=row['query']
        title=row['title']
        neg_title=row['neg_title']
        yield {'query':query, 'title':title,'neg_title':neg_title}

def read_test(src_path, is_predict=False):
    data=pd.read_csv(src_path,sep='\t')
    for index, row in tqdm(data.iterrows()):
        query=row['query']
        title=row['title']
        label=row['label']
        yield {'query':query, 'title':title,'label':label}


test_file='./dev_ranking_demo.csv'
train_file='./train_ranking_demo.csv'

train_ds=load_dataset(read,src_path=train_file,lazy=False)
dev_ds=load_dataset(read_test,src_path=test_file,lazy=False)
print('打印一条训练集')
print(train_ds[0])
print('打印一条验证集')
print(dev_ds[0])

class PairwiseMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin

        # hidden_size -> 1, calculate similarity
        self.similarity = nn.Linear(self.ptm.config["hidden_size"], 1)

    # 用于导出静态图模型来计算概率
    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, None], dtype='int64'),paddle.static.InputSpec(shape=[None, None], dtype='int64')])
    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids,
                                        position_ids, attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        # 计算相似度
        sim = self.similarity(cls_embedding)
        return sim


    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        sim_score = F.sigmoid(sim_score)
        return sim_score

    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids=None,
                neg_token_type_ids=None,
                pos_position_ids=None,
                neg_position_ids=None,
                pos_attention_mask=None,
                neg_attention_mask=None):

        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)

        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)

        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)

        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)

        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)

        labels = paddle.full(
            shape=[pos_cls_embedding.shape[0]], fill_value=1.0, dtype='float32')

        loss = F.margin_ranking_loss(
            pos_sim, neg_sim, labels, margin=self.margin)

        return loss

# 关键参数
margin=0.2 # 推荐取值 0.0 ~ 0.2
eval_step=100
max_seq_length=128
epochs=3
batch_size=32
warmup_proportion=0.0
weight_decay=0.0
save_step=100

# 基于 ERNIE-3.0-Medium-zh 热启训练单塔 Pair-wise 排序模型，并定义数据读取的 DataLoader
pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        'ernie-3.0-medium-zh')
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
        'ernie-3.0-medium-zh')

trans_func_train = partial(
        convert_pairwise_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

trans_func_eval = partial(
        convert_pairwise_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        phase="eval")

batchify_fn_train = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # pos_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # pos_pair_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # neg_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64')  # neg_pair_segment
    ): [data for data in fn(samples)]

batchify_fn_eval = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=batch_size,
        batchify_fn=batchify_fn_train,
        trans_fn=trans_func_train)

dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=batch_size,
        batchify_fn=batchify_fn_eval,
        trans_fn=trans_func_eval)
model = PairwiseMatching(pretrained_model, margin=margin)