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

# 导入PaddleNLP相关的库
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.downloader import get_path_from_url
from visualdl import LogWriter
from utils.data import convert_pairwise_example
from utils.base_model import SemanticIndexBase


# 有监督语义索引
# 数据准备

# 使用文献的的query, title, keywords，构造带正标签的数据集，不包含负标签样本
#
# ```
# 宁夏社区图书馆服务体系布局现状分析	       宁夏社区图书馆服务体系布局现状分析社区图书馆,社区图书馆服务,社区图书馆服务体系
# 人口老龄化对京津冀经济	                 京津冀人口老龄化对区域经济增长的影响京津冀,人口老龄化,区域经济增长,固定效应模型
# 英语广告中的模糊语	                  模糊语在英语广告中的应用及其功能模糊语,英语广告,表现形式,语用功能
# 甘氨酸二肽的合成	                      甘氨酸二肽合成中缩合剂的选择甘氨酸,缩合剂,二肽
# ```
# 数据读取逻辑
def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            # 可以看到有监督数据使用query title pair的
            # 所以text_a和text_b不一样
            yield {'text_a': data[0], 'text_b': data[1]}

train_set_file='./train.csv'
train_ds = load_dataset(
        read_text_pair, data_path=train_set_file, lazy=False)
# # 打印3条文本
# for i in range(3):
#     print(train_ds[i])


# 展示3条数据
# for i  in range(3):
#     print(train_ds[i])

# 明文数据 -> ID 序列训练数
class SemanticIndexBatchNeg(SemanticIndexBase):
    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.3,
                 scale=30,
                 output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.sacle = scale

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype())

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        # 分布式批采样器加载数据的一个子集。
        # 每个进程可以传递给DataLoader一个DistributedBatchSampler的实例，每个进程加载原始数据的一个子集。
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # 批采样器的基础实现，
        # 用于 paddle.io.DataLoader 中迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致。
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    # 组装mini-batch
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

def convert_example(example, tokenizer, max_seq_length=512, do_evalute=False):

    result = []

    for key, text in example.items():
        if 'label' in key:
            # do_evaluate
            result += [example['label']]
        else:
            # do_train
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            result += [input_ids, token_type_ids]

    return result
# 关键参数, 定义模型训练的超参，优化器等等。
scale=20 # 推荐值: 10 ~ 30
margin=0.1 # 推荐值: 0.0 ~ 0.2
# 最大序列长度
max_seq_length=64
epochs=1
learning_rate=5E-5
warmup_proportion=0.0
weight_decay=0.0
save_steps=10
batch_size=64
output_emb_size=256

pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        'ernie-3.0-medium-zh')
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-3.0-medium-zh')
trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # query_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # query_segment
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # title_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # tilte_segment
    ): [data for data in fn(samples)]

train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
# Inbatch-Negatives
model = SemanticIndexBatchNeg(
        pretrained_model,
        margin=margin,
        scale=scale,
        output_emb_size=output_emb_size)

num_training_steps = len(train_data_loader) * epochs

lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                         warmup_proportion)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)


# 模型训练过程如下：1.从dataloader中取出小批量数据 2.输入到模型中做前向 3.求损失函数 3.反向传播更新梯度
def do_train(model, train_data_loader):
    global_step = 0
    tic_train = time.time()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch

            loss = model(
                query_input_ids=query_input_ids,
                title_input_ids=title_input_ids,
                query_token_type_ids=query_token_type_ids,
                title_token_type_ids=title_token_type_ids)

            global_step += 1
            if global_step % 5 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % save_steps == 0:
                save_path = os.path.join(save_dir, "model_%d" % global_step)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_param_path = os.path.join(save_path, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                tokenizer.save_pretrained(save_path)


do_train(model, train_data_loader)























