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


# 数据读取逻辑
def read_simcse_text(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            data = line.rstrip()
            # 这里的text_a和text_b是一样的
            yield {'text_a': data, 'text_b': data}


# 数据集路径
train_set_file = './train_demo.csv'

train_ds = load_dataset(read_simcse_text, data_path=train_set_file, lazy=False)


# 展示3条数据
# for i  in range(3):
#     print(train_ds[i])

# 明文数据 -> ID 序列训练数据

# 在训练神经网络之前，我们需要构建小批量的数据，所以需要借助Dataloader
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


# 语义索引的维度最大为64，可以根据自己的情况调节长度
max_seq_length = 64
# 根据经验 batch_size越大效果越好
batch_size = 32
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-3.0-medium-zh')
# 给convert_example赋予默认的值，如tokenizer，max_seq_length
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
# [pad]对齐的函数
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # query_input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # query_segment
    Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # title_input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # tilte_segment
): [data for data in fn(samples)]

# 构建训练的Dataloader
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


# 接下来搭建SimCSE模型，主要部分是用query和title分别得到embedding向量，然后计算余弦相似度。
class SimCSE(nn.Layer):
    def __init__(self,
                 pretrained_model,
                 dropout=None,
                 margin=0.0,
                 scale=20,
                 output_emb_size=None):

        super().__init__()

        self.ptm = pretrained_model
        # 显式的加一个dropout来控制
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size,
        # 考虑到性能和效率，我们推荐把output_emb_size设置成256
        # 向量越大，语义信息越丰富，但消耗资源越多
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                768, output_emb_size, weight_attr=weight_attr)

        self.margin = margin
        # 为了使余弦相似度更容易收敛，我们选择把计算出来的余弦相似度扩大scale倍，一般设置成20左右
        self.sacle = scale

    # 加入jit注释能够把该提取向量的函数导出成静态图
    # 对应input_id,token_type_id两个
    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                                      paddle.static.InputSpec(shape=[None, None], dtype='int64')])
    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             with_pooler=True):

        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)

        if with_pooler == False:
            cls_embedding = sequence_output[:, 0, :]

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)
        cls_embedding = self.dropout(cls_embedding)
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/normalize_cn.html
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None):

        # 第 1 次编码: 文本经过无监督语义索引模型编码后的语义向量
        # [N, 768]
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        # 第 2 次编码: 文本经过无监督语义索引模型编码后的语义向量
        # [N, 768]
        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids,
            title_attention_mask)

        # 相似度矩阵: [N, N]
        cosine_sim = paddle.matmul(
            query_cls_embedding, title_cls_embedding, transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        # 填充self.margin值，比如margin为0.2，query_cls_embedding.shape[0]=2
        # margin_diag: [0.2,0.2]
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype())
        # input paddle.diag(margin_diag): [[0.2,0],[0,0.2]]
        # input cosine_sim : [[1.0,0.6],[0.6,1.0]]
        # output cosine_sim: [[0.8,0.6],[0.6,0.8]]
        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.sacle

        # 转化成多分类任务: 对角线元素是正例，其余元素为负例
        # labels : [0,1,2,3]
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        # labels : [[0],[1],[2],[3]]
        labels = paddle.reshape(labels, shape=[-1, 1])

        # 交叉熵损失函数
        loss = F.cross_entropy(input=cosine_sim, label=labels)
        return loss


# 训练配置：关键参数
scale = 20  # 推荐值: 10 ~ 30
margin = 0.1  # 推荐值: 0.0 ~ 0.2
# SimCSE的dropout的参数，也可以使用预训练语言模型默认的dropout参数
dropout = 0.2
# 向量映射的维度，默认的输出是768维，推荐通过线性层映射成256维
output_emb_size = 256
# 训练的epoch数目
epochs = 1
weight_decay = 0.0
# 学习率
learning_rate = 5E-5
warmup_proportion = 0.0

# 加载与训练模型 1. 加载预训练模型 ERNIE 3.0-Medium 进行热启 2. 定义优化器 AdamOptimizer
# 设置 ERNIE-3.0-Medium-zh 预训练模型
model_name_or_path = 'ernie-3.0-medium-zh'
pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
    model_name_or_path,
    hidden_dropout_prob=dropout,
    attention_probs_dropout_prob=dropout)
print("loading model from {}".format(model_name_or_path))

# 实例化SimCSE，SimCSE使用的Encoder是ERNIE-3.0-Medium-zh
model = SimCSE(
    pretrained_model,
    margin=margin,
    scale=scale,
    output_emb_size=output_emb_size)
# 训练的总步数
num_training_steps = len(train_data_loader) * epochs
# warmpup操作，学习率先上升后下降
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps,
                                     warmup_proportion)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]
# 设置优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

# 模型训练
save_dir = 'checkpoint'
save_steps = 100
time_start = time.time()
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch
        # 其中query和title为同一条数据
        loss = model(
            query_input_ids=query_input_ids,
            title_input_ids=title_input_ids,
            query_token_type_ids=query_token_type_ids,
            title_token_type_ids=title_token_type_ids)
        # 每隔10个step进行打印日志
        global_step += 1
        if global_step % 10 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                  % (global_step, epoch, step, loss,
                     10 / (time.time() - tic_train)))
            tic_train = time.time()
        # 反向
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
        # 每隔save_steps保存模型
        if global_step % save_steps == 0:
            save_path = os.path.join(save_dir, "model_%d" % (global_step))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_param_path = os.path.join(save_path, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_path)
time_end = time.time()
print('totally cost {} seconds'.format(time_end - time_start))

# 模型部署首先需要把模型转换成静态图模型。
output_path='./output/recall'
model.eval()
# Convert to static graph with specific input description
model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ])
# Save in static graph model.
save_path = os.path.join(output_path, "inference")
print(save_path)
paddle.jit.save(model, save_path)