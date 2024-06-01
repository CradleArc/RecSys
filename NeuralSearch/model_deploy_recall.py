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

class RecallPredictor(object):
    def __init__(self,
                 model_dir,
                 device="gpu",
                 max_seq_length=128,
                 batch_size=32,
                 use_tensorrt=False,
                 precision="fp32",
                 cpu_threads=10,
                 enable_mkldnn=False):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        model_file = model_dir + "/output/recall/inference.get_pooled_embedding.pdmodel"
        params_file = model_dir + "/output/recall/inference.get_pooled_embedding.pdiparams"
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)

        if device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[precision]

            if use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(cpu_threads)
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])



    def extract_embedding(self, data, tokenizer):
        """
        Predicts the data labels.
        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        Returns:
            results(obj:`dict`): All the feature vectors.
        """

        examples = []
        for text in data:
            input_ids, segment_ids = convert_example_recall_infer(text, tokenizer)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # segment
        ): fn(samples)

        input_ids, segment_ids = batchify_fn(examples)
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(segment_ids)
        self.predictor.run()
        logits = self.output_handle.copy_to_cpu()
        return logits

    def predict(self, data, tokenizer):
        """
        Predicts the data labels.
        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        Returns:
            results(obj:`dict`): All the predictions probs.
        """

        examples = []
        for idx, text in enumerate(data):
            input_ids, segment_ids = convert_example_recall_infer({idx: text[0]}, tokenizer)
            title_ids, title_segment_ids = convert_example_recall_infer({
                idx: text[1]
            }, tokenizer)
            examples.append(
                (input_ids, segment_ids, title_ids, title_segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # segment
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
        ): fn(samples)


        query_ids, query_segment_ids, title_ids, title_segment_ids = batchify_fn(
            examples)
        self.input_handles[0].copy_from_cpu(query_ids)
        self.input_handles[1].copy_from_cpu(query_segment_ids)
        self.predictor.run()
        query_logits = self.output_handle.copy_to_cpu()

        self.input_handles[0].copy_from_cpu(title_ids)
        self.input_handles[1].copy_from_cpu(title_segment_ids)
        self.predictor.run()
        title_logits = self.output_handle.copy_to_cpu()

        result = [
            float(1 - spatial.distance.cosine(arr1, arr2))
            for arr1, arr2 in zip(query_logits, title_logits)
        ]
        return result

model_dir = './output/recall'
# device='gpu'
device='cpu'
max_seq_length=64
use_tensorrt = False
batch_size =32
precision = 'fp32'
cpu_threads = 1
enable_mkldnn =False
predictor = RecallPredictor(model_dir, device, max_seq_length,
                          batch_size, use_tensorrt, precision,
                          cpu_threads, enable_mkldnn)


id2corpus = {0: '国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据'}
corpus_list = [{idx: text} for idx, text in id2corpus.items()]
res = predictor.extract_embedding(corpus_list, tokenizer)
print('抽取向量')
print(res.shape)
print(res)

corpus_list = [['中西方语言与文化的差异', '中西方文化差异以及语言体现中西方文化,差异,语言体现'],
                   ['中西方语言与文化的差异', '飞桨致力于让深度学习技术的创新与应用更简单']]
res = predictor.predict(corpus_list, tokenizer)
print('计算相似度')
print(res)

# 导出静态图接下来就是部署了，目前部署支持C++和Pipeline两种方式，由于aistudio不支持部署环境，需要部署的话可以参考链接:
# (https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search/recall/in_batch_negative/deploy)