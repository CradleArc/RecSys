## # 本地建一个语义检索系统


## 一、项目说明


检索系统存在于我们日常使用的很多产品中，比如商品搜索系统、学术文献检索系，短视频相关性推荐等等，本方案提供了检索系统完整实现。限定场景是用户通过输入检索词 Query，快速在海量数据中查找相似文档。

所谓语义检索（也称基于向量的检索），是指检索系统不再拘泥于用户 Query 字面本身，而是能精准捕捉到用户 Query 后面的真正意图并以此来搜索，从而更准确地向用户返回最符合的结果。通过使用最先进的语义索引模型找到文本的向量表示，在高维向量空间中对它们进行索引，并度量查询向量与索引文档的相似程度，从而解决了关键词索引带来的缺陷。

例如下面两组文本 Pair，如果基于关键词去计算相似度，两组的相似度是相同的。而从实际语义上看，第一组相似度高于第二组。

```
车头如何放置车牌    前牌照怎么装
车头如何放置车牌    后牌照怎么装
```

语义检索系统的关键就在于，采用语义而非关键词方式进行召回，达到更精准、更广泛得召回相似结果的目的。


通常检索业务的数据都比较庞大，都会分为召回（索引）、排序两个环节。召回阶段主要是从至少千万级别的候选集合里面，筛选出相关的文档，这样候选集合的数目就会大大降低，在之后的排序阶段就可以使用一些复杂的模型做精细化或者个性化的排序。一般采用多路召回策略（例如关键词召回、热点召回、语义召回结合等），多路召回结果聚合后，经过统一的打分以后选出最优的 TopK 的结果。

本项目基于[PaddleNLP Neural Search](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search)。

以下是Neural Search的系统流程图，其中左侧为召回环节，核心是语义向量抽取模块；右侧是排序环节，核心是排序模型。图中红色虚线框表示在线计算，黑色虚线框表示离线批量处理。下面我们分别介绍召回中的语义向量抽取模块，以及排序模型。

![](https://ai-studio-static-online.cdn.bcebos.com/ee8e631f1288401fbaff199d5974f02646034b5de7eb47f2a3f58edb72967711)

### PaddleNLP Neural Search 系统特色
    
+ 低门槛
    + 手把手搭建起检索系统
    + 无需标注数据也能构建检索系统
    + 提供 训练、预测、ANN 引擎一站式能力

+ 效果好
    + 针对多种数据场景的专业方案
        + 仅有无监督数据: SimCSE
        + 仅有有监督数据: InBatchNegative
        + 兼具无监督数据 和 有监督数据：融合模型
    + 进一步优化方案: 面向领域的预训练 Domain-adaptive Pretraining 
+ 性能快
    + 基于 Paddle Inference 快速抽取向量
    + 基于 Milvus 快速查询和高性能建库


## 二、安装说明

AI Studio平台默认安装了Paddle和PaddleNLP，并定期更新版本。 如需手动更新，可参考如下说明：

* paddlepaddle >= 2.3    
[安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

* 最新版本 PaddleNLP 

使用如下命令确保安装最新版PaddleNLP：
```
!pip install --upgrade paddlenlp
```


* python >= 3.6   

## 三、召回模型方案实践

### 方案简介

首先利用业务上的无标注数据对SimCSE上进行无监督训练，训练导出模型，然后利用In-batch Negatives的策略在有监督数据上进行训练得到最终的召回模型。利用召回模型抽取向量，然后插入到Milvus召回系统中，进行召回。


### 无监督语义索引

#### 数据准备

我们基于开源的文献检索数据集构造生成了面向语义索引的训练集、评估集、召回库。

采用文献的 query,title,keywords 三个字段内容，构造无标签数据集，每一行只有一条文本，要么是query，要么就是title和keywords的拼接句子。

样例数据如下:
```
睡眠障碍与常见神经系统疾病的关系睡眠觉醒障碍,神经系统疾病,睡眠,快速眼运动,细胞增殖,阿尔茨海默病
城市道路交通流中观仿真研究
城市道路交通流中观仿真研究智能运输系统;城市交通管理;计算机仿真;城市道路;交通流;路径选择
网络健康可信性研究
网络健康可信性研究网络健康信息;可信性;评估模式
脑瘫患儿家庭复原力的影响因素及干预模式雏形 研究
脑瘫患儿家庭复原力的影响因素及干预模式雏形研究脑瘫患儿;家庭功能;干预模式
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
地西他滨与HA方案治疗骨髓增生异常综合征转化的急性髓系白血病患者近期疗效比较
个案工作 社会化
个案社会工作介入社区矫正再社会化研究——以东莞市清溪镇为例社会工作者;社区矫正人员;再社会化;角色定位
圆周运动加速度角速度
圆周运动向心加速度物理意义的理论分析匀速圆周运动,向心加速度,物理意义,角速度,物理量,线速度,周期
```

注：这里采用少量demo数据用于演示训练流程。预测阶段直接调用基于全量数据训练出来的模型进行预测。

## 四、排序方案实践

### 方案简介

基于ERNIE-3.0-Medium-zh训练Pair-wise模型。Pair-wise 匹配模型适合将文本对相似度作为特征之一输入到上层排序模块进行排序的应用场景。

双塔模型，使用ERNIE-3.0-medium-zh预训练模型，使用margin_ranking_loss训练模型。


### 数据准备

使用点击（作为正样本）和展现未点击（作为负样本）数据构造排序阶段的训练集

样例数据如下:
```
个人所得税税务筹划      基于新个税视角下的个人所得税纳税筹划分析新个税;个人所得税;纳税筹划      个人所得税工资薪金税务筹划研究个人所得税,工资薪金,税务筹划
液压支架底座受力分析    ZY4000/09/19D型液压支架的有限元分析液压支架,有限元分析,两端加载,偏载,扭转       基于ANSYS的液压支架多工况受力分析液压支架,四种工况,仿真分析,ANSYS,应力集中,优化
迟发性血管痉挛  西洛他唑治疗动脉瘤性蛛网膜下腔出血后脑血管痉挛的Meta分析西洛他唑,蛛网膜下腔出血,脑血管痉挛,Meta分析     西洛他唑治疗动脉瘤性蛛网膜下腔出血后脑血管痉挛的Meta分析西洛他唑,蛛网膜下腔出血,脑血管痉挛,Meta分析
氧化亚硅        复合溶胶-凝胶一锅法制备锂离子电池氧化亚硅/碳复合负极材料氧化亚硅,溶胶-凝胶法,纳米颗粒,负极,锂离子电池   负载型聚酰亚胺-二氧化硅-银杂化膜的制备和表征聚酰亚胺,二氧化硅,银,杂化膜,促进传输
```

# 五、参考文献

[1] Tianyu Gao, Xingcheng Yao, Danqi Chen: [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). EMNLP (1) 2021: 6894-6910

[2] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906). Preprint 2020.

[3] Shuohuan Wang, Yu Sun, Yang Xiang, Zhihua Wu, Siyu Ding, Weibao Gong, Shikun Feng, Junyuan Shang, Yanbin Zhao, Chao Pang, Jiaxiang Liu, Xuyi Chen, Yuxiang Lu, Weixin Liu, et al.:
[ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731). Preprint 2021.


