# 基于DCN-V2模型的点击率预估模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data # 数据样例
        ├── sample_train.txt #训练数据样例
├── __init__.py
├── README.md #文档
├── net.py #组网文件
├── dygraph_model.py #动态图模型文件
├── static_model.py #静态图模型文件
├── config.yaml #样本数据训练推理配置文件
├── config_big.yaml #全量数据训练推理配置文件
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的DCN-V2模型：

```text
@inproceedings{DeepAndCross,
  title={DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems},
  author={Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi},
  year={2020}
}
```

## 数据准备
### 数据来源
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  

### 一键下载训练及测试数据
全量数据集解析过程:
1. 确认您当前所在目录为PaddleRec/models/rank/dcnv2
2. 进入paddlerec/datasets/criteo目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。自动处理数据转化为可直接进行训练的格式。解压后全量训练数据放置于`./slot_train_data_full`，全量测试数据放置于`./slot_test_data_full`

``` bash
cd ../../../datasets/criteo
sh run.sh
``` 


## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec模型目录"PaddleRec/models/rank/dcnv2"目录下执行下面的命令即可快速启动训练： 

动态图训练：
```
python ../../../tools/trainer.py -m ./config.yaml
```

动态图推理
```
python ../../../tools/infer.py -m ./config.yaml
```


## 模型组网
在DCN v1的基础上，DCN v2有两种结构，一种是并列的cross network和dnn层，一种是先叠加cross network再叠加dnn。两种方法在不同数据集上表现不一致，各有千秋。
如论文中所述，在criteo数据集上采用stack结构
![image](https://user-images.githubusercontent.com/66041642/149084601-6dab8f78-f681-460d-8ae6-c92f871cefa9.png)

作者们所提出的cross网络图示如下

![image](https://user-images.githubusercontent.com/66041642/149084631-275520c3-9fb2-4926-895e-bc5f80e51e43.png)

权重矩阵的奇异值变化趋势以及低秩专家的组合图示如下

![image](https://user-images.githubusercontent.com/66041642/149084656-2a86504c-a2fc-4049-88ec-0c7e98af00bb.png)



## 效果复现
为了方便使用者能够快速的跑通模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  

| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------| :------ | :------| 
| dcnv2 |  80.26   |   512   |  1 | 约 3 小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/dcnv2
2. 在"criteo data"全量数据目录下，运行数据一键处理脚本，命令如下：  
``` 
cd ../../../datasets/criteo
sh run.sh
```
3. 退回dcnv2目录中，配置改为使用config_big.yaml中的参数  

4. 运行命令，模型会进行一个epoch的训练，并获得相应auc指标  
动态图训练：
```
python ../../../tools/trainer.py -m ./config_big.yaml
```


5. 经过全量数据训练后，执行推理：
动态图推理
```
python ../../../tools/infer.py -m ./config_big.yaml
```
注意训练-预测，模型存储文件位置的一致性；
## 进阶使用
  
## FAQ