# mindspore-pemi

#### 介绍
Infusing Hierarchical Guidance into Prompt Tuning: A Parameter-Efficient Framework for Multi-level Implicit Discourse Relation Recognition

ACL 2023

Mindspore代码实现

#### 目录结构
├─[dataset.py](dataset.py)  内容  
├─[global_func.py](global_func.py)  
├─[log.py](log.py)  
├─[main.py](main.py)  
├─[model.py](model.py)  
├─[model_config.py](model_config.py)  
├─[early_stopping.py](early_stopping.py)  
├─model_data  
|     ├─wheel  
|     |   └mindnlp-0.2.0.20231227-py3-none-any.whl
#### 模型架构
整个模型按照论文中叙述分为三个部分：  
1）**软提示微调编码器**；应用类似P-Tuning的方式冻结除输入中软提示外所有的模型参数，仅微调输入中的数个软提示。  
2）**层次标签精炼方法**（Hierarchical Label Refining, HLR）；本文主要创新方法，用于在标签嵌入空间中建立层次化联系的同时减少参数依赖，
3）**层次间联合学习**；

#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx
