# Mindspore-PEMI

#### 介绍
*[Infusing Hierarchical Guidance into Prompt Tuning: A Parameter-Efficient Framework for Multi-level Implicit Discourse Relation Recognition](https://aclanthology.org/2023.acl-long.357.pdf)（ACL 2023）*

Mindspore代码实现

#### 模型架构
整个模型按照论文中叙述分为三个部分： 

1）**软提示微调编码器**；应用类似P-Tuning的方式冻结除输入中软提示外所有的模型参数，仅微调输入中的数个软提示。 

2）**层次标签精炼方法**（Hierarchical Label Refining, HLR）；本文主要创新方法，用于在标签嵌入空间中建立层次化联系的同时减少参数依赖。

3）**层次间联合学习**；对隐式篇章关系的多个层次分类进行联合学习，使用的分类器即是2）中生成的标签嵌入。

整体模型架构图如下：

<img src="https://joker-typora-bucket.oss-cn-beijing.aliyuncs.com/picture/typora/2024/01/04/2024-01-04%2023:12:56.jpg" style="zoom: 33%;" />

#### 目录结构

├─[dataset.py](dataset.py)  数据集处理文件  
├─[global_func.py](global_func.py)  通用函数文件  
├─[log.py](log.py)   日志函数文件   
├─[main.py](main.py)   主程序文件   
├─[model.py](model.py)   模型文件  
├─[model_config.py](model_config.py)  模型基本超参数文件  
├─[early_stopping.py](early_stopping.py)  早停机制类文件  
├─model_data  模型依赖数据文件夹  
|     ├─dataset  
|     |   └PDTB3_connectives.txt  
|     |   └PDTB_connectives.txt  
|     ├─wheel  
|     |   └mindnlp-0.2.0.20231227-py3-none-any.whl

#### 运行教程

1. 安装依赖包，主要的包版本在requirements.txt中定义（MindNLP使用了Dally版本，放在了`model_data/wheel`下）

   ``````python
   pip install -r requirements.txt

2. 将PDTB 2.0和3.0的csv文件放入到`model_data/dataset`下，并命名为`PDTB.csv`和`PDTB3.csv`。

3. 将MindNLP的`roberta-base`预训练模型参数放到`model_data/pretrained_model`下。

4. 执行下面命令：

   ```python
   python3 main.py
   ```

#### 运行参数说明

##### 基本设置

1.  `-d`,`--device`,设备类型，包括`CPU/GPU/Ascend`。
2.  `-di`,`--device_id`,显卡虚拟号，可传入一个或多个显卡号。
3.  `-dn`,`--dataset_name`,选择的数据集名称，包括`PDTBDataset`和`PDTB3Dataset`两个选择。

##### 模型设置

1.  `-im`,`--input_form`，提示模板，其中包括`<cls>`分类标识、`<mask>`掩码标识、`<sen1>`和`<sen2>`两个论元标识以及`<p:x>`标识提示词组，其中`x`为当前位置提示词数量。
2.  `-hi`,`--hierarchy`,层次数，可以选择`1，2，3`层。
3.  `-mn`,`--model_name`,预训练模型名称，目前仅支持`roberta-base`。
4.  `-f`,`--freeze`,是否冻结预训练模型参数。
5.  `-lm`,`--label_mode`，标签自下而上生成的模式，`0`为不建立层次化联系，`5`为论文中自学习权重模式。

##### 训练设置

1.  `-lr`,`--learning_rate`,训练学习率。
2.  `-lf`,`--log_freq`,打印日志的频率，输入整数，为经过的迭代数。
3.  `-bs`,`--batch_size`,批处理数量。
4.  `-e`,`--epochs`,最大迭代数。
5.  `-es`,`--early_stopping`,是否设置早停机制。（默认为5次性能不上升后停止）
6.  `-tr`,`--total_repeat`,进行多少次重复实验。
7.  `-si`,`--saved_id`,保存号，默认为-1，即根据文件夹序号判定保存号。
8.  `-ds`,`--dev_step`,验证集验证的间隔，输入整数，为经过的迭代数。
9.  `-sp`,`--saved_path`,保存位置，默认保存位置为`model_data/saved_models`。
