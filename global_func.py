# @Time 2023/12/27 20:58
# @Author: Joker
# @File: global_func.py
# @Software: PyCharm

from model_config import BertConfig
import os
from dataset import PromptDataset


def obtain_max_id(path=BertConfig.model_save_path,
                  prefix="",
                  suffix="",
                  limits=None):
    """
    获取最大的id，用于挑选模型
    :param limits: 搜索id限制的范围
    :param path: 目标路径
    :param prefix: 搜索文件的前缀
    :param suffix: 搜索文件的后缀
    :return:
    """
    if limits is None:
        limits = [0, 100000]
    max_idx = -1
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.startswith(prefix) and file_name.endswith(suffix):
            target = int(file_name.split("_")[-1].split(".")[0])
            if limits[0] <= target <= limits[1]:
                max_idx = max(max_idx, target)
    return max_idx


def normal_dataset_split(dataset):
    """
    对数据集进行常规分割
    :param dataset: 数据集对象
    :return:
    """
    # 如果有外面prompt的外包装，就取得里面基础的dataset
    if isinstance(dataset, PromptDataset):
        origin_dataset = dataset.base_dataset
    else:
        origin_dataset = dataset
    dataset_name = origin_dataset.file_name
    from copy import deepcopy
    if BertConfig.dataset_split.get(dataset_name, None) is not None:
        sections = BertConfig.dataset_split[dataset_name]
        dataset_list = [deepcopy(dataset), deepcopy(dataset), deepcopy(dataset)]
        for idx in range(len(sections)):
            dataset_list[idx].base_dataset.corpus = dataset_list[idx].base_dataset.corpus[dataset_list[idx].base_dataset.corpus["Section"].isin(sections[idx])]
            dataset_list[idx].base_dataset.corpus.reset_index(inplace=True, drop=True)
            # dataset_list[idx] = PromptDataset(dataset_list[idx], input_form="<p:4><sen1><p:4><mask><p:4><sep><p:4><sen2><p:4>")
        # indices = [origin_dataset.corpus[origin_dataset.corpus["Section"].isin(sections[j])].index.tolist()
        #            for j in range(len(sections))]
        # return [Subset(dataset, indices[i]) for i in range(len(sections))]
        return dataset_list
    else:
        raise Exception("暂无对应的dataset分割方式！")
    #     train_num = int(len(origin_dataset) * 0.8)
    #     eval_num = int(len(origin_dataset) * 0.1)
    #     test_num = len(origin_dataset) - train_num - eval_num
    #     return random_split(dataset, [train_num, eval_num, test_num])