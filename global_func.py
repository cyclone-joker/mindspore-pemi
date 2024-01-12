# @Time 2023/12/27 20:58
# @Author: Joker
# @File: global_func.py
# @Software: PyCharm

from model_config import BertConfig
import os
from log import log
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
        return dataset_list
    else:
        raise Exception("暂无对应的dataset分割方式！")


def calculate_average_result(metric_dict,
                             final_dict,
                             cur_repeat,
                             total_repeat):
    """
    对最终结果进行平均加和的操作，方便进行统计
    :param final_dict: 最终结果字典，用于存储各迭代次数的结果平均
    :param metric_dict: 统计结果字典
    :param cur_repeat: 当前重复次数，用验证是否进行取余和输出
    :param total_repeat: 总共重复实验次数
    :return:
    """
    for key in metric_dict.keys():
        if isinstance(metric_dict[key], list):
            if not final_dict.get(key, None):
                final_dict[key] = [0.0]*len(metric_dict[key])
            for i in range(len(metric_dict[key])):
                final_dict[key][i] = final_dict[key][i] + metric_dict[key][i]/total_repeat
        else:
            final_dict[key] = final_dict.get(key, 0.0) + metric_dict[key]/total_repeat
    # 最后一轮平均完成后汇报结果
    if cur_repeat == total_repeat:
        for key in final_dict.keys():
            log.info("Averaged Test {0}: {1}".format(key, final_dict[key]))
