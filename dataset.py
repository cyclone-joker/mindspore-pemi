# @Time 2023/12/27 20:33
# @Author: Joker
# @File: dataset.py
# @Software: PyCharm

import pandas
import os
from functools import partial
import re
import numpy as np
from model_config import BertConfig
import mindspore
from mindnlp.transformers import RobertaTokenizer
from log import log


class BaseDataset:
    def __init__(self,
                 relation_type,
                 model_name,
                 hierarchy,
                 file_name,
                 use_cols,
                 max_sequence_len=BertConfig.max_sequence_len,
                 dataset_path=BertConfig.dataset_path):

        self.relation_type = relation_type
        self.model_name = model_name
        self.hierarchy = hierarchy
        self.max_sequence_len = max_sequence_len
        self.file_name = file_name
        self.dataset_path = dataset_path

        if self.file_name in ["TED_CDB", "HIT-CDTB"]:
            file_encoding = "gbk"
        else:
            file_encoding = "utf-8"
        full_corpus = pandas.read_csv(os.path.join(self.dataset_path, self.file_name + ".csv"),
                                      usecols=use_cols,
                                      encoding=file_encoding)
        self.corpus = self._operate_corpus(full_corpus)
        self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(BertConfig.pretrained_model_path,
                                                                       self.model_name))
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.encode_func = partial(self.tokenizer.encode_plus,
                                   add_special_tokens=True,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=self.max_sequence_len,
                                   return_attention_mask=True,
                                   return_token_type_ids=True,
                                   return_tensors="ms",
                                   )

    def _operate_corpus(self, full_corpus):
        """
        基本函数方法，用来根据条件过滤原始数据集的数据
        :param full_corpus:
        :return:
        """
        return full_corpus

    def operate_hierarchical_labels(self, res_dict, idx):
        """
        将层次化的标签放入到res_dict当中，继承类必须完成
        :param res_dict:
        :param idx:
        :return:
        """
        pass

    def __len__(self):
        return self.corpus.shape[0]

    def data_statistics(self, with_section=False):
        pass

    def _class_filter(self, threshold, del_classes=None):
        """
        进行类别过滤,主要依据threshold大小
        :param threshold:
        :param del_classes: 需要删除的指定类别
        :return:
        """
        pass

    def get_conns(self, col_names):
        """
        用于提取所有存在的连接词，可能这个函数只用一次
        :param col_names 用于提取连接词的列名列表
        :return:
        """
        use_cols = ["Relation"]
        use_cols.extend(col_names)
        full_corpus = pandas.read_csv(os.path.join(BertConfig.dataset_path, self.file_name + ".csv"),
                                      usecols=use_cols,
                                      encoding="utf-8")
        conn_list = []
        for col in col_names:
            conn_list.extend(full_corpus[col].dropna().to_list())
        conn_list = list(set([conn.strip() for conn in conn_list]))
        if conn_list.count("NONE") > 0:
            conn_list.remove("NONE")
        conn_list.sort()
        print(conn_list)
        print(len(conn_list))
        with open(os.path.join(BertConfig.dataset_path, self.file_name + "_connectives.txt"), "w+",
                  encoding="utf-8") as f:
            for conn in conn_list:
                f.write(conn + "\n")

    def __getitem__(self, idx):
        res_dict = {}
        self.operate_hierarchical_labels(res_dict, idx)
        arg1, arg2 = self.corpus.loc[idx, "Arg1_RawText"], self.corpus.loc[idx, "Arg2_RawText"]
        temp_dict = self.encode_func(text=arg1, text_pair=arg2)
        for key in temp_dict.keys():
            res_dict[key] = temp_dict[key].squeeze(0)
        # roberta最前面为<s>，bert最前面为[CLS]，roberta后面接</s>，bert为[SEP]
        sen1_len, sen2_len = len(self.tokenizer.tokenize(arg1)), len(self.tokenizer.tokenize(arg2))
        res_dict["sen_range"] = mindspore.tensor([1, sen1_len,
                                                  3 + sen1_len if "roberta" in self.model_name else 2 + sen1_len,
                                                  sen2_len], dtype=mindspore.int32)

        return res_dict


# @Time 2022/3/11 15:38
# @Author: Joker
# @File: prompt_dataset.py
# @Software: PyCharm


class PDTBDataset(BaseDataset):
    def __init__(self,
                 **kwargs
                 ):
        """
        PDTB 2.0数据集初始化
        :param is_transfer: 是否进行领域适应的设置，
                            如果是，则合成为7分类
                            Pragmatic x会和x合并，
                            List会和Conjunction进行合并,
                            Asynchronous和Synchrony进行合并
        :param kwargs:
        """
        kwargs["file_name"] = "PDTB"
        kwargs["use_cols"] = ["Relation", "Section", "Conn1",
                              "ConnHeadSemClass1", "ConnHeadSemClass2",
                              "Arg1_RawText", "Arg2_RawText"]
        self.relation_key = "ConnHeadSemClass1"
        self.expand_relation_key = "ConnHeadSemClass2"
        self.label_keys = ["top_level", "second_level", "third_level"]
        super(PDTBDataset, self).__init__(**kwargs)

        self.relation_hierarchies = {}
        for i in range(min(self.hierarchy, 2)):
            self.relation_hierarchies[self.label_keys[i]] = \
                self.corpus[self.relation_key].str.split(".").map(
                    lambda x: ".".join(x[:i + 1])).value_counts().index.tolist().copy()
            self.relation_hierarchies[self.label_keys[i]].sort()
        # 连接词分类表
        if self.hierarchy == 3:
            self.relation_hierarchies["third_level"] = self.load_conn_list()
        # 过滤掉数量太少的类别
        self._class_filter([0, 10, 0])
        self.corpus.reset_index(inplace=True, drop=True)

    def _replace_relations(self, relation_dict):
        """
        使用pandas将某些关系合并，转换为新的关系
        :param relation_dict:
        :return:
        """
        for level in relation_dict.keys():
            for key, val in relation_dict[level].items():
                self.corpus[self.relation_key] = self.corpus[self.relation_key].str.replace(key, val)
                # 如果新加的类型不存在，则添加到关系类型当中去
                if val not in self.relation_hierarchies[level]:
                    self.relation_hierarchies[level].append(val)

    def load_conn_list(self):
        """
        创建连接词列表
        :return:
        """
        conn_list = []
        conn_path = os.path.join(self.dataset_path, self.file_name + "_connectives.txt")
        if not os.path.exists(conn_path):
            self.get_conns("Conn1")
        with open(conn_path, "r") as conn_txt:
            for conn in conn_txt.readlines():
                conn_list.append(conn.replace("\n", ""))
        return conn_list

    def _operate_corpus(self, corpus):
        """
        对corpus数据集中的数据进行筛选
        如果进行11分类至少要有2级标注
        :param corpus: pandas数据集对象
        :return:
        """
        corpus = corpus[corpus["Relation"].str.contains(self.relation_type)].copy()
        # corpus = corpus[corpus["Section"].isin([i for i in range(2, 23)])]
        # 如果存在两个关系，将其扩充为两条数据
        double_corpus = corpus[~corpus[self.expand_relation_key].isnull()]
        # 筛选来自训练集的双标签数据
        double_label_trains = double_corpus[
            double_corpus["Section"].isin(BertConfig.dataset_split[self.file_name][0])].copy()
        # 将这些标签数据变为Class1，并且将Conn2变为Conn1
        double_label_trains[self.relation_key] = double_label_trains[self.expand_relation_key].copy()
        if self.file_name == "PDTB3":
            double_label_trains["Conn1"] = double_label_trains["Conn2"]
        if len(double_corpus) > 0:
            corpus = pandas.concat([corpus, double_corpus], ignore_index=True)
        # 如果进行二层次分类，则需要将没有二层次分类标识的数据删除掉
        if self.hierarchy >= 2:
            corpus = corpus[corpus[self.relation_key].str.split(".").map(len) >= 2]
        return corpus

    def _class_filter(self, threshold, del_classes=None):
        """
        根据类别的数量进行过滤，删除数量较少或者指定的类别
        :param threshold: list 对于各个层次的筛选数量
        :param del_classes: 指定删除的类别列表
        :return:
        """
        filtered_classes = []
        if del_classes is not None and len(del_classes) > 0:
            filtered_classes.extend(del_classes)
        if any(item > 0 for item in threshold):
            # 先过滤前两层
            for i in range(self.hierarchy):
                if i < 2:
                    count_frame = self.corpus[self.relation_key]. \
                        str.split(".").map(lambda x: ".".join(x[:i + 1])).value_counts()
                else:
                    count_frame = self.corpus["Conn1"].str.strip().value_counts()
                    # if self.file_name == "PDTB3":
                    #     filtered_classes.extend([item for item in self.relation_hierarchies[self.label_keys[i]]
                    #                              if item not in count_frame.index.tolist()])
                filtered_classes.extend(count_frame[count_frame < threshold[i]].index.tolist())

        if len(filtered_classes) > 0:
            # 从关系库中删除关系
            for m in range(len(filtered_classes)):
                for key in self.relation_hierarchies.keys():
                    if filtered_classes[m] in self.relation_hierarchies[key]:
                        self.relation_hierarchies[key].remove(filtered_classes[m])
                        # 从数据集中删除数据
                        self.corpus = self.corpus[self.corpus[self.relation_key].str.find(filtered_classes[m]) == -1]

    def __len__(self):
        return self.corpus.shape[0]

    def operate_hierarchical_labels(self, res_dict, idx):
        levels = []
        for i in range(min(self.hierarchy, 2)):
            levels.append(".".join(self.corpus.loc[idx, self.relation_key].split(".")[:i + 1]))
        # 如果对第二层次进行迁移，将11分类变为8分类
        if self.hierarchy == 3:
            levels.append(self.corpus.loc[idx, "Conn1"])
        for i in range(self.hierarchy):
            res_dict[self.label_keys[i]] = self.relation_hierarchies[self.label_keys[i]].index(levels[i])

    def data_statistics(self, with_section=False):
        # 首先统计每个类别的数量
        for i in range(self.hierarchy):
            temp_corpus = self.corpus[self.relation_key].str.split(".").map(lambda x: ".".join(x[:i + 1]))
            print("The {}-level relation statistics are shown as below:\n{}".format(i + 1, temp_corpus.value_counts()))
            print("total counts: {}".format(temp_corpus.shape[0]))
        if with_section:
            # 统计不同Section的数量分布
            sections = list(set(self.corpus["Section"].values))
            for section in sections:
                section_corpus = self.corpus[self.corpus["Section"].str.contains(section)]
                class_corpus = section_corpus[self.relation_key].str.split(".").map(lambda x: x[0])
                print("Section {} relation statistics are shown as below:\n{}".format(section,
                                                                                      class_corpus.value_counts()))
                print("total counts: {}".format(class_corpus.shape[0]))

    def calculate_upper_to_lower(self, threshold=2):
        """
        输出上层对下层的数据统计
        :param threshold:
        :return:
        """
        assert self.hierarchy >= 2, "计算该函数必须保证为二层以上分类!"
        count_list = []
        if self.hierarchy == 2:
            target_upper_list = self.relation_hierarchies["top_level"]
            target_lower_list = self.relation_hierarchies["second_level"]
            target_key = self.relation_key
        else:
            target_upper_list = self.relation_hierarchies["second_level"]
            target_lower_list = self.relation_hierarchies["third_level"]
            target_key = "Conn1"
        for label, label_word in enumerate(target_upper_list):
            label_corpus = self.corpus[self.corpus.ConnHeadSemClass1.str.contains(label_word)]
            temp_dict = {key: val for key, val in label_corpus[target_key].str.split(".").map(
                lambda x: ".".join(x[:self.hierarchy]) if self.hierarchy < 3 else x[0]).value_counts().to_dict().items()
                         if val > threshold}
            count_list.append({target_lower_list.index(key): val for key, val in temp_dict.items()})
        return count_list

    @property
    def num_cls(self):
        """
        获取层次对应的输出类别数
        :return:
        """
        return [len(self.relation_hierarchies[self.label_keys[i]]) for i in range(self.hierarchy)]


# @Time 2022/9/7 16:41
# @Author: Joker
# @File: prompt_dataset.py
# @Software: PyCharm


class PromptDataset:
    def __init__(self,
                 dataset=None,
                 input_form: str = None
                 ):
        """
        包装类，用来处理基于提示学习的输入
        :param input_form: 输入模板抽象形式
        :param dataset:基础数据集

        """
        # 初始化数据集
        self.base_dataset = dataset
        self.input_form, self.special_tokens = self.operate_input_form(input_form)
        self.origin_vocab_size = len(self.base_dataset.tokenizer)
        log.info("origin vocab size: {}".format(self.origin_vocab_size))
        self.base_dataset.tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        log.info('current vocab size: {}'.format(len(self.base_dataset.tokenizer)))

    def operate_input_form(self, input_forms: str):
        '''
        对句子组成结构进行替换,主要将prompt的部分改为多个<prompt:x>标签
        :param input_forms:
        :return:
        '''
        prompt_tokens_list = re.findall(re.compile(r'<p:\d+>'), input_forms)
        prompt_tokens = " ".join(prompt_tokens_list)
        prompt_nums = re.findall(re.compile(r'\d+'), prompt_tokens)
        prompt_nums = [int(item) for item in prompt_nums]
        special_tokens = ['<prompt{}>'.format(i) for i in range(sum(prompt_nums))]
        special_tokens.append("<sen1>")
        special_tokens.append("<sen2>")
        current_nums = 0
        for i in range(len(prompt_nums)):
            input_forms = input_forms.replace(prompt_tokens_list[i],
                                              "".join([special_tokens[m] for m in
                                                       range(current_nums, current_nums + prompt_nums[i])]), 1)
            current_nums += prompt_nums[i]
        # 将<mask><sep><cls>标记都进行替换
        input_forms = input_forms.replace("<mask>", self.base_dataset.tokenizer.mask_token). \
            replace("<sep>", self.base_dataset.tokenizer.sep_token). \
            replace("<cls>", self.base_dataset.tokenizer.cls_token)
        return input_forms, special_tokens

    def _calculate_prompt_idx(self, full_sentence):
        """
        计算prompt词的位置指导向量
        :param full_sentence: 完整的句子，分为raw_text以及token ids两种
        :return:
        """
        return np.array([i for i in range(full_sentence.shape[0]) if full_sentence[i] >= self.origin_vocab_size])

    def _get_input_dict(self, sentence_list, res_dict):
        """
        将输入包装为适合prompt的id张量形式
        :param sentence_list:
        :param res_dict:
        :return:
        """
        tokenizer = self.base_dataset.tokenizer
        sen1, sen2 = sentence_list
        sen1_tokens, sen2_tokens = tokenizer.tokenize(sen1), \
            tokenizer.tokenize(sen2)
        sen1_length, sen2_length = len(sen1_tokens), len(sen2_tokens)
        # 如果sen1过长，则需要进行提前截断，否则影响提示词的插入
        sen1_length = min(sen1_length, BertConfig.max_sequence_len)
        sen1_tokens = sen1_tokens[:sen1_length]
        template_token_list = tokenizer.tokenize(self.input_form)
        sen1_start = template_token_list.index("<sen1>")
        origin_sen2_start = template_token_list.index("<sen2>")
        sen2_start = origin_sen2_start + sen1_length - 1
        # 将各部分进行拼接
        full_tokens = template_token_list[:sen1_start] + sen1_tokens + \
                      template_token_list[sen1_start + 1: origin_sen2_start] + sen2_tokens \
                      + template_token_list[origin_sen2_start + 1:]
        final_token_index = sen2_start + sen2_length - 1
        # 如果还是超过，则从sen2中不断切除内容，直到长度小于等于最大长度
        while len(full_tokens) > BertConfig.max_sequence_len:
            full_tokens.pop(final_token_index)
            final_token_index -= 1

        full_ids = tokenizer.convert_tokens_to_ids(full_tokens)
        sen_reprs = np.array([tokenizer.pad_token_id] * BertConfig.max_sequence_len, dtype=np.int32)
        sen_reprs[:len(full_ids)] = full_ids
        # 生成attention mask
        # attention_mask = ms.ops.where(sen_reprs == self.base_dataset.tokenizer.pad_token_id,
        #                               ms.ops.zeros_like(sen_reprs), ms.ops.ones_like(sen_reprs))
        attention_mask = (sen_reprs != self.base_dataset.tokenizer.pad_token_id).astype(np.int32)

        # attention_mask = ms.ops.ones_like(sen_reprs)
        # 生成token_type_ids
        token_type_ids = np.zeros(sen_reprs.shape, dtype=np.int32)
        if "roberta" not in self.base_dataset.model_name.lower():
            token_type_ids[sen2_start:] = 1
        token_type_ids = token_type_ids.astype(np.int32)
        # token_type_ids = ms.tensor(token_type_ids, dtype=ms.int32)
        # 如果存在<mask>标识，那么标明mask在句子中的位置
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id in full_ids:
            mask_pos = full_ids.index(mask_token_id)
            res_dict["mask_pos"] = mask_pos
        else:
            # 没有mask就表明cls的位置
            res_dict["mask_pos"] = np.array(0)

        res_dict.update({"input_ids": sen_reprs,
                         "attention_mask": attention_mask,
                         "token_type_ids": token_type_ids,
                         "sen_range": [sen1_start, sen1_length, sen2_start, sen2_length]})

    def __getitem__(self, idx):
        arg1 = self.base_dataset.corpus["Arg1_RawText"][idx]
        arg2 = self.base_dataset.corpus["Arg2_RawText"][idx]
        # 首先将两个args放入到定制的内容当中
        res_dict = {}
        self.base_dataset.operate_hierarchical_labels(res_dict, idx)

        self._get_input_dict([arg1, arg2], res_dict)
        prompt_idx = self._calculate_prompt_idx(res_dict["input_ids"])
        # 对于超越长度的句子，进行一步更改，将<prompt>标识重新加回去
        assert prompt_idx.shape[0] == len(self.special_tokens) - 2, print(arg1 + "\n" + arg2)
        res_dict["prompt_idx"] = prompt_idx

        return (res_dict["input_ids"], res_dict["attention_mask"], res_dict["token_type_ids"],
                res_dict["mask_pos"], res_dict["prompt_idx"], res_dict["sen_range"], res_dict["top_level"],
                res_dict["second_level"], res_dict["third_level"])

    def __len__(self):
        return len(self.base_dataset)

    @property
    def output_size(self):
        return self.base_dataset.output_size
