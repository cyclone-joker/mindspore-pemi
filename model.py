# @Time 2023/12/27 20:35
# @Author: Joker
# @File: model.py
# @Software: PyCharm

import mindspore
import mindspore as ms
import os
from model_config import BertConfig
import sys
from mindnlp.transformers import BertTokenizer, BertModel, RobertaForMaskedLM, RobertaTokenizer
from global_func import obtain_max_id
import json
from log import log


class BaseClassifier(mindspore.nn.Cell):
    def __init__(self,
                 tokenizer,
                 hierarchy,
                 model_name,
                 class_mode,
                 freeze,
                 model_save_path=BertConfig.model_save_path,
                 **kwargs):
        super(BaseClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.hierarchy = hierarchy
        self.model_name = model_name
        self.class_mode = class_mode
        self.freeze = freeze
        self.model_save_path = model_save_path
        self.saved_id = -1
        self.save_count = 0
        self.load_success = False
        # 设置模型
        if 'roberta' in self.model_name:
            # self.model = RobertaForMaskedLM.from_pretrained(os.path.join(BertConfig.pretrained_model_path,
            #                                                              model_name))
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
        else:
            self.model = BertModel.from_pretrained(model_name)
        # auto_mixed_precision(self.model, 'O1')

        self.current_vocab_size = len(self.tokenizer)
        self.model.resize_token_embeddings(self.current_vocab_size)
        # 暂且设置prompt数量就为20
        self.prompt_size = len(self.tokenizer.additional_special_tokens) - 2
        log.info("prompt size: {}".format(self.prompt_size))

    def construct(self, inputs, with_output=True):
        input_dict = {"input_ids": inputs["input_ids"],
                      "attention_mask": inputs["attention_mask"],
                      "token_type_ids": inputs["token_type_ids"]}
        model_outputs = self.model(**input_dict)
        final_reprs, arg1_reprs, arg2_reprs = self.get_final_reprs(bert_outputs=model_outputs,
                                                                   inputs=inputs)
        result_dict = {"final_repr": final_reprs,
                       "arg1_repr": arg1_reprs,
                       "arg2_repr": arg2_reprs}
        if with_output:
            result_dict["output"] = self.classify_layer(final_reprs)
        return result_dict

    def get_final_reprs(self, bert_outputs, **kwargs):
        arg1_repr, arg2_repr = self.get_mean_reprs(bert_outputs[0],
                                                   kwargs["inputs"])
        if self.class_mode == 0:
            final_repr = bert_outputs.pooler_output
        elif self.class_mode == 1:
            final_repr = (arg1_repr + arg2_repr) / 2
        elif self.class_mode == 2:
            final_repr = self.gate_interaction(arg1_repr, arg2_repr)
        else:
            raise Exception("no such class mode!")
        return final_repr, arg1_repr, arg2_repr

    def get_mean_reprs(self, last_hidden_state, inputs):
        """
        对于有意义的词的表示取平均，从而获得该论元的表示
        :param last_hidden_state:
        :param inputs:
        :return:
        """
        batch_size = last_hidden_state.shape[0]
        hidden_size = last_hidden_state.shape[-1]
        arg1_reprs = ms.ops.zeros((batch_size, hidden_size))
        arg2_reprs = ms.ops.zeros_like(arg1_reprs)

        for i in range(batch_size):
            arg1_hidden_repr = last_hidden_state[i, inputs["sen_range"][i, 0]:inputs["sen_range"][i, 0] +
                                                                              inputs["sen_range"][i, 1], :]
            arg2_hidden_repr = last_hidden_state[i, inputs["sen_range"][i, 2]: inputs["sen_range"][i, 2] +
                                                                               inputs["sen_range"][i, 3], :]

            arg1_reprs[i, :] = ms.ops.mean(arg1_hidden_repr, axis=0)
            arg2_reprs[i, :] = ms.ops.mean(arg2_hidden_repr, axis=0)

        return arg1_reprs, arg2_reprs

    # 冻结BERT参数操作
    def _freeze_bert_parameters(self):
        frozen_parameters = self.model.get_parameters()
        for p in frozen_parameters:
            p.requires_grad = False

    def save_model(self,
                   step=0,
                   repeat_time=0,
                   save_path=None):
        """
        对模型参数进行保存
        :param repeat_time: 重复序列，对于重复实验或者k-折交叉验证比较有用
        :param save_path:
        :param step: 训练步数/迭代epoch数
        :return:
        """
        if save_path is not None:
            dir_name = os.path.join(save_path, "checkpoint_{:06d}".format(repeat_time * 100000 + step))
            # 自己规定路径，saved_id就没有意义了
            self.saved_id = 0
        else:
            if self.saved_id < 0:
                if repeat_time > 0 or self.save_count > 0:
                    self.saved_id = obtain_max_id()
                else:
                    self.saved_id = obtain_max_id() + 1
            dir_name = os.path.join(self.model_save_path,
                                    "Multi-level_Prompt_{}".format(self.saved_id),
                                    "checkpoint_{:06d}".format(repeat_time * 100000 + step))
        # print(dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model.pkl")
        param_path = os.path.join(dir_name, "params.json")
        config_dict = self.get_basic_params()
        config_dict.pop("tokenizer")
        ms.save_checkpoint(self.state_dict(), file_path)
        param_file = open(param_path, "w")
        json.dump(config_dict, param_file)
        param_file.close()
        self.save_count += 1

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        重写state dict保存机制
        :param destination:
        :param prefix:
        :param keep_vars:
        :return:
        """
        states = super().state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.freeze:
            # 保证只保留下面几部分的参数，其他都可以扔掉
            key_to_remove = [key for key in states.keys() if "prompt_params" not in key
                             and "classify_layer" not in key and "weighted_list" not in key]
            for key in key_to_remove:
                del states[key]

        return states

    @classmethod
    def load_model(cls,
                   repeat_time=0,
                   param_dict=None,
                   load_path=None,
                   limits=None,
                   classifier=None
                   ):
        """
        进行模型加载
        :param classifier: 如果有现成的分类器可以直接加载
        :param repeat_time: 重复次数
        :param limits: 对于加载区域的局限
        :param param_dict: 分类器参数列表
        :param load_path: 加载路径，可以规定要加载的其他路径
        :param limits: 100000以下是source，100000+为target，主要用于区分同时使用两个模型的情况
        :return:
        """
        # 首先创建模型
        if classifier is None:
            classifier_name = cls.__name__
            classifier = getattr(sys.modules["model_code.prompt_roberta.kernel.classifier"],
                                 classifier_name)(**param_dict)
        if limits is None:
            limits = [repeat_time * 100000, (repeat_time + 1) * 100000]
        if load_path is not None:
            dir_name = load_path
            param_dict["saved_id"] = 0
        else:
            dir_name = os.path.join(BertConfig.model_save_path,
                                    "Multi-level_Prompt_{}".format(param_dict["saved_id"]))
            print(dir_name)

        if os.path.exists(dir_name):
            # 找到最后一次保存的检查点
            max_step = obtain_max_id(path=dir_name,
                                     prefix="checkpoint",
                                     limits=limits)
            file_path = os.path.join(dir_name, "checkpoint_{:06d}".format(max_step))
            model_path = os.path.join(file_path, "model.pkl")
            # 将保存号保留下来
            classifier.saved_id = param_dict["saved_id"]
            if os.path.exists(model_path):
                log.info("load pretrained model for {} dev step ...".format(max_step))
                param_dict = ms.load_checkpoint(model_path)
                ms.load_param_into_net(classifier, param_dict)
            else:
                log.info("Checkpoints don't exist. Init pretrained model...")
        else:
            log.info("Init pretrained model...")

        return classifier

    @property
    def model_hidden_size(self):
        return self.model.config.hidden_size

    def get_basic_params(self):
        return {"tokenizer": self.tokenizer, "model_name": self.model_name,
                "hierarchy": self.hierarchy, "class_mode": self.class_mode,
                "saved_id": self.saved_id, "freeze": self.freeze}


# P-Tuning分类器
class WARPClassifier(BaseClassifier):
    def __init__(self,
                 num_cls,
                 **kwargs
                 ):
        kwargs["class_mode"] = 0
        super(WARPClassifier, self).__init__(**kwargs)
        self.num_cls = num_cls
        # 设置dropout为0，因为这种调参方式基本不会出现过拟合的问题
        self.model.config.hidden_dropout_prob = 0.0
        self.model.config.attention_probs_dropout_prob = 0.0
        # 更改mask lm，不使用decoder部分，将其扔掉
        # self.model.mlmloss.dense = ms.nn.Identity()
        self.model.lm_head.decoder = ms.nn.Identity()
        # 对模型参数进行冻结
        if self.freeze:
            self._freeze_bert_parameters()

        self.classify_layer = LabelEmbeddingLayer(embeddings=self.model.roberta.embeddings.word_embeddings,
                                                  num_cls=sum(self.num_cls) if isinstance(self.num_cls, list)
                                                  else self.num_cls,
                                                  output_size=self.model.config.hidden_size,)
        # 将原本的transformer的word embedding层进行替换
        injector = PromptInjector(tokenizer=self.tokenizer,
                                  old_embeddings=self.model.roberta.embeddings.word_embeddings,
                                  prompt_size=self.prompt_size)
        self.model.roberta.embeddings.word_embeddings = injector

    def construct(self, inputs, with_output=True):
        batch_size = inputs["input_ids"].shape[0]
        input_dict = {"input_ids": inputs["input_ids"],
                      "attention_mask": inputs["attention_mask"],
                      "token_type_ids": inputs["token_type_ids"],
                      "return_dict": True}
        # "masked_lm_positions": inputs["mask_pos"]}
        # "masked_lm_weights": ms.ops.ones(8, dtype=ms.int32),
        # "masked_lm_ids": ms.tensor([103]*batch_size, dtype=ms.int32)}
        model_outputs = self.model(**input_dict)
        # 获取平均的标识作为论元独立的表示
        # arg1_reprs, arg2_reprs = self.get_mean_reprs(model_outputs[1], inputs)

        assert inputs.get("mask_pos") is not None, "没有在输入模板当中设置<mask>的位置"
        final_reprs = model_outputs.logits[ms.ops.arange(0, batch_size, 1), inputs["mask_pos"], :]
        # final_reprs = model_outputs.logits[tuple(range(0, 8, 1)), inputs["mask_pos"], :]
        # final_reprs = ms.ops.zeros((batch_size, model_outputs.logits.shape[-1]))
        # for i in range(batch_size):
        #     final_reprs[batch_size] = model_outputs.logits[i, inputs['mask_pos'][i], :]
        output_dict = {"final_repr": final_reprs}
        # "arg1_repr": arg1_reprs,
        # "arg2_repr": arg2_reprs}
        if with_output:
            output_dict["output"] = self.classify_layer(final_reprs)
        return output_dict

    def get_basic_params(self):
        basic_params = super(WARPClassifier, self).get_basic_params()
        basic_params["init_prompt_with_embeddings"] = self.init_prompt_with_embeddings
        basic_params["num_cls"] = self.num_cls
        return basic_params


class MultiLabelClassifier(WARPClassifier):
    def __init__(self,
                 label_mode,
                 **kwargs):
        """
        进行多任务分类器计算
        :param label_mode:
        0：相当于多任务分类, 分类器之间没有任何指导
        1，2，3，4: 进行多层次label embedding转换
        1: 使用平均方式
        2: 使用max-pooling方式
        3: 使用绝对值最大方式
        4: 使用加权求和方式，该方式权重固定，使用count数的开方
        5: 使用加权求和方式，但是该方式的权重是自己学习的
        6: 使用加权求和方式，(1-β^n)/(1-β)
        :param num_cls: 传参为一个列表，用于指示各个层级的类别数量
        """
        if kwargs["hierarchy"] < 2:
            raise ValueError("multi label classifier must operate at least 2 hierarchical categories!")
        # 使用父类方法进行初始化
        super(MultiLabelClassifier, self).__init__(**kwargs)
        # super().__init__(**kwargs)
        self.label_mode = label_mode
        self.threshold = 1
        # 在初始化时计算range_list
        self.range_list = self._get_range_list()
        if self.label_mode in [4, 5, 6]:
            self.weighted_list = self._get_weighted_list()

    def calculate_upper_label_embeddings(self):
        """
        从底层计算所有层次的类别嵌入，这种方式是从子类嵌入中获取父类的内容
        :return:
        """
        if self.label_mode > 0:
            label_embeds = self.classify_layer.label_embeddings
            label_embed_list = [label_embeds(ms.ops.arange(sum(self.num_cls[:self.hierarchy - 1]),
                                                           sum(self.num_cls)))]
            # 创建空白向量，存储计算完成的向量
            for j in range(self.hierarchy - 2, -1, -1):
                label_embed_list.append(ms.ops.zeros(self.num_cls[j], self.model_hidden_size))
            # 按照归属关系进行聚类
            for idx in range(len(self.range_list)):
                for i in range(len(self.range_list[idx])):
                    if idx == 0:
                        # 原始向量从label embedding中获取
                        lower_embeds = label_embeds(self.range_list[idx][i] + self.num_cls[1])
                    else:
                        lower_embeds = label_embed_list[idx][self.range_list[idx][i]]

                    if self.range_list[idx][i].shape[0] > 1:
                        if self.label_mode == 1:
                            label_embed_list[idx + 1][i] = ms.ops.mean(lower_embeds, axis=0)
                        elif self.label_mode == 2:
                            label_embed_list[idx + 1][i] = ms.ops.max(lower_embeds, axis=0).values
                        elif self.label_mode == 3:
                            x_max = ms.ops.max(lower_embeds, axis=0).values
                            x_min = ms.ops.min(lower_embeds, axis=0).values
                            label_embed_list[idx + 1][i] = ms.ops.where(x_max > x_min.abs(), x_max, x_min)
                        elif self.label_mode in [4, 6]:
                            # 进行加权求和的方式
                            label_embed_list[idx + 1][i] = self.weighted_list[idx][i] @ lower_embeds
                        elif self.label_mode == 5:
                            weighted_tensor = self.weighted_list[idx][i]
                            label_embed_list[idx + 1][i] = (weighted_tensor / weighted_tensor.norm(dim=0, ord=1,
                                                                                                   keepdim=True)) \
                                                           @ lower_embeds
                    else:
                        # 如果下层类别只有一个子类，直接复制到上级即可
                        label_embed_list[idx + 1][i] = lower_embeds
        else:
            label_embed_list = self.get_label_embeddings()

        return label_embed_list

    def get_label_embeddings(self) -> list:
        """
        返回所有层次的label embeddings的列表
        :return:
        """
        label_embeddings = self.classify_layer.label_embeddings
        index_list = [ms.ops.arange(self.num_cls[0], sum(self.num_cls[:2])),
                      ms.ops.arange(0, self.num_cls[0])]
        if self.hierarchy == 3:
            index_list.insert(0, ms.ops.arange(sum(self.num_cls[:2]), sum(self.num_cls)))
        label_embed_list = []
        for idx in range(len(index_list)):
            label_embed_list.append(label_embeddings(index_list[idx]))
        return label_embed_list

    def save_label_embeddings(self):
        """
        在梯度完成后将前面层次的标签嵌入保存到Embedding当中
        :return: None
        """
        label_embedding_list = self.calculate_upper_label_embeddings()
        label_embeddings_weight = self.classify_layer.label_embeddings.weight
        index_list = [ms.ops.arange(0, self.num_cls[0])]
        if self.hierarchy == 3:
            index_list.insert(0, ms.ops.arange(self.num_cls[0], sum(self.num_cls[:2])))
        for idx in range(len(index_list)):
            label_embeddings_weight[index_list[idx]] = label_embedding_list[idx + 1]

    def _get_range_list(self):
        """
        获取每个上层类别对应下层类别的列表，对于数量过少的类别进行过滤
        :return:
        """
        range_list = [[ms.tensor([key for key, val in pro.items() if val > self.threshold])
                       for pro in BertConfig.top2second]]
        if self.hierarchy == 3:
            range_list.insert(0, [ms.tensor([key for key, val in pro.items() if val > self.threshold])
                                  for pro in BertConfig.second2conn])
        return range_list

    def _get_weighted_list(self):
        origin_list = [[{key: val for key, val in pro_dict.items() if val > self.threshold}
                        for pro_dict in BertConfig.top2second]]
        if self.hierarchy == 3:
            origin_list.insert(0, [{key: val for key, val in pro_dict.items() if val > self.threshold}
                                   for pro_dict in BertConfig.second2conn])
        # 只有在4，5，6 情况下需要进行权重加和
        if self.label_mode == 4:
            weighted_list = []
            for idx in range(len(origin_list)):
                weighted_tensors = [ms.tensor([count for count in val.values()]).sqrt() for val in origin_list[idx]]
                weighted_list.append([tensor / tensor.norm(dim=0, keepdim=True, ord=1) for tensor in weighted_tensors])
        elif self.label_mode == 5:
            # 需要创建相关权重参数矩阵
            weighted_list = []
            for idx in range(len(origin_list)):
                weighted_list.append(
                    [ms.Parameter(ms.ops.ones(len(val)).div(len(val)), requires_grad=True)
                     for val in origin_list[idx]])
        elif self.label_mode == 6:
            weighted_list = []
            beta = 0.9
            for idx in range(len(origin_list)):
                weight_tensors = [(1 - beta ** ms.tensor([count for count in val.values()])) / (1 - beta)
                                  for val in origin_list[idx]]
                weighted_list.append(
                    [tensor / ms.ops.norm(tensor, dim=0, keepdim=True, ord=1) for tensor in weight_tensors])
                # weighted_list.append([F.softmax(tensor, dim=0) for tensor in weight_tensors])
        else:
            raise ValueError("计算权重不应该选择label mode {}".format(self.label_mode))
        return weighted_list

    def construct(self, inputs, with_output=True):
        output_dict = super().construct(inputs, False)
        if self.phase == 'train':
            third_level, second_level, top_level = self.calculate_upper_label_embeddings()
        else:
            third_level, second_level, top_level = self.get_label_embeddings()
        label_embeddings_dict = {"top_level": top_level, "second_level": second_level}
        if self.hierarchy == 3:
            label_embeddings_dict["third_level"] = third_level
        output_dict["label_embeddings"] = label_embeddings_dict
        # 进行分类
        if with_output:
            output_dict["output"] = {key: output_dict["final_repr"] @ val.t() for key, val in
                                     label_embeddings_dict.items()}

        return output_dict

    def get_basic_params(self):
        basic_params = super(MultiLabelClassifier, self).get_basic_params()
        basic_params["label_mode"] = self.label_mode
        basic_params["threshold"] = self.threshold
        return basic_params


# 注入提示的类
class PromptInjector(ms.nn.Cell):
    def __init__(self,
                 tokenizer,
                 old_embeddings,
                 prompt_size):
        """
        用于替换原本embedding的类
        :param tokenizer: 分词器，主要用于获取词表中某些词的id
        :param old_embeddings: 原始embedding
        :param prompt_size: 插入的软提示词数
        """
        super(PromptInjector, self).__init__()
        self.vocab_size = old_embeddings.vocab_size
        self.embeddings = old_embeddings
        self.prompt_size = prompt_size
        self.prompt_ids = [i for i in range(self.vocab_size - self.prompt_size - 2,
                                            self.vocab_size - 2)]
        self.prompt_params = ms.Parameter(
            ms.ops.zeros((self.prompt_size, old_embeddings.embedding_size), dtype=ms.float32),
            requires_grad=True)
        self.prompt_params.set_data(self.embeddings(
            ms.tensor([tokenizer.mask_token_id] * prompt_size)))

    def construct(self, inputs):
        embeddings_output = self.embeddings(inputs)
        for idx, prompt_id in enumerate(self.prompt_ids):
            mask = inputs == prompt_id
            embeddings_output = ms.ops.where(mask.unsqueeze(-1), self.prompt_params[idx], embeddings_output)
        # print(embeddings_output.shape)
        # print(type(embeddings_output))
        # print(embeddings_output)
        return embeddings_output


# 标签嵌入层
class LabelEmbeddingLayer(ms.nn.Cell):
    def __init__(self,
                 embeddings,
                 num_cls,
                 output_size):
        """
        用于prompt模式下获取最终分类结果的层次
        :param embeddings:
        :param num_cls:
        :param output_size:
        """
        super(LabelEmbeddingLayer, self).__init__()
        self.output_size = output_size
        self.num_cls = num_cls
        self.label_embeddings = ms.nn.Embedding(self.num_cls, self.output_size)
        self.label_embeddings.weight.set_data(ms.ops.normal((self.num_cls, self.output_size),
                                                            mean=embeddings.weight.mean().item(),
                                                            stddev=embeddings.weight.std().item()))

    def construct(self, final_repr):
        all_label_embeddings = self.label_embeddings(ms.ops.arange(0, self.num_cls))
        return final_repr @ all_label_embeddings.t()
