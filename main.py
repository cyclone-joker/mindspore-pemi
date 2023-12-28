# @Time 2023/11/23 19:52
# @Author: Joker
# @File: main.py
# @Software: PyCharm

from model import MultiLabelClassifier
from global_func import *
from dataset import *
import mindspore
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from log import log
import os
from tqdm import tqdm


from mindnlp.transformers import BertTokenizer, BertModel, RobertaForMaskedLM, RobertaTokenizer
from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback, EarlyStopCallback
from mindnlp.metrics import Accuracy, F1Score


if __name__ == '__main__':
    ms.set_context(device_target='GPU',
                   device_id=0,
                   mode=ms.PYNATIVE_MODE)
                   # pynative_synchronize=True)
                   # mode=ms.GRAPH_MODE,
                   # save_graphs=True,
                   # save_graphs_path='model_data/save_graph')
    # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
    ms.set_seed(101)
    freeze = True
    num_epochs = 15
    hierarchy = 3
    num_cls = [4, 11, 102]
    log_step = 100
    learning_rate = 1e-3
    model_name = 'roberta-base'
    hierarchy_key = ["top_level", "second_level", "third_level"]

    d1 = PDTBDataset(relation_type="Implicit", model_name=model_name, hierarchy=3)
    d2 = PromptDataset(d1, input_form="<p:4><sen1><p:4><mask><p:4><sep><p:4><sen2><p:4>")
    dataset_list = normal_dataset_split(d2)
    dataloader_list = [GeneratorDataset(d,
                                        column_names=["input_ids", "attention_mask",
                                                      "token_type_ids", "mask_pos", "prompt_idx", "sen_range",
                                                      "top_level", "second_level", "third_level"],
                                        shuffle=True).batch(8) for d in dataset_list]

    classifier = MultiLabelClassifier(label_mode=5,
                                      num_cls=num_cls,
                                      tokenizer=d1.tokenizer,
                                      hierarchy=hierarchy,
                                      model_name=model_name,
                                      freeze=freeze)
    log.info("freeze: {}".format(freeze))
    log.info("trainable params: {}".format(classifier.trainable_params()))

    optimizer = ms.nn.Adam(params=classifier.trainable_params(), learning_rate=learning_rate)
    loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")


    def forward_fn(data):
        output = classifier(data)
        loss_total = ms.Tensor(0.0)
        for i in range(hierarchy):
            # print(output["output"][hierarchy_key[i]].shape)
            # print(data[hierarchy_key[i]].shape)
            loss = loss_fn(output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
            loss_total += loss
        return loss_total, output["output"]


    grad_fn = mindspore.value_and_grad(fn=forward_fn, has_aux=False, grad_position=None, weights=optimizer.parameters)
    loss_sum = 0
    acc_metrics = [ms.train.Accuracy('classification'), ms.train.Accuracy('classification'),
                   ms.train.Accuracy('classification')]
    f1_metrics = [ms.train.F1(), ms.train.F1(), ms.train.F1()]

    # 训练阶段
    classifier.set_train(True)
    with tqdm(total=num_epochs * len(dataloader_list[0]), leave=False) as tbar:
        for epoch in range(num_epochs):
            for idx, data in enumerate(dataloader_list[0].create_dict_iterator(num_epochs=num_epochs, output_numpy=False)):
                (loss, logits), grads = grad_fn(data)
                loss = ms.ops.depend(loss, optimizer(grads))
                loss_sum += loss.asnumpy()
                for i in range(hierarchy):
                    acc_metrics[i].update(logits[hierarchy_key[i]], data[hierarchy_key[i]])
                    f1_metrics[i].update(logits[hierarchy_key[i]], data[hierarchy_key[i]])
                if (idx + 1) % log_step == 0:
                    print()
                    log.info("Epoch {} Step {} Average Loss: {}".format(epoch+1, idx+1, loss_sum / (idx + 1)))
                    log.info("Epoch {} Step {} Acc: {}".format(epoch+1, idx + 1,
                                                               [acc_metrics[i].eval() for i in range(hierarchy)]))
                    temp_res = [f1_metrics[i].eval() for i in range(hierarchy)]
                    # print("{} f1: {}".format(hierarchy_key[i], f1_metrics[i].eval()))
                    log.info("Epoch {} Step {} Macro F1: {}".format(epoch+1, idx + 1,
                                                                    [sum(temp_res[j])/len(temp_res[j]) for j in range(hierarchy)]))
                tbar.update(1)
    # 保存一下前面层次生成的标签嵌入
    classifier.save_label_embeddings()
    # 保存一下模型
    ms.save_checkpoint(classifier, os.path.join(BertConfig.model_save_path, "pemi.ckpt"))

    # 在测试集上验证效果
    classifier.set_train(False)
    for i in range(hierarchy):
        acc_metrics[i].clear()
        f1_metrics[i].clear()
    for idx, data in tqdm(enumerate(dataloader_list[2].create_dict_iterator(num_epochs=1, output_numpy=False)),
                          total=len(dataloader_list[2]), leave=False):
        eval_output = classifier(data)
        for i in range(hierarchy):
            acc_metrics[i].update(eval_output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
            f1_metrics[i].update(eval_output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
    # 结束时打印一下测试集结果
    log.info("Test Average Loss: {}".format(loss_sum / (idx + 1)))
    log.info("Test Acc: {}".format([acc_metrics[i].eval() for i in range(hierarchy)]))
    temp_res = [f1_metrics[i].eval() for i in range(hierarchy)]
    # print("{} f1: {}".format(hierarchy_key[i], f1_metrics[i].eval()))
    log.info("Test Macro f1: {}".format([sum(temp_res[j]) / len(temp_res[j]) for j in range(hierarchy)]))