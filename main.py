# @Time 2023/11/23 19:52
# @Author: Joker
# @File: main.py
# @Software: PyCharm

from model import MultiLabelClassifier
from global_func import *
from dataset import *
import mindspore
import mindspore as ms
from mindspore.train import Accuracy, F1
from mindspore.dataset import GeneratorDataset
from log import log
import os
from tqdm import tqdm
import argparse
import sys


def main_process(args):
    """
    运行主函数
    :param args:
    :return:
    """
    ms.set_seed(101)
    if len(args.device_id) == 1:
        ms.set_context(device_target=args.device,
                       device_id=int(args.device_id[0]),
                       mode=ms.PYNATIVE_MODE)
                       # pynative_synchronize=True)
                       # mode=ms.GRAPH_MODE,
                       # save_graphs=True,
                       # save_graphs_path='model_data/save_graph')
    else:
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL)

    hierarchy_key = ["top_level", "second_level", "third_level"]

    d1 = getattr(sys.modules["dataset"], args.dataset_name)(relation_type="Implicit",
                                                            model_name=args.model_name,
                                                            hierarchy=args.hierarchy)
    d2 = PromptDataset(d1, input_form=args.input_form)
    dataset_list = normal_dataset_split(d2)
    dataloader_list = [GeneratorDataset(d,
                                        column_names=["input_ids", "attention_mask",
                                                      "token_type_ids", "mask_pos", "prompt_idx", "sen_range",
                                                      "top_level", "second_level", "third_level"],
                                        shuffle=False).batch(args.batch_size) for d in dataset_list]

    classifier = MultiLabelClassifier(label_mode=args.label_mode,
                                      num_cls=d1.num_cls,
                                      tokenizer=d1.tokenizer,
                                      hierarchy=args.hierarchy,
                                      model_name=args.model_name,
                                      freeze=args.freeze)
    # log.info(classifier.weight_units)
    log.info("freeze: {}".format(args.freeze))
    log.info("trainable params: {}".format(classifier.trainable_params()))
    log.info("hierarchical sense number: {}".format(d1.num_cls))
    optimizer = ms.nn.Adam(params=classifier.trainable_params(), learning_rate=args.learning_rate)
    loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    def forward_fn(data):
        output = classifier(data)
        loss_total = ms.Tensor(0.0)
        for i in range(args.hierarchy):
            loss = loss_fn(output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
            loss_total += loss
        return loss_total, output["output"]

    grad_fn = mindspore.value_and_grad(fn=forward_fn, has_aux=False, grad_position=None, weights=optimizer.parameters)
    acc_metrics = [Accuracy('classification'), Accuracy('classification'),
                   Accuracy('classification')]
    f1_metrics = [F1(), F1(), F1()]

    def clear_metrics():
        for i in range(args.hierarchy):
            acc_metrics[i].clear()
            f1_metrics[i].clear()

    # 训练阶段
    classifier.set_train(True)
    with tqdm(total=args.epochs * len(dataloader_list[0]), leave=False) as tbar:
        for epoch in range(args.epochs):
            loss_sum = 0
            for idx, data in enumerate(
                    dataloader_list[0].create_dict_iterator(num_epochs=args.epochs, output_numpy=False)):
                (loss, logits), grads = grad_fn(data)
                # loss = ms.ops.depend(loss, optimizer(grads))
                loss_sum += loss.asnumpy()
                for i in range(args.hierarchy):
                    acc_metrics[i].update(logits[hierarchy_key[i]], data[hierarchy_key[i]])
                    f1_metrics[i].update(logits[hierarchy_key[i]], data[hierarchy_key[i]])
                if (idx + 1) % args.log_freq == 0:
                    print()
                    # print([getattr(classifier.weight_units, "l1_weight_units_{}".format(i)).value() for i in range(4)])
                    # print(classifier.get_label_embeddings()[:4])
                    # print([param.value() for param in classifier.weight_units.l2_weight_units])
                    log.info("Epoch {} Step {} Average Loss: {}".format(epoch + 1, idx + 1, loss_sum / (idx + 1)))
                    log.info("Epoch {} Step {} Acc: {}".format(epoch + 1, idx + 1,
                                                               [acc_metrics[i].eval() * 100 for i in range(args.hierarchy)]))
                    log.info("Epoch {} Step {} Macro F1: {}".format(epoch + 1, idx + 1,
                                                                    [f1_metrics[i].eval(average=True) * 100 for i in range(args.hierarchy)]))
                tbar.update(1)
        clear_metrics()
    # 保存一下前面层次生成的标签嵌入
    classifier.save_label_embeddings()
    # 保存一下模型
    ms.save_checkpoint(classifier, os.path.join(BertConfig.model_save_path, "pemi.ckpt"))

    # 在测试集上验证效果
    classifier.set_train(False)
    clear_metrics()
    for idx, data in tqdm(enumerate(dataloader_list[2].create_dict_iterator(num_epochs=1, output_numpy=False)),
                          total=len(dataloader_list[2]), leave=False):
        eval_output = classifier(data)
        for i in range(args.hierarchy):
            acc_metrics[i].update(eval_output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
            f1_metrics[i].update(eval_output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
    # 结束时打印一下测试集结果
    log.info("Test Acc: {}".format([acc_metrics[i].eval() for i in range(args.hierarchy)]))
    log.info("Test Macro f1: {}".format([f1_metrics[i].eval(average=True) * 100 for i in range(args.hierarchy)]))


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser(description="PEMI Mindspore")
    # 输入设置
    parser.add_argument("-im", "--input_form", default="<p:4><sen1><p:4><mask><p:4><sep><p:4><sen2><p:4>", type=str)
    parser.add_argument("-dn", "--dataset_name", default="PDTBDataset", type=str)
    parser.add_argument('-d', '--device', default="GPU", type=str, choices=["GPU", "CPU", "Ascend"])
    parser.add_argument('-di', '--device_id', nargs='+')
    # 分类器挑选
    parser.add_argument("-hi", "--hierarchy", default=3, type=int)
    parser.add_argument("-mn", "--model_name", default="roberta-base", type=str)
    # parser.add_argument("-cm", "--class_mode", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("-f", "--freeze", action="store_true")
    # parser.add_argument("-cn", "--classifier_name", default="MultiLabelClassifier", type=str)
    parser.add_argument("-lm", "--label_mode", default=0, type=int)
    # 训练器设置
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-lf", "--log_freq", default=100, type=int)
    parser.add_argument("-bs", "--batch_size", default=8, type=int)
    parser.add_argument("-e", "--epochs", default=10, type=int)
    parser.add_argument("-es", "--early_stopping", action="store_true")
    parser.add_argument("-tr", "--total_repeat", default=5, type=int)
    parser.add_argument("-si", "--saved_id", default=-1, type=int)
    parser.add_argument("-ds", "--dev_step", default=500, type=int)
    parser.add_argument("-sp", "--saved_path", default=None, type=str)
    args = parser.parse_args()

    main_process(args)
