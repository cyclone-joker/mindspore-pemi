# @Time 2023/11/23 19:52
# @Author: Joker
# @File: main.py
# @Software: PyCharm

from model import MultiLabelClassifier
from global_func import *
from dataset import *
import mindspore as ms
from mindspore.train import Accuracy, F1
from mindspore.dataset import GeneratorDataset
from log import log
from early_stopping import EarlyStopping
from tqdm import tqdm
import argparse
import sys


def main_process(args):
    """
    运行主函数
    :param args:
    :return:
    """
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

    if args.saved_id < 0:
        args.saved_id = obtain_max_id(prefix="Multi-level_Prompt") + 1
    log.info("当前保存号: {}".format(args.saved_id))

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
    # 统计最终平均结果的字典
    final_dict = {}
    for repeat_time in range(args.total_repeat):
        log.info("重复实验计数：{}".format(repeat_time + 1))
        once_process(args=args,
                     repeat_time=repeat_time,
                     loader_list=dataloader_list,
                     source_dataset=d1,
                     seed=100 + repeat_time,
                     final_dict=final_dict)


def clear_metrics(metrics):
    for metric_list in metrics:
        for i in range(args.hierarchy):
            metric_list[i].clear()


def once_process(args,
                 repeat_time,
                 loader_list,
                 source_dataset,
                 seed=100,
                 final_dict=None):
    ms.set_seed(seed)
    np.random.seed(seed)
    hierarchy_key = ["top_level", "second_level", "third_level"]
    model_param_dict = {"label_mode": args.label_mode,
                        "num_cls": source_dataset.num_cls,
                        "tokenizer": source_dataset.tokenizer,
                        "hierarchy": args.hierarchy,
                        "model_name": args.model_name,
                        "freeze": args.freeze,
                        "saved_id": args.saved_id}

    classifier = MultiLabelClassifier.load_model(repeat_time=repeat_time,
                                                 param_dict=model_param_dict)
    log.info("freeze: {}".format(args.freeze))
    log.info("trainable params: {}".format(classifier.trainable_params()))
    log.info("hierarchical sense number: {}".format(source_dataset.num_cls))
    log.info("learning rate: {}".format(args.learning_rate))
    train_steps = len(loader_list[0]) * args.epochs
    # 设置warmup长度为总步数的10%
    warmup_lr = ms.nn.WarmUpLR(args.learning_rate, train_steps // 10)
    optimizer = ms.nn.Adam(params=classifier.trainable_params(), learning_rate=args.learning_rate)

    # 是否选择使用早停机制
    if args.early_stopping:
        es = EarlyStopping(classifier.save_model, repeat_time)
    loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    def forward_fn(data):
        output = classifier(data)
        loss_total = ms.Tensor(0.0)
        for i in range(args.hierarchy):
            loss = loss_fn(output["output"][hierarchy_key[i]], data[hierarchy_key[i]])
            loss_total += loss
        return loss_total, output

    grad_fn = ms.value_and_grad(fn=forward_fn,
                                has_aux=False,
                                grad_position=None,
                                weights=classifier.trainable_params())
    acc_metrics = [Accuracy('classification'), Accuracy('classification'),
                   Accuracy('classification')]
    f1_metrics = [F1(), F1(), F1()]
    total_step = 0
    is_break = False

    # 训练阶段
    classifier.set_train(True)
    with tqdm(total=train_steps, leave=True, position=0) as tbar:
        for epoch in range(args.epochs):
            loss_sum = 0
            for idx, data in enumerate(
                    loader_list[0].create_dict_iterator(num_epochs=args.epochs, output_numpy=False)):
                total_step += 1
                (loss, outputs), grads = grad_fn(data)
                if args.label_mode == 5:
                    grads = grads[:2] + outputs["weight_units_grads"]
                grads = ms.ops.clip_by_global_norm(grads, 1.0)
                optimizer.learning_rate = warmup_lr(ms.Tensor(total_step, ms.float32))
                optimizer(grads)
                # loss = ms.ops.depend(loss, optimizer(grads))
                loss_sum += loss.asnumpy()

                # 保存一下前面层次生成的标签嵌入
                classifier.save_label_embeddings()
                for i in range(args.hierarchy):
                    acc_metrics[i].update(outputs["output"][hierarchy_key[i]],
                                          data[hierarchy_key[i]])
                    f1_metrics[i].update(outputs["output"][hierarchy_key[i]],
                                         data[hierarchy_key[i]])
                # 打印训练集结果
                if total_step % args.log_freq == 0:
                    print()
                    # print(classifier.get_label_embeddings()[:4])
                    print([param.value() for param in classifier.weight_units.l1_weight_units])
                    log.info("Current Learning Rate: {:.6f}".format(optimizer.learning_rate.item()))
                    log.info("Epoch {} Step {} Average Loss: {}".format(epoch + 1, total_step, loss_sum / (idx + 1)))
                    log.info("Epoch {} Step {} Acc: {}".format(epoch + 1, total_step,
                                                               [round(acc_metrics[i].eval() * 100, 6) for i in
                                                                range(args.hierarchy)]))
                    log.info("Epoch {} Step {} Macro F1: {}".format(epoch + 1, total_step,
                                                                    [round(f1_metrics[i].eval(average=True) * 100, 6)
                                                                     for i in
                                                                     range(args.hierarchy)]))
                # 进行验证集测试，用于选出最佳模型
                if args.early_stopping and total_step % args.dev_step == 0:
                    acc_res, f1_res = test_process(loader=loader_list[1],
                                                   model=classifier,
                                                   metrics=[acc_metrics, f1_metrics],
                                                   mode="Dev",
                                                   repeat_time=repeat_time)
                    is_break = es.apply(cur_res=[acc_res[-1], f1_res[-1]], cur_step=total_step)
                    if is_break:
                        break
                tbar.update(1)
            clear_metrics([acc_metrics, f1_metrics])
            if is_break:
                break
    # 在测试集上进行测试
    test_acc, test_f1 = test_process(loader=loader_list[2],
                                     model=classifier,
                                     metrics=[acc_metrics, f1_metrics],
                                     repeat_time=repeat_time,
                                     mode="Test",
                                     load_best_model=True)
    # 打印多次重复实验平均结果
    calculate_average_result({"Acc": test_acc, "Macro-F1": test_f1},
                             final_dict,
                             repeat_time,
                             total_repeat=args.total_repeat)


def test_process(loader,
                 metrics,
                 mode="Test",
                 load_best_model=False,
                 repeat_time=-1,
                 model=None):
    """
    验证和测试过程
    :type load_best_model: 是否加载最佳模型进行验证
    :type repeat_time: 重复实验次数
    :param loader: 数据集
    :param model: 传入的模型
    :param metrics: 计算指标
    :param mode: Test or Dev
    :return:
    """
    hierarchy_key = ["top_level", "second_level", "third_level"]
    if load_best_model and mode == "Test":
        # 加载验证集上效果最佳模型
        model = MultiLabelClassifier.load_model(param_dict=model.get_basic_params(),
                                                repeat_time=repeat_time)

    # 在测试集上验证效果
    model.set_train(False)
    clear_metrics(metrics)
    with tqdm(total=len(loader), leave=True, position=0) as tbar:
        for idx, eval_data in enumerate(loader.create_dict_iterator(num_epochs=1, output_numpy=False)):
            eval_output = model(eval_data)
            for metric_list in metrics:
                for i in range(args.hierarchy):
                    metric_list[i].update(eval_output["output"][hierarchy_key[i]],
                                          eval_data[hierarchy_key[i]])
            tbar.update(1)
    acc_res = [round(metrics[0][i].eval() * 100, 6) for i in range(args.hierarchy)]
    f1_res = [round(metrics[1][i].eval(average=True) * 100, 6) for i in range(args.hierarchy)]
    # 结束时打印一下测试集结果
    log.info("{} Acc: {}".format(mode, acc_res))
    log.info("{} Macro f1: {}".format(mode, f1_res))
    model.set_train(True)
    return acc_res, f1_res


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser(description="PEMI Mindspore")
    # 基本设置
    parser.add_argument('-d', '--device', default="GPU", type=str, choices=["GPU", "CPU", "Ascend"])
    parser.add_argument('-di', '--device_id', nargs='+')
    parser.add_argument("-dn", "--dataset_name", default="PDTBDataset", type=str)
    # 模型设置
    parser.add_argument("-im", "--input_form", default="<p:4><sen1><p:4><mask><p:4><sep><p:4><sen2><p:4>", type=str)
    # parser.add_argument("-im", "--input_form", default="<cls><sen1><sen2>", type=str)
    parser.add_argument("-hi", "--hierarchy", default=3, type=int, choices=[1, 2, 3])
    parser.add_argument("-mn", "--model_name", default="roberta-base", type=str)
    parser.add_argument("-f", "--freeze", action="store_true")
    # parser.add_argument("-cn", "--classifier_name", default="MultiLabelClassifier", type=str)
    parser.add_argument("-lm", "--label_mode", default=0, type=int, choices=range(1, 7))
    # 训练设置
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
