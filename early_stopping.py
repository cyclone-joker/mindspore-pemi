# @Time 2024/1/3 19:45
# @Author: Joker
# @File: early_stopping.py
# @Software: PyCharm
from log import log


class EarlyStopping:
    def __init__(self, model_save_func, repeat_time, max_step=5):
        self.max_step = max_step
        self.counter = 0
        self.best_res = None
        self.last_best_step = 0
        self.model_save_func = model_save_func
        self.is_break = False
        self.repeat_time = repeat_time

    def apply(self, cur_res, cur_step):
        """
        :param cur_res: 传入的是[acc,f1]的列表，只要其中一个上升，就可以继续训练
        :param cur_step: 当前步
        :return:
        """
        if self.best_res is None:
            self.best_res = cur_res
            self.last_best_step = cur_step
            self.model_save_func(step=cur_step,
                                 repeat_time=self.repeat_time)
        elif cur_res[0] <= self.best_res[0] and cur_res[1] <= self.best_res[1]:
            self.counter += 1
            if self.counter >= self.max_step:
                self.is_break = True
                log.info("早停机制触发，在{}步停止训练...".format(cur_step))

        else:
            self.best_res = cur_res
            self.last_best_step = cur_step
            # 保存模型
            self.model_save_func(step=cur_step,
                                 repeat_time=self.repeat_time)
            self.counter = 0
        return self.is_break

    def clear(self):
        self.best_res = None
        self.counter = 0
        self.is_break = False
        self.last_best_step = 0
        self.model_save_func = None
