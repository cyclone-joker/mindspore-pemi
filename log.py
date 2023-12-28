# @Time 2023/12/28 10:52
# @Author: Joker
# @File: logs.py
# @Software: PyCharm

from model_config import BertConfig
import logging
import os


class Log:
    def __init__(self):
        self.log_path = BertConfig.log_save_path
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        self.formatter = logging.Formatter(fmt)
        self.sh = logging.StreamHandler()
        self.sh.setLevel(logging.INFO)
        self.sh.setFormatter(self.formatter)
        self.fh = logging.FileHandler(os.path.join(self.log_path, "pemi_log.txt"))
        self.fh.setFormatter(self.formatter)
        self.log = logging.getLogger("mindspore pemi")
        self.log.setLevel(logging.INFO)
        self.log.addHandler(self.sh)
        self.log.addHandler(self.fh)

    def get_log(self):
        return self.log


logger = Log()
log = logger.get_log()