import time
import datetime
import os
import sys
import numpy as np
import logging
import random
import configs.config as config
from shutil import copyfile
import torch
from tensorboardX import SummaryWriter

def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('wavelet')
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger

class MyDebugger():
    pre_fix = config.debug_base_folder

    def __init__(self, exp_name: str, fix_rand_seed=None, is_save_print_to_file=True,
                 config_path=os.path.join('configs', 'config.py')):
        if fix_rand_seed is not None:
            np.random.seed(seed=fix_rand_seed)
            random.seed(fix_rand_seed)
            torch.manual_seed(fix_rand_seed)
        if isinstance(exp_name, str):
            self.exp_name = exp_name
        else:
            self.exp_name = '_'.join(exp_name)
        # self._debug_dir_name = os.path.join(os.path.dirname(__file__), MyDebugger.pre_fix,
        #                                     datetime.datetime.fromtimestamp(time.time()).strftime(
        #                                         f'%Y-%m-%d_%H-%M-%S_{self.model_name}'))
        self._debug_dir_name = os.path.join(os.getcwd(), MyDebugger.pre_fix, exp_name)
        self._write_dir_name = os.path.join(self._debug_dir_name, 'tensorboard')
        # self._debug_dir_name = os.path.join(os.path.dirname(__file__), self._debug_dir_name)
        print("=================== Program Start ====================")
        print(f"Output directory: {self._debug_dir_name}")
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self._init_debug_dir()
        log_file = os.path.join(self._debug_dir_name, f'{timestamp}.log')
        self.logger = get_root_logger(log_file=log_file)
        self.logger.info("Directory {} established".format(self._debug_dir_name))
        self.writer = SummaryWriter(log_dir=self._write_dir_name)

        ######## redirect the standard output
        # if is_save_print_to_file:
        #     sys.stdout = open(self.file_path("print.log"), 'w')
        #
        #     ######## print the dir again on the log
        #     print("=================== Program Start ====================")
        #     print(f"Output directory: {self._debug_dir_name}")

        ########  copy config file to
        config_file_save_path = self.file_path(os.path.basename(config_path))
        assert os.path.exists(config_path)
        copyfile(config_path, config_file_save_path)
        self.logger.info(f"config file created at {config_file_save_path}")

    def file_path(self, file_name):
        return os.path.join(self._debug_dir_name, file_name)

    def set_direcotry_name(self, name):
        self._debug_dir_name = name

    def _init_debug_dir(self):
        # init root debug dir
        if not os.path.exists(MyDebugger.pre_fix):
            os.mkdir(MyDebugger.pre_fix)
        if not os.path.exists(self._debug_dir_name):
            os.mkdir(self._debug_dir_name)
        if not os.path.exists(self._write_dir_name):
            os.mkdir(self._write_dir_name)

    def save_text(self, idx, save_type, filepath):
        self.logger.info(f"Epoch {idx} {save_type} saved in {filepath}")

    def add_scalar(self, *scalar):
        self.writer.add_scalar(*scalar)

    @staticmethod
    def get_save_text(save_type):
        return f"{save_type} saved in "


if __name__ == '__main__':
    debugger = MyDebugger('testing')
    # file can save in the path
    file_path = debugger.file_path('file_to_be_save.txt')
