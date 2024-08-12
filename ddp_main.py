# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import sys
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import argparse
import shutil
import time
from tools.ddp_trainer import DiffusionTrainer
from lib.distributed import multi_proc_run, ErrorHandler
import random


def single_proc_run(config):
    if not torch.cuda.is_available():
        raise Exception('No GPUs FOUND.')
    trainer = DiffusionTrainer(config)
    if config.train.is_train:
        trainer.train()
    else:
        trainer.test()

def get_args():
    parser = argparse.ArgumentParser('DiffComplete')
    parser.add_argument('--config', type=str, help='name of config file')
    args = parser.parse_args()
    return args


@hydra.main(config_path='configs', config_name='epn_control_train.yaml')
def main(config):
    # fix seed
    np.random.seed(config.misc.seed)
    torch.manual_seed(config.misc.seed)
    torch.cuda.manual_seed(config.misc.seed)

    # Convert to dict
    if config.exp.num_gpus > 1:
        multi_proc_run(config.exp.num_gpus, fun=single_proc_run, fun_args=(config,))
    else:
        single_proc_run(config)

if __name__ == '__main__':
    __spec__ = None
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # os.environ["OMP_NUM_THREADS"] = "4"
    main()