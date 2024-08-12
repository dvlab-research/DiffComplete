# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC
from pathlib import Path
from collections import defaultdict

import random
import numpy as np
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader

import datasets.transforms as t
from datasets.dataloader import InfSampler, DistributedInfSampler
from lib.distributed import get_world_size


class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Test = 2
  Debug = 3


def datasetphase_2str(arg):
  if arg == DatasetPhase.Train:
    return 'train'
  elif arg == DatasetPhase.Val:
    return 'val'
  elif arg == DatasetPhase.Test:
    return 'test'
  else:
    raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
  if arg.upper() == 'TRAIN':
    return DatasetPhase.Train
  elif arg.upper() == 'VAL':
    return DatasetPhase.Val
  elif arg.upper() == 'TEST':
    return DatasetPhase.Test
  else:
    raise ValueError('phase must be one of train/val/test')


class DictDataset(Dataset, ABC):

    def __init__(self,
                 data_paths,
                 input_transform=None,
                 target_transform=None,
                 cache=False,
                 data_root='/'):
        """
        data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
        """
        Dataset.__init__(self)

        # Allows easier path concatenation
        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.data_root = data_root
        self.data_paths = sorted(data_paths)

        self.input_transform = input_transform
        self.target_transform = target_transform

        # dictionary of input
        self.data_loader_dict = {
            'input': (self.load_input, self.input_transform),
            'target': (self.load_target, self.target_transform)
        }

        # For large dataset, do not cache
        self.cache = cache
        self.cache_dict = defaultdict(dict)
        self.loading_key_order = ['input', 'target']

    def load_input(self, index):
        raise NotImplementedError

    def load_target(self, index):
        raise NotImplementedError

    def get_classnames(self):
        pass

    def reorder_result(self, result):
        return result

    def __getitem__(self, index):
        out_array = []
        for k in self.loading_key_order:
            loader, transformer = self.data_loader_dict[k]
            v = loader(index)
            if transformer:
                v = transformer(v)
            out_array.append(v)
        return out_array

    def __len__(self):
        return len(self.data_paths)