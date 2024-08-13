# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import errno
import time
import torch
import numpy as np
from omegaconf import OmegaConf
from lib.distributed import get_world_size

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def load_state_with_same_shape(model, weights):
    # weights['conv1.kernel'] = weights['conv1.kernel'].repeat([1,3,1])/3.0
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith('module.'):
        logging.info("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('encoder.'):
        logging.info("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition('encoder.')[2]: weights[k] for k in weights.keys()}

    # print(weights.items())
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print("===================")
    # print(model_state)

    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best, scaler=None, postfix=None):
    mkdir_p('weights')
    filename = f"checkpoint_{config.net.network}_iter{iteration}.pth"
    if config.train.overwrite_weights:
        filename = f"checkpoint_{config.net.network}.pth"
    if postfix is not None:
        filename = f"checkpoint_{config.net.network}_{postfix}.pth"
    checkpoint_file = 'weights/' + filename

    _model = model.module if get_world_size() > 1 else model
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.net.network,
        'state_dict': _model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    if hasattr(config.train, 'mix_precision') and config.train.mix_precision:
        state['scalar'] = scaler.state_dict()

    if best is not None:
        state['best'] = best
        state['best_iter'] = iteration

    # Save config
    OmegaConf.save(config, 'config.yaml')

    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")

    if postfix == None:
        # Delete symlink if it exists
        if os.path.exists('weights/weights.pth'):
            os.remove('weights/weights.pth')
        # Create symlink
        os.system('ln -s {} weights/weights.pth'.format(filename))


def checkpoint_control(model, optimizer, epoch, iteration, config, best, scaler=None, postfix=None):
    mkdir_p('weights')
    filename = f"checkpoint_{config.net.controlnet}_iter{iteration}.pth"
    if config.train.overwrite_weights:
        filename = f"checkpoint_{config.net.controlnet}.pth"
    if postfix is not None:
        filename = f"checkpoint_{config.net.controlnet}_{postfix}.pth"
    checkpoint_file = 'weights/' + filename

    _model = model.module if get_world_size() > 1 else model
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'arch': config.net.controlnet,
        'state_dict': _model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    if hasattr(config.train, 'mix_precision') and config.train.mix_precision:
        state['scalar'] = scaler.state_dict()

    if best is not None:
        state['best'] = best
        state['best_iter'] = iteration


    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")

    if postfix == None:
        # Delete symlink if it exists
        if os.path.exists('weights/weights.pth'):
            os.remove('weights/weights.pth')
        # Create symlink
        os.system('ln -s {} weights/weights.pth'.format(filename))

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
                            (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def debug_on():
    import sys
    import pdb
    import functools
    import traceback

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
    return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

    def __init__(self, prime=5279):
        self.prime = prime

    def __call__(self, time, batch):
        return self.hash(time, batch)

    def hash(self, time, batch):
        return self.prime * batch + time

    def dehash(self, key):
        time = key % self.prime
        batch = key / self.prime
        return time, batch


