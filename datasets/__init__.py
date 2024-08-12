from torch.utils.data import DataLoader
from lib.distributed import get_world_size
from datasets.dataloader import InfSampler, DistributedInfSampler, SequentialDistributedSampler
import datasets.transforms as t
from datasets.dataset import str2datasetphase_type, DatasetPhase

from datasets import epn_control

DATASETS = []

def add_datasets(module):
  DATASETS.extend([getattr(module, a) for a in dir(module) if 'Dataset' in a])

add_datasets(epn_control)

def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           num_workers,
                           shuffle,
                           repeat,
                           collate,
                           augment_data,
                           batch_size,
                           input_transform=None,
                           target_transform=None,
                           persistent_workers=False):
    if isinstance(phase, str):
        phase = str2datasetphase_type(phase)

    # Transform: currently None
    transform_train = []
    if augment_data:
        if input_transform is not None:
            transform_train += input_transform

    if len(transform_train) > 0:
        transforms = t.Compose(transform_train)
    else:
        transforms = None

    dataset = DatasetClass(
        config,
        input_transform=transforms,
        target_transform=target_transform,
        cache=config.data.cache_data,
        augment_data=augment_data,
        phase=phase)

    if collate:
        collate_fn = t.collate_fn_factory()

        data_args = {
            'dataset': dataset,
            'num_workers': num_workers,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'persistent_workers': persistent_workers
        }
    else:
        data_args = {
            'dataset': dataset,
            'num_workers': num_workers,
            'batch_size': batch_size,
            'persistent_workers': persistent_workers
        }

    if repeat:
        if get_world_size() > 1:
            data_args['sampler'] = DistributedInfSampler(dataset, shuffle=shuffle)
        else:
            data_args['sampler'] = InfSampler(dataset, shuffle)

    else:
        data_args['shuffle'] = shuffle

    if config.train.is_train == False and config.test.partial_shape == True and get_world_size() > 1:
        data_args['sampler'] = SequentialDistributedSampler(dataset, batch_size=batch_size)

    data_loader = DataLoader(**data_args)

    return data_loader