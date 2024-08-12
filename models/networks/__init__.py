import torch.nn as nn
from models.networks import resunet3d, controlnet

# Import network
MODELS = []

def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])

add_models(resunet3d)
add_models(controlnet)

def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS

def load_network(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass

def initialize_network(NetClass, config):
  if not isinstance(NetClass, type) or not issubclass(NetClass, nn.Module):
    raise TypeError("network class must be a subclass of nn.Module")

  if isinstance(config.net.channel_mult, str):
    config.net.channel_mult = list(map(int,config.net.channel_mult.split(',')))

  if not config.net.attention_resolutions:
    config.net.attention_resolutions = []
  else:
    config.net.attention_resolutions = list(map(int,config.net.attention_resolutions.split(',')))

  model = NetClass(
    in_channels=config.net.in_channels, model_channels=config.net.model_channels,
    out_channels=2 if hasattr(config.diffusion, 'diffusion_learn_sigma')
                      and config.diffusion.diffusion_learn_sigma else 1,  # 1
    num_res_blocks=config.net.num_res_blocks,  # 3
    channel_mult=config.net.channel_mult,  # (1, 2, 2, 2)
    attention_resolutions=config.net.attention_resolutions,  # []
    dropout=0,
    dims=3,
    activation=config.net.unet_activation if hasattr(config.net, 'unet_activation') else None
  )

  return model

def initialize_controlnet(NetClass, config):
  if not isinstance(NetClass, type) or not issubclass(NetClass, nn.Module):
    raise TypeError("network class must be a subclass of nn.Module")

  if isinstance(config.net.channel_mult, str):
    config.net.channel_mult = list(map(int,config.net.channel_mult.split(',')))

  if not config.net.attention_resolutions:
    config.net.attention_resolutions = []
  else:
    config.net.attention_resolutions = list(map(int,config.net.attention_resolutions.split(',')))

  model = NetClass(
    in_channels=config.net.in_channels, model_channels=config.net.model_channels,
    hint_channels = config.net.hint_channels,
    out_channels=2 if hasattr(config.diffusion, 'diffusion_learn_sigma')
                      and config.diffusion.diffusion_learn_sigma else 1,  # 1
    num_res_blocks=config.net.num_res_blocks,  # 3
    channel_mult=config.net.channel_mult,  # (1, 2, 2, 2)
    attention_resolutions=config.net.attention_resolutions,  # []
    dropout=0,
    dims=3,
    activation=config.net.unet_activation if hasattr(config.net, 'unet_activation') else None
  )

  return model
