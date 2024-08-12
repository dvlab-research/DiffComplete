import models.diffusion.gaussian_diffusion as gaussian_diffusion
from models.diffusion.gaussian_diffusion import space_timesteps
from models.diffusion.common import ModelMeanType, ModelVarType, LossType

ModelMeanTypeDict= {
  'PREVIOUS_X': ModelMeanType.PREVIOUS_X,
  'START_X': ModelMeanType.START_X,
  'EPSILON': ModelMeanType.EPSILON
}

ModelVarTypeDict= {
  'LEARNED': ModelVarType.LEARNED,
  'FIXED_SMALL': ModelVarType.FIXED_SMALL,
  'FIXED_LARGE': ModelVarType.FIXED_LARGE,
  'LEARNED_RANGE': ModelVarType.LEARNED_RANGE
}

LossTypeDict= {
  'MSE': LossType.MSE,
  'RESCALED_MSE': LossType.RESCALED_MSE,
  'KL': LossType.KL,
  'RESCALED_KL': LossType.RESCALED_KL
}

MODELS = []

def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Diffusion' in a])

add_models(gaussian_diffusion)


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS

def load_diff_model(name):
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

def initialize_diff_model(DiffusionClass, betas, config):
  model_var_type = ModelVarTypeDict[config.diffusion.model_var_type]
  model_mean_type = ModelMeanTypeDict[config.diffusion.model_mean_type]
  loss_type = LossTypeDict[config.diffusion.loss_type]

  if not 'Spaced' in DiffusionClass.__name__:
    model = DiffusionClass(betas=betas,
                            model_var_type=model_var_type,
                            model_mean_type=model_mean_type,
                            loss_type=loss_type,
                            rescale_timesteps=config.diffusion.rescale_timestep
                            if hasattr(config.diffusion, 'rescale_timestep') else False # False
                           )
  else:
    respacing = [config.diffusion.step // config.diffusion.respacing]
        
    model = DiffusionClass(use_timesteps=space_timesteps(config.diffusion.step, respacing),
                           betas=betas,
                           model_var_type=model_var_type,
                           model_mean_type=model_mean_type,
                           loss_type=loss_type
                           )

  return model