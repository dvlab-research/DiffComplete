import logging
import os
import shutil
import warnings
import mcubes
import numpy as np
import torch
import torch.nn as nn
from lib.distributed import get_world_size, all_gather, is_master_proc
from models.diffusion import load_diff_model, initialize_diff_model
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from skimage.measure import marching_cubes

from lib.utils import Timer, AverageMeter
from lib.visualize import visualize_mesh

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def test(model, control_model, data_loader, config):


  is_master = is_master_proc(config.exp.num_gpus) if config.exp.num_gpus > 1 else True
  cur_device = torch.cuda.current_device()
  global_timer, iter_timer = Timer(), Timer()

  bs = config.test.test_batch_size // config.exp.num_gpus

  model.eval()
  control_model.eval()

  if is_master:
      logging.info('===> Start testing')
  global_timer.tic()

  # Clear cache (when run in test mode, cleanup training cache)
  torch.cuda.empty_cache()

  # Split test data into different gpus
  test_cnt = len(data_loader) // config.exp.num_gpus
  test_iter = int(config.net.control_weights[:-4].split('iter')[1])

  cls = config.data.class_id
  save_folder = 'completion_results'
  os.makedirs(save_folder, exist_ok=True)
  save_folder = os.path.join(save_folder, str(cls), str(test_iter))
  os.makedirs(save_folder, exist_ok=True)
  noise_folder = os.path.join(save_folder, 'noise')
  os.makedirs(noise_folder, exist_ok=True)

  npz_folder = os.path.join('completion_results_npz', str(cls), str(test_iter))
  os.makedirs(npz_folder, exist_ok=True)

  # Setting of Diffusion Models
  clip_noise = config.test.clip_noise
  use_ddim = config.test.use_ddim
  ddim_eta = config.test.ddim_eta
  betas = get_named_beta_schedule(config.diffusion.beta_schedule,
                                  config.diffusion.step,
                                  config.diffusion.scale_ratio)
  DiffusionClass = load_diff_model(config.diffusion.test_model)
  diffusion_model = initialize_diff_model(DiffusionClass, betas, config)

  data_iter = data_loader.__iter__()

  iter_timer.tic()


  if config.exp.num_gpus == 1:

      with torch.no_grad():
          for m in range(test_cnt):
              scan_ids, observe, gt = data_iter.next()
              sign = observe[:, 1].numpy()
              bs = observe.size(0)
              noise = None
              model_kwargs = {
                  'noise_save_path': os.path.join(noise_folder, f'{scan_ids[0]}noise.pt')}
              model_kwargs["hint"] = observe.to(cur_device) # torch.Size([1, 2, 32, 32, 32])

              # # # Visualize range scans (by SDF)
              # for i in range(len(observe)):
              #     single_observe = observe[i]
              #     obs_sdf = single_observe[0].numpy()
              #     scan_id = scan_ids[i]
              #     sdf_vertices, sdf_traingles = mcubes.marching_cubes(obs_sdf, 0.5)
              #     out_file = os.path.join(save_folder, f'{scan_id}input.obj')
              #     mcubes.export_obj(sdf_vertices, sdf_traingles, out_file)
              #     # print(f"Save {out_file}!")
              #
              # # Visualize GT DF
              # gt_df = gt.numpy()
              # for i in range(len(gt_df)):
              #     gt_single = gt_df[i]
              #     scan_id = scan_ids[i]
              #     vertices, traingles = mcubes.marching_cubes(gt_single, 0.5)
              #     # vertices = (vertices.astype(np.float32) - 0.5) / config.exp.res - 0.5
              #     out_file = os.path.join(save_folder, f'{scan_id}gt.obj')
              #     mcubes.export_obj(vertices, traingles, out_file)
              #     # print(f"Save {out_file}!")

              if use_ddim:
                  low_samples = diffusion_model.ddim_sample_loop(model=model,
                                                                 shape=[bs, 1] + [config.exp.res] * 3,
                                                                 device=cur_device,
                                                                 clip_denoised=clip_noise, progress=True,
                                                                 noise=noise,
                                                                 eta=ddim_eta,
                                                                 model_kwargs=model_kwargs).detach()
              else:
                  low_samples = diffusion_model.p_sample_loop(model=model,
                                                              control_model=control_model,
                                                              shape=[bs, 1] + [config.exp.res] * 3,
                                                              device=cur_device,
                                                              clip_denoised=clip_noise, progress=True, noise=noise,
                                                              model_kwargs=model_kwargs).detach()

              low_samples = low_samples.cpu().numpy()[:, 0]
              if config.data.log_df == True:
                  low_samples = np.exp(low_samples) - 1
              low_samples = np.clip(low_samples, 0, config.data.trunc_distance)

              # Visualize predicted DF
              for i in range(len(low_samples)):
                  low_sample = low_samples[i]
                  scan_id = scan_ids[i]
                  # You can choose more advanced surface extraining methods for TDF outputs
                  vertices, traingles = mcubes.marching_cubes(low_sample, 0.5)
                  out_file = os.path.join(save_folder, f'{scan_id}output.obj')
                  mcubes.export_obj(vertices, traingles, out_file)
                  out_npy_file = os.path.join(npz_folder, f'{scan_id}output.npy')
                  np.save(out_npy_file, low_sample)

  else:
      with torch.no_grad():
          for scan_id, observe, gt in data_loader:
              sign = observe[:, 1].numpy()
              noise = None
              model_kwargs = {
                  'noise_save_path': os.path.join(noise_folder, f'{scan_id[0]}noise.pt')}
              model_kwargs["hint"] = observe.to(cur_device)  # torch.Size([1, 2, 32, 32, 32])

              if use_ddim:
                  low_samples = diffusion_model.ddim_sample_loop(model=model,
                                                                 shape=[bs, 1] + [config.exp.res] * 3,
                                                                 device=cur_device,
                                                                 clip_denoised=clip_noise, progress=True,
                                                                 noise=noise,
                                                                 eta=ddim_eta,
                                                                 model_kwargs=model_kwargs).detach()
              else:
                  low_samples = diffusion_model.p_sample_loop(model=model,
                                                              control_model=control_model,
                                                              shape=[bs, 1] + [config.exp.res] * 3,
                                                              device=cur_device,
                                                              clip_denoised=clip_noise, progress=True, noise=noise,
                                                              model_kwargs=model_kwargs).detach()

              low_samples = low_samples.cpu().numpy()[:, 0]
              if config.data.log_df == True:
                  low_samples = np.exp(low_samples) - 1
              low_samples = np.clip(low_samples, 0, config.data.trunc_distance)

              # You can visualize the results here

  iter_time = iter_timer.toc(False)
  global_time = global_timer.toc(False)
