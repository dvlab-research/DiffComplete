import numpy as np
import logging
import os
import sys
import torch
import torch.nn.functional as F
import tqdm

from torch import nn
from torch.serialization import default_restore_location
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from lib.distributed import get_world_size, all_gather, is_master_proc
from lib.solvers import initialize_optimizer, initialize_scheduler
from datasets import load_dataset, initialize_data_loader
from lib.utils import checkpoint, checkpoint_control, Timer, AverageMeter, load_state_with_same_shape, count_parameters

from models.networks import load_network, initialize_network, initialize_controlnet
from models.diffusion import load_diff_model, initialize_diff_model
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from models.modules.resample import UniformSampler, LossSecondMomentResampler, LossAwareSampler

from tools.test import test as test_

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class DiffusionTrainer:
    def __init__(self, config):

        self.is_master = is_master_proc(config.exp.num_gpus) if config.exp.num_gpus > 1 else True
        self.cur_device = torch.cuda.current_device()

        # Load the configurations
        self.setup_logging()

        # Use the previous configurations if the training breaks
        # if config.train.is_train == True and config.train.debug == False:
        #     if os.path.exists('config.yaml'):
        #         logging.info('===> Loading exsiting config file')
        #         config = OmegaConf.load('config.yaml')
        #         logging.info('===> Loaded exsiting config file')
        #     logging.info('===> Configurations')
        #     logging.info(config)

        # Dataloader
        DatasetClass = load_dataset(config.data.dataset)
        logging.info('===> Initializing dataloader')
        self.train_data_loader = initialize_data_loader(
            DatasetClass, config, phase=config.train.train_phase,
            num_workers=config.exp.num_workers, augment_data=False,
            shuffle=True, repeat=True, collate=config.data.collate_fn,
            batch_size=config.exp.batch_size // config.exp.num_gpus,
            persistent_workers=config.data.persistent_workers
        )

        self.test_data_loader = initialize_data_loader(
            DatasetClass, config, phase=config.test.test_phase,
            num_workers=config.exp.num_workers, augment_data=False,
            shuffle=config.test.partial_shape==None, repeat=False, collate=config.data.collate_fn,
            batch_size=config.test.test_batch_size // config.exp.num_gpus,
            persistent_workers=config.data.persistent_workers
        )

        # Main network initialization
        logging.info('===> Building model')
        NetClass = load_network(config.net.network)
        model = initialize_network(NetClass, config)

        # ControlNet initialization
        logging.info('===> Building model')
        ControlNet = load_network(config.net.controlnet)
        control_model = initialize_controlnet(ControlNet, config)

        logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
        logging.info('===> Number of trainable parameters: {}: {}'.format(ControlNet.__name__, count_parameters(control_model)))
        logging.info(model)
        logging.info(control_model)

        # Load weights for the main network and control network
        if config.net.weights:
            logging.info('===> Loading weights: ' + config.net.weights)
            state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            matched_weights = load_state_with_same_shape(model, state['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(matched_weights)
            model.load_state_dict(model_dict)

        if config.net.control_weights:
            logging.info('===> Loading weights: ' + config.net.control_weights)
            config.net.control_weights = default(config.net.control_weights, config.net.weights)
            control_state = torch.load(config.net.control_weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))

            control_matched_weights = load_state_with_same_shape(control_model, control_state['state_dict'])
            control_model_dict = control_model.state_dict()
            control_model_dict.update(control_matched_weights)
            control_model.load_state_dict(control_model_dict)

        model = model.cuda()
        if config.exp.num_gpus > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[self.cur_device],
                output_device=self.cur_device,
                broadcast_buffers=False,
                # find_unused_parameters=True
            )

        control_model = control_model.cuda()
        if config.exp.num_gpus > 1:
            control_model = torch.nn.parallel.DistributedDataParallel(
                module=control_model, device_ids=[self.cur_device],
                output_device=self.cur_device,
                broadcast_buffers=False,
            )

        self.config = config
        self.skip_validate = config.exp.skip_validate
        self.model = model
        self.control_model = control_model

        # Diffusion model
        # linear, 1000, 1.0
        betas = get_named_beta_schedule(config.diffusion.beta_schedule,
                                        config.diffusion.step,
                                        config.diffusion.scale_ratio)
        DiffusionClass = load_diff_model(config.diffusion.model)
        self.diffusion_model = initialize_diff_model(DiffusionClass, betas, config)


        # Sample
        if config.diffusion.sampler == 'uniform':
            self.sampler = UniformSampler(self.diffusion_model)
        elif config.diffusion.sampler == 'second-order':
            self.sampler = LossSecondMomentResampler(self.diffusion_model)
        else:
            raise Exception("Unknown Sampler.....")

        if self.is_master:
            self.writer = SummaryWriter(log_dir='tensorboard')

        self.optimizer, self.scheduler = self.configure_optimizers(config)

        # # fix parameters for training
        # if config.train.fine_tune_encoder == True:
        #     assert config.net.weights is not None, "Please specify the pre-trained weights for fine-tuning"
        #     for name, p in self.model.named_parameters():
        #         if 'time_embed' in name or 'input_blocks' in name or 'middle_block' in name:
        #             p.requires_grad = True
        #         else:
        #             p.requires_grad = False

        # Mixed precision training
        if hasattr(config.train, 'mix_precision') and self.config.train.mix_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Continue training from the checkpoint (TBD)
        if config.train.is_train:
            checkpoint_fn = 'weights/weights.pth'
            self.min_loss = 100
            self.best = -1
            self.curr_iter, self.epoch, self.is_training = 1, 1, True

    def configure_optimizers(self, config):
        params = list(self.control_model.parameters())
        params += list(self.model.parameters())

        optimizer = initialize_optimizer(params, config.optimizer)
        if config.optimizer.lr_decay:  # False
            scheduler = initialize_scheduler(self.optimizer, config.optimizer)
        else:
            scheduler = None
        return optimizer, scheduler

    def setup_logging(self):

        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.WARN)
        if self.is_master:
            logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(
            format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
            datefmt='%m/%d %H:%M:%S',
            handlers=[ch])

    def load_state(self, state):
        if get_world_size() > 1:
            _model = self.model.module
        else:
            _model = self.model
        _model.load_state_dict(state)

    def set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.config.misc.seed + self.curr_iter
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test(self):
        return test_(self.model, self.control_model, self.test_data_loader, self.config)


    def validate(self):
        if not self.skip_validate:
            val_loss, val_score, _, = test_(self.model, self.val_data_loader, self.config)
            self.writer.add_scalar('val/loss', val_loss, self.curr_iter)
            self.writer.add_scalar('val/score', val_score, self.curr_iter)

            if val_score > self.best:
                self.best = val_score
                self.best_iter = self.curr_iter
                checkpoint(self.model, self.optimizer, self.epoch, self.curr_iter, self.config,
                           self.best, self.scaler, postfix="best")
                logging.info("Current best score: {:.3f} at iter {}".format(self.best, self.best_iter))

        checkpoint(self.model, self.optimizer, self.epoch, self.curr_iter, self.config, self.best, self.scaler)
        checkpoint_control(self.control_model, self.optimizer, self.epoch, self.curr_iter, self.config, self.best, self.scaler)

    def train(self):
        ## To be checked
        self.model.train()
        self.control_model.train()

        # Configuration
        data_timer, iter_timer = Timer(), Timer()
        fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()
        data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
        fw_time_avg, bw_time_avg, ddp_time_avg = AverageMeter(), AverageMeter(), AverageMeter()

        losses = {
            'total_loss': AverageMeter(),
            'mse_loss': AverageMeter()
        }

        # Train the network
        logging.info('===> Start training on {} GPUs, batch-size={}'.format(
            get_world_size(), self.config.exp.batch_size))

        data_iter = self.train_data_loader.__iter__()  # (distributed) infinite sampler
        while self.is_training:
            for _ in range(len(self.train_data_loader)):
                self.optimizer.zero_grad()
                data_time = 0
                batch_losses = {'total_loss': 0.0,
                                'mse_loss': 0.0}
                iter_timer.tic()

                # set random seed for every iteration for trackability
                self.set_seed()
                total_loss = 0.0
                mse_loss = 0.0
                # Get training data
                data_timer.tic()
                scan_id, input_sdf, gt_df = data_iter.next()
                shape_gt = gt_df.unsqueeze(1).to(self.cur_device)

                if self.config.data.dataset != 'ControlledEPNDataset':
                    input_sdf = input_sdf.unsqueeze(1).to(self.cur_device)
                else:
                    input_sdf = input_sdf.to(self.cur_device)

                data_time += data_timer.toc(False)
                # Feed forward
                fw_timer.tic()

                if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
                    with autocast():
                        t, t_weights = self.sampler.sample(shape_gt.size(0), device=self.cur_device)
                        iterative_loss = self.diffusion_model.training_losses(model=self.model,
                                                                              control_model=self.control_model,
                                                                              x_start=shape_gt,
                                                                              hint=input_sdf,
                                                                              t=t,
                                                                              weighted_loss=self.config.train.weighted_loss)
                        mse_loss += torch.mean(iterative_loss['loss'] * t_weights)
                else:
                    t, t_weights = self.sampler.sample(shape_gt.size(0), device=self.cur_device)
                    iterative_loss = self.diffusion_model.training_losses(model=self.model,
                                                                          control_model=self.control_model,
                                                                          x_start=shape_gt,
                                                                          hint=input_sdf,
                                                                          t=t,
                                                                          weighted_loss=self.config.train.weighted_loss)
                    mse_loss += torch.mean(iterative_loss['loss'] * t_weights)

                # Compute and accumulate gradient
                total_loss += mse_loss

                # bp the loss
                fw_timer.toc(False)
                bw_timer.tic()

                if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                if self.config.train.use_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip_value)

                bw_timer.toc(False)

                # gather information
                logging_output = {'total_loss': total_loss.item(), 'mse_loss': mse_loss.item()}

                ddp_timer.tic()
                if self.config.exp.num_gpus > 1:
                    logging_output = all_gather(logging_output)
                    logging_output = {w: np.mean([
                        a[w] for a in logging_output]
                    ) for w in logging_output[0]}

                batch_losses['total_loss'] += logging_output['total_loss']
                batch_losses['mse_loss'] += logging_output['mse_loss']
                ddp_timer.toc(False)

                # Update number of steps
                if hasattr(self.config.train, 'mix_precision') and self.config.train.mix_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # print(self.model.state_dict()['input_blocks.14.0.in_layers.0.bias'])
                # print(self.model.state_dict()['output_blocks.15.0.in_layers.2.bias'])

                data_time_avg.update(data_time)
                iter_time_avg.update(iter_timer.toc(False))
                fw_time_avg.update(fw_timer.diff)
                bw_time_avg.update(bw_timer.diff)
                ddp_time_avg.update(ddp_timer.diff)

                losses['total_loss'].update(batch_losses['total_loss'], shape_gt.size(0))
                losses['mse_loss'].update(batch_losses['mse_loss'], shape_gt.size(0))

                if self.curr_iter >= self.config.train.max_iter:
                    self.is_training = False
                    break

                last_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.state_dict()['param_groups'][0]['lr']
                if self.curr_iter % self.config.train.stat_freq == 0 or self.curr_iter == 1:
                    # lrs = ', '.join(['{:.3e}'.format(x) for x in last_lr])
                    # debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}, LR: {}\t".format(
                    #     self.epoch, self.curr_iter, len(self.train_data_loader),
                    #     losses['total_loss'].avg, lrs)
                    lr = '{:.3e}'.format(last_lr)
                    debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}, LR: {}\t".format(
                        self.epoch, self.curr_iter, len(self.train_data_loader),
                        losses['total_loss'].avg, lr)
                    debug_str += "Data time: {:.4f}, Forward time: {:.4f}, Backward time: {:.4f}, DDP time: {:.4f}, Total iter time: {:.4f}".format(
                        data_time_avg.avg, fw_time_avg.avg, bw_time_avg.avg, ddp_time_avg.avg,
                        iter_time_avg.avg)
                    logging.info(debug_str)
                    # Reset timers
                    data_time_avg.reset()
                    iter_time_avg.reset()

                    # Write logs
                    if self.is_master:
                        self.writer.add_scalar('train/loss', losses['total_loss'].avg, self.curr_iter)
                        self.writer.add_scalar('train/learning_rate', last_lr, self.curr_iter)

                    # clear loss
                    losses['total_loss'].reset()
                    losses['mse_loss'].reset()

                # Validation
                if self.curr_iter % self.config.train.val_freq == 0 and self.is_master:
                    self.validate()
                    if not self.skip_validate:
                        self.model.train()
                        self.control_model.train()


                if self.curr_iter % self.config.train.empty_cache_freq == 0:
                    # Clear cache
                    torch.cuda.empty_cache()

                # End of iteration
                self.curr_iter += 1

            # max_memory_allocated = torch.cuda.max_memory_allocated(self.cur_device) / (1024 ** 2)
            # logging.info(f"End of Epoch {self.epoch + 1}, Max memory allocated: {max_memory_allocated:.2f} MiB")

            self.epoch += 1

        # Explicit memory cleanup
        if hasattr(data_iter, 'cleanup'):
            data_iter.cleanup()

        # max_memory_allocated = torch.cuda.max_memory_allocated(self.cur_device) / (1024 ** 2)
        # print(f"End of training, Max memory allocated: {max_memory_allocated:.2f} MiB")

        # Save the final model
        if self.is_master:
            self.validate()