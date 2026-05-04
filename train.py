# -*- coding: utf-8 -*-
from tqdm import trange, tqdm
import torch
import os
from trainers import gen_trainers, disc_trainers
from utilities.utils.logger import Logger
from torch.utils.data import DataLoader
from utilities.utils.dataset import DatasetRepeater
from torch.optim.lr_scheduler import MultiStepLR


def load_pretrained_model(model, path, name='generator', device='cpu'):
    cpk = torch.load(path, map_location=device)
    if name in cpk:
        model.load_state_dict(cpk[name], strict='optimizer' not in name)
    return model


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def split_params(module):
    """Split module parameters into main and quantile (aux) groups."""
    main = [p for n, p in module.named_parameters() if not n.endswith(".quantiles")]
    aux = [p for n, p in module.named_parameters() if n.endswith(".quantiles")]
    return main, aux


def train_rdac(config, dataset, generator, kp_detector, discriminator, **kwargs):
    train_params = config['train_params']
    step = train_params['step']
    aux_optimizer = None
    pretrained_cpk_path = config['dataset_params']['cpk_path']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if step == 0:
        parameters, aux_parameters = split_params(generator)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=train_params['lr_aux'])
        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'], betas=train_params['betas'])

    elif step == 1:
        freeze(generator)
        freeze(kp_detector)
        sdc_main, sdc_aux = split_params(generator.sdc.train())
        tdc_main, tdc_aux = split_params(generator.tdc.train())
        parameters = sdc_main + tdc_main
        aux_parameters = sdc_aux + tdc_aux
        unfreeze(aux_parameters)
        assert parameters, "No trainable parameters found for step 1"
        unfreeze(parameters)
        aux_optimizer = torch.optim.Adam(aux_parameters, lr=train_params['lr_aux'])
        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'], betas=train_params['betas'])

    elif step == 2:
        freeze(kp_detector)
        freeze(generator)
        parameters = list(generator.refinement_network.parameters())
        unfreeze(parameters)
        gen_optimizer = torch.optim.Adam(parameters, lr=train_params['lr'], betas=train_params['betas'])

    elif step == 3:
        freeze(kp_detector)
        freeze(generator.sdc)
        freeze(generator.tdc)
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=train_params['lr'], betas=train_params['betas'])
    else:
        raise NotImplementedError("Unknown training step [step < 0 or step > 3]")

    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if pretrained_cpk_path != '':
        for model, name in [(generator, 'generator'), (kp_detector, 'kp_detector'),
                            (gen_optimizer, 'gen_optimzer'), (discriminator, 'discriminator'),
                            (disc_optimizer, 'disc_optimizer')]:
            load_pretrained_model(model, path=pretrained_cpk_path, name=name, device=device)
        if aux_optimizer is not None:
            load_pretrained_model(aux_optimizer, path=pretrained_cpk_path, name='aux_optimzer', device=device)

    if config['model_params']['generator_params']['ref_coder']:
        tic_weights = torch.load("checkpoints/tic.pth.tar", map_location=device, weights_only=True)
        generator.ref_coder.load_state_dict(tic_weights['tic'], strict=True)

    generator_full = gen_trainers['rdac'](kp_detector, generator, discriminator, config)
    discriminator_full = disc_trainers['rdac'](discriminator, train_params,
                                               disc_type=config['model_params']['discriminator_params']['disc_type'])

    schedulers = [MultiStepLR(gen_optimizer, train_params['epoch_milestones'], gamma=0.1),
                  MultiStepLR(disc_optimizer, train_params['epoch_milestones'], gamma=0.1)]
    if aux_optimizer is not None:
        schedulers.append(MultiStepLR(aux_optimizer, train_params['epoch_milestones'], gamma=0.1))

    if torch.cuda.is_available():
        generator_full, discriminator_full = generator_full.cuda(), discriminator_full.cuda()
        if torch.cuda.device_count() > 1:
            generator_full = CustomDataParallel(generator_full)
            discriminator_full = CustomDataParallel(discriminator_full)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=kwargs['num_workers'], drop_last=True, pin_memory=True)
    res_params = config['model_params']['generator_params']['residual_coder_params']

    with Logger(log_dir=kwargs['log_dir'], visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(0, train_params['num_epochs']):
            for x in tqdm(dataloader, leave=False):
                if torch.cuda.is_available():
                    x = {k: v.cuda() for k, v in x.items()}

                params = {**kwargs, 'variable_bitrate': res_params['variable_bitrate'],
                          'bitrate_levels': res_params['levels']}
                losses_generator, generated = generator_full(x, **params)

                losses_ = {}
                for key in ('distortion', 'rate', 'perp_distortion'):
                    if key in generated:
                        losses_[key] = generated[key].mean().detach().cpu().item()

                loss = sum(val.mean() for val in losses_generator.values())
                loss.backward()
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                aux_loss = 0
                if aux_optimizer is not None:
                    aux_loss = generator.sdc.aux_loss()
                    if generator.tdc is not None:
                        aux_loss += generator.tdc.aux_loss()
                    aux_loss.backward()
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()

                # Train the discriminator network
                losses_discriminator = discriminator_full(x, generated)
                disc_loss = sum(val.mean() for val in losses_discriminator.values())
                disc_loss.backward()
                disc_optimizer.step()
                disc_optimizer.zero_grad()

                losses_generator.update(losses_discriminator)
                losses = {k: v.mean().detach().cpu().numpy() for k, v in losses_generator.items()}
                losses.update(losses_)
                if aux_loss > 0:
                    losses["aux_loss"] = aux_loss.mean().detach().cpu().item()
                logger.log_iter(losses=losses)
                if kwargs['debug']:
                    break

            for s in schedulers:
                s.step()

            state_dict = {'generator': generator, 'kp_detector': kp_detector,
                          'gen_optimizer': gen_optimizer, 'discriminator': discriminator,
                          'disc_optimizer': disc_optimizer}
            if aux_optimizer is not None:
                state_dict['aux_optimizer'] = aux_optimizer
            logger.log_epoch(epoch, state_dict, inp=x, out=generated)
            if kwargs['debug']:
                break


train_functions = {'rdac': train_rdac}


class CustomDataParallel(torch.nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)