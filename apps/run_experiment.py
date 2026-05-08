import torch
import os
import yaml
from shutil import copy
from argparse import ArgumentParser
from time import gmtime, strftime
from codec import generator as rdac_generator, kpd as rdac_kpd, discriminator as rdac_discriminator
from apps.train import train_functions
from apps.test import test_functions
from data.dataset import FramesDataset, HDACFramesDataset, MRFramesDataset

DATASET_MAP = {
    'mvac': MRFramesDataset, 'mrdac': MRFramesDataset,
    'hdac': HDACFramesDataset, 'hdac_hf': HDACFramesDataset,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "compress", "test"])
    parser.add_argument("--model_id", default=None, help="model id to use (overrides config filename)")
    parser.add_argument("--project_id", default='Animation-Based-Codecs', help="project name")
    parser.add_argument("--log_dir", default='artifacts', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="Use pretrained generator and kp detector")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--debug", action="store_true", help="Test on one batch to debug")
    parser.add_argument("--num_workers", default=4, type=int, help="num of cpu cores for dataloading")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_id = opt.model_id or os.path.basename(opt.config).split('.')[0]
    device = opt.device_ids[0] if torch.cuda.is_available() else 'cpu'

    if opt.mode == 'train':
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1]) if opt.checkpoint else \
                  os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0]) + '_' + strftime("%d_%m_%y_%H_%M_%S", gmtime())
    else:
        log_dir = os.path.join(opt.log_dir, model_id)

    # Build models and move to device
    generator_params = {**config['model_params']['common_params'], **config['model_params']['generator_params']}
    generator = rdac_generator.RDAC_Generator(**generator_params).to(device)
    print(f"##..{generator.__class__.__name__} LOADED..##")

    kpd_params = {**config['model_params']['common_params'], **config['model_params']['kp_detector_params']}
    kp_detector = rdac_kpd.RDAC_KPD(**kpd_params).to(device)

    disc_params = {**config['model_params']['common_params'], **config['model_params']['discriminator_params']}
    discriminator = rdac_discriminator.MultiScaleDiscriminator(**disc_params).to(device)

    # Select dataset class
    dataset_cls = DATASET_MAP.get(model_id, FramesDataset)
    dataset = dataset_cls(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir + '/img_aug', exist_ok=True)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)

        params = {'project_id': opt.project_id, 'debug': opt.debug, 'model_id': model_id,
                  'checkpoint': opt.checkpoint, 'log_dir': log_dir,
                  'device_ids': opt.device_ids, 'num_workers': opt.num_workers}
        train_functions[model_id](config, dataset, generator, kp_detector, discriminator, **params)

    elif opt.mode == 'test':
        params = {'model_id': model_id, 'checkpoint': opt.checkpoint, 'log_dir': log_dir}
        test_functions[model_id](config, dataset, generator, kp_detector, **params)
