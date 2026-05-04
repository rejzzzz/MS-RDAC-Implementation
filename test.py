import os
import json
import torch
import numpy as np
import imageio
import torch.nn.functional as F
from PIL import Image
from tqdm import trange
from train import load_pretrained_model
from utilities.metrics import Metrics
from utilities.utils.visualizer import Visualizer
from utilities.image_coders import ImageCoder
from utilities.utils.coding_utils import frame2tensor, tensor2frame
from utilities.entropy_coders import KpEntropyCoder
from utilities.anchors import HEVC, VVC_VTM, VvenC


def tensor2rgb(tensor):
    '''1x3xHxW -> HxWx3 (uint8)'''
    return (tensor.detach().cpu().squeeze().numpy() * 255.0).astype(np.uint8)


class Models:
    def __init__(self, animation_model, kp_detector_model, image_coder, kp_coder):
        self.animation_model = animation_model
        self.kp_detector_model = kp_detector_model
        self.image_coder = image_coder
        self.kp_coder = kp_coder


class Inputs:
    def __init__(self, eval_params=None):
        self.num_frames = eval_params['num_frames']
        self.gop_size = eval_params['gop_size']
        self.eval_params = eval_params
        self.fps = eval_params['fps']
        self.device = 'cpu'
        self.video = None
        self.base_layer_qp = eval_params.get('bl_qp', 50)
        self.gops = []
        self.original_video = []

    def create_gops(self):
        num_gops = max(1, self.num_frames // self.gop_size)
        for idx in range(num_gops):
            self.gops.append(self.video[idx * self.gop_size: idx * self.gop_size + self.gop_size])


class Outputs:
    def __init__(self, out_path='results'):
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0
        self.decoded_video = []
        self.visualization = []
        self.animated_video = []
        self.f_dec = open(f"{out_path}/decoded.rgb", 'wb')

    def update_decoded(self, dec_frame):
        chw = tensor2rgb(dec_frame)
        chw.tofile(self.f_dec)
        self.decoded_video.append(np.transpose(chw, [1, 2, 0]))

    def update_bits_and_time(self, info):
        self.total_bits += info['bitstring_size']
        self.enc_time += info['time']['enc_time']
        self.dec_time += info['time']['dec_time']

    def get_bitrate(self, fps, num_frames):
        return (self.total_bits * fps) / (1000 * num_frames)


# --- Codec runners ---

CODEC_MAP = {'hevc': HEVC, 'vvc': VVC_VTM, 'vvenc': VvenC}

def run_codec(codec_name, gop, qp, fps=10):
    N, H, W, _ = gop.shape
    print(f"Running {codec_name.upper()}..")
    if codec_name == 'vvenc':
        params = {'qp': qp, 'fps': fps, 'frame_dim': f"{H}x{W}",
                  'gop_size': N, 'sequence': gop, 'out_path': 'vvc_logs/'}
    elif codec_name == 'vvc':
        params = {'qp': qp, 'fps': fps, 'frame_dim': [H, W],
                  'gop_size': N, 'n_frames': N, 'sequence': gop}
    else:  # hevc
        params = {'qp': qp, 'sequence': gop, 'gop_size': N,
                  'fps': fps, 'frame_dim': (H, W)}
    return CODEC_MAP[codec_name](**params).run()


class ConventionalCodec:
    def __init__(self, eval_params):
        self.codec_name = eval_params['ref_codec']
        self.num_frames = eval_params['num_frames']
        self.gop_size = eval_params['gop_size']
        self.fps = eval_params['fps']
        self.qp = eval_params['qp']
        self.video = None
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0
        self.gops = []
        self.original_video = []
        self.decoded_video = []

    def create_gops(self):
        num_gops = max(1, self.num_frames // self.gop_size)
        for idx in range(num_gops):
            self.gops.append(self.video[idx * self.gop_size: idx * self.gop_size + self.gop_size])

    def get_bitrate(self, fps=None):
        return (self.total_bits * (fps or self.fps)) / (1000 * self.num_frames)

    def run(self):
        for gop in self.gops:
            info = run_codec(self.codec_name, gop, self.qp)
            self.decoded_video.extend(info['dec_frames'])
            self.original_video.extend(list(gop))
            self.total_bits += info['bitstring_size']
            self.enc_time += info['time']['enc_time']
            self.dec_time += info['time']['dec_time']


def resize_frames(frames, scale_factor=1):
    if scale_factor == 1:
        return frames
    N, H, W, _ = frames.shape
    return np.array([np.asarray(Image.fromarray(frames[i]).resize(
        (int(H * scale_factor), int(W * scale_factor)), Image.Resampling.LANCZOS)) for i in range(N)])


def resize_video_if_needed(video, target_h, target_w):
    N, h, w, _ = video.shape
    if h != target_h or w != target_w:
        return np.array([np.array(Image.fromarray(video[i]).resize(
            (target_w, target_h), Image.LANCZOS)) for i in range(N)])
    return video


def predictive_coder(models, input_data, visualizer, out_path='results'):
    output_data = Outputs(out_path=out_path)
    for gop in input_data.gops:
        input_data.original_video.extend(gop)
        org_reference = frame2tensor(gop[0]).to(input_data.device)
        dec_reference_info = models.image_coder(org_reference)
        output_data.update_bits_and_time(dec_reference_info)

        reference_frame = dec_reference_info['decoded']
        if isinstance(reference_frame, np.ndarray):
            reference_frame = frame2tensor(reference_frame)
        reference_frame = reference_frame.to(input_data.device)

        with torch.no_grad():
            kp_reference = models.kp_detector_model(reference_frame)
            models.kp_coder.kp_reference = kp_reference
            ref_fts = models.animation_model.reference_ft_encoder(reference_frame)

            output_data.decoded_video.append(tensor2frame(reference_frame))
            output_data.animated_video.append(tensor2frame(reference_frame))

            prev_latent = None
            for idx in trange(1, input_data.gop_size):
                target_frame = frame2tensor(gop[idx]).to(input_data.device)
                kp_target = models.kp_detector_model(target_frame)
                kp_coding_info = models.kp_coder.encode_kp(kp_target=kp_target)
                output_data.update_bits_and_time(kp_coding_info)

                kp_target_hat = kp_coding_info['kp_hat']
                anim_params = {'reference_frame': reference_frame, 'ref_fts': ref_fts,
                               'kp_reference': kp_reference, 'kp_target': kp_target_hat}
                animated_frame = models.animation_model.generate_animation(anim_params)
                residual_frame = target_frame - animated_frame
                eval_params = {
                    'rate_idx': input_data.eval_params['rd_point'],
                    'q_value': input_data.eval_params['q_value'],
                    'use_skip': input_data.eval_params['use_skip'],
                    'skip_thresh': input_data.eval_params['skip_thresh']
                }

                if idx == 1:
                    res_coding_info, skip = models.animation_model.compress_spatial_residual(
                        residual_frame, prev_latent, **eval_params)
                    prev_res_hat = res_coding_info['res_hat']
                else:
                    temporal_residual_frame = residual_frame - prev_res_hat
                    res_coding_info, skip = models.animation_model.compress_temporal_residual(
                        temporal_residual_frame, prev_latent, **eval_params)
                    if not skip:
                        prev_res_hat = res_coding_info['res_hat'] + prev_res_hat

                if not skip:
                    prev_latent = res_coding_info['prev_latent']
                    output_data.update_bits_and_time(res_coding_info)

                enh_prediction = (animated_frame + prev_res_hat).clamp(0, 1)
                output_data.animated_video.append(tensor2frame(animated_frame))
                output_data.decoded_video.append(tensor2frame(enh_prediction))

                viz_params = {'reference_frame': reference_frame, 'target_frame': target_frame,
                              'res': residual_frame, 'res_hat': prev_res_hat,
                              'prediction': animated_frame, 'enhanced_prediction': enh_prediction,
                              **anim_params}
                output_data.visualization.append(visualizer.visualize(**viz_params))
    return output_data


def test(config, dataset, animation_model_arch, kp_detector_arch, **kwargs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']
    visualizer = Visualizer(**config['visualizer_params'])
    target_h, target_w = config['dataset_params']['frame_shape'][:2]

    if model_id == 'baselines':
        monitor = Metrics(config['eval_params']['metrics'], config['eval_params'].get('temporal', False))
        all_metrics = {}
        for x in dataset:
            codec = ConventionalCodec(config['eval_params'])
            video = x['video']
            codec.num_frames = min(num_frames, video.shape[0])
            codec.video = video
            name = x['name']
            os.makedirs(os.path.join(kwargs['log_dir'], name.split('.')[0]), exist_ok=True)
            codec.create_gops()
            codec.run()
            metrics = monitor.compute_metrics(codec.video[:codec.num_frames], codec.decoded_video)
            metrics['bitrate'] = codec.get_bitrate()
            all_metrics[name] = metrics
        with open(f"{kwargs['log_dir']}/metrics_{codec.qp}.json", 'w') as f:
            json.dump(all_metrics, f, indent=4)
        return

    # Non-baseline models
    per_frame = config['eval_params'].get('per_frame_metrics', config['eval_params'].get('temporal', False))
    monitor = Metrics(config['eval_params']['metrics'], per_frame)
    pretrained_cpk_path = kwargs.get('checkpoint') or config['dataset_params'].get('cpk_path')
    rd_point = config['eval_params']['rd_point']

    animation_model = animation_model_arch
    kp_detector_model = kp_detector_arch
    if pretrained_cpk_path is not None:
        animation_model = load_pretrained_model(animation_model, path=pretrained_cpk_path, device=device)
        kp_detector_model = load_pretrained_model(kp_detector_model, path=pretrained_cpk_path,
                                                  name='kp_detector', device=device)

    animation_model.eval()
    if 'rdac' in model_id:
        animation_model.sdc.update(force=True)
        animation_model.tdc.update(force=True)

    kp_detector_model.eval()
    if torch.cuda.is_available():
        animation_model, kp_detector_model = animation_model.cuda(), kp_detector_model.cuda()

    models = Models(animation_model, kp_detector_model,
                    ImageCoder(config['eval_params']['qp'], config['eval_params']['ref_codec']),
                    KpEntropyCoder())

    all_metrics = {}
    with torch.no_grad():
        for x in dataset:
            video = resize_video_if_needed(x['video'], target_h, target_w)
            n_frames = min(num_frames, video.shape[0])

            input_data = Inputs(config['eval_params'])
            input_data.device = device
            input_data.num_frames = n_frames
            input_data.video = video
            input_data.create_gops()

            name = x['name']
            out_path = os.path.join(kwargs['log_dir'], name.split('.')[0])
            os.makedirs(out_path, exist_ok=True)

            output_data = predictive_coder(models, input_data, visualizer, out_path=out_path)

            imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4", output_data.decoded_video, fps=10, codec='h264')
            imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4", output_data.visualization, fps=10, codec='h264')

            if len(output_data.animated_video) == len(output_data.decoded_video):
                comp_vid = np.concatenate((np.array(input_data.original_video),
                                           np.array(output_data.animated_video),
                                           np.array(output_data.decoded_video)), axis=2)
            else:
                comp_vid = np.concatenate((np.array(input_data.original_video),
                                           np.array(output_data.decoded_video)), axis=2)
            imageio.mimsave(f"{out_path}/{rd_point}_anim_enh.mp4", comp_vid, fps=10, codec='h264')

            metrics = monitor.compute_metrics(input_data.original_video, output_data.decoded_video)
            metrics['bitrate'] = output_data.get_bitrate(input_data.fps, input_data.num_frames)
            all_metrics[name] = metrics
            print(metrics)

    with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
        json.dump(all_metrics, f, indent=4)


test_functions = {'rdac': test, 'crdac': test}
