import os
import numpy as np
import imageio.v3 as iio
from skimage import io, img_as_float32
from skimage.transform import resize
from torch.utils.data import Dataset
from .augmentation import AllAugmentationTransform
from functools import partial


class FramesDataset(Dataset):
    """Dataset of videos (.mp4/.gif or frame folders)."""

    def __init__(self, train_dir, test_dir, frame_shape=(256, 256, 3), is_train=True,
                 augmentation_params=None, num_sources=2, target_delta=2, **kwargs):
        self.is_train = is_train
        self.frame_shape = tuple(frame_shape)
        self.root_dir = train_dir if is_train else test_dir
        self.videos = os.listdir(self.root_dir)
        self.num_sources = num_sources
        self.tgt_delta = target_delta
        self.transform = AllAugmentationTransform(**augmentation_params) if is_train else None

    def __len__(self):
        return len(self.videos)

    def _get_resize_fn(self):
        if self.frame_shape is not None:
            return partial(resize, output_shape=self.frame_shape)
        return img_as_float32

    def _count_frames(self, path):
        if os.path.isdir(path):
            return os.listdir(path), len(os.listdir(path))
        return None, iio.improps(path, plugin='pyav').shape[0]

    def _read_frames(self, path, frame_idx, frames_list=None):
        resize_fn = self._get_resize_fn()
        if frames_list is not None:  # directory of frames
            if isinstance(frames_list[0], bytes):
                return [resize_fn(io.imread(os.path.join(path, frames_list[i].decode('utf-8')))) for i in frame_idx]
            return [resize_fn(io.imread(os.path.join(path, frames_list[i]))) for i in frame_idx]
        return [resize_fn(iio.imread(path, plugin='pyav', index=i)) for i in frame_idx]

    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)

        if not self.is_train:
            video = np.array(iio.imread(path, plugin="pyav"))
            return {'video': video, 'name': name.split('.')[0]}

        frames_list, n_frames = self._count_frames(path)
        src_idx = np.random.choice(n_frames // 2)
        drv_idx = np.random.choice(range(n_frames // 2, n_frames - (self.num_sources * self.tgt_delta)))
        frame_idx = [src_idx, drv_idx]

        for _ in range(self.num_sources - 2):
            drv_idx += self.tgt_delta
            frame_idx.append(drv_idx)

        video_array = self._read_frames(path, frame_idx, frames_list)
        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        for i in range(self.num_sources):
            key = 'reference' if i == 0 else f'target_{i - 1}'
            out[key] = video_array[i].transpose((2, 0, 1))
        return out


class MRFramesDataset(FramesDataset):
    """Multi-reference dataset variant."""

    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)

        if not self.is_train:
            video = np.array(iio.imread(path, plugin="pyav"))
            return {'video': video, 'name': name.split('.')[0]}

        frames_list, n_frames = self._count_frames(path)
        frame_idx = np.random.choice(range(n_frames), size=self.num_sources, replace=False)

        video_array = self._read_frames(path, frame_idx, frames_list)
        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {'rf_weights': frame_idx}
        for i in range(self.num_sources):
            key = f'reference_{i}' if i < self.num_sources - 1 else 'target_0'
            out[key] = video_array[i].transpose((2, 0, 1))
        return out


class HDACFramesDataset(Dataset):
    """Hybrid animation dataset with optional base layer."""

    def __init__(self, train_dir, test_dir, frame_shape=(256, 256, 3), is_train=True,
                 base_layer=False, augmentation_params=None, num_sources=2,
                 base_layer_params=None, target_delta=2, **kwargs):
        self.is_train = is_train
        self.frame_shape = tuple(frame_shape)
        self.root_dir = train_dir if is_train else test_dir
        self.videos = os.listdir(self.root_dir)
        self.num_sources = num_sources
        self.base_layer = base_layer
        self.base_layer_params = base_layer_params
        self.tgt_delta = target_delta
        self.transform = AllAugmentationTransform(**augmentation_params) if is_train else None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)
        out = {}

        if not self.is_train:
            video = np.array(iio.imread(path, plugin="pyav"))
            return {'video': video, 'name': name.split('.')[0]}

        if self.base_layer:
            bl_params = self.base_layer_params
            if bl_params['variable_quality']:
                bl_qp = np.random.choice(list(bl_params['qp_values'].keys()))
            else:
                bl_qp = '50'
            bl_path = os.path.join(f"{bl_params['dir']}/{bl_params['bl_codec']}_bl/{bl_qp}", name)
            out.update({'lambda_value': bl_params['qp_values'][bl_qp]['lmbda'],
                        'bitrate': bl_params['qp_values'][bl_qp]['bitrate']})

        n_frames = iio.improps(path, plugin='pyav').shape[0]
        src_idx = np.random.choice(n_frames // 2)
        drv_idx = np.random.choice(range(n_frames // 2, n_frames - (self.num_sources * self.tgt_delta)))
        frame_idx = [src_idx, drv_idx]

        for _ in range(self.num_sources - 2):
            drv_idx += np.random.choice(range(self.tgt_delta))
            frame_idx.append(drv_idx)

        video_array = img_as_float32([iio.imread(path, plugin='pyav', index=i) for i in frame_idx])

        if self.base_layer:
            bl_video_array = img_as_float32([iio.imread(bl_path, plugin="pyav", index=i) for i in frame_idx])
            video_array = np.concatenate([video_array, bl_video_array], axis=0)

        if self.transform is not None:
            video_array = self.transform(video_array)

        for i in range(self.num_sources):
            frame = video_array[i]
            if i == 0:
                out['reference'] = frame.transpose((2, 0, 1))
            else:
                out[f'target_{i - 1}'] = frame.transpose((2, 0, 1))
                if self.base_layer:
                    out[f'base_layer_{i - 1}'] = video_array[i + self.num_sources].transpose((2, 0, 1))
        return out


class DatasetRepeater(Dataset):
    """Pass several times over the same dataset for better i/o performance."""

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
