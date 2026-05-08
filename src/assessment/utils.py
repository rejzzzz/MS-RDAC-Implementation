import torch
import numpy as np
from typing import Dict, Tuple, List, Union, Sequence, Any

data_range: List[int] = [0, 1]
_COLOR_MATRICES = {
    '601': (0.299, 0.587, 0.114, 1.772, 1.402),
    '709': (0.2126, 0.7152, 0.0722, 1.8556, 1.5748),
    '2020': (0.2627, 0.6780, 0.0593, 1.8814, 1.4747)
}

def load_image_array(rgb_data: np.ndarray, color_conv: str = '709', def_bits: int = 8, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    rgb_t = convert_and_round_plane(torch.tensor(rgb_data, dtype=torch.float, device=device).permute(2, 0, 1), [0, 255], data_range, def_bits).unsqueeze(0)
    yuv_t = round_plane(rgb_to_yuv(rgb_t, color_conv).clamp(min(data_range), max(data_range)), def_bits)
    return {'Y': yuv_t[0, 0], 'U': yuv_t[0, 1], 'V': yuv_t[0, 2]}

def load_image(filename: str, color_conv: str = '709', def_bits: int = 8, device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    from PIL import Image
    im = Image.open(filename)
    mode, rgb_data = im.mode, np.array(im.convert('RGB'))
    im.close()
    if def_bits == -1:
        def_bits = int(mode.split(';')[1]) if ';' in mode else 8
    rgb_t = convert_and_round_plane(torch.tensor(rgb_data, dtype=torch.float, device=device).permute(2, 0, 1), [0, 255], data_range, def_bits).unsqueeze(0)
    yuv_t = round_plane(rgb_to_yuv(rgb_t, color_conv).clamp(min(data_range), max(data_range)), def_bits)
    return {'Y': yuv_t[0, 0], 'U': yuv_t[0, 1], 'V': yuv_t[0, 2]}, rgb_t


def round_plane(plane: torch.Tensor, bits: int) -> torch.Tensor:
    return plane.mul((1 << bits) - 1).round().div((1 << bits) - 1)

def convertup_and_round_plane(plane: torch.Tensor, cur_range: Sequence[Union[int, float]], new_range: Sequence[Union[int, float]], bits: int) -> torch.Tensor:
    return convert_range(plane, cur_range, new_range).mul((1 << bits) - 1).round()

def convert_and_round_plane(plane: torch.Tensor, cur_range: Sequence[Union[int, float]], new_range: Sequence[Union[int, float]], bits: int) -> torch.Tensor:
    return round_plane(convert_range(plane, cur_range, new_range), bits)

def convert_range(plane: torch.Tensor, cur_range: Sequence[Union[int, float]], new_range: Sequence[Union[int, float]] = [0, 1]) -> torch.Tensor:
    c_min, c_max, n_min, n_max = float(cur_range[0]), float(cur_range[1]), float(new_range[0]), float(new_range[1])
    if c_min == n_min and c_max == n_max: return plane
    return (plane + c_min) * (n_max - n_min) / (c_max - c_min) - n_min

def convert_yuvdict_to_tensor(yuv: Dict[str, torch.Tensor], device: str = 'cpu') -> torch.Tensor:
    size = yuv['Y'].shape
    ans = torch.zeros((1, 3, size[-2], size[-1]), dtype=torch.float, device=torch.device(device))
    ans[0, 0] = yuv['Y']
    ans[0, 1] = yuv.get('U', yuv['Y'])
    ans[0, 2] = yuv.get('V', yuv['Y'])
    return ans

def color_conv_matrix(color_conv: str = '709') -> Tuple[float, float, float, float, float]:
    if color_conv not in _COLOR_MATRICES: raise NotImplementedError(f"Color conversion {color_conv} not supported")
    return _COLOR_MATRICES[color_conv]


def rgb_to_yuv(image: torch.Tensor, color_conv: str = '709') -> torch.Tensor:
    """Convert RGB image to YUV. Image shape: (*, 3, H, W), range: (0, 1)."""
    if not isinstance(image, torch.Tensor) or len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Expected torch.Tensor with shape (*, 3, H, W), got {type(image)} {getattr(image, "shape", None)}')
    r, g, b = image[..., 0, :, :], image[..., 1, :, :], image[..., 2, :, :]
    a1, b1, c1, d1, e1 = color_conv_matrix(color_conv)
    y = a1 * r + b1 * g + c1 * b
    return torch.stack([y, (b - y) / d1 + 0.5, (r - y) / e1 + 0.5], -3)

def yuv_to_rgb(image: torch.Tensor, color_conv: str = '709') -> torch.Tensor:
    """Convert YUV image to RGB. Image shape: (*, 3, H, W), range: (0, 1)."""
    if not isinstance(image, torch.Tensor) or len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f'Expected torch.Tensor with shape (*, 3, H, W), got {type(image)} {getattr(image, "shape", None)}')
    y, u, v = image[..., 0, :, :], image[..., 1, :, :] - 0.5, image[..., 2, :, :] - 0.5
    a, b, c, d, e = color_conv_matrix(color_conv)
    return torch.stack([y + e * v, y - (c * d / b) * u - (a * e / b) * v, y + d * u], -3)

def write_yuv(yuv: Dict[str, Union[torch.Tensor, np.ndarray]], f: Union[str, Any], bits: int = 8) -> None:
    """Write YUV dict to file."""
    data_types = {1: np.uint8, 2: np.uint16, 4: np.uint32}
    nr_bytes = int(np.ceil(bits / 8))
    if nr_bytes not in data_types: raise NotImplementedError(f'Bitdepth {bits} not supported')
    
    yuv_converted = {}
    for plane in yuv:
        tensor_val = yuv[plane] if isinstance(yuv[plane], torch.Tensor) else torch.from_numpy(yuv[plane])
        yuv_converted[plane] = convertup_and_round_plane(tensor_val, data_range, data_range, bits).cpu().numpy()
    
    lst = []
    for plane in ['Y', 'U', 'V']:
        if plane in yuv_converted: lst.extend(yuv_converted[plane].ravel().tolist())
    
    np.array(lst).astype(data_types[nr_bytes]).tofile(f)

