import os, json, torch, imageio, numpy as np
from typing import List, Dict, Any
from skimage import img_as_ubyte, img_as_float32
from pathlib import Path

def get_gops(video, gop_size):
    n = video.shape[0]
    return [video] if gop_size > n and n > 2 else [video[i:min(i+gop_size, n)] for i in range(0, n, gop_size)]

def rgb2t(x, device='cpu'):
    x = torch.tensor(img_as_float32(x).transpose(2,0,1), dtype=torch.float32).unsqueeze(0)
    return x.cuda() if torch.cuda.is_available() and device=='cuda' else x

def t2rgb(x):
    return img_as_ubyte(np.transpose(torch.squeeze(x, dim=0).data.cpu().numpy(), [1, 2, 0]))

def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def fsize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return os.path.getsize(filepath) * 8

def read_bits(filepath: str):
    with open(filepath, 'rb') as f:
        return f.read()

def bitrate(bits: int, fps: float, frames: int) -> float:
    return round((bits * fps) / (1000 * frames), 2)

def save_vids(path: str, videos: Dict[str, List[np.ndarray]], meta: Dict[str, Any]) -> None:
    out = os.path.join(path, meta['c_name'], meta['name'])
    os.makedirs(out, exist_ok=True)
    for k in ['decoded', 'visualization', 'mask']:
        if k in videos:
            imageio.mimsave(f"{out}/{meta['l_name']}_{k}.mp4", videos[k], fps=meta['fps'])

def save_mets(path: str, mets: Dict[str, List[float]], meta: Dict[str, Any]) -> None:
    out = os.path.join(path, meta['c_name'])
    os.makedirs(out, exist_ok=True)
    fp = f"{out}/{meta['c_name']}_metrics.json"
    
    m = {'fps': meta['fps'], 'bitrate': mets['bitrate'], **mets['metrics']}
    all_m = json.load(open(fp)) if os.path.exists(fp) else {}
    
    if meta['name'] not in all_m:
        all_m[meta['name']] = {}
    all_m[meta['name']][meta['l_name']] = m
    
    with open(fp, 'w') as f:
        json.dump(all_m, f)
