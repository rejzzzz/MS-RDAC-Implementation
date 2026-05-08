import torch, numpy as np
from typing import Union, Protocol

class Generator(Protocol):
    def forward(self): ...

class KPD(Protocol):
    def forward(self): ...

def load_model(model: Union[Generator, KPD], path: str, name: str='generator', device: str='cpu'):
    model.load_state_dict(torch.load(path, map_location=device)[name], strict=True)
    return model

def f2t(frame, cuda=True):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
    else:
        frame = frame.permute(0,3,1,2)
    return frame / 255.0

def t2f(tensor):
    return (tensor.detach().cpu().squeeze().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)

def coord_grid(spatial_size, type):
    h, w = spatial_size
    x = (2 * (torch.arange(w).type(type) / (w - 1)) - 1)
    y = (2 * (torch.arange(h).type(type) / (h - 1)) - 1)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([xx, yy], dim=2)

def kp2gauss(mean, spatial_size, kp_var=0.01):
    grid = coord_grid(spatial_size, mean.type())
    shape = (1,) * (len(mean.shape) - 1) + grid.shape
    grid = grid.view(*shape).repeat(*mean.shape[:len(mean.shape)-1], 1, 1, 1)
    mean = mean.view(*mean.shape[:len(mean.shape)-1], 1, 1, 2)
    return torch.exp(-0.5 * ((grid - mean) ** 2).sum(-1) / kp_var)

def get_rd_pt(path):
    pth = path.split('/')[-1].split('.')[0]
    return int(pth.split('_')[-1]) if 'rd' in pth else 1