import os, torch, imageio, collections, numpy as np, torch.nn.functional as F
from skimage.draw import disk
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def draw_hm(hm, cm, bg=(0,0,0)):
    bg = np.array(bg).reshape((1, 1, 1, 3))
    parts, wts = [], []
    for i in range(hm.shape[-1]):
        c = np.array(cm(i / hm.shape[-1]))[:3].reshape((1, 1, 1, 3))
        p = hm[:, :, :, i:(i + 1)] / np.max(hm[:, :, :, i:(i + 1)], axis=(1, 2), keepdims=True)
        wts.append(p)
        parts.append(p * c)
    w = np.maximum(1, sum(wts))
    return sum(parts) / w + (1 - np.minimum(1, w)) * bg

class Viz:
    def __init__(self, kp_sz=5, frame_dim=(256,256,3), draw_border=False, cm='gist_rainbow', bg=(0, 0, 0)):
        self.kp_sz, self.draw_border, self.cm, self.bg = kp_sz, draw_border, plt.get_cmap(cm), bg
        self.h, self.w, self.c = frame_dim if isinstance(frame_dim, tuple) else (256, 256, 3)

    def draw_kp(self, img, kp):
        img = np.copy(img)
        sz = np.array(img.shape[:2][::-1])[np.newaxis]
        kp = sz * (kp + 1) / 2
        for i, k in enumerate(kp):
            rr, cc = disk((k[1], k[0]), self.kp_sz, shape=img.shape[:2])
            img[rr, cc] = np.array(self.cm(i / len(kp)))[:3]
        return img

    def col_kp(self, imgs, kp):
        return self.col(np.array([self.draw_kp(v, k) for v, k in zip(imgs, kp)]))

    def col(self, imgs):
        if self.draw_border:
            imgs = np.copy(imgs)
            imgs[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(imgs), axis=0)

    def grid(self, *args):
        return np.concatenate([self.col_kp(arg[0], arg[1]) if isinstance(arg, tuple) else self.col(arg) for arg in args], axis=1)

    def detach(self, frame, sz=None):
        sz = sz or [self.h, self.w]
        return np.transpose(F.interpolate(frame.data.cpu(), size=sz).numpy(), [0, 2, 3, 1])

    def viz(self, **out):
        imgs = []
        for k in ['kp_target_0', 'kp_target']:
            out.pop(k, None)
        
        for i in range(4):
            if f'reference_{i}' in out:
                ref = np.transpose(out[f'reference_{i}'].data.cpu(), [0, 2, 3, 1])
                imgs.append((ref, out[f'kp_reference_{i}']['value'].data.cpu().numpy()))
                H, W = ref.shape[1:3]
        
        if 'reference' in out:
            ref = np.transpose(out['reference'].data.cpu(), [0, 2, 3, 1])
            if 'kp_src' in out:
                imgs.append((ref, -out['kp_src'].data.cpu().numpy()))
            if 'kp_reference' in out:
                imgs.append((ref, out['kp_reference']['value'].data.cpu().numpy()))
            else:
                imgs.append(ref)
            H, W = ref.shape[1:3]
        
        for i, k in enumerate([x for x in sorted(out.keys()) if 'target' in x and '_target' not in x]):
            for hf_k in [f'hf_details_{i}']:
                if hf_k in out:
                    for hf in out[hf_k]:
                        hf_m = (torch.tanh(F.interpolate(torch.mean(hf, dim=1, keepdim=True), size=[256,256]))+1.0)/2.0
                        imgs.append(self.detach(hf_m.repeat(1,3,1,1), [H,W]))
            
            for lk in [f'base_layer_{i}', k, f'occlusion_map_{i}', f'prediction_{i}']:
                if lk in out:
                    if lk == f'occlusion_map_{i}':
                        occ = F.interpolate(out[lk].data.cpu().repeat(1, 3, 1, 1), size=(H,W)).numpy()
                        imgs.append(np.transpose(occ, [0, 2, 3, 1]))
                    else:
                        imgs.append(self.detach(out[lk], [H,W]))
        
        img = self.grid(*imgs)
        return (255 * img).astype(np.uint8)

class Log:
    def __init__(self, log_dir: str, ckpt_freq: int=100, viz_params: Dict[str, Any]=None, zfill: int=8, log_file: str='log.txt', mode: str='test'):
        self.loss_list, self.cpk_dir = [], log_dir
        self.viz_dir = os.path.join(log_dir, 'train-vis')
        self.log_f = open(os.path.join(log_dir, log_file), 'a')
        self.zfill, self.ckpt_freq, self.epoch = zfill, ckpt_freq, 0
        self.viz = Viz(**(viz_params or {}))
        self.best_loss, self.names, self.mode, self.epoch_losses = float('inf'), None, mode, []

    def log_scores(self, loss_names: List[str]):
        loss_mean = np.array(self.loss_list).mean(axis=0)
        loss_str = "; ".join([f"{n} - {v:.5f}" for n, v in zip(loss_names, loss_mean)])
        try:
            self.epoch_losses = {y[0].replace(' ',''): float(y[1]) for y in [x.split('-') for x in loss_str.split(';')]}
            print(f"{str(self.epoch).zfill(self.zfill)}) {loss_str}", file=self.log_f)
            self.loss_list = []
            self.log_f.flush()
        except ValueError:
            print(f"Logging error: {loss_str}")

    def viz_rec(self, inp: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], name: str=None):
        vd = f"{self.viz_dir}_{name}" if name else self.viz_dir
        os.makedirs(vd, exist_ok=True)
        img = self.viz.viz(**{**inp, **out})
        imageio.imsave(os.path.join(vd, f"{str(self.epoch).zfill(self.zfill)}-rec.png"), img)
        return img
    
    def save_ckpt(self, emergent=False):
        ckpt = {k: v.state_dict() for k, v in self.models.items()} | {'epoch': self.epoch}
        cp = os.path.join(self.cpk_dir, f"{str(self.epoch).zfill(self.zfill)}-new-checkpoint.pth.tar")
        if not (os.path.exists(cp) and emergent):
            torch.save(ckpt, cp)
        prev = os.path.join(self.cpk_dir, f"{str(self.epoch-self.ckpt_freq).zfill(self.zfill)}-new-checkpoint.pth.tar")
        os.path.isfile(prev) and os.remove(prev)
            
    @staticmethod
    def load_ckpt(ckpt_path, gen=None, kpd=None, opt=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=device)
        gen and gen.load_state_dict(ckpt.get('generator', {}), strict=False)
        kpd and kpd.load_state_dict(ckpt.get('kp_detector', {}), strict=False)
        opt and opt.load_state_dict(ckpt.get('optimizer', {}), strict=False)
        return ckpt.get('epoch', 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        'models' in self.__dict__ and self.save_ckpt()
        self.log_f.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        self.names = self.names or list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models=None, inp=None, out=None, name=None):
        self.epoch = epoch
        if models:
            self.models = models
            if (self.epoch + 1) % self.ckpt_freq == 0:
                self.save_ckpt()
        self.log_scores(self.names)
        img = self.viz_rec(inp, out, name=name)
        return img, self.epoch_losses
