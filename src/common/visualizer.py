import numpy as np
import torch.nn.functional as F
from skimage.draw import disk
import matplotlib.pyplot as plt

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
