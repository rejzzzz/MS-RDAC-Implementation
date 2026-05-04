import torch
import torch.nn as nn
import torch.nn.functional as F
from rdac.train_utils import ImagePyramide, detach_kp


class DACDiscriminatorFullModel(nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """
    def __init__(self, discriminator=None, train_params=None, **kwargs):
        super(DACDiscriminatorFullModel, self).__init__()
        self.discriminator = discriminator
        self.train_params = train_params
        self.disc_type = kwargs['disc_type']
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.loss_weights = train_params['loss_weights']

    def compute_multiscale(self, real, decoded, kp_target):
        pyramide_real = self.pyramid(real)
        pyramide_generated = self.pyramid(decoded)
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_target))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_target))

        loss = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            loss += self.loss_weights['discriminator_gan'] * value.mean()
        return loss

    def _non_saturating_loss(self, D_real_logits, D_gen_logits):
        D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
            target=torch.ones_like(D_real_logits))
        D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
            target=torch.zeros_like(D_gen_logits))
        D_loss = D_loss_real + D_loss_gen
        return D_loss

    def compute_patch_disc(self, real, decoded, context, model='disc'):
        disc_out = self.discriminator(torch.cat([real, decoded], dim=1), context)
        loss = self._non_saturating_loss(disc_out.d_real_logits, disc_out.d_gen_logits)
        return loss

    def forward(self, x, generated, model='discriminator'):
        loss = 0.0
        num_targets = len([tgt for tgt in x.keys() if 'target' in tgt])
        for idx in range(num_targets):
            real = x[f'target_{idx}']
            prediction = generated[f'prediction_{idx}'].detach()

            if self.disc_type == 'multi_scale':
                loss += self.compute_multiscale(real, prediction, generated[f'kp_target_{idx}'])
            else:
                loss += self.compute_patch_disc(real, prediction, generated[f'context_{idx}'], 'disc')
        loss = (loss / num_targets) * self.train_params['loss_weights']['discriminator_gan']
        return {'disc_gan': loss}
