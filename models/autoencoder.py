from .utils import DiagonalGaussianDistribution
from .decoder import Decoder
from .encoder import Encoder

import torch
import torch.nn as nn
from typing import TypedDict, List
from PIL import Image
from taming.modules.losses.vqperceptual import LPIPS, NLayerDiscriminator, hinge_d_loss, vanilla_d_loss, weights_init, adopt_weight
import lightning as L

# from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

class BatchDict(TypedDict):
    images: List[Image.Image]
    texts : List[str]

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=(1, )) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, opt_target,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss

        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if opt_target == 'GEN_PART':
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if opt_target == 'DISC_PART':
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
        
class AutoEncoder(L.LightningModule):
    def __init__(self, cfg, colorize_nlabels=None, monitor=None, ignore_keys:List[str]=[]):
        super().__init__()
        # DotDict form
        self.cfg = cfg
        self.automatic_optimization = False

        self.encoder = Encoder(**cfg.encoder)
        self.decoder = Decoder(**cfg.decoder)

        self.quant_conv = torch.nn.Conv2d(2*cfg.encoder.z_channels, 2 * cfg.common.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(cfg.common.embed_dim, cfg.decoder.z_channels, 1)
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        if cfg.common.ckpt_path is not None:
            self.init_from_ckpt(cfg.common.ckpt_path, ignore_keys=ignore_keys)
        
        self.loss = LPIPSWithDiscriminator(disc_start=cfg.loss_fn.disc_start, 
                                           kl_weight=cfg.loss_fn.kl_weight, 
                                           disc_weight=cfg.loss_fn.disc_weight)
        
    def init_from_ckpt(self, path:str, ignore_keys:List[str]):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x: torch.Tensor):
        # given x, this encodes the x into posterior mean and log variance 
        h = self.encoder(x)
        moments = self.quant_conv(h) # Mean and Log var
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor):
        # z : latent variable
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    def forward(self, input, sample_posterior:bool = True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def training_step(self, batch, batch_idx):
        #  -> torch.Tensor | Mapping[str, Any] | None
        # ldm.modules.losses.LPIPSWithDiscriminator
        # train encoder, decoder
        opt_ae, opt_disc = self.optimizers()
        self.toggle_optimizer(opt_ae)
        
        reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(batch, reconstructions, posterior, 'GEN_PART', self.global_step, 
                                        last_layer=self.get_last_layer(), split='train')
        self.manual_backward(aeloss)
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        
        self.toggle_optimizer(opt_disc)

        discloss, log_dict_disc = self.loss(batch, reconstructions, posterior, 'DISC_PART', self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        
        
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.cfg.common.lr
        opt1 = torch.optim.Adam(list(self.encoder.parameters())+
                               list(self.decoder.parameters())+
                               list(self.quant_conv.parameters())+
                               list(self.post_quant_conv.parameters()) + 
                               list(self.loss.discriminator.parameters()),
                               lr=lr, betas=(0.5, 0.9))
        
        opt2 = torch.optim.Adam(list(self.loss.discriminator.parameters()),
                               lr=lr, betas=(0.5, 0.9))
        return [opt1, opt2], []
    
    
if __name__ == "__main__":
    # Following disc compresses an image into 1/64 image shape and get scores.
    pass