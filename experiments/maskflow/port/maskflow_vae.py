"""2511 RGB codec with official model-dtype affine normalization boundaries."""
from __future__ import annotations

from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from maskflow_math import normalize_vae_latents, denormalize_vae_latents


class MaskFlowVAE(QwenVAE):
    def encode(self, latents):
        if latents.ndim == 4:
            latents = latents[:, :, None]
        moments = self.quant_conv(self.encoder(latents))
        # Official eval uses the posterior mode, not a sampled posterior.
        return normalize_vae_latents(moments[:, :16], backend="mlx")

    def decode(self, latents):
        if latents.ndim == 4:
            latents = latents[:, :, None]
        return self.decoder(self.post_quant_conv(denormalize_vae_latents(latents, backend="mlx")))
