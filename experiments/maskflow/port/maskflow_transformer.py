"""Isolated 2511 compatibility fixes against Diffusers 0.37.1.

The pinned MLX-Gen checkout is unmodified. Its base forward is reproduced
here only because it calls its own class's static RoPE helper explicitly.
Diffusers computes text RoPE for physical padded width, and its output AdaLN
has a learned linear bias. Both differences are checked by tiny CPU parity.
"""
from __future__ import annotations

import mlx.core as mx

from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer


class MaskFlowTransformer(QwenTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, bias=True)

    @staticmethod
    def _compute_rotary_embeddings(encoder_hidden_states_mask, pos_embed, img_shapes):
        width = int(encoder_hidden_states_mask.shape[1])
        return pos_embed(video_fhw=img_shapes, txt_seq_lens=[width] * int(encoder_hidden_states_mask.shape[0]))

    def __call__(self, t, config, hidden_states, encoder_hidden_states,
                 encoder_hidden_states_mask, qwen_image_ids=None,
                 cond_image_grid=None, controlnet_block_samples=None):
        if controlnet_block_samples is not None:
            raise ValueError("MaskFlow has no ControlNet branch")
        if qwen_image_ids is not None:
            raise ValueError("Full-canvas MaskFlow uses rectangular official RoPE grids")
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = self._compute_timestep(t, config)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        img_shapes = self._compute_image_shapes(config=config, cond_image_grid=cond_image_grid)
        modulate_index = None
        if self.zero_cond_t:
            timestep = mx.concatenate([timestep, mx.zeros_like(timestep)], axis=0)
            modulate_index = self._compute_modulate_index(
                batch_size=batch_size, img_shapes=img_shapes,
                sequence_length=hidden_states.shape[1])
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        rotary = self._compute_rotary_embeddings(encoder_hidden_states_mask, self.pos_embed, img_shapes)
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = self._apply_transformer_block(
                idx=idx, block=block, hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings, image_rotary_embeddings=rotary,
                modulate_index=modulate_index)
        if self.zero_cond_t:
            text_embeddings = mx.split(text_embeddings, 2, axis=0)[0]
        return self.proj_out(self.norm_out(hidden_states, text_embeddings))
