"""Single-use Qwen conditioning encoder: materialize and release each layer.

Weights retain the same native checkpoint validation and arithmetic. This is
inference component offloading, not pruning, distillation or fewer layers.
The encoder is consumed after one prompt; it cannot be reused for generation.
"""
import mlx.core as mx
import numpy as np
import json
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import (
    LanguageModel, QwenImage21TextEncoder,
)
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel


def load_layer(owner, layer, prefix):
    directory = getattr(owner, "_encoder_directory", None)
    if directory is not None:
        from low_memory_loader import load_component
        return load_component(layer, directory, prefix=prefix)
    return layer


class StreamingVisionModel(Qwen3VLVisionModel):
    def __call__(self, hidden_states, grid_thw, return_deepstack=False):
        if self.patch_embed is None:
            raise RuntimeError("Streaming vision encoder has already been consumed")
        self.patch_embed = load_layer(self, self.patch_embed, "visual.patch_embed")
        self.pos_embed = load_layer(self, self.pos_embed, "visual.pos_embed")
        hidden_states = self.patch_embed(hidden_states)
        positions = self._fast_pos_embed_interpolate(self.spatial_merge_size, self.pos_embed, self.num_grid_per_side, grid_thw)
        if self.preserve_input_dtype:
            positions = positions.astype(hidden_states.dtype)
        hidden_states = hidden_states + positions
        mx.eval(hidden_states)
        print(json.dumps({"encoder_phase": "vision_patch", "tokens": hidden_states.shape[0], "mlx_active_bytes": mx.get_active_memory()}), flush=True)
        self.patch_embed = self.pos_embed = None
        del positions
        rotary = self._rot_pos_emb(self.rotary_pos_emb, self.spatial_merge_size, grid_thw)
        length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(length, -1)
        rotary = rotary.reshape(length, -1)
        emb = mx.concatenate([rotary, rotary], axis=-1)
        position_embeddings = (mx.cos(emb), mx.sin(emb))
        lengths = []
        for index in range(grid_thw.shape[0]):
            t, h, w = [int(grid_thw[index, dim].item()) for dim in range(3)]
            lengths.extend([h * w * t] * t)
        cu_seqlens = mx.array([0] + [sum(lengths[:index + 1]) for index in range(len(lengths))], dtype=mx.int32)
        mx.eval(position_embeddings, cu_seqlens)
        deepstack = [] if return_deepstack else None
        for index in range(len(self.blocks)):
            block = load_layer(self, self.blocks[index], f"visual.blocks.{index}")
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            if return_deepstack and index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(index)
                self.deepstack_merger_list[merger_index] = load_layer(self, self.deepstack_merger_list[merger_index], f"visual.deepstack_merger_list.{merger_index}")
                features = self.deepstack_merger_list[merger_index](hidden_states)
                mx.eval(hidden_states, features)
                deepstack.append(features)
                self.deepstack_merger_list[merger_index] = None
            else:
                mx.eval(hidden_states)
            self.blocks[index] = None
            del block
            mx.clear_cache()
            if index % 8 == 0:
                print(json.dumps({"encoder_phase": "vision_block", "index": index, "mlx_active_bytes": mx.get_active_memory()}), flush=True)
        self.merger = load_layer(self, self.merger, "visual.merger")
        result = self.merger(hidden_states)
        mx.eval(result)
        self.merger = None
        mx.clear_cache()
        return result, deepstack


class StreamingLanguageModel(LanguageModel):
    def __call__(self, hidden, positions, image_indices=None, deepstack=None):
        rope = self.rotary_emb(hidden, positions)
        idx = mx.arange(hidden.shape[1])
        mask = (idx[:, None] >= idx[None, :])[None, None]
        mx.eval(rope, mask)
        for index in range(len(self.layers)):
            layer = self.layers[index]
            if layer is None:
                raise RuntimeError("Streaming encoder has already been consumed")
            layer = load_layer(self, layer, f"language_model.layers.{index}")
            hidden, cache = layer(hidden, attention_mask=mask, position_embeddings=rope)
            if deepstack is not None and index < len(deepstack):
                hidden[:, image_indices] += deepstack[index].astype(hidden.dtype)[None]
            mx.eval(hidden)
            self.layers[index] = None
            del layer, cache
            mx.clear_cache()
            if index % 8 == 0:
                print(json.dumps({"encoder_phase": "language_block", "index": index, "mlx_active_bytes": mx.get_active_memory()}), flush=True)
        return hidden


class StreamingTextEncoder(QwenImage21TextEncoder):
    def __call__(self, input_ids, pixel_values=None, image_grid_thw=None):
        if input_ids.shape[0] != 1:
            raise ValueError("One prompt required")
        if self.language_model.embed_tokens is None:
            raise RuntimeError("Streaming encoder has already been consumed")
        self.language_model.embed_tokens = load_layer(self, self.language_model.embed_tokens, "language_model.embed_tokens")
        hidden = self.language_model.embed_tokens(input_ids)
        mx.eval(hidden)
        print(json.dumps({"encoder_phase": "embedding", "tokens": input_ids.shape[-1], "mlx_active_bytes": mx.get_active_memory()}), flush=True)
        self.language_model.embed_tokens = None
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            mx.eval(features, *deepstack)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count mismatch")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
            mx.eval(hidden)
            del features
        self.visual = None
        mx.clear_cache()
        positions = self.position_ids(input_ids, image_grid_thw)
        return self.language_model(hidden, positions, image_indices, deepstack)


def consume_encoder(module, directory=None):
    module.__class__ = StreamingTextEncoder
    module.language_model.__class__ = StreamingLanguageModel
    module.visual.__class__ = StreamingVisionModel
    if directory is not None:
        module._encoder_directory = directory
        module.language_model._encoder_directory = directory
        module.visual._encoder_directory = directory
    return module
