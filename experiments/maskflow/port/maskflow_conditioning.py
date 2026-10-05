"""Faithful MaskFlow source/mask VLM conditioning, without loading weights.

The upstream 2511 fast image processor receives BF16 [0,1] tensors and still
has ``do_rescale=True``. This intentionally preserves its second 1/255 rescale.
Writing those tensors to PNG and using the stock MLX image processor changes
that behavior, and also quantizes the softened mask. No temporary PNG is used.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from PIL import Image
from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder

BASE_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
PROCESSOR_CONFIG_SHA256 = "e98b81513c5d03209ac71194b3cd5303b90956097ffdb9e2aa64d6d1a13ae74e"
PROCESSOR_CONFIG_URL = (
    "https://huggingface.co/Qwen/Qwen-Image-Edit-2511/resolve/"
    f"{BASE_REVISION}/processor/preprocessor_config.json"
)
EDIT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
TEMPLATE_DROP_INDEX = 64


class MaskFlowEncoder(QwenEncoder):
    """Keep the pretrained BF16 vision input boundary before Qwen VL features.

    Assigning this subclass to an existing encoder changes only this method;
    it creates no model arrays and does not modify the pinned runtime class.
    """

    def get_image_features(self, pixel_values, image_grid_thw):
        import mlx.core as mx

        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")
        pixel_values = pixel_values.astype(mx.bfloat16)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        merge_size = int(getattr(self.visual, "spatial_merge_size", 2))
        split_sizes = (image_grid_thw.prod(axis=-1) // (merge_size**2)).astype(mx.int32).tolist()
        if sum(int(size) for size in split_sizes) != image_embeds.shape[0]:
            raise ValueError("Vision feature count does not match image_grid_thw split sizes")
        image_embeds_split = []
        start_idx = 0
        for size in split_sizes:
            end_idx = start_idx + int(size)
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split


def _float_array(value: Any) -> np.ndarray:
    """Copy CPU, Torch, or materialized MLX arrays to float32 without uint8 loss."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    elif type(value).__module__.startswith("mlx"):
        import mlx.core as mx

        value = np.asarray(value.astype(mx.float32))
    return np.asarray(value, dtype=np.float32)


def _bchw(value: Any) -> np.ndarray:
    array = _float_array(value)
    if array.ndim == 2:
        array = array[None, None]
    elif array.ndim == 3:
        if array.shape[-1] in (1, 3):
            array = array.transpose(2, 0, 1)[None]
        elif array.shape[0] in (1, 3):
            array = array[None]
        else:
            raise ValueError(f"Ambiguous condition image shape {array.shape}")
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] not in (1, 3):
        raise ValueError(f"Expected one BCHW source/mask image, got {array.shape}")
    if array.shape[1] == 1:
        array = np.repeat(array, 3, axis=1)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("VLM source/mask must contain finite [0,1] values")
    return np.ascontiguousarray(array)


def prepare_vlm_conditions(
    source_canvas: Image.Image,
    processed_mask: Any,
    *,
    max_condition_resolution: int = 384 * 384,
    divisible_by: int = 32,
) -> dict[str, np.ndarray]:
    """Prepare source Lanczos and already-softened mask nearest images.

    Prefer passing ``prepare_condition_images(...)["vlm_conditions"]`` directly
    to ``encode_cfg_branches``; this fallback exists for callers with a PIL
    canvas and a full model-resolution processed mask.
    """
    import torch
    import torch.nn.functional as F

    canvas = source_canvas.convert("RGB")
    width, height = canvas.size
    ratio = width / height
    ch = math.floor(math.sqrt(max_condition_resolution / ratio) / divisible_by) * divisible_by
    cw = math.floor(math.sqrt(max_condition_resolution * ratio) / divisible_by) * divisible_by
    if min(ch, cw) <= 0:
        raise ValueError("Condition dimensions must be positive")
    # Source is uint8 PIL already, equal to upstream source.float()*255 round.
    resized = canvas if canvas.size == (cw, ch) else canvas.resize((cw, ch), Image.Resampling.LANCZOS)
    source = torch.from_numpy(np.array(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0)
    source = source.to(torch.bfloat16).float().numpy()
    mask = torch.from_numpy(_bchw(processed_mask)).to(torch.bfloat16)
    if mask.shape[-2:] != (height, width):
        raise ValueError("Processed mask must have the full source canvas dimensions")
    mask = F.interpolate(mask.float(), size=(ch, cw), mode="nearest").to(torch.bfloat16).float().numpy()
    return {"source": source, "mask": mask}


def build_official_processor(raw_tokenizer: Any):
    """Use the pinned 2511 fast processor's exact image settings on CPU."""
    from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor

    image_processor = Qwen2VLImageProcessor(
        do_convert_rgb=True,
        do_resize=True,
        resample=3,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        min_pixels=3136,
        max_pixels=12845056,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    # Transformers 5.x requires a video processor even for image-only calls.
    return Qwen2VLProcessor(
        image_processor=image_processor,
        video_processor=Qwen2VLVideoProcessor(),
        tokenizer=raw_tokenizer,
    )


def format_prompt(prompt: str, num_images: int = 2) -> str:
    prefix = "".join(
        f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>"
        for index in range(num_images)
    )
    return EDIT_TEMPLATE.format(prefix + prompt)


def tokenize_conditioning(
    prompt: str,
    vlm_conditions: dict[str, Any],
    *,
    qwen_vl_tokenizer: Any = None,
    raw_tokenizer: Any = None,
    processor: Any = None,
) -> dict[str, Any]:
    """Return actual processor inputs; no forced 1024 padding/truncation.

    Source then mask are included for positive *and* empty-negative branches.
    Inputs stay float/BF16 throughout CPU spatial and processor operations.
    """
    import torch

    if processor is None:
        if raw_tokenizer is None:
            if qwen_vl_tokenizer is None:
                raise ValueError("A local raw tokenizer or qwen_vl_tokenizer is required")
            raw_tokenizer = qwen_vl_tokenizer.processor.tokenizer
        processor = build_official_processor(raw_tokenizer)
    conditions = [_bchw(vlm_conditions[key]) for key in ("source", "mask")]
    if conditions[0].shape != conditions[1].shape:
        raise ValueError("VLM source and mask geometry must match")
    images = [torch.from_numpy(value).to(torch.bfloat16) for value in conditions]
    result = processor(
        text=[format_prompt(prompt)],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    output = {
        key: value.detach().cpu().numpy() if value.dtype != torch.bfloat16 else value.float().cpu().numpy()
        for key, value in result.items()
        if key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    }
    grid = output["image_grid_thw"]
    output["metadata"] = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "formatted_prompt_sha256": hashlib.sha256(format_prompt(prompt).encode()).hexdigest(),
        "input_shape_bchw": list(conditions[0].shape),
        "processor_grid_thw": grid.tolist(),
        "processor_pixel_shape": list(output["pixel_values"].shape),
        "pixel_values_sha256_float32": hashlib.sha256(output["pixel_values"].astype(np.float32).tobytes()).hexdigest(),
        "input_ids_sha256_int64": hashlib.sha256(output["input_ids"].astype(np.int64).tobytes()).hexdigest(),
        "input_token_count": int(output["attention_mask"].sum()),
        "embedding_token_count": int(output["attention_mask"].sum()) - TEMPLATE_DROP_INDEX,
        "merged_vision_tokens_per_image": [int(np.prod(row)) // 4 for row in grid],
        "template_drop_index": TEMPLATE_DROP_INDEX,
        "padding": "actual per-branch length; no forced 1024",
        "truncation": False,
        "pixel_processing": "upstream BF16 [0,1] tensor -> FastTorch resize/1-255 rescale/CLIP normalization",
        "softmask_png_quantization": False,
    }
    return output


def encode_cfg_branches(
    *,
    prompt: str,
    qwen_vl_tokenizer: Any,
    qwen_vl_encoder: Any,
    negative_prompt: str = "",
    vlm_conditions: dict[str, Any] | None = None,
    source_canvas: Image.Image | None = None,
    processed_mask: Any = None,
) -> dict[str, Any]:
    """Encode pm/nm independently and materialize before the VL is released."""
    import mlx.core as mx

    if vlm_conditions is None:
        if source_canvas is None or processed_mask is None:
            raise ValueError("Pass prepared vlm_conditions or source_canvas and processed_mask")
        vlm_conditions = prepare_vlm_conditions(source_canvas, processed_mask)
    processor = build_official_processor(qwen_vl_tokenizer.processor.tokenizer)
    branches = {}
    metadata = {}
    for name, text in (("pm", prompt), ("nm", negative_prompt or "")):
        tokenized = tokenize_conditioning(text, vlm_conditions, processor=processor)
        inputs = {
            "input_ids": mx.array(tokenized["input_ids"], dtype=mx.int32),
            "attention_mask": mx.array(tokenized["attention_mask"], dtype=mx.int32),
            "pixel_values": mx.array(tokenized["pixel_values"], dtype=mx.bfloat16),
            "image_grid_thw": mx.array(tokenized["image_grid_thw"], dtype=mx.int32),
        }
        embeds, mask = qwen_vl_encoder(**inputs)
        embeds = embeds.astype(mx.bfloat16)
        mask = mask.astype(mx.int32)
        mx.eval(embeds, mask)
        if embeds.shape[1] != tokenized["metadata"]["embedding_token_count"]:
            raise ValueError("VL encoder output length differs from official 64-token template trim")
        branches[name] = {"prompt_embeds": embeds, "prompt_embeds_mask": mask}
        metadata[name] = tokenized["metadata"]
        metadata[name]["embedding_shape"] = list(embeds.shape)
        metadata[name]["embedding_dtype"] = str(embeds.dtype)
        del inputs, tokenized
    branches["metadata"] = {
        "processor_config_sha256": PROCESSOR_CONFIG_SHA256,
        "processor_config_url": PROCESSOR_CONFIG_URL,
        "condition_order": ["source", "mask"],
        "negative_empty_encoded_independently": True,
        "branches": metadata,
    }
    return branches
