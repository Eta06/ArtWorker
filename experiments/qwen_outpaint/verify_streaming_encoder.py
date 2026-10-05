"""Reference arithmetic parity for destructive layer offloading, CPU-sized model."""
import json
import tempfile
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from streaming_encoder import StreamingLanguageModel, StreamingVisionModel
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import LanguageModel


def save_layers(directory, parameters, root):
    groups = {}
    for key, value in parameters:
        key = root + "." + key
        match = re.match(r'(language_model\.layers\.\d+|visual\.blocks\.\d+|visual\.deepstack_merger_list\.\d+)\.', key)
        group = match.group(1) if match else key.rsplit('.', 1)[0]
        groups.setdefault(group, {})[key] = value
    files = []
    for index, (group, values) in enumerate(groups.items()):
        name = root + str(index) + '.safetensors'
        mx.save_safetensors(str(directory / name), values, {'quantization_level': '4'})
        files.append(dict(group=group, file=name))
    return files


def main():
    mx.set_cache_limit(0)
    mx.random.seed(7)
    config = dict(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=128,
        attention_bias=False, rms_norm_eps=1e-6, rope_theta=10000,
        rope_scaling={"mrope_section": [6, 5, 5]})
    normal = LanguageModel(config)
    normal.update(tree_unflatten([(k, v.astype(mx.float16)) for k, v in tree_flatten(normal.parameters())]))
    nn.quantize(normal, bits=4, group_size=64)
    parameters = [(k, v) for k, v in tree_flatten(normal.parameters())]
    hidden = mx.random.normal((1, 9, 128)).astype(mx.float16)
    positions = mx.broadcast_to(mx.arange(9)[None, None], (3, 1, 9))
    indices = mx.array([2, 3, 4, 5])
    stack = [mx.random.normal((4, 128)).astype(mx.float16) for _ in range(3)]
    records = []
    for with_vision in (False, True):
        streamed = StreamingLanguageModel(config)
        nn.quantize(streamed, bits=4, group_size=64)
        streamed.update(tree_unflatten(parameters), strict=True)
        # Copy input: deepstack additions intentionally mutate each branch's hidden state.
        a = normal(hidden + mx.zeros_like(hidden), positions, indices, stack if with_vision else None)
        b = streamed(hidden + mx.zeros_like(hidden), positions, indices, stack if with_vision else None)
        mx.eval(a, b)
        error = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        released = all(layer is None for layer in streamed.layers)
        rejected_reuse = False
        try:
            streamed(hidden, positions)
        except RuntimeError:
            rejected_reuse = True
        if error != 0 or not released or not rejected_reuse:
            raise AssertionError((error, released, rejected_reuse))
        records.append({"deepstack": with_vision, "max_abs_error": error,
            "all_decoder_layers_released": released, "reuse_rejected": rejected_reuse})
    vision_config = dict(patch_size=2, temporal_patch_size=2, in_channels=3,
        hidden_size=128, num_heads=4, intermediate_size=256, depth=3,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=128,
        deepstack_visual_indexes=[0, 1, 2], preserve_input_dtype=True)
    vision = Qwen3VLVisionModel(**vision_config)
    vision.update(tree_unflatten([(k, v.astype(mx.float16)) for k, v in tree_flatten(vision.parameters())]))
    nn.quantize(vision, bits=4, group_size=64)
    vision_parameters = tree_flatten(vision.parameters())
    pixels = mx.random.normal((16, 24)).astype(mx.float16)
    grid = mx.array([[1, 4, 4]])
    vision_records = []
    for with_stack in (False, True):
        streamed = StreamingVisionModel(**vision_config)
        nn.quantize(streamed, bits=4, group_size=64)
        streamed.update(tree_unflatten(vision_parameters), strict=True)
        a, a_stack = vision(pixels, grid, return_deepstack=with_stack)
        b, b_stack = streamed(pixels, grid, return_deepstack=with_stack)
        pairs = [(a, b)] + list(zip(a_stack or [], b_stack or []))
        errors = []
        for expected, actual in pairs:
            mx.eval(expected, actual)
            errors.append(float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))))
        released = streamed.patch_embed is None and streamed.pos_embed is None and streamed.merger is None and all(block is None for block in streamed.blocks)
        mergers_released = not with_stack or all(merger is None for merger in streamed.deepstack_merger_list)
        rejected_reuse = False
        try:
            streamed(pixels, grid)
        except RuntimeError:
            rejected_reuse = True
        if any(errors) or not released or not mergers_released or not rejected_reuse:
            raise AssertionError((errors, released, mergers_released, rejected_reuse))
        vision_records.append(dict(deepstack=with_stack, max_abs_errors=errors,
            all_used_layers_released=released and mergers_released, reuse_rejected=rejected_reuse))
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        files = save_layers(directory, parameters, 'language_model') + save_layers(directory, vision_parameters, 'visual')
        (directory / 'repack_manifest.json').write_text(json.dumps(dict(files=files)))
        streamed = StreamingLanguageModel(config)
        streamed._encoder_directory = directory
        streamed.rotary_emb.update(normal.rotary_emb.parameters())
        a = normal(hidden + mx.zeros_like(hidden), positions, indices, stack)
        b = streamed(hidden + mx.zeros_like(hidden), positions, indices, stack)
        mx.eval(a, b)
        language_error = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        streamed = StreamingVisionModel(**vision_config)
        streamed._encoder_directory = directory
        streamed.rotary_pos_emb.update(vision.rotary_pos_emb.parameters())
        a, a_stack = vision(pixels, grid, return_deepstack=True)
        b, b_stack = streamed(pixels, grid, return_deepstack=True)
        errors = []
        for expected, actual in [(a,b)] + list(zip(a_stack,b_stack)):
            mx.eval(expected, actual)
            errors.append(float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)))))
        from low_memory_loader import load_component
        embedding = load_component(nn.Embedding(512,128), directory, prefix='language_model.embed_tokens')
        a, b = normal.embed_tokens(mx.array([1,3,5])), embedding(mx.array([1,3,5]))
        mx.eval(a,b)
        embedding_error = float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))))
        if language_error or any(errors) or embedding_error:
            raise AssertionError((language_error, errors, embedding_error))
        ondemand = dict(language_max_abs_error=language_error, vision_max_abs_errors=errors, embedding_max_abs_error=embedding_error)
        language_directory = directory / 'sequential'
        language_directory.mkdir()
        save_layers(language_directory, parameters, 'language_model')
        wrapper = nn.Module()
        wrapper.language_model = LanguageModel(config)
        wrapper = load_component(wrapper, language_directory, sequential=True)
        a = normal(hidden + mx.zeros_like(hidden), positions, indices, stack)
        b = wrapper.language_model(hidden + mx.zeros_like(hidden), positions, indices, stack)
        mx.eval(a,b)
        sequential_error = float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))))
        if sequential_error:
            raise AssertionError(sequential_error)
        ondemand['sequential_component_max_abs_error'] = sequential_error
    result = {"status": "passed", "cases": records, "vision_cases": vision_records,
        "ondemand_checkpoint_cases": ondemand,
        "scope": "Reduced actual decoder and vision classes, affine Q4, FP16, with/without deepstack. Not full VLM image quality or iPhone proof."}
    Path(__file__).with_name("streaming_encoder_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
