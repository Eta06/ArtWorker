"""Weight-free NumPy/MLX port of the pinned official MaskFlow inference math.

Official revision: 3125f1ecd72f5a4e068c5a1954e9d6a8df38d443.
Arrays use the official NCHW / B,C,1,H,W / B,sequence,64 conventions.
NumPy BF16 values are represented by float32 values rounded to BF16; pass
``dtype='bfloat16'`` explicitly for NumPy. MLX keeps a native BF16 dtype.
No weights, PyTorch, ControlNet, or denoising implementation is imported here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

OFFICIAL_REVISION = "3125f1ecd72f5a4e068c5a1954e9d6a8df38d443"
BRANCHES = ("pm", "pn", "nm", "nn")
VAE_SCALE_FACTOR = 8
VAE_CHANNELS = 16
PATCH_SIZE = 2
LATENTS_MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
LATENTS_STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916)


def bfloat16_round(x: Any) -> np.ndarray:
    """IEEE round-to-nearest-even FP32 -> BF16 -> FP32, including NaNs."""
    value = np.asarray(x, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = rounded & np.uint32(0xFFFF0000)
    is_nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(is_nan, bits | np.uint32(0x00400000), rounded).astype(np.uint32)
    return rounded.view(np.float32)


class _Ops:
    def __init__(self, backend: str):
        if backend == "numpy":
            self.xp = np
        elif backend == "mlx":
            import mlx.core as mx
            self.xp = mx
        else:
            raise ValueError("backend must be 'numpy' or 'mlx'")
        self.backend = backend

    def dtype(self, x, dtype=None):
        if dtype is not None:
            if str(dtype) not in ("float32", "bfloat16"):
                raise ValueError("dtype must be 'float32', 'bfloat16', or None")
            return str(dtype)
        return "bfloat16" if str(getattr(x, "dtype", "")) == "mlx.core.bfloat16" else "float32"

    def cast(self, x, dtype="float32"):
        if self.backend == "numpy":
            return bfloat16_round(x) if dtype == "bfloat16" else np.asarray(x, dtype=np.float32)
        return self.xp.array(x).astype(self.xp.bfloat16 if dtype == "bfloat16" else self.xp.float32)

    def arithmetic(self, a, b, op, dtype):
        # Explicit boundaries reproduce separate eager PyTorch BF16 operations.
        # Python scalar coefficients retain FP32 precision during arithmetic.
        # Scalar *tensor* coefficients have a different PyTorch promotion rule;
        # callers cast those explicitly before entering this helper.
        return self.cast(op(self.cast(a), self.cast(b)), dtype)

    def add(self, a, b, dtype):
        return self.arithmetic(a, b, lambda x, y: x + y, dtype)

    def sub(self, a, b, dtype):
        return self.arithmetic(a, b, lambda x, y: x - y, dtype)

    def mul(self, a, b, dtype):
        return self.arithmetic(a, b, lambda x, y: x * y, dtype)

    def div(self, a, b, dtype):
        return self.arithmetic(a, b, lambda x, y: x / y, dtype)


def sigma_grid(num_steps=50, img_seq_len=2304, *, backend="numpy", base_image_seq_len=256,
               base_shift=0.5, max_image_seq_len=8192, max_shift=0.9, shift_power=1,
               time_shift_type="exponential", use_dynamic_shifting=True, shift=1.0):
    """Official linspace(1,1/N,N), dynamic shift, then one appended zero.

    There is no terminal-sigma stretching. For 512x1152, packed length is 2304.
    ``timesteps`` are the unshifted diagnostic t grid. The model receives the
    shifted ``sigmas[:-1]`` (cast to model dtype), as in official eval_step.
    """
    if num_steps <= 0 or img_seq_len <= 0:
        raise ValueError("num_steps and img_seq_len must be positive")
    o = _Ops(backend)
    t = o.cast(np.linspace(1.0, 1.0 / num_steps, num_steps, endpoint=True).astype(np.float32))
    scale = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    mu = scale * img_seq_len + base_shift - scale * base_image_seq_len if use_dynamic_shifting else shift
    if time_shift_type in ("exponential", "linear"):
        factor = np.float32(np.exp(mu) if time_shift_type == "exponential" else mu)
        sigmas = factor / (factor + (1.0 / t - 1.0) ** shift_power)
        d_sigmas_dt = (np.float32(shift_power) / factor) * ((sigmas / t) ** 2) * ((1.0 / t - 1.0) ** (shift_power - 1))
    else:
        sigmas, d_sigmas_dt = t, o.xp.ones_like(t)
    return {"timesteps": t, "sigmas": o.xp.concatenate((o.cast(sigmas), o.cast([0.0]))),
            "d_sigmas_dt": o.cast(d_sigmas_dt), "mu": float(mu)}


def pack_latents(latents, *, backend="numpy"):
    """Official 2x2 patch packing: 16 VAE channels become 64 token channels."""
    o = _Ops(backend)
    x = o.xp.array(latents)
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError("Only the official single-frame image case is supported")
        x = x[:, :, 0]
    if x.ndim != 4:
        raise ValueError("Expected [B,C,H,W] or [B,C,1,H,W]")
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("Latent height and width must be divisible by 2")
    return x.reshape(b, c, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(b, h * w // 4, c * 4)


def unpack_latents(latents, height, width, vae_scale_factor=8, *, backend="numpy"):
    """Inverse official packing, with Diffusers' image-to-latent rounding."""
    o = _Ops(backend)
    x = o.xp.array(latents)
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    if x.ndim != 3 or x.shape[-1] % 4 or x.shape[1] != h * w // 4:
        raise ValueError("Packed tensor does not match requested image geometry")
    b, _, dim = x.shape
    return x.reshape(b, h // 2, w // 2, dim // 4, 2, 2).transpose(0, 3, 1, 4, 2, 5).reshape(b, dim // 4, 1, h, w)


def _nearest_resize(x, height, width, o):
    # F.interpolate(mode='nearest') uses floor(output_index * input/output).
    h, w = x.shape[-2:]
    iy = np.floor(np.arange(height) * (h / height)).astype(np.int32)
    ix = np.floor(np.arange(width) * (w / width)).astype(np.int32)
    if o.backend == "mlx":
        iy, ix = o.xp.array(iy), o.xp.array(ix)
    return o.xp.take(o.xp.take(x, iy, axis=-2), ix, axis=-1)


def dilate_mask(mask, kernel_size=25, *, backend="numpy", dtype=None):
    o = _Ops(backend)
    dtype = o.dtype(mask, dtype)
    x = o.cast(mask, dtype)
    k = int(kernel_size) + 1 - int(kernel_size) % 2
    if k < 1 or x.ndim != 4:
        raise ValueError("Expected NCHW mask and positive kernel_size")
    p = k // 2
    # Separable square max pool. Zero is sufficient for the [0,1] mask domain,
    # but negative infinity preserves official max-pool semantics for any input.
    for axis in (-2, -1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (p, p)
        padded = o.xp.pad(x, pad, constant_values=-float("inf"))
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(0, x.shape[axis])
        result = padded[tuple(slices)]
        for i in range(1, k):
            slices[axis] = slice(i, i + x.shape[axis])
            result = o.xp.maximum(result, padded[tuple(slices)])
        x = result
    return o.cast(x, dtype)


def blur_mask(mask, kernel_size=25, sigma=25.0, *, backend="numpy", dtype=None):
    """Torchvision Gaussian, reflect pad, then the official mask-dependent edit.

    BF16 kernel construction follows torchvision's operation cast points.
    A 2D kernel is used because a separable convolution introduces an extra
    BF16 rounding point that the official conv2d does not have.
    """
    o = _Ops(backend)
    dtype = o.dtype(mask, dtype)
    x = o.cast(mask, dtype)
    k = int(kernel_size) + 1 - int(kernel_size) % 2
    p = k // 2
    if k < 1 or sigma <= 0 or x.ndim != 4 or min(x.shape[-2:]) <= p:
        raise ValueError("Invalid Gaussian kernel/sigma or reflect-padding geometry")
    coords = o.cast(np.linspace(-p, p, k, dtype=np.float32), dtype)
    exponent = o.mul(-0.5, o.cast(o.div(coords, float(sigma), dtype) ** 2, dtype), dtype)
    kernel = o.cast(o.xp.exp(exponent), dtype)
    kernel = o.div(kernel, o.cast(o.xp.sum(o.cast(kernel)), dtype), dtype)
    kernel2d = o.mul(kernel[:, None], kernel[None, :], dtype)
    # Reflection (excluding edge pixel) indices match F.pad(mode='reflect').
    iy = np.pad(np.arange(x.shape[-2], dtype=np.int32), (p, p), mode="reflect")
    ix = np.pad(np.arange(x.shape[-1], dtype=np.int32), (p, p), mode="reflect")
    if o.backend == "mlx":
        padded = o.xp.take(o.xp.take(x, o.xp.array(iy), axis=-2), o.xp.array(ix), axis=-1)
        # Conv2d is NHWC and weights OHWI. Fold channels into the batch to
        # avoid grouped backend differences; each channel uses the same kernel.
        b, c, h, w = x.shape
        conv_input = padded.reshape(b * c, h + 2 * p, w + 2 * p, 1)
        conv_weight = kernel2d.reshape(1, k, k, 1)
        blurred = o.xp.conv2d(o.cast(conv_input), o.cast(conv_weight)).reshape(b, c, h, w)
    else:
        padded = np.take(np.take(x, iy, axis=-2), ix, axis=-1)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(-2, -1))
        # einsum avoids allocating all H*W*k*k products for a full canvas.
        blurred = np.einsum("bchwkl,kl->bchw", windows, np.asarray(kernel2d), optimize=False)
    blurred = o.cast(blurred, dtype)
    return o.xp.where(x >= 1.0, o.cast(1.0, dtype), o.mul(blurred, 2.0, dtype))


def process_mask(mask, *, backend="numpy", dtype="bfloat16", dilation_kernel=25,
                 blur_kernel=25, blur_sigma=25.0):
    """Apply model-dtype cast, one dilation, one blur at full model resolution."""
    o = _Ops(backend)
    value = o.cast(mask, dtype)
    if dilation_kernel > 0:
        value = dilate_mask(value, dilation_kernel, backend=backend, dtype=dtype)
    if blur_kernel > 0:
        value = blur_mask(value, blur_kernel, blur_sigma, backend=backend, dtype=dtype)
    return value


def encode_mask(mask, vae_scale_factor=8, vae_channels=16, *, backend="numpy", dtype=None):
    """Control mask: nearest 8x, channel mean/repeat, not the VAE mask image."""
    o = _Ops(backend)
    dtype = o.dtype(mask, dtype)
    x = o.cast(mask, dtype)
    h, w = x.shape[-2:]
    x = _nearest_resize(x, h // vae_scale_factor, w // vae_scale_factor, o)
    x = o.cast(o.xp.mean(o.cast(x), axis=1, keepdims=True), dtype)
    x = o.xp.repeat(x, vae_channels, axis=1)[:, :, None]
    return o.cast(o.xp.clip(x, 0.0, 1.0), dtype)


def neighbor_sum(x, *, backend="numpy", dtype=None):
    """Four adjacent pixels, with zero edge padding (no wrap/reflect)."""
    o = _Ops(backend)
    dtype = o.dtype(x, dtype)
    value = o.cast(x, dtype)
    if value.ndim != 4:
        raise ValueError("Expected [B,C,H,W]")
    padded = o.xp.pad(o.cast(value), [(0, 0), (0, 0), (1, 1), (1, 1)])
    result = padded[:, :, :-2, 1:-1] + padded[:, :, 1:-1, :-2]
    result = result + padded[:, :, 1:-1, 2:] + padded[:, :, 2:, 1:-1]
    return o.cast(result, dtype)


def poisson_refine(g, x_S, M, soft_M, poisson_lambda_e=1.0, poisson_lambda_s=1.0,
                   poisson_num_iter=50, poisson_momentum=0.1, eps=1e-6,
                   disable_progress_bar=True, *, backend="numpy", dtype=None):
    """Official weighted four-neighbor soft Poisson, Jacobi with momentum.

    M is binary (processed latent mask > 0); soft_M supplies edge weights.
    Official masks promote the solve to FP32 even when g/source are BF16.
    The standalone return is FP32; the pipeline repacks and casts to BF16.
    """
    del disable_progress_bar
    o = _Ops(backend)
    dtype = o.dtype(g, dtype)
    g, x_S = o.cast(g, dtype), o.cast(x_S, dtype)
    M = o.xp.broadcast_to(o.cast(M), g.shape)
    soft_M = o.xp.broadcast_to(o.cast(soft_M), g.shape)
    y = M * o.cast(g) + (1.0 - M) * o.cast(x_S)
    D = neighbor_sum(o.xp.ones_like(g), backend=backend, dtype=dtype)
    nsum_soft_M = neighbor_sum(soft_M, backend=backend)
    weight_sum = 0.5 * soft_M * o.cast(D) + 0.5 * nsum_soft_M
    nsum_g = neighbor_sum(g, backend=backend, dtype=dtype)
    nsum_g_soft_M = neighbor_sum(soft_M * o.cast(g), backend=backend)
    weighted_nsum_g = 0.5 * soft_M * o.cast(nsum_g) + 0.5 * nsum_g_soft_M
    div_g = weight_sum * o.cast(g) - weighted_nsum_g
    x_S_out = (1.0 - M) * o.cast(x_S)
    nsum_x_S = neighbor_sum(x_S_out, backend=backend)
    nsum_x_S_soft_M = neighbor_sum(soft_M * x_S_out, backend=backend)
    soft_boundary = 0.5 * soft_M * nsum_x_S + 0.5 * nsum_x_S_soft_M
    diag = weight_sum + poisson_lambda_e * soft_M + poisson_lambda_s * (1.0 - soft_M)
    b = div_g + soft_boundary + poisson_lambda_e * soft_M * o.cast(g) + poisson_lambda_s * (1.0 - soft_M) * o.cast(x_S)
    for _ in range(poisson_num_iter):
        y_in = M * y
        nsum_y_in = neighbor_sum(y_in, backend=backend)
        nsum_y_in_soft_M = neighbor_sum(soft_M * y_in, backend=backend)
        y_next = (0.5 * soft_M * nsum_y_in + 0.5 * nsum_y_in_soft_M + b) / o.xp.maximum(diag, eps)
        y_next = M * y_next + (1.0 - M) * o.cast(x_S)
        y = poisson_momentum * y + (1.0 - poisson_momentum) * y_next
    return y


def branch_conditions(branch):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown CFG branch {branch!r}")
    return branch[0] == "p", branch[1] == "m"


def cfg_coefficients(text_scale=1.0, mask_scale=1.0, interaction_scale=None):
    interaction = text_scale if interaction_scale is None else interaction_scale
    return {"pm": interaction, "pn": text_scale - interaction,
            "nm": mask_scale - interaction, "nn": 1.0 - text_scale - mask_scale + interaction}


def required_branches(text_scale=1.0, mask_scale=1.0, interaction_scale=None, rescale=True):
    coefficients = cfg_coefficients(text_scale, mask_scale, interaction_scale)
    return [name for name in BRANCHES if coefficients[name] != 0 or (rescale and name == "pm")]


def combine_predictions(predictions, text_scale=4.0, mask_scale=1.0, interaction_scale=None,
                        rescale=True, *, backend="numpy", dtype=None):
    """CFG4 default uses pm + nm; norm rescale is per token's last axis."""
    o = _Ops(backend)
    needed = required_branches(text_scale, mask_scale, interaction_scale, rescale)
    missing = set(needed) - set(predictions)
    if missing:
        raise ValueError(f"Missing CFG predictions: {sorted(missing)}")
    dtype = o.dtype(predictions[needed[0]], dtype)
    predictions = {name: o.cast(value, dtype) for name, value in predictions.items()}
    coefficients = cfg_coefficients(text_scale, mask_scale, interaction_scale)
    if mask_scale == 1 and coefficients["pn"] == 0 and text_scale not in (0, 1):
        combined = o.add(predictions["nm"], o.mul(text_scale, o.sub(predictions["pm"], predictions["nm"], dtype), dtype), dtype)
    else:
        terms = [o.mul(coefficients[name], predictions[name], dtype) for name in needed if coefficients[name] != 0]
        combined = terms[0]
        for term in terms[1:]:
            combined = o.add(combined, term, dtype)
    if rescale:
        def norm(x):
            return o.xp.maximum(o.cast(o.xp.sqrt(o.xp.sum(o.cast(x) ** 2, axis=-1, keepdims=True)), dtype), o.cast(1e-6, dtype))
        combined = o.mul(combined, o.div(norm(predictions["pm"]), norm(combined), dtype), dtype)
    return combined


def _expand_sigma(sigma, ndim, o):
    sigma = o.cast(sigma)
    while sigma.ndim < ndim:
        sigma = sigma[..., None]
    return sigma


def add_noise_by_sigmas(noise, x0, sigmas, source=None, mask=None, *, unmask_with="noisy_source",
                        background_noise_power=1.0, backend="numpy", dtype=None):
    o = _Ops(backend)
    dtype = o.dtype(x0, dtype)
    x0, noise = o.cast(o.cast(x0, dtype)), o.cast(o.cast(noise, dtype))
    sigmas = _expand_sigma(sigmas, x0.ndim, o)
    target = (1.0 - sigmas) * x0 + sigmas * noise
    if source is None or mask is None:
        return o.cast(target, dtype)
    source = o.cast(o.cast(source, dtype))
    mask_native = o.cast(mask, dtype)
    mask = o.cast(mask_native)
    # Official (1-mask) is evaluated before the shaped FP32 sigma expression
    # promotes the multiplication; retaining this BF16 subtraction matters.
    inverse_mask = o.cast(o.sub(1.0, mask_native, dtype))
    if unmask_with == "target":
        background = x0
    elif unmask_with == "source":
        background = source
    elif unmask_with == "noisy_target":
        background = target
    elif unmask_with == "noisy_source":
        background_sigmas = sigmas ** background_noise_power
        background = (1.0 - background_sigmas) * source + background_sigmas * noise
    else:
        raise ValueError(f"Unsupported unmask_with {unmask_with!r}")
    background_product = (o.mul(inverse_mask, background, dtype)
                          if unmask_with == "source" else inverse_mask * background)
    return o.cast(mask * target + background_product, dtype)


def scheduler_step(xt, vt, curr_sigma, next_sigma, d_sigma_dt=None, source=None, mask=None,
                   noise=None, *, unmask_with="noisy_source", background_noise_power=1.0,
                   backend="numpy", dtype=None, mask_dtype=None, velocity_dtype=None):
    """Euler followed by official source/noise bridge at NEXT sigma.

    d_sigma_dt is accepted for source-compatible calls and deliberately unused.
    A scalar FP32 sigma multiplied by BF16 vt is BF16 in PyTorch; a shaped
    sigma promotes the product to FP32. Explicit casts preserve that distinction.
    """
    del d_sigma_dt
    o = _Ops(backend)
    dtype = o.dtype(xt, dtype)
    # Poisson inversion returns FP32 despite BF16 xt. MLX can infer it from vt;
    # NumPy's emulated BF16 representation needs an explicit velocity_dtype for
    # this mixed case. Without one, dtype describes both emulated input arrays.
    if velocity_dtype is None and backend == "numpy":
        velocity_dtype = dtype
    velocity_dtype = o.dtype(vt, velocity_dtype)
    delta = o.cast(next_sigma) - o.cast(curr_sigma)
    product_dtype = velocity_dtype if delta.ndim == 0 else "float32"
    delta_for_product = o.cast(delta, product_dtype) if delta.ndim == 0 else delta
    euler = o.cast(o.cast(o.cast(xt, dtype)) + o.mul(delta_for_product, o.cast(vt, velocity_dtype), product_dtype), dtype)
    if source is None or mask is None or noise is None:
        return euler
    if mask_dtype is None and backend == "numpy":
        mask_dtype = dtype
    mask_dtype = o.dtype(mask, mask_dtype)
    first_dtype = "float32" if mask_dtype == "float32" or dtype == "float32" else "bfloat16"
    mask = o.cast(mask, mask_dtype)
    source, noise = o.cast(source, dtype), o.cast(noise, dtype)
    if unmask_with in ("source", "target"):
        background = source
        background_dtype = dtype
    elif unmask_with in ("noisy_source", "noisy_target"):
        sigma = o.cast(next_sigma) ** (background_noise_power if unmask_with == "noisy_source" else 1.0)
        background_dtype = dtype if sigma.ndim == 0 else "float32"
        source_sigma = o.cast(1.0 - sigma, background_dtype) if sigma.ndim == 0 else 1.0 - sigma
        noise_sigma = o.cast(sigma, background_dtype) if sigma.ndim == 0 else sigma
        background = o.add(o.mul(source_sigma, source, background_dtype), o.mul(noise_sigma, noise, background_dtype), background_dtype)
    else:
        raise ValueError(f"Unsupported unmask_with {unmask_with!r}")
    second_dtype = "float32" if mask_dtype == "float32" or background_dtype == "float32" else "bfloat16"
    output_dtype = "float32" if first_dtype == "float32" or second_dtype == "float32" else "bfloat16"
    # A shaped FP32 sigma can promote only the background term. The masked
    # Euler product is still BF16 if both of its operands are BF16.
    first = o.mul(mask, euler, first_dtype)
    second = o.mul(o.sub(1.0, mask, mask_dtype), background, second_dtype)
    return o.add(first, second, output_dtype)


def predict_x0(xt, sigma, d_sigma_dt, v, source=None, mask=None, noise=None, *,
               unmask_with="noisy_source", background_noise_power=1.0,
               backend="numpy", dtype=None):
    del d_sigma_dt
    o = _Ops(backend)
    dtype = o.dtype(xt, dtype)
    xt, v = o.cast(o.cast(xt, dtype)), o.cast(v)
    sigma = _expand_sigma(sigma, xt.ndim, o)
    x0 = xt - sigma * v
    if unmask_with == "noisy_source" and background_noise_power != 1.0:
        if source is None or mask is None or noise is None:
            raise ValueError("Nonlinear path inverse requires source, mask, noise")
        correction = (background_noise_power - 1.0) * sigma ** background_noise_power
        x0 = x0 + (1.0 - o.cast(mask)) * correction * (o.cast(noise) - o.cast(source))
    return o.cast(x0, dtype)


def apply_poisson_to_prediction(xt, pred, curr_sigma, source, mask_latents, noise, height, width,
                                *, backend="numpy", dtype=None, unmask_with="noisy_source",
                                background_noise_power=1.0, poisson_lambda_e=1.0,
                                poisson_lambda_s=1.0, poisson_num_iter=50, poisson_momentum=0.1):
    """Official x0 FP32 estimate -> BF16 -> Poisson -> BF16 -> FP32 velocity.

    The default gamma=1 subtraction happens in model dtype, then division by
    the expanded FP32 sigma promotes the velocity to FP32.
    The nonlinear correction branch retains the FP32 pre-refinement estimate.
    """
    o = _Ops(backend)
    dtype = o.dtype(xt, dtype)
    xt, source, mask_latents, noise = [o.cast(v, dtype) for v in (xt, source, mask_latents, noise)]
    source_grid = unpack_latents(source, height, width, backend=backend)[:, :, 0]
    soft_M = unpack_latents(mask_latents, height, width, backend=backend)[:, :, 0]
    x0_packed = predict_x0(o.cast(xt), curr_sigma, None, pred, source, mask_latents, noise,
                           backend=backend, dtype="float32", unmask_with=unmask_with,
                           background_noise_power=background_noise_power)
    g = unpack_latents(o.cast(x0_packed, dtype), height, width, backend=backend)[:, :, 0]
    refined = poisson_refine(g, source_grid, soft_M > 0, soft_M,
                             poisson_lambda_e, poisson_lambda_s, poisson_num_iter,
                             poisson_momentum, backend=backend, dtype=dtype)
    refined = o.cast(pack_latents(refined, backend=backend), dtype)
    sigma = _expand_sigma(curr_sigma, xt.ndim, o)
    denominator = o.xp.maximum(sigma, 1e-4)
    if unmask_with != "noisy_source" or background_noise_power == 1.0:
        return o.cast(o.sub(xt, refined, dtype)) / denominator
    return o.cast(pred) + (x0_packed - o.cast(refined)) / denominator


def prepare_canvas(source, *, fill="black", height=1152, width=512, known_top=320):
    """Declared full-canvas inputs; 512x512 source occupies y=320:832.

    Accept HWC, CHW, or 1CHW RGB uint8 / float32 [0,1]. Black=0, gray=0.5.
    Mask 1 denotes expansion; mask 0 denotes known source before morphology.
    No aspect-ratio crop or automatic max-resolution upscale occurs here.
    """
    x = np.asarray(source)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x.transpose(2, 0, 1)
    if x.shape != (3, 512, 512) or width != 512 or height != 1152 or known_top != 320:
        raise ValueError("This declared protocol requires 512x512 -> 512x1152 at y320:832")
    x = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
    if fill not in ("black", "gray") or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Expected RGB [0,1] source and fill='black' or 'gray'")
    fill_value = 0.0 if fill == "black" else 0.5
    canvas = np.full((1, 3, height, width), fill_value, dtype=np.float32)
    canvas[:, :, known_top:known_top + 512] = x
    mask = np.ones_like(canvas)
    mask[:, :, known_top:known_top + 512] = 0.0
    return {"source_canvas": canvas, "mask": mask,
            "metadata": {"width": width, "height": height, "known_box_xyxy": [0, 320, 512, 832],
                         "fill": fill, "fill_value": fill_value, "edit_mask_value": 1,
                         "preprocess_geometry": "aligned full canvas; no crop or scale"}}


def _resize_rgb_pil(image, height, width):
    from PIL import Image
    image = np.asarray(image, dtype=np.float32)
    if image.shape[-2:] == (height, width):
        return image
    outputs = []
    for item in image:
        # Official resize_image rounds before uint8 conversion and uses PIL Lanczos.
        pixels = np.rint(item.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        resized = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
        outputs.append(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(outputs)


def prepare_condition_images(source_canvas, pixel_mask, *, backend="numpy", dtype="bfloat16",
                              max_condition_resolution=384 * 384, divisible_by=32,
                              dilation_kernel=25, blur_kernel=25, blur_sigma=25.0):
    """Weight-free preprocessing of BOTH source+mask VLM/VAE inputs.

    dit_conditions must each be separately VAE-encoded (argmax/mode), normalized
    with Qwen's per-channel mean/std, then packed. The control mask here is only
    interpolated; it cannot replace the independently VAE-encoded mask image.
    """
    o = _Ops(backend)
    source_cpu = np.asarray(source_canvas, dtype=np.float32)
    if source_cpu.ndim != 4 or source_cpu.shape[1] != 3 or tuple(pixel_mask.shape) != source_cpu.shape:
        raise ValueError("Aligned source/mask must each be [B,3,H,W]")
    h, w = source_cpu.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError("Full canvas dimensions must be divisible by 16")
    aspect = w / h
    ch = math.floor(math.sqrt(max_condition_resolution / aspect) / divisible_by) * divisible_by
    cw = math.floor(math.sqrt(max_condition_resolution * aspect) / divisible_by) * divisible_by
    source = o.cast(source_cpu, dtype)
    mask = process_mask(pixel_mask, backend=backend, dtype=dtype, dilation_kernel=dilation_kernel,
                        blur_kernel=blur_kernel, blur_sigma=blur_sigma)
    vlm_source = o.cast(_resize_rgb_pil(source_cpu, ch, cw), dtype)
    vlm_mask = o.cast(_nearest_resize(o.cast(mask), ch, cw, o), dtype)
    def normalized_image(x):
        return o.sub(o.mul(x, 2.0, dtype), 1.0, dtype)[:, :, None]
    control = encode_mask(mask, backend=backend, dtype=dtype)
    return {"raw_source": source, "mask": mask,
            "dit_conditions": {"source": normalized_image(source), "mask": normalized_image(mask)},
            "vlm_conditions": {"source": vlm_source, "mask": vlm_mask},
            "mask_latents": pack_latents(control, backend=backend),
            "image_shapes": [[(1, h // 16, w // 16)] * 3 for _ in range(source.shape[0])],
            "height": h, "width": w,
            "metadata": {"vlm_height": ch, "vlm_width": cw, "vae_condition_order": ["source", "mask"],
                         "vlm_condition_order": ["source", "mask"], "vae_sample_mode": "argmax",
                         "transformer_conditioning": "token concatenation: target, source, mask",
                         "packed_dimension": 64, "controlnet_context_channels": None}}


def normalize_vae_latents(latents, *, backend="numpy", dtype=None, mean=LATENTS_MEAN, std=LATENTS_STD):
    o = _Ops(backend)
    dtype = o.dtype(latents, dtype)
    x = o.cast(latents, dtype)
    shape = (1, len(mean)) + (1,) * (x.ndim - 2)
    return o.div(o.sub(x, o.cast(mean, dtype).reshape(shape), dtype), o.cast(std, dtype).reshape(shape), dtype)


def denormalize_vae_latents(latents, *, backend="numpy", dtype=None, mean=LATENTS_MEAN, std=LATENTS_STD):
    o = _Ops(backend)
    dtype = o.dtype(latents, dtype)
    x = o.cast(latents, dtype)
    shape = (1, len(mean)) + (1,) * (x.ndim - 2)
    return o.add(o.mul(x, o.cast(std, dtype).reshape(shape), dtype), o.cast(mean, dtype).reshape(shape), dtype)


def pixel_blend(output, raw_source, processed_mask, *, backend="numpy", dtype=None, output_dtype=None):
    """Official final blend; exact preservation applies where processed mask==0.

    Dilation/blur edits a boundary strip of the original square. This operation
    does not claim exact preservation of the original uint8 512x512 image.
    """
    o = _Ops(backend)
    dtype = o.dtype(raw_source, dtype)
    mask = o.cast(processed_mask, dtype)
    output_dtype = o.dtype(output, output_dtype)
    output = o.cast(output, output_dtype)
    first_dtype = "float32" if output_dtype == "float32" or dtype == "float32" else dtype
    first = o.mul(mask, output, first_dtype)
    second = o.mul(o.sub(1.0, mask, dtype), o.cast(raw_source, dtype), dtype)
    return o.add(first, second, first_dtype)


def hard_paste_original(output, original, *, known_top=320):
    """Separate NumPy output policy restoring original square exactly.

    Input/output use matching pixel dtype/layout; this is not MaskFlow math.
    """
    value, source = np.array(output, copy=True), np.asarray(original)
    if value.ndim == 3 and value.shape[-1] == 3 and source.shape == (512, 512, 3):
        value[known_top:known_top + 512, :512] = source
    elif value.ndim == 4 and value.shape[:2] == (1, 3) and source.shape == (1, 3, 512, 512):
        value[:, :, known_top:known_top + 512, :512] = source
    elif value.ndim == 3 and value.shape[0] == 3 and source.shape == (3, 512, 512):
        value[:, known_top:known_top + 512, :512] = source
    else:
        raise ValueError("Expected matching HWC, CHW or 1CHW RGB pixel layouts")
    if value.dtype != source.dtype:
        raise ValueError("Matching pixel dtypes are required for exact hard paste")
    return value
