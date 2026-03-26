from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pywt
import tifffile
from scipy import integrate, special
from tqdm import tqdm


BACKGROUND_MODE_MAP = {
    "strong_high_snr": 1,
    "weak_high_snr": 2,
    "weak_low_snr": 3,
    "medium_low_snr": 4,
    "strong_low_snr": 5,
    "none": 6,
}

DEBLURRING_METHOD_MAP = {
    "lucy_richardson": 1,
    "landweber": 2,
    "none": 3,
}

OVERSAMPLING_METHOD_MAP = {
    "spatial": 1,
    "fourier": 2,
    "none": 3,
}

BACKEND_CHOICES = {"auto", "cpu", "cuda"}
MODE_CHOICES = {"exact_cpu_full", "windowed_gpu"}


@dataclass
class SparseSIMConfig:
    pixel_size_nm: float
    wavelength_nm: float
    effective_na: float
    sparse_iter: int = 120
    fidelity: float = 150.0
    z_continuity: float = 1.0
    sparsity: float = 6.0
    deconv_iter: int = 8
    background_mode: str = "none"
    deblurring_method: str = "lucy_richardson"
    oversampling_method: str = "none"
    debug_max_slices: int | None = None
    three_d: bool = True
    psf_z_m: float = 0.0
    psf_integration_samples: int = 1024
    mode: str = "exact_cpu_full"
    window_size: int = 32
    halo: int = 4
    backend: str = "cpu"
    gpu_device_index: int = 0
    show_progress: bool = True

    def validate(self) -> None:
        if self.pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be > 0")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be > 0")
        if self.effective_na <= 0:
            raise ValueError("effective_na must be > 0")
        if self.sparse_iter <= 0:
            raise ValueError("sparse_iter must be > 0")
        if self.fidelity <= 0:
            raise ValueError("fidelity must be > 0")
        if self.sparsity < 0:
            raise ValueError("sparsity must be >= 0")
        if self.z_continuity < 0:
            raise ValueError("z_continuity must be >= 0")
        if self.deconv_iter <= 0:
            raise ValueError("deconv_iter must be > 0")
        if self.background_mode not in BACKGROUND_MODE_MAP:
            raise ValueError(f"Unsupported background_mode: {self.background_mode}")
        if self.deblurring_method not in DEBLURRING_METHOD_MAP:
            raise ValueError(f"Unsupported deblurring_method: {self.deblurring_method}")
        if self.oversampling_method not in OVERSAMPLING_METHOD_MAP:
            raise ValueError(f"Unsupported oversampling_method: {self.oversampling_method}")
        if self.debug_max_slices is not None and self.debug_max_slices <= 0:
            raise ValueError("debug_max_slices must be > 0 when provided")
        if self.psf_integration_samples < 16:
            raise ValueError("psf_integration_samples must be >= 16")
        if self.mode not in MODE_CHOICES:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3")
        if self.halo < 0:
            raise ValueError("halo must be >= 0")
        if self.mode == "windowed_gpu" and self.window_size <= 2 * self.halo:
            raise ValueError("window_size must be > 2 * halo for windowed_gpu mode")
        if self.backend not in BACKEND_CHOICES:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.gpu_device_index < 0:
            raise ValueError("gpu_device_index must be >= 0")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tiff_stack(path: Path) -> tuple[np.ndarray, str]:
    arr = tifffile.imread(path)
    original_dtype = str(arr.dtype)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D stack, but got shape={arr.shape}")

    return arr, original_dtype


def convert_for_saving(img: np.ndarray, original_dtype: str, original_max: float) -> np.ndarray:
    img = np.asarray(img)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)

    if original_max > 0:
        img = img * original_max

    dtype = np.dtype(original_dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        img = np.clip(np.rint(img), info.min, info.max).astype(dtype)
    else:
        img = img.astype(dtype)
    return img


def progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if enabled:
        return tqdm(iterable, **kwargs)
    return iterable


def resolve_backend(name: str, gpu_device_index: int) -> dict[str, Any]:
    backend = name.lower()
    if backend == "auto":
        try:
            cache_dir = Path(__file__).resolve().parent / ".cupy_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(__file__).resolve().parent / ".cupy_tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
            os.environ.setdefault("TMP", str(temp_dir))
            os.environ.setdefault("TEMP", str(temp_dir))
            os.environ.setdefault("TMPDIR", str(temp_dir))
            tempfile.tempdir = str(temp_dir)
            import cupy as cp  # type: ignore

            cp.cuda.Device(gpu_device_index).use()
            return {
                "name": "cuda",
                "xp": cp,
                "fft": cp.fft,
                "device_index": gpu_device_index,
                "device_name": cp.cuda.runtime.getDeviceProperties(gpu_device_index)["name"].decode("utf-8"),
            }
        except Exception:
            backend = "cpu"

    if backend == "cpu":
        return {
            "name": "cpu",
            "xp": np,
            "fft": np.fft,
            "device_index": None,
            "device_name": None,
        }

    if backend == "cuda":
        cache_dir = Path(__file__).resolve().parent / ".cupy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(__file__).resolve().parent / ".cupy_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
        os.environ.setdefault("TMP", str(temp_dir))
        os.environ.setdefault("TEMP", str(temp_dir))
        os.environ.setdefault("TMPDIR", str(temp_dir))
        tempfile.tempdir = str(temp_dir)
        try:
            import cupy as cp  # type: ignore
        except ImportError as exc:
            raise ImportError("backend='cuda' requires CuPy to be installed.") from exc

        cp.cuda.Device(gpu_device_index).use()
        return {
            "name": "cuda",
            "xp": cp,
            "fft": cp.fft,
            "device_index": gpu_device_index,
            "device_name": cp.cuda.runtime.getDeviceProperties(gpu_device_index)["name"].decode("utf-8"),
        }

    raise ValueError(f"Unsupported backend: {name}")


def to_backend_array(arr: Any, backend: dict[str, Any], dtype: Any = None) -> Any:
    xp = backend["xp"]
    if dtype is None:
        return xp.asarray(arr)
    return xp.asarray(arr, dtype=dtype)


def to_cpu_array(arr: Any, backend: dict[str, Any]) -> np.ndarray:
    if backend["name"] == "cuda":
        return backend["xp"].asnumpy(arr)
    return np.asarray(arr)


def array_max(arr: Any, backend: dict[str, Any]) -> float:
    return float(to_cpu_array(backend["xp"].max(arr), backend))


def release_backend_memory(backend: dict[str, Any]) -> None:
    if backend["name"] != "cuda":
        return
    cp = backend["xp"]
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def low_frequency_resolve(coeffs: list[Any], dlevel: int) -> list[Any]:
    c_an = coeffs[0]
    vec: list[Any] = [c_an]
    for i in range(1, dlevel + 1):
        c_h, c_v, c_d = coeffs[i]
        zeros = np.zeros_like(c_h)
        vec.append((zeros, zeros, zeros))
    return vec


def background_estimation(
    imgs: np.ndarray,
    th: int = 1,
    dlevel: int = 7,
    wavename: str = "db6",
    iterations: int = 3,
    show_progress: bool = True,
) -> np.ndarray:
    imgs = np.asarray(imgs, dtype=np.float32)
    x, y, z = imgs.shape[1], imgs.shape[2], imgs.shape[0]

    padded = imgs
    if x < y:
        pad_h = y - x
        padded = np.pad(imgs, ((0, 0), (0, pad_h), (0, 0)), mode="symmetric")

    background = np.zeros_like(padded, dtype=np.float32)
    for frame in progress(range(padded.shape[0]), show_progress, desc="Background", leave=False):
        initial = padded[frame]
        res = initial.copy()
        b_iter = np.zeros_like(initial, dtype=np.float32)
        for _ in range(iterations):
            coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
            low_only = low_frequency_resolve(coeffs, dlevel)
            b_iter = pywt.waverec2(low_only, wavelet=wavename)
            b_iter = b_iter[: initial.shape[0], : initial.shape[1]]
            if th > 0:
                eps = np.sqrt(np.abs(res)) / 2.0
                ind = initial > (b_iter + eps)
                res[ind] = b_iter[ind] + eps[ind]

                coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                low_only = low_frequency_resolve(coeffs, dlevel)
                b_iter = pywt.waverec2(low_only, wavelet=wavename)
                b_iter = b_iter[: initial.shape[0], : initial.shape[1]]
        background[frame] = b_iter

    return background[:, :x, :y].astype(np.float32)


def apply_background_mode(stack: np.ndarray, mode: str, show_progress: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    code = BACKGROUND_MODE_MAP[mode]
    stack = np.asarray(stack, dtype=np.float32)

    if code == 6:
        return stack.copy(), None

    if code == 1:
        backgrounds = background_estimation(stack / 2.0, show_progress=show_progress)
        return stack - backgrounds, backgrounds
    if code == 2:
        backgrounds = background_estimation(stack / 2.5, show_progress=show_progress)
        return stack - backgrounds, backgrounds

    sub_temp = stack.copy()
    mean_val = float(np.mean(stack))
    if code == 3:
        sub_temp[sub_temp > mean_val] = mean_val
    elif code == 4:
        sub_temp[sub_temp > (mean_val / 2.0)] = mean_val / 2.0
    elif code == 5:
        sub_temp[sub_temp > (mean_val / 2.5)] = mean_val / 2.5

    backgrounds = background_estimation(sub_temp, show_progress=show_progress)
    return stack - backgrounds, backgrounds


def spatial_upsample(stack: np.ndarray, factor: int = 2) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        out = np.zeros((stack.shape[0] * factor, stack.shape[1] * factor), dtype=np.float32)
        out[0::factor, 0::factor] = stack
        return out

    out = np.zeros((stack.shape[0], stack.shape[1] * factor, stack.shape[2] * factor), dtype=np.float32)
    out[:, 0::factor, 0::factor] = stack
    return out


def f_interp_2d(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    img_h, img_w = img.shape
    new_h, new_w = new_shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Invalid new shape: {new_shape}")

    nyq_h = int(np.ceil((img_h + 1) / 2))
    nyq_w = int(np.ceil((img_w + 1) / 2))
    scale = (new_h / img_h) * (new_w / img_w)

    spectrum = scale * np.fft.fft2(img)
    interp = np.zeros((new_h, new_w), dtype=np.complex64)

    interp[:nyq_h, :nyq_w] = spectrum[:nyq_h, :nyq_w]
    interp[new_h - (img_h - nyq_h) :, :nyq_w] = spectrum[nyq_h:, :nyq_w]
    interp[:nyq_h, new_w - (img_w - nyq_w) :] = spectrum[:nyq_h, nyq_w:]
    interp[new_h - (img_h - nyq_h) :, new_w - (img_w - nyq_w) :] = spectrum[nyq_h:, nyq_w:]

    if img_h % 2 == 0 and new_h != img_h:
        interp[nyq_h - 1, :] /= 2.0
        interp[nyq_h - 1 + (new_h - img_h), :] = interp[nyq_h - 1, :]
    if img_w % 2 == 0 and new_w != img_w:
        interp[:, nyq_w - 1] /= 2.0
        interp[:, nyq_w - 1 + (new_w - img_w)] = interp[:, nyq_w - 1]

    return np.fft.ifft2(interp).real.astype(np.float32)


def fourier_upsample(stack: np.ndarray, factor: int = 2) -> np.ndarray:
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    out = np.zeros((stack.shape[0], stack.shape[1] * factor, stack.shape[2] * factor), dtype=np.float32)
    for i in range(stack.shape[0]):
        img = stack[i]
        pad_h = img.shape[0] / 2.0
        pad_w = img.shape[1] / 2.0
        pad_top = int(np.ceil(pad_h))
        pad_left = int(np.ceil(pad_w))
        pad_bottom = int(np.floor(pad_h))
        pad_right = int(np.floor(pad_w))
        padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="symmetric")

        interp = f_interp_2d(padded, (factor * padded.shape[0] - (factor - 1), factor * padded.shape[1] - (factor - 1)))
        idx_h = int(np.ceil(img.shape[0] / 2.0) + 1 + (factor - 1) * np.floor(img.shape[0] / 2.0))
        idx_w = int(np.ceil(img.shape[1] / 2.0) + 1 + (factor - 1) * np.floor(img.shape[1] / 2.0))
        out[i] = interp[idx_h : idx_h + factor * img.shape[0], idx_w : idx_w + factor * img.shape[1]]

    if squeeze:
        return out[0]
    return out


def select_kernel_support(image_min_dim: int) -> int:
    if 17 < image_min_dim < 33:
        return 8
    if 33 < image_min_dim < 65:
        return 16
    if 65 < image_min_dim < 129:
        return 32
    if 129 < image_min_dim < 257:
        return 64
    return 64


def generate_physical_psf(pixel_m: float, wavelength_m: float, support: int, na: float, z_m: float = 0.0, num_samples: int = 1024) -> np.ndarray:
    sin2 = (1.0 - (1.0 - na**2)) / 2.0
    u = 8.0 * np.pi * z_m * sin2 / wavelength_m
    x = np.arange(-support * pixel_m, support * pixel_m + pixel_m, pixel_m, dtype=np.float64)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    radius = np.sqrt(xx**2 + yy**2)

    p = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
    phase = np.exp(0.5j * u * (p**2))
    arg = (2.0 * np.pi * na / wavelength_m) * radius[..., None] * p[None, None, :]
    integrand = 2.0 * phase[None, None, :] * special.j0(arg)
    ip = integrate.trapezoid(integrand, p, axis=-1)
    ipsf = np.abs(ip**2)
    ipsf /= np.sum(ipsf)
    return ipsf.astype(np.float32)


def build_physical_kernel(height: int, width: int, config: SparseSIMConfig) -> np.ndarray:
    image_min_dim = min(height, width)
    support = select_kernel_support(image_min_dim)
    pixel_m = config.pixel_size_nm * 1e-9
    wavelength_m = config.wavelength_nm * 1e-9

    if config.oversampling_method != "none":
        pixel_m = pixel_m / 2.0

    kernel = generate_physical_psf(
        pixel_m=pixel_m,
        wavelength_m=wavelength_m,
        support=support,
        na=config.effective_na,
        z_m=config.psf_z_m,
        num_samples=config.psf_integration_samples,
    )
    return kernel / np.sum(kernel)


def psf2otf(psf: np.ndarray, out_shape: tuple[int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    psf = xp.asarray(psf, dtype=xp.float32)
    pad_h = out_shape[0] - psf.shape[0]
    pad_w = out_shape[1] - psf.shape[1]
    padded = xp.pad(psf, ((0, pad_h), (0, pad_w)), mode="constant")
    for axis, size in enumerate(psf.shape):
        padded = xp.roll(padded, -int(size / 2), axis=axis)
    otf = fftmod.fftn(padded)
    return otf


def rliter(yk: Any, data: Any, otf: Any, backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    blurred = xp.maximum(fftmod.ifftn(otf * fftmod.fftn(yk)).real, xp.float32(1e-6))
    return fftmod.fftn(data / blurred)


def deblur_core(data: Any, kernel: np.ndarray, iteration: int, rule: int, backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    kernel = xp.asarray(kernel, dtype=xp.float32)
    kernel = kernel / xp.sum(kernel)
    dx, dy = data.shape
    border = int(np.floor(min(dx, dy) / 6))
    padded = xp.pad(data, ((border, border), (border, border)), mode="edge")

    yk = padded.astype(xp.float32)
    xk = xp.zeros_like(yk, dtype=xp.float32)
    vk = xp.zeros_like(yk, dtype=xp.float32)
    otf = psf2otf(kernel, yk.shape, backend)

    if rule == 2:
        t = 1.0
        gamma1 = 1.0
        for i in range(iteration):
            if i == 0:
                xk_prev = yk.copy()
                xk = yk + t * fftmod.ifftn(xp.conj(otf) * (fftmod.fftn(yk) - (otf * fftmod.fftn(yk)))).real
            else:
                gamma2 = 0.5 * np.sqrt(4 * gamma1 * gamma1 + gamma1**4) - gamma1**2
                beta = -gamma2 * (1.0 - 1.0 / gamma1)
                yk_update = xk + beta * (xk - xk_prev)
                yk = yk_update + t * fftmod.ifftn(xp.conj(otf) * (fftmod.fftn(padded) - (otf * fftmod.fftn(yk_update)))).real
                yk = xp.maximum(yk, xp.float32(1e-6)).astype(xp.float32)
                gamma1 = gamma2
                xk_prev = xk.copy()
                xk = yk
    else:
        for i in range(iteration):
            xk_prev = xk.copy()
            rl_part = rliter(yk, padded, otf, backend)
            denominator = fftmod.ifftn(fftmod.fftn(xp.ones_like(padded)) * otf).real
            xk = yk * fftmod.ifftn(xp.conj(otf) * rl_part).real / xp.maximum(denominator, xp.float32(1e-6))
            xk = xp.maximum(xk, xp.float32(1e-6)).astype(xp.float32)

            vk_prev = vk.copy()
            vk = xp.maximum(xk - yk, xp.float32(1e-6)).astype(xp.float32)
            if i == 0:
                yk = xp.maximum(xk, xp.float32(1e-6)).astype(xp.float32)
            else:
                alpha = xp.sum(vk_prev * vk) / (xp.sum(vk_prev * vk_prev) + xp.float32(1e-10))
                alpha = xp.maximum(xp.minimum(alpha, xp.float32(1.0)), xp.float32(1e-6))
                yk = xp.maximum(xk + alpha * (xk - xk_prev), xp.float32(1e-6)).astype(xp.float32)
                yk = xp.nan_to_num(yk, nan=xp.float32(1e-6), posinf=xp.float32(1e-6), neginf=xp.float32(1e-6))

    yk = xp.maximum(yk, xp.float32(0))
    return yk[border : yk.shape[0] - border, border : yk.shape[1] - border].astype(xp.float32)


def iterative_deblur(
    stack: Any,
    kernel: np.ndarray,
    iteration: int,
    rule: int,
    backend: dict[str, Any],
    show_progress: bool = True,
) -> np.ndarray:
    xp = backend["xp"]
    stack_cpu = np.asarray(to_cpu_array(stack, backend), dtype=np.float32)
    if stack_cpu.ndim == 2:
        deblurred = deblur_core(to_backend_array(stack_cpu, backend, dtype=xp.float32), kernel, iteration, rule, backend)
        result = to_cpu_array(deblurred, backend).astype(np.float32, copy=False)
        release_backend_memory(backend)
        return result

    output = np.zeros_like(stack_cpu, dtype=np.float32)
    for i in progress(range(stack_cpu.shape[0]), show_progress, desc="Deblur", leave=False):
        slice_backend = to_backend_array(stack_cpu[i], backend, dtype=xp.float32)
        output[i] = to_cpu_array(deblur_core(slice_backend, kernel, iteration, rule, backend), backend)
        release_backend_memory(backend)
    return output


def forward_diff(data: Any, step: float, dim: int, backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    r, n, m = data.shape
    size = [int(r), int(n), int(m)]
    position = [0, 0, 0]
    temp1 = xp.zeros((size[0] + 1, size[1] + 1, size[2] + 1), dtype=xp.float32)
    temp2 = xp.zeros((size[0] + 1, size[1] + 1, size[2] + 1), dtype=xp.float32)

    size[dim] += 1
    position[dim] += 1
    temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data

    size[dim] -= 1
    temp2[0 : size[0], 0 : size[1], 0 : size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] += 1

    out = temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]]
    return -out


def back_diff(data: Any, step: float, dim: int, backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    r, n, m = data.shape
    size = [int(r), int(n), int(m)]
    position = [0, 0, 0]
    temp1 = xp.zeros((size[0] + 1, size[1] + 1, size[2] + 1), dtype=xp.float32)
    temp2 = xp.zeros((size[0] + 1, size[1] + 1, size[2] + 1), dtype=xp.float32)

    temp1[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data

    size[dim] += 1
    position[dim] += 1
    temp2[position[0] : size[0], position[1] : size[1], position[2] : size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] -= 1
    return temp1[0 : size[0], 0 : size[1], 0 : size[2]]


def shrink(x: Any, lagrangian: float, backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    return xp.sign(x) * xp.maximum(xp.abs(x) - (1.0 / lagrangian), xp.float32(0.0))


def iter_xx(g: Any, bxx: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gxx = back_diff(forward_diff(g, 1, 1, backend), 1, 1, backend)
    dxx = shrink(gxx + bxx, mu, backend)
    bxx = bxx + (gxx - dxx)
    lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1, backend), 1, 1, backend)
    return lxx, bxx


def iter_xy(g: Any, bxy: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gxy = forward_diff(forward_diff(g, 1, 1, backend), 1, 2, backend)
    dxy = shrink(gxy + bxy, mu, backend)
    bxy = bxy + (gxy - dxy)
    lxy = para * back_diff(back_diff(dxy - bxy, 1, 2, backend), 1, 1, backend)
    return lxy, bxy


def iter_xz(g: Any, bxz: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gxz = forward_diff(forward_diff(g, 1, 1, backend), 1, 0, backend)
    dxz = shrink(gxz + bxz, mu, backend)
    bxz = bxz + (gxz - dxz)
    lxz = para * back_diff(back_diff(dxz - bxz, 1, 0, backend), 1, 1, backend)
    return lxz, bxz


def iter_yy(g: Any, byy: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gyy = back_diff(forward_diff(g, 1, 2, backend), 1, 2, backend)
    dyy = shrink(gyy + byy, mu, backend)
    byy = byy + (gyy - dyy)
    lyy = para * back_diff(forward_diff(dyy - byy, 1, 2, backend), 1, 2, backend)
    return lyy, byy


def iter_yz(g: Any, byz: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gyz = forward_diff(forward_diff(g, 1, 2, backend), 1, 0, backend)
    dyz = shrink(gyz + byz, mu, backend)
    byz = byz + (gyz - dyz)
    lyz = para * back_diff(back_diff(dyz - byz, 1, 0, backend), 1, 2, backend)
    return lyz, byz


def iter_zz(g: Any, bzz: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    gzz = back_diff(forward_diff(g, 1, 0, backend), 1, 0, backend)
    dzz = shrink(gzz + bzz, mu, backend)
    bzz = bzz + (gzz - dzz)
    lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0, backend), 1, 0, backend)
    return lzz, bzz


def iter_sparse(gsparse: Any, bsparse: Any, para: float, mu: float, backend: dict[str, Any]) -> tuple[Any, Any]:
    dsparse = shrink(gsparse + bsparse, mu, backend)
    bsparse = bsparse + (gsparse - dsparse)
    lsparse = para * (dsparse - bsparse)
    return lsparse, bsparse


def operation_xx(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_xx = xp.array([[[1, -2, 1]]], dtype=xp.float32)
    return fftmod.fftn(delta_xx, gsize) * xp.conj(fftmod.fftn(delta_xx, gsize))


def operation_xy(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_xy = xp.array([[[1, -1], [-1, 1]]], dtype=xp.float32)
    return fftmod.fftn(delta_xy, gsize) * xp.conj(fftmod.fftn(delta_xy, gsize))


def operation_xz(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_xz = xp.array([[[1, -1]], [[-1, 1]]], dtype=xp.float32)
    return fftmod.fftn(delta_xz, gsize) * xp.conj(fftmod.fftn(delta_xz, gsize))


def operation_yy(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_yy = xp.array([[[1], [-2], [1]]], dtype=xp.float32)
    return fftmod.fftn(delta_yy, gsize) * xp.conj(fftmod.fftn(delta_yy, gsize))


def operation_yz(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_yz = xp.array([[[1], [-1]], [[-1], [1]]], dtype=xp.float32)
    return fftmod.fftn(delta_yz, gsize) * xp.conj(fftmod.fftn(delta_yz, gsize))


def operation_zz(gsize: tuple[int, int, int], backend: dict[str, Any]) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    delta_zz = xp.array([[[1]], [[-2]], [[1]]], dtype=xp.float32)
    return fftmod.fftn(delta_zz, gsize) * xp.conj(fftmod.fftn(delta_zz, gsize))


def sparse_hessian(
    stack: np.ndarray,
    iteration_num: int,
    fidelity: float,
    sparsity: float,
    contiz: float,
    backend: dict[str, Any],
    mu: float = 1.0,
    show_progress: bool = True,
) -> Any:
    xp = backend["xp"]
    fftmod = backend["fft"]
    contiz = float(np.sqrt(contiz))
    f1 = np.asarray(stack, dtype=np.float32)
    if f1.ndim == 2:
        contiz = 0.0
        f = np.repeat(f1[np.newaxis, ...], 3, axis=0)
        squeeze = True
    elif f1.shape[0] < 3:
        contiz = 0.0
        f = np.zeros((3, f1.shape[1], f1.shape[2]), dtype=np.float32)
        f[: f1.shape[0]] = f1
        for i in range(f1.shape[0], 3):
            f[i] = f[1]
        squeeze = False
    else:
        f = f1
        squeeze = False

    f = f / np.maximum(np.max(f), 1e-6)
    f = to_backend_array(f, backend, dtype=xp.float32)
    imgsize = f.shape
    xxfft = operation_xx(imgsize, backend)
    yyfft = operation_yy(imgsize, backend)
    zzfft = operation_zz(imgsize, backend)
    xyfft = operation_xy(imgsize, backend)
    xzfft = operation_xz(imgsize, backend)
    yzfft = operation_yz(imgsize, backend)

    operationfft = xxfft + yyfft + (contiz**2) * zzfft + 2 * xyfft + 2 * contiz * xzfft + 2 * contiz * yzfft
    normalize = (fidelity / mu) + (sparsity**2) + operationfft

    bxx = xp.zeros(imgsize, dtype=xp.float32)
    byy = xp.zeros(imgsize, dtype=xp.float32)
    bzz = xp.zeros(imgsize, dtype=xp.float32)
    bxy = xp.zeros(imgsize, dtype=xp.float32)
    bxz = xp.zeros(imgsize, dtype=xp.float32)
    byz = xp.zeros(imgsize, dtype=xp.float32)
    bl1 = xp.zeros(imgsize, dtype=xp.float32)

    g_update = (fidelity / mu) * f
    for iteration in progress(range(iteration_num), show_progress, desc="Sparse Hessian", leave=False):
        g_update_fft = fftmod.fftn(g_update)
        if iteration == 0:
            g = fftmod.ifftn(g_update_fft / (fidelity / mu)).real.astype(xp.float32)
        else:
            g = fftmod.ifftn(g_update_fft / normalize).real.astype(xp.float32)

        g_update = (fidelity / mu) * f
        lxx, bxx = iter_xx(g, bxx, 1, mu, backend)
        g_update += lxx
        lyy, byy = iter_yy(g, byy, 1, mu, backend)
        g_update += lyy
        lzz, bzz = iter_zz(g, bzz, contiz**2, mu, backend)
        g_update += lzz
        lxy, bxy = iter_xy(g, bxy, 2, mu, backend)
        g_update += lxy
        lxz, bxz = iter_xz(g, bxz, 2 * contiz, mu, backend)
        g_update += lxz
        lyz, byz = iter_yz(g, byz, 2 * contiz, mu, backend)
        g_update += lyz
        lsparse, bl1 = iter_sparse(g, bl1, sparsity, mu, backend)
        g_update += lsparse

    g = xp.maximum(g, xp.float32(0))
    if squeeze:
        return g[1]
    return g


def prepare_stack_for_sparse_hessian(
    stack: np.ndarray,
    debug_max_slices: int | None,
    allow_prefix_padding: bool = True,
) -> tuple[np.ndarray, int, bool]:
    stack = np.asarray(stack, dtype=np.float32)
    original_z = stack.shape[0]
    if original_z < 3:
        pad_count = 3 - original_z
        pad = np.repeat(stack[-1:, :, :], pad_count, axis=0)
        stack = np.concatenate([stack, pad], axis=0)

    if debug_max_slices is not None and original_z > debug_max_slices:
        stack = stack[:debug_max_slices]
        original_z = debug_max_slices

    applied_prefix_padding = False
    if allow_prefix_padding and debug_max_slices is None and stack.shape[0] > 3:
        padded = np.empty((stack.shape[0] + 2, stack.shape[1], stack.shape[2]), dtype=stack.dtype)
        padded[2:] = stack
        padded[1] = padded[3]
        padded[0] = padded[4]
        stack = padded
        applied_prefix_padding = True

    return stack, original_z, applied_prefix_padding


def restore_sparse_hessian_output(stack: np.ndarray, original_z: int, applied_prefix_padding: bool) -> np.ndarray:
    restored = stack
    if applied_prefix_padding and restored.shape[0] > 3:
        restored = restored[2:]
    return restored[:original_z]


def run_sparse_hessian_volume(
    sparse_input: np.ndarray,
    config: SparseSIMConfig,
    backend: dict[str, Any],
    allow_prefix_padding: bool,
    show_progress: bool,
) -> np.ndarray:
    prepared_input, original_z, applied_prefix_padding = prepare_stack_for_sparse_hessian(
        sparse_input,
        debug_max_slices=None,
        allow_prefix_padding=allow_prefix_padding,
    )
    sparse_output = sparse_hessian(
        prepared_input,
        iteration_num=config.sparse_iter,
        fidelity=config.fidelity,
        sparsity=config.sparsity,
        contiz=config.z_continuity,
        backend=backend,
        show_progress=show_progress,
    )
    restored = restore_sparse_hessian_output(sparse_output, original_z, applied_prefix_padding)
    restored_cpu = to_cpu_array(restored, backend).astype(np.float32, copy=False)
    release_backend_memory(backend)
    return restored_cpu


def compute_window_plan(z_size: int, window_size: int, halo: int) -> list[dict[str, int]]:
    if z_size <= 0:
        return []

    core_size = window_size - 2 * halo
    if core_size <= 0:
        raise ValueError("window_size must be greater than 2 * halo")

    plan: list[dict[str, int]] = []
    output_start = 0
    while output_start < z_size:
        output_end = min(output_start + core_size, z_size)

        context_start = max(0, output_start - halo)
        context_end = min(z_size, output_end + halo)
        target_size = min(window_size, z_size)
        current_size = context_end - context_start

        if current_size < target_size:
            shift_left = min(context_start, target_size - current_size)
            context_start -= shift_left
            current_size = context_end - context_start
            if current_size < target_size:
                context_end = min(z_size, context_start + target_size)
                context_start = max(0, context_end - target_size)

        plan.append(
            {
                "output_start": output_start,
                "output_end": output_end,
                "context_start": context_start,
                "context_end": context_end,
                "local_output_start": output_start - context_start,
                "local_output_end": output_end - context_start,
            }
        )
        output_start = output_end

    return plan


def sparse_hessian_windowed(
    sparse_input: np.ndarray,
    config: SparseSIMConfig,
    backend: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    plan = compute_window_plan(sparse_input.shape[0], config.window_size, config.halo)
    output = np.zeros_like(sparse_input, dtype=np.float32)

    iterator = progress(plan, config.show_progress, desc="Sparse Hessian Windows", leave=False)
    for window in iterator:
        context = sparse_input[window["context_start"] : window["context_end"]]
        local_result = run_sparse_hessian_volume(
            context,
            config,
            backend,
            allow_prefix_padding=(window["context_start"] == 0 and config.debug_max_slices is None),
            show_progress=False,
        )
        output[window["output_start"] : window["output_end"]] = local_result[
            window["local_output_start"] : window["local_output_end"]
        ]

    return output, {"window_plan": plan}


def run_sparse_sim(stack_zyx: np.ndarray, config: SparseSIMConfig) -> tuple[np.ndarray, dict[str, Any]]:
    config.validate()
    stack = np.asarray(stack_zyx, dtype=np.float32)

    original_max = float(np.max(stack))
    if original_max <= 0:
        raise ValueError("Input image max intensity must be > 0")

    normalized = stack / original_max
    if normalized.shape[0] < 1:
        raise ValueError("Input stack must contain at least one slice")
    if config.debug_max_slices is not None and normalized.shape[0] > config.debug_max_slices:
        normalized = normalized[: config.debug_max_slices]

    background_corrected, background = apply_background_mode(normalized, config.background_mode, show_progress=config.show_progress)
    background_corrected[background_corrected < 0] = 0
    background_corrected = background_corrected / np.maximum(np.max(background_corrected), 1e-6)

    if config.oversampling_method == "spatial":
        sparse_input = spatial_upsample(background_corrected)
    elif config.oversampling_method == "fourier":
        sparse_input = fourier_upsample(background_corrected)
    else:
        sparse_input = background_corrected

    effective_backend_name = config.backend
    if config.mode == "exact_cpu_full":
        effective_backend_name = "cpu"

    backend = resolve_backend(effective_backend_name, config.gpu_device_index)
    window_meta: dict[str, Any] | None = None

    if config.mode == "windowed_gpu":
        sparse_output, window_meta = sparse_hessian_windowed(sparse_input, config, backend)
    else:
        sparse_output = run_sparse_hessian_volume(
            sparse_input,
            config,
            backend,
            allow_prefix_padding=(config.debug_max_slices is None),
            show_progress=config.show_progress,
        )

    sparse_output = np.asarray(sparse_output, dtype=np.float32)
    sparse_output = sparse_output / max(float(np.max(sparse_output)), 1e-6)

    deblur_rule = DEBLURRING_METHOD_MAP[config.deblurring_method]
    if deblur_rule != 3:
        kernel = build_physical_kernel(sparse_output.shape[1], sparse_output.shape[2], config)
        final_output = iterative_deblur(
            sparse_output,
            kernel,
            config.deconv_iter,
            deblur_rule,
            backend,
            show_progress=config.show_progress,
        )
        final_output = final_output / max(float(np.max(final_output)), 1e-6)
    else:
        kernel = None
        final_output = sparse_output

    final_output_cpu = np.asarray(final_output, dtype=np.float32)
    meta = {
        "input_shape_zyx": list(stack.shape),
        "processed_input_shape_zyx": list(normalized.shape),
        "background_shape_zyx": list(background.shape) if background is not None else None,
        "sparse_input_shape_zyx": list(sparse_input.shape),
        "output_shape_zyx": list(final_output_cpu.shape),
        "original_max_intensity": original_max,
        "kernel_shape": list(kernel.shape) if kernel is not None else None,
        "mode": config.mode,
        "backend": backend["name"],
        "gpu_device_index": backend["device_index"],
        "gpu_device_name": backend["device_name"],
        "window_size": config.window_size if config.mode == "windowed_gpu" else None,
        "halo": config.halo if config.mode == "windowed_gpu" else None,
        "window_count": len(window_meta["window_plan"]) if window_meta is not None else None,
        "window_plan": window_meta["window_plan"] if window_meta is not None else None,
        "config": asdict(config),
    }
    return final_output_cpu, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MATLAB-compatible Sparse-SIM step-1 runner for 3D TIFF stacks")
    parser.add_argument("--input", required=True, help="Path to input TIFF stack, expected in (Z, H, W).")
    parser.add_argument("--output", required=True, help="Path to output TIFF stack.")
    parser.add_argument("--params_json", help="Optional path to save the resolved parameter JSON.")
    parser.add_argument("--meta_json", help="Optional path to save run metadata JSON.")
    parser.add_argument("--pixel_size_nm", type=float, required=True, help="GUI Pixel size (nm).")
    parser.add_argument("--wavelength_nm", type=float, required=True, help="GUI Wave length (nm).")
    parser.add_argument("--effective_na", type=float, required=True, help="GUI Effective NA.")
    parser.add_argument("--sparse_iter", type=int, default=120, help="GUI Sparse iteration times.")
    parser.add_argument("--fidelity", type=float, default=150.0, help="GUI Image fidelity.")
    parser.add_argument("--z_continuity", type=float, default=1.0, help="GUI t(z) axial continuity.")
    parser.add_argument("--sparsity", type=float, default=6.0, help="GUI Sparsity.")
    parser.add_argument("--deconv_iter", type=int, default=8, help="GUI Iterative deblur times.")
    parser.add_argument(
        "--background_mode",
        choices=sorted(BACKGROUND_MODE_MAP.keys()),
        default="none",
        help="Background mode aligned to MATLAB GUI semantics.",
    )
    parser.add_argument(
        "--deblurring_method",
        choices=sorted(DEBLURRING_METHOD_MAP.keys()),
        default="lucy_richardson",
        help="Deblurring method aligned to MATLAB GUI semantics.",
    )
    parser.add_argument(
        "--oversampling_method",
        choices=sorted(OVERSAMPLING_METHOD_MAP.keys()),
        default="none",
        help="Oversampling method aligned to MATLAB GUI semantics.",
    )
    parser.add_argument("--mode", choices=sorted(MODE_CHOICES), default="exact_cpu_full", help="Execution mode for Sparse Hessian reconstruction.")
    parser.add_argument("--window_size", type=int, default=32, help="Z-window size used in windowed_gpu mode.")
    parser.add_argument("--halo", type=int, default=4, help="Z halo size on each side used in windowed_gpu mode.")
    parser.add_argument("--debug_max_slices", type=int, help="Optional debug cap on the number of slices.")
    parser.add_argument("--psf_integration_samples", type=int, default=1024, help="Integration samples used for physical PSF generation.")
    parser.add_argument("--backend", choices=sorted(BACKEND_CHOICES), default="cpu", help="Execution backend. 'cuda' requires CuPy.")
    parser.add_argument("--gpu_device_index", type=int, default=0, help="Zero-based CUDA device index used when backend is 'cuda' or 'auto'.")
    parser.add_argument("--hide_progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    stack, original_dtype = read_tiff_stack(input_path)
    config = SparseSIMConfig(
        pixel_size_nm=args.pixel_size_nm,
        wavelength_nm=args.wavelength_nm,
        effective_na=args.effective_na,
        sparse_iter=args.sparse_iter,
        fidelity=args.fidelity,
        z_continuity=args.z_continuity,
        sparsity=args.sparsity,
        deconv_iter=args.deconv_iter,
        background_mode=args.background_mode,
        deblurring_method=args.deblurring_method,
        oversampling_method=args.oversampling_method,
        mode=args.mode,
        window_size=args.window_size,
        halo=args.halo,
        debug_max_slices=args.debug_max_slices,
        psf_integration_samples=args.psf_integration_samples,
        backend=args.backend,
        gpu_device_index=args.gpu_device_index,
        show_progress=not args.hide_progress,
    )

    output_float, meta = run_sparse_sim(stack, config)
    output_to_save = convert_for_saving(output_float, original_dtype, meta["original_max_intensity"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, output_to_save)

    if args.params_json:
        save_json(Path(args.params_json).expanduser().resolve(), asdict(config))
    if args.meta_json:
        meta["output_dtype"] = str(output_to_save.dtype)
        save_json(Path(args.meta_json).expanduser().resolve(), meta)


if __name__ == "__main__":
    main()
