from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from local_normalization import process_stack
from sparse_sim_matlab import SparseSIMConfig, convert_for_saving, run_sparse_sim, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step 1 Sparse-SIM and step 2 local normalization in sequence.")
    parser.add_argument("--input", required=True, help="Path to raw TIFF input.")
    parser.add_argument("--output_dir", required=True, help="Directory for all step-1 and step-2 outputs.")
    parser.add_argument("--config_json", required=True, help="Path to Sparse-SIM step-1 JSON config.")
    parser.add_argument("--channel_index", type=int, default=0, help="Channel index to use when the input TIFF contains channels.")
    parser.add_argument("--save_extracted_input", action="store_true", help="Save the extracted step-1 input stack as TIFF.")
    parser.add_argument("--mode", choices=["exact_cpu_full", "windowed_gpu"], help="Optional override for step-1 execution mode.")
    parser.add_argument("--window_size", type=int, help="Optional override for step-1 z-window size.")
    parser.add_argument("--halo", type=int, help="Optional override for step-1 z-halo size.")
    parser.add_argument("--sparsity", type=float, help="Optional override for step-1 sparsity.")
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], help="Optional override for step-1 execution backend.")
    parser.add_argument("--gpu_device_index", type=int, help="Optional override for step-1 zero-based CUDA device index.")
    parser.add_argument("--hide_progress", action="store_true", help="Disable tqdm progress bars in step-1.")
    parser.add_argument("--ln_radius", type=int, default=30, help="Local normalization window radius.")
    parser.add_argument("--ln_bias", type=float, default=5e-4, help="Local normalization bias term.")
    parser.add_argument("--ln_output_dtype", choices=["uint16", "float32"], default="uint16", help="Local normalization output dtype.")
    return parser.parse_args()


def load_stack(path: Path, channel_index: int) -> tuple[np.ndarray, str, str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = getattr(series, "axes", "")

    arr = np.asarray(arr)
    original_dtype = str(arr.dtype)

    if arr.ndim == 2:
        stack = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        stack = arr
    elif arr.ndim == 4:
        if "C" not in axes:
            raise ValueError(f"4D input is ambiguous without a channel axis: axes={axes}, shape={arr.shape}")
        c_axis = axes.index("C")
        if not 0 <= channel_index < arr.shape[c_axis]:
            raise ValueError(f"channel_index={channel_index} is out of range for shape={arr.shape} and axes={axes}")
        stack = np.take(arr, indices=channel_index, axis=c_axis)
        remaining_axes = "".join(ax for ax in axes if ax != "C")
        if remaining_axes != "ZYX":
            raise ValueError(f"Expected remaining axes to be ZYX after channel selection, but got {remaining_axes}")
    else:
        raise ValueError(f"Unsupported input shape={arr.shape}, axes={axes}")

    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D ZYX stack after extraction, but got shape={stack.shape}")

    return np.asarray(stack), original_dtype, axes


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config_json).expanduser().resolve()

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode is not None:
        config_data["mode"] = args.mode
    if args.window_size is not None:
        config_data["window_size"] = args.window_size
    if args.halo is not None:
        config_data["halo"] = args.halo
    if args.sparsity is not None:
        config_data["sparsity"] = args.sparsity
    if args.backend is not None:
        config_data["backend"] = args.backend
    if args.gpu_device_index is not None:
        config_data["gpu_device_index"] = args.gpu_device_index
    if args.hide_progress:
        config_data["show_progress"] = False
    config = SparseSIMConfig(**config_data)

    stack, original_dtype, input_axes = load_stack(input_path, args.channel_index)

    if args.save_extracted_input:
        tifffile.imwrite(output_dir / "step1_input_extracted.tif", stack)

    step1_float, step1_meta = run_sparse_sim(stack, config)
    step1_saved = convert_for_saving(step1_float, original_dtype, step1_meta["original_max_intensity"])
    step1_path = output_dir / "step1_sparse_sim.tif"
    tifffile.imwrite(step1_path, step1_saved)

    step2_array = process_stack(
        step1_saved,
        radius=args.ln_radius,
        bias=args.ln_bias,
        output_dtype=args.ln_output_dtype,
    )
    step2_path = output_dir / "step2_local_normalization.tif"
    tifffile.imwrite(step2_path, step2_array)

    pipeline_meta: dict[str, Any] = {
        "input_file": str(input_path),
        "input_axes": input_axes,
        "selected_channel_index": args.channel_index,
        "extracted_shape_zyx": list(stack.shape),
        "step1_output": str(step1_path),
        "step2_output": str(step2_path),
        "step1_meta": step1_meta,
        "step2": {
            "radius": args.ln_radius,
            "bias": args.ln_bias,
            "output_dtype": args.ln_output_dtype,
            "output_shape_zyx": list(step2_array.shape),
            "output_dtype_actual": str(step2_array.dtype),
        },
    }

    save_json(output_dir / "step1_params.json", asdict(config))
    save_json(output_dir / "step12_meta.json", pipeline_meta)


if __name__ == "__main__":
    main()
