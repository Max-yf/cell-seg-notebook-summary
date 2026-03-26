from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import tifffile

from sparse_sim_matlab import SparseSIMConfig, convert_for_saving, read_tiff_stack, run_sparse_sim, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step-1 sparse deconvolution using a JSON config file.")
    parser.add_argument("--input", required=True, help="Path to input TIFF stack, expected in (Z, H, W).")
    parser.add_argument("--output", required=True, help="Path to output TIFF stack.")
    parser.add_argument("--config_json", required=True, help="Path to Sparse-SIM JSON config.")
    parser.add_argument("--params_json", help="Optional path to save the resolved parameter JSON.")
    parser.add_argument("--meta_json", help="Optional path to save run metadata JSON.")
    parser.add_argument("--mode", choices=["exact_cpu_full", "windowed_gpu"], help="Optional override for the execution mode.")
    parser.add_argument("--window_size", type=int, help="Optional override for z-window size in windowed_gpu mode.")
    parser.add_argument("--halo", type=int, help="Optional override for z-halo size in windowed_gpu mode.")
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], help="Optional override for the execution backend.")
    parser.add_argument("--gpu_device_index", type=int, help="Optional override for the zero-based CUDA device index.")
    parser.add_argument("--hide_progress", action="store_true", help="Disable tqdm progress bars for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    config_path = Path(args.config_json).expanduser().resolve()

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode is not None:
        config_data["mode"] = args.mode
    if args.window_size is not None:
        config_data["window_size"] = args.window_size
    if args.halo is not None:
        config_data["halo"] = args.halo
    if args.backend is not None:
        config_data["backend"] = args.backend
    if args.gpu_device_index is not None:
        config_data["gpu_device_index"] = args.gpu_device_index
    if args.hide_progress:
        config_data["show_progress"] = False
    config = SparseSIMConfig(**config_data)

    stack, original_dtype = read_tiff_stack(input_path)
    output_float, meta = run_sparse_sim(stack, config)
    output_to_save = convert_for_saving(output_float, original_dtype, meta["original_max_intensity"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, output_to_save)

    if args.params_json:
        save_json(Path(args.params_json).expanduser().resolve(), asdict(config))
    if args.meta_json:
        meta["output_dtype"] = str(output_to_save.dtype)
        meta["config_json"] = str(config_path)
        save_json(Path(args.meta_json).expanduser().resolve(), meta)


if __name__ == "__main__":
    main()
