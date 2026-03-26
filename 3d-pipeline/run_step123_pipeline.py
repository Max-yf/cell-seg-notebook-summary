from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import tifffile

from run_step12_pipeline import main as run_step12_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step 1, step 2, and step 3 in sequence.")
    parser.add_argument("--input", required=True, help="Path to raw TIFF input.")
    parser.add_argument("--output_dir", required=True, help="Directory for all outputs.")
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
    parser.add_argument("--cellpose_model", default=None, help="Path to the finetuned Cellpose-SAM model.")
    parser.add_argument("--cellpose_config", default=None, help="Path to the Cellpose-SAM config JSON.")
    parser.add_argument("--use_gpu_step3", action="store_true", help="Use GPU for step 3 if available.")
    parser.add_argument("--step3_batch_size_3d", type=int, default=4, help="Cellpose step-3 batch size.")
    parser.add_argument("--step3_bsize", type=int, default=256, help="Cellpose step-3 tile/block size.")
    parser.add_argument("--step3_tile_overlap", type=float, default=0.1, help="Cellpose step-3 tile overlap fraction.")
    parser.add_argument("--step3_augment", action="store_true", help="Enable Cellpose tile augmentation in step 3.")
    parser.add_argument("--step3_gpu_device_index", type=int, default=None, help="Optional zero-based CUDA device index for step 3.")
    return parser.parse_args()


def resolve_default_cellpose_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    candidate_roots = [script_dir, script_dir.parent]
    for root in candidate_roots:
        model_path = root / "model" / "finetuned_cellpose_sam_model"
        config_path = root / "model" / "model_config.json"
        if model_path.exists() and config_path.exists():
            return model_path.resolve(), config_path.resolve()
    return (
        (script_dir / "model" / "finetuned_cellpose_sam_model").resolve(),
        (script_dir / "model" / "model_config.json").resolve(),
    )


def run_step12(args: argparse.Namespace) -> None:
    argv = [
        "--input",
        args.input,
        "--output_dir",
        args.output_dir,
        "--config_json",
        args.config_json,
        "--channel_index",
        str(args.channel_index),
        "--ln_radius",
        str(args.ln_radius),
        "--ln_bias",
        str(args.ln_bias),
        "--ln_output_dtype",
        args.ln_output_dtype,
    ]
    if args.save_extracted_input:
        argv.append("--save_extracted_input")
    if args.mode is not None:
        argv.extend(["--mode", args.mode])
    if args.window_size is not None:
        argv.extend(["--window_size", str(args.window_size)])
    if args.halo is not None:
        argv.extend(["--halo", str(args.halo)])
    if args.sparsity is not None:
        argv.extend(["--sparsity", str(args.sparsity)])
    if args.backend is not None:
        argv.extend(["--backend", args.backend])
    if args.gpu_device_index is not None:
        argv.extend(["--gpu_device_index", str(args.gpu_device_index)])
    if args.hide_progress:
        argv.append("--hide_progress")

    from run_step12_pipeline import main as run_step12_pipeline_main

    sys.argv = ["run_step12_pipeline.py", *argv]
    run_step12_pipeline_main()


def run_step3(
    step2_path: Path,
    output_dir: Path,
    model_path: Path,
    config_path: Path,
    use_gpu: bool,
    batch_size_3d: int,
    bsize: int,
    tile_overlap: float,
    augment: bool,
    gpu_device_index: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_infer_3d.py"),
        "--input",
        str(step2_path),
        "--output",
        str(output_dir),
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--batch_size_3d",
        str(batch_size_3d),
        "--bsize",
        str(bsize),
        "--tile_overlap",
        str(tile_overlap),
    ]
    if use_gpu:
        cmd.append("--use_gpu")
    if augment:
        cmd.append("--augment")

    env = None
    if gpu_device_index is not None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device_index)

    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step12_output_dir = output_dir / "step12"
    step12_output_dir.mkdir(parents=True, exist_ok=True)

    step3_output_dir = output_dir / "step3_cellpose"
    step3_output_dir.mkdir(parents=True, exist_ok=True)

    # Run step 1 + 2 into a nested folder first.
    step12_args = argparse.Namespace(**vars(args))
    step12_args.output_dir = str(step12_output_dir)
    run_step12(step12_args)

    step2_path = step12_output_dir / "step2_local_normalization.tif"
    if not step2_path.exists():
        raise FileNotFoundError(f"Step 2 output not found: {step2_path}")

    default_model_path, default_config_path = resolve_default_cellpose_paths()
    model_path = Path(args.cellpose_model).resolve() if args.cellpose_model else default_model_path
    config_path = Path(args.cellpose_config).resolve() if args.cellpose_config else default_config_path

    run_step3(
        step2_path,
        step3_output_dir,
        model_path,
        config_path,
        args.use_gpu_step3,
        args.step3_batch_size_3d,
        args.step3_bsize,
        args.step3_tile_overlap,
        args.step3_augment,
        args.step3_gpu_device_index,
    )

    summary = {
        "step12_output_dir": str(step12_output_dir),
        "step3_output_dir": str(step3_output_dir),
        "step2_input_to_step3": str(step2_path),
        "cellpose_model": str(model_path),
        "cellpose_config": str(config_path),
        "use_gpu_step3": bool(args.use_gpu_step3),
        "step3_batch_size_3d": int(args.step3_batch_size_3d),
        "step3_bsize": int(args.step3_bsize),
        "step3_tile_overlap": float(args.step3_tile_overlap),
        "step3_augment": bool(args.step3_augment),
        "step3_gpu_device_index": args.step3_gpu_device_index,
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
