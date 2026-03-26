#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_infer_3d.py

用途：
    使用微调后的 Cellpose-SAM 模型对单个 3D tif/tiff stack 进行分割，
    输出 mask.tif、meta.json、params.json、run.log，方便后续排错与追踪。

当前交付版默认参数：
    CELLPROB_THRESHOLD = 0
    MIN_SIZE = 50
    ANISOTROPY = 1
    DIAMETER = None
    RESCALE = 1
    DO_3D = True
    Z_AXIS = 0
    BATCH_SIZE_3D = 4
    STITCH_THRESHOLD = 0.0
    FLOW_THRESHOLD = 0.4

这些默认值的设计目标是：尽量保证识别出的细胞形状清晰、结果稳定，方便后续分析。

参数说明：参见README.md

输入要求：
    - 只支持单个 3D .tif / .tiff 文件
    - 默认数据维度顺序为 (Z, H, W)
    - 因此默认 z_axis = 0

示例：
    python run_infer_3d.py \
        --input /path/to/input.tif \
        --output /path/to/output_dir \
        --model ./model/finetuned_cellpose_sam_model \
        --config ./model/model_config.json \
        --use_gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile as tiff
from cellpose import io, models


# =========================
# 默认参数（交付版）
# =========================
DEFAULT_CELLPROB_THRESHOLD = 0.0
DEFAULT_MIN_SIZE = 50
DEFAULT_ANISOTROPY = 1.0
DEFAULT_DIAMETER = None
DEFAULT_RESCALE = 1.0
DEFAULT_DO_3D = True
DEFAULT_Z_AXIS = 0
DEFAULT_BATCH_SIZE_3D = 4
DEFAULT_BSIZE = 256
DEFAULT_STITCH_THRESHOLD = 0.0
DEFAULT_FLOW_THRESHOLD = 0.4
DEFAULT_TILE_OVERLAP = 0.1
DEFAULT_AUGMENT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use finetuned Cellpose-SAM model for single-stack 3D inference."
    )

    # 基本输入输出
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single 3D .tif/.tiff stack.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory. mask.tif / meta.json / params.json / run.log will be saved here.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./model/finetuned_cellpose_sam_model",
        help="Path to finetuned Cellpose-SAM model file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./model/model_config.json",
        help="Path to model config json file. Used for record/tracing only.",
    )

    # 当前交付版默认参数
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=DEFAULT_CELLPROB_THRESHOLD,
        help="Default = 0. Lower is looser, higher is stricter.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help="Default = 50. Remove very small objects.",
    )
    parser.add_argument(
        "--anisotropy",
        type=float,
        default=DEFAULT_ANISOTROPY,
        help="Default = 1. Do not change casually.",
    )
    parser.add_argument(
        "--diameter",
        type=str,
        default="None",
        help="Default = None. Keep None to avoid triggering internal scaling behavior.",
    )
    parser.add_argument(
        "--rescale",
        type=float,
        default=DEFAULT_RESCALE,
        help="Default = 1. Only used when diameter=None.",
    )
    parser.add_argument(
        "--do_3D",
        action="store_true",
        default=DEFAULT_DO_3D,
        help="Use Cellpose 3D mode. Default is enabled.",
    )
    parser.add_argument(
        "--z_axis",
        type=int,
        default=DEFAULT_Z_AXIS,
        help="Default = 0, meaning input shape should be (Z, H, W).",
    )
    parser.add_argument(
        "--batch_size_3d",
        type=int,
        default=DEFAULT_BATCH_SIZE_3D,
        help="Default = 4. Lower this first if memory is not enough.",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=DEFAULT_BSIZE,
        help="Default = 256. Tile/block size used internally by Cellpose.",
    )
    parser.add_argument(
        "--stitch_threshold",
        type=float,
        default=DEFAULT_STITCH_THRESHOLD,
        help="Default = 0.0. Keep default unless you know exactly why to change it.",
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=DEFAULT_FLOW_THRESHOLD,
        help="Default = 0.4. In current a=1 setup this is usually not the first parameter to tune.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=float,
        default=DEFAULT_TILE_OVERLAP,
        help="Default = 0.1. Fraction of overlap between internal Cellpose tiles.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=DEFAULT_AUGMENT,
        help="Enable Cellpose tile augmentation and averaging.",
    )

    # 其它选项
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    parser.add_argument(
        "--save_flows",
        action="store_true",
        help="Whether to save raw flow outputs as flows.npy. Default is off.",
    )

    return parser.parse_args()


def parse_diameter_arg(diameter_str: str) -> float | None:
    """
    支持：
        --diameter None
        --diameter none
        --diameter 8
        --diameter 8.0
    """
    if diameter_str is None:
        return None

    s = str(diameter_str).strip().lower()
    if s in {"none", "null"}:
        return None

    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --diameter value: {diameter_str!r}. Use a number or 'None'."
        ) from exc


def ensure_input_suffix_ok(input_path: Path) -> None:
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(
            f"Only single 3D .tif/.tiff is supported, but got: {input_path.name}"
        )


def ensure_input_is_3d(img: np.ndarray) -> None:
    if img.ndim != 3:
        raise ValueError(
            f"Expected a single 3D stack with shape (Z, H, W), "
            f"but got shape={img.shape} (ndim={img.ndim})."
        )


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_infer_3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件日志
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 终端日志
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def count_instances(masks: np.ndarray) -> int:
    if masks.size == 0:
        return 0
    return int(masks.max())


def main() -> None:
    args = parse_args()

    start_time = time.time()
    start_time_str = now_str()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    logger = setup_logger(log_path)

    # 解析 diameter
    diameter_value = parse_diameter_arg(args.diameter)

    # 实际参数字典：后面写进 params.json
    actual_params: dict[str, Any] = {
        "cellprob_threshold": float(args.cellprob_threshold),
        "min_size": int(args.min_size),
        "anisotropy": float(args.anisotropy),
        "diameter": diameter_value,
        "rescale": float(args.rescale),
        "do_3D": bool(args.do_3D),
        "z_axis": int(args.z_axis),
        "batch_size_3d": int(args.batch_size_3d),
        "bsize": int(args.bsize),
        "stitch_threshold": float(args.stitch_threshold),
        "flow_threshold": float(args.flow_threshold),
        "tile_overlap": float(args.tile_overlap),
        "augment": bool(args.augment),
        "use_gpu": bool(args.use_gpu),
        "save_flows": bool(args.save_flows),
    }

    # 先把 params.json 存下来，后面即使报错也能知道当时用了什么参数
    save_json(output_dir / "params.json", actual_params)

    meta: dict[str, Any] = {
        "task": "3d_inference",
        "success": False,
        "start_time": start_time_str,
        "end_time": None,
        "elapsed_seconds": None,
        "input_image": str(input_path),
        "input_suffix": input_path.suffix.lower(),
        "image_shape": None,
        "image_dtype": None,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "output_mask": str(output_dir / "mask.tif"),
        "output_params": str(output_dir / "params.json"),
        "output_log": str(log_path),
        "output_flows": str(output_dir / "flows.npy") if args.save_flows else None,
        "num_instances": None,
        "error_type": None,
        "error_message": None,
    }

    try:
        logger.info("=" * 90)
        logger.info("3D inference started")
        logger.info("=" * 90)
        logger.info(f"Input image       : {input_path}")
        logger.info(f"Output dir        : {output_dir}")
        logger.info(f"Model path        : {model_path}")
        logger.info(f"Config path       : {config_path}")
        logger.info("Actual parameters :")
        for k, v in actual_params.items():
            logger.info(f"  - {k}: {v}")

        # 基础检查
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        ensure_input_suffix_ok(input_path)

        # Redirect Cellpose's own log files into the run output folder so the
        # pipeline does not depend on write access to the user home directory.
        io.logger_setup(cp_path=str(output_dir / ".cellpose"))

        # 读取图像
        logger.info("Loading input image ...")
        img = io.imread(str(input_path))
        img = np.asarray(img)
        ensure_input_is_3d(img)

        meta["image_shape"] = list(img.shape)
        meta["image_dtype"] = str(img.dtype)

        logger.info(f"Image shape       : {img.shape}")
        logger.info(f"Image dtype       : {img.dtype}")

        # 加载模型
        logger.info("Loading model ...")
        model = models.CellposeModel(
            gpu=args.use_gpu,
            pretrained_model=str(model_path),
        )

        # 推理
        logger.info("Running model.eval(...) ...")
        masks, flows, styles = model.eval(
            img,
            channels=[0, 0],          # 单通道灰度图
            channel_axis=None,
            z_axis=args.z_axis,
            do_3D=args.do_3D,
            diameter=diameter_value,
            rescale=args.rescale,
            anisotropy=args.anisotropy,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            stitch_threshold=args.stitch_threshold,
            min_size=args.min_size,
            batch_size=args.batch_size_3d,
            bsize=args.bsize,
            tile_overlap=args.tile_overlap,
            augment=args.augment,
        )

        # 保存 mask
        mask_path = output_dir / "mask.tif"
        logger.info("Saving mask.tif ...")
        tiff.imwrite(mask_path, masks.astype(np.uint16))

        # 可选保存 flows
        if args.save_flows:
            flow_path = output_dir / "flows.npy"
            logger.info("Saving flows.npy ...")
            np.save(flow_path, flows, allow_pickle=True)

        # 统计实例数
        num_instances = count_instances(masks)
        meta["num_instances"] = num_instances
        meta["success"] = True

        end_time = time.time()
        meta["end_time"] = now_str()
        meta["elapsed_seconds"] = round(end_time - start_time, 4)

        # 保存 meta
        save_json(output_dir / "meta.json", meta)

        logger.info("=" * 90)
        logger.info("3D inference finished successfully")
        logger.info(f"Saved mask        : {mask_path}")
        logger.info(f"Saved meta        : {output_dir / 'meta.json'}")
        logger.info(f"Saved params      : {output_dir / 'params.json'}")
        logger.info(f"Saved log         : {log_path}")
        if args.save_flows:
            logger.info(f"Saved flows       : {output_dir / 'flows.npy'}")
        logger.info(f"Num instances     : {num_instances}")
        logger.info(f"Elapsed seconds   : {meta['elapsed_seconds']}")
        logger.info("=" * 90)

    except Exception as exc:
        end_time = time.time()
        meta["success"] = False
        meta["end_time"] = now_str()
        meta["elapsed_seconds"] = round(end_time - start_time, 4)
        meta["error_type"] = type(exc).__name__
        meta["error_message"] = str(exc)

        save_json(output_dir / "meta.json", meta)

        logger.error("=" * 90)
        logger.error("3D inference failed")
        logger.error(f"Error type        : {type(exc).__name__}")
        logger.error(f"Error message     : {exc}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"meta.json written : {output_dir / 'meta.json'}")
        logger.error(f"params.json written: {output_dir / 'params.json'}")
        logger.error("=" * 90)
        raise


if __name__ == "__main__":
    main()
