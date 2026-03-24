#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 3.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = None
RESCALE = 1.75
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[5]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_cp3p5_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp3p5__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp3p5__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[7]:


# -*- coding: utf-8 -*-
"""
make_manual_overlay_3d_tif.py

作用：
- 读取 3D 原图 tif
- 读取 3D mask tif（label mask, 0=background, >0=instance）
- 生成用于人工巡检的 3D overlay tif
- overlay 采用“半透明烘焙”方式：mask 不会挡死原图背景

输出：
- overlay_fill_3d.tif      : 半透明填充版（适合整体看覆盖）
- overlay_boundary_3d.tif  : 边界高亮版（适合看边界贴不贴）
- meta.json                : 记录输入输出和参数

说明：
- 输出 tif 为 RGB 3D stack，shape = (Z, H, W, 3), dtype=uint8
- 推荐用 Fiji / ImageJ 打开检查
"""

from pathlib import Path
import json
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "manual_overlay_3d_tif"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 要处理的 mask 列表
# 你后面要加别的结果，往这里继续 append 就行
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_cp2p5_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp2p5__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp2p5__ms50__full_brain_3d_masks_raw.tif",
    },
]

# ---------------------------
# 显示参数
# ---------------------------

# 原图显示拉伸的百分位
RAW_NORM_PMIN = 1.0
RAW_NORM_PMAX = 99.0

# 填充版 overlay 的透明度（越大 mask 越显眼）
# 建议 0.18 ~ 0.35
FILL_ALPHA = 0.25

# 边界版 overlay 的透明度
BOUNDARY_ALPHA = 0.95

# 边界粗细：通过多次膨胀实现
BOUNDARY_DILATE_ITERS = 1

# 颜色：红色
MASK_COLOR_RGB = np.array([255, 0, 0], dtype=np.float32)

# 是否同时输出边界版
SAVE_BOUNDARY_VERSION = True

# tif 压缩
TIFF_COMPRESSION = "zlib"

# 是否保存一个只包含“有 mask 的切片索引”的 json
SAVE_MASKED_Z_LIST = True


# =========================================================
# 1) 基础工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def robust_normalize_to_uint8(
    vol: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.0
) -> np.ndarray:
    """
    把 3D 原图归一化到 uint8 [0,255]
    采用全局百分位拉伸，避免每层亮度风格乱跳
    """
    x = np.asarray(vol, dtype=np.float32)

    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)

    if not np.isfinite(lo):
        lo = float(np.min(x))
    if not np.isfinite(hi):
        hi = float(np.max(x))

    if hi <= lo:
        # 兜底
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
        return x

    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def gray3_to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    """
    输入: (Z,H,W) uint8
    输出: (Z,H,W,3) uint8
    """
    return np.repeat(gray_u8[..., None], 3, axis=-1)


def alpha_blend_inplace(
    base_rgb_u8: np.ndarray,
    mask_bool: np.ndarray,
    color_rgb: np.ndarray,
    alpha: float,
):
    """
    在 base_rgb_u8 上原地做 alpha blend

    base_rgb_u8: (H, W, 3) uint8
    mask_bool  : (H, W) bool
    color_rgb  : (3,)
    alpha      : 0~1
    """
    if alpha <= 0:
        return
    if not np.any(mask_bool):
        return

    base_rgb_u8[mask_bool] = (
        (1.0 - alpha) * base_rgb_u8[mask_bool].astype(np.float32)
        + alpha * color_rgb[None, :].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def get_boundary(mask2d: np.ndarray, dilate_iters: int = 1) -> np.ndarray:
    """
    从二值 mask 提取边界。
    """
    m = mask2d.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)

    eroded = ndi.binary_erosion(m)
    boundary = m & (~eroded)

    if dilate_iters > 0:
        boundary = ndi.binary_dilation(boundary, iterations=dilate_iters)

    return boundary


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 2) 核心：生成 overlay stack
# =========================================================
def build_fill_overlay_stack(
    raw_u8: np.ndarray,
    mask_label: np.ndarray,
    alpha: float = 0.25,
    color_rgb: np.ndarray = np.array([255, 0, 0], dtype=np.float32),
) -> np.ndarray:
    """
    生成半透明填充版 overlay
    输入:
      raw_u8    : (Z,H,W), uint8
      mask_label: (Z,H,W), 整数标签图，>0 视为前景
    输出:
      overlay_rgb: (Z,H,W,3), uint8
    """
    zdim = raw_u8.shape[0]
    overlay_rgb = gray3_to_rgb(raw_u8)

    for z in range(zdim):
        fg = mask_label[z] > 0
        if np.any(fg):
            alpha_blend_inplace(
                overlay_rgb[z],
                fg,
                color_rgb=color_rgb,
                alpha=alpha,
            )
        if (z + 1) % 20 == 0 or z == zdim - 1:
            print(f"[fill overlay] done {z+1}/{zdim}")

    return overlay_rgb


def build_boundary_overlay_stack(
    raw_u8: np.ndarray,
    mask_label: np.ndarray,
    alpha: float = 0.95,
    color_rgb: np.ndarray = np.array([255, 0, 0], dtype=np.float32),
    dilate_iters: int = 1,
) -> np.ndarray:
    """
    生成边界高亮版 overlay
    输入:
      raw_u8    : (Z,H,W), uint8
      mask_label: (Z,H,W), 整数标签图，>0 视为前景
    输出:
      overlay_rgb: (Z,H,W,3), uint8
    """
    zdim = raw_u8.shape[0]
    overlay_rgb = gray3_to_rgb(raw_u8)

    for z in range(zdim):
        fg = mask_label[z] > 0
        bd = get_boundary(fg, dilate_iters=dilate_iters)
        if np.any(bd):
            alpha_blend_inplace(
                overlay_rgb[z],
                bd,
                color_rgb=color_rgb,
                alpha=alpha,
            )
        if (z + 1) % 20 == 0 or z == zdim - 1:
            print(f"[boundary overlay] done {z+1}/{zdim}")

    return overlay_rgb


# =========================================================
# 3) 单个 mask 的处理流程
# =========================================================
def process_one_mask_item(
    raw_path: Path,
    item: dict,
    out_root: Path,
):
    tag = item["tag"]
    mask_path = Path(item["mask_tif_path"]).resolve()

    print("=" * 100)
    print(f"开始处理: {tag}")
    print("=" * 100)
    print(f"RAW : {raw_path}")
    print(f"MASK: {mask_path}")

    if not raw_path.exists():
        raise FileNotFoundError(f"raw tif 不存在: {raw_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask tif 不存在: {mask_path}")

    tag_out_dir = out_root / tag
    ensure_dir(tag_out_dir)

    # 读取数据
    raw = tiff.imread(raw_path)
    mask = tiff.imread(mask_path)

    raw = np.asarray(raw)
    mask = np.asarray(mask)

    print(f"raw shape : {raw.shape}, dtype={raw.dtype}")
    print(f"mask shape: {mask.shape}, dtype={mask.dtype}")

    if raw.ndim != 3:
        raise ValueError(f"期望 raw 是 3D (Z,H,W)，实际得到: {raw.shape}")
    if mask.ndim != 3:
        raise ValueError(f"期望 mask 是 3D (Z,H,W)，实际得到: {mask.shape}")
    if raw.shape != mask.shape:
        raise ValueError(
            f"raw 和 mask shape 不一致: raw={raw.shape}, mask={mask.shape}"
        )

    # 原图归一化到 uint8
    print("开始归一化原图...")
    raw_u8 = robust_normalize_to_uint8(
        raw,
        pmin=RAW_NORM_PMIN,
        pmax=RAW_NORM_PMAX,
    )

    masked_z = np.where((mask > 0).reshape(mask.shape[0], -1).any(axis=1))[0].tolist()

    # 半透明填充版
    print("生成半透明填充版 overlay 3D tif ...")
    overlay_fill = build_fill_overlay_stack(
        raw_u8=raw_u8,
        mask_label=mask,
        alpha=FILL_ALPHA,
        color_rgb=MASK_COLOR_RGB,
    )

    overlay_fill_path = tag_out_dir / "overlay_fill_3d.tif"
    print(f"保存: {overlay_fill_path}")
    tiff.imwrite(
        overlay_fill_path,
        overlay_fill,
        photometric="rgb",
        compression=TIFF_COMPRESSION,
    )

    # 边界版
    overlay_boundary_path = None
    if SAVE_BOUNDARY_VERSION:
        print("生成边界高亮版 overlay 3D tif ...")
        overlay_boundary = build_boundary_overlay_stack(
            raw_u8=raw_u8,
            mask_label=mask,
            alpha=BOUNDARY_ALPHA,
            color_rgb=MASK_COLOR_RGB,
            dilate_iters=BOUNDARY_DILATE_ITERS,
        )

        overlay_boundary_path = tag_out_dir / "overlay_boundary_3d.tif"
        print(f"保存: {overlay_boundary_path}")
        tiff.imwrite(
            overlay_boundary_path,
            overlay_boundary,
            photometric="rgb",
            compression=TIFF_COMPRESSION,
        )

    # 元信息
    meta = {
        "tag": tag,
        "raw_tif_path": str(raw_path),
        "mask_tif_path": str(mask_path),
        "shape_zyx": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "mask_dtype": str(mask.dtype),
        "masked_z_count": len(masked_z),
        "masked_z_min": int(masked_z[0]) if len(masked_z) > 0 else None,
        "masked_z_max": int(masked_z[-1]) if len(masked_z) > 0 else None,
        "output_overlay_fill_3d_tif": str(overlay_fill_path),
        "output_overlay_boundary_3d_tif": str(overlay_boundary_path) if overlay_boundary_path else None,
        "params": {
            "RAW_NORM_PMIN": RAW_NORM_PMIN,
            "RAW_NORM_PMAX": RAW_NORM_PMAX,
            "FILL_ALPHA": FILL_ALPHA,
            "BOUNDARY_ALPHA": BOUNDARY_ALPHA,
            "BOUNDARY_DILATE_ITERS": BOUNDARY_DILATE_ITERS,
            "MASK_COLOR_RGB": MASK_COLOR_RGB.tolist(),
            "SAVE_BOUNDARY_VERSION": SAVE_BOUNDARY_VERSION,
            "TIFF_COMPRESSION": TIFF_COMPRESSION,
        }
    }
    save_json(meta, tag_out_dir / "meta.json")

    if SAVE_MASKED_Z_LIST:
        save_json(masked_z, tag_out_dir / "masked_z_list.json")

    print("完成。")
    print(f"输出目录: {tag_out_dir}")
    print(f"有 mask 的 z 层数: {len(masked_z)}")


# =========================================================
# 4) main
# =========================================================
def main():
    ensure_dir(OUT_ROOT)

    print("=" * 100)
    print("Manual Overlay 3D TIF")
    print("=" * 100)
    print(f"OUT_ROOT : {OUT_ROOT}")
    print(f"RAW PATH : {DEFAULT_RAW_TIF_PATH}")
    print(f"MASK 数量 : {len(MASK_ITEMS)}")

    for item in MASK_ITEMS:
        process_one_mask_item(
            raw_path=DEFAULT_RAW_TIF_PATH,
            item=item,
            out_root=OUT_ROOT,
        )

    print("=" * 100)
    print("全部完成")
    print("=" * 100)


if __name__ == "__main__":
    main()


# In[8]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = None
RESCALE = 1.75
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[ ]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p25_cp2p5_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp3p5__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp3p5__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[ ]:





# In[9]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[16]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_a1_flow02_fs2",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1__cp2p5__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1__cp2p5__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[10]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[15]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_a1_flow02_fs2",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp2p5__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp2p5__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[18]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2
FLOW3D_SMOOTH = [1,0,0]

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_fs100_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
        flow3D_smooth=FLOW3D_SMOOTH,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,
        "FLOW3D_SMOOTH": FLOW3D_SMOOTH,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[19]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2
FLOW3D_SMOOTH = 1

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_fs100_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
        flow3D_smooth=FLOW3D_SMOOTH,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,
        "FLOW3D_SMOOTH": FLOW3D_SMOOTH,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[20]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_a1_flow02_fs1",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_fs100_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp2p5__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp2p5__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[21]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 1
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.8

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[22]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_a1_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale125_dNone_cp2p5_ms50_flow02_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[24]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 1
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.4

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rs1_dNone_cp1_ms50_flow04_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[25]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1p75_a1_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rs1_dNone_cp1_ms50_flow04_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[26]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 0
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.4

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rs1_dNone_cp0_ms50_flow04_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[27]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1_a1_cp0_flow04",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rs1_dNone_cp0_ms50_flow04_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp0__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp0__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[ ]:





# In[29]:


# ==========================================
# Cell Full3D-StepA-Top1 | Version B (rescale版)
# 不再对 full-brain 3D volume 做外部 xyz 插值，
# 直接用原始 3D stack + 微调模型做 3D 推理，只保存 raw masks
#
# 目的：
#   1) 让真实预测时约 8 px 的目标，通过 rescale=14/8=1.75，
#      更接近微调时约 14 px 的尺度分布
#   2) 避免传 diameter=8 触发 Cellpose-SAM 内部往 30 px 参考直径缩放
#
# 当前策略：
#   - ANISOTROPY = 2.0   (保持不变，用于 3D spacing)
#   - DIAMETER = None    (关键：不走 diameter->30px 那套逻辑)
#   - RESCALE = 14.0/8.0 = 1.75
#   - CELLPROB_THRESHOLD = 2.5
#   - MIN_SIZE = 50
#   - BATCH_SIZE_3D = 4
#
# 注意：
#   - 输入给 model.eval 的是原始 raw stack
#   - 不再做外部 xyz*2 插值
#   - 输出 mask 的 shape 与原始 raw shape 对应
# ==========================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Full3D-StepA-Top1 | rescale=1.75 + diameter=None + do_3D")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_*"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入来源：仍然读取 3.11
# --------------------------------------------------
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 固定参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 1
MIN_SIZE = 50

ANISOTROPY = 1
DIAMETER = None
RESCALE = 1
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

FLOW_THRESHOLD = 0.2

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale1_dNone_cp1_ms50_flow02_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
rs_str = str(RESCALE).replace(".", "p").replace("-", "m")

run_name = (
    f"{TARGET_TAG}"
    f"__a{ANISOTROPY}"
    f"__dNone"
    f"__rs{rs_str}"
    f"__cp{cp_str}"
    f"__ms{MIN_SIZE}"
)

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()

for p in [RUN_ROOT, RAW_MASK_ROOT, STAT_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def estimate_nbytes_gb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 3)

# --------------------------------------------------
# 5) 自动找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None
selected_snapshot = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

    cand = None
    if best_model_path and Path(best_model_path).exists():
        cand = Path(best_model_path).resolve()
    elif final_model_path and Path(final_model_path).exists():
        cand = Path(final_model_path).resolve()

    if cand is not None:
        selected_model_path = cand
        selected_snapshot_path = sf.resolve()
        selected_snapshot = snap

if selected_model_path is None:
    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型")

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

(SNAP_ROOT / "selected_model_path.txt").write_text(
    str(selected_model_path) + "\n",
    encoding="utf-8"
)
(SNAP_ROOT / "selected_config_path.txt").write_text(
    str(selected_snapshot_path) + "\n",
    encoding="utf-8"
)
if selected_snapshot is not None:
    (SNAP_ROOT / "selected_config_snapshot.json").write_text(
        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------
# 6) 输出路径
# --------------------------------------------------
mask_out = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
meta_out = STAT_ROOT / f"{run_name}__stepA_meta.json"

print("🎯 run_name :", run_name)
print("mask_out    :", mask_out)
print("meta_out    :", meta_out)

if mask_out.exists() and meta_out.exists() and not FORCE_RERUN:
    print("⏭️ 已存在且 FORCE_RERUN=False，直接跳过")
else:
    # --------------------------------------------------
    # 7) 读取 raw
    # --------------------------------------------------
    print("\n[1/4] 读取 raw")
    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"

    original_shape = tuple(stack.shape)
    original_dtype = str(stack.dtype)

    print("✅ original stack shape:", original_shape, stack.dtype)
    print(f"   original size ~ {estimate_nbytes_gb(np.asarray(stack)):.3f} GB")

    # 为了后续更稳，统一转 float32
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32, copy=False)

    # --------------------------------------------------
    # 8) 加载模型
    # --------------------------------------------------
    print("\n[2/4] 加载模型")
    model = models.CellposeModel(
        gpu=True,
        pretrained_model=str(selected_model_path),
    )
    print("✅ model loaded")

    # --------------------------------------------------
    # 9) full-brain 3D eval
    # --------------------------------------------------
    print("\n[3/4] full-brain 3D eval on original stack with rescale")
    print(f"🚦 BATCH_SIZE_3D        = {BATCH_SIZE_3D}")
    print(f"🚦 ANISOTROPY           = {ANISOTROPY}")
    print(f"🚦 DIAMETER             = {DIAMETER}")
    print(f"🚦 RESCALE              = {RESCALE}")
    print(f"🚦 CELLPROB_THRESHOLD   = {CELLPROB_THRESHOLD}")
    print(f"🚦 MIN_SIZE             = {MIN_SIZE}")
    print(f"🚦 DO_3D                = {DO_3D}")
    print(f"🚦 Z_AXIS               = {Z_AXIS}")

    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        rescale=RESCALE,
        cellprob_threshold=CELLPROB_THRESHOLD,
        flow_threshold=FLOW_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
       
    )

    eval_elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   eval_elapsed_s :", eval_elapsed_s)
    print("   masks shape    :", masks.shape)
    print("   masks dtype    :", masks.dtype)
    print(f"   masks size ~ {estimate_nbytes_gb(np.asarray(masks)):.3f} GB")

    # --------------------------------------------------
    # 10) 保存 raw masks
    # --------------------------------------------------
    print("\n[4/4] 保存 raw masks")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(mask_out), masks)
    else:
        tiff.imwrite(str(mask_out), masks.astype(np.uint32))

    meta = {
        "run_name": run_name,
        "step": "A_save_raw_mask_only_rescale_dNone",
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),

        "original_shape": list(original_shape),
        "original_dtype": original_dtype,

        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "rescale": RESCALE,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "do_3d": DO_3D,
        "z_axis": Z_AXIS,
        "FLOW_THRESHOLD": FLOW_THRESHOLD,

        "eval_elapsed_s": eval_elapsed_s,

        "notes": [
            "Original stack was sent to model.eval without external xyz interpolation",
            "diameter=None + rescale=1.75 is used to avoid diameter-driven resize to 30 px",
            "anisotropy=2.0 is kept for 3D spacing handling"
        ],
    }

    meta_out.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Version B StepA DONE")


# In[30]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_fixed5.py

固定 5 条 mask 路径的 hollow artifact 分析脚本

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
- 每个 mask 的 summary.json / slice_metrics.csv / instance_metrics.csv
"""

from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


# =========================================================
# 0) 配置区
# =========================================================

# ---------------------------
# 输出根目录
# ---------------------------
BATCH_OUT_ROOT = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "hollow_analysis_batch_fixed5"
).resolve()

# ---------------------------
# 原始 full-brain 3D tif（统一配给所有 mask）
# ---------------------------
DEFAULT_RAW_TIF_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/"
    "iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# ---------------------------
# 固定 5 条路径清单
# tag 建议短而清楚，方便横向比较
# ---------------------------
MASK_ITEMS = [
    {
        "tag": "rs1_a1_cp1_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale1_dNone_cp1_ms50_flow02_v1/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50/raw_masks/P21_lr9e5_wd8e3__a1__dNone__rs1__cp1__ms50__full_brain_3d_masks_raw.tif",
    },
    
]

# ---------------------------
# 日志控制
# ---------------------------
VERBOSE_INSTANCE_PROGRESS = False   # True 时恢复“每 200 个实例打印一次”
PRINT_TOP_SUMMARY_ROWS = 10         # 最终终端打印前多少行 summary

# ---------------------------
# 切片级判定参数
# ---------------------------
ABS_MIN_AREA = 30
REL_VALID_AREA_RATIO = 0.20

MIN_HOLE_AREA = 8
HOLE_RATIO_STRONG = 0.12
HOLE_RATIO_MID = 0.08

CORE_DIST_RATIO = 0.50
MIN_CORE_PIXELS = 6
CENTER_OCC_STRONG = 0.35
CENTER_OCC_MID = 0.50

SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45

# 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 50   # 从 80 收紧一点，目录不至于太炸


# =========================================================
# 1) 工具函数
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def longest_true_run(bool_list):
    best = 0
    cur = 0
    for v in bool_list:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_center_region_from_filled_mask(filled_mask, core_dist_ratio=0.5, min_core_pixels=6):
    if filled_mask.sum() == 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    dist = ndi.distance_transform_edt(filled_mask)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros_like(filled_mask, dtype=bool), 0.0

    core_mask = dist >= (core_dist_ratio * max_dist)

    if core_mask.sum() < min_core_pixels:
        vals = dist[filled_mask]
        if vals.size == 0:
            return np.zeros_like(filled_mask, dtype=bool), max_dist

        k = min(min_core_pixels, vals.size)
        kth = np.partition(vals, -k)[-k]
        core_mask = (dist >= kth) & filled_mask

    return core_mask, float(max_dist)


def compute_slice_metrics(
    slice_mask_2d,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    instance_max_area=0,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4
):
    eps = 1e-8
    s = slice_mask_2d.astype(bool)
    area_raw = int(s.sum())

    valid_area_th = max(abs_min_area, int(math.ceil(rel_valid_area_ratio * instance_max_area)))
    is_valid_slice = area_raw >= valid_area_th

    fill_mask = None
    hole_mask = None
    core_mask = None

    area_fill = area_raw
    area_hole = 0
    hole_ratio = 0.0
    center_occupancy = np.nan
    severity = 0.0
    is_hollow_slice = False

    if is_valid_slice:
        fill_mask = ndi.binary_fill_holes(s)
        hole_mask = fill_mask & (~s)

        area_fill = int(fill_mask.sum())
        area_hole = int(hole_mask.sum())
        hole_ratio = area_hole / (area_fill + eps)

        core_mask, _ = compute_center_region_from_filled_mask(
            fill_mask,
            core_dist_ratio=core_dist_ratio,
            min_core_pixels=min_core_pixels
        )

        if core_mask.sum() > 0:
            center_occupancy = float(s[core_mask].mean())
        else:
            center_occupancy = np.nan

        center_badness = 0.0 if np.isnan(center_occupancy) else (1.0 - center_occupancy)
        severity = severity_alpha * hole_ratio + severity_beta * center_badness
        severity = float(np.clip(severity, 0.0, 1.0))

        cond_a = (area_hole >= min_hole_area) and (hole_ratio >= hole_ratio_strong)
        cond_b = (not np.isnan(center_occupancy)) and (center_occupancy <= center_occ_strong)
        cond_c = (
            (area_hole >= min_hole_area)
            and (hole_ratio >= hole_ratio_mid)
            and (not np.isnan(center_occupancy))
            and (center_occupancy <= center_occ_mid)
        )

        is_hollow_slice = bool(cond_a or cond_b or cond_c)

    return {
        "area_raw": area_raw,
        "valid_area_th": valid_area_th,
        "is_valid_slice": bool(is_valid_slice),
        "area_fill": int(area_fill),
        "area_hole": int(area_hole),
        "hole_ratio": float(hole_ratio),
        "center_occupancy": safe_float(center_occupancy),
        "severity": float(severity),
        "is_hollow_slice": bool(is_hollow_slice),
        "fill_mask": fill_mask,
        "hole_mask": hole_mask,
        "core_mask": core_mask,
    }


def save_slice_visualization(
    out_png_path: Path,
    raw_slice,
    mask_slice,
    fill_mask,
    hole_mask,
    core_mask,
    title_text=""
):
    has_raw = raw_slice is not None
    ncols = 5 if has_raw else 4
    fig = plt.figure(figsize=(4.5 * ncols, 4.5))
    idx = 1

    if has_raw:
        ax = fig.add_subplot(1, ncols, idx)
        idx += 1
        ax.imshow(raw_slice, cmap="gray")
        ax.set_title("Raw")
        ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Mask")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(fill_mask, cmap="gray")
    ax.set_title("Fill Holes")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    idx += 1
    ax.imshow(hole_mask, cmap="gray")
    ax.set_title("Hole Region")
    ax.axis("off")

    ax = fig.add_subplot(1, ncols, idx)
    overlay_base = raw_slice if has_raw else mask_slice.astype(np.float32)
    ax.imshow(overlay_base, cmap="gray")
    if hole_mask is not None:
        yy, xx = np.where(hole_mask)
        ax.scatter(xx, yy, s=3)
    if core_mask is not None and core_mask.sum() > 0:
        yy2, xx2 = np.where(core_mask)
        ax.scatter(xx2, yy2, s=1)
    ax.set_title("Overlay")
    ax.axis("off")

    fig.suptitle(title_text, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def parse_cp_numeric(cp_tag: str):
    mapping = {
        "cp0": 0,
        "cp1": 1,
        "cpneg1": -1,
        "cpm1": -1,
    }
    return mapping.get(cp_tag, np.nan)


def parse_flow_numeric(flow_tag: str):
    mapping = {
        "flow02": 0.2,
        "flow04": 0.4,
        "flow08": 0.8,
    }
    return mapping.get(flow_tag, np.nan)


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
    tag=None,
    flow_tag=None,
    cp_tag=None,
    abs_min_area=30,
    rel_valid_area_ratio=0.20,
    min_hole_area=8,
    hole_ratio_strong=0.12,
    hole_ratio_mid=0.08,
    core_dist_ratio=0.50,
    min_core_pixels=6,
    center_occ_strong=0.35,
    center_occ_mid=0.50,
    severity_alpha=0.6,
    severity_beta=0.4,
    instance_min_hollow_slices=2,
    instance_min_hollow_ratio=0.20,
    instance_min_longest_run=2,
    instance_max_severity_th=0.45,
    save_vis=True,
    topk_vis_per_instance=3,
    max_vis_instances=80,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    vis_dir = out_dir / "vis_hollow_slices"
    if save_vis:
        ensure_dir(vis_dir)

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"    ⚠ raw shape 不匹配，跳过 raw: raw={raw.shape}, mask={masks.shape}")
            raw = None

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    if num_instances == 0:
        raise ValueError("mask 中没有正实例标签。")

    obj_slices = ndi.find_objects(masks)

    slice_rows = []
    instance_rows = []
    hollow_instance_vis_count = 0

    for idx_i, label_id in enumerate(label_ids, start=1):
        if label_id - 1 >= len(obj_slices):
            continue

        bbox = obj_slices[label_id - 1]
        if bbox is None:
            continue

        z_sl, y_sl, x_sl = bbox
        sub_mask_labels = masks[z_sl, y_sl, x_sl]
        inst_3d = (sub_mask_labels == label_id)

        slice_areas = inst_3d.sum(axis=(1, 2))
        instance_max_area = int(slice_areas.max()) if slice_areas.size > 0 else 0
        total_voxels = int(inst_3d.sum())

        z_indices_local = np.where(slice_areas > 0)[0]
        if len(z_indices_local) == 0:
            continue

        per_inst_slice_results = []

        for zl in z_indices_local:
            z_global = z_sl.start + zl
            s2d = inst_3d[zl]

            m = compute_slice_metrics(
                slice_mask_2d=s2d,
                abs_min_area=abs_min_area,
                rel_valid_area_ratio=rel_valid_area_ratio,
                instance_max_area=instance_max_area,
                min_hole_area=min_hole_area,
                hole_ratio_strong=hole_ratio_strong,
                hole_ratio_mid=hole_ratio_mid,
                core_dist_ratio=core_dist_ratio,
                min_core_pixels=min_core_pixels,
                center_occ_strong=center_occ_strong,
                center_occ_mid=center_occ_mid,
                severity_alpha=severity_alpha,
                severity_beta=severity_beta,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "label_id": int(label_id),
                "z": int(z_global),
                "bbox_z0": int(z_sl.start),
                "bbox_z1": int(z_sl.stop - 1),
                "bbox_y0": int(y_sl.start),
                "bbox_y1": int(y_sl.stop - 1),
                "bbox_x0": int(x_sl.start),
                "bbox_x1": int(x_sl.stop - 1),
                "instance_total_voxels": int(total_voxels),
                "instance_max_area": int(instance_max_area),
                "area_raw": int(m["area_raw"]),
                "valid_area_th": int(m["valid_area_th"]),
                "is_valid_slice": int(m["is_valid_slice"]),
                "area_fill": int(m["area_fill"]),
                "area_hole": int(m["area_hole"]),
                "hole_ratio": float(m["hole_ratio"]),
                "center_occupancy": safe_float(m["center_occupancy"]),
                "severity": float(m["severity"]),
                "is_hollow_slice": int(m["is_hollow_slice"]),
            }
            slice_rows.append(row)

            per_inst_slice_results.append({
                "z_global": z_global,
                "slice_mask_2d": s2d,
                "metrics": m,
            })

        valid_flags = [r["metrics"]["is_valid_slice"] for r in per_inst_slice_results]
        hollow_flags = [r["metrics"]["is_hollow_slice"] for r in per_inst_slice_results]

        valid_slices = int(sum(valid_flags))
        hollow_slices = int(sum(hollow_flags))
        hollow_ratio = hollow_slices / max(valid_slices, 1)

        longest_hollow_run = longest_true_run(
            [bool(h and v) for h, v in zip(hollow_flags, valid_flags)]
        )

        severities = [
            r["metrics"]["severity"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_severity = float(max(severities)) if len(severities) > 0 else 0.0
        mean_severity = float(np.mean(severities)) if len(severities) > 0 else 0.0

        hole_ratios = [
            r["metrics"]["hole_ratio"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"]
        ]
        max_hole_ratio = float(max(hole_ratios)) if len(hole_ratios) > 0 else 0.0

        center_occs = [
            r["metrics"]["center_occupancy"]
            for r in per_inst_slice_results
            if r["metrics"]["is_valid_slice"] and not np.isnan(r["metrics"]["center_occupancy"])
        ]
        min_center_occupancy = float(min(center_occs)) if len(center_occs) > 0 else np.nan

        is_hollow_instance = (
            (
                hollow_slices >= instance_min_hollow_slices
                and hollow_ratio >= instance_min_hollow_ratio
            )
            or (longest_hollow_run >= instance_min_longest_run)
            or (
                max_severity >= instance_max_severity_th
                and hollow_slices >= INSTANCE_MIN_SLICES_FOR_MAX_SEV
            )
        )

        inst_row = {
            "tag": tag,
            "flow_tag": flow_tag,
            "cp_tag": cp_tag,
            "flow_value": parse_flow_numeric(flow_tag),
            "cp_value": parse_cp_numeric(cp_tag),

            "label_id": int(label_id),
            "z_min": int(z_sl.start),
            "z_max": int(z_sl.stop - 1),
            "y_min": int(y_sl.start),
            "y_max": int(y_sl.stop - 1),
            "x_min": int(x_sl.start),
            "x_max": int(x_sl.stop - 1),
            "total_voxels": int(total_voxels),
            "instance_max_area": int(instance_max_area),
            "num_nonzero_slices": int(len(z_indices_local)),
            "valid_slices": int(valid_slices),
            "hollow_slices": int(hollow_slices),
            "hollow_ratio": float(hollow_ratio),
            "longest_hollow_run": int(longest_hollow_run),
            "max_hole_ratio": float(max_hole_ratio),
            "min_center_occupancy": safe_float(min_center_occupancy),
            "max_severity": float(max_severity),
            "mean_severity": float(mean_severity),
            "is_hollow_instance": int(is_hollow_instance),
        }
        instance_rows.append(inst_row)

        if save_vis and is_hollow_instance and hollow_instance_vis_count < max_vis_instances:
            hollow_instance_vis_count += 1
            candidates = []
            for r in per_inst_slice_results:
                m = r["metrics"]
                if m["is_valid_slice"] and m["is_hollow_slice"]:
                    candidates.append(r)

            candidates = sorted(
                candidates,
                key=lambda x: x["metrics"]["severity"],
                reverse=True
            )[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"tag={tag}, label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"{tag}_label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if VERBOSE_INSTANCE_PROGRESS and (idx_i % 200 == 0 or idx_i == num_instances):
            print(f"    已处理 {idx_i}/{num_instances} 个实例")

    slice_df = pd.DataFrame(slice_rows)
    inst_df = pd.DataFrame(instance_rows)

    if len(slice_df) > 0:
        slice_df = slice_df.sort_values(["label_id", "z"]).reset_index(drop=True)
    if len(inst_df) > 0:
        inst_df = inst_df.sort_values(
            ["is_hollow_instance", "max_severity", "hollow_ratio"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    slice_csv = out_dir / "slice_metrics.csv"
    inst_csv = out_dir / "instance_metrics.csv"
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")
    inst_df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    total_instances = int(len(inst_df))
    hollow_instances = int(inst_df["is_hollow_instance"].sum()) if total_instances > 0 else 0
    hollow_instance_ratio = hollow_instances / max(total_instances, 1)

    if len(slice_df) > 0:
        valid_slice_df = slice_df[slice_df["is_valid_slice"] == 1]
        total_valid_slices = int(len(valid_slice_df))
        hollow_slices = int(valid_slice_df["is_hollow_slice"].sum()) if total_valid_slices > 0 else 0
        hollow_slice_ratio = hollow_slices / max(total_valid_slices, 1)
        mean_hollow_severity = float(valid_slice_df["severity"].mean()) if total_valid_slices > 0 else 0.0
    else:
        total_valid_slices = 0
        hollow_slices = 0
        hollow_slice_ratio = 0.0
        mean_hollow_severity = 0.0

    summary = {
        "tag": tag,
        "flow_tag": flow_tag,
        "cp_tag": cp_tag,
        "flow_value": parse_flow_numeric(flow_tag),
        "cp_value": parse_cp_numeric(cp_tag),

        "mask_tif_path": str(mask_tif_path),
        "raw_tif_path": str(raw_tif_path) if raw_tif_path is not None else None,
        "shape": [int(Z), int(Y), int(X)],
        "params": {
            "ABS_MIN_AREA": abs_min_area,
            "REL_VALID_AREA_RATIO": rel_valid_area_ratio,
            "MIN_HOLE_AREA": min_hole_area,
            "HOLE_RATIO_STRONG": hole_ratio_strong,
            "HOLE_RATIO_MID": hole_ratio_mid,
            "CORE_DIST_RATIO": core_dist_ratio,
            "MIN_CORE_PIXELS": min_core_pixels,
            "CENTER_OCC_STRONG": center_occ_strong,
            "CENTER_OCC_MID": center_occ_mid,
            "SEVERITY_ALPHA": severity_alpha,
            "SEVERITY_BETA": severity_beta,
            "INSTANCE_MIN_HOLLOW_SLICES": instance_min_hollow_slices,
            "INSTANCE_MIN_HOLLOW_RATIO": instance_min_hollow_ratio,
            "INSTANCE_MIN_LONGEST_RUN": instance_min_longest_run,
            "INSTANCE_MAX_SEVERITY_TH": instance_max_severity_th,
            "INSTANCE_MIN_SLICES_FOR_MAX_SEV": INSTANCE_MIN_SLICES_FOR_MAX_SEV,
        },
        "total_instances": int(total_instances),
        "hollow_instances": int(hollow_instances),
        "hollow_instance_ratio": float(hollow_instance_ratio),
        "total_valid_slices": int(total_valid_slices),
        "hollow_slices": int(hollow_slices),
        "hollow_slice_ratio": float(hollow_slice_ratio),
        "mean_hollow_severity": float(mean_hollow_severity),
        "outputs": {
            "slice_csv": str(slice_csv),
            "instance_csv": str(inst_csv),
            "vis_dir": str(vis_dir) if save_vis else None,
        }
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return slice_df, inst_df, summary


# =========================================================
# 3) 固定清单批量模式
# =========================================================
def run_fixed_list_analysis(mask_items, batch_out_root, default_raw_tif_path=None):
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Hollow Artifact Analysis | fixed 5-mask list")
    print("=" * 100)
    print(f"输出目录: {batch_out_root}")
    print(f"样本数  : {len(mask_items)}")
    print(f"raw tif : {default_raw_tif_path}")
    print("-" * 100)

    batch_rows = []
    ok_count = 0
    fail_count = 0
    failed_tags = []

    for idx, item in enumerate(mask_items, start=1):
        tag = item["tag"]
        flow_tag = item.get("flow_tag")
        cp_tag = item.get("cp_tag")
        mask_path = Path(item["mask_tif_path"]).resolve()
        out_dir = batch_out_root / sanitize_name(tag)

        print(f"[{idx}/{len(mask_items)}] 开始: {tag}")
        print(f"    mask: {mask_path}")

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,
                tag=tag,
                flow_tag=flow_tag,
                cp_tag=cp_tag,

                abs_min_area=ABS_MIN_AREA,
                rel_valid_area_ratio=REL_VALID_AREA_RATIO,

                min_hole_area=MIN_HOLE_AREA,
                hole_ratio_strong=HOLE_RATIO_STRONG,
                hole_ratio_mid=HOLE_RATIO_MID,

                core_dist_ratio=CORE_DIST_RATIO,
                min_core_pixels=MIN_CORE_PIXELS,
                center_occ_strong=CENTER_OCC_STRONG,
                center_occ_mid=CENTER_OCC_MID,

                severity_alpha=SEVERITY_ALPHA,
                severity_beta=SEVERITY_BETA,

                instance_min_hollow_slices=INSTANCE_MIN_HOLLOW_SLICES,
                instance_min_hollow_ratio=INSTANCE_MIN_HOLLOW_RATIO,
                instance_min_longest_run=INSTANCE_MIN_LONGEST_RUN,
                instance_max_severity_th=INSTANCE_MAX_SEVERITY_TH,

                save_vis=SAVE_VIS,
                topk_vis_per_instance=TOPK_VIS_PER_INSTANCE,
                max_vis_instances=MAX_VIS_INSTANCES,
            )

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": summary["total_instances"],
                "hollow_instances": summary["hollow_instances"],
                "hollow_instance_ratio": summary["hollow_instance_ratio"],
                "total_valid_slices": summary["total_valid_slices"],
                "hollow_slices": summary["hollow_slices"],
                "hollow_slice_ratio": summary["hollow_slice_ratio"],
                "mean_hollow_severity": summary["mean_hollow_severity"],
                "status": "ok",
            }

            ok_count += 1
            print(
                f"    完成: {tag} | "
                f"inst={summary['total_instances']} | "
                f"hollow_inst_ratio={summary['hollow_instance_ratio']:.4f} | "
                f"hollow_slice_ratio={summary['hollow_slice_ratio']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            failed_tags.append(tag)

            row = {
                "tag": tag,
                "flow_tag": flow_tag,
                "cp_tag": cp_tag,
                "flow_value": parse_flow_numeric(flow_tag),
                "cp_value": parse_cp_numeric(cp_tag),

                "mask_tif_path": str(mask_path),
                "out_dir": str(out_dir),
                "total_instances": np.nan,
                "hollow_instances": np.nan,
                "hollow_instance_ratio": np.nan,
                "total_valid_slices": np.nan,
                "hollow_slices": np.nan,
                "hollow_slice_ratio": np.nan,
                "mean_hollow_severity": np.nan,
                "status": f"failed: {e}",
            }

            print(f"    失败: {tag} | {e}")

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

        print("-" * 100)

    batch_df = pd.DataFrame(batch_rows)
    batch_csv = batch_out_root / "batch_summary.csv"
    batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("✅ Batch 分析完成")
    print(f"汇总表: {batch_csv}")
    print(f"成功数: {ok_count}")
    print(f"失败数: {fail_count}")
    if failed_tags:
        print(f"失败样本: {failed_tags}")

    if len(batch_df) > 0:
        ok_df = batch_df[batch_df["status"] == "ok"].copy()
        if len(ok_df) > 0:
            ok_df = ok_df.sort_values(
                ["hollow_instance_ratio", "hollow_slice_ratio", "mean_hollow_severity"],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            print("-" * 100)
            print("按 hollow_instance_ratio 排序的结果预览：")
            cols = [
                "tag",
                "flow_tag",
                "cp_tag",
                "total_instances",
                "hollow_instances",
                "hollow_instance_ratio",
                "hollow_slice_ratio",
                "mean_hollow_severity",
            ]
            print(ok_df[cols].head(PRINT_TOP_SUMMARY_ROWS).to_string(index=False))
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    run_fixed_list_analysis(
        mask_items=MASK_ITEMS,
        batch_out_root=BATCH_OUT_ROOT,
        default_raw_tif_path=DEFAULT_RAW_TIF_PATH
    )


# In[ ]:




