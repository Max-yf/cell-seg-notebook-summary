#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================
# Cell 3.19-2p5D-StepA：
# 扫描 2.5D 参数（do_3D=False + stitch_threshold>0）
# 只负责：
#   1) 读取 full-brain 3D stack
#   2) 扫描 min_size / stitch_threshold / flow_threshold
#   3) 保存 raw 2.5D masks
#   4) 记录每组参数、耗时、显存占用
#   5) 导出 grid 和 stepA summary
#
# 不负责：
#   - 后处理过滤
#   - 统计分析
#   - overlay / top-k 可视化
# 这些留到 StepB
# ==========================================

import os
import gc
import json
import time
import math
import itertools
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None


print("=" * 100)
print("🚀 Cell 3.19-2p5D-StepA | 2.5D sweep: save raw masks + memory logs")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 路径：这次全部挂到 3.19 时间戳目录下
# --------------------------------------------------
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
RUNS_ROOT = (ROOT / "runs").resolve()

# 你这次明确要求“全部保存在时间戳为 3.19 的文件夹里”
# 这里单独新建一个 3.19 实验目录，不去污染旧 3.18 / 3.11
OUT_EXP_DIR = (
    RUNS_ROOT / f"exp_20260319_2p5d_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
).resolve()

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# 模型来源：沿用你当前最常用的实验 config 目录去找 TARGET_TAG 对应模型
SRC_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (SRC_EXP_DIR / "config").resolve()

# --------------------------------------------------
# 1) 核心配置：你最常改的地方
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"

# ===== 2.5D 固定参数 =====
DO_3D = False
Z_AXIS = 0
DIAMETER = 8.0
CELLPROB_THRESHOLD = 2.5
BATCH_SIZE = 8   # 先固定一档，后面你要模拟 4090 时再调
CHANNELS = [0, 0]  # 灰度图常用写法
NORMALIZE = True

# ===== 本轮要扫描的参数 =====
MIN_SIZE_LIST = [0, 30, 50, 100]
STITCH_THRESHOLD_LIST = [0.0, 0.1, 0.2, 0.3]
FLOW_THRESHOLD_LIST = [0.4, 0.6, 0.8]

# 是否强制重跑已有结果
FORCE_RERUN = False

# 是否每组都落盘保存 mask
SAVE_MASK = True

# 是否记录 nvidia-smi
ENABLE_NVIDIA_SMI = True

# 如果你只想先小试几组，可以改成 True
DEBUG_SMALL_GRID = False

# --------------------------------------------------
# 2) 输出目录：全部放在 3.19 目录下
# --------------------------------------------------
STEPA_ROOT = (OUT_EXP_DIR / "stepA_2p5d_sweep").resolve()
MASK_ROOT = (STEPA_ROOT / TARGET_TAG / "masks").resolve()
LOG_ROOT = (STEPA_ROOT / TARGET_TAG / "logs").resolve()
META_ROOT = (STEPA_ROOT / TARGET_TAG / "meta").resolve()
SUMMARY_ROOT = (STEPA_ROOT / TARGET_TAG / "summary").resolve()
CODE_ROOT = (STEPA_ROOT / TARGET_TAG / "code").resolve()

for p in [OUT_EXP_DIR, STEPA_ROOT, MASK_ROOT, LOG_ROOT, META_ROOT, SUMMARY_ROOT, CODE_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("\n📂 路径检查")
print("ROOT            :", ROOT)
print("RUNS_ROOT       :", RUNS_ROOT)
print("SRC_EXP_DIR     :", SRC_EXP_DIR)
print("CFG_DIR         :", CFG_DIR)
print("OUT_EXP_DIR     :", OUT_EXP_DIR)
print("STEPA_ROOT      :", STEPA_ROOT)
print("MASK_ROOT       :", MASK_ROOT)
print("LOG_ROOT        :", LOG_ROOT)
print("META_ROOT       :", META_ROOT)
print("SUMMARY_ROOT    :", SUMMARY_ROOT)
print("CODE_ROOT       :", CODE_ROOT)
print("RAW_3D_STACK    :", RAW_3D_STACK_PATH)

assert ROOT.exists(), f"❌ ROOT 不存在: {ROOT}"
assert RUNS_ROOT.exists(), f"❌ RUNS_ROOT 不存在: {RUNS_ROOT}"
assert SRC_EXP_DIR.exists(), f"❌ SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"
assert CFG_DIR.exists(), f"❌ CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW_3D_STACK_PATH 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 3) 工具函数
# --------------------------------------------------
def stage(msg: str):
    print(f"\n🔹 {msg}")

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def safe_write_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def cp_to_str(x):
    s = str(x)
    s = s.replace(".", "p").replace("-", "m")
    return s

def get_selected_model_path(cfg_dir: Path, target_tag: str):
    """
    从 config_*.json 中找和 TARGET_TAG 匹配的 best/final model path
    """
    config_files = sorted(cfg_dir.glob("config_*.json"))
    assert len(config_files) > 0, f"❌ CFG_DIR 下没找到 config_*.json: {cfg_dir}"

    selected_model_path = None
    selected_snapshot_path = None

    for sf in config_files:
        snap = read_json(sf)
        if not isinstance(snap, dict):
            continue

        tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
        if tag != target_tag:
            continue

        best_model_path = snap.get("best_model_path")
        final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")

        if best_model_path and Path(best_model_path).exists():
            selected_model_path = str(Path(best_model_path).resolve())
            selected_snapshot_path = str(sf.resolve())
            break
        elif final_model_path and Path(final_model_path).exists():
            selected_model_path = str(Path(final_model_path).resolve())
            selected_snapshot_path = str(sf.resolve())
            break

    assert selected_model_path is not None, f"❌ 没找到 TARGET_TAG={target_tag} 对应可用模型路径"
    return selected_model_path, selected_snapshot_path

def get_gpu_info_from_nvidia_smi():
    """
    返回当前机器所有 GPU 的显存信息
    """
    if not ENABLE_NVIDIA_SMI:
        return None

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        rows = []
        for line in out:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 4:
                continue
            idx, name, mem_used, mem_total = parts
            rows.append({
                "gpu_index": int(idx),
                "gpu_name": name,
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
            })
        return rows
    except Exception as e:
        print(f"⚠️ nvidia-smi 读取失败: {e}")
        return None

def get_gpu_peak_from_rows(rows):
    if rows is None or len(rows) == 0:
        return None
    peak_used_mb = max(r["memory_used_mb"] for r in rows)
    peak_row = sorted(rows, key=lambda x: x["memory_used_mb"], reverse=True)[0]
    return {
        "peak_gpu_index": peak_row["gpu_index"],
        "peak_gpu_name": peak_row["gpu_name"],
        "peak_memory_used_mb": peak_row["memory_used_mb"],
        "peak_memory_total_mb": peak_row["memory_total_mb"],
    }

def reset_torch_peak_memory():
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        # 对所有可见 GPU 重置峰值
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats(i)
    except Exception as e:
        print(f"⚠️ reset_peak_memory_stats 失败: {e}")

def collect_torch_peak_memory():
    """
    记录所有可见 GPU 的 PyTorch 峰值显存
    """
    if torch is None:
        return None
    if not torch.cuda.is_available():
        return None

    rows = []
    try:
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
            reserv = torch.cuda.max_memory_reserved(i) / (1024 ** 2)
            rows.append({
                "gpu_index": i,
                "torch_peak_allocated_mb": float(alloc),
                "torch_peak_reserved_mb": float(reserv),
            })
        return rows
    except Exception as e:
        print(f"⚠️ collect_torch_peak_memory 失败: {e}")
        return None

def torch_peak_summary(rows):
    if rows is None or len(rows) == 0:
        return None
    peak_alloc = max(r["torch_peak_allocated_mb"] for r in rows)
    peak_resv = max(r["torch_peak_reserved_mb"] for r in rows)
    peak_alloc_row = sorted(rows, key=lambda x: x["torch_peak_allocated_mb"], reverse=True)[0]
    peak_resv_row = sorted(rows, key=lambda x: x["torch_peak_reserved_mb"], reverse=True)[0]
    return {
        "torch_peak_allocated_mb_max": peak_alloc,
        "torch_peak_reserved_mb_max": peak_resv,
        "torch_peak_allocated_gpu_index": peak_alloc_row["gpu_index"],
        "torch_peak_reserved_gpu_index": peak_resv_row["gpu_index"],
    }

def build_run_name(
    target_tag,
    diameter,
    cellprob_threshold,
    min_size,
    stitch_threshold,
    flow_threshold
):
    return (
        f"{target_tag}"
        f"__2p5d"
        f"__d{cp_to_str(diameter)}"
        f"__cp{cp_to_str(cellprob_threshold)}"
        f"__ms{cp_to_str(min_size)}"
        f"__st{cp_to_str(stitch_threshold)}"
        f"__ft{cp_to_str(flow_threshold)}"
    )

# --------------------------------------------------
# 4) 找模型
# --------------------------------------------------
stage("定位模型路径")
SELECTED_MODEL_PATH, SELECTED_SNAPSHOT_PATH = get_selected_model_path(CFG_DIR, TARGET_TAG)

print("✅ SELECTED_MODEL_PATH   :", SELECTED_MODEL_PATH)
print("✅ SELECTED_SNAPSHOT_PATH:", SELECTED_SNAPSHOT_PATH)

# --------------------------------------------------
# 5) 读取 full-brain raw stack
# --------------------------------------------------
stage("读取 raw 3D stack")
raw_3d = tiff.imread(str(RAW_3D_STACK_PATH))
print("✅ raw_3d.shape:", raw_3d.shape, raw_3d.dtype)

assert raw_3d.ndim == 3, f"❌ 预期 raw_3d 是 3D stack，但拿到 shape={raw_3d.shape}"

# 可选：做一次小范围 debug
if DEBUG_SMALL_GRID:
    MIN_SIZE_LIST = [50]
    STITCH_THRESHOLD_LIST = [0.0, 0.1]
    FLOW_THRESHOLD_LIST = [0.4]
    print("⚠️ DEBUG_SMALL_GRID=True，当前网格已缩小")

# --------------------------------------------------
# 6) 生成参数网格
# --------------------------------------------------
stage("生成参数网格")
grid_rows = []

for min_size, stitch_threshold, flow_threshold in itertools.product(
    MIN_SIZE_LIST,
    STITCH_THRESHOLD_LIST,
    FLOW_THRESHOLD_LIST
):
    run_name = build_run_name(
        target_tag=TARGET_TAG,
        diameter=DIAMETER,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=min_size,
        stitch_threshold=stitch_threshold,
        flow_threshold=flow_threshold,
    )

    mask_path = (MASK_ROOT / f"{run_name}__raw_masks.tif").resolve()
    meta_json = (META_ROOT / f"{run_name}__meta.json").resolve()
    log_json = (LOG_ROOT / f"{run_name}__gpu_log.json").resolve()

    grid_rows.append({
        "run_name": run_name,
        "target_tag": TARGET_TAG,
        "model_path": SELECTED_MODEL_PATH,
        "snapshot_path": SELECTED_SNAPSHOT_PATH,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "do_3D": DO_3D,
        "z_axis": Z_AXIS,
        "diameter": DIAMETER,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": min_size,
        "stitch_threshold": stitch_threshold,
        "flow_threshold": flow_threshold,
        "batch_size": BATCH_SIZE,
        "normalize": NORMALIZE,
        "channels": str(CHANNELS),
        "mask_path": str(mask_path),
        "meta_json": str(meta_json),
        "log_json": str(log_json),
    })

grid_df = pd.DataFrame(grid_rows)
grid_csv = (CODE_ROOT / f"grid_2p5d_sweep_{TARGET_TAG}.csv").resolve()
grid_df.to_csv(grid_csv, index=False)

print(f"✅ 参数组合数: {len(grid_df)}")
print("✅ grid_csv   :", grid_csv)
display(grid_df.head())

# --------------------------------------------------
# 7) 初始化模型
# --------------------------------------------------
stage("初始化 Cellpose 模型")
model = models.CellposeModel(
    gpu=True,
    pretrained_model=SELECTED_MODEL_PATH,
)

print("✅ model loaded")

# --------------------------------------------------
# 8) 主循环：逐组跑 2.5D 推理
# --------------------------------------------------
stage("开始 StepA 扫描")
summary_rows = []

for idx, row in grid_df.iterrows():
    run_name = row["run_name"]
    mask_path = Path(row["mask_path"]).resolve()
    meta_json = Path(row["meta_json"]).resolve()
    log_json = Path(row["log_json"]).resolve()

    print("\n" + "-" * 100)
    print(f"[{idx + 1}/{len(grid_df)}] 🚀 run_name = {run_name}")
    print("-" * 100)

    if mask_path.exists() and meta_json.exists() and (not FORCE_RERUN):
        print("⏭️ 已存在结果，跳过")
        old_meta = read_json(meta_json)
        if isinstance(old_meta, dict):
            summary_rows.append(old_meta)
        continue

    # ---------- 推理前准备 ----------
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    reset_torch_peak_memory()

    smi_before = get_gpu_info_from_nvidia_smi()
    t0 = time.time()

    success = False
    err_msg = None
    pred_shape = None
    pred_dtype = None

    try:
        # =========================================================
        # 2.5D 核心：
        #   do_3D=False
        #   stitch_threshold > 0 时，Cellpose 会做 2D + stitch
        # =========================================================
        masks, flows, styles = model.eval(
            raw_3d,
            batch_size=int(row["batch_size"]),
            channels=CHANNELS,
            diameter=float(row["diameter"]),
            do_3D=bool(row["do_3D"]),
            z_axis=int(row["z_axis"]),
            stitch_threshold=float(row["stitch_threshold"]),
            cellprob_threshold=float(row["cellprob_threshold"]),
            flow_threshold=float(row["flow_threshold"]),
            min_size=int(row["min_size"]),
            normalize=bool(row["normalize"]),
        )

        elapsed_s = time.time() - t0

        pred_shape = tuple(masks.shape)
        pred_dtype = str(masks.dtype)

        if SAVE_MASK:
            tiff.imwrite(str(mask_path), masks.astype(np.uint32))
            print("✅ saved mask:", mask_path)

        success = True

    except Exception as e:
        elapsed_s = time.time() - t0
        err_msg = repr(e)
        print(f"❌ 推理失败: {err_msg}")

    smi_after = get_gpu_info_from_nvidia_smi()
    torch_peak_rows = collect_torch_peak_memory()

    # 汇总显存
    smi_peak = get_gpu_peak_from_rows(smi_after) if smi_after is not None else None
    torch_peak = torch_peak_summary(torch_peak_rows)

    gpu_log = {
        "run_name": run_name,
        "smi_before": smi_before,
        "smi_after": smi_after,
        "torch_peak_rows": torch_peak_rows,
        "smi_peak_summary": smi_peak,
        "torch_peak_summary": torch_peak,
    }
    safe_write_json(gpu_log, log_json)

    meta = {
        "run_name": run_name,
        "target_tag": row["target_tag"],
        "model_path": row["model_path"],
        "snapshot_path": row["snapshot_path"],
        "raw_3d_stack_path": row["raw_3d_stack_path"],
        "mask_path": str(mask_path),
        "meta_json": str(meta_json),
        "log_json": str(log_json),

        "do_3D": bool(row["do_3D"]),
        "z_axis": int(row["z_axis"]),
        "diameter": float(row["diameter"]),
        "cellprob_threshold": float(row["cellprob_threshold"]),
        "min_size": int(row["min_size"]),
        "stitch_threshold": float(row["stitch_threshold"]),
        "flow_threshold": float(row["flow_threshold"]),
        "batch_size": int(row["batch_size"]),
        "normalize": bool(row["normalize"]),
        "channels": row["channels"],

        "success": success,
        "error_msg": err_msg,
        "elapsed_s": float(elapsed_s),
        "pred_shape": str(pred_shape),
        "pred_dtype": pred_dtype,

        # nvidia-smi 峰值摘要
        "smi_peak_gpu_index": None if smi_peak is None else smi_peak["peak_gpu_index"],
        "smi_peak_gpu_name": None if smi_peak is None else smi_peak["peak_gpu_name"],
        "smi_peak_memory_used_mb": None if smi_peak is None else smi_peak["peak_memory_used_mb"],
        "smi_peak_memory_total_mb": None if smi_peak is None else smi_peak["peak_memory_total_mb"],

        # torch 峰值摘要
        "torch_peak_allocated_mb_max": None if torch_peak is None else torch_peak["torch_peak_allocated_mb_max"],
        "torch_peak_reserved_mb_max": None if torch_peak is None else torch_peak["torch_peak_reserved_mb_max"],
        "torch_peak_allocated_gpu_index": None if torch_peak is None else torch_peak["torch_peak_allocated_gpu_index"],
        "torch_peak_reserved_gpu_index": None if torch_peak is None else torch_peak["torch_peak_reserved_gpu_index"],

        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    safe_write_json(meta, meta_json)
    summary_rows.append(meta)

    print(f"✅ success                 : {success}")
    print(f"✅ elapsed_s               : {elapsed_s:.2f}")
    print(f"✅ pred_shape              : {pred_shape}")
    print(f"✅ pred_dtype              : {pred_dtype}")
    print(f"✅ smi_peak_memory_used_mb : {meta['smi_peak_memory_used_mb']}")
    print(f"✅ torch_peak_allocated_mb : {meta['torch_peak_allocated_mb_max']}")
    print(f"✅ torch_peak_reserved_mb  : {meta['torch_peak_reserved_mb_max']}")

    # 清理
    try:
        del masks, flows, styles
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# --------------------------------------------------
# 9) 导出 StepA summary
# --------------------------------------------------
stage("导出 StepA summary")
summary_df = pd.DataFrame(summary_rows)

summary_csv = (SUMMARY_ROOT / f"summary_stepA_2p5d_sweep_{TARGET_TAG}.csv").resolve()
summary_df.to_csv(summary_csv, index=False)

print("✅ summary_csv:", summary_csv)
print("\n📊 StepA 结果预览")
display(
    summary_df[
        [
            "run_name",
            "success",
            "elapsed_s",
            "min_size",
            "stitch_threshold",
            "flow_threshold",
            "smi_peak_memory_used_mb",
            "torch_peak_allocated_mb_max",
            "torch_peak_reserved_mb_max",
            "mask_path",
        ]
    ].sort_values(
        by=["success", "stitch_threshold", "min_size", "flow_threshold"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)
)

print("\n" + "=" * 100)
print("✅ 2.5D StepA 扫描完成")
print("OUT_EXP_DIR :", OUT_EXP_DIR)
print("GRID CSV    :", grid_csv)
print("SUMMARY CSV :", summary_csv)
print("=" * 100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# ==========================================
# Cell 3.19-2p5D-StepB：
# 读取 StepA 已保存的 2.5D raw masks，做统计分析与可视化
#
# 负责：
#   1) 读取 StepA summary csv
#   2) 逐组读取 raw mask
#   3) 计算 total_cells / median_volume / p90_volume / max_volume
#   4) 计算 large_obj_ratio_ge_2x_median
#   5) 输出每组固定模板预览图
#   6) 汇总结果并排序
#
# 不负责：
#   - 重新推理
#   - 改模型参数
# ==========================================

import gc
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

print("=" * 100)
print("📊 Cell 3.19-2p5D-StepB | Analyze saved 2.5D masks")
print("=" * 100)

# --------------------------------------------------
# 0) 路径：这里要与你刚才 StepA 的 OUT_EXP_DIR 对齐
# --------------------------------------------------
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
RUNS_ROOT = (ROOT / "runs").resolve()

# ✅ 这里务必改成你刚才 StepA 实际打印出来的 OUT_EXP_DIR
OUT_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260319_2p5d_sweep_请替换成你实际时间戳"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

STEPA_ROOT = (OUT_EXP_DIR / "stepA_2p5d_sweep").resolve()
MASK_ROOT = (STEPA_ROOT / TARGET_TAG / "masks").resolve()
LOG_ROOT = (STEPA_ROOT / TARGET_TAG / "logs").resolve()
META_ROOT = (STEPA_ROOT / TARGET_TAG / "meta").resolve()
SUMMARY_ROOT = (STEPA_ROOT / TARGET_TAG / "summary").resolve()

STEPB_ROOT = (OUT_EXP_DIR / "stepB_2p5d_analysis" / TARGET_TAG).resolve()
STAT_ROOT = (STEPB_ROOT / "stats").resolve()
FIG_ROOT = (STEPB_ROOT / "figures").resolve()
TABLE_ROOT = (STEPB_ROOT / "tables").resolve()

for p in [STEPB_ROOT, STAT_ROOT, FIG_ROOT, TABLE_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

STEPA_SUMMARY_CSV = (SUMMARY_ROOT / f"summary_stepA_2p5d_sweep_{TARGET_TAG}.csv").resolve()

print("\n📂 路径检查")
print("OUT_EXP_DIR       :", OUT_EXP_DIR)
print("STEPA_SUMMARY_CSV :", STEPA_SUMMARY_CSV)
print("RAW_3D_STACK_PATH :", RAW_3D_STACK_PATH)
print("STEPB_ROOT        :", STEPB_ROOT)
print("STAT_ROOT         :", STAT_ROOT)
print("FIG_ROOT          :", FIG_ROOT)
print("TABLE_ROOT        :", TABLE_ROOT)

assert OUT_EXP_DIR.exists(), f"❌ OUT_EXP_DIR 不存在: {OUT_EXP_DIR}"
assert STEPA_SUMMARY_CSV.exists(), f"❌ StepA summary csv 不存在: {STEPA_SUMMARY_CSV}"
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW_3D_STACK_PATH 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 1) 工具函数
# --------------------------------------------------
def stage(msg: str):
    print(f"\n🔹 {msg}")

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def percentile_safe(arr, q, default=np.nan):
    if arr is None or len(arr) == 0:
        return default
    return float(np.percentile(arr, q))

def choose_z_slices(nz: int):
    """
    固定挑几层用于做预览，避免每组图不一致
    """
    if nz <= 1:
        return [0]
    if nz <= 5:
        return list(range(nz))

    cand = sorted(set([
        0,
        nz // 4,
        nz // 2,
        (3 * nz) // 4,
        nz - 1
    ]))
    return [int(x) for x in cand]

def compute_object_volumes(mask_3d: np.ndarray):
    """
    输入 uint/int 标签 mask，返回：
      labels: 去掉背景后的标签 id
      volumes: 每个标签体素数
    """
    vals, counts = np.unique(mask_3d, return_counts=True)
    keep = vals != 0
    vals = vals[keep]
    counts = counts[keep]
    return vals.astype(np.int64), counts.astype(np.int64)

def topk_objects(labels, volumes, k=8):
    if len(labels) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(-volumes)
    order = order[:k]
    return labels[order], volumes[order]

def bbox3d_from_mask(mask_bool: np.ndarray):
    """
    返回一个 bool mask 的 3D bbox: z0,z1,y0,y1,x0,x1 (闭区间->切片时右边+1)
    """
    coords = np.argwhere(mask_bool)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)

def pad_bbox3d(bbox, shape, pad_z=2, pad_y=20, pad_x=20):
    z0, z1, y0, y1, x0, x1 = bbox
    nz, ny, nx = shape
    z0 = max(0, z0 - pad_z)
    z1 = min(nz - 1, z1 + pad_z)
    y0 = max(0, y0 - pad_y)
    y1 = min(ny - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(nx - 1, x1 + pad_x)
    return z0, z1, y0, y1, x0, x1

def make_overlay_rgb(gray2d, label2d, alpha=0.45):
    """
    做一个简单 overlay：
      - 背景用灰度
      - label 区域叠加红色
    """
    gray = gray2d.astype(np.float32)
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min())
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    rgb = np.stack([gray, gray, gray], axis=-1)

    mask = label2d > 0
    if np.any(mask):
        rgb[..., 0][mask] = (1 - alpha) * rgb[..., 0][mask] + alpha * 1.0
        rgb[..., 1][mask] = (1 - alpha) * rgb[..., 1][mask] + alpha * 0.1
        rgb[..., 2][mask] = (1 - alpha) * rgb[..., 2][mask] + alpha * 0.1

    return np.clip(rgb, 0, 1)

# --------------------------------------------------
# 2) 读取 StepA summary + raw stack
# --------------------------------------------------
stage("读取 StepA summary")
stepa_df = pd.read_csv(STEPA_SUMMARY_CSV)
print("✅ stepa_df.shape:", stepa_df.shape)
display(stepa_df.head())

stage("读取 raw 3D stack")
raw_3d = tiff.imread(str(RAW_3D_STACK_PATH))
print("✅ raw_3d.shape:", raw_3d.shape, raw_3d.dtype)

assert raw_3d.ndim == 3, f"❌ 预期 raw_3d 为 3D，实际拿到: {raw_3d.shape}"

# 只分析成功的组
stepa_df_ok = stepa_df[stepa_df["success"] == True].copy()
stepa_df_ok = stepa_df_ok.reset_index(drop=True)

print(f"✅ 成功组数: {len(stepa_df_ok)} / {len(stepa_df)}")
assert len(stepa_df_ok) > 0, "❌ 没有 success=True 的参数组，无法分析"

# --------------------------------------------------
# 3) 主循环：逐组分析
# --------------------------------------------------
stage("逐组分析 saved masks")
analysis_rows = []

for idx, row in stepa_df_ok.iterrows():
    run_name = row["run_name"]
    mask_path = Path(row["mask_path"]).resolve()
    meta_json = Path(row["meta_json"]).resolve() if "meta_json" in row else None
    log_json = Path(row["log_json"]).resolve() if "log_json" in row else None

    print("\n" + "-" * 100)
    print(f"[{idx + 1}/{len(stepa_df_ok)}] 📦 run_name = {run_name}")
    print("-" * 100)

    assert mask_path.exists(), f"❌ mask 不存在: {mask_path}"

    mask_3d = tiff.imread(str(mask_path))
    print("✅ mask_3d.shape:", mask_3d.shape, mask_3d.dtype)

    assert mask_3d.shape == raw_3d.shape, (
        f"❌ mask 与 raw shape 不一致: mask={mask_3d.shape}, raw={raw_3d.shape}"
    )

    labels, volumes = compute_object_volumes(mask_3d)

    total_cells = int(len(labels))
    sum_volume = int(volumes.sum()) if len(volumes) > 0 else 0
    mean_volume = float(volumes.mean()) if len(volumes) > 0 else np.nan
    median_volume = float(np.median(volumes)) if len(volumes) > 0 else np.nan
    p90_volume = percentile_safe(volumes, 90)
    max_volume = int(volumes.max()) if len(volumes) > 0 else 0
    min_volume_nonzero = int(volumes.min()) if len(volumes) > 0 else 0

    if len(volumes) > 0 and np.isfinite(median_volume) and median_volume > 0:
        large_ratio_ge_2x_median = float((volumes >= (2.0 * median_volume)).mean())
        large_ratio_ge_3x_median = float((volumes >= (3.0 * median_volume)).mean())
    else:
        large_ratio_ge_2x_median = np.nan
        large_ratio_ge_3x_median = np.nan

    # 单层占有情况：帮助看碎裂 / 过度连接
    nz = mask_3d.shape[0]
    nonzero_per_slice = np.array([(mask_3d[z] > 0).sum() for z in range(nz)], dtype=np.int64)
    n_labels_per_slice = np.array([len(np.unique(mask_3d[z])) - (1 if 0 in mask_3d[z] else 0) for z in range(nz)], dtype=np.int64)

    # 读取 StepA 里的 meta/log（如果有）
    meta = read_json(meta_json) if (meta_json is not None and meta_json.exists()) else {}
    glog = read_json(log_json) if (log_json is not None and log_json.exists()) else {}

    # --------------------------------------------------
    # 3.1 固定模板预览图
    # --------------------------------------------------
    fig_dir = (FIG_ROOT / run_name).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    z_slices = choose_z_slices(mask_3d.shape[0])

    # 图1：固定切片 overlay 预览
    fig1_path = (fig_dir / f"{run_name}__overlay_preview.png").resolve()

    ncols = len(z_slices)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, z in enumerate(z_slices):
        raw2d = raw_3d[z]
        mask2d = mask_3d[z]

        axes[0, j].imshow(raw2d, cmap="gray")
        axes[0, j].set_title(f"raw z={z}")
        axes[0, j].axis("off")

        overlay = make_overlay_rgb(raw2d, mask2d, alpha=0.45)
        axes[1, j].imshow(overlay)
        axes[1, j].set_title(f"overlay z={z}")
        axes[1, j].axis("off")

    plt.suptitle(
        f"{run_name}\n"
        f"cells={total_cells}, median={median_volume:.1f}, p90={p90_volume:.1f}, max={max_volume}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("✅ saved overlay preview:", fig1_path)

    # 图2：最大对象局部预览
    fig2_path = (fig_dir / f"{run_name}__largest_objects_preview.png").resolve()

    top_labels, top_volumes = topk_objects(labels, volumes, k=6)

    if len(top_labels) > 0:
        ncols = min(3, len(top_labels))
        nrows = int(math.ceil(len(top_labels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for ax in axes.ravel():
            ax.axis("off")

        for k, (lab, vol) in enumerate(zip(top_labels, top_volumes)):
            obj = (mask_3d == lab)
            bbox = bbox3d_from_mask(obj)
            bbox = pad_bbox3d(bbox, mask_3d.shape, pad_z=2, pad_y=20, pad_x=20)

            z0, z1, y0, y1, x0, x1 = bbox
            zc = (z0 + z1) // 2

            raw_crop = raw_3d[zc, y0:y1+1, x0:x1+1]
            obj_crop = obj[zc, y0:y1+1, x0:x1+1]

            overlay = make_overlay_rgb(raw_crop, obj_crop.astype(np.uint8), alpha=0.50)

            ax = axes.ravel()[k]
            ax.imshow(overlay)
            ax.set_title(
                f"label={int(lab)} | vol={int(vol)}\n"
                f"z=[{z0},{z1}] mid={zc}"
            )
            ax.axis("off")

        plt.suptitle(f"{run_name}\nLargest objects preview", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("✅ saved largest-objects preview:", fig2_path)
    else:
        fig2_path = None
        print("⚠️ no foreground objects, skip largest-object preview")

    # 图3：每层对象数 / 占有像素数
    fig3_path = (fig_dir / f"{run_name}__slice_profile.png").resolve()

    fig = plt.figure(figsize=(12, 5))
    plt.plot(n_labels_per_slice, label="n_labels_per_slice")
    plt.plot(nonzero_per_slice, label="nonzero_pixels_per_slice")
    plt.title(f"{run_name}\nSlice-wise profile")
    plt.xlabel("z slice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("✅ saved slice profile:", fig3_path)

    # --------------------------------------------------
    # 3.2 汇总记录
    # --------------------------------------------------
    analysis_row = {
        "run_name": run_name,
        "mask_path": str(mask_path),

        "diameter": row.get("diameter", np.nan),
        "cellprob_threshold": row.get("cellprob_threshold", np.nan),
        "min_size": row.get("min_size", np.nan),
        "stitch_threshold": row.get("stitch_threshold", np.nan),
        "flow_threshold": row.get("flow_threshold", np.nan),
        "batch_size": row.get("batch_size", np.nan),

        "elapsed_s_stepA": row.get("elapsed_s", np.nan),

        "total_cells": total_cells,
        "sum_volume": sum_volume,
        "mean_volume": mean_volume,
        "median_volume": median_volume,
        "p90_volume": p90_volume,
        "max_volume": max_volume,
        "min_volume_nonzero": min_volume_nonzero,
        "large_obj_ratio_ge_2x_median": large_ratio_ge_2x_median,
        "large_obj_ratio_ge_3x_median": large_ratio_ge_3x_median,

        "mean_nonzero_pixels_per_slice": float(nonzero_per_slice.mean()),
        "max_nonzero_pixels_per_slice": int(nonzero_per_slice.max()),
        "mean_n_labels_per_slice": float(n_labels_per_slice.mean()),
        "max_n_labels_per_slice": int(n_labels_per_slice.max()),

        # StepA 已记录的显存摘要（如果有）
        "smi_peak_memory_used_mb": row.get("smi_peak_memory_used_mb", np.nan),
        "torch_peak_allocated_mb_max": row.get("torch_peak_allocated_mb_max", np.nan),
        "torch_peak_reserved_mb_max": row.get("torch_peak_reserved_mb_max", np.nan),

        "overlay_preview_path": str(fig1_path),
        "largest_objects_preview_path": None if fig2_path is None else str(fig2_path),
        "slice_profile_path": str(fig3_path),
    }

    # 保存单组 stats json
    stat_json = (STAT_ROOT / f"{run_name}__stats.json").resolve()
    stat_json.write_text(json.dumps(analysis_row, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_row["stat_json"] = str(stat_json)

    analysis_rows.append(analysis_row)

    # 清理
    del mask_3d
    gc.collect()

# --------------------------------------------------
# 4) 汇总总表 + 排序
# --------------------------------------------------
stage("汇总分析结果")
analysis_df = pd.DataFrame(analysis_rows)

raw_table_csv = (TABLE_ROOT / f"analysis_2p5d_raw_{TARGET_TAG}.csv").resolve()
analysis_df.to_csv(raw_table_csv, index=False)

print("✅ raw analysis table:", raw_table_csv)

# --------------------------------------------------
# 5) 生成几个常用排序表
# --------------------------------------------------
# 你现在还没有 GT，所以这里不做 Dice/IoU 排序
# 先按“更少巨物粘连 + 不至于总数太离谱 + 显存可接受”来给一个粗排视角

rank_df = analysis_df.copy()

# 一个简单的启发式综合分：
# - large object ratio 越低越好
# - max_volume 越低越好（避免巨型怪物）
# - total_cells 先不直接强行最小/最大，而是用 log 缩放轻度参与
eps = 1e-8
rank_df["score_large2"] = rank_df["large_obj_ratio_ge_2x_median"].fillna(1.0)
rank_df["score_large3"] = rank_df["large_obj_ratio_ge_3x_median"].fillna(1.0)
rank_df["score_maxvol"] = np.log1p(rank_df["max_volume"].fillna(1e9))
rank_df["score_totalcells"] = -np.log1p(rank_df["total_cells"].fillna(0))

rank_df["heuristic_score"] = (
    2.0 * rank_df["score_large2"] +
    2.5 * rank_df["score_large3"] +
    0.5 * rank_df["score_maxvol"] +
    0.1 * rank_df["score_totalcells"]
)

rank_by_heuristic = rank_df.sort_values(
    by=["heuristic_score", "large_obj_ratio_ge_2x_median", "max_volume", "torch_peak_allocated_mb_max"],
    ascending=[True, True, True, True]
).reset_index(drop=True)

rank_by_mem = rank_df.sort_values(
    by=["torch_peak_allocated_mb_max", "torch_peak_reserved_mb_max", "elapsed_s_stepA"],
    ascending=[True, True, True]
).reset_index(drop=True)

rank_by_cells = rank_df.sort_values(
    by=["total_cells", "median_volume"],
    ascending=[False, True]
).reset_index(drop=True)

rank_heuristic_csv = (TABLE_ROOT / f"rank_2p5d_heuristic_{TARGET_TAG}.csv").resolve()
rank_mem_csv = (TABLE_ROOT / f"rank_2p5d_memory_{TARGET_TAG}.csv").resolve()
rank_cells_csv = (TABLE_ROOT / f"rank_2p5d_cells_{TARGET_TAG}.csv").resolve()

rank_by_heuristic.to_csv(rank_heuristic_csv, index=False)
rank_by_mem.to_csv(rank_mem_csv, index=False)
rank_by_cells.to_csv(rank_cells_csv, index=False)

print("✅ rank_heuristic_csv:", rank_heuristic_csv)
print("✅ rank_mem_csv      :", rank_mem_csv)
print("✅ rank_cells_csv    :", rank_cells_csv)

# --------------------------------------------------
# 6) 屏幕预览
# --------------------------------------------------
stage("结果预览：heuristic 排序 Top")
show_cols = [
    "run_name",
    "min_size",
    "stitch_threshold",
    "flow_threshold",
    "total_cells",
    "median_volume",
    "p90_volume",
    "max_volume",
    "large_obj_ratio_ge_2x_median",
    "large_obj_ratio_ge_3x_median",
    "elapsed_s_stepA",
    "torch_peak_allocated_mb_max",
    "torch_peak_reserved_mb_max",
    "heuristic_score",
]

display(rank_by_heuristic[show_cols].head(15))

stage("结果预览：heuristic 排序 Top（含显存）")
show_cols = [
    "run_name",
    "min_size",
    "stitch_threshold",
    "flow_threshold",
    "total_cells",
    "median_volume",
    "p90_volume",
    "max_volume",
    "large_obj_ratio_ge_2x_median",
    "large_obj_ratio_ge_3x_median",
    "elapsed_s_stepA",
    "smi_peak_memory_used_mb",
    "torch_peak_allocated_mb_max",
    "torch_peak_reserved_mb_max",
    "heuristic_score",
]
display(rank_by_heuristic[show_cols].head(20))

stage("结果预览：按显存从小到大排序")
mem_cols = [
    "run_name",
    "min_size",
    "stitch_threshold",
    "flow_threshold",
    "elapsed_s_stepA",
    "smi_peak_memory_used_mb",
    "torch_peak_allocated_mb_max",
    "torch_peak_reserved_mb_max",
    "total_cells",
    "median_volume",
    "max_volume",
    "large_obj_ratio_ge_2x_median",
]
display(rank_by_mem[mem_cols].head(20))


# In[ ]:


# ==========================================
# Cell 3.19-2p5D-StepC：
# 从 StepB 结果里自动提取 Top-N 候选，生成总览拼图
#
# 负责：
#   1) 读取 StepB 排序表
#   2) 选择 Top-N 候选
#   3) 汇总每组 overlay / largest-object / slice-profile
#   4) 导出候选清单 csv
#   5) 生成总览拼图，方便人工快速挑参数
# ==========================================
from IPython.display import display

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("=" * 100)
print("🧩 Cell 3.19-2p5D-StepC | Make top-N candidate overview")
print("=" * 100)

# --------------------------------------------------
# 0) 路径：改成你的真实 3.19 OUT_EXP_DIR
# --------------------------------------------------
from IPython.display import display

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
RUNS_ROOT = (ROOT / "runs").resolve()

OUT_EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_2026320_25D_test/runs"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

STEPA_ROOT = (OUT_EXP_DIR / "stepA_2p5d_sweep").resolve()
MASK_ROOT = (STEPA_ROOT / TARGET_TAG / "masks").resolve()
LOG_ROOT = (STEPA_ROOT / TARGET_TAG / "logs").resolve()
META_ROOT = (STEPA_ROOT / TARGET_TAG / "meta").resolve()
SUMMARY_ROOT = (STEPA_ROOT / TARGET_TAG / "summary").resolve()

STEPB_ROOT = (OUT_EXP_DIR / "stepB_2p5d_analysis" / TARGET_TAG).resolve()
STAT_ROOT = (STEPB_ROOT / "stats").resolve()
FIG_ROOT = (STEPB_ROOT / "figures").resolve()
TABLE_ROOT = (STEPB_ROOT / "tables").resolve()

for p in [STEPC_ROOT, OVERVIEW_ROOT, TABLE_OUT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1) 读取 StepB 排序表
# --------------------------------------------------
RANK_CSV = (TABLE_ROOT / f"rank_2p5d_heuristic_{TARGET_TAG}.csv").resolve()

assert OUT_EXP_DIR.exists(), f"❌ OUT_EXP_DIR 不存在: {OUT_EXP_DIR}"
assert RANK_CSV.exists(), f"❌ rank csv 不存在: {RANK_CSV}"

rank_df = pd.read_csv(RANK_CSV)
print("✅ rank_df.shape:", rank_df.shape)

# --------------------------------------------------
# 2) 取 Top-N 候选
# --------------------------------------------------
TOP_N = 8
rank_top = rank_df.head(TOP_N).copy().reset_index(drop=True)

top_csv = (TABLE_OUT_ROOT / f"top{TOP_N}_2p5d_candidates_{TARGET_TAG}.csv").resolve()
rank_top.to_csv(top_csv, index=False)
print("✅ top_csv:", top_csv)

display(
    rank_top[
        [
            "run_name",
            "min_size",
            "stitch_threshold",
            "flow_threshold",
            "total_cells",
            "median_volume",
            "max_volume",
            "large_obj_ratio_ge_2x_median",
            "smi_peak_memory_used_mb",
            "torch_peak_allocated_mb_max",
            "torch_peak_reserved_mb_max",
            "heuristic_score",
        ]
    ]
)

# --------------------------------------------------
# 3) 工具函数
# --------------------------------------------------
def read_img_or_none(path_str):
    if pd.isna(path_str):
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return mpimg.imread(str(p))
    except Exception:
        return None

# --------------------------------------------------
# 4) 生成总览拼图
# 每一行一组参数：
#   col1 = overlay preview
#   col2 = largest objects preview
#   col3 = slice profile
# --------------------------------------------------
nrows = len(rank_top)
ncols = 3

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
if nrows == 1:
    axes = np.array(axes).reshape(1, ncols)

for i, row in rank_top.iterrows():
    run_name = row["run_name"]

    overlay_path = row.get("overlay_preview_path", None)
    largest_path = row.get("largest_objects_preview_path", None)
    profile_path = row.get("slice_profile_path", None)

    overlay_img = read_img_or_none(overlay_path)
    largest_img = read_img_or_none(largest_path)
    profile_img = read_img_or_none(profile_path)

    imgs = [overlay_img, largest_img, profile_img]
    titles = [
        "overlay preview",
        "largest objects",
        "slice profile",
    ]

    desc = (
        f"{run_name}\n"
        f"ms={row['min_size']}, st={row['stitch_threshold']}, ft={row['flow_threshold']}\n"
        f"cells={row['total_cells']}, med={row['median_volume']:.1f}, max={row['max_volume']}\n"
        f"large2={row['large_obj_ratio_ge_2x_median']:.4f}, "
        f"torch_alloc={row['torch_peak_allocated_mb_max']:.1f} MB, "
        f"smi={row['smi_peak_memory_used_mb']:.1f} MB"
    )

    for j in range(ncols):
        ax = axes[i, j]
        img = imgs[j]
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=12)
        ax.set_title(titles[j], fontsize=11)
        ax.axis("off")

    # 把这一行的大标题写在左侧第一张图上方
    axes[i, 0].text(
        0.0, 1.15, desc,
        transform=axes[i, 0].transAxes,
        fontsize=11,
        va="bottom",
        ha="left"
    )

plt.tight_layout()

overview_png = (OVERVIEW_ROOT / f"top{TOP_N}_overview_{TARGET_TAG}.png").resolve()
plt.savefig(overview_png, dpi=180, bbox_inches="tight")
plt.close(fig)

print("✅ overview_png:", overview_png)

# --------------------------------------------------
# 5) 额外导出一个“显存友好 Top-N”表
# --------------------------------------------------
mem_top = rank_df.sort_values(
    by=[
        "torch_peak_allocated_mb_max",
        "torch_peak_reserved_mb_max",
        "heuristic_score",
    ],
    ascending=[True, True, True]
).head(TOP_N).copy().reset_index(drop=True)

mem_top_csv = (TABLE_OUT_ROOT / f"top{TOP_N}_memory_friendly_{TARGET_TAG}.csv").resolve()
mem_top.to_csv(mem_top_csv, index=False)
print("✅ mem_top_csv:", mem_top_csv)

display(
    mem_top[
        [
            "run_name",
            "min_size",
            "stitch_threshold",
            "flow_threshold",
            "total_cells",
            "median_volume",
            "max_volume",
            "large_obj_ratio_ge_2x_median",
            "smi_peak_memory_used_mb",
            "torch_peak_allocated_mb_max",
            "torch_peak_reserved_mb_max",
            "heuristic_score",
        ]
    ]
)

print("\n" + "=" * 100)
print("✅ StepC 完成")
print("TOP CSV      :", top_csv)
print("OVERVIEW PNG :", overview_png)
print("MEM TOP CSV  :", mem_top_csv)
print("=" * 100)


# In[ ]:





# In[ ]:





# In[ ]:




