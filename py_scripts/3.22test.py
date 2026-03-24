#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================
# StepA-Post-ParallelAligned | 对 full-brain 3D raw mask 做“逐 2D 填坑”
# 并行对齐版：
#   1) 与朴素版保持同一输入/输出目录风格
#   2) 仍然是“逐实例、逐 z 切片”的 2D binary_fill_holes
#   3) 不再跳过小实例（与朴素版对齐）
#   4) 使用 bbox + 多进程加速，尽量保持结果一致
#   5) 新建单独子文件夹保存，避免覆盖旧版本
#
# 说明：
#   - 算法意图与第一个脚本一致：不会重新分割，不会新增实例 ID
#   - 正常情况下不会改变细胞计数，只会填补实例内部封闭空洞
#   - 相比朴素版，主要优化在于：不再每个实例全图扫描，而是先建 label->coords
# ==========================================

import os
import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes

print("=" * 100)
print("🩹 StepA-Post-ParallelAligned | Fill holes slice-by-slice (2D) for 3D instance masks")
print("=" * 100)

# --------------------------------------------------
# 0) 路径配置：已对齐到第一个脚本的真实路径
# --------------------------------------------------
RAW_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/"
    "raw_masks/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif"
).resolve()

META_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/"
    "stats/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__stepA_meta.json"
).resolve()

# 新版本单独存到新子文件夹，避免和第一个脚本的 post_fillholes_2d 混在一起
OUT_ROOT = RAW_MASK_PATH.parent.parent / "post_fillholes_2d_parallel_aligned_v1"
OUT_MASK_DIR = OUT_ROOT / "filled_masks"
OUT_STAT_DIR = OUT_ROOT / "stats"

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_STAT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MASK_PATH = OUT_MASK_DIR / RAW_MASK_PATH.name.replace(
    "_masks_raw.tif",
    "_masks_filled2d_parallel_aligned.tif"
)

OUT_META_PATH = OUT_STAT_DIR / META_PATH.name.replace(
    "__stepA_meta.json",
    "__fillholes2d_parallel_aligned_meta.json"
)

# --------------------------------------------------
# 1) 可调参数
# --------------------------------------------------
# 与第一个脚本对齐的核心点：不跳过小实例
MIN_VOXELS_SKIP_FILL = 0

# 并行参数：先保守一点，别一上来开太猛
NUM_WORKERS = min(8, os.cpu_count() or 4)
CHUNK_SIZE = 200

# 是否强制用 uint32 保存；默认保持输入 dtype，与第一个脚本一致
SAVE_UINT32 = False

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
def chunk_list(xs, chunk_size):
    for i in range(0, len(xs), chunk_size):
        yield xs[i:i + chunk_size]


def build_label_coords(mask3d: np.ndarray):
    """
    一次性扫描所有非零体素，构造：
        label -> list[(z,y,x), ...]
    避免像第一个脚本那样每个实例都全图做 `masks == obj_id`
    """
    coords = np.argwhere(mask3d > 0)
    label_to_coords = defaultdict(list)

    for z, y, x in coords:
        lab = int(mask3d[z, y, x])
        label_to_coords[lab].append((int(z), int(y), int(x)))

    return label_to_coords


def process_one_instance(obj_id: int, coords_list, min_voxels_skip_fill=0):
    """
    对单个实例做：
      - bbox 裁剪
      - 逐 z 的 binary_fill_holes

    返回：
      {
        obj_id,
        bbox,
        filled_local_coords,
        stats...
      }
    """
    n_vox = len(coords_list)
    if n_vox == 0:
        return None

    coords = np.asarray(coords_list, dtype=np.int32)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    dz = int(z1 - z0 + 1)
    dy = int(y1 - y0 + 1)
    dx = int(x1 - x0 + 1)

    sub_mask = np.zeros((dz, dy, dx), dtype=bool)
    zz = coords[:, 0] - z0
    yy = coords[:, 1] - y0
    xx = coords[:, 2] - x0
    sub_mask[zz, yy, xx] = True

    # 与朴素版对齐：默认所有实例都做 fill
    if n_vox < min_voxels_skip_fill:
        filled_local = sub_mask
        obj_added_voxels = 0
        obj_slices_touched = 0
    else:
        filled_local = np.zeros_like(sub_mask, dtype=bool)
        obj_added_voxels = 0
        obj_slices_touched = 0

        for z_local in range(dz):
            sl = sub_mask[z_local]
            if not sl.any():
                continue

            sl_filled = binary_fill_holes(sl)
            added = int(sl_filled.sum() - sl.sum())

            if added > 0:
                obj_added_voxels += added
                obj_slices_touched += 1

            filled_local[z_local] = sl_filled

    filled_local_coords = np.argwhere(filled_local)

    return {
        "obj_id": int(obj_id),
        "z0": int(z0), "z1": int(z1),
        "y0": int(y0), "y1": int(y1),
        "x0": int(x0), "x1": int(x1),
        "dz": int(dz), "dy": int(dy), "dx": int(dx),
        "n_vox_raw": int(n_vox),
        "n_vox_filled": int(filled_local.sum()),
        "added_voxels": int(obj_added_voxels),
        "slices_touched": int(obj_slices_touched),
        "changed": bool(obj_added_voxels > 0),
        "filled_local_coords": filled_local_coords.astype(np.int32),
    }


def worker_process(batch_items, min_voxels_skip_fill):
    """
    一个 worker 处理一批实例。
    batch_items: [(obj_id, coords_list), ...]
    """
    results = []
    batch_changed_instances = 0
    batch_added_voxels = 0
    batch_slices_touched = 0

    for obj_id, coords_list in batch_items:
        out = process_one_instance(
            obj_id=obj_id,
            coords_list=coords_list,
            min_voxels_skip_fill=min_voxels_skip_fill,
        )
        if out is None:
            continue

        results.append(out)

        if out["changed"]:
            batch_changed_instances += 1
            batch_added_voxels += out["added_voxels"]
            batch_slices_touched += out["slices_touched"]

    return {
        "results": results,
        "batch_changed_instances": int(batch_changed_instances),
        "batch_added_voxels": int(batch_added_voxels),
        "batch_slices_touched": int(batch_slices_touched),
    }


# --------------------------------------------------
# 3) 读取 raw mask
# --------------------------------------------------
print("\n[1/6] 读取 raw mask")
if not RAW_MASK_PATH.exists():
    raise FileNotFoundError(f"找不到 RAW_MASK_PATH: {RAW_MASK_PATH}")

masks = tiff.imread(str(RAW_MASK_PATH))
masks = np.asarray(masks)

print(f"raw mask path   : {RAW_MASK_PATH}")
print(f"raw mask shape  : {masks.shape}")
print(f"raw mask dtype  : {masks.dtype}")
print(f"raw mask min/max: {masks.min()} / {masks.max()}")

if masks.ndim != 3:
    raise ValueError(f"期望 3D mask，实际 ndim={masks.ndim}，shape={masks.shape}")

mask_shape = tuple(int(v) for v in masks.shape)

# --------------------------------------------------
# 4) 收集实例 ID（并建立 label -> coords）
# --------------------------------------------------
print("\n[2/6] 收集实例 ID + 构建 label -> coords 映射")
t0 = time.time()
t_build = time.time()

label_to_coords = build_label_coords(masks)
obj_ids = sorted(label_to_coords.keys())
num_instances_raw = int(len(obj_ids))

print(f"num_instances_raw = {num_instances_raw}")
print(f"build_label_coords elapsed = {time.time() - t_build:.2f}s")

# --------------------------------------------------
# 5) 逐 2D 填坑（并行版）
# --------------------------------------------------
print("\n[3/6] 逐实例、逐 z 切片填坑（并行加速）")
print(f"NUM_WORKERS          = {NUM_WORKERS}")
print(f"CHUNK_SIZE           = {CHUNK_SIZE}")
print(f"MIN_VOXELS_SKIP_FILL = {MIN_VOXELS_SKIP_FILL}")

items = [(obj_id, label_to_coords[obj_id]) for obj_id in obj_ids]
batches = list(chunk_list(items, CHUNK_SIZE))
num_batches = len(batches)

if SAVE_UINT32:
    filled = np.zeros(mask_shape, dtype=np.uint32)
else:
    filled = np.zeros_like(masks, dtype=masks.dtype)

num_changed_instances = 0
num_total_added_voxels = 0
num_slices_touched = 0
processed_instances = 0
largest_added_cases = []

t_parallel = time.time()

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = [
        ex.submit(worker_process, batch, MIN_VOXELS_SKIP_FILL)
        for batch in batches
    ]

    for fut_idx, fut in enumerate(as_completed(futures), start=1):
        batch_out = fut.result()

        num_changed_instances += batch_out["batch_changed_instances"]
        num_total_added_voxels += batch_out["batch_added_voxels"]
        num_slices_touched += batch_out["batch_slices_touched"]

        results = batch_out["results"]
        for out in results:
            obj_id = out["obj_id"]
            z0, y0, x0 = out["z0"], out["y0"], out["x0"]
            local_coords = out["filled_local_coords"]

            if local_coords.size > 0:
                zz = local_coords[:, 0] + z0
                yy = local_coords[:, 1] + y0
                xx = local_coords[:, 2] + x0
                filled[zz, yy, xx] = obj_id

            processed_instances += 1

            if out["added_voxels"] > 0:
                largest_added_cases.append({
                    "obj_id": int(obj_id),
                    "added_voxels": int(out["added_voxels"]),
                    "n_vox_raw": int(out["n_vox_raw"]),
                    "n_vox_filled": int(out["n_vox_filled"]),
                    "slices_touched": int(out["slices_touched"]),
                    "bbox": {
                        "z0": int(out["z0"]), "z1": int(out["z1"]),
                        "y0": int(out["y0"]), "y1": int(out["y1"]),
                        "x0": int(out["x0"]), "x1": int(out["x1"]),
                        "dz": int(out["dz"]), "dy": int(out["dy"]), "dx": int(out["dx"]),
                    }
                })

        if fut_idx % 10 == 0 or fut_idx == num_batches:
            elapsed_now = time.time() - t_parallel
            print(
                f"[progress] batch {fut_idx:>4d}/{num_batches} | "
                f"processed_instances={processed_instances}/{num_instances_raw} | "
                f"changed_instances={num_changed_instances} | "
                f"added_voxels={num_total_added_voxels} | "
                f"elapsed={elapsed_now:.1f}s"
            )

parallel_elapsed = time.time() - t_parallel
elapsed = time.time() - t0

# --------------------------------------------------
# 6) 保存 filled mask + meta
# --------------------------------------------------
print("\n[4/6] 保存 filled mask")
tiff.imwrite(str(OUT_MASK_PATH), filled)
print(f"✅ saved filled mask : {OUT_MASK_PATH}")

print("\n[5/6] 保存 meta json")
filled_ids = np.unique(filled)
num_instances_filled = int(len(filled_ids) - (1 if 0 in filled_ids else 0))

largest_added_cases = sorted(
    largest_added_cases,
    key=lambda x: x["added_voxels"],
    reverse=True
)
top20_largest_added = largest_added_cases[:20]

meta = {
    "raw_mask_path": str(RAW_MASK_PATH),
    "input_meta_path": str(META_PATH),
    "output_mask_path": str(OUT_MASK_PATH),
    "shape": list(map(int, masks.shape)),
    "dtype_in": str(masks.dtype),
    "dtype_out": str(filled.dtype),
    "num_instances_raw": int(num_instances_raw),
    "num_instances_filled": int(num_instances_filled),
    "num_changed_instances": int(num_changed_instances),
    "num_total_added_voxels": int(num_total_added_voxels),
    "num_slices_touched": int(num_slices_touched),
    "elapsed_sec_total": round(elapsed, 4),
    "elapsed_sec_parallel_only": round(parallel_elapsed, 4),
    "method": "parallel-aligned bbox slice-wise 2D binary_fill_holes on each instance",
    "note": (
        "这是与朴素版对齐的并行版本：逐实例、逐z切片做2D填坑；"
        "默认不跳过小实例，尽量与第一个脚本保持结果一致。"
    ),
    "num_workers": int(NUM_WORKERS),
    "chunk_size": int(CHUNK_SIZE),
    "min_voxels_skip_fill": int(MIN_VOXELS_SKIP_FILL),
    "top20_largest_added_cases": top20_largest_added,
}

with open(OUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ saved meta        : {OUT_META_PATH}")

print("\n[6/6] DONE")
print(f"✅ num_instances_raw    : {num_instances_raw}")
print(f"✅ num_instances_filled : {num_instances_filled}")
print(f"✅ changed_instances    : {num_changed_instances}")
print(f"✅ total_added_voxels   : {num_total_added_voxels}")
print(f"✅ slices_touched       : {num_slices_touched}")
print(f"✅ parallel elapsed     : {parallel_elapsed:.2f}s")
print(f"✅ total elapsed        : {elapsed:.2f}s")
print("=" * 100)


# In[2]:


# ==========================================
# StepA-Post-ParallelAligned-v2 | 对 full-brain 3D raw mask 做“逐 2D 修补小孔”
#
# 这版相对上一版的变化：
#   1) 不再只做 binary_fill_holes
#   2) 改为：逐实例、逐 z 切片
#        binary_closing(3x3) -> remove_small_holes(area_threshold=16)
#   3) 如果环境里没有 scikit-image，则自动回退为：
#        binary_closing(3x3) -> binary_fill_holes
#   4) 仍保留 bbox + 多进程加速
#   5) 输出到新的子文件夹，且 masks 文件名与上一版区分开
#
# 说明：
#   - 不重新分割，不新增实例 ID
#   - 目标是修“边缘细小裂缝 + 小孔”，比单纯 fill_holes 更强一些
#   - 默认仍然不跳过小实例，尽量保持和朴素版一致的处理范围
# ==========================================

import os
import json
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes, binary_closing

# --------------------------------------------------
# 可选：优先用 skimage 的 remove_small_holes
# 如果没有，就自动回退到 binary_fill_holes
# --------------------------------------------------
try:
    from skimage.morphology import remove_small_holes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    remove_small_holes = None

print("=" * 100)
print("🩹 StepA-Post-ParallelAligned-v2 | closing + small-hole filling slice-by-slice (2D)")
print("=" * 100)

# --------------------------------------------------
# 0) 路径配置：已对齐到你的真实路径
# --------------------------------------------------
RAW_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/"
    "raw_masks/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif"
).resolve()

META_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
    "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
    "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/"
    "stats/"
    "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__stepA_meta.json"
).resolve()

# 新版本单独存目录，避免和上一版 post_fillholes_2d_parallel_aligned_v1 混在一起
OUT_ROOT = RAW_MASK_PATH.parent.parent / "post_fillholes_2d_parallel_aligned_v2"
OUT_MASK_DIR = OUT_ROOT / "filled_masks"
OUT_STAT_DIR = OUT_ROOT / "stats"

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_STAT_DIR.mkdir(parents=True, exist_ok=True)

# 注意这里故意改名字，和上一版 _filled2d_parallel_aligned 区分开
OUT_MASK_PATH = OUT_MASK_DIR / RAW_MASK_PATH.name.replace(
    "_masks_raw.tif",
    "_masks_filled2d_parallel_aligned_closing_smallholes.tif"
)

OUT_META_PATH = OUT_STAT_DIR / META_PATH.name.replace(
    "__stepA_meta.json",
    "__fillholes2d_parallel_aligned_closing_smallholes_meta.json"
)

# --------------------------------------------------
# 1) 可调参数
# --------------------------------------------------
# 与第一版对齐：默认不跳过小实例
MIN_VOXELS_SKIP_FILL = 0

# 并行参数
NUM_WORKERS = min(8, os.cpu_count() or 4)
CHUNK_SIZE = 200

# 保存 dtype：默认跟输入保持一致
SAVE_UINT32 = False

# 修补策略参数
CLOSING_KERNEL = np.ones((3, 3), dtype=bool)   # 轻微闭运算，先补边缘裂缝
SMALL_HOLE_AREA_THRESHOLD = 16                 # 只填小孔，先从 16 开始

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
def chunk_list(xs, chunk_size):
    for i in range(0, len(xs), chunk_size):
        yield xs[i:i + chunk_size]


def build_label_coords(mask3d: np.ndarray):
    """
    一次性扫描所有非零体素，构造：
        label -> list[(z,y,x), ...]
    避免每个实例都全图 masks == obj_id
    """
    coords = np.argwhere(mask3d > 0)
    label_to_coords = defaultdict(list)

    for z, y, x in coords:
        lab = int(mask3d[z, y, x])
        label_to_coords[lab].append((int(z), int(y), int(x)))

    return label_to_coords


def fill_one_slice(sl: np.ndarray):
    """
    单层修补逻辑：
      1) binary_closing: 先补很细的小裂缝
      2) remove_small_holes(area_threshold=...)
         如果没有 skimage，则回退到 binary_fill_holes
    """
    sl_closed = binary_closing(sl, structure=CLOSING_KERNEL)

    if HAS_SKIMAGE:
        sl_filled = remove_small_holes(
            sl_closed,
            area_threshold=SMALL_HOLE_AREA_THRESHOLD
        )
    else:
        sl_filled = binary_fill_holes(sl_closed)

    return sl_filled.astype(bool)


def process_one_instance(obj_id: int, coords_list, min_voxels_skip_fill=0):
    """
    对单个实例做：
      - bbox 裁剪
      - 逐 z 修补：closing + small holes filling

    返回：
      {
        obj_id,
        bbox,
        filled_local_coords,
        stats...
      }
    """
    n_vox = len(coords_list)
    if n_vox == 0:
        return None

    coords = np.asarray(coords_list, dtype=np.int32)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    dz = int(z1 - z0 + 1)
    dy = int(y1 - y0 + 1)
    dx = int(x1 - x0 + 1)

    sub_mask = np.zeros((dz, dy, dx), dtype=bool)
    zz = coords[:, 0] - z0
    yy = coords[:, 1] - y0
    xx = coords[:, 2] - x0
    sub_mask[zz, yy, xx] = True

    if n_vox < min_voxels_skip_fill:
        filled_local = sub_mask
        obj_added_voxels = 0
        obj_slices_touched = 0
    else:
        filled_local = np.zeros_like(sub_mask, dtype=bool)
        obj_added_voxels = 0
        obj_slices_touched = 0

        for z_local in range(dz):
            sl = sub_mask[z_local]
            if not sl.any():
                continue

            sl_fixed = fill_one_slice(sl)
            added = int(sl_fixed.sum() - sl.sum())

            if added > 0:
                obj_added_voxels += added
                obj_slices_touched += 1

            filled_local[z_local] = sl_fixed

    filled_local_coords = np.argwhere(filled_local)

    return {
        "obj_id": int(obj_id),
        "z0": int(z0), "z1": int(z1),
        "y0": int(y0), "y1": int(y1),
        "x0": int(x0), "x1": int(x1),
        "dz": int(dz), "dy": int(dy), "dx": int(dx),
        "n_vox_raw": int(n_vox),
        "n_vox_filled": int(filled_local.sum()),
        "added_voxels": int(obj_added_voxels),
        "slices_touched": int(obj_slices_touched),
        "changed": bool(obj_added_voxels > 0),
        "filled_local_coords": filled_local_coords.astype(np.int32),
    }


def worker_process(batch_items, min_voxels_skip_fill):
    """
    一个 worker 处理一批实例。
    batch_items: [(obj_id, coords_list), ...]
    """
    results = []
    batch_changed_instances = 0
    batch_added_voxels = 0
    batch_slices_touched = 0

    for obj_id, coords_list in batch_items:
        out = process_one_instance(
            obj_id=obj_id,
            coords_list=coords_list,
            min_voxels_skip_fill=min_voxels_skip_fill,
        )
        if out is None:
            continue

        results.append(out)

        if out["changed"]:
            batch_changed_instances += 1
            batch_added_voxels += out["added_voxels"]
            batch_slices_touched += out["slices_touched"]

    return {
        "results": results,
        "batch_changed_instances": int(batch_changed_instances),
        "batch_added_voxels": int(batch_added_voxels),
        "batch_slices_touched": int(batch_slices_touched),
    }


# --------------------------------------------------
# 3) 读取 raw mask
# --------------------------------------------------
print("\n[1/6] 读取 raw mask")
if not RAW_MASK_PATH.exists():
    raise FileNotFoundError(f"找不到 RAW_MASK_PATH: {RAW_MASK_PATH}")

masks = tiff.imread(str(RAW_MASK_PATH))
masks = np.asarray(masks)

print(f"raw mask path   : {RAW_MASK_PATH}")
print(f"raw mask shape  : {masks.shape}")
print(f"raw mask dtype  : {masks.dtype}")
print(f"raw mask min/max: {masks.min()} / {masks.max()}")

if masks.ndim != 3:
    raise ValueError(f"期望 3D mask，实际 ndim={masks.ndim}，shape={masks.shape}")

mask_shape = tuple(int(v) for v in masks.shape)

# --------------------------------------------------
# 4) 收集实例 ID（并建立 label -> coords）
# --------------------------------------------------
print("\n[2/6] 收集实例 ID + 构建 label -> coords 映射")
t0 = time.time()
t_build = time.time()

label_to_coords = build_label_coords(masks)
obj_ids = sorted(label_to_coords.keys())
num_instances_raw = int(len(obj_ids))

print(f"num_instances_raw = {num_instances_raw}")
print(f"build_label_coords elapsed = {time.time() - t_build:.2f}s")

# --------------------------------------------------
# 5) 逐 2D 修补（并行版）
# --------------------------------------------------
print("\n[3/6] 逐实例、逐 z 切片修补（并行加速）")
print(f"NUM_WORKERS                = {NUM_WORKERS}")
print(f"CHUNK_SIZE                 = {CHUNK_SIZE}")
print(f"MIN_VOXELS_SKIP_FILL       = {MIN_VOXELS_SKIP_FILL}")
print(f"HAS_SKIMAGE                = {HAS_SKIMAGE}")
print(f"SMALL_HOLE_AREA_THRESHOLD  = {SMALL_HOLE_AREA_THRESHOLD}")
print(f"CLOSING_KERNEL shape       = {CLOSING_KERNEL.shape}")

items = [(obj_id, label_to_coords[obj_id]) for obj_id in obj_ids]
batches = list(chunk_list(items, CHUNK_SIZE))
num_batches = len(batches)

if SAVE_UINT32:
    filled = np.zeros(mask_shape, dtype=np.uint32)
else:
    filled = np.zeros_like(masks, dtype=masks.dtype)

num_changed_instances = 0
num_total_added_voxels = 0
num_slices_touched = 0
processed_instances = 0
largest_added_cases = []

t_parallel = time.time()

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = [
        ex.submit(worker_process, batch, MIN_VOXELS_SKIP_FILL)
        for batch in batches
    ]

    for fut_idx, fut in enumerate(as_completed(futures), start=1):
        batch_out = fut.result()

        num_changed_instances += batch_out["batch_changed_instances"]
        num_total_added_voxels += batch_out["batch_added_voxels"]
        num_slices_touched += batch_out["batch_slices_touched"]

        results = batch_out["results"]
        for out in results:
            obj_id = out["obj_id"]
            z0, y0, x0 = out["z0"], out["y0"], out["x0"]
            local_coords = out["filled_local_coords"]

            if local_coords.size > 0:
                zz = local_coords[:, 0] + z0
                yy = local_coords[:, 1] + y0
                xx = local_coords[:, 2] + x0
                filled[zz, yy, xx] = obj_id

            processed_instances += 1

            if out["added_voxels"] > 0:
                largest_added_cases.append({
                    "obj_id": int(obj_id),
                    "added_voxels": int(out["added_voxels"]),
                    "n_vox_raw": int(out["n_vox_raw"]),
                    "n_vox_filled": int(out["n_vox_filled"]),
                    "slices_touched": int(out["slices_touched"]),
                    "bbox": {
                        "z0": int(out["z0"]), "z1": int(out["z1"]),
                        "y0": int(out["y0"]), "y1": int(out["y1"]),
                        "x0": int(out["x0"]), "x1": int(out["x1"]),
                        "dz": int(out["dz"]), "dy": int(out["dy"]), "dx": int(out["dx"]),
                    }
                })

        if fut_idx % 10 == 0 or fut_idx == num_batches:
            elapsed_now = time.time() - t_parallel
            print(
                f"[progress] batch {fut_idx:>4d}/{num_batches} | "
                f"processed_instances={processed_instances}/{num_instances_raw} | "
                f"changed_instances={num_changed_instances} | "
                f"added_voxels={num_total_added_voxels} | "
                f"elapsed={elapsed_now:.1f}s"
            )

parallel_elapsed = time.time() - t_parallel
elapsed = time.time() - t0

# --------------------------------------------------
# 6) 保存 filled mask + meta
# --------------------------------------------------
print("\n[4/6] 保存 filled mask")
tiff.imwrite(str(OUT_MASK_PATH), filled)
print(f"✅ saved filled mask : {OUT_MASK_PATH}")

print("\n[5/6] 保存 meta json")
filled_ids = np.unique(filled)
num_instances_filled = int(len(filled_ids) - (1 if 0 in filled_ids else 0))

largest_added_cases = sorted(
    largest_added_cases,
    key=lambda x: x["added_voxels"],
    reverse=True
)
top20_largest_added = largest_added_cases[:20]

meta = {
    "raw_mask_path": str(RAW_MASK_PATH),
    "input_meta_path": str(META_PATH),
    "output_mask_path": str(OUT_MASK_PATH),
    "shape": list(map(int, masks.shape)),
    "dtype_in": str(masks.dtype),
    "dtype_out": str(filled.dtype),
    "num_instances_raw": int(num_instances_raw),
    "num_instances_filled": int(num_instances_filled),
    "num_changed_instances": int(num_changed_instances),
    "num_total_added_voxels": int(num_total_added_voxels),
    "num_slices_touched": int(num_slices_touched),
    "elapsed_sec_total": round(elapsed, 4),
    "elapsed_sec_parallel_only": round(parallel_elapsed, 4),
    "method": (
        "parallel-aligned bbox slice-wise 2D "
        "binary_closing(3x3) + "
        f"{'remove_small_holes' if HAS_SKIMAGE else 'binary_fill_holes'} "
        "on each instance"
    ),
    "note": (
        "这版比单纯 binary_fill_holes 更强："
        "先用轻微 closing 补细裂缝，再填小孔；"
        "如果环境没有 scikit-image，则自动回退为 closing + binary_fill_holes。"
    ),
    "num_workers": int(NUM_WORKERS),
    "chunk_size": int(CHUNK_SIZE),
    "min_voxels_skip_fill": int(MIN_VOXELS_SKIP_FILL),
    "has_skimage": bool(HAS_SKIMAGE),
    "small_hole_area_threshold": int(SMALL_HOLE_AREA_THRESHOLD),
    "closing_kernel_shape": list(CLOSING_KERNEL.shape),
    "top20_largest_added_cases": top20_largest_added,
}

with open(OUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ saved meta        : {OUT_META_PATH}")

print("\n[6/6] DONE")
print(f"✅ num_instances_raw    : {num_instances_raw}")
print(f"✅ num_instances_filled : {num_instances_filled}")
print(f"✅ changed_instances    : {num_changed_instances}")
print(f"✅ total_added_voxels   : {num_total_added_voxels}")
print(f"✅ slices_touched       : {num_slices_touched}")
print(f"✅ parallel elapsed     : {parallel_elapsed:.2f}s")
print(f"✅ total elapsed        : {elapsed:.2f}s")
print("=" * 100)


# In[7]:


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


# In[1]:


# -*- coding: utf-8 -*-
"""
hollow_artifact_analysis_batch.py

支持两种模式：
1) 单文件模式：指定一个 MASK_TIF_PATH
2) 批量模式：指定 BATCH_SEARCH_ROOT，自动递归搜索 raw_masks/*.tif

输出：
- 每个 mask 一个独立分析目录
- 总汇总 batch_summary.csv
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
# 运行模式
# ---------------------------
# MODE = "single" 或 "batch"
MODE = "batch"

# ---------------------------
# 单文件模式配置
# ---------------------------
MASK_TIF_PATH = r"/path/to/your/3d_mask.tif"
RAW_TIF_PATH = None
OUT_DIR = r"/path/to/output_hollow_analysis"

# ---------------------------
# 批量模式配置
# ---------------------------
BATCH_SEARCH_ROOT = r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_20260321_v1"
BATCH_OUT_ROOT = r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_20260321_v1/hollow_analysis_batch"

# 如果你有一个固定的原始 full brain 3D tif，可以填这里；
# 批量模式下每个 mask 都默认配这个 raw
DEFAULT_RAW_TIF_PATH = r"/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"   # 没有就设为 None

# 是否只搜索 raw_masks 目录下的 tif
ONLY_SEARCH_RAW_MASKS_DIR = True

# tif 文件名必须包含这些关键词之一，才视为候选 mask
MASK_NAME_KEYWORDS = ["mask", "masks"]

# tif 文件名如果包含这些关键词，就跳过
EXCLUDE_NAME_KEYWORDS = ["filled", "overlay", "vis", "figure", "preview"]


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

'''SEVERITY_ALPHA = 0.6
SEVERITY_BETA = 0.4'''
SEVERITY_ALPHA = 0.75
SEVERITY_BETA = 0.25

# ---------------------------
# 实例级判定参数
# ---------------------------
INSTANCE_MIN_HOLLOW_SLICES = 2
INSTANCE_MIN_HOLLOW_RATIO = 0.20
INSTANCE_MIN_LONGEST_RUN = 2
INSTANCE_MAX_SEVERITY_TH = 0.45


# v2: 防止只靠单层 max_severity 就把整个实例打成 hollow
INSTANCE_MIN_SLICES_FOR_MAX_SEV = 2

# ---------------------------
# 可视化参数
# ---------------------------
SAVE_VIS = True
TOPK_VIS_PER_INSTANCE = 3
MAX_VIS_INSTANCES = 80


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
        # 安全裁剪，避免浮点边界问题
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
    """
    把路径名转成适合建文件夹的安全名字
    """
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:200]


def should_use_as_mask_tif(tif_path: Path) -> bool:
    """
    判断某个 tif 是否应该作为 mask 输入
    """
    name = tif_path.name.lower()

    if tif_path.suffix.lower() not in [".tif", ".tiff"]:
        return False

    if ONLY_SEARCH_RAW_MASKS_DIR:
        if tif_path.parent.name.lower() != "raw_masks":
            return False

    if not any(k in name for k in MASK_NAME_KEYWORDS):
        return False

    if any(k in name for k in EXCLUDE_NAME_KEYWORDS):
        return False

    return True


def discover_mask_tifs(search_root):
    """
    自动递归搜索候选 mask tif
    """
    search_root = Path(search_root)
    if not search_root.exists():
        raise FileNotFoundError(f"搜索根目录不存在：{search_root}")

    candidates = []
    for p in search_root.rglob("*"):
        if p.is_file() and should_use_as_mask_tif(p):
            candidates.append(p)

    candidates = sorted(set(candidates))
    return candidates


# =========================================================
# 2) 单个 mask 的主分析函数
# =========================================================
def analyze_hollow_artifacts(
    mask_tif_path,
    out_dir,
    raw_tif_path=None,
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

    print("=" * 100)
    print("🧪 Hollow Artifact Analysis")
    print("=" * 100)
    print(f"MASK_TIF_PATH : {mask_tif_path}")
    print(f"RAW_TIF_PATH  : {raw_tif_path}")
    print(f"OUT_DIR       : {out_dir}")

    masks = tiff.imread(mask_tif_path)
    if masks.ndim != 3:
        raise ValueError(f"mask tif 必须是 3D (Z,Y,X)，当前 shape={masks.shape}")

    masks = np.asarray(masks)
    Z, Y, X = masks.shape
    print(f"✅ masks shape = {masks.shape}, dtype = {masks.dtype}")

    raw = None
    if raw_tif_path is not None and str(raw_tif_path).strip() != "" and str(raw_tif_path).lower() != "none":
        raw = tiff.imread(raw_tif_path)
        raw = np.asarray(raw)
        if raw.shape != masks.shape:
            print(f"⚠️ raw 与 mask shape 不一致，已跳过 raw。raw={raw.shape}, mask={masks.shape}")
            raw = None
        else:
            print(f"✅ raw shape   = {raw.shape}, dtype = {raw.dtype}")
    else:
        print("ℹ️ 未提供 raw tif，可视化只显示 mask。")

    label_ids = np.unique(masks)
    label_ids = label_ids[label_ids > 0]
    num_instances = len(label_ids)

    print(f"✅ num_instances = {num_instances}")

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

            candidates = sorted(candidates, key=lambda x: x["metrics"]["severity"], reverse=True)[:topk_vis_per_instance]

            for rank_j, r in enumerate(candidates, start=1):
                z_global = r["z_global"]
                m = r["metrics"]
                raw_slice = raw[z_global, y_sl, x_sl] if raw is not None else None

                title_text = (
                    f"label={label_id}, z={z_global}, rank={rank_j}\n"
                    f"area_raw={m['area_raw']}, hole_area={m['area_hole']}, "
                    f"hole_ratio={m['hole_ratio']:.4f}, "
                    f"center_occ={m['center_occupancy'] if not np.isnan(m['center_occupancy']) else 'nan'}, "
                    f"severity={m['severity']:.4f}"
                )

                out_png = vis_dir / f"label_{int(label_id):06d}_z_{int(z_global):04d}_rank_{rank_j}.png"
                save_slice_visualization(
                    out_png_path=out_png,
                    raw_slice=raw_slice,
                    mask_slice=r["slice_mask_2d"],
                    fill_mask=m["fill_mask"],
                    hole_mask=m["hole_mask"],
                    core_mask=m["core_mask"],
                    title_text=title_text,
                )

        if idx_i % 200 == 0 or idx_i == num_instances:
            print(f"🔹 已处理 {idx_i}/{num_instances} 个实例")

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

    print("=" * 100)
    print("✅ 分析完成")
    print(f"✅ slice_csv   : {slice_csv}")
    print(f"✅ instance_csv: {inst_csv}")
    print(f"✅ summary_json: {summary_json}")
    if save_vis:
        print(f"✅ vis_dir     : {vis_dir}")
    print("-" * 100)
    print(f"✅ total_instances       : {total_instances}")
    print(f"✅ hollow_instances      : {hollow_instances}")
    print(f"✅ hollow_instance_ratio : {hollow_instance_ratio:.6f}")
    print(f"✅ total_valid_slices    : {total_valid_slices}")
    print(f"✅ hollow_slices         : {hollow_slices}")
    print(f"✅ hollow_slice_ratio    : {hollow_slice_ratio:.6f}")
    print(f"✅ mean_hollow_severity  : {mean_hollow_severity:.6f}")
    print("=" * 100)

    return slice_df, inst_df, summary


# =========================================================
# 3) 批量模式
# =========================================================
def run_batch_analysis(
    batch_search_root,
    batch_out_root,
    default_raw_tif_path=None
):
    batch_search_root = Path(batch_search_root)
    batch_out_root = Path(batch_out_root)
    ensure_dir(batch_out_root)

    print("=" * 100)
    print("🚀 Batch Hollow Artifact Analysis")
    print("=" * 100)
    print(f"BATCH_SEARCH_ROOT : {batch_search_root}")
    print(f"BATCH_OUT_ROOT    : {batch_out_root}")
    print(f"DEFAULT_RAW_TIF   : {default_raw_tif_path}")

    mask_tifs = discover_mask_tifs(batch_search_root)

    print(f"✅ 共发现 {len(mask_tifs)} 个候选 mask tif")
    for i, p in enumerate(mask_tifs, start=1):
        print(f"  [{i:02d}] {p}")

    if len(mask_tifs) == 0:
        print("⚠️ 没找到符合规则的 mask tif。")
        return

    batch_rows = []

    for idx, mask_path in enumerate(mask_tifs, start=1):
        print("\n" + "#" * 100)
        print(f"### [{idx}/{len(mask_tifs)}] 开始分析")
        print(mask_path)
        print("#" * 100)

        # 给每个 mask 建一个独立输出目录
        rel_parent = mask_path.parent.parent.relative_to(batch_search_root) if mask_path.parent.parent.is_relative_to(batch_search_root) else Path(mask_path.stem)
        exp_name = sanitize_name(str(rel_parent / mask_path.stem))
        out_dir = batch_out_root / exp_name

        try:
            _, _, summary = analyze_hollow_artifacts(
                mask_tif_path=str(mask_path),
                raw_tif_path=default_raw_tif_path,
                out_dir=out_dir,

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

        except Exception as e:
            print(f"❌ 分析失败: {mask_path}")
            print(f"❌ 错误信息: {e}")

            row = {
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

        batch_rows.append(row)

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = batch_out_root / "batch_summary.csv"
        batch_df.to_csv(batch_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("✅ Batch 分析完成")
    print(f"✅ 汇总表: {batch_out_root / 'batch_summary.csv'}")
    print("=" * 100)


# =========================================================
# 4) 主程序入口
# =========================================================
if __name__ == "__main__":
    if MODE == "single":
        analyze_hollow_artifacts(
            mask_tif_path=MASK_TIF_PATH,
            raw_tif_path=RAW_TIF_PATH,
            out_dir=OUT_DIR,

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

    elif MODE == "batch":
        run_batch_analysis(
            batch_search_root=BATCH_SEARCH_ROOT,
            batch_out_root=BATCH_OUT_ROOT,
            default_raw_tif_path=DEFAULT_RAW_TIF_PATH
        )

    else:
        raise ValueError(f"未知 MODE: {MODE}")


# In[2]:


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
        "tag": "flow02_cp0",
        "flow_tag": "flow02",
        "cp_tag": "cp0",
        "mask_tif_path": Path(
            "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
            "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
            "fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/raw_masks/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif"
        )
    },
    {
        "tag": "flow04_cp0",
        "flow_tag": "flow04",
        "cp_tag": "cp0",
        "mask_tif_path": Path(
            "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
            "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
            "fullbrain_top1_rescale175_dNone_cp0_ms50_flow04_v1/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/raw_masks/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif"
        )
    },
    {
        "tag": "flow04_cp1",
        "flow_tag": "flow04",
        "cp_tag": "cp1",
        "mask_tif_path": Path(
            "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
            "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
            "fullbrain_top1_rescale175_dNone_cp0_ms50_flow04_v1/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp1__ms50/raw_masks/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp1__ms50__full_brain_3d_masks_raw.tif"
        )
    },
    {
        "tag": "flow04_cpneg1",
        "flow_tag": "flow04",
        "cp_tag": "cpneg1",
        "mask_tif_path": Path(
            "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
            "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
            "fullbrain_top1_rescale175_dNone_cp0_ms50_flow04_v1/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cpm1__ms50/raw_masks/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cpm1__ms50__full_brain_3d_masks_raw.tif"
        )
    },
    {
        "tag": "flow08_cp0",
        "flow_tag": "flow08",
        "cp_tag": "cp0",
        "mask_tif_path": Path(
            "/gpfs/share/home/2306391536/projects/cell_seg/runs/"
            "exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/"
            "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/raw_masks/"
            "P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif"
        )
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


# In[3]:


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
        "tag": "rs1p25_cp0",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_20260321_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p25__cp0__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p25__cp0__ms50__full_brain_3d_masks_raw.tif",
    },
    {
        "tag": "rs1p75_cp0",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_20260321_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp0__ms50__full_brain_3d_masks_raw.tif",
    },
    {
        "tag": "rs1p75_cp2p5",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_20260321_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp2p5__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp2p5__ms50__full_brain_3d_masks_raw.tif",
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


# In[6]:


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


# In[8]:


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
        "tag": "rs1p25_cp1_flow02",
        "mask_tif_path": r"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/fullbrain_top1_rescale175_dNone_cp0_ms50_flow02_v1/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp1__ms50/raw_masks/P21_lr9e5_wd8e3__a2.0__dNone__rs1p75__cp1__ms50__full_brain_3d_masks_raw.tif",
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




