#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tifffile

tif_path = "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"

with tifffile.TiffFile(tif_path) as tif:
    print("pages:", len(tif.pages))
    print("series shape:", tif.series[0].shape)
    print("series axes:", tif.series[0].axes)
    print("imagej_metadata:", tif.imagej_metadata)
    print("ome_metadata:", tif.ome_metadata)
    print("first page tags:")
    for tag in tif.pages[0].tags.values():
        print(tag.name, tag.value)


# In[ ]:


# ==========================================
# Cell X：基于当前脚本改造的 3D 参数搜索版（crop sweep）
# 功能：
# 1) 只使用当前 notebook 的 ROOT / EXP_DIR / CFG_DIR
# 2) baseline 复用旧结果，不重复跑
# 3) 当前模型固定为 TARGET_TAG（如 P21_lr9e5_wd8e3）
# 4) 在 crop 上扫描 anisotropy / diameter / stitch_threshold
# 5) 每组保存 mask / stats / overlay
# 6) 导出 summary_3d_sweep.csv，并生成 top-k 总览图
# ==========================================
import json
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models, io

print("=" * 100)
print("🚀 Cell X | 3D parameter sweep (crop version) under current EXP_DIR")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 依赖当前 notebook 前面已经定义好的变量
# --------------------------------------------------
# --------------------------------------------------
# 0) 路径初始化：直接手动写死，别再复用旧变量
# --------------------------------------------------
from pathlib import Path

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

CFG_DIR = (EXP_DIR / "config").resolve()

print("📌 当前上下文")
print("ROOT   :", ROOT)
print("EXP_DIR:", EXP_DIR)
print("CFG_DIR:", CFG_DIR)

assert ROOT.exists(), f"ROOT 不存在: {ROOT}"
assert EXP_DIR.exists(), f"EXP_DIR 不存在: {EXP_DIR}"
assert CFG_DIR.exists(), f"CFG_DIR 不存在: {CFG_DIR}"

print("📌 当前上下文")
print("ROOT   :", ROOT)
print("EXP_DIR:", EXP_DIR)
print("CFG_DIR:", CFG_DIR)


# --------------------------------------------------
# 1) 你最常改的参数：只需要改这里
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"
# TARGET_TAG = "P22_lr9e5_wd9e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

# baseline：直接复用旧结果，不重复跑
BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()
BASELINE_NAME = "Baseline_cpsam"

# ======= 可视化 / crop 位置 =======
# 你之前看的区域
Z_VIS = 100
Y0, Y1 = 478, 734
X0, X1 = 353, 609

# 可选：为了参数搜索更稳一点，也可以让 crop 在 z 上取一小段体积
# 下面默认取 Z_VIS 前后各 16 层，共 33 层
Z_HALF_SPAN = 16

# ======= 3D 推理固定参数 =======
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 16

# ======= 搜索参数（核心）=======
ANISOTROPY_LIST = [1, 2, 4, 6, 8]
DIAMETER_LIST = [6, 7, 8, 9]
STITCH_THRESHOLD_LIST = [0.0, 0.1, 0.25]

# 只想先小试几组的话，可把上面改短，例如：
# ANISOTROPY_LIST = [1, 4, 8]
# DIAMETER_LIST = [6, 7, 8]
# STITCH_THRESHOLD_LIST = [0.0, 0.1]

# 是否强制重跑已存在的某组结果
FORCE_RERUN = False

# top-k 可视化总览
TOPK_SHOW = 6

# --------------------------------------------------
# 2) 输出目录：全部挂到当前 EXP_DIR 下
# --------------------------------------------------
SWEEP_ROOT = (EXP_DIR / "sweep_3d_compare").resolve()
MASK_ROOT = (SWEEP_ROOT / TARGET_TAG / "masks").resolve()
FIG_ROOT = (SWEEP_ROOT / TARGET_TAG / "figures").resolve()
STAT_ROOT = (SWEEP_ROOT / TARGET_TAG / "stats").resolve()

SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
MASK_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)

print("\n📂 输出目录")
print("SWEEP_ROOT:", SWEEP_ROOT)
print("MASK_ROOT :", MASK_ROOT)
print("FIG_ROOT  :", FIG_ROOT)
print("STAT_ROOT :", STAT_ROOT)

# --------------------------------------------------
# 3) 小工具
# --------------------------------------------------
def stage(msg: str):
    print(f"\n🔹 {msg}")

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary(mask2d: np.ndarray):
    fg = (mask2d > 0).astype(np.uint8)
    up    = np.roll(fg, -1, axis=0)
    down  = np.roll(fg,  1, axis=0)
    left  = np.roll(fg, -1, axis=1)
    right = np.roll(fg,  1, axis=1)
    bd = fg & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    bd[0, :] = 0
    bd[-1, :] = 0
    bd[:, 0] = 0
    bd[:, -1] = 0
    return bd.astype(bool)

def overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "blue":
        overlay[bd] = [0.0, 0.6, 1.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [1.0, 0.0, 1.0]

    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_sizes = counts[valid]

    total_cells = int(valid.sum())
    if total_cells == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    max_v = float(np.max(obj_sizes))
    large_ratio = float(np.mean(obj_sizes >= (2.0 * max(median_v, 1.0))))

    return {
        "total_cells": total_cells,
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
    }

def run_name_from_params(tag, anisotropy, diameter, stitch):
    # 为了路径安全，把小数点替成 p
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__s{s}"

# --------------------------------------------------
# 4) 只从当前 EXP_DIR/config 中找 TARGET_TAG 对应模型
# --------------------------------------------------
stage("[1/8] 在当前 EXP_DIR/config 中定位模型")

config_files = sorted(CFG_DIR.glob("config_*.json"))
assert len(config_files) > 0, f"当前 EXP_DIR 的 config 下没有 snapshot: {CFG_DIR}"

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

    if best_model_path and Path(best_model_path).exists():
        selected_model_path = str(Path(best_model_path).resolve())
        selected_snapshot_path = str(sf.resolve())
        selected_snapshot = snap
        break
    elif final_model_path and Path(final_model_path).exists():
        selected_model_path = str(Path(final_model_path).resolve())
        selected_snapshot_path = str(sf.resolve())
        selected_snapshot = snap
        break

assert selected_model_path is not None, f"当前 EXP_DIR/config 下没找到 {TARGET_TAG} 的可用模型"

print("✅ TARGET_TAG            :", TARGET_TAG)
print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

# --------------------------------------------------
# 5) 读取 raw / baseline，并构造 crop 体积
# --------------------------------------------------
stage("[2/8] 读取 raw 3D 数据与 baseline 旧结果")

assert RAW_3D_STACK_PATH.exists(), f"3D 原始数据不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"baseline 结果不存在: {BASELINE_MASK_PATH}"

stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))

assert stack.ndim == 3, f"期望 raw 为 3D stack，实际 shape={stack.shape}"
assert baseline_masks.shape == stack.shape, (
    f"baseline mask shape={baseline_masks.shape} 与 raw={stack.shape} 不一致"
)

Z, Y, X = stack.shape
assert 0 <= Z_VIS < Z, f"Z_VIS={Z_VIS} 超出范围 0~{Z-1}"
assert 0 <= Y0 < Y1 <= Y, f"Y crop 越界: {(Y0, Y1)} vs Y={Y}"
assert 0 <= X0 < X1 <= X, f"X crop 越界: {(X0, X1)} vs X={X}"

Z0 = max(0, Z_VIS - Z_HALF_SPAN)
Z1 = min(Z, Z_VIS + Z_HALF_SPAN + 1)

stack_crop = stack[Z0:Z1, Y0:Y1, X0:X1]
baseline_crop = baseline_masks[Z0:Z1, Y0:Y1, X0:X1]

print("✅ raw stack shape     :", stack.shape, "| dtype:", stack.dtype)
print("✅ baseline loaded     :", BASELINE_MASK_PATH)
print(f"✅ crop box (zyx)      : z=({Z0},{Z1}), y=({Y0},{Y1}), x=({X0},{X1})")
print("✅ crop stack shape    :", stack_crop.shape)

# baseline 统计
base_stats_crop = summarize_mask(baseline_crop)

# --------------------------------------------------
# 6) 参数组合
# --------------------------------------------------
stage("[3/8] 生成参数组合")

param_grid = list(itertools.product(
    ANISOTROPY_LIST,
    DIAMETER_LIST,
    STITCH_THRESHOLD_LIST
))

print(f"✅ 参数组合总数: {len(param_grid)}")
for i, (a, d, s) in enumerate(param_grid[:10], 1):
    print(f"  {i:02d}. anisotropy={a}, diameter={d}, stitch={s}")
if len(param_grid) > 10:
    print("  ...")

# --------------------------------------------------
# 7) 加载模型，只加载一次
# --------------------------------------------------
stage("[4/8] 加载当前模型")

model = models.CellposeModel(
    gpu=True,
    pretrained_model=selected_model_path
)
print("✅ 模型已加载")

# --------------------------------------------------
# 8) 循环跑每组参数
# --------------------------------------------------
stage("[5/8] 开始参数搜索（crop 上跑）")

summary_rows = []

for idx, (anisotropy, diameter, stitch) in enumerate(param_grid, 1):
    run_name = run_name_from_params(TARGET_TAG, anisotropy, diameter, stitch)

    this_mask_path = MASK_ROOT / f"{run_name}_crop_masks.tif"
    this_stats_path = STAT_ROOT / f"{run_name}_stats.json"
    this_fig_path = FIG_ROOT / f"{run_name}_overlay.png"

    print("\n" + "-" * 100)
    print(f"[{idx}/{len(param_grid)}] {run_name}")
    print("-" * 100)

    need_run = FORCE_RERUN or (not this_mask_path.exists())

    if not need_run:
        print(f"✅ 已存在，直接复用: {this_mask_path}")
        masks = tiff.imread(str(this_mask_path))
        elapsed_s = None
    else:
        print(
            f"🔥 Running crop 3D eval | "
            f"anisotropy={anisotropy}, diameter={diameter}, stitch={stitch}"
        )
        t0 = time.time()

        masks, _, _ = model.eval(
            stack_crop,
            diameter=diameter,
            do_3D=DO_3D,
            stitch_threshold=stitch,
            z_axis=Z_AXIS,
            batch_size=BATCH_SIZE_3D,
            anisotropy=anisotropy,
            progress=True
        )
        masks = np.asarray(masks)

        elapsed_s = float(time.time() - t0)
        tiff.imwrite(str(this_mask_path), masks.astype(np.uint32))
        print(f"✅ 推理完成，耗时 {elapsed_s/60:.2f} min")

    assert masks.shape == stack_crop.shape, (
        f"预测 shape={masks.shape} 与 crop={stack_crop.shape} 不一致"
    )

    stats = summarize_mask(masks)

    # 取 crop 中间层可视化
    z_mid_local = masks.shape[0] // 2
    z_mid_global = Z0 + z_mid_local

    raw_mid = stack_crop[z_mid_local]
    base_mid = baseline_crop[z_mid_local]
    this_mid = masks[z_mid_local]

    raw_mid_norm = normalize_img(raw_mid)
    base_overlay = overlay_boundary(raw_mid, base_mid, color="red")
    this_overlay = overlay_boundary(raw_mid, this_mid, color="green")

    # 单独保存对比图
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(raw_mid_norm, cmap="gray")
    axes[0].set_title(f"Raw crop | global z={z_mid_global}")
    axes[0].axis("off")

    axes[1].imshow(base_overlay)
    axes[1].set_title(
        f"{BASELINE_NAME}\n"
        f"cells={base_stats_crop['total_cells']} | "
        f"medV={base_stats_crop['median_volume']:.1f} | "
        f"largeR={base_stats_crop['large_obj_ratio_ge_2x_median']:.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(this_overlay)
    axes[2].set_title(
        f"a={anisotropy}, d={diameter}, s={stitch}\n"
        f"cells={stats['total_cells']} | "
        f"medV={stats['median_volume']:.1f} | "
        f"largeR={stats['large_obj_ratio_ge_2x_median']:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(this_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 存 json 记录
    stats_record = {
        "target_tag": TARGET_TAG,
        "selected_model_path": selected_model_path,
        "selected_snapshot_path": selected_snapshot_path,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "baseline_mask_path": str(BASELINE_MASK_PATH),
        "crop_box": {
            "z0": int(Z0), "z1": int(Z1),
            "y0": int(Y0), "y1": int(Y1),
            "x0": int(X0), "x1": int(X1),
            "z_vis_center": int(Z_VIS),
        },
        "mask_path": str(this_mask_path),
        "fig_path": str(this_fig_path),
        "shape": list(masks.shape),
        "elapsed_s": elapsed_s,
        "anisotropy": anisotropy,
        "diameter_3d": diameter,
        "do_3d": DO_3D,
        "stitch_threshold": stitch,
        "z_axis": Z_AXIS,
        "batch_size_3d": BATCH_SIZE_3D,
        **stats
    }
    this_stats_path.write_text(
        json.dumps(stats_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    row = {
        "run_name": run_name,
        "anisotropy": anisotropy,
        "diameter": diameter,
        "stitch_threshold": stitch,
        "elapsed_s": elapsed_s,
        "mask_path": str(this_mask_path),
        "fig_path": str(this_fig_path),
        **stats
    }
    summary_rows.append(row)

    print("✅ done")
    print("   mask :", this_mask_path)
    print("   fig  :", this_fig_path)
    print("   cells:", stats["total_cells"])
    print("   medV :", round(stats["median_volume"], 2))
    print("   largeR:", round(stats["large_obj_ratio_ge_2x_median"], 4))

# --------------------------------------------------
# 9) 汇总成表并排序
# --------------------------------------------------
stage("[6/8] 汇总搜索结果")

summary_df = pd.DataFrame(summary_rows)

# 一个比较土但够用的排序逻辑：
# - 先让 large_obj_ratio 低（粘连嫌疑小）
# - 再让 total_cells 高一点（别少得离谱）
# - 再让 median_volume 不至于过大
summary_df = summary_df.sort_values(
    ["large_obj_ratio_ge_2x_median", "total_cells", "median_volume"],
    ascending=[True, False, True]
).reset_index(drop=True)

summary_csv = SWEEP_ROOT / f"summary_3d_sweep_{TARGET_TAG}.csv"
summary_df.to_csv(summary_csv, index=False)

print("✅ summary saved:", summary_csv)
display_cols = [
    "run_name",
    "anisotropy",
    "diameter",
    "stitch_threshold",
    "total_cells",
    "median_volume",
    "p90_volume",
    "max_volume",
    "large_obj_ratio_ge_2x_median",
    "elapsed_s",
]
display(summary_df[display_cols].head(15))

# --------------------------------------------------
# 10) top-k 总览图
# --------------------------------------------------
stage("[7/8] 生成 top-k 总览图")

k = min(TOPK_SHOW, len(summary_df))
top_df = summary_df.head(k).copy()

ncols = 3
nrows = int(np.ceil(k / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
axes = np.array(axes).reshape(-1)

for ax in axes:
    ax.axis("off")

for i, (_, row) in enumerate(top_df.iterrows()):
    fig_path = Path(row["fig_path"])
    if fig_path.exists():
        img = plt.imread(str(fig_path))
        axes[i].imshow(img)
        axes[i].set_title(
            f"rank#{i+1}\n"
            f"a={row['anisotropy']}, d={row['diameter']}, s={row['stitch_threshold']}\n"
            f"cells={int(row['total_cells'])} | "
            f"medV={row['median_volume']:.1f} | "
            f"largeR={row['large_obj_ratio_ge_2x_median']:.3f}",
            fontsize=10
        )
        axes[i].axis("off")

topk_fig_path = SWEEP_ROOT / f"top{TOPK_SHOW}_overview_{TARGET_TAG}.png"
plt.tight_layout()
plt.savefig(topk_fig_path, dpi=180, bbox_inches="tight")
plt.show()

print("✅ top-k 总览图已保存:", topk_fig_path)

# --------------------------------------------------
# 11) 保存 sweep 配置
# --------------------------------------------------
stage("[8/8] 保存 sweep 配置")

record = {
    "target_tag": TARGET_TAG,
    "selected_model_path": selected_model_path,
    "selected_snapshot_path": selected_snapshot_path,
    "exp_dir": str(EXP_DIR),
    "cfg_dir": str(CFG_DIR),
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_name": BASELINE_NAME,
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "crop_box": {
        "z0": int(Z0), "z1": int(Z1),
        "y0": int(Y0), "y1": int(Y1),
        "x0": int(X0), "x1": int(X1),
        "z_vis_center": int(Z_VIS),
        "z_half_span": int(Z_HALF_SPAN),
    },
    "do_3d": DO_3D,
    "z_axis": Z_AXIS,
    "batch_size_3d": BATCH_SIZE_3D,
    "anisotropy_list": ANISOTROPY_LIST,
    "diameter_list": DIAMETER_LIST,
    "stitch_threshold_list": STITCH_THRESHOLD_LIST,
    "force_rerun": FORCE_RERUN,
    "topk_show": TOPK_SHOW,
    "summary_csv": str(summary_csv),
    "topk_fig_path": str(topk_fig_path),
    "baseline_crop_stats": base_stats_crop,
}
record_path = SWEEP_ROOT / f"sweep_config_{TARGET_TAG}.json"
record_path.write_text(
    json.dumps(record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("✅ sweep 配置记录已保存:", record_path)

print("\n" + "=" * 100)
print("🏁 全部完成")
print("summary_csv :", summary_csv)
print("top-k 总览图:", topk_fig_path)
print("mask 根目录 :", MASK_ROOT)
print("figure 根目录:", FIG_ROOT)
print("=" * 100)


# In[ ]:


from pathlib import Path

runs_root = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs")
cand = sorted(runs_root.glob("exp_20260311_ori_train_9runs_400ep_es50_*"))

print("匹配到的目录：")
for p in cand:
    print(p)


# In[ ]:


from pathlib import Path

baseline_candidates = list(
    Path("/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results").rglob("full_brain_3d_masks.tif")
)

print("找到的 baseline candidates:")
for p in baseline_candidates:
    print(" -", p)

assert len(baseline_candidates) > 0, "一个 baseline 3D mask 都没找到，请检查目录"

BASELINE_MASK_PATH = baseline_candidates[0].resolve()
print("使用 BASELINE_MASK_PATH:", BASELINE_MASK_PATH)


# In[ ]:


# ==========================================
# Cell New-1：新增一轮 3D crop 参数扫描
# 参数：
#   ANISOTROPY_LIST = [1.5, 2.0, 2.5]
#   DIAMETER_LIST = [7, 8, 9]
#   CELLPROB_THRESHOLD_LIST = [-1, 0, 1]
#   STITCH_THRESHOLD = 0.0
# ==========================================
import json
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models, io

print("=" * 100)
print("🚀 Cell New-1 | New crop sweep with anisotropy / diameter / cellprob")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 路径初始化（手动写死，避免旧变量污染）
# --------------------------------------------------
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

CFG_DIR = (EXP_DIR / "config").resolve()

print("📌 当前上下文")
print("ROOT   :", ROOT)
print("EXP_DIR:", EXP_DIR)
print("CFG_DIR:", CFG_DIR)

assert ROOT.exists(), f"ROOT 不存在: {ROOT}"
assert EXP_DIR.exists(), f"EXP_DIR 不存在: {EXP_DIR}"
assert CFG_DIR.exists(), f"CFG_DIR 不存在: {CFG_DIR}"

# --------------------------------------------------
# 1) 主要参数
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()
BASELINE_NAME = "Baseline_cpsam"

# 之前那个 crop
Z_VIS = 100
Y0, Y1 = 478, 734
X0, X1 = 353, 609
Z_HALF_SPAN = 16

DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 16

# ===== 新一轮参数 =====
ANISOTROPY_LIST = [1.5, 2.0, 2.5]
DIAMETER_LIST = [7, 8, 9]
CELLPROB_THRESHOLD_LIST = [-1, 0, 1]
STITCH_THRESHOLD = 0.0

FORCE_RERUN = False
TOPK_SHOW = 6

# 给这轮 sweep 一个名字
SWEEP_NAME = "sweep_cp_v2_a1p5_2p5_d7_9_cp-1_1"

# --------------------------------------------------
# 2) 输出目录
# --------------------------------------------------
SWEEP_ROOT = (EXP_DIR / "sweep_3d_compare" / SWEEP_NAME).resolve()
MASK_ROOT = (SWEEP_ROOT / TARGET_TAG / "masks").resolve()
FIG_ROOT = (SWEEP_ROOT / TARGET_TAG / "figures").resolve()
STAT_ROOT = (SWEEP_ROOT / TARGET_TAG / "stats").resolve()

SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
MASK_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)

print("\n📂 输出目录")
print("SWEEP_ROOT:", SWEEP_ROOT)
print("MASK_ROOT :", MASK_ROOT)
print("FIG_ROOT  :", FIG_ROOT)
print("STAT_ROOT :", STAT_ROOT)

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

def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary(mask2d: np.ndarray):
    fg = (mask2d > 0).astype(np.uint8)
    up    = np.roll(fg, -1, axis=0)
    down  = np.roll(fg,  1, axis=0)
    left  = np.roll(fg, -1, axis=1)
    right = np.roll(fg,  1, axis=1)
    bd = fg & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    bd[0, :] = 0
    bd[-1, :] = 0
    bd[:, 0] = 0
    bd[:, -1] = 0
    return bd.astype(bool)

def overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "blue":
        overlay[bd] = [0.0, 0.6, 1.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [1.0, 0.0, 1.0]

    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_sizes = counts[valid]

    total_cells = int(valid.sum())
    if total_cells == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    max_v = float(np.max(obj_sizes))
    large_ratio = float(np.mean(obj_sizes >= (2.0 * max(median_v, 1.0))))

    return {
        "total_cells": total_cells,
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
    }

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

# --------------------------------------------------
# 4) 定位模型
# --------------------------------------------------
stage("[1/8] 在当前 EXP_DIR/config 中定位模型")

config_files = sorted(CFG_DIR.glob("config_*.json"))
assert len(config_files) > 0, f"当前 EXP_DIR 的 config 下没有 snapshot: {CFG_DIR}"

selected_model_path = None
selected_snapshot_path = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue

    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
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

assert selected_model_path is not None, f"当前 EXP_DIR/config 下没找到 {TARGET_TAG} 的可用模型"

print("✅ TARGET_TAG            :", TARGET_TAG)
print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

# --------------------------------------------------
# 5) 读取 raw / baseline / crop
# --------------------------------------------------
stage("[2/8] 读取 raw 3D 数据与 baseline 旧结果")

assert RAW_3D_STACK_PATH.exists(), f"3D 原始数据不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"baseline 结果不存在: {BASELINE_MASK_PATH}"

stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))

assert stack.ndim == 3, f"期望 raw 为 3D stack，实际 shape={stack.shape}"
assert baseline_masks.shape == stack.shape, (
    f"baseline mask shape={baseline_masks.shape} 与 raw={stack.shape} 不一致"
)

Z, Y, X = stack.shape
assert 0 <= Z_VIS < Z
assert 0 <= Y0 < Y1 <= Y
assert 0 <= X0 < X1 <= X

Z0 = max(0, Z_VIS - Z_HALF_SPAN)
Z1 = min(Z, Z_VIS + Z_HALF_SPAN + 1)

stack_crop = stack[Z0:Z1, Y0:Y1, X0:X1]
baseline_crop = baseline_masks[Z0:Z1, Y0:Y1, X0:X1]

print("✅ raw stack shape     :", stack.shape, "| dtype:", stack.dtype)
print("✅ baseline loaded     :", BASELINE_MASK_PATH)
print(f"✅ crop box (zyx)      : z=({Z0},{Z1}), y=({Y0},{Y1}), x=({X0},{X1})")
print("✅ crop stack shape    :", stack_crop.shape)

base_stats_crop = summarize_mask(baseline_crop)

# --------------------------------------------------
# 6) 参数组合
# --------------------------------------------------
stage("[3/8] 生成参数组合")

param_grid = list(itertools.product(
    ANISOTROPY_LIST,
    DIAMETER_LIST,
    CELLPROB_THRESHOLD_LIST
))

print(f"✅ 参数组合总数: {len(param_grid)}")
for i, (a, d, cp) in enumerate(param_grid[:10], 1):
    print(f"  {i:02d}. anisotropy={a}, diameter={d}, cellprob={cp}")
if len(param_grid) > 10:
    print("  ...")

# --------------------------------------------------
# 7) 加载模型
# --------------------------------------------------
stage("[4/8] 加载当前模型")

model = models.CellposeModel(
    gpu=True,
    pretrained_model=selected_model_path
)
print("✅ 模型已加载")

# --------------------------------------------------
# 8) 循环跑每组参数
# --------------------------------------------------
stage("[5/8] 开始参数搜索（crop 上跑）")

summary_rows = []

for idx, (anisotropy, diameter, cellprob) in enumerate(param_grid, 1):
    run_name = run_name_from_params(TARGET_TAG, anisotropy, diameter, cellprob, STITCH_THRESHOLD)

    this_mask_path = MASK_ROOT / f"{run_name}_crop_masks.tif"
    this_stats_path = STAT_ROOT / f"{run_name}_stats.json"
    this_fig_path = FIG_ROOT / f"{run_name}_overlay.png"

    print("\n" + "-" * 100)
    print(f"[{idx}/{len(param_grid)}] {run_name}")
    print("-" * 100)

    need_run = FORCE_RERUN or (not this_mask_path.exists())

    if not need_run:
        print(f"✅ 已存在，直接复用: {this_mask_path}")
        masks = tiff.imread(str(this_mask_path))
        elapsed_s = np.nan
    else:
        print(
            f"🔥 Running crop 3D eval | "
            f"anisotropy={anisotropy}, diameter={diameter}, "
            f"cellprob={cellprob}, stitch={STITCH_THRESHOLD}"
        )
        t0 = time.time()

        masks, _, _ = model.eval(
            stack_crop,
            diameter=diameter,
            do_3D=DO_3D,
            stitch_threshold=STITCH_THRESHOLD,
            z_axis=Z_AXIS,
            batch_size=BATCH_SIZE_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob,
            progress=True
        )
        masks = np.asarray(masks)

        elapsed_s = float(time.time() - t0)
        tiff.imwrite(str(this_mask_path), masks.astype(np.uint32))
        print(f"✅ 推理完成，耗时 {elapsed_s/60:.2f} min")

    assert masks.shape == stack_crop.shape

    stats = summarize_mask(masks)

    z_mid_local = masks.shape[0] // 2
    z_mid_global = Z0 + z_mid_local

    raw_mid = stack_crop[z_mid_local]
    base_mid = baseline_crop[z_mid_local]
    this_mid = masks[z_mid_local]

    raw_mid_norm = normalize_img(raw_mid)
    base_overlay = overlay_boundary(raw_mid, base_mid, color="red")
    this_overlay = overlay_boundary(raw_mid, this_mid, color="green")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(raw_mid_norm, cmap="gray")
    axes[0].set_title(f"Raw crop | global z={z_mid_global}")
    axes[0].axis("off")

    axes[1].imshow(base_overlay)
    axes[1].set_title(
        f"{BASELINE_NAME}\n"
        f"cells={base_stats_crop['total_cells']} | "
        f"medV={base_stats_crop['median_volume']:.1f} | "
        f"largeR={base_stats_crop['large_obj_ratio_ge_2x_median']:.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(this_overlay)
    axes[2].set_title(
        f"a={anisotropy}, d={diameter}, cp={cellprob}\n"
        f"cells={stats['total_cells']} | "
        f"medV={stats['median_volume']:.1f} | "
        f"largeR={stats['large_obj_ratio_ge_2x_median']:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(this_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    stats_record = {
        "target_tag": TARGET_TAG,
        "selected_model_path": selected_model_path,
        "selected_snapshot_path": selected_snapshot_path,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "baseline_mask_path": str(BASELINE_MASK_PATH),
        "crop_box": {
            "z0": int(Z0), "z1": int(Z1),
            "y0": int(Y0), "y1": int(Y1),
            "x0": int(X0), "x1": int(X1),
            "z_vis_center": int(Z_VIS),
        },
        "mask_path": str(this_mask_path),
        "fig_path": str(this_fig_path),
        "shape": list(masks.shape),
        "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
        "anisotropy": anisotropy,
        "diameter_3d": diameter,
        "cellprob_threshold": cellprob,
        "do_3d": DO_3D,
        "stitch_threshold": STITCH_THRESHOLD,
        "z_axis": Z_AXIS,
        "batch_size_3d": BATCH_SIZE_3D,
        **stats
    }
    this_stats_path.write_text(
        json.dumps(stats_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    row = {
        "run_name": run_name,
        "sweep_name": SWEEP_NAME,
        "anisotropy": anisotropy,
        "diameter": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": STITCH_THRESHOLD,
        "elapsed_s": elapsed_s,
        "mask_path": str(this_mask_path),
        "fig_path": str(this_fig_path),
        **stats
    }
    summary_rows.append(row)

    print("✅ done")
    print("   mask  :", this_mask_path)
    print("   fig   :", this_fig_path)
    print("   cells :", stats["total_cells"])
    print("   medV  :", round(stats["median_volume"], 2))
    print("   largeR:", round(stats["large_obj_ratio_ge_2x_median"], 4))

# --------------------------------------------------
# 9) 汇总
# --------------------------------------------------
stage("[6/8] 汇总搜索结果")

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    ["large_obj_ratio_ge_2x_median", "total_cells", "median_volume"],
    ascending=[True, False, True]
).reset_index(drop=True)

summary_csv = SWEEP_ROOT / f"summary_3d_sweep_{TARGET_TAG}.csv"
summary_df.to_csv(summary_csv, index=False)

print("✅ summary saved:", summary_csv)
display_cols = [
    "run_name", "anisotropy", "diameter", "cellprob_threshold",
    "stitch_threshold", "total_cells", "median_volume",
    "p90_volume", "max_volume", "large_obj_ratio_ge_2x_median", "elapsed_s",
]
display(summary_df[display_cols].head(15))

# --------------------------------------------------
# 10) top-k 总览图
# --------------------------------------------------
stage("[7/8] 生成 top-k 总览图")

k = min(TOPK_SHOW, len(summary_df))
top_df = summary_df.head(k).copy()

ncols = 3
nrows = int(np.ceil(k / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
axes = np.array(axes).reshape(-1)

for ax in axes:
    ax.axis("off")

for i, (_, row) in enumerate(top_df.iterrows()):
    fig_path = Path(row["fig_path"])
    if fig_path.exists():
        img = plt.imread(str(fig_path))
        axes[i].imshow(img)
        axes[i].set_title(
            f"rank#{i+1}\n"
            f"a={row['anisotropy']}, d={row['diameter']}, cp={row['cellprob_threshold']}\n"
            f"cells={int(row['total_cells'])} | "
            f"medV={row['median_volume']:.1f} | "
            f"largeR={row['large_obj_ratio_ge_2x_median']:.3f}",
            fontsize=10
        )
        axes[i].axis("off")

topk_fig_path = SWEEP_ROOT / f"top{TOPK_SHOW}_overview_{TARGET_TAG}.png"
plt.tight_layout()
plt.savefig(topk_fig_path, dpi=180, bbox_inches="tight")
plt.show()

print("✅ top-k 总览图已保存:", topk_fig_path)

# --------------------------------------------------
# 11) 保存 sweep 配置
# --------------------------------------------------
stage("[8/8] 保存 sweep 配置")

record = {
    "target_tag": TARGET_TAG,
    "selected_model_path": selected_model_path,
    "selected_snapshot_path": selected_snapshot_path,
    "exp_dir": str(EXP_DIR),
    "cfg_dir": str(CFG_DIR),
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_name": BASELINE_NAME,
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "crop_box": {
        "z0": int(Z0), "z1": int(Z1),
        "y0": int(Y0), "y1": int(Y1),
        "x0": int(X0), "x1": int(X1),
        "z_vis_center": int(Z_VIS),
        "z_half_span": int(Z_HALF_SPAN),
    },
    "do_3d": DO_3D,
    "z_axis": Z_AXIS,
    "batch_size_3d": BATCH_SIZE_3D,
    "anisotropy_list": ANISOTROPY_LIST,
    "diameter_list": DIAMETER_LIST,
    "cellprob_threshold_list": CELLPROB_THRESHOLD_LIST,
    "stitch_threshold": STITCH_THRESHOLD,
    "force_rerun": FORCE_RERUN,
    "topk_show": TOPK_SHOW,
    "summary_csv": str(summary_csv),
    "topk_fig_path": str(topk_fig_path),
    "baseline_crop_stats": base_stats_crop,
}
record_path = SWEEP_ROOT / f"sweep_config_{TARGET_TAG}.json"
record_path.write_text(
    json.dumps(record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("✅ sweep 配置记录已保存:", record_path)

print("\n" + "=" * 100)
print("🏁 全部完成")
print("summary_csv :", summary_csv)
print("top-k 总览图:", topk_fig_path)
print("mask 根目录 :", MASK_ROOT)
print("figure 根目录:", FIG_ROOT)
print("=" * 100)


# In[ ]:


#检查文件是否在

from pathlib import Path
import json

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (EXP_DIR / "config").resolve()
TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

OLD_SUMMARY_CSV = (
    EXP_DIR / "sweep_3d_compare" / f"summary_3d_sweep_{TARGET_TAG}.csv"
).resolve()

NEW_SUMMARY_CSV = (
    EXP_DIR / "sweep_3d_compare" / "sweep_cp_v2_a1p5_2p5_d7_9_cp-1_1" / f"summary_3d_sweep_{TARGET_TAG}.csv"
).resolve()

print("ROOT exists        :", ROOT.exists(), ROOT)
print("EXP_DIR exists     :", EXP_DIR.exists(), EXP_DIR)
print("CFG_DIR exists     :", CFG_DIR.exists(), CFG_DIR)
print("RAW exists         :", RAW_3D_STACK_PATH.exists(), RAW_3D_STACK_PATH)
print("BASELINE exists    :", BASELINE_MASK_PATH.exists(), BASELINE_MASK_PATH)
print("OLD summary exists :", OLD_SUMMARY_CSV.exists(), OLD_SUMMARY_CSV)
print("NEW summary exists :", NEW_SUMMARY_CSV.exists(), NEW_SUMMARY_CSV)

config_files = sorted(CFG_DIR.glob("config_*.json"))
print("config count:", len(config_files))

selected_model_path = None
selected_snapshot_path = None

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
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

print("selected_model_path   :", selected_model_path)
print("selected_snapshot_path:", selected_snapshot_path)


# In[ ]:


# ==========================================
# Cell Rescue-Rank1-StepA：只跑 full-brain rank1，并优先保存 mask
# ==========================================
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🚑 Rescue-Rank1-StepA | Run full-brain rank1 only and save mask first")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
full_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"

MASK_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)

assert CFG_DIR.exists(), f"CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

RUN_ONLY_RANK = 1
BATCH_SIZE_3D = 4   # 比原来的 16 更稳；如果还炸，再改成 2

# --------------------------------------------------
# 1) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# --------------------------------------------------
# 2) 找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
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

assert selected_model_path is not None, f"没找到 {TARGET_TAG} 对应模型"

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

# --------------------------------------------------
# 3) 读取 top2 和 raw
# --------------------------------------------------
print("\n[Step 1/5] 读取 top2 参数表")
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 1, "top2 参数表为空"

print("[Step 2/5] 读取 raw 3D stack")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
print("✅ stack shape:", stack.shape, "dtype:", stack.dtype)

# --------------------------------------------------
# 4) 加载模型
# --------------------------------------------------
print("[Step 3/5] 加载模型")
model = models.CellposeModel(gpu=True, pretrained_model=selected_model_path)
print("✅ model loaded")

# --------------------------------------------------
# 5) 只跑 rank1，并尽快保存 mask
# --------------------------------------------------
summary_rows = []

for rank_idx, (_, row) in enumerate(top2_df.iterrows(), 1):
    if rank_idx != RUN_ONLY_RANK:
        print(f"⏭️ 跳过 rank#{rank_idx}")
        continue

    anisotropy = float(row["anisotropy"])
    diameter = float(row["diameter"])
    cellprob = float(row["cellprob_threshold"])
    stitch = float(row["stitch_threshold"])

    run_name = f"FINAL_rank{rank_idx}__" + run_name_from_params(
        TARGET_TAG, anisotropy, diameter, cellprob, stitch
    )

    out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
    out_stats_path = STAT_ROOT / f"{run_name}__stepA_minimal.json"

    print("\n" + "=" * 100)
    print(f"rank#{rank_idx}: a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}")
    print("mask exists:", out_mask_path.exists(), out_mask_path)
    print("=" * 100)

    if out_mask_path.exists():
        print("✅ 已存在 mask，直接跳过推理")
        elapsed_s = np.nan
        masks_shape = None
        masks_dtype = None
    else:
        print("[Step 4/5] 开始 eval")
        print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
        t0 = time.time()

        # ---- 核心推理 ----
        masks, flows, styles = model.eval(
            stack,
            diameter=diameter,
            do_3D=True,
            stitch_threshold=stitch,
            z_axis=0,
            batch_size=BATCH_SIZE_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob,
            progress=True
        )
        print("✅ eval returned")

        # ---- 立即转数组并保存 ----
        masks = np.asarray(masks)
        elapsed_s = float(time.time() - t0)
        masks_shape = list(masks.shape)
        masks_dtype = str(masks.dtype)

        print("✅ masks asarray done")
        print("   masks shape:", masks_shape)
        print("   masks dtype:", masks_dtype)
        print(f"   elapsed: {elapsed_s/60:.2f} min")

        print("[Step 5/5] 写出 mask tif")
        # 尽量避免不必要拷贝；若已经是整数类型，直接写
        if np.issubdtype(masks.dtype, np.integer):
            tiff.imwrite(str(out_mask_path), masks)
        else:
            tiff.imwrite(str(out_mask_path), masks.astype(np.uint32))

        print("✅ mask tif saved:", out_mask_path)

        # 释放非必要对象
        del flows, styles
        safe_cuda_cleanup()

    minimal_record = {
        "rank_idx": rank_idx,
        "target_tag": TARGET_TAG,
        "selected_model_path": selected_model_path,
        "selected_snapshot_path": selected_snapshot_path,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_path": str(out_mask_path),
        "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
        "anisotropy": anisotropy,
        "diameter_3d": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "masks_shape": masks_shape,
        "masks_dtype": masks_dtype,
        "step": "A_save_mask_only",
    }
    out_stats_path.write_text(
        json.dumps(minimal_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("✅ minimal json saved:", out_stats_path)

    summary_rows.append({
        "rank_idx": rank_idx,
        "run_name": run_name,
        "anisotropy": anisotropy,
        "diameter": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "elapsed_s": elapsed_s,
        "mask_path": str(out_mask_path),
        "status": "MASK_SAVED_ONLY"
    })

# 保存一个 stepA 简版 summary，避免污染你原来的 full summary 逻辑
stepA_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}__stepA_mask_only.csv"
pd.DataFrame(summary_rows).to_csv(stepA_summary_csv, index=False)

print("\n✅ StepA summary saved:", stepA_summary_csv)
display(pd.DataFrame(summary_rows))


# In[ ]:


# ==========================================
# Cell Rescue-Rank1-StepB（替换版）：
# 读取已保存 rank1 mask，做实例级统计 + 正确边界可视化 + Top-K 大对象局部排查
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

print("=" * 100)
print("🧪 Rescue-Rank1-StepB | Analyze saved rank1 mask with instance boundaries and local crops")
print("=" * 100)

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"
RUN_ONLY_RANK = 1

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()
FIG_ROOT = (FULL_ROOT / "figures").resolve()
CROP_ROOT = (FULL_ROOT / "figures" / "largest_obj_crops_rank1").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
full_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"

STAT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)
CROP_ROOT.mkdir(parents=True, exist_ok=True)

assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"BASELINE 不存在: {BASELINE_MASK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

# --------------------------------------------------
# 1) 可调参数
# --------------------------------------------------
Z_VIS_LIST = [98, 100, 102]     # 主对比图看的 z 层
TOPK_LARGEST = 8                # 排查最大的前 K 个对象
CROP_HALF_SIZE = 96             # 局部裁剪半径，实际 crop 大小约 192 x 192
SAVE_MAX_N_CROPS = 6            # 最多输出多少个局部 crop 图

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    """
    实例边界：
    只要相邻像素 label 不同，就认为是边界。
    这样能显示实例之间的内部边界，而不是只显示整体前景外轮廓。
    """
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)

    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))

    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr

    bd[0, :] = False
    bd[-1, :] = False
    bd[:, 0] = False
    bd[:, -1] = False
    return bd

def overlay_instance_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 1.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_labels = labels[valid]
    obj_sizes = counts[valid]

    if len(obj_sizes) == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "p95_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
            "n_ge_3x_median": 0,
            "n_ge_5x_median": 0,
            "top10_volumes": [],
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    p95_v = float(np.percentile(obj_sizes, 95))
    max_v = float(np.max(obj_sizes))
    med_safe = max(median_v, 1.0)

    large_ratio = float(np.mean(obj_sizes >= (2.0 * med_safe)))
    n_ge_3x = int(np.sum(obj_sizes >= 3.0 * med_safe))
    n_ge_5x = int(np.sum(obj_sizes >= 5.0 * med_safe))
    top10 = sorted(obj_sizes.tolist(), reverse=True)[:10]

    return {
        "total_cells": int(len(obj_sizes)),
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "p95_volume": p95_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
        "n_ge_3x_median": n_ge_3x,
        "n_ge_5x_median": n_ge_5x,
        "top10_volumes": top10,
    }

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

def get_topk_objects(mask3d: np.ndarray, topk=10):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    labels = labels[valid]
    counts = counts[valid]

    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    out = []
    for lab, vol in zip(labels[:topk], counts[:topk]):
        coords = np.argwhere(mask3d == lab)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)

        out.append({
            "label": int(lab),
            "volume": int(vol),
            "z_min": int(z0), "z_max": int(z1),
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "z_center": int(cz), "y_center": int(cy), "x_center": int(cx),
            "z_span": int(z1 - z0 + 1),
            "y_span": int(y1 - y0 + 1),
            "x_span": int(x1 - x0 + 1),
        })
    return out

def safe_crop_2d(arr2d, cy, cx, half_size=96):
    H, W = arr2d.shape
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size)
    return arr2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def save_largest_object_crops(
    raw_stack,
    baseline_masks,
    pred_masks,
    largest_objs,
    crop_root,
    max_n=6,
    half_size=96
):
    crop_rows = []

    for i, obj in enumerate(largest_objs[:max_n], 1):
        lab = obj["label"]
        cz, cy, cx = obj["z_center"], obj["y_center"], obj["x_center"]

        raw2d = raw_stack[cz]
        base2d = baseline_masks[cz]
        pred2d = pred_masks[cz]

        raw_crop, crop_box = safe_crop_2d(raw2d, cy, cx, half_size=half_size)
        base_crop, _ = safe_crop_2d(base2d, cy, cx, half_size=half_size)
        pred_crop, _ = safe_crop_2d(pred2d, cy, cx, half_size=half_size)

        raw_norm = normalize_img(raw_crop)
        base_overlay = overlay_instance_boundary(raw_crop, base_crop, color="red")
        pred_overlay = overlay_instance_boundary(raw_crop, pred_crop, color="green")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title(
            f"Raw crop\nz={cz}, y={cy}, x={cx}"
        )
        axes[0].axis("off")

        axes[1].imshow(base_overlay)
        axes[1].set_title("Baseline boundary")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(
            f"Pred boundary\nlabel={lab}, vol={obj['volume']}\n"
            f"zspan={obj['z_span']} yspan={obj['y_span']} xspan={obj['x_span']}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        crop_path = crop_root / f"largest_obj_rank{i:02d}__label{lab}__z{cz}_y{cy}_x{cx}.png"
        plt.savefig(crop_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        crop_rows.append({
            "rank_in_largest": i,
            "label": lab,
            "volume": obj["volume"],
            "z_center": cz,
            "y_center": cy,
            "x_center": cx,
            "z_span": obj["z_span"],
            "y_span": obj["y_span"],
            "x_span": obj["x_span"],
            "crop_path": str(crop_path),
            "crop_box_y0y1x0x1": str(crop_box),
        })

    return crop_rows

# --------------------------------------------------
# 3) 读取参数表并锁定 rank1
# --------------------------------------------------
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 1, "top2 参数表为空"

row = top2_df.iloc[RUN_ONLY_RANK - 1]
anisotropy = float(row["anisotropy"])
diameter = float(row["diameter"])
cellprob = float(row["cellprob_threshold"])
stitch = float(row["stitch_threshold"])

run_name = f"FINAL_rank{RUN_ONLY_RANK}__" + run_name_from_params(
    TARGET_TAG, anisotropy, diameter, cellprob, stitch
)

out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
out_stats_path = STAT_ROOT / f"{run_name}__stats.json"
out_fig_path = FIG_ROOT / f"{run_name}__compare_multiz.png"
out_largest_csv = STAT_ROOT / f"{run_name}__largest_objects.csv"
out_largest_summary_json = STAT_ROOT / f"{run_name}__largest_objects.json"
out_crop_csv = STAT_ROOT / f"{run_name}__largest_object_crops.csv"

assert out_mask_path.exists(), f"mask tif 不存在，请先跑 StepA: {out_mask_path}"

print("✅ mask path:", out_mask_path)

# --------------------------------------------------
# 4) 读取数据
# --------------------------------------------------
print("[1/5] 读取 raw / baseline / saved mask")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))
masks = tiff.imread(str(out_mask_path))

assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
assert baseline_masks.shape == stack.shape, "baseline 与 raw shape 不一致"
assert masks.shape == stack.shape, "saved mask 与 raw shape 不一致"

print("✅ stack shape     :", stack.shape, stack.dtype)
print("✅ baseline shape  :", baseline_masks.shape, baseline_masks.dtype)
print("✅ saved mask shape:", masks.shape, masks.dtype)

# --------------------------------------------------
# 5) 统计 mask
# --------------------------------------------------
print("[2/5] 统计 mask")
stats = summarize_mask(masks)
print("✅ stats:")
for k, v in stats.items():
    print(f"   - {k}: {v}")

largest_objs = get_topk_objects(masks, topk=TOPK_LARGEST)
largest_df = pd.DataFrame(largest_objs)
largest_df.to_csv(out_largest_csv, index=False)
print("✅ largest object table saved:", out_largest_csv)

# --------------------------------------------------
# 6) 多 z 层主对比图（实例边界）
# --------------------------------------------------
print("[3/5] 生成多 z 层实例边界对比图")
valid_z_list = [z for z in Z_VIS_LIST if 0 <= z < stack.shape[0]]
assert len(valid_z_list) > 0, "Z_VIS_LIST 全部越界"

fig, axes = plt.subplots(len(valid_z_list), 3, figsize=(14, 4.2 * len(valid_z_list)))
if len(valid_z_list) == 1:
    axes = np.array([axes])

for i, z in enumerate(valid_z_list):
    raw2d = stack[z]
    base2d = baseline_masks[z]
    pred2d = masks[z]

    raw_norm = normalize_img(raw2d)
    base_overlay = overlay_instance_boundary(raw2d, base2d, color="red")
    pred_overlay = overlay_instance_boundary(raw2d, pred2d, color="green")

    axes[i, 0].imshow(raw_norm, cmap="gray")
    axes[i, 0].set_title(f"Raw | z={z}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(base_overlay)
    axes[i, 1].set_title(f"Baseline | z={z}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_overlay)
    axes[i, 2].set_title(
        f"Rank#{RUN_ONLY_RANK} | z={z}\n"
        f"a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}\n"
        f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
        f"maxV={stats['max_volume']:.1f} | n>=3xmed={stats['n_ge_3x_median']}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("✅ figure saved:", out_fig_path)

# --------------------------------------------------
# 7) 最大对象局部 crop 排查
# --------------------------------------------------
print("[4/5] 生成最大对象局部 crop 排查图")
crop_rows = save_largest_object_crops(
    raw_stack=stack,
    baseline_masks=baseline_masks,
    pred_masks=masks,
    largest_objs=largest_objs,
    crop_root=CROP_ROOT,
    max_n=SAVE_MAX_N_CROPS,
    half_size=CROP_HALF_SIZE
)
crop_df = pd.DataFrame(crop_rows)
crop_df.to_csv(out_crop_csv, index=False)
print("✅ largest object crop table saved:", out_crop_csv)

largest_summary = {
    "run_name": run_name,
    "topk_largest": largest_objs,
    "crop_rows": crop_rows,
}
out_largest_summary_json.write_text(
    json.dumps(largest_summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ largest object summary json saved:", out_largest_summary_json)

# --------------------------------------------------
# 8) 写 stats + 更新 full summary
# --------------------------------------------------
print("[5/5] 写 stats 和汇总表")
stats_record = {
    "rank_idx": RUN_ONLY_RANK,
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_json": str(out_largest_summary_json),
    "largest_object_crop_csv": str(out_crop_csv),
    "shape": list(masks.shape),
    "anisotropy": anisotropy,
    "diameter_3d": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    **stats
}
out_stats_path.write_text(
    json.dumps(stats_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ stats json saved:", out_stats_path)

new_full_df = pd.DataFrame([{
    "rank_idx": RUN_ONLY_RANK,
    "run_name": run_name,
    "anisotropy": anisotropy,
    "diameter": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    "elapsed_s": np.nan,
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_crop_csv": str(out_crop_csv),
    **stats
}])

if full_summary_csv.exists():
    old_full_df = pd.read_csv(full_summary_csv)
else:
    old_full_df = pd.DataFrame()

if len(old_full_df) > 0:
    merged_full_df = pd.concat([old_full_df, new_full_df], ignore_index=True)
    merged_full_df = merged_full_df.drop_duplicates(subset=["rank_idx"], keep="last").sort_values("rank_idx")
else:
    merged_full_df = new_full_df

merged_full_df.to_csv(full_summary_csv, index=False)

print("\n✅ full summary saved:", full_summary_csv)
print("\n✅ largest objects:")
display(largest_df)

print("\n✅ largest object crops:")
display(crop_df)

print("\n✅ merged full summary:")
display(merged_full_df)


# In[ ]:


# ==========================================
# Cell Rescue-Rank2-StepA：只跑 full-brain rank2，并优先保存 mask
# ==========================================
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🚑 Rescue-Rank2-StepA | Run full-brain rank2 only and save mask first")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
CFG_DIR = (EXP_DIR / "config").resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"

MASK_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)

assert CFG_DIR.exists(), f"CFG_DIR 不存在: {CFG_DIR}"
assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

RUN_ONLY_RANK = 2
BATCH_SIZE_3D = 8   # 如果还炸，改成 2

# --------------------------------------------------
# 1) 工具函数
# --------------------------------------------------
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# --------------------------------------------------
# 2) 找模型
# --------------------------------------------------
config_files = sorted(CFG_DIR.glob("config_*.json"))
selected_model_path = None
selected_snapshot_path = None

for sf in config_files:
    snap = read_json(sf)
    if not isinstance(snap, dict):
        continue
    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")
    if tag != TARGET_TAG:
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

assert selected_model_path is not None, f"没找到 {TARGET_TAG} 对应模型"

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

# --------------------------------------------------
# 3) 读取 top2 和 raw
# --------------------------------------------------
print("\n[Step 1/5] 读取 top2 参数表")
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 2, "top2 参数表少于 2 条"

print("[Step 2/5] 读取 raw 3D stack")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
print("✅ stack shape:", stack.shape, "dtype:", stack.dtype)

# --------------------------------------------------
# 4) 加载模型
# --------------------------------------------------
print("[Step 3/5] 加载模型")
model = models.CellposeModel(gpu=True, pretrained_model=selected_model_path)
print("✅ model loaded")

# --------------------------------------------------
# 5) 只跑 rank2，并尽快保存 mask
# --------------------------------------------------
summary_rows = []

for rank_idx, (_, row) in enumerate(top2_df.iterrows(), 1):
    if rank_idx != RUN_ONLY_RANK:
        print(f"⏭️ 跳过 rank#{rank_idx}")
        continue

    anisotropy = float(row["anisotropy"])
    diameter = float(row["diameter"])
    cellprob = float(row["cellprob_threshold"])
    stitch = float(row["stitch_threshold"])

    run_name = f"FINAL_rank{rank_idx}__" + run_name_from_params(
        TARGET_TAG, anisotropy, diameter, cellprob, stitch
    )

    out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
    out_stats_path = STAT_ROOT / f"{run_name}__stepA_minimal.json"

    print("\n" + "=" * 100)
    print(f"rank#{rank_idx}: a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}")
    print("mask exists:", out_mask_path.exists(), out_mask_path)
    print("=" * 100)

    masks_shape = None
    masks_dtype = None

    if out_mask_path.exists():
        print("✅ 已存在 mask，直接跳过推理")
        elapsed_s = np.nan
    else:
        print("[Step 4/5] 开始 eval")
        print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
        t0 = time.time()

        masks, flows, styles = model.eval(
            stack,
            diameter=diameter,
            do_3D=True,
            stitch_threshold=stitch,
            z_axis=0,
            batch_size=BATCH_SIZE_3D,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob,
            progress=True
        )
        print("✅ eval returned")

        masks = np.asarray(masks)
        elapsed_s = float(time.time() - t0)
        masks_shape = list(masks.shape)
        masks_dtype = str(masks.dtype)

        print("✅ masks asarray done")
        print("   masks shape:", masks_shape)
        print("   masks dtype:", masks_dtype)
        print(f"   elapsed: {elapsed_s/60:.2f} min")

        print("[Step 5/5] 写出 mask tif")
        if np.issubdtype(masks.dtype, np.integer):
            tiff.imwrite(str(out_mask_path), masks)
        else:
            tiff.imwrite(str(out_mask_path), masks.astype(np.uint32))

        print("✅ mask tif saved:", out_mask_path)

        del flows, styles
        safe_cuda_cleanup()

    minimal_record = {
        "rank_idx": rank_idx,
        "target_tag": TARGET_TAG,
        "selected_model_path": selected_model_path,
        "selected_snapshot_path": selected_snapshot_path,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_path": str(out_mask_path),
        "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
        "anisotropy": anisotropy,
        "diameter_3d": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "masks_shape": masks_shape,
        "masks_dtype": masks_dtype,
        "step": "A_save_mask_only",
    }
    out_stats_path.write_text(
        json.dumps(minimal_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("✅ minimal json saved:", out_stats_path)

    summary_rows.append({
        "rank_idx": rank_idx,
        "run_name": run_name,
        "anisotropy": anisotropy,
        "diameter": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "elapsed_s": elapsed_s,
        "mask_path": str(out_mask_path),
        "status": "MASK_SAVED_ONLY"
    })

stepA_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}__rank2_stepA_mask_only.csv"
pd.DataFrame(summary_rows).to_csv(stepA_summary_csv, index=False)

print("\n✅ StepA summary saved:", stepA_summary_csv)
display(pd.DataFrame(summary_rows))


# In[ ]:


# ==========================================
# Cell Rescue-Rank2-StepB（替换版）：
# 读取已保存 rank2 mask，做实例级统计 + 正确边界可视化 + Top-K 大对象局部排查
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

print("=" * 100)
print("🧪 Rescue-Rank2-StepB | Analyze saved rank2 mask with instance boundaries and local crops")
print("=" * 100)

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"
RUN_ONLY_RANK = 2

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()
FIG_ROOT = (FULL_ROOT / "figures").resolve()
CROP_ROOT = (FULL_ROOT / "figures" / "largest_obj_crops").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
full_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"

STAT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)
CROP_ROOT.mkdir(parents=True, exist_ok=True)

assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"BASELINE 不存在: {BASELINE_MASK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

# --------------------------------------------------
# 1) 可调参数
# --------------------------------------------------
Z_VIS_LIST = [98, 100, 102]     # 主对比图看的 z 层
TOPK_LARGEST = 8                # 排查最大的前 K 个对象
CROP_HALF_SIZE = 96             # 局部裁剪半径，实际 crop 大小约 192 x 192
SAVE_MAX_N_CROPS = 6            # 最多输出多少个局部 crop 图

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    """
    实例边界：
    只要相邻像素 label 不同，就认为是边界。
    和原来的二值前景边界不同，这样能看见实例之间的内部边界。
    """
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)

    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))

    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr

    bd[0, :] = False
    bd[-1, :] = False
    bd[:, 0] = False
    bd[:, -1] = False
    return bd

def overlay_instance_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 1.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_labels = labels[valid]
    obj_sizes = counts[valid]

    if len(obj_sizes) == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "p95_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
            "n_ge_3x_median": 0,
            "n_ge_5x_median": 0,
            "top10_volumes": [],
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    p95_v = float(np.percentile(obj_sizes, 95))
    max_v = float(np.max(obj_sizes))
    med_safe = max(median_v, 1.0)

    large_ratio = float(np.mean(obj_sizes >= (2.0 * med_safe)))
    n_ge_3x = int(np.sum(obj_sizes >= 3.0 * med_safe))
    n_ge_5x = int(np.sum(obj_sizes >= 5.0 * med_safe))
    top10 = sorted(obj_sizes.tolist(), reverse=True)[:10]

    return {
        "total_cells": int(len(obj_sizes)),
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "p95_volume": p95_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
        "n_ge_3x_median": n_ge_3x,
        "n_ge_5x_median": n_ge_5x,
        "top10_volumes": top10,
    }

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

def get_topk_objects(mask3d: np.ndarray, topk=10):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    labels = labels[valid]
    counts = counts[valid]

    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    out = []
    for lab, vol in zip(labels[:topk], counts[:topk]):
        coords = np.argwhere(mask3d == lab)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)

        out.append({
            "label": int(lab),
            "volume": int(vol),
            "z_min": int(z0), "z_max": int(z1),
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "z_center": int(cz), "y_center": int(cy), "x_center": int(cx),
            "z_span": int(z1 - z0 + 1),
            "y_span": int(y1 - y0 + 1),
            "x_span": int(x1 - x0 + 1),
        })
    return out

def safe_crop_2d(arr2d, cy, cx, half_size=96):
    H, W = arr2d.shape
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size)
    return arr2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def save_largest_object_crops(
    raw_stack,
    baseline_masks,
    pred_masks,
    largest_objs,
    crop_root,
    max_n=6,
    half_size=96
):
    crop_rows = []

    for i, obj in enumerate(largest_objs[:max_n], 1):
        lab = obj["label"]
        cz, cy, cx = obj["z_center"], obj["y_center"], obj["x_center"]

        raw2d = raw_stack[cz]
        base2d = baseline_masks[cz]
        pred2d = pred_masks[cz]

        raw_crop, crop_box = safe_crop_2d(raw2d, cy, cx, half_size=half_size)
        base_crop, _ = safe_crop_2d(base2d, cy, cx, half_size=half_size)
        pred_crop, _ = safe_crop_2d(pred2d, cy, cx, half_size=half_size)

        raw_norm = normalize_img(raw_crop)
        base_overlay = overlay_instance_boundary(raw_crop, base_crop, color="red")
        pred_overlay = overlay_instance_boundary(raw_crop, pred_crop, color="green")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title(
            f"Raw crop\nz={cz}, y={cy}, x={cx}"
        )
        axes[0].axis("off")

        axes[1].imshow(base_overlay)
        axes[1].set_title("Baseline boundary")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(
            f"Pred boundary\nlabel={lab}, vol={obj['volume']}\n"
            f"zspan={obj['z_span']} yspan={obj['y_span']} xspan={obj['x_span']}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        crop_path = crop_root / f"largest_obj_rank{i:02d}__label{lab}__z{cz}_y{cy}_x{cx}.png"
        plt.savefig(crop_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        crop_rows.append({
            "rank_in_largest": i,
            "label": lab,
            "volume": obj["volume"],
            "z_center": cz,
            "y_center": cy,
            "x_center": cx,
            "z_span": obj["z_span"],
            "y_span": obj["y_span"],
            "x_span": obj["x_span"],
            "crop_path": str(crop_path),
            "crop_box_y0y1x0x1": str(crop_box),
        })

    return crop_rows

# --------------------------------------------------
# 3) 读取参数表并锁定 rank2
# --------------------------------------------------
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 2, "top2 参数表少于 2 条"

row = top2_df.iloc[RUN_ONLY_RANK - 1]
anisotropy = float(row["anisotropy"])
diameter = float(row["diameter"])
cellprob = float(row["cellprob_threshold"])
stitch = float(row["stitch_threshold"])

run_name = f"FINAL_rank{RUN_ONLY_RANK}__" + run_name_from_params(
    TARGET_TAG, anisotropy, diameter, cellprob, stitch
)

out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
out_stats_path = STAT_ROOT / f"{run_name}__stats.json"
out_fig_path = FIG_ROOT / f"{run_name}__compare_multiz.png"
out_largest_csv = STAT_ROOT / f"{run_name}__largest_objects.csv"
out_largest_summary_json = STAT_ROOT / f"{run_name}__largest_objects.json"

assert out_mask_path.exists(), f"mask tif 不存在，请先跑 StepA: {out_mask_path}"

print("✅ mask path:", out_mask_path)

# --------------------------------------------------
# 4) 读取数据
# --------------------------------------------------
print("[1/5] 读取 raw / baseline / saved mask")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))
masks = tiff.imread(str(out_mask_path))

assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
assert baseline_masks.shape == stack.shape, "baseline 与 raw shape 不一致"
assert masks.shape == stack.shape, "saved mask 与 raw shape 不一致"

print("✅ stack shape     :", stack.shape, stack.dtype)
print("✅ baseline shape  :", baseline_masks.shape, baseline_masks.dtype)
print("✅ saved mask shape:", masks.shape, masks.dtype)

# --------------------------------------------------
# 5) 统计 mask
# --------------------------------------------------
print("[2/5] 统计 mask")
stats = summarize_mask(masks)
print("✅ stats:")
for k, v in stats.items():
    print(f"   - {k}: {v}")

largest_objs = get_topk_objects(masks, topk=TOPK_LARGEST)
largest_df = pd.DataFrame(largest_objs)
largest_df.to_csv(out_largest_csv, index=False)
print("✅ largest object table saved:", out_largest_csv)

# --------------------------------------------------
# 6) 多 z 层主对比图（实例边界）
# --------------------------------------------------
print("[3/5] 生成多 z 层实例边界对比图")
valid_z_list = [z for z in Z_VIS_LIST if 0 <= z < stack.shape[0]]
assert len(valid_z_list) > 0, "Z_VIS_LIST 全部越界"

fig, axes = plt.subplots(len(valid_z_list), 3, figsize=(14, 4.2 * len(valid_z_list)))
if len(valid_z_list) == 1:
    axes = np.array([axes])

for i, z in enumerate(valid_z_list):
    raw2d = stack[z]
    base2d = baseline_masks[z]
    pred2d = masks[z]

    raw_norm = normalize_img(raw2d)
    base_overlay = overlay_instance_boundary(raw2d, base2d, color="red")
    pred_overlay = overlay_instance_boundary(raw2d, pred2d, color="green")

    axes[i, 0].imshow(raw_norm, cmap="gray")
    axes[i, 0].set_title(f"Raw | z={z}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(base_overlay)
    axes[i, 1].set_title(f"Baseline | z={z}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_overlay)
    axes[i, 2].set_title(
        f"Pred | z={z}\n"
        f"a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}\n"
        f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
        f"maxV={stats['max_volume']:.1f} | n>=3xmed={stats['n_ge_3x_median']}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("✅ figure saved:", out_fig_path)

# --------------------------------------------------
# 7) 最大对象局部 crop 排查
# --------------------------------------------------
print("[4/5] 生成最大对象局部 crop 排查图")
crop_rows = save_largest_object_crops(
    raw_stack=stack,
    baseline_masks=baseline_masks,
    pred_masks=masks,
    largest_objs=largest_objs,
    crop_root=CROP_ROOT,
    max_n=SAVE_MAX_N_CROPS,
    half_size=CROP_HALF_SIZE
)
crop_df = pd.DataFrame(crop_rows)

out_crop_csv = STAT_ROOT / f"{run_name}__largest_object_crops.csv"
crop_df.to_csv(out_crop_csv, index=False)
print("✅ largest object crop table saved:", out_crop_csv)

largest_summary = {
    "run_name": run_name,
    "topk_largest": largest_objs,
    "crop_rows": crop_rows,
}
out_largest_summary_json.write_text(
    json.dumps(largest_summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ largest object summary json saved:", out_largest_summary_json)

# --------------------------------------------------
# 8) 写 stats + 更新 full summary
# --------------------------------------------------
print("[5/5] 写 stats 和汇总表")
stats_record = {
    "rank_idx": RUN_ONLY_RANK,
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_json": str(out_largest_summary_json),
    "largest_object_crop_csv": str(out_crop_csv),
    "shape": list(masks.shape),
    "anisotropy": anisotropy,
    "diameter_3d": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    **stats
}
out_stats_path.write_text(
    json.dumps(stats_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ stats json saved:", out_stats_path)

new_full_df = pd.DataFrame([{
    "rank_idx": RUN_ONLY_RANK,
    "run_name": run_name,
    "anisotropy": anisotropy,
    "diameter": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    "elapsed_s": np.nan,
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_crop_csv": str(out_crop_csv),
    **stats
}])

if full_summary_csv.exists():
    old_full_df = pd.read_csv(full_summary_csv)
else:
    old_full_df = pd.DataFrame()

if len(old_full_df) > 0:
    merged_full_df = pd.concat([old_full_df, new_full_df], ignore_index=True)
    merged_full_df = merged_full_df.drop_duplicates(subset=["rank_idx"], keep="last").sort_values("rank_idx")
else:
    merged_full_df = new_full_df

merged_full_df.to_csv(full_summary_csv, index=False)

print("\n✅ rank2 full summary saved:", full_summary_csv)
print("\n✅ 最大对象表：")
display(largest_df)

print("\n✅ 最大对象 crop 表：")
display(crop_df)

print("\n✅ 汇总表：")
display(merged_full_df)


# In[ ]:


# ==========================================
# Cell Pack-Results：打包 rank1 / rank2 分析结果为 zip
# ==========================================
from pathlib import Path
import zipfile
import pandas as pd
from datetime import datetime

print("=" * 100)
print("📦 Pack-Results | Collect useful figures / csv / json and zip them")
print("=" * 100)

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()
FIG_ROOT = (FULL_ROOT / "figures").resolve()

PACK_ROOT = (FULL_ROOT / "delivery").resolve()
PACK_ROOT.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_PATH = PACK_ROOT / f"{TARGET_TAG}__rank1_rank2_analysis_delivery_{ts}.zip"
MANIFEST_CSV = PACK_ROOT / f"{TARGET_TAG}__rank1_rank2_analysis_delivery_manifest_{ts}.csv"

assert MERGE_ROOT.exists(), f"MERGE_ROOT 不存在: {MERGE_ROOT}"
assert FULL_ROOT.exists(), f"FULL_ROOT 不存在: {FULL_ROOT}"

print("✅ MERGE_ROOT :", MERGE_ROOT)
print("✅ FULL_ROOT  :", FULL_ROOT)
print("✅ ZIP_PATH   :", ZIP_PATH)

# --------------------------------------------------
# 1) 收集候选文件
# --------------------------------------------------
candidate_files = []

# ---- MERGE_ROOT 下的重要汇总 ----
merge_patterns = [
    f"top2_params_{TARGET_TAG}.csv",
    f"full_brain_top2_summary_{TARGET_TAG}.csv",
    f"full_brain_top2_summary_{TARGET_TAG}__rank2_stepA_mask_only.csv",
]

for pat in merge_patterns:
    for p in MERGE_ROOT.glob(pat):
        if p.exists() and p.is_file():
            candidate_files.append(p)

# ---- stats 下的重要文件 ----
if STAT_ROOT.exists():
    stats_patterns = [
        "*.json",
        "*largest_objects.csv",
        "*largest_object_crops.csv",
        "*largest_objects.json",
    ]
    for pat in stats_patterns:
        for p in STAT_ROOT.glob(pat):
            if p.exists() and p.is_file():
                candidate_files.append(p)

# ---- figures 下的重要图 ----
if FIG_ROOT.exists():
    fig_patterns = [
        "*compare*.png",
        "largest_obj_crops/*.png",
        "largest_obj_crops_rank1/*.png",
    ]
    for pat in fig_patterns:
        for p in FIG_ROOT.glob(pat):
            if p.exists() and p.is_file():
                candidate_files.append(p)

# 去重 + 排序
candidate_files = sorted(set(candidate_files))

print(f"✅ 找到候选文件数: {len(candidate_files)}")

# --------------------------------------------------
# 2) 生成 manifest
# --------------------------------------------------
manifest_rows = []
for p in candidate_files:
    try:
        rel = p.relative_to(FULL_ROOT)
    except Exception:
        rel = p.name

    manifest_rows.append({
        "filename": p.name,
        "relative_path": str(rel),
        "abs_path": str(p),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 4),
        "suffix": p.suffix.lower(),
        "parent": p.parent.name,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
print("✅ manifest saved:", MANIFEST_CSV)

# 把 manifest 自己也加入 zip
files_to_zip = candidate_files + [MANIFEST_CSV]

# --------------------------------------------------
# 3) 写 zip
# --------------------------------------------------
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in files_to_zip:
        try:
            arcname = p.relative_to(FULL_ROOT)
            arcname = Path(TARGET_TAG) / arcname
        except Exception:
            arcname = Path(TARGET_TAG) / p.name

        zf.write(p, arcname=str(arcname))
        print("  +", arcname)

print("\n✅ 打包完成:", ZIP_PATH)

# --------------------------------------------------
# 4) 输出摘要
# --------------------------------------------------
total_size_mb = sum(p.stat().st_size for p in files_to_zip if p.exists()) / 1024 / 1024

print("\n" + "=" * 100)
print("📦 打包摘要")
print("=" * 100)
print("zip 文件 :", ZIP_PATH)
print("manifest :", MANIFEST_CSV)
print(f"文件数量 : {len(files_to_zip)}")
print(f"总大小   : {total_size_mb:.2f} MB")

display(manifest_df.sort_values(["suffix", "relative_path"]).reset_index(drop=True))


# In[ ]:





# In[ ]:


# ==========================================
# Cell Baseline-StepA：用 baseline(cpsam) + rank1 参数跑 full-brain，并优先保存 mask
# ==========================================
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Baseline-StepA | Run baseline(cpsam) with rank1 params and save mask first")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
BASELINE_ROOT = (MERGE_ROOT / "Baseline_cpsam_rank1params").resolve()
MASK_ROOT = (BASELINE_ROOT / "masks").resolve()
STAT_ROOT = (BASELINE_ROOT / "stats").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"

MASK_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)

assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

RUN_ONLY_RANK = 1
BATCH_SIZE_3D = 8   # 如果还炸，再改成 2

# --------------------------------------------------
# 1) 工具函数
# --------------------------------------------------
def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# --------------------------------------------------
# 2) 读取 rank1 参数和 raw
# --------------------------------------------------
print("\n[Step 1/4] 读取 rank1 参数")
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 1, "top2 参数表为空"

row = top2_df.iloc[RUN_ONLY_RANK - 1]
anisotropy = float(row["anisotropy"])
diameter = float(row["diameter"])
cellprob = float(row["cellprob_threshold"])
stitch = float(row["stitch_threshold"])

print("✅ rank1 params:")
print("   anisotropy        :", anisotropy)
print("   diameter          :", diameter)
print("   cellprob_threshold:", cellprob)
print("   stitch_threshold  :", stitch)

print("[Step 2/4] 读取 raw 3D stack")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
print("✅ stack shape:", stack.shape, "dtype:", stack.dtype)

# --------------------------------------------------
# 3) 加载 baseline 模型
# --------------------------------------------------
print("[Step 3/4] 加载 baseline 模型 cpsam")
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
print("✅ baseline model loaded: cpsam")

# --------------------------------------------------
# 4) 推理并优先保存 mask
# --------------------------------------------------
run_name = "BASELINE_cpsam__rank1params__" + run_name_from_params(
    TARGET_TAG, anisotropy, diameter, cellprob, stitch
)

out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
out_stats_path = STAT_ROOT / f"{run_name}__stepA_minimal.json"

print("\n" + "=" * 100)
print(f"Baseline(cpsam) with rank1 params")
print("mask exists:", out_mask_path.exists(), out_mask_path)
print("=" * 100)

masks_shape = None
masks_dtype = None

if out_mask_path.exists():
    print("✅ 已存在 baseline mask，直接跳过推理")
    elapsed_s = np.nan
else:
    print("[Step 4/4] 开始 eval")
    print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        diameter=diameter,
        do_3D=True,
        stitch_threshold=stitch,
        z_axis=0,
        batch_size=BATCH_SIZE_3D,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob,
        progress=True
    )
    print("✅ eval returned")

    masks = np.asarray(masks)
    elapsed_s = float(time.time() - t0)
    masks_shape = list(masks.shape)
    masks_dtype = str(masks.dtype)

    print("✅ masks asarray done")
    print("   masks shape:", masks_shape)
    print("   masks dtype:", masks_dtype)
    print(f"   elapsed: {elapsed_s/60:.2f} min")

    print("写出 baseline mask tif")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(out_mask_path), masks)
    else:
        tiff.imwrite(str(out_mask_path), masks.astype(np.uint32))

    print("✅ baseline mask tif saved:", out_mask_path)

    del flows, styles
    safe_cuda_cleanup()

minimal_record = {
    "model_name": "cpsam",
    "param_source": "rank1_of_top2_csv",
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "mask_path": str(out_mask_path),
    "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
    "anisotropy": anisotropy,
    "diameter_3d": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    "masks_shape": masks_shape,
    "masks_dtype": masks_dtype,
    "step": "A_save_mask_only",
}
out_stats_path.write_text(
    json.dumps(minimal_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ minimal json saved:", out_stats_path)

stepA_summary_csv = BASELINE_ROOT / "baseline_cpsam_rank1params__stepA_mask_only.csv"
pd.DataFrame([{
    "model_name": "cpsam",
    "run_name": run_name,
    "anisotropy": anisotropy,
    "diameter": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    "elapsed_s": elapsed_s,
    "mask_path": str(out_mask_path),
    "status": "MASK_SAVED_ONLY"
}]).to_csv(stepA_summary_csv, index=False)

print("\n✅ StepA summary saved:", stepA_summary_csv)
display(pd.read_csv(stepA_summary_csv))


# In[ ]:


import subprocess
print(subprocess.run(["which", "sbatch"], capture_output=True, text=True).stdout)


# In[ ]:


# ==========================================
# Cell Baseline-StepB：分析 baseline(cpsam)+rank1参数 的 mask，并与 rank1/rank2 汇总对比
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

print("=" * 100)
print("🧪 Baseline-StepB | Analyze baseline(cpsam)+rank1params and compare with fine-tuned runs")
print("=" * 100)

# --------------------------------------------------
# 0) 固定路径
# --------------------------------------------------
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"
RUN_ONLY_RANK = 1
Z_VIS = 100

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_NATIVE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
BASELINE_ROOT = (MERGE_ROOT / "Baseline_cpsam_rank1params").resolve()
MASK_ROOT = (BASELINE_ROOT / "masks").resolve()
STAT_ROOT = (BASELINE_ROOT / "stats").resolve()
FIG_ROOT = (BASELINE_ROOT / "figures").resolve()

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
finetune_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"
baseline_compare_csv = MERGE_ROOT / f"compare_baseline_cpsam_rank1params_vs_finetune_{TARGET_TAG}.csv"

STAT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

assert RAW_3D_STACK_PATH.exists(), f"RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_NATIVE_MASK_PATH.exists(), f"baseline native mask 不存在: {BASELINE_NATIVE_MASK_PATH}"
assert top2_csv.exists(), f"缺少 top2 参数表: {top2_csv}"

# --------------------------------------------------
# 1) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary(mask2d: np.ndarray):
    fg = (mask2d > 0).astype(np.uint8)
    up    = np.roll(fg, -1, axis=0)
    down  = np.roll(fg,  1, axis=0)
    left  = np.roll(fg, -1, axis=1)
    right = np.roll(fg,  1, axis=1)
    bd = fg & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    bd[0, :] = 0
    bd[-1, :] = 0
    bd[:, 0] = 0
    bd[:, -1] = 0
    return bd.astype(bool)

def overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary(mask2d)
    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "blue":
        overlay[bd] = [0.0, 0.8, 1.0]
    else:
        overlay[bd] = [1.0, 1.0, 0.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_sizes = counts[valid]
    total_cells = int(valid.sum())
    if total_cells == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
        }
    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    max_v = float(np.max(obj_sizes))
    large_ratio = float(np.mean(obj_sizes >= (2.0 * max(median_v, 1.0))))
    return {
        "total_cells": total_cells,
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
    }

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"

# --------------------------------------------------
# 2) 锁定 baseline(cpsam)+rank1参数 的 mask
# --------------------------------------------------
top2_df = pd.read_csv(top2_csv)
assert len(top2_df) >= 1, "top2 参数表为空"

row = top2_df.iloc[RUN_ONLY_RANK - 1]
anisotropy = float(row["anisotropy"])
diameter = float(row["diameter"])
cellprob = float(row["cellprob_threshold"])
stitch = float(row["stitch_threshold"])

run_name = "BASELINE_cpsam__rank1params__" + run_name_from_params(
    TARGET_TAG, anisotropy, diameter, cellprob, stitch
)

out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
out_stats_path = STAT_ROOT / f"{run_name}__stats.json"
out_fig_path = FIG_ROOT / f"{run_name}__compare_z{Z_VIS}.png"

assert out_mask_path.exists(), f"baseline mask tif 不存在，请先跑 StepA: {out_mask_path}"

print("✅ baseline mask path:", out_mask_path)

# --------------------------------------------------
# 3) 读取数据并统计
# --------------------------------------------------
print("[1/4] 读取 raw / baseline-native / saved-baseline-rank1params mask")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_native_masks = tiff.imread(str(BASELINE_NATIVE_MASK_PATH))
masks = tiff.imread(str(out_mask_path))

assert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"
assert baseline_native_masks.shape == stack.shape, "baseline native 与 raw shape 不一致"
assert masks.shape == stack.shape, "saved baseline-rank1params mask 与 raw shape 不一致"

print("✅ stack shape              :", stack.shape, stack.dtype)
print("✅ baseline native shape    :", baseline_native_masks.shape, baseline_native_masks.dtype)
print("✅ saved baseline-rank1 shape:", masks.shape, masks.dtype)

print("[2/4] 统计 baseline-rank1params mask")
stats = summarize_mask(masks)
print("✅ stats:", stats)

# --------------------------------------------------
# 4) 出图
# --------------------------------------------------
print("[3/4] 生成对比图")
raw_mid = stack[Z_VIS]
base_native_mid = baseline_native_masks[Z_VIS]
this_mid = masks[Z_VIS]

raw_norm = normalize_img(raw_mid)
base_native_overlay = overlay_boundary(raw_mid, base_native_mid, color="red")
this_overlay = overlay_boundary(raw_mid, this_mid, color="blue")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(raw_norm, cmap="gray")
axes[0].set_title(f"Raw | z={Z_VIS}")
axes[0].axis("off")

axes[1].imshow(base_native_overlay)
axes[1].set_title("Baseline_native_saved")
axes[1].axis("off")

axes[2].imshow(this_overlay)
axes[2].set_title(
    f"Baseline cpsam + rank1 params\n"
    f"a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}\n"
    f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
    f"largeR={stats['large_obj_ratio_ge_2x_median']:.3f}"
)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("✅ figure saved:", out_fig_path)

# --------------------------------------------------
# 5) 写 stats + 生成对比表
# --------------------------------------------------
print("[4/4] 写 stats 和 compare summary")
stats_record = {
    "model_name": "cpsam",
    "param_source": "rank1_of_top2_csv",
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_native_mask_path": str(BASELINE_NATIVE_MASK_PATH),
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    "shape": list(masks.shape),
    "anisotropy": anisotropy,
    "diameter_3d": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    **stats
}
out_stats_path.write_text(
    json.dumps(stats_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ stats json saved:", out_stats_path)

baseline_row = pd.DataFrame([{
    "model_type": "baseline_cpsam_rank1params",
    "rank_idx": 0,
    "run_name": run_name,
    "anisotropy": anisotropy,
    "diameter": diameter,
    "cellprob_threshold": cellprob,
    "stitch_threshold": stitch,
    "elapsed_s": np.nan,
    "mask_path": str(out_mask_path),
    "fig_path": str(out_fig_path),
    **stats
}])

if finetune_summary_csv.exists():
    fine_df = pd.read_csv(finetune_summary_csv).copy()
    fine_df["model_type"] = fine_df["rank_idx"].map({
        1: "finetuned_rank1",
        2: "finetuned_rank2"
    }).fillna("finetuned_other")
else:
    fine_df = pd.DataFrame()

compare_df = pd.concat([baseline_row, fine_df], ignore_index=True, sort=False)
compare_df.to_csv(baseline_compare_csv, index=False)

print("✅ compare summary saved:", baseline_compare_csv)
display(compare_df[[
    "model_type", "rank_idx", "run_name",
    "anisotropy", "diameter", "cellprob_threshold", "stitch_threshold",
    "total_cells", "mean_volume", "median_volume", "p90_volume", "max_volume",
    "large_obj_ratio_ge_2x_median", "mask_path"
]])


# In[ ]:





# In[ ]:





# In[ ]:


#3.18测试


# In[3]:


get_ipython().run_cell_magic('writefile', '/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py', '')


# In[6]:


get_ipython().run_cell_magic('writefile', '/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py', '# ==========================================\n# Cell Sweep-SBatch-Submit (3.11读模型 / 3.18存输出)：\n# 生成 3D crop sweep 的参数表 + 单任务脚本 + sbatch array 脚本，并一键提交\n# 扫描：\n#   cellprob_threshold\n#   min_size\n# 固定：\n#   anisotropy=2.0\n#   diameter=8\n#   do_3D=True\n# ==========================================\nimport os\nimport json\nimport textwrap\nimport itertools\nimport subprocess\nfrom pathlib import Path\nfrom datetime import datetime\n\nimport pandas as pd\n\nprint("=" * 100)\nprint("🚀 Sweep-SBatch-Submit | 3.11读模型 / 3.18存输出")\nprint("=" * 100)\n\n# --------------------------------------------------\n# 0) 固定总根目录\n# --------------------------------------------------\nROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()\nassert ROOT.exists(), f"ROOT 不存在: {ROOT}"\n\n# --------------------------------------------------\n# 1) 输入来源：仍然读取 3.11 实验\n# --------------------------------------------------\nSRC_EXP_DIR = Path(\n    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"\n).resolve()\n\nCFG_DIR = (SRC_EXP_DIR / "config").resolve()\n\nTARGET_TAG = "P21_lr9e5_wd8e3"\n\nRAW_3D_STACK_PATH = Path(\n    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"\n).resolve()\n\nBASELINE_MASK_PATH = Path(\n    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"\n).resolve()\n\nassert SRC_EXP_DIR.exists(), f"SRC_EXP_DIR 不存在: {SRC_EXP_DIR}"\nassert CFG_DIR.exists(), f"CFG_DIR 不存在: {CFG_DIR}"\nassert RAW_3D_STACK_PATH.exists(), f"RAW_3D_STACK_PATH 不存在: {RAW_3D_STACK_PATH}"\nassert BASELINE_MASK_PATH.exists(), f"BASELINE_MASK_PATH 不存在: {BASELINE_MASK_PATH}"\n\n# --------------------------------------------------\n# 2) 输出根目录：新建 3.18 实验目录\n# --------------------------------------------------\nRUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")\nOUT_EXP_DIR = Path(\n    f"/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_{RUN_STAMP}"\n).resolve()\nOUT_EXP_DIR.mkdir(parents=True, exist_ok=True)\n\nprint("📥 SRC_EXP_DIR :", SRC_EXP_DIR)\nprint("📤 OUT_EXP_DIR :", OUT_EXP_DIR)\n\n# --------------------------------------------------\n# 3) 这轮 sweep 参数\n# --------------------------------------------------\nSWEEP_NAME = "sweep_crop_cp_minsize_a2_d8_20260318_v1"\n\nANISOTROPY = 2.0\nDIAMETER = 8.0\nDO_3D = True\nZ_AXIS = 0\nBATCH_SIZE_3D = 8\nSTITCH_THRESHOLD = 0.0\n\nCELLPROB_THRESHOLD_LIST = [1.0, 1.5, 2.0, 2.5]\nMIN_SIZE_LIST = [50, 100, 150, 200]\n\n# crop 位置（沿用你当前 notebook）\nZ_VIS = 100\nY0, Y1 = 478, 734\nX0, X1 = 353, 609\nZ_HALF_SPAN = 16\n\nFORCE_RERUN = False\n\n# --------------------------------------------------\n# 4) Slurm 配置\n# --------------------------------------------------\nSLURM_PARTITION = "GPUA800"   # 需要时改成 GPUH100\nSLURM_GRES = "gpu:1"\nSLURM_CPUS_PER_TASK = 8\nSLURM_MEM = "64G"\nSLURM_TIME = "12:00:00"\n\n# 你当前常用 micromamba 环境\nENV_ACTIVATE = """\nsource ~/.bashrc\nmicromamba activate cpsm\n"""\n\n# --------------------------------------------------\n# 5) 输出目录（全部挂到 3.18）\n# --------------------------------------------------\nSWEEP_ROOT = (OUT_EXP_DIR / "sweep_3d_compare" / SWEEP_NAME).resolve()\nCODE_ROOT  = (SWEEP_ROOT / "code").resolve()\nLOG_ROOT   = (SWEEP_ROOT / "slurm_logs").resolve()\nMASK_ROOT  = (SWEEP_ROOT / TARGET_TAG / "masks").resolve()\nSTAT_ROOT  = (SWEEP_ROOT / TARGET_TAG / "stats").resolve()\nFIG_ROOT   = (SWEEP_ROOT / TARGET_TAG / "figures").resolve()\nSNAP_ROOT  = (SWEEP_ROOT / "snapshot").resolve()\n\nfor p in [SWEEP_ROOT, CODE_ROOT, LOG_ROOT, MASK_ROOT, STAT_ROOT, FIG_ROOT, SNAP_ROOT]:\n    p.mkdir(parents=True, exist_ok=True)\n\nprint("📂 SWEEP_ROOT:", SWEEP_ROOT)\nprint("📂 CODE_ROOT :", CODE_ROOT)\nprint("📂 LOG_ROOT  :", LOG_ROOT)\n\n# --------------------------------------------------\n# 6) 从 3.11 config 中自动找到当前 TARGET_TAG 对应模型\n# --------------------------------------------------\ndef read_json(path: Path):\n    try:\n        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))\n    except Exception:\n        return None\n\nconfig_files = sorted(CFG_DIR.glob("config_*.json"))\nselected_model_path = None\nselected_snapshot_path = None\nselected_snapshot = None\n\nfor sf in config_files:\n    snap = read_json(sf)\n    if not isinstance(snap, dict):\n        continue\n\n    tag = snap.get("tag") or snap.get("model_tag") or snap.get("run_tag")\n    if tag != TARGET_TAG:\n        continue\n\n    best_model_path = snap.get("best_model_path")\n    final_model_path = snap.get("final_model_path") or snap.get("model_dir") or snap.get("eval_model_path")\n\n    cand = None\n    if best_model_path and Path(best_model_path).exists():\n        cand = Path(best_model_path).resolve()\n    elif final_model_path and Path(final_model_path).exists():\n        cand = Path(final_model_path).resolve()\n\n    if cand is not None:\n        selected_model_path = cand\n        selected_snapshot_path = sf.resolve()\n        selected_snapshot = snap\n\nif selected_model_path is None:\n    raise RuntimeError(f"❌ 没找到 TARGET_TAG={TARGET_TAG} 对应模型，请检查 {CFG_DIR}")\n\nprint("✅ selected_model_path   :", selected_model_path)\nprint("✅ selected_snapshot_path:", selected_snapshot_path)\n\n# --------------------------------------------------\n# 7) 在 3.18 目录里保存一份快照（只做记录，不改调用来源）\n# --------------------------------------------------\n(SNAP_ROOT / "selected_model_path.txt").write_text(\n    str(selected_model_path) + "\\n", encoding="utf-8"\n)\n(SNAP_ROOT / "selected_config_path.txt").write_text(\n    str(selected_snapshot_path) + "\\n", encoding="utf-8"\n)\nif selected_snapshot is not None:\n    (SNAP_ROOT / "selected_config_snapshot.json").write_text(\n        json.dumps(selected_snapshot, indent=2, ensure_ascii=False),\n        encoding="utf-8"\n    )\n\nmeta = {\n    "src_exp_dir": str(SRC_EXP_DIR),\n    "out_exp_dir": str(OUT_EXP_DIR),\n    "target_tag": TARGET_TAG,\n    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),\n    "baseline_mask_path": str(BASELINE_MASK_PATH),\n    "selected_model_path": str(selected_model_path),\n    "selected_config_path": str(selected_snapshot_path),\n}\n(SNAP_ROOT / "run_meta.json").write_text(\n    json.dumps(meta, indent=2, ensure_ascii=False),\n    encoding="utf-8"\n)\n\nprint("✅ snapshot saved to:", SNAP_ROOT)\n\n# --------------------------------------------------\n# 8) 生成参数表\n# --------------------------------------------------\nrows = []\nfor cp, ms in itertools.product(CELLPROB_THRESHOLD_LIST, MIN_SIZE_LIST):\n    rows.append({\n        "target_tag": TARGET_TAG,\n        "model_path": str(selected_model_path),\n        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),\n        "baseline_mask_path": str(BASELINE_MASK_PATH),\n        "src_exp_dir": str(SRC_EXP_DIR),\n        "out_exp_dir": str(OUT_EXP_DIR),\n        "sweep_root": str(SWEEP_ROOT),\n        "mask_root": str(MASK_ROOT),\n        "stat_root": str(STAT_ROOT),\n        "fig_root": str(FIG_ROOT),\n        "anisotropy": ANISOTROPY,\n        "diameter": DIAMETER,\n        "cellprob_threshold": cp,\n        "min_size": ms,\n        "stitch_threshold": STITCH_THRESHOLD,\n        "do_3d": DO_3D,\n        "z_axis": Z_AXIS,\n        "batch_size_3d": BATCH_SIZE_3D,\n        "z_vis": Z_VIS,\n        "z_half_span": Z_HALF_SPAN,\n        "y0": Y0, "y1": Y1,\n        "x0": X0, "x1": X1,\n        "force_rerun": FORCE_RERUN,\n    })\n\ngrid_df = pd.DataFrame(rows)\ngrid_csv = CODE_ROOT / "grid_cp_minsize.csv"\ngrid_df.to_csv(grid_csv, index=False)\n\nprint(f"✅ grid saved: {grid_csv}")\nprint(grid_df[["cellprob_threshold", "min_size"]].to_string(index=False))\n\n# --------------------------------------------------\n# 9) 生成单任务 python 脚本\n# --------------------------------------------------\nrun_one_py = CODE_ROOT / "run_crop_sweep_one.py"\n\nrun_one_code = r\'\'\'\nimport os\nimport gc\nimport json\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport tifffile as tiff\nimport matplotlib.pyplot as plt\n\nfrom cellpose import models, io\n\ntry:\n    import torch\nexcept ImportError:\n    torch = None\n\nprint("=" * 100)\nprint("🧪 run_crop_sweep_one.py")\nprint("=" * 100)\n\nio.logger_setup()\n\nGRID_CSV = Path(os.environ["GRID_CSV"]).resolve()\nTASK_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])\n\nprint("GRID_CSV:", GRID_CSV)\nprint("TASK_ID :", TASK_ID)\n\ngrid = pd.read_csv(GRID_CSV)\nassert 0 <= TASK_ID < len(grid), f"TASK_ID 越界: {TASK_ID}, len(grid)={len(grid)}"\nrow = grid.iloc[TASK_ID].to_dict()\n\ndef safe_cuda_cleanup():\n    gc.collect()\n    if torch is not None and torch.cuda.is_available():\n        try:\n            torch.cuda.empty_cache()\n        except Exception:\n            pass\n\ndef normalize_img(img2d: np.ndarray):\n    img = img2d.astype(np.float32)\n    lo, hi = np.percentile(img, [1, 99])\n    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)\n    return img\n\ndef mask_to_boundary_instance(mask2d: np.ndarray):\n    m = mask2d.astype(np.int64)\n    bd = np.zeros_like(m, dtype=bool)\n\n    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))\n    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))\n\n    bd[:-1, :] |= diff_ud\n    bd[1:,  :] |= diff_ud\n    bd[:, :-1] |= diff_lr\n    bd[:, 1: ] |= diff_lr\n\n    bd[0, :] = False\n    bd[-1, :] = False\n    bd[:, 0] = False\n    bd[:, -1] = False\n    return bd\n\ndef overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):\n    raw_norm = normalize_img(raw2d)\n    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)\n    bd = mask_to_boundary_instance(mask2d)\n    if color == "red":\n        overlay[bd] = [1.0, 0.0, 0.0]\n    elif color == "green":\n        overlay[bd] = [0.0, 1.0, 0.0]\n    elif color == "blue":\n        overlay[bd] = [0.0, 0.6, 1.0]\n    else:\n        overlay[bd] = [1.0, 1.0, 0.0]\n    return overlay\n\ndef label_stats_3d(mask3d: np.ndarray):\n    labels, counts = np.unique(mask3d, return_counts=True)\n    keep = labels > 0\n    labels = labels[keep]\n    counts = counts[keep]\n\n    if len(labels) == 0:\n        return {\n            "total_cells": 0,\n            "median_volume": 0.0,\n            "p90_volume": 0.0,\n            "max_volume": 0.0,\n            "min_volume_kept": 0.0,\n        }\n\n    return {\n        "total_cells": int(len(labels)),\n        "median_volume": float(np.median(counts)),\n        "p90_volume": float(np.percentile(counts, 90)),\n        "max_volume": float(np.max(counts)),\n        "min_volume_kept": float(np.min(counts)),\n    }\n\ntarget_tag = str(row["target_tag"])\nmodel_path = Path(row["model_path"]).resolve()\nraw_path = Path(row["raw_3d_stack_path"]).resolve()\nbaseline_mask_path = Path(row["baseline_mask_path"]).resolve()\n\nmask_root = Path(row["mask_root"]).resolve()\nstat_root = Path(row["stat_root"]).resolve()\nfig_root = Path(row["fig_root"]).resolve()\n\nanisotropy = float(row["anisotropy"])\ndiameter = float(row["diameter"])\ncellprob = float(row["cellprob_threshold"])\nmin_size = int(row["min_size"])\nstitch = float(row["stitch_threshold"])\ndo_3d = bool(row["do_3d"])\nz_axis = int(row["z_axis"])\nbatch_size_3d = int(row["batch_size_3d"])\n\nz_vis = int(row["z_vis"])\nz_half_span = int(row["z_half_span"])\ny0, y1 = int(row["y0"]), int(row["y1"])\nx0, x1 = int(row["x0"]), int(row["x1"])\n\nforce_rerun = str(row["force_rerun"]).lower() == "true"\n\ncp_str = str(cellprob).replace(".", "p").replace("-", "m")\nrun_name = f"{target_tag}__a{anisotropy}__d{diameter}__cp{cp_str}__ms{min_size}"\n\nmask_out = mask_root / f"{run_name}__crop3d_masks.tif"\nstat_out = stat_root / f"{run_name}__stats.json"\nfig_out  = fig_root  / f"{run_name}__overlay_z{z_vis}.png"\n\nprint("\\n🎯 run_name:", run_name)\nprint("mask_out:", mask_out)\nprint("stat_out:", stat_out)\nprint("fig_out :", fig_out)\n\nif mask_out.exists() and stat_out.exists() and fig_out.exists() and not force_rerun:\n    print("⏭️ 已存在且 force_rerun=False，跳过")\n    raise SystemExit(0)\n\nprint("\\n[1/5] 读取 raw / baseline")\nstack = tiff.imread(str(raw_path))\nbaseline = tiff.imread(str(baseline_mask_path))\n\nassert stack.ndim == 3, f"raw 不是 3D: {stack.shape}"\nassert baseline.ndim == 3, f"baseline 不是 3D: {baseline.shape}"\n\nz0 = max(0, z_vis - z_half_span)\nz1 = min(stack.shape[0], z_vis + z_half_span + 1)\n\ncrop_raw = stack[z0:z1, y0:y1, x0:x1]\ncrop_base = baseline[z0:z1, y0:y1, x0:x1]\n\nprint("crop_raw shape :", crop_raw.shape)\nprint("crop_base shape:", crop_base.shape)\n\nprint("\\n[2/5] 加载模型")\nmodel = models.CellposeModel(\n    gpu=True,\n    pretrained_model=str(model_path),\n)\n\nprint("\\n[3/5] 跑 3D eval")\nt0 = time.time()\nmasks, flows, styles = model.eval(\n    crop_raw,\n    do_3D=do_3d,\n    z_axis=z_axis,\n    anisotropy=anisotropy,\n    diameter=diameter,\n    cellprob_threshold=cellprob,\n    min_size=min_size,\n    stitch_threshold=stitch,\n    batch_size=batch_size_3d,\n)\nelapsed_s = time.time() - t0\n\nsafe_cuda_cleanup()\n\nprint("✅ eval done, elapsed_s =", elapsed_s)\nprint("masks shape:", masks.shape, "dtype:", masks.dtype)\n\nprint("\\n[4/5] 保存 mask + stats")\ntiff.imwrite(str(mask_out), masks.astype(np.uint32))\n\nstats = label_stats_3d(masks)\nstats.update({\n    "run_name": run_name,\n    "target_tag": target_tag,\n    "anisotropy": anisotropy,\n    "diameter": diameter,\n    "cellprob_threshold": cellprob,\n    "min_size": min_size,\n    "stitch_threshold": stitch,\n    "elapsed_s": elapsed_s,\n    "mask_path": str(mask_out),\n    "z_range": [int(z0), int(z1)],\n    "crop_box": [int(y0), int(y1), int(x0), int(x1)],\n})\n\nstat_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")\nprint("✅ stats saved:", stat_out)\n\nprint("\\n[5/5] 保存 overlay")\nlocal_z = z_vis - z0\noverlay_pred = overlay_boundary(crop_raw[local_z], masks[local_z], color="green")\noverlay_base = overlay_boundary(crop_raw[local_z], crop_base[local_z], color="red")\n\nplt.figure(figsize=(15, 5))\nplt.subplot(1, 3, 1)\nplt.imshow(normalize_img(crop_raw[local_z]), cmap="gray")\nplt.title(f"raw z={z_vis}")\nplt.axis("off")\n\nplt.subplot(1, 3, 2)\nplt.imshow(overlay_base)\nplt.title("baseline (red)")\nplt.axis("off")\n\nplt.subplot(1, 3, 3)\nplt.imshow(overlay_pred)\nplt.title(f"pred (green)\\ncp={cellprob}, min_size={min_size}")\nplt.axis("off")\n\nplt.tight_layout()\nplt.savefig(fig_out, dpi=180, bbox_inches="tight")\nplt.close()\n\nprint("✅ figure saved:", fig_out)\nprint("\\n🎉 TASK DONE")\n\'\'\'\nrun_one_py.write_text(run_one_code, encoding="utf-8")\nprint(f"✅ run_one script saved: {run_one_py}")\n\n# --------------------------------------------------\n# 10) 生成 sbatch array 脚本\n# --------------------------------------------------\nsbatch_sh = CODE_ROOT / "submit_crop_sweep_array.sbatch"\narray_max = len(grid_df) - 1\n\nsbatch_code = f"""#!/bin/bash\n#SBATCH -J cpmsweep0318\n#SBATCH -p {SLURM_PARTITION}\n#SBATCH --gres={SLURM_GRES}\n#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}\n#SBATCH --mem={SLURM_MEM}\n#SBATCH -t {SLURM_TIME}\n#SBATCH -o {LOG_ROOT}/slurm_%A_%a.out\n#SBATCH -e {LOG_ROOT}/slurm_%A_%a.err\n#SBATCH --array=0-{array_max}\n\nset -e\nset -u\n\necho "============================================================"\necho "🚀 SLURM job started"\necho "JOB_ID=$SLURM_JOB_ID"\necho "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"\necho "HOSTNAME=$(hostname)"\necho "START_TIME=$(date)"\necho "============================================================"\n\n{ENV_ACTIVATE}\n\nexport GRID_CSV="{grid_csv}"\npython "{run_one_py}"\n\necho "============================================================"\necho "✅ SLURM job finished"\necho "END_TIME=$(date)"\necho "============================================================"\n"""\nsbatch_sh.write_text(sbatch_code, encoding="utf-8")\nos.chmod(sbatch_sh, 0o755)\nprint(f"✅ sbatch script saved: {sbatch_sh}")\n\n# --------------------------------------------------\n# 11) 输出路径总览\n# --------------------------------------------------\nprint("\\n" + "=" * 100)\nprint("📌 输出路径总览")\nprint("=" * 100)\nprint("OUT_EXP_DIR :", OUT_EXP_DIR)\nprint("SWEEP_ROOT  :", SWEEP_ROOT)\nprint("CODE_ROOT   :", CODE_ROOT)\nprint("LOG_ROOT    :", LOG_ROOT)\nprint("MASK_ROOT   :", MASK_ROOT)\nprint("STAT_ROOT   :", STAT_ROOT)\nprint("FIG_ROOT    :", FIG_ROOT)\nprint("SNAP_ROOT   :", SNAP_ROOT)\nprint("grid_csv    :", grid_csv)\nprint("run_one_py  :", run_one_py)\nprint("sbatch_sh   :", sbatch_sh)\n\n# --------------------------------------------------\n# 12) 提交\n# --------------------------------------------------\nprint("\\n🚀 提交 sbatch ...")\nres = subprocess.run(["sbatch", str(sbatch_sh)], capture_output=True, text=True)\n\nprint("returncode:", res.returncode)\nprint("stdout:")\nprint(res.stdout.strip() if res.stdout else "(empty)")\nprint("stderr:")\nprint(res.stderr.strip() if res.stderr else "(empty)")\n\nif res.returncode != 0:\n    raise RuntimeError("❌ sbatch 提交失败，请检查 partition / 环境激活命令 / sbatch 是否可用")\n\nprint("\\n🎉 已提交。后面断网也不影响，Slurm 会继续跑。")\n')


# In[7]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

text = text.replace(
    '#SBATCH --array=0-{array_max}',
    '#SBATCH --array=0-{array_max}%2'
)

py_path.write_text(text, encoding="utf-8")
print("done")


# In[5]:


get_ipython().system('mkdir -p /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318')


# In[8]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

for line in text.splitlines():
    if "--array=" in line:
        print(line)


# In[9]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8").splitlines()

for i, line in enumerate(text, 1):
    if "--array=" in line:
        start = max(0, i-5)
        end = min(len(text), i+5)
        for j in range(start, end):
            print(f"{j+1}: {text[j]}")


# In[10]:


get_ipython().system('python /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py')


# In[11]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

old = '''ENV_ACTIVATE = """
source ~/.bashrc
micromamba activate cpsm
"""'''

new = '''ENV_ACTIVATE = """
eval "$(micromamba shell hook --shell bash)"
micromamba activate cpsm
"""'''

text = text.replace(old, new)
py_path.write_text(text, encoding="utf-8")

print("done")


# In[12]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

capture = False
for line in text.splitlines():
    if "ENV_ACTIVATE" in line:
        capture = True
    if capture:
        print(line)
        if line.strip() == '"""' and "ENV_ACTIVATE" not in line:
            break


# In[13]:


get_ipython().system('python /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py')


# In[14]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

text = text.replace('set -u\n', '')
py_path.write_text(text, encoding="utf-8")

print("done")


# In[15]:


from pathlib import Path

py_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py")
text = py_path.read_text(encoding="utf-8")

for line in text.splitlines():
    if "set -e" in line or "set -u" in line:
        print(line)


# In[16]:


get_ipython().system('python /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318/submit_sweep_0318.py')


# In[ ]:





# In[ ]:





# In[20]:


#视奸进程，只看最后50行
get_ipython().system('tail -n 50 /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260318_ori_train_9runs_400ep_es50_20260318_160213/sweep_3d_compare/sweep_crop_cp_minsize_a2_d8_20260318_v1/slurm_logs/slurm_1318510_0.out')


# In[21]:


# ==========================================
# Cell Analyze-1：汇总并分析 3D crop sweep（适配 sbatch 并行输出）
# 扫描参数：
#   cellprob_threshold
#   min_size
# 固定：
#   anisotropy=2.0
#   diameter=8
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 100)
print("📊 Cell Analyze-1 | Analyze finished crop sweep results")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()

exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_* 目录"

OUT_EXP_DIR = exp_dirs[0].resolve()
SWEEP_NAME = "sweep_crop_cp_minsize_a2_d8_20260318_v1"
SWEEP_ROOT = (OUT_EXP_DIR / "sweep_3d_compare" / SWEEP_NAME).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"
STAT_ROOT = (SWEEP_ROOT / TARGET_TAG / "stats").resolve()
FIG_ROOT  = (SWEEP_ROOT / TARGET_TAG / "figures").resolve()
MASK_ROOT = (SWEEP_ROOT / TARGET_TAG / "masks").resolve()

TOPK_SHOW = 6

print("📂 OUT_EXP_DIR:", OUT_EXP_DIR)
print("📂 SWEEP_ROOT :", SWEEP_ROOT)
print("📂 STAT_ROOT  :", STAT_ROOT)
print("📂 FIG_ROOT   :", FIG_ROOT)
print("📂 MASK_ROOT  :", MASK_ROOT)

assert SWEEP_ROOT.exists(), f"❌ SWEEP_ROOT 不存在: {SWEEP_ROOT}"
assert STAT_ROOT.exists(), f"❌ STAT_ROOT 不存在: {STAT_ROOT}"

# --------------------------------------------------
# 1) 读取所有 stats json
# --------------------------------------------------
stat_files = sorted(STAT_ROOT.glob("*__stats.json"))
assert len(stat_files) > 0, f"❌ 没找到 stats json: {STAT_ROOT}"

print(f"\n✅ 找到 stats 文件数: {len(stat_files)}")

rows = []
bad_files = []

for sf in stat_files:
    try:
        data = json.loads(sf.read_text(encoding="utf-8", errors="ignore"))
        rows.append(data)
    except Exception as e:
        bad_files.append((sf, str(e)))

if bad_files:
    print("\n⚠️ 以下 stats 文件读取失败：")
    for p, err in bad_files:
        print(" -", p)
        print("   ", err)

summary_df = pd.DataFrame(rows)
assert len(summary_df) > 0, "❌ 没有成功读入任何 stats"

print("\n✅ 成功读入记录数:", len(summary_df))
print("columns:", list(summary_df.columns))

# --------------------------------------------------
# 2) 补充一些辅助列
# --------------------------------------------------
def safe_exists(x):
    try:
        return Path(str(x)).exists()
    except Exception:
        return False

if "fig_path" not in summary_df.columns:
    # 兼容你之前脚本 fig_path 可能没写入 json 的情况
    if "run_name" in summary_df.columns:
        summary_df["fig_path"] = summary_df["run_name"].apply(
            lambda x: str(FIG_ROOT / f"{x}__overlay_z100.png")
        )

summary_df["fig_exists"] = summary_df["fig_path"].apply(safe_exists) if "fig_path" in summary_df.columns else False
summary_df["mask_exists"] = summary_df["mask_path"].apply(safe_exists) if "mask_path" in summary_df.columns else False

# --------------------------------------------------
# 3) 排序逻辑
# --------------------------------------------------
# 你这轮重点是：
# - 粘连少：max_volume / p90_volume / median_volume 不要太离谱
# - 细胞数不要太少
# - min_size 太大可能会删掉很多真细胞
#
# 暂时先用一个实用排序：
# 1) max_volume 小
# 2) p90_volume 小
# 3) total_cells 大
# 4) median_volume 小
#
# 这不是绝对真理，只是第一轮机器排序；最后还要看 overlay 图。
sort_cols = []
ascending = []

if "max_volume" in summary_df.columns:
    sort_cols.append("max_volume")
    ascending.append(True)

if "p90_volume" in summary_df.columns:
    sort_cols.append("p90_volume")
    ascending.append(True)

if "total_cells" in summary_df.columns:
    sort_cols.append("total_cells")
    ascending.append(False)

if "median_volume" in summary_df.columns:
    sort_cols.append("median_volume")
    ascending.append(True)

summary_df = summary_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

# 加 rank
summary_df.insert(0, "rank_auto", np.arange(1, len(summary_df) + 1))

# --------------------------------------------------
# 4) 保存 summary csv
# --------------------------------------------------
summary_csv = SWEEP_ROOT / f"summary_3d_sweep_{TARGET_TAG}_cp_minsize.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n✅ summary saved:", summary_csv)

display_cols = [
    c for c in [
        "rank_auto",
        "run_name",
        "cellprob_threshold",
        "min_size",
        "total_cells",
        "median_volume",
        "p90_volume",
        "max_volume",
        "min_volume_kept",
        "elapsed_s",
        "fig_exists",
        "mask_exists",
    ] if c in summary_df.columns
]

display(summary_df[display_cols].head(16))

# --------------------------------------------------
# 5) 生成 top-k 总览图
# --------------------------------------------------
k = min(TOPK_SHOW, len(summary_df))
top_df = summary_df.head(k).copy()

print(f"\n✅ 生成 top-{k} 总览图")

ncols = 3
nrows = int(np.ceil(k / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
axes = np.array(axes).reshape(-1)

for ax in axes:
    ax.axis("off")

for i, (_, row) in enumerate(top_df.iterrows()):
    fig_path = Path(row["fig_path"])
    if fig_path.exists():
        img = plt.imread(str(fig_path))
        axes[i].imshow(img)
        axes[i].set_title(
            f"rank#{int(row['rank_auto'])}\n"
            f"cp={row.get('cellprob_threshold', 'NA')}, ms={row.get('min_size', 'NA')}\n"
            f"cells={int(row.get('total_cells', 0))} | "
            f"medV={row.get('median_volume', np.nan):.1f} | "
            f"p90={row.get('p90_volume', np.nan):.1f} | "
            f"max={row.get('max_volume', np.nan):.1f}",
            fontsize=10
        )
        axes[i].axis("off")
    else:
        axes[i].text(0.5, 0.5, f"Missing figure\n{fig_path.name}", ha="center", va="center")
        axes[i].axis("off")

topk_fig_path = SWEEP_ROOT / f"top{TOPK_SHOW}_overview_{TARGET_TAG}_cp_minsize.png"
plt.tight_layout()
plt.savefig(topk_fig_path, dpi=180, bbox_inches="tight")
plt.show()

print("✅ top-k 总览图已保存:", topk_fig_path)

# --------------------------------------------------
# 6) 输出推荐 top2
# --------------------------------------------------
print("\n" + "=" * 100)
print("🏆 Auto top-2 candidates")
print("=" * 100)

top2 = summary_df.head(2).copy()

for i, (_, row) in enumerate(top2.iterrows(), 1):
    print(f"\n[{i}] {row.get('run_name', 'NA')}")
    print(f"    cp        : {row.get('cellprob_threshold', 'NA')}")
    print(f"    min_size  : {row.get('min_size', 'NA')}")
    print(f"    total     : {row.get('total_cells', 'NA')}")
    print(f"    medianV   : {row.get('median_volume', 'NA')}")
    print(f"    p90V      : {row.get('p90_volume', 'NA')}")
    print(f"    maxV      : {row.get('max_volume', 'NA')}")
    print(f"    fig_path  : {row.get('fig_path', 'NA')}")
    print(f"    mask_path : {row.get('mask_path', 'NA')}")

# --------------------------------------------------
# 7) 保存分析记录
# --------------------------------------------------
record = {
    "out_exp_dir": str(OUT_EXP_DIR),
    "sweep_root": str(SWEEP_ROOT),
    "target_tag": TARGET_TAG,
    "stat_root": str(STAT_ROOT),
    "fig_root": str(FIG_ROOT),
    "mask_root": str(MASK_ROOT),
    "summary_csv": str(summary_csv),
    "topk_fig_path": str(topk_fig_path),
    "num_stat_files": int(len(stat_files)),
    "topk_show": int(TOPK_SHOW),
    "sort_rule": {
        "sort_cols": sort_cols,
        "ascending": ascending,
        "note": "先机器粗排，再人工看 overlay 图定最终 top2"
    }
}
record_path = SWEEP_ROOT / f"analysis_record_{TARGET_TAG}_cp_minsize.json"
record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n✅ 分析记录已保存:", record_path)

print("\n" + "=" * 100)
print("🏁 分析完成")
print("summary_csv :", summary_csv)
print("top-k 图    :", topk_fig_path)
print("=" * 100)


# In[2]:


# ==========================================
# Cell ROI-1：生成粗 brain ROI（直接显示预览图）
# 作用：
#   1) 从 full raw 3D 生成宽松 brain ROI
#   2) 保存 ROI mask / meta / preview
#   3) 直接在 cell 输出里显示检查图
# ==========================================
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_closing,
    binary_dilation,
    label,
)

print("=" * 100)
print("🧠 Cell ROI-1 | Build coarse brain ROI and preview inline")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()

exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_* 目录"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入路径
# --------------------------------------------------
RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top2_roi_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
ROI_ROOT = (FULL_ROOT / "roi").resolve()
ROI_ROOT.mkdir(parents=True, exist_ok=True)

ROI_MASK_PATH = ROI_ROOT / "brain_roi_mask.tif"
ROI_META_PATH = ROI_ROOT / "roi_meta.json"
ROI_PREVIEW_PATH = ROI_ROOT / "brain_roi_preview.png"

print("📂 OUT_EXP_DIR      :", OUT_EXP_DIR)
print("📂 FULL_ROOT        :", FULL_ROOT)
print("📂 ROI_ROOT         :", ROI_ROOT)
print("📂 RAW_3D_STACK_PATH:", RAW_3D_STACK_PATH)

# --------------------------------------------------
# 3) 参数（第一版尽量宽松，别把脑裁小了）
# --------------------------------------------------
# 高斯平滑，适当抑制噪声；z 向少平滑一点
GAUSSIAN_SIGMA = (0.8, 1.2, 1.2)

# 阈值用高分位数之间的线性插值：
# thr = p_low + alpha * (p_high - p_low)
P_LOW = 40
P_HIGH = 99.5
ALPHA = 0.18

# 形态学操作
CLOSING_ITERS = 2
DILATION_ITERS = 4

# 预览显示哪些 z
N_PREVIEW_SLICES = 5

# 是否覆盖已有 ROI
FORCE_RERUN = True

# --------------------------------------------------
# 4) 读取 raw
# --------------------------------------------------
raw = tiff.imread(str(RAW_3D_STACK_PATH))
assert raw.ndim == 3, f"❌ raw 不是 3D: {raw.shape}"

print("\n✅ raw shape:", raw.shape, raw.dtype)

if ROI_MASK_PATH.exists() and not FORCE_RERUN:
    print("⚠️ ROI 已存在且 FORCE_RERUN=False，直接读取已有 ROI")
    roi = tiff.imread(str(ROI_MASK_PATH)).astype(bool)
else:
    # --------------------------------------------------
    # 5) 轻微平滑
    # --------------------------------------------------
    print("\n[1/6] Gaussian smoothing ...")
    raw_s = gaussian_filter(raw.astype(np.float32), sigma=GAUSSIAN_SIGMA)

    # --------------------------------------------------
    # 6) 宽松阈值
    # --------------------------------------------------
    print("[2/6] Thresholding ...")
    p_low = np.percentile(raw_s, P_LOW)
    p_high = np.percentile(raw_s, P_HIGH)
    thr = p_low + ALPHA * (p_high - p_low)

    fg = raw_s > thr

    print(f"    p{P_LOW}   = {p_low:.3f}")
    print(f"    p{P_HIGH} = {p_high:.3f}")
    print(f"    thr       = {thr:.3f}")
    print(f"    fg ratio  = {fg.mean():.4f}")

    # --------------------------------------------------
    # 7) 闭运算 + 填洞
    # --------------------------------------------------
    print("[3/6] Closing + fill holes ...")
    roi0 = fg.copy()
    for _ in range(CLOSING_ITERS):
        roi0 = binary_closing(roi0)

    roi0 = binary_fill_holes(roi0)

    # --------------------------------------------------
    # 8) 保留最大连通域
    # --------------------------------------------------
    print("[4/6] Keep largest connected component ...")
    lab, ncomp = label(roi0)
    assert ncomp > 0, "❌ 阈值后没有任何连通域，说明阈值太高了"

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest_id = counts.argmax()
    roi1 = (lab == largest_id)

    print(f"    connected components = {ncomp}")
    print(f"    largest component voxels = {int(counts[largest_id])}")

    # --------------------------------------------------
    # 9) 再膨胀，给脑边界留 buffer
    # --------------------------------------------------
    print("[5/6] Dilate ROI for safety margin ...")
    roi = roi1.copy()
    for _ in range(DILATION_ITERS):
        roi = binary_dilation(roi)

    # 再填一次洞，更稳一点
    roi = binary_fill_holes(roi)

    print(f"    final roi ratio = {roi.mean():.4f}")

    # --------------------------------------------------
    # 10) 保存 ROI
    # --------------------------------------------------
    print("[6/6] Save ROI ...")
    tiff.imwrite(str(ROI_MASK_PATH), roi.astype(np.uint8))
    print("✅ saved:", ROI_MASK_PATH)

    meta = {
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "roi_mask_path": str(ROI_MASK_PATH),
        "raw_shape": list(raw.shape),
        "gaussian_sigma": list(GAUSSIAN_SIGMA),
        "threshold_percentile_low": P_LOW,
        "threshold_percentile_high": P_HIGH,
        "threshold_alpha": ALPHA,
        "threshold_value": float(thr),
        "closing_iters": CLOSING_ITERS,
        "dilation_iters": DILATION_ITERS,
        "roi_ratio": float(roi.mean()),
        "largest_component_voxels": int(counts[largest_id]),
    }
    ROI_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ saved:", ROI_META_PATH)

# --------------------------------------------------
# 11) 预览：选几个代表性 z 层，直接在输出里显示
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

z_nonzero = np.where(roi.sum(axis=(1, 2)) > 0)[0]
assert len(z_nonzero) > 0, "❌ ROI 在所有 z 都为空"

z_min, z_max = int(z_nonzero.min()), int(z_nonzero.max())
z_samples = np.linspace(z_min, z_max, N_PREVIEW_SLICES).round().astype(int)

print("\n✅ preview z slices:", z_samples.tolist())

fig, axes = plt.subplots(len(z_samples), 3, figsize=(15, 4 * len(z_samples)))
if len(z_samples) == 1:
    axes = np.array([axes])

for r, z in enumerate(z_samples):
    raw2d = raw[z]
    roi2d = roi[z].astype(bool)

    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    # ROI 区域给一点青绿色高亮
    overlay[roi2d] = 0.65 * overlay[roi2d] + 0.35 * np.array([0.0, 1.0, 1.0])

    axes[r, 0].imshow(raw_norm, cmap="gray")
    axes[r, 0].set_title(f"raw | z={z}")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(roi2d, cmap="gray")
    axes[r, 1].set_title(f"roi mask | z={z}")
    axes[r, 1].axis("off")

    axes[r, 2].imshow(overlay)
    axes[r, 2].set_title(f"raw + roi overlay | z={z}")
    axes[r, 2].axis("off")

plt.suptitle("Coarse Brain ROI Preview", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(ROI_PREVIEW_PATH, dpi=180, bbox_inches="tight")
plt.show()

print("\n✅ preview saved:", ROI_PREVIEW_PATH)

# --------------------------------------------------
# 12) 再额外输出一些整体信息
# --------------------------------------------------
roi_vox = int(roi.sum())
raw_vox = int(np.prod(raw.shape))
print("\n" + "=" * 100)
print("📌 ROI Summary")
print("=" * 100)
print("raw shape        :", raw.shape)
print("roi voxels       :", roi_vox)
print("all voxels       :", raw_vox)
print("roi ratio        :", f"{roi_vox / raw_vox:.4f}")
print("z range covered  :", (z_min, z_max))
print("ROI_MASK_PATH    :", ROI_MASK_PATH)
print("ROI_META_PATH    :", ROI_META_PATH)
print("ROI_PREVIEW_PATH :", ROI_PREVIEW_PATH)
print("=" * 100)


# In[3]:


# ==========================================
# Cell ROI-1b：生成粗 brain ROI（低频 envelope 版，直接显示预览）
# 思路：
#   1) 对 full raw 3D 做强平滑，抹掉单个细胞纹理
#   2) 在低频图上做阈值，提取脑主体大轮廓
#   3) 保留最大连通域
#   4) 填洞 + 闭运算 + 膨胀
#   5) 保存 ROI，并直接在 cell 输出里显示
# ==========================================
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_closing,
    binary_dilation,
    label,
)

print("=" * 100)
print("🧠 Cell ROI-1b | Build coarse brain ROI from low-frequency envelope")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位最新 3.18 实验目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()

exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 exp_20260318_ori_train_9runs_400ep_es50_* 目录"

OUT_EXP_DIR = exp_dirs[0].resolve()

# --------------------------------------------------
# 1) 输入路径
# --------------------------------------------------
RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top2_roi_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
ROI_ROOT = (FULL_ROOT / "roi").resolve()
ROI_ROOT.mkdir(parents=True, exist_ok=True)

ROI_MASK_PATH = ROI_ROOT / "brain_roi_mask_lowfreq_v1.tif"
ROI_META_PATH = ROI_ROOT / "roi_meta_lowfreq_v1.json"
ROI_PREVIEW_PATH = ROI_ROOT / "brain_roi_preview_lowfreq_v1.png"

print("📂 OUT_EXP_DIR      :", OUT_EXP_DIR)
print("📂 FULL_ROOT        :", FULL_ROOT)
print("📂 ROI_ROOT         :", ROI_ROOT)
print("📂 RAW_3D_STACK_PATH:", RAW_3D_STACK_PATH)

# --------------------------------------------------
# 3) 参数
# --------------------------------------------------
# 关键：强平滑，把单个细胞和细碎纹理糊掉
# z向少平滑一点，xy向多平滑一点
LOWFREQ_SIGMA = (2.0, 12.0, 12.0)

# 在低频图上阈值
# 用较高分位数区间里的线性插值，尽量抓主体而不是背景
P_LOW = 60
P_HIGH = 99.8
ALPHA = 0.22

# 形态学操作
CLOSING_ITERS = 2
DILATION_ITERS = 6

# 预览层数
N_PREVIEW_SLICES = 5

# 是否覆盖
FORCE_RERUN = True

# --------------------------------------------------
# 4) 读取 raw
# --------------------------------------------------
raw = tiff.imread(str(RAW_3D_STACK_PATH))
assert raw.ndim == 3, f"❌ raw 不是 3D: {raw.shape}"

print("\n✅ raw shape:", raw.shape, raw.dtype)

# --------------------------------------------------
# 5) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

# --------------------------------------------------
# 6) 生成 ROI
# --------------------------------------------------
if ROI_MASK_PATH.exists() and not FORCE_RERUN:
    print("⚠️ ROI 已存在且 FORCE_RERUN=False，直接读取已有 ROI")
    roi = tiff.imread(str(ROI_MASK_PATH)).astype(bool)
    lowfreq = None
    thr = None
else:
    print("\n[1/7] Build low-frequency envelope ...")
    lowfreq = gaussian_filter(raw.astype(np.float32), sigma=LOWFREQ_SIGMA)

    print("[2/7] Threshold on low-frequency image ...")
    p_low = np.percentile(lowfreq, P_LOW)
    p_high = np.percentile(lowfreq, P_HIGH)
    thr = p_low + ALPHA * (p_high - p_low)

    fg = lowfreq > thr

    print(f"    p{P_LOW}   = {p_low:.3f}")
    print(f"    p{P_HIGH} = {p_high:.3f}")
    print(f"    thr       = {thr:.3f}")
    print(f"    fg ratio  = {fg.mean():.4f}")

    print("[3/7] Fill holes + closing ...")
    roi0 = binary_fill_holes(fg)
    for _ in range(CLOSING_ITERS):
        roi0 = binary_closing(roi0)

    print("[4/7] Keep largest connected component ...")
    lab, ncomp = label(roi0)
    assert ncomp > 0, "❌ 阈值后没有任何连通域，说明阈值太高"

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest_id = counts.argmax()
    roi1 = (lab == largest_id)

    print(f"    connected components = {ncomp}")
    print(f"    largest component voxels = {int(counts[largest_id])}")

    print("[5/7] Fill holes again ...")
    roi2 = binary_fill_holes(roi1)

    print("[6/7] Dilate for safety margin ...")
    roi = roi2.copy()
    for _ in range(DILATION_ITERS):
        roi = binary_dilation(roi)

    print(f"    final roi ratio = {roi.mean():.4f}")

    print("[7/7] Save ROI ...")
    tiff.imwrite(str(ROI_MASK_PATH), roi.astype(np.uint8))
    print("✅ saved:", ROI_MASK_PATH)

    meta = {
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "roi_mask_path": str(ROI_MASK_PATH),
        "raw_shape": list(raw.shape),
        "lowfreq_sigma": list(LOWFREQ_SIGMA),
        "threshold_percentile_low": P_LOW,
        "threshold_percentile_high": P_HIGH,
        "threshold_alpha": ALPHA,
        "threshold_value": float(thr),
        "closing_iters": CLOSING_ITERS,
        "dilation_iters": DILATION_ITERS,
        "roi_ratio": float(roi.mean()),
        "largest_component_voxels": int(counts[largest_id]),
    }
    ROI_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ saved:", ROI_META_PATH)

# --------------------------------------------------
# 7) 选预览 z 层
# --------------------------------------------------
z_nonzero = np.where(roi.sum(axis=(1, 2)) > 0)[0]
assert len(z_nonzero) > 0, "❌ ROI 在所有 z 都为空"

z_min, z_max = int(z_nonzero.min()), int(z_nonzero.max())
z_samples = np.linspace(z_min, z_max, N_PREVIEW_SLICES).round().astype(int)

print("\n✅ preview z slices:", z_samples.tolist())

# 如果是读已有 ROI，没有 lowfreq，这里补算一份仅用于显示
if lowfreq is None:
    lowfreq = gaussian_filter(raw.astype(np.float32), sigma=LOWFREQ_SIGMA)

# --------------------------------------------------
# 8) 直接显示预览
# 每层显示：
#   raw
#   lowfreq
#   roi mask
#   raw + roi overlay
# --------------------------------------------------
fig, axes = plt.subplots(len(z_samples), 4, figsize=(20, 4 * len(z_samples)))
if len(z_samples) == 1:
    axes = np.array([axes])

for r, z in enumerate(z_samples):
    raw2d = raw[z]
    low2d = lowfreq[z]
    roi2d = roi[z].astype(bool)

    raw_norm = normalize_img(raw2d)
    low_norm = normalize_img(low2d)

    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    overlay[roi2d] = 0.65 * overlay[roi2d] + 0.35 * np.array([0.0, 1.0, 1.0])

    axes[r, 0].imshow(raw_norm, cmap="gray")
    axes[r, 0].set_title(f"raw | z={z}")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(low_norm, cmap="gray")
    axes[r, 1].set_title(f"lowfreq | z={z}")
    axes[r, 1].axis("off")

    axes[r, 2].imshow(roi2d, cmap="gray")
    axes[r, 2].set_title(f"roi mask | z={z}")
    axes[r, 2].axis("off")

    axes[r, 3].imshow(overlay)
    axes[r, 3].set_title(f"raw + roi overlay | z={z}")
    axes[r, 3].axis("off")

plt.suptitle("Coarse Brain ROI Preview (Low-frequency Envelope)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(ROI_PREVIEW_PATH, dpi=180, bbox_inches="tight")
plt.show()

print("\n✅ preview saved:", ROI_PREVIEW_PATH)

# --------------------------------------------------
# 9) 输出摘要
# --------------------------------------------------
roi_vox = int(roi.sum())
raw_vox = int(np.prod(raw.shape))

print("\n" + "=" * 100)
print("📌 ROI Summary")
print("=" * 100)
print("raw shape        :", raw.shape)
print("roi voxels       :", roi_vox)
print("all voxels       :", raw_vox)
print("roi ratio        :", f"{roi_vox / raw_vox:.4f}")
print("z range covered  :", (z_min, z_max))
print("ROI_MASK_PATH    :", ROI_MASK_PATH)
print("ROI_META_PATH    :", ROI_META_PATH)
print("ROI_PREVIEW_PATH :", ROI_PREVIEW_PATH)
print("=" * 100)


# In[1]:


# ==========================================
# Cell Full3D-StepA-Top1：
# 直挂 Jupyter 跑 full-brain 3D，只保存 raw masks
# 读取 3.11 模型，输出到最新 3.18
# 参数：
#   - cellprob_threshold = 2.5
#   - min_size = 50
#   - batch_size = 4
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
print("🧪 Full3D-StepA-Top1 | full-brain 3D raw mask only")
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
# 2) 固定参数：Top1
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = 8.0
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
run_name = f"{TARGET_TAG}__a{ANISOTROPY}__d{DIAMETER}__cp{cp_str}__ms{MIN_SIZE}"

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

(SNAP_ROOT / "selected_model_path.txt").write_text(str(selected_model_path) + "\n", encoding="utf-8")
(SNAP_ROOT / "selected_config_path.txt").write_text(str(selected_snapshot_path) + "\n", encoding="utf-8")
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
    print("✅ stack shape:", stack.shape, stack.dtype)

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
    print("\n[3/4] full-brain 3D eval")
    print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   elapsed_s :", elapsed_s)
    print("   masks shape:", masks.shape)
    print("   masks dtype:", masks.dtype)

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
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),
        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "elapsed_s": elapsed_s,
        "stack_shape": list(stack.shape),
        "step": "A_save_raw_mask_only",
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Top1 StepA DONE")


# In[3]:


# ==========================================
# Cell Full3D-StepB-Top1-Analyze：
# 适配新版 StepA-Top1 输出：
# 读取已保存 raw mask，做实例级统计 + 
正确边界可视化 + Top-K 大对象局部排查
#
# 对应 StepA 输出：
#   FULL_ROOT / run_name / raw_masks / {run_name}__full_brain_3d_masks_raw.tif
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

print("=" * 100)
print("🧪 Full3D-StepB-Top1-Analyze | Analyze saved Top1 raw mask with instance boundaries and local crops")
print("=" * 100)

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
# 1) 固定输入
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"❌ BASELINE 不存在: {BASELINE_MASK_PATH}"

# --------------------------------------------------
# 2) 对应 Top1 的固定参数（必须和 StepA 一致）
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = 8.0
STITCH_THRESHOLD = 0.0

# --------------------------------------------------
# 3) 输出目录（必须和 StepA 一致）
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
run_name = f"{TARGET_TAG}__a{ANISOTROPY}__d{DIAMETER}__cp{cp_str}__ms{MIN_SIZE}"

RUN_ROOT = (FULL_ROOT / run_name).resolve()
RAW_MASK_ROOT = (RUN_ROOT / "raw_masks").resolve()
STAT_ROOT = (RUN_ROOT / "stats").resolve()
FIG_ROOT = (RUN_ROOT / "figures").resolve()
CROP_ROOT = (FIG_ROOT / "largest_obj_crops").resolve()

for p in [SUMMARY_ROOT, STAT_ROOT, FIG_ROOT, CROP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 输入 / 输出路径
# --------------------------------------------------
mask_path = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
stepA_meta_path = STAT_ROOT / f"{run_name}__stepA_meta.json"

out_stats_path = STAT_ROOT / f"{run_name}__analysis_stats.json"
out_fig_path = FIG_ROOT / f"{run_name}__compare_multiz.png"
out_largest_csv = STAT_ROOT / f"{run_name}__largest_objects.csv"
out_largest_summary_json = STAT_ROOT / f"{run_name}__largest_objects.json"
out_crop_csv = STAT_ROOT / f"{run_name}__largest_object_crops.csv"

full_summary_csv = SUMMARY_ROOT / "fullbrain_top1_analysis_summary.csv"

assert mask_path.exists(), f"❌ mask tif 不存在，请先跑 StepA: {mask_path}"

print("🎯 run_name         :", run_name)
print("📂 mask_path        :", mask_path)
print("📂 stepA_meta_path  :", stepA_meta_path)
print("📂 out_stats_path   :", out_stats_path)
print("📂 out_fig_path     :", out_fig_path)
print("📂 out_largest_csv  :", out_largest_csv)
print("📂 out_crop_csv     :", out_crop_csv)
print("📂 full_summary_csv :", full_summary_csv)

# --------------------------------------------------
# 5) 可调参数
# --------------------------------------------------
Z_VIS_LIST = [98, 100, 102]     # 主对比图看的 z 层
TOPK_LARGEST = 8                # 排查最大的前 K 个对象
CROP_HALF_SIZE = 96             # 局部裁剪半径，实际 crop 大小约 192 x 192
SAVE_MAX_N_CROPS = 6            # 最多输出多少个局部 crop 图

# --------------------------------------------------
# 6) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    """
    实例边界：
    只要相邻像素 label 不同，就认为是边界。
    这样能显示实例之间的内部边界，而不是只显示整体前景外轮廓。
    """
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)

    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))

    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr

    bd[0, :] = False
    bd[-1, :] = False
    bd[:, 0] = False
    bd[:, -1] = False
    return bd

def overlay_instance_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 1.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_sizes = counts[valid]

    if len(obj_sizes) == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "p95_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
            "n_ge_3x_median": 0,
            "n_ge_5x_median": 0,
            "top10_volumes": [],
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    p95_v = float(np.percentile(obj_sizes, 95))
    max_v = float(np.max(obj_sizes))
    med_safe = max(median_v, 1.0)

    large_ratio = float(np.mean(obj_sizes >= (2.0 * med_safe)))
    n_ge_3x = int(np.sum(obj_sizes >= 3.0 * med_safe))
    n_ge_5x = int(np.sum(obj_sizes >= 5.0 * med_safe))
    top10 = sorted(obj_sizes.tolist(), reverse=True)[:10]

    return {
        "total_cells": int(len(obj_sizes)),
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "p95_volume": p95_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
        "n_ge_3x_median": n_ge_3x,
        "n_ge_5x_median": n_ge_5x,
        "top10_volumes": top10,
    }

def get_topk_objects(mask3d: np.ndarray, topk=10):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    labels = labels[valid]
    counts = counts[valid]

    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    out = []
    for lab, vol in zip(labels[:topk], counts[:topk]):
        coords = np.argwhere(mask3d == lab)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)

        out.append({
            "label": int(lab),
            "volume": int(vol),
            "z_min": int(z0), "z_max": int(z1),
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "z_center": int(cz), "y_center": int(cy), "x_center": int(cx),
            "z_span": int(z1 - z0 + 1),
            "y_span": int(y1 - y0 + 1),
            "x_span": int(x1 - x0 + 1),
        })
    return out

def safe_crop_2d(arr2d, cy, cx, half_size=96):
    H, W = arr2d.shape
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size)
    return arr2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def save_largest_object_crops(
    raw_stack,
    baseline_masks,
    pred_masks,
    largest_objs,
    crop_root,
    max_n=6,
    half_size=96
):
    crop_rows = []

    for i, obj in enumerate(largest_objs[:max_n], 1):
        lab = obj["label"]
        cz, cy, cx = obj["z_center"], obj["y_center"], obj["x_center"]

        raw2d = raw_stack[cz]
        base2d = baseline_masks[cz]
        pred2d = pred_masks[cz]

        raw_crop, crop_box = safe_crop_2d(raw2d, cy, cx, half_size=half_size)
        base_crop, _ = safe_crop_2d(base2d, cy, cx, half_size=half_size)
        pred_crop, _ = safe_crop_2d(pred2d, cy, cx, half_size=half_size)

        raw_norm = normalize_img(raw_crop)
        base_overlay = overlay_instance_boundary(raw_crop, base_crop, color="red")
        pred_overlay = overlay_instance_boundary(raw_crop, pred_crop, color="green")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title(f"Raw crop\nz={cz}, y={cy}, x={cx}")
        axes[0].axis("off")

        axes[1].imshow(base_overlay)
        axes[1].set_title("Baseline boundary")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(
            f"Pred boundary\nlabel={lab}, vol={obj['volume']}\n"
            f"zspan={obj['z_span']} yspan={obj['y_span']} xspan={obj['x_span']}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        crop_path = crop_root / f"largest_obj_rank{i:02d}__label{lab}__z{cz}_y{cy}_x{cx}.png"
        plt.savefig(crop_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        crop_rows.append({
            "rank_in_largest": i,
            "label": lab,
            "volume": obj["volume"],
            "z_center": cz,
            "y_center": cy,
            "x_center": cx,
            "z_span": obj["z_span"],
            "y_span": obj["y_span"],
            "x_span": obj["x_span"],
            "crop_path": str(crop_path),
            "crop_box_y0y1x0x1": str(crop_box),
        })

    return crop_rows

# --------------------------------------------------
# 7) 读取数据
# --------------------------------------------------
print("[1/5] 读取 raw / baseline / saved raw mask")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))
masks = tiff.imread(str(mask_path))

assert stack.ndim == 3, f"❌ raw 不是 3D: {stack.shape}"
assert baseline_masks.shape == stack.shape, "❌ baseline 与 raw shape 不一致"
assert masks.shape == stack.shape, "❌ saved mask 与 raw shape 不一致"

print("✅ stack shape     :", stack.shape, stack.dtype)
print("✅ baseline shape  :", baseline_masks.shape, baseline_masks.dtype)
print("✅ saved mask shape:", masks.shape, masks.dtype)

# --------------------------------------------------
# 8) 统计 mask
# --------------------------------------------------
print("[2/5] 统计 mask")
stats = summarize_mask(masks)
print("✅ stats:")
for k, v in stats.items():
    print(f"   - {k}: {v}")

largest_objs = get_topk_objects(masks, topk=TOPK_LARGEST)
largest_df = pd.DataFrame(largest_objs)
largest_df.to_csv(out_largest_csv, index=False)
print("✅ largest object table saved:", out_largest_csv)

# --------------------------------------------------
# 9) 多 z 层主对比图（实例边界）
# --------------------------------------------------
print("[3/5] 生成多 z 层实例边界对比图")
valid_z_list = [z for z in Z_VIS_LIST if 0 <= z < stack.shape[0]]
assert len(valid_z_list) > 0, "❌ Z_VIS_LIST 全部越界"

fig, axes = plt.subplots(len(valid_z_list), 3, figsize=(14, 4.2 * len(valid_z_list)))
if len(valid_z_list) == 1:
    axes = np.array([axes])

for i, z in enumerate(valid_z_list):
    raw2d = stack[z]
    base2d = baseline_masks[z]
    pred2d = masks[z]

    raw_norm = normalize_img(raw2d)
    base_overlay = overlay_instance_boundary(raw2d, base2d, color="red")
    pred_overlay = overlay_instance_boundary(raw2d, pred2d, color="green")

    axes[i, 0].imshow(raw_norm, cmap="gray")
    axes[i, 0].set_title(f"Raw | z={z}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(base_overlay)
    axes[i, 1].set_title(f"Baseline | z={z}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_overlay)
    axes[i, 2].set_title(
        f"Top1 | z={z}\n"
        f"a={ANISOTROPY}, d={DIAMETER}, cp={CELLPROB_THRESHOLD}, ms={MIN_SIZE}\n"
        f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
        f"maxV={stats['max_volume']:.1f} | n>=3xmed={stats['n_ge_3x_median']}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("✅ figure saved:", out_fig_path)

# --------------------------------------------------
# 10) 最大对象局部 crop 排查
# --------------------------------------------------
print("[4/5] 生成最大对象局部 crop 排查图")
crop_rows = save_largest_object_crops(
    raw_stack=stack,
    baseline_masks=baseline_masks,
    pred_masks=masks,
    largest_objs=largest_objs,
    crop_root=CROP_ROOT,
    max_n=SAVE_MAX_N_CROPS,
    half_size=CROP_HALF_SIZE
)
crop_df = pd.DataFrame(crop_rows)
crop_df.to_csv(out_crop_csv, index=False)
print("✅ largest object crop table saved:", out_crop_csv)

largest_summary = {
    "run_name": run_name,
    "topk_largest": largest_objs,
    "crop_rows": crop_rows,
}
out_largest_summary_json.write_text(
    json.dumps(largest_summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ largest object summary json saved:", out_largest_summary_json)

# --------------------------------------------------
# 11) 写 stats + 更新 full summary
# --------------------------------------------------
print("[5/5] 写 stats 和汇总表")

stepA_meta = None
if stepA_meta_path.exists():
    try:
        stepA_meta = json.loads(stepA_meta_path.read_text(encoding="utf-8"))
    except Exception:
        stepA_meta = None

stats_record = {
    "run_name": run_name,
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "mask_path": str(mask_path),
    "stepA_meta_path": str(stepA_meta_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_json": str(out_largest_summary_json),
    "largest_object_crop_csv": str(out_crop_csv),
    "shape": list(masks.shape),
    "anisotropy": ANISOTROPY,
    "diameter_3d": DIAMETER,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "stepA_elapsed_s": None if stepA_meta is None else stepA_meta.get("elapsed_s"),
    **stats
}
out_stats_path.write_text(
    json.dumps(stats_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ stats json saved:", out_stats_path)

new_full_df = pd.DataFrame([{
    "run_name": run_name,
    "anisotropy": ANISOTROPY,
    "diameter": DIAMETER,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "stepA_elapsed_s": None if stepA_meta is None else stepA_meta.get("elapsed_s"),
    "mask_path": str(mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_crop_csv": str(out_crop_csv),
    **stats
}])

if full_summary_csv.exists():
    old_full_df = pd.read_csv(full_summary_csv)
else:
    old_full_df = pd.DataFrame()

if len(old_full_df) > 0:
    merged_full_df = pd.concat([old_full_df, new_full_df], ignore_index=True)
    merged_full_df = merged_full_df.drop_duplicates(subset=["run_name"], keep="last")
else:
    merged_full_df = new_full_df

merged_full_df.to_csv(full_summary_csv, index=False)

print("\n✅ full summary saved:", full_summary_csv)
print("\n✅ largest objects:")
display(largest_df)

print("\n✅ largest object crops:")
display(crop_df)

print("\n✅ merged full summary:")
display(merged_full_df)


# In[ ]:


# ==========================================
# Cell Full3D-StepA-Top2：
# 直挂 Jupyter 跑 full-brain 3D，只保存 raw masks
# 读取 3.11 模型，输出到最新 3.18
# 参数：
#   - cellprob_threshold = 2.5
#   - min_size = 100
#   - batch_size = 4
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
print("🧪 Full3D-StepA-Top2 | full-brain 3D raw mask only")
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
# 2) 固定参数：Top2
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 100

ANISOTROPY = 2.0
DIAMETER = 8.0
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SNAP_ROOT = (FULL_ROOT / "snapshot").resolve()

for p in [FULL_ROOT, SNAP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
run_name = f"{TARGET_TAG}__a{ANISOTROPY}__d{DIAMETER}__cp{cp_str}__ms{MIN_SIZE}"

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

(SNAP_ROOT / "selected_model_path.txt").write_text(str(selected_model_path) + "\n", encoding="utf-8")
(SNAP_ROOT / "selected_config_path.txt").write_text(str(selected_snapshot_path) + "\n", encoding="utf-8")
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
    print("✅ stack shape:", stack.shape, stack.dtype)

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
    print("\n[3/4] full-brain 3D eval")
    print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        do_3D=DO_3D,
        z_axis=Z_AXIS,
        anisotropy=ANISOTROPY,
        diameter=DIAMETER,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_threshold=STITCH_THRESHOLD,
        batch_size=BATCH_SIZE_3D,
        progress=True,
    )

    elapsed_s = time.time() - t0
    masks = np.asarray(masks)

    print("✅ eval done")
    print("   elapsed_s :", elapsed_s)
    print("   masks shape:", masks.shape)
    print("   masks dtype:", masks.dtype)

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
        "model_path": str(selected_model_path),
        "selected_snapshot_path": str(selected_snapshot_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_out": str(mask_out),
        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "cellprob_threshold": CELLPROB_THRESHOLD,
        "min_size": MIN_SIZE,
        "stitch_threshold": STITCH_THRESHOLD,
        "batch_size_3d": BATCH_SIZE_3D,
        "elapsed_s": elapsed_s,
        "stack_shape": list(stack.shape),
        "step": "A_save_raw_mask_only",
    }
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("✅ saved mask :", mask_out)
    print("✅ saved meta :", meta_out)

    del masks, flows, styles, model, stack
    safe_cuda_cleanup()

print("\n🎉 Top2 StepA DONE")


# In[ ]:


# ==========================================
# Cell Full3D-StepB-Analyze-And-Filter-v2
# 适配新版 Jupyter StepA 两个模型：
#   1) cp=2.5, min_size=50
#   2) cp=2.5, min_size=100
#
# 功能：
#   - 读取 StepA raw masks
#   - 做实例级统计
#   - 生成多 z 层边界对比图
#   - 生成 Top-K 大对象局部排查图
#   - 做外围过滤
#   - 保存 filtered masks
#
# 路径规则：
#   - 公共 summary 只保存一份到 FULL_ROOT/summary
#   - 每个 run 专属文件保存到各自 run_root 下，避免冲突
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

print("=" * 100)
print("🧪 Full3D-StepB-Analyze-And-Filter-v2")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 3.18 实验目录"

OUT_EXP_DIR = exp_dirs[0].resolve()
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 SUMMARY_ROOT:", SUMMARY_ROOT)

# --------------------------------------------------
# 1) 公共输入（只读，不重复保存）
# --------------------------------------------------
RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"❌ BASELINE 不存在: {BASELINE_MASK_PATH}"

# --------------------------------------------------
# 2) 目标 run（适配你现在两个 StepA）
# --------------------------------------------------
TARGET_RUNS = [
    {"cellprob_threshold": 2.5, "min_size": 50},
    {"cellprob_threshold": 2.5, "min_size": 100},
]

TARGET_TAG = "P21_lr9e5_wd8e3"
ANISOTROPY = 2.0
DIAMETER = 8.0

def cp_to_str(cp):
    return str(cp).replace(".", "p").replace("-", "m")

TARGET_RUN_NAMES = [
    f"{TARGET_TAG}__a{ANISOTROPY}__d{DIAMETER}__cp{cp_to_str(x['cellprob_threshold'])}__ms{x['min_size']}"
    for x in TARGET_RUNS
]

print("\n🎯 target runs:")
for rn in TARGET_RUN_NAMES:
    print("   -", rn)

# --------------------------------------------------
# 3) 可调参数：可视化
# --------------------------------------------------
Z_VIS_LIST = [98, 100, 102]     # 主对比图看的 z 层
TOPK_LARGEST = 8                # 排查最大的前 K 个对象
CROP_HALF_SIZE = 96             # 实际 crop 大小约 192 x 192
SAVE_MAX_N_CROPS = 6            # 最多输出多少个局部 crop 图

# --------------------------------------------------
# 4) 可调参数：过滤
# --------------------------------------------------
GRAPH_MIN_VOLUME = 80
CENTROID_Q_LOW = 0.5
CENTROID_Q_HIGH = 99.5
Z_SCALE = 2.0
GRAPH_RADIUS = 35.0
ENABLE_BBOX_FILTER = True

# --------------------------------------------------
# 5) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)

    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))

    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr

    bd[0, :] = False
    bd[-1, :] = False
    bd[:, 0] = False
    bd[:, -1] = False
    return bd

def overlay_instance_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 1.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_labels = labels[valid]
    obj_sizes = counts[valid]

    if len(obj_sizes) == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "p95_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
            "n_ge_3x_median": 0,
            "n_ge_5x_median": 0,
            "top10_volumes": [],
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    p95_v = float(np.percentile(obj_sizes, 95))
    max_v = float(np.max(obj_sizes))
    med_safe = max(median_v, 1.0)

    large_ratio = float(np.mean(obj_sizes >= (2.0 * med_safe)))
    n_ge_3x = int(np.sum(obj_sizes >= 3.0 * med_safe))
    n_ge_5x = int(np.sum(obj_sizes >= 5.0 * med_safe))
    top10 = sorted(obj_sizes.tolist(), reverse=True)[:10]

    return {
        "total_cells": int(len(obj_sizes)),
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "p95_volume": p95_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
        "n_ge_3x_median": n_ge_3x,
        "n_ge_5x_median": n_ge_5x,
        "top10_volumes": top10,
    }

def get_topk_objects(mask3d: np.ndarray, topk=10):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    labels = labels[valid]
    counts = counts[valid]

    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    out = []
    for lab, vol in zip(labels[:topk], counts[:topk]):
        coords = np.argwhere(mask3d == lab)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)

        out.append({
            "label": int(lab),
            "volume": int(vol),
            "z_min": int(z0), "z_max": int(z1),
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "z_center": int(cz), "y_center": int(cy), "x_center": int(cx),
            "z_span": int(z1 - z0 + 1),
            "y_span": int(y1 - y0 + 1),
            "x_span": int(x1 - x0 + 1),
        })
    return out

def safe_crop_2d(arr2d, cy, cx, half_size=96):
    H, W = arr2d.shape
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size)
    return arr2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def save_largest_object_crops(
    raw_stack,
    baseline_masks,
    pred_masks,
    largest_objs,
    crop_root,
    max_n=6,
    half_size=96
):
    crop_rows = []

    for i, obj in enumerate(largest_objs[:max_n], 1):
        lab = obj["label"]
        cz, cy, cx = obj["z_center"], obj["y_center"], obj["x_center"]

        raw2d = raw_stack[cz]
        base2d = baseline_masks[cz]
        pred2d = pred_masks[cz]

        raw_crop, crop_box = safe_crop_2d(raw2d, cy, cx, half_size=half_size)
        base_crop, _ = safe_crop_2d(base2d, cy, cx, half_size=half_size)
        pred_crop, _ = safe_crop_2d(pred2d, cy, cx, half_size=half_size)

        raw_norm = normalize_img(raw_crop)
        base_overlay = overlay_instance_boundary(raw_crop, base_crop, color="red")
        pred_overlay = overlay_instance_boundary(raw_crop, pred_crop, color="green")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title(f"Raw crop\nz={cz}, y={cy}, x={cx}")
        axes[0].axis("off")

        axes[1].imshow(base_overlay)
        axes[1].set_title("Baseline boundary")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(
            f"Pred boundary\nlabel={lab}, vol={obj['volume']}\n"
            f"zspan={obj['z_span']} yspan={obj['y_span']} xspan={obj['x_span']}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        crop_path = crop_root / f"largest_obj_rank{i:02d}__label{lab}__z{cz}_y{cy}_x{cx}.png"
        plt.savefig(crop_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        crop_rows.append({
            "rank_in_largest": i,
            "label": lab,
            "volume": obj["volume"],
            "z_center": cz,
            "y_center": cy,
            "x_center": cx,
            "z_span": obj["z_span"],
            "y_span": obj["y_span"],
            "x_span": obj["x_span"],
            "crop_path": str(crop_path),
            "crop_box_y0y1x0x1": str(crop_box),
        })

    return crop_rows

def get_instance_stats(mask3d: np.ndarray):
    labels = np.unique(mask3d)
    labels = labels[labels > 0]

    rows = []
    for lb in labels:
        zz, yy, xx = np.where(mask3d == lb)
        rows.append({
            "label": int(lb),
            "volume": int(len(zz)),
            "cz": float(np.mean(zz)),
            "cy": float(np.mean(yy)),
            "cx": float(np.mean(xx)),
            "zmin": int(np.min(zz)),
            "zmax": int(np.max(zz)),
            "ymin": int(np.min(yy)),
            "ymax": int(np.max(yy)),
            "xmin": int(np.min(xx)),
            "xmax": int(np.max(xx)),
        })
    return pd.DataFrame(rows)

def keep_largest_centroid_cluster(df_stats: pd.DataFrame, z_scale=2.0, radius=35.0):
    if len(df_stats) == 0:
        return set()

    pts = df_stats[["cz", "cy", "cx"]].to_numpy(dtype=float).copy()
    pts[:, 0] *= z_scale

    tree = cKDTree(pts)
    pairs = list(tree.query_pairs(r=radius))

    if len(pairs) == 0:
        idx = int(df_stats["volume"].idxmax())
        return {int(df_stats.loc[idx, "label"])}

    rows = []
    cols = []
    for i, j in pairs:
        rows.extend([i, j])
        cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(len(df_stats), len(df_stats)))

    n_comp, labels_cc = connected_components(graph, directed=False)
    df_tmp = df_stats.copy().reset_index(drop=True)
    df_tmp["cc"] = labels_cc

    group = df_tmp.groupby("cc").agg(
        n_inst=("label", "count"),
        sum_vol=("volume", "sum"),
    ).reset_index()

    group = group.sort_values(["n_inst", "sum_vol"], ascending=[False, False]).reset_index(drop=True)
    best_cc = int(group.iloc[0]["cc"])

    kept = set(df_tmp.loc[df_tmp["cc"] == best_cc, "label"].astype(int).tolist())
    return kept

def apply_label_keep(mask3d: np.ndarray, keep_labels: set):
    out = mask3d.copy()
    all_labels = np.unique(out)
    all_labels = all_labels[all_labels > 0]
    remove = [lb for lb in all_labels if int(lb) not in keep_labels]
    if len(remove) == 0:
        return out

    for lb in remove:
        out[out == lb] = 0
    return out

# --------------------------------------------------
# 6) 公共读取：raw + baseline，只读一次，避免重复
# --------------------------------------------------
print("\n[0/6] 读取公共 raw / baseline（只读一次）")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))

assert stack.ndim == 3, f"❌ raw 不是 3D: {stack.shape}"
assert baseline_masks.shape == stack.shape, "❌ baseline 与 raw shape 不一致"

print("✅ stack shape    :", stack.shape, stack.dtype)
print("✅ baseline shape :", baseline_masks.shape, baseline_masks.dtype)

analysis_summary_rows = []
filter_summary_rows = []

# --------------------------------------------------
# 7) 逐个 run 处理
# --------------------------------------------------
for run_name in TARGET_RUN_NAMES:
    print("\n" + "=" * 100)
    print("🎯 run_name:", run_name)
    print("=" * 100)

    run_root = (FULL_ROOT / run_name).resolve()
    raw_mask_root = (run_root / "raw_masks").resolve()
    stat_root = (run_root / "stats").resolve()
    fig_root = (run_root / "figures").resolve()

    crop_root = (fig_root / "largest_obj_crops").resolve()
    filtered_mask_root = (run_root / "filtered_masks").resolve()
    filtered_stat_root = (run_root / "filtered_stats").resolve()

    for p in [stat_root, fig_root, crop_root, filtered_mask_root, filtered_stat_root]:
        p.mkdir(parents=True, exist_ok=True)

    raw_mask_path = raw_mask_root / f"{run_name}__full_brain_3d_masks_raw.tif"
    assert raw_mask_path.exists(), f"❌ raw mask 不存在，请先跑 StepA: {raw_mask_path}"

    # run 专属输出
    out_stats_path = stat_root / f"{run_name}__analysis_stats.json"
    out_fig_path = fig_root / f"{run_name}__compare_multiz.png"
    out_largest_csv = stat_root / f"{run_name}__largest_objects.csv"
    out_largest_summary_json = stat_root / f"{run_name}__largest_objects.json"
    out_crop_csv = stat_root / f"{run_name}__largest_object_crops.csv"

    filtered_mask_path = filtered_mask_root / f"{run_name}__full_brain_3d_masks_filtered.tif"
    raw_instance_stats_csv = filtered_stat_root / f"{run_name}__instance_stats_raw.csv"
    filtered_summary_json = filtered_stat_root / f"{run_name}__filter_summary.json"

    print("📂 run_root               :", run_root)
    print("📂 raw_mask_path          :", raw_mask_path)
    print("📂 out_stats_path         :", out_stats_path)
    print("📂 out_fig_path           :", out_fig_path)
    print("📂 out_largest_csv        :", out_largest_csv)
    print("📂 out_crop_csv           :", out_crop_csv)
    print("📂 filtered_mask_path     :", filtered_mask_path)
    print("📂 raw_instance_stats_csv :", raw_instance_stats_csv)
    print("📂 filtered_summary_json  :", filtered_summary_json)

    # 从 run_name 里反解析参数，避免再维护一份 grid
    if "__cp2p5__ms50" in run_name:
        cellprob = 2.5
        min_size = 50
    elif "__cp2p5__ms100" in run_name:
        cellprob = 2.5
        min_size = 100
    else:
        raise RuntimeError(f"❌ 无法从 run_name 识别参数: {run_name}")

    print("[1/6] 读取 raw mask")
    masks = tiff.imread(str(raw_mask_path))
    assert masks.shape == stack.shape, f"❌ saved mask 与 raw shape 不一致: {masks.shape} vs {stack.shape}"
    print("✅ saved mask shape:", masks.shape, masks.dtype)

    print("[2/6] 统计 mask")
    stats = summarize_mask(masks)
    for k, v in stats.items():
        print(f"   - {k}: {v}")

    largest_objs = get_topk_objects(masks, topk=TOPK_LARGEST)
    largest_df = pd.DataFrame(largest_objs)
    largest_df.to_csv(out_largest_csv, index=False)
    print("✅ largest object table saved:", out_largest_csv)

    print("[3/6] 生成多 z 层实例边界对比图")
    valid_z_list = [z for z in Z_VIS_LIST if 0 <= z < stack.shape[0]]
    assert len(valid_z_list) > 0, "❌ Z_VIS_LIST 全部越界"

    fig, axes = plt.subplots(len(valid_z_list), 3, figsize=(14, 4.2 * len(valid_z_list)))
    if len(valid_z_list) == 1:
        axes = np.array([axes])

    for i, z in enumerate(valid_z_list):
        raw2d = stack[z]
        base2d = baseline_masks[z]
        pred2d = masks[z]

        raw_norm = normalize_img(raw2d)
        base_overlay = overlay_instance_boundary(raw2d, base2d, color="red")
        pred_overlay = overlay_instance_boundary(raw2d, pred2d, color="green")

        axes[i, 0].imshow(raw_norm, cmap="gray")
        axes[i, 0].set_title(f"Raw | z={z}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(base_overlay)
        axes[i, 1].set_title(f"Baseline | z={z}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(
            f"Pred | z={z}\n"
            f"a={ANISOTROPY}, d={DIAMETER}, cp={cellprob}, ms={min_size}\n"
            f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
            f"maxV={stats['max_volume']:.1f} | n>=3xmed={stats['n_ge_3x_median']}"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("✅ figure saved:", out_fig_path)

    print("[4/6] 生成最大对象局部 crop 排查图")
    crop_rows = save_largest_object_crops(
        raw_stack=stack,
        baseline_masks=baseline_masks,
        pred_masks=masks,
        largest_objs=largest_objs,
        crop_root=crop_root,
        max_n=SAVE_MAX_N_CROPS,
        half_size=CROP_HALF_SIZE
    )
    crop_df = pd.DataFrame(crop_rows)
    crop_df.to_csv(out_crop_csv, index=False)
    print("✅ largest object crop table saved:", out_crop_csv)

    largest_summary = {
        "run_name": run_name,
        "topk_largest": largest_objs,
        "crop_rows": crop_rows,
    }
    out_largest_summary_json.write_text(
        json.dumps(largest_summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("✅ largest object summary json saved:", out_largest_summary_json)

    analysis_record = {
        "run_name": run_name,
        "target_tag": TARGET_TAG,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "baseline_mask_path": str(BASELINE_MASK_PATH),
        "raw_mask_path": str(raw_mask_path),
        "fig_path": str(out_fig_path),
        "largest_object_csv": str(out_largest_csv),
        "largest_object_json": str(out_largest_summary_json),
        "largest_object_crop_csv": str(out_crop_csv),
        "shape": list(masks.shape),
        "anisotropy": ANISOTROPY,
        "diameter_3d": DIAMETER,
        "cellprob_threshold": cellprob,
        "min_size": min_size,
        **stats
    }
    out_stats_path.write_text(
        json.dumps(analysis_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("✅ analysis stats json saved:", out_stats_path)

    analysis_summary_rows.append({
        "run_name": run_name,
        "anisotropy": ANISOTROPY,
        "diameter": DIAMETER,
        "cellprob_threshold": cellprob,
        "min_size": min_size,
        "raw_mask_path": str(raw_mask_path),
        "fig_path": str(out_fig_path),
        "largest_object_csv": str(out_largest_csv),
        "largest_object_crop_csv": str(out_crop_csv),
        **stats
    })

    print("[5/6] 外围过滤")
    df_stats = get_instance_stats(masks)
    assert len(df_stats) > 0, f"❌ 没有任何实例: {run_name}"
    df_stats.to_csv(raw_instance_stats_csv, index=False)
    print("✅ raw instance stats saved:", raw_instance_stats_csv)

    z_lo, z_hi = np.percentile(df_stats["cz"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])
    y_lo, y_hi = np.percentile(df_stats["cy"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])
    x_lo, x_hi = np.percentile(df_stats["cx"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])

    if ENABLE_BBOX_FILTER:
        bbox_keep = df_stats[
            (df_stats["cz"] >= z_lo) & (df_stats["cz"] <= z_hi) &
            (df_stats["cy"] >= y_lo) & (df_stats["cy"] <= y_hi) &
            (df_stats["cx"] >= x_lo) & (df_stats["cx"] <= x_hi)
        ].copy()
    else:
        bbox_keep = df_stats.copy()

    print("bbox_keep instances:", len(bbox_keep))

    graph_df = bbox_keep[bbox_keep["volume"] >= GRAPH_MIN_VOLUME].copy()
    print("graph_df instances:", len(graph_df))

    if len(graph_df) == 0:
        print("⚠️ 没有实例满足 GRAPH_MIN_VOLUME，退化成 bbox_keep")
        keep_labels = set(bbox_keep["label"].astype(int).tolist())
    else:
        keep_labels = keep_largest_centroid_cluster(
            graph_df,
            z_scale=Z_SCALE,
            radius=GRAPH_RADIUS
        )

        cc_df = bbox_keep[bbox_keep["label"].isin(list(keep_labels))].copy()

        z_lo2, z_hi2 = cc_df["cz"].min(), cc_df["cz"].max()
        y_lo2, y_hi2 = cc_df["cy"].min(), cc_df["cy"].max()
        x_lo2, x_hi2 = cc_df["cx"].min(), cc_df["cx"].max()

        expanded = bbox_keep[
            (bbox_keep["cz"] >= z_lo2 - 5) & (bbox_keep["cz"] <= z_hi2 + 5) &
            (bbox_keep["cy"] >= y_lo2 - 25) & (bbox_keep["cy"] <= y_hi2 + 25) &
            (bbox_keep["cx"] >= x_lo2 - 25) & (bbox_keep["cx"] <= x_hi2 + 25)
        ].copy()

        keep_labels = set(expanded["label"].astype(int).tolist())

    print("kept labels:", len(keep_labels))

    print("[6/6] 保存 filtered mask + filter summary")
    mask_filtered = apply_label_keep(masks, keep_labels)
    tiff.imwrite(str(filtered_mask_path), mask_filtered.astype(np.uint32))
    print("✅ saved filtered mask:", filtered_mask_path)

    raw_total = int(df_stats["label"].nunique())
    filt_df = df_stats[df_stats["label"].isin(list(keep_labels))].copy()
    filt_total = int(filt_df["label"].nunique())

    filter_summary = {
        "run_name": run_name,
        "cellprob_threshold": cellprob,
        "min_size": min_size,
        "raw_mask_path": str(raw_mask_path),
        "filtered_mask_path": str(filtered_mask_path),
        "raw_total_instances": raw_total,
        "filtered_total_instances": filt_total,
        "removed_instances": int(raw_total - filt_total),
        "removed_ratio": float((raw_total - filt_total) / max(raw_total, 1)),
        "graph_min_volume": GRAPH_MIN_VOLUME,
        "centroid_q_low": CENTROID_Q_LOW,
        "centroid_q_high": CENTROID_Q_HIGH,
        "z_scale": Z_SCALE,
        "graph_radius": GRAPH_RADIUS,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "stats_csv_raw": str(raw_instance_stats_csv),
    }
    filtered_summary_json.write_text(
        json.dumps(filter_summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("✅ saved filter summary:", filtered_summary_json)

    filter_summary_rows.append(filter_summary)

# --------------------------------------------------
# 8) 公共 summary：只保存一份
# --------------------------------------------------
analysis_summary_df = pd.DataFrame(analysis_summary_rows)
analysis_summary_csv = SUMMARY_ROOT / "fullbrain_top2_analysis_summary.csv"
analysis_summary_df.to_csv(analysis_summary_csv, index=False)

filter_summary_df = pd.DataFrame(filter_summary_rows)
filter_summary_csv = SUMMARY_ROOT / "fullbrain_top2_filter_summary.csv"
filter_summary_df.to_csv(filter_summary_csv, index=False)

print("\n" + "=" * 100)
print("🏁 StepB analyze+filter finished")
print("📂 analysis_summary_csv:", analysis_summary_csv)
print("📂 filter_summary_csv  :", filter_summary_csv)
print("=" * 100)

print("\n✅ analysis summary:")
display(analysis_summary_df)

print("\n✅ filter summary:")
display(filter_summary_df)


# In[ ]:


# ==========================================
# Cell Full3D-StepB-Compare-v2
# 显示 raw vs filtered 的 whole-brain dense slices
# 直接在 cell 输出里看
# ==========================================
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 100)
print("🖼️ Full3D-StepB-Compare-v2 | raw vs filtered whole-brain slices")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 3.18 实验目录"

OUT_EXP_DIR = exp_dirs[0].resolve()
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()

filter_summary_csv = SUMMARY_ROOT / "fullbrain_top2_filter_summary.csv"
assert filter_summary_csv.exists(), f"❌ filter_summary_csv 不存在: {filter_summary_csv}"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

TOPN_SLICES = 3

# 可选：
RUN_NAME_OVERRIDE = None
# 例如：
# RUN_NAME_OVERRIDE = "P21_lr9e5_wd8e3__a2.0__d8.0__cp2p5__ms50"
# RUN_NAME_OVERRIDE = "P21_lr9e5_wd8e3__a2.0__d8.0__cp2p5__ms100"

summary_df = pd.read_csv(filter_summary_csv)
assert len(summary_df) > 0, "❌ summary_df 为空"

if RUN_NAME_OVERRIDE is not None:
    target_df = summary_df[summary_df["run_name"] == RUN_NAME_OVERRIDE].copy()
    assert len(target_df) == 1, f"❌ 没找到 run_name={RUN_NAME_OVERRIDE}"
    row = target_df.iloc[0]
else:
    row = summary_df.iloc[0]

run_name = row["run_name"]
raw_mask_path = Path(row["raw_mask_path"]).resolve()
filtered_mask_path = Path(row["filtered_mask_path"]).resolve()

print("🎯 run_name          :", run_name)
print("raw_mask_path       :", raw_mask_path)
print("filtered_mask_path  :", filtered_mask_path)

raw = tiff.imread(str(RAW_3D_STACK_PATH))
mask_raw = tiff.imread(str(raw_mask_path))
mask_filt = tiff.imread(str(filtered_mask_path))

assert raw.shape == mask_raw.shape == mask_filt.shape, "❌ raw/rawmask/filtmask shape 不一致"

def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)
    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))
    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr
    bd[0, :] = bd[-1, :] = bd[:, 0] = bd[:, -1] = False
    return bd

def overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)
    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 0.0]
    return overlay

def count_instances_per_slice(mask3d: np.ndarray):
    out = []
    for z in range(mask3d.shape[0]):
        labels = np.unique(mask3d[z])
        labels = labels[labels > 0]
        out.append(int(len(labels)))
    return np.array(out, dtype=int)

cells_per_z = count_instances_per_slice(mask_raw)
dense_df = pd.DataFrame({
    "z": np.arange(mask_raw.shape[0]),
    "num_cells_raw": cells_per_z
}).sort_values(by=["num_cells_raw", "z"], ascending=[False, True]).reset_index(drop=True)

top_df = dense_df.head(TOPN_SLICES).copy()
display(top_df)

fig, axes = plt.subplots(len(top_df), 3, figsize=(16, 5 * len(top_df)))
if len(top_df) == 1:
    axes = np.array([axes])

for r, (_, info) in enumerate(top_df.iterrows()):
    z = int(info["z"])

    axes[r, 0].imshow(normalize_img(raw[z]), cmap="gray")
    axes[r, 0].set_title(f"raw | z={z}")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(overlay_boundary(raw[z], mask_raw[z], color="red"))
    axes[r, 1].set_title(f"raw mask overlay | z={z}")
    axes[r, 1].axis("off")

    axes[r, 2].imshow(overlay_boundary(raw[z], mask_filt[z], color="green"))
    axes[r, 2].set_title(f"filtered overlay | z={z}")
    axes[r, 2].axis("off")

plt.suptitle(run_name, fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

print("\n✅ 比较完成。红色=过滤前，绿色=过滤后。")


# In[ ]:


# ==========================================
# Cell Full3D-StepB-Analyze-And-Filter：
# 读取 StepA 的 raw masks，做统计 + 外围过滤 + 保存 filtered masks
# 过滤逻辑 v1：
#   1) robust centroid bbox
#   2) centroid graph 最大主群
#   3) 小体积实例先剔除出建图集合（避免噪点主导）
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

print("=" * 100)
print("🧪 Full3D-StepB-Analyze-And-Filter")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 3.18 实验目录"

OUT_EXP_DIR = exp_dirs[0].resolve()
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
CODE_ROOT = (FULL_ROOT / "code").resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()
SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

grid_csv = CODE_ROOT / "grid_fullbrain_top2.csv"
assert grid_csv.exists(), f"❌ grid_csv 不存在: {grid_csv}"

grid_df = pd.read_csv(grid_csv)

print("📂 FULL_ROOT :", FULL_ROOT)
print("📂 SUMMARY_ROOT:", SUMMARY_ROOT)

# --------------------------------------------------
# 1) 过滤参数（第一版先求稳）
# --------------------------------------------------
# 小于这个体素数的实例，不参与“主群建图”
GRAPH_MIN_VOLUME = 80

# 质心 robust bbox：只保留主分布
CENTROID_Q_LOW = 0.5
CENTROID_Q_HIGH = 99.5

# 建图半径（单位：voxel 坐标系）
# z 方向更稀，先给 z 乘一个缩放
Z_SCALE = 2.0
GRAPH_RADIUS = 35.0

# 是否在 bbox 外的实例直接删
ENABLE_BBOX_FILTER = True

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
def get_instance_stats(mask3d: np.ndarray):
    labels = np.unique(mask3d)
    labels = labels[labels > 0]

    rows = []
    for lb in labels:
        zz, yy, xx = np.where(mask3d == lb)
        rows.append({
            "label": int(lb),
            "volume": int(len(zz)),
            "cz": float(np.mean(zz)),
            "cy": float(np.mean(yy)),
            "cx": float(np.mean(xx)),
            "zmin": int(np.min(zz)),
            "zmax": int(np.max(zz)),
            "ymin": int(np.min(yy)),
            "ymax": int(np.max(yy)),
            "xmin": int(np.min(xx)),
            "xmax": int(np.max(xx)),
        })
    return pd.DataFrame(rows)

def keep_largest_centroid_cluster(df_stats: pd.DataFrame, z_scale=2.0, radius=35.0):
    if len(df_stats) == 0:
        return set()

    pts = df_stats[["cz", "cy", "cx"]].to_numpy(dtype=float).copy()
    pts[:, 0] *= z_scale

    tree = cKDTree(pts)
    pairs = list(tree.query_pairs(r=radius))

    if len(pairs) == 0:
        # 没有边，退化成保留体积最大的那个
        idx = int(df_stats["volume"].idxmax())
        return {int(df_stats.loc[idx, "label"])}

    rows = []
    cols = []
    for i, j in pairs:
        rows.extend([i, j])
        cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(len(df_stats), len(df_stats)))

    n_comp, labels_cc = connected_components(graph, directed=False)
    df_tmp = df_stats.copy().reset_index(drop=True)
    df_tmp["cc"] = labels_cc

    # 按组件中实例数最多优先；若打平，按总体积最大
    group = df_tmp.groupby("cc").agg(
        n_inst=("label", "count"),
        sum_vol=("volume", "sum"),
    ).reset_index()

    group = group.sort_values(["n_inst", "sum_vol"], ascending=[False, False]).reset_index(drop=True)
    best_cc = int(group.iloc[0]["cc"])

    kept = set(df_tmp.loc[df_tmp["cc"] == best_cc, "label"].astype(int).tolist())
    return kept

def apply_label_keep(mask3d: np.ndarray, keep_labels: set):
    out = mask3d.copy()
    all_labels = np.unique(out)
    all_labels = all_labels[all_labels > 0]
    remove = [lb for lb in all_labels if int(lb) not in keep_labels]
    if len(remove) == 0:
        return out

    remove = np.array(remove)
    # 简单做法：逐个清零，数量通常可接受
    for lb in remove:
        out[out == lb] = 0
    return out

rows_summary = []

# --------------------------------------------------
# 3) 逐个 run 分析
# --------------------------------------------------
for _, row in grid_df.iterrows():
    run_name = row["run_name"]
    run_root = Path(row["run_root"]).resolve()
    raw_mask_root = Path(row["raw_mask_root"]).resolve()
    stat_root = Path(row["stat_root"]).resolve()
    fig_root = Path(row["fig_root"]).resolve()

    filtered_mask_root = (run_root / "filtered_masks").resolve()
    filtered_stat_root = (run_root / "filtered_stats").resolve()
    for p in [filtered_mask_root, filtered_stat_root]:
        p.mkdir(parents=True, exist_ok=True)

    raw_mask_path = raw_mask_root / f"{run_name}__full_brain_3d_masks_raw.tif"
    assert raw_mask_path.exists(), f"❌ raw mask 不存在: {raw_mask_path}"

    print("\n" + "-" * 100)
    print("🎯 run_name:", run_name)
    print("raw_mask_path:", raw_mask_path)

    mask_raw = tiff.imread(str(raw_mask_path))
    assert mask_raw.ndim == 3, f"❌ raw mask 不是 3D: {mask_raw.shape}"

    print("[1/5] 提取实例统计 ...")
    df_stats = get_instance_stats(mask_raw)
    assert len(df_stats) > 0, f"❌ 没有任何实例: {run_name}"

    stats_csv = filtered_stat_root / f"{run_name}__instance_stats_raw.csv"
    df_stats.to_csv(stats_csv, index=False)

    # --------------------------------------------------
    # bbox filter
    # --------------------------------------------------
    print("[2/5] robust centroid bbox ...")
    z_lo, z_hi = np.percentile(df_stats["cz"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])
    y_lo, y_hi = np.percentile(df_stats["cy"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])
    x_lo, x_hi = np.percentile(df_stats["cx"], [CENTROID_Q_LOW, CENTROID_Q_HIGH])

    if ENABLE_BBOX_FILTER:
        bbox_keep = df_stats[
            (df_stats["cz"] >= z_lo) & (df_stats["cz"] <= z_hi) &
            (df_stats["cy"] >= y_lo) & (df_stats["cy"] <= y_hi) &
            (df_stats["cx"] >= x_lo) & (df_stats["cx"] <= x_hi)
        ].copy()
    else:
        bbox_keep = df_stats.copy()

    print("bbox_keep instances:", len(bbox_keep))

    # --------------------------------------------------
    # 最大主群
    # --------------------------------------------------
    print("[3/5] centroid largest-cluster filtering ...")
    graph_df = bbox_keep[bbox_keep["volume"] >= GRAPH_MIN_VOLUME].copy()
    print("graph_df instances:", len(graph_df))

    if len(graph_df) == 0:
        print("⚠️ 没有实例满足 GRAPH_MIN_VOLUME，退化成 bbox_keep")
        keep_labels = set(bbox_keep["label"].astype(int).tolist())
    else:
        keep_labels = keep_largest_centroid_cluster(
            graph_df,
            z_scale=Z_SCALE,
            radius=GRAPH_RADIUS
        )

        # 对于 bbox_keep 中落在主群邻域的点，一起保留；这里先简单策略：
        # 保留主群标签 + 主群 bbox 内的实例
        cc_df = bbox_keep[bbox_keep["label"].isin(list(keep_labels))].copy()

        z_lo2, z_hi2 = cc_df["cz"].min(), cc_df["cz"].max()
        y_lo2, y_hi2 = cc_df["cy"].min(), cc_df["cy"].max()
        x_lo2, x_hi2 = cc_df["cx"].min(), cc_df["cx"].max()

        expanded = bbox_keep[
            (bbox_keep["cz"] >= z_lo2 - 5) & (bbox_keep["cz"] <= z_hi2 + 5) &
            (bbox_keep["cy"] >= y_lo2 - 25) & (bbox_keep["cy"] <= y_hi2 + 25) &
            (bbox_keep["cx"] >= x_lo2 - 25) & (bbox_keep["cx"] <= x_hi2 + 25)
        ].copy()

        keep_labels = set(expanded["label"].astype(int).tolist())

    print("kept labels:", len(keep_labels))

    # --------------------------------------------------
    # 应用过滤
    # --------------------------------------------------
    print("[4/5] apply filtered labels ...")
    mask_filtered = apply_label_keep(mask_raw, keep_labels)

    filtered_mask_path = filtered_mask_root / f"{run_name}__full_brain_3d_masks_filtered.tif"
    tiff.imwrite(str(filtered_mask_path), mask_filtered.astype(np.uint32))
    print("✅ saved:", filtered_mask_path)

    # --------------------------------------------------
    # 保存 summary
    # --------------------------------------------------
    print("[5/5] save summaries ...")
    raw_total = int(df_stats["label"].nunique())
    filt_df = df_stats[df_stats["label"].isin(list(keep_labels))].copy()
    filt_total = int(filt_df["label"].nunique())

    summary = {
        "run_name": run_name,
        "cellprob_threshold": float(row["cellprob_threshold"]),
        "min_size": int(row["min_size"]),
        "raw_mask_path": str(raw_mask_path),
        "filtered_mask_path": str(filtered_mask_path),
        "raw_total_instances": raw_total,
        "filtered_total_instances": filt_total,
        "removed_instances": int(raw_total - filt_total),
        "removed_ratio": float((raw_total - filt_total) / max(raw_total, 1)),
        "graph_min_volume": GRAPH_MIN_VOLUME,
        "centroid_q_low": CENTROID_Q_LOW,
        "centroid_q_high": CENTROID_Q_HIGH,
        "z_scale": Z_SCALE,
        "graph_radius": GRAPH_RADIUS,
        "bbox_filter_enabled": ENABLE_BBOX_FILTER,
        "stats_csv_raw": str(stats_csv),
    }

    summary_json = filtered_stat_root / f"{run_name}__filter_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ saved:", summary_json)

    rows_summary.append(summary)

summary_df = pd.DataFrame(rows_summary)
summary_csv = SUMMARY_ROOT / "fullbrain_top2_filter_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 100)
print("🏁 StepB analyze+filter finished")
print("summary_csv:", summary_csv)
display(summary_df)
print("=" * 100)


# In[ ]:


# ==========================================
# Cell Full3D-StepB-Compare：
# 显示 raw vs filtered 的 whole-brain top3 dense slices
# 直接在 cell 输出里看，不用下载
# ==========================================
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 100)
print("🖼️ Full3D-StepB-Compare | raw vs filtered whole-brain slices")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位目录
# --------------------------------------------------
RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs").resolve()
exp_dirs = sorted(
    RUNS_ROOT.glob("exp_20260318_ori_train_9runs_400ep_es50_*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)
assert len(exp_dirs) > 0, "❌ 没找到 3.18 实验目录"

OUT_EXP_DIR = exp_dirs[0].resolve()
FULL_RUN_NAME = "fullbrain_top2_postfilter_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()

summary_csv = SUMMARY_ROOT / "fullbrain_top2_filter_summary.csv"
assert summary_csv.exists(), f"❌ summary_csv 不存在: {summary_csv}"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()
assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

TOPN_SLICES = 3
RUN_NAME_OVERRIDE = None
# 例如：
# RUN_NAME_OVERRIDE = "P21_lr9e5_wd8e3__a2.0__d8.0__cp2p5__ms50"

summary_df = pd.read_csv(summary_csv)
assert len(summary_df) > 0, "❌ summary_df 为空"

if RUN_NAME_OVERRIDE is not None:
    target_df = summary_df[summary_df["run_name"] == RUN_NAME_OVERRIDE].copy()
    assert len(target_df) == 1, f"❌ 没找到 run_name={RUN_NAME_OVERRIDE}"
    row = target_df.iloc[0]
else:
    row = summary_df.iloc[0]

run_name = row["run_name"]
raw_mask_path = Path(row["raw_mask_path"]).resolve()
filtered_mask_path = Path(row["filtered_mask_path"]).resolve()

print("🎯 run_name          :", run_name)
print("raw_mask_path       :", raw_mask_path)
print("filtered_mask_path  :", filtered_mask_path)

raw = tiff.imread(str(RAW_3D_STACK_PATH))
mask_raw = tiff.imread(str(raw_mask_path))
mask_filt = tiff.imread(str(filtered_mask_path))

assert raw.shape == mask_raw.shape == mask_filt.shape, "❌ raw/rawmask/filtmask shape 不一致"

def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)
    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))
    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr
    bd[0, :] = bd[-1, :] = bd[:, 0] = bd[:, -1] = False
    return bd

def overlay_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)
    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 0.0]
    return overlay

def count_instances_per_slice(mask3d: np.ndarray):
    out = []
    for z in range(mask3d.shape[0]):
        labels = np.unique(mask3d[z])
        labels = labels[labels > 0]
        out.append(int(len(labels)))
    return np.array(out, dtype=int)

cells_per_z = count_instances_per_slice(mask_raw)
dense_df = pd.DataFrame({
    "z": np.arange(mask_raw.shape[0]),
    "num_cells_raw": cells_per_z
}).sort_values(by=["num_cells_raw", "z"], ascending=[False, True]).reset_index(drop=True)

top_df = dense_df.head(TOPN_SLICES).copy()
display(top_df)

fig, axes = plt.subplots(len(top_df), 3, figsize=(16, 5 * len(top_df)))
if len(top_df) == 1:
    axes = np.array([axes])

for r, (_, info) in enumerate(top_df.iterrows()):
    z = int(info["z"])

    axes[r, 0].imshow(normalize_img(raw[z]), cmap="gray")
    axes[r, 0].set_title(f"raw | z={z}")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(overlay_boundary(raw[z], mask_raw[z], color="red"))
    axes[r, 1].set_title(f"raw mask overlay | z={z}")
    axes[r, 1].axis("off")

    axes[r, 2].imshow(overlay_boundary(raw[z], mask_filt[z], color="green"))
    axes[r, 2].set_title(f"filtered overlay | z={z}")
    axes[r, 2].axis("off")

plt.suptitle(run_name, fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

print("\n✅ 比较完成。红色=过滤前，绿色=过滤后。")


# In[ ]:





# In[1]:


# ==========================================
# Cell Baseline-StepA-Top1Aligned：
# 用 baseline(cpsam) + 与微调模型 Top1 完全相同的参数
# 跑 full-brain 3D，并优先保存 mask
#
# 目的：
#   - 与 finetuned model 做公平对比
#   - 唯一变量：模型本身
# ==========================================
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from cellpose import models, io

try:
    import torch
except ImportError:
    torch = None

print("=" * 100)
print("🧪 Baseline-StepA-Top1Aligned | Run baseline(cpsam) with same params as finetuned Top1")
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
# 1) 固定路径
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"

# --------------------------------------------------
# 2) 与微调模型 Top1 完全对齐的参数
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = 8.0
DO_3D = True
Z_AXIS = 0
BATCH_SIZE_3D = 4
STITCH_THRESHOLD = 0.0
FORCE_RERUN = False

print("✅ aligned params:")
print("   cellprob_threshold :", CELLPROB_THRESHOLD)
print("   min_size           :", MIN_SIZE)
print("   anisotropy         :", ANISOTROPY)
print("   diameter           :", DIAMETER)
print("   do_3D              :", DO_3D)
print("   z_axis             :", Z_AXIS)
print("   batch_size_3d      :", BATCH_SIZE_3D)
print("   stitch_threshold   :", STITCH_THRESHOLD)

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "baseline_vs_finetuned_20260318_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
BASELINE_ROOT = (FULL_ROOT / "baseline_cpsam_top1_aligned").resolve()
MASK_ROOT = (BASELINE_ROOT / "masks").resolve()
STAT_ROOT = (BASELINE_ROOT / "stats").resolve()

for p in [FULL_ROOT, BASELINE_ROOT, MASK_ROOT, STAT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

cp_str = str(CELLPROB_THRESHOLD).replace(".", "p").replace("-", "m")
run_name = f"BASELINE_cpsam__{TARGET_TAG}__a{ANISOTROPY}__d{DIAMETER}__cp{cp_str}__ms{MIN_SIZE}"

out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
out_stats_path = STAT_ROOT / f"{run_name}__stepA_meta.json"
stepA_summary_csv = BASELINE_ROOT / "baseline_cpsam_top1_aligned__stepA_mask_only.csv"

print("📂 OUT_EXP_DIR      :", OUT_EXP_DIR)
print("📂 FULL_ROOT        :", FULL_ROOT)
print("📂 BASELINE_ROOT    :", BASELINE_ROOT)
print("🎯 run_name         :", run_name)
print("📂 out_mask_path    :", out_mask_path)
print("📂 out_stats_path   :", out_stats_path)
print("📂 stepA_summary_csv:", stepA_summary_csv)

# --------------------------------------------------
# 4) 工具函数
# --------------------------------------------------
def safe_cuda_cleanup():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# --------------------------------------------------
# 5) 读取 raw 3D stack
# --------------------------------------------------
print("\n[Step 1/3] 读取 raw 3D stack")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
assert stack.ndim == 3, f"❌ raw 不是 3D: {stack.shape}"
print("✅ stack shape:", stack.shape, "dtype:", stack.dtype)

# --------------------------------------------------
# 6) 加载 baseline 模型
# --------------------------------------------------
print("[Step 2/3] 加载 baseline 模型 cpsam")
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
print("✅ baseline model loaded: cpsam")

# --------------------------------------------------
# 7) 推理并优先保存 mask
# --------------------------------------------------
masks_shape = None
masks_dtype = None

print("\n" + "=" * 100)
print("Baseline(cpsam) with Top1-aligned params")
print("mask exists:", out_mask_path.exists(), out_mask_path)
print("=" * 100)

if out_mask_path.exists() and out_stats_path.exists() and not FORCE_RERUN:
    print("✅ 已存在 baseline mask 和 meta，直接跳过推理")
    elapsed_s = np.nan
else:
    print("[Step 3/3] 开始 eval")
    print(f"🚦 BATCH_SIZE_3D = {BATCH_SIZE_3D}")
    t0 = time.time()

    masks, flows, styles = model.eval(
        stack,
        diameter=DIAMETER,
        do_3D=DO_3D,
        stitch_threshold=STITCH_THRESHOLD,
        z_axis=Z_AXIS,
        batch_size=BATCH_SIZE_3D,
        anisotropy=ANISOTROPY,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        progress=True
    )
    print("✅ eval returned")

    masks = np.asarray(masks)
    elapsed_s = float(time.time() - t0)
    masks_shape = list(masks.shape)
    masks_dtype = str(masks.dtype)

    print("✅ masks asarray done")
    print("   masks shape:", masks_shape)
    print("   masks dtype:", masks_dtype)
    print(f"   elapsed: {elapsed_s/60:.2f} min")

    print("写出 baseline mask tif")
    if np.issubdtype(masks.dtype, np.integer):
        tiff.imwrite(str(out_mask_path), masks)
    else:
        tiff.imwrite(str(out_mask_path), masks.astype(np.uint32))

    print("✅ baseline mask tif saved:", out_mask_path)

    del flows, styles
    safe_cuda_cleanup()

minimal_record = {
    "model_name": "cpsam",
    "param_source": "manual_align_to_finetuned_top1",
    "target_tag": TARGET_TAG,
    "run_name": run_name,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "mask_path": str(out_mask_path),
    "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
    "anisotropy": ANISOTROPY,
    "diameter_3d": DIAMETER,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "batch_size_3d": BATCH_SIZE_3D,
    "do_3D": DO_3D,
    "z_axis": Z_AXIS,
    "masks_shape": masks_shape,
    "masks_dtype": masks_dtype,
    "step": "A_save_mask_only",
}
out_stats_path.write_text(
    json.dumps(minimal_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ minimal json saved:", out_stats_path)

summary_df = pd.DataFrame([{
    "model_name": "cpsam",
    "run_name": run_name,
    "anisotropy": ANISOTROPY,
    "diameter": DIAMETER,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "batch_size_3d": BATCH_SIZE_3D,
    "elapsed_s": elapsed_s,
    "mask_path": str(out_mask_path),
    "status": "MASK_SAVED_ONLY"
}])
summary_df.to_csv(stepA_summary_csv, index=False)

print("\n✅ StepA summary saved:", stepA_summary_csv)
display(summary_df)


# In[ ]:





# In[ ]:


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

FLOW_THRESHOLD = 0.4

# --------------------------------------------------
# 3) 输出目录
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale175_dNone_cp0_ms50_flow08_v1"
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


# In[4]:


# ==========================================
# Cell Full3D-StepB-Top1-Analyze：
# 适配新版 StepA-Top1 (Version B / rescale版) 输出：
# 读取已保存 raw mask，做实例级统计 +
# 正确边界可视化 + Top-K 大对象局部排查
#
# 对应新版 StepA 输出：
#   FULL_ROOT / run_name / raw_masks / {run_name}__full_brain_3d_masks_raw.tif
#
# 适配点：
#   - FULL_RUN_NAME 改为 fullbrain_top1_rescale175_dNone_20260321_v1
#   - run_name 改为包含 dNone + rs1p75
#   - 分析展示参数改为读取新版 StepA 的 diameter=None, rescale=1.75
# ==========================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

print("=" * 100)
print("🧪 Full3D-StepB-Top1-Analyze | Analyze saved Top1 raw mask (Version B / rescale)")
print("=" * 100)

# --------------------------------------------------
# 0) 自动定位最新实验目录
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
# 1) 固定输入
# --------------------------------------------------
TARGET_TAG = "P21_lr9e5_wd8e3"

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"❌ RAW 不存在: {RAW_3D_STACK_PATH}"
assert BASELINE_MASK_PATH.exists(), f"❌ BASELINE 不存在: {BASELINE_MASK_PATH}"

# --------------------------------------------------
# 2) 对应新版 StepA 的固定参数（必须和 StepA 一致）
# --------------------------------------------------
CELLPROB_THRESHOLD = 2.5
MIN_SIZE = 50

ANISOTROPY = 2.0
DIAMETER = None
RESCALE = 14.0 / 8.0   # 1.75
STITCH_THRESHOLD = 0.0

# --------------------------------------------------
# 3) 输出目录（必须和新版 StepA 一致）
# --------------------------------------------------
FULL_RUN_NAME = "fullbrain_top1_rescale175_dNone_20260321_v1"
FULL_ROOT = (OUT_EXP_DIR / FULL_RUN_NAME).resolve()
SUMMARY_ROOT = (FULL_ROOT / "summary").resolve()

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
CROP_ROOT = (FIG_ROOT / "largest_obj_crops").resolve()

for p in [SUMMARY_ROOT, STAT_ROOT, FIG_ROOT, CROP_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("📂 OUT_EXP_DIR :", OUT_EXP_DIR)
print("📂 FULL_ROOT   :", FULL_ROOT)
print("📂 RUN_ROOT    :", RUN_ROOT)

# --------------------------------------------------
# 4) 输入 / 输出路径
# --------------------------------------------------
mask_path = RAW_MASK_ROOT / f"{run_name}__full_brain_3d_masks_raw.tif"
stepA_meta_path = STAT_ROOT / f"{run_name}__stepA_meta.json"

out_stats_path = STAT_ROOT / f"{run_name}__analysis_stats.json"
out_fig_path = FIG_ROOT / f"{run_name}__compare_multiz.png"
out_largest_csv = STAT_ROOT / f"{run_name}__largest_objects.csv"
out_largest_summary_json = STAT_ROOT / f"{run_name}__largest_objects.json"
out_crop_csv = STAT_ROOT / f"{run_name}__largest_object_crops.csv"

full_summary_csv = SUMMARY_ROOT / "fullbrain_top1_analysis_summary.csv"

assert mask_path.exists(), f"❌ mask tif 不存在，请先跑 StepA: {mask_path}"

print("🎯 run_name         :", run_name)
print("📂 mask_path        :", mask_path)
print("📂 stepA_meta_path  :", stepA_meta_path)
print("📂 out_stats_path   :", out_stats_path)
print("📂 out_fig_path     :", out_fig_path)
print("📂 out_largest_csv  :", out_largest_csv)
print("📂 out_crop_csv     :", out_crop_csv)
print("📂 full_summary_csv :", full_summary_csv)

# --------------------------------------------------
# 5) 可调参数
# --------------------------------------------------
Z_VIS_LIST = [98, 100, 102]     # 主对比图看的 z 层
TOPK_LARGEST = 8                # 排查最大的前 K 个对象
CROP_HALF_SIZE = 96             # 局部裁剪半径，实际 crop 大小约 192 x 192
SAVE_MAX_N_CROPS = 6            # 最多输出多少个局部 crop 图

# --------------------------------------------------
# 6) 工具函数
# --------------------------------------------------
def normalize_img(img2d: np.ndarray):
    img = img2d.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    return img

def mask_to_boundary_instance(mask2d: np.ndarray):
    """
    实例边界：
    只要相邻像素 label 不同，就认为是边界。
    这样能显示实例之间的内部边界，而不是只显示整体前景外轮廓。
    """
    m = mask2d.astype(np.int64)
    bd = np.zeros_like(m, dtype=bool)

    diff_ud = (m[:-1, :] != m[1:, :]) & (((m[:-1, :] > 0) | (m[1:, :] > 0)))
    diff_lr = (m[:, :-1] != m[:, 1:]) & (((m[:, :-1] > 0) | (m[:, 1:] > 0)))

    bd[:-1, :] |= diff_ud
    bd[1:,  :] |= diff_ud
    bd[:, :-1] |= diff_lr
    bd[:, 1: ] |= diff_lr

    bd[0, :] = False
    bd[-1, :] = False
    bd[:, 0] = False
    bd[:, -1] = False
    return bd

def overlay_instance_boundary(raw2d: np.ndarray, mask2d: np.ndarray, color="green"):
    raw_norm = normalize_img(raw2d)
    overlay = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
    bd = mask_to_boundary_instance(mask2d)

    if color == "red":
        overlay[bd] = [1.0, 0.0, 0.0]
    elif color == "green":
        overlay[bd] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        overlay[bd] = [1.0, 1.0, 0.0]
    else:
        overlay[bd] = [0.0, 1.0, 1.0]
    return overlay

def summarize_mask(mask3d: np.ndarray):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    obj_sizes = counts[valid]

    if len(obj_sizes) == 0:
        return {
            "total_cells": 0,
            "mean_volume": 0.0,
            "median_volume": 0.0,
            "p90_volume": 0.0,
            "p95_volume": 0.0,
            "max_volume": 0.0,
            "large_obj_ratio_ge_2x_median": 0.0,
            "n_ge_3x_median": 0,
            "n_ge_5x_median": 0,
            "top10_volumes": [],
        }

    median_v = float(np.median(obj_sizes))
    mean_v = float(np.mean(obj_sizes))
    p90_v = float(np.percentile(obj_sizes, 90))
    p95_v = float(np.percentile(obj_sizes, 95))
    max_v = float(np.max(obj_sizes))
    med_safe = max(median_v, 1.0)

    large_ratio = float(np.mean(obj_sizes >= (2.0 * med_safe)))
    n_ge_3x = int(np.sum(obj_sizes >= 3.0 * med_safe))
    n_ge_5x = int(np.sum(obj_sizes >= 5.0 * med_safe))
    top10 = sorted(obj_sizes.tolist(), reverse=True)[:10]

    return {
        "total_cells": int(len(obj_sizes)),
        "mean_volume": mean_v,
        "median_volume": median_v,
        "p90_volume": p90_v,
        "p95_volume": p95_v,
        "max_volume": max_v,
        "large_obj_ratio_ge_2x_median": large_ratio,
        "n_ge_3x_median": n_ge_3x,
        "n_ge_5x_median": n_ge_5x,
        "top10_volumes": top10,
    }

def get_topk_objects(mask3d: np.ndarray, topk=10):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = labels > 0
    labels = labels[valid]
    counts = counts[valid]

    order = np.argsort(counts)[::-1]
    labels = labels[order]
    counts = counts[order]

    out = []
    for lab, vol in zip(labels[:topk], counts[:topk]):
        coords = np.argwhere(mask3d == lab)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)

        out.append({
            "label": int(lab),
            "volume": int(vol),
            "z_min": int(z0), "z_max": int(z1),
            "y_min": int(y0), "y_max": int(y1),
            "x_min": int(x0), "x_max": int(x1),
            "z_center": int(cz), "y_center": int(cy), "x_center": int(cx),
            "z_span": int(z1 - z0 + 1),
            "y_span": int(y1 - y0 + 1),
            "x_span": int(x1 - x0 + 1),
        })
    return out

def safe_crop_2d(arr2d, cy, cx, half_size=96):
    H, W = arr2d.shape
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size)
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size)
    return arr2d[y0:y1, x0:x1], (y0, y1, x0, x1)

def save_largest_object_crops(
    raw_stack,
    baseline_masks,
    pred_masks,
    largest_objs,
    crop_root,
    max_n=6,
    half_size=96
):
    crop_rows = []

    for i, obj in enumerate(largest_objs[:max_n], 1):
        lab = obj["label"]
        cz, cy, cx = obj["z_center"], obj["y_center"], obj["x_center"]

        raw2d = raw_stack[cz]
        base2d = baseline_masks[cz]
        pred2d = pred_masks[cz]

        raw_crop, crop_box = safe_crop_2d(raw2d, cy, cx, half_size=half_size)
        base_crop, _ = safe_crop_2d(base2d, cy, cx, half_size=half_size)
        pred_crop, _ = safe_crop_2d(pred2d, cy, cx, half_size=half_size)

        raw_norm = normalize_img(raw_crop)
        base_overlay = overlay_instance_boundary(raw_crop, base_crop, color="red")
        pred_overlay = overlay_instance_boundary(raw_crop, pred_crop, color="green")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_norm, cmap="gray")
        axes[0].set_title(f"Raw crop\nz={cz}, y={cy}, x={cx}")
        axes[0].axis("off")

        axes[1].imshow(base_overlay)
        axes[1].set_title("Baseline boundary")
        axes[1].axis("off")

        axes[2].imshow(pred_overlay)
        axes[2].set_title(
            f"Pred boundary\nlabel={lab}, vol={obj['volume']}\n"
            f"zspan={obj['z_span']} yspan={obj['y_span']} xspan={obj['x_span']}"
        )
        axes[2].axis("off")

        plt.tight_layout()
        crop_path = crop_root / f"largest_obj_rank{i:02d}__label{lab}__z{cz}_y{cy}_x{cx}.png"
        plt.savefig(crop_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        crop_rows.append({
            "rank_in_largest": i,
            "label": lab,
            "volume": obj["volume"],
            "z_center": cz,
            "y_center": cy,
            "x_center": cx,
            "z_span": obj["z_span"],
            "y_span": obj["y_span"],
            "x_span": obj["x_span"],
            "crop_path": str(crop_path),
            "crop_box_y0y1x0x1": str(crop_box),
        })

    return crop_rows

# --------------------------------------------------
# 7) 读取数据
# --------------------------------------------------
print("[1/5] 读取 raw / baseline / saved raw mask")
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))
masks = tiff.imread(str(mask_path))

assert stack.ndim == 3, f"❌ raw 不是 3D: {stack.shape}"
assert baseline_masks.shape == stack.shape, "❌ baseline 与 raw shape 不一致"
assert masks.shape == stack.shape, "❌ saved mask 与 raw shape 不一致"

print("✅ stack shape     :", stack.shape, stack.dtype)
print("✅ baseline shape  :", baseline_masks.shape, baseline_masks.dtype)
print("✅ saved mask shape:", masks.shape, masks.dtype)

# --------------------------------------------------
# 8) 统计 mask
# --------------------------------------------------
print("[2/5] 统计 mask")
stats = summarize_mask(masks)
print("✅ stats:")
for k, v in stats.items():
    print(f"   - {k}: {v}")

largest_objs = get_topk_objects(masks, topk=TOPK_LARGEST)
largest_df = pd.DataFrame(largest_objs)
largest_df.to_csv(out_largest_csv, index=False)
print("✅ largest object table saved:", out_largest_csv)

# --------------------------------------------------
# 9) 多 z 层主对比图（实例边界）
# --------------------------------------------------
print("[3/5] 生成多 z 层实例边界对比图")
valid_z_list = [z for z in Z_VIS_LIST if 0 <= z < stack.shape[0]]
assert len(valid_z_list) > 0, "❌ Z_VIS_LIST 全部越界"

fig, axes = plt.subplots(len(valid_z_list), 3, figsize=(14, 4.2 * len(valid_z_list)))
if len(valid_z_list) == 1:
    axes = np.array([axes])

for i, z in enumerate(valid_z_list):
    raw2d = stack[z]
    base2d = baseline_masks[z]
    pred2d = masks[z]

    raw_norm = normalize_img(raw2d)
    base_overlay = overlay_instance_boundary(raw2d, base2d, color="red")
    pred_overlay = overlay_instance_boundary(raw2d, pred2d, color="green")

    axes[i, 0].imshow(raw_norm, cmap="gray")
    axes[i, 0].set_title(f"Raw | z={z}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(base_overlay)
    axes[i, 1].set_title(f"Baseline | z={z}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_overlay)
    axes[i, 2].set_title(
        f"Top1 | z={z}\n"
        f"a={ANISOTROPY}, d={DIAMETER}, rs={RESCALE}, cp={CELLPROB_THRESHOLD}, ms={MIN_SIZE}\n"
        f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
        f"maxV={stats['max_volume']:.1f} | n>=3xmed={stats['n_ge_3x_median']}"
    )
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print("✅ figure saved:", out_fig_path)

# --------------------------------------------------
# 10) 最大对象局部 crop 排查
# --------------------------------------------------
print("[4/5] 生成最大对象局部 crop 排查图")
crop_rows = save_largest_object_crops(
    raw_stack=stack,
    baseline_masks=baseline_masks,
    pred_masks=masks,
    largest_objs=largest_objs,
    crop_root=CROP_ROOT,
    max_n=SAVE_MAX_N_CROPS,
    half_size=CROP_HALF_SIZE
)
crop_df = pd.DataFrame(crop_rows)
crop_df.to_csv(out_crop_csv, index=False)
print("✅ largest object crop table saved:", out_crop_csv)

largest_summary = {
    "run_name": run_name,
    "topk_largest": largest_objs,
    "crop_rows": crop_rows,
}
out_largest_summary_json.write_text(
    json.dumps(largest_summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ largest object summary json saved:", out_largest_summary_json)

# --------------------------------------------------
# 11) 写 stats + 更新 full summary
# --------------------------------------------------
print("[5/5] 写 stats 和汇总表")

stepA_meta = None
if stepA_meta_path.exists():
    try:
        stepA_meta = json.loads(stepA_meta_path.read_text(encoding="utf-8"))
    except Exception:
        stepA_meta = None

stepA_elapsed_s = None
if isinstance(stepA_meta, dict):
    stepA_elapsed_s = stepA_meta.get("eval_elapsed_s", stepA_meta.get("elapsed_s"))

stats_record = {
    "run_name": run_name,
    "target_tag": TARGET_TAG,
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "baseline_mask_path": str(BASELINE_MASK_PATH),
    "mask_path": str(mask_path),
    "stepA_meta_path": str(stepA_meta_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_json": str(out_largest_summary_json),
    "largest_object_crop_csv": str(out_crop_csv),
    "shape": list(masks.shape),
    "anisotropy": ANISOTROPY,
    "diameter_3d": DIAMETER,
    "rescale": RESCALE,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "stepA_eval_elapsed_s": stepA_elapsed_s,
    **stats
}
out_stats_path.write_text(
    json.dumps(stats_record, indent=2, ensure_ascii=False),
    encoding="utf-8"
)
print("✅ stats json saved:", out_stats_path)

new_full_df = pd.DataFrame([{
    "run_name": run_name,
    "anisotropy": ANISOTROPY,
    "diameter": DIAMETER,
    "rescale": RESCALE,
    "cellprob_threshold": CELLPROB_THRESHOLD,
    "min_size": MIN_SIZE,
    "stitch_threshold": STITCH_THRESHOLD,
    "stepA_eval_elapsed_s": stepA_elapsed_s,
    "mask_path": str(mask_path),
    "fig_path": str(out_fig_path),
    "largest_object_csv": str(out_largest_csv),
    "largest_object_crop_csv": str(out_crop_csv),
    **stats
}])

if full_summary_csv.exists():
    old_full_df = pd.read_csv(full_summary_csv)
else:
    old_full_df = pd.DataFrame()

if len(old_full_df) > 0:
    merged_full_df = pd.concat([old_full_df, new_full_df], ignore_index=True)
    merged_full_df = merged_full_df.drop_duplicates(subset=["run_name"], keep="last")
else:
    merged_full_df = new_full_df

merged_full_df.to_csv(full_summary_csv, index=False)

print("\n✅ full summary saved:", full_summary_csv)
print("\n✅ largest objects:")
display(largest_df)

print("\n✅ largest object crops:")
display(crop_df)

print("\n✅ merged full summary:")
display(merged_full_df)


# In[ ]:


# ==========================================
# StepA-Post | 对 full-brain 3D raw mask 做“逐 2D 填坑”
#
# 功能：
#   1) 读取 StepA 输出的 3D label mask
#   2) 对每个实例 ID，在每个 z 切片上分别做 binary_fill_holes
#   3) 输出 filled mask（不覆盖 raw）
#   4) 保存统计信息 json
#
# 说明：
#   - 这是“按实例、按切片”的保守后处理
#   - 不会重新分割，不会新增新实例 ID
#   - 正常情况下不会改变细胞计数，只会让部分空心细胞变实心
# ==========================================

import json
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes

print("=" * 100)
print("🩹 StepA-Post | Fill holes slice-by-slice (2D) for 3D instance masks")
print("=" * 100)

# --------------------------------------------------
# 0) 路径配置：按你这次真实输出写死
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

OUT_ROOT = RAW_MASK_PATH.parent.parent / "post_fillholes_2d"
OUT_MASK_DIR = OUT_ROOT / "filled_masks"
OUT_STAT_DIR = OUT_ROOT / "stats"

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUT_STAT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MASK_PATH = OUT_MASK_DIR / RAW_MASK_PATH.name.replace(
    "_masks_raw.tif",
    "_masks_filled2d.tif"
)

OUT_META_PATH = OUT_STAT_DIR / META_PATH.name.replace(
    "__stepA_meta.json",
    "__fillholes2d_meta.json"
)

# --------------------------------------------------
# 1) 读取 raw mask
# --------------------------------------------------
print("\n[1/5] 读取 raw mask")
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

# --------------------------------------------------
# 2) 获取实例 ID
# --------------------------------------------------
print("\n[2/5] 收集实例 ID")
obj_ids = np.unique(masks)
obj_ids = obj_ids[obj_ids != 0]

num_instances_raw = int(len(obj_ids))
print(f"num_instances_raw = {num_instances_raw}")

# --------------------------------------------------
# 3) 逐 2D 填坑
# --------------------------------------------------
print("\n[3/5] 逐实例、逐 z 切片填坑")
t0 = time.time()

filled = np.zeros_like(masks, dtype=masks.dtype)

num_changed_instances = 0
num_total_added_voxels = 0
num_slices_touched = 0

Z = masks.shape[0]

for i, obj_id in enumerate(obj_ids, start=1):
    obj = (masks == obj_id)   # bool, shape=(Z, Y, X)
    obj_filled = np.zeros_like(obj, dtype=bool)

    obj_added_voxels = 0
    obj_slices_touched = 0

    for z in range(Z):
        sl = obj[z]

        # 如果这一层本来没有该实例，就跳过
        if not sl.any():
            continue

        sl_filled = binary_fill_holes(sl)

        added_voxels = int(sl_filled.sum() - sl.sum())
        if added_voxels > 0:
            obj_slices_touched += 1
            obj_added_voxels += added_voxels

        obj_filled[z] = sl_filled

    filled[obj_filled] = obj_id

    if obj_added_voxels > 0:
        num_changed_instances += 1
        num_total_added_voxels += obj_added_voxels
        num_slices_touched += obj_slices_touched

    if i % 500 == 0 or i == num_instances_raw:
        print(
            f"[progress] {i:>6d}/{num_instances_raw} | "
            f"changed_instances={num_changed_instances} | "
            f"added_voxels={num_total_added_voxels}"
        )

elapsed = time.time() - t0

# --------------------------------------------------
# 4) 保存 filled mask
# --------------------------------------------------
print("\n[4/5] 保存 filled mask")
tiff.imwrite(str(OUT_MASK_PATH), filled)

print(f"✅ saved filled mask : {OUT_MASK_PATH}")

# --------------------------------------------------
# 5) 保存统计信息
# --------------------------------------------------
print("\n[5/5] 保存 meta json")

num_instances_filled = int(len(np.unique(filled)) - (1 if 0 in np.unique(filled) else 0))

meta = {
    "raw_mask_path": str(RAW_MASK_PATH),
    "input_meta_path": str(META_PATH),
    "output_mask_path": str(OUT_MASK_PATH),
    "shape": list(map(int, masks.shape)),
    "dtype": str(masks.dtype),
    "num_instances_raw": num_instances_raw,
    "num_instances_filled": num_instances_filled,
    "num_changed_instances": int(num_changed_instances),
    "num_total_added_voxels": int(num_total_added_voxels),
    "num_slices_touched": int(num_slices_touched),
    "elapsed_sec": round(elapsed, 4),
    "method": "slice-wise 2D binary_fill_holes on each instance",
    "note": (
        "这是逐实例、逐z切片的2D填坑；"
        "正常情况下不会改变实例数量，只会填补实例内部封闭空洞。"
    ),
}

with open(OUT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ saved meta        : {OUT_META_PATH}")

print("\nDone.")
print("=" * 100)


# In[ ]:




