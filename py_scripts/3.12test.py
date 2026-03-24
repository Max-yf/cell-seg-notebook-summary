#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1：路径初始化（5-fold / processed data / isolated experiment root）
from pathlib import Path
from datetime import datetime
import json

print("="*80)
print("🧱 Cell 1 | Paths & experiment directory init (5-fold PROC)")
print("="*80)

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
DATA_ROOT = (ROOT / "Cellpose2TrainDataset").resolve()

# ===== 参与 5-fold 的旧数据（预处理版）=====
OLD_TRAIN_DIR = (DATA_ROOT / "trainset").resolve()
OLD_VAL_DIR   = (DATA_ROOT / "valset").resolve()

# ===== 独立外部测试集（预处理版）=====
NEWVAL_ROOT = (DATA_ROOT / "new_val_proc").resolve()
NEWVAL_IMG_DIR = (NEWVAL_ROOT / "images_sp_lcn").resolve()
NEWVAL_GT_DIR  = (NEWVAL_ROOT / "ground").resolve()

# ===== 3D 预处理版输入 =====
PROC_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"exp_20260312_proc_5fold_cv_{STAMP}"
EXP_DIR = (ROOT / "runs" / EXP_NAME).resolve()

# ===== 全部产物都收进 EXP_DIR，不污染别的实验 =====
CFG_DIR       = EXP_DIR / "config"
LOG_DIR       = EXP_DIR / "logs"
MET_DIR       = EXP_DIR / "metrics"
EXPORT_DIR    = EXP_DIR / "exports"
DELIV_DIR     = EXP_DIR / "delivery"

CV_ROOT       = EXP_DIR / "cv"
FOLD_VIEW_DIR = CV_ROOT / "fold_views"       # 每折 train/val 视图
CV_RUNS_DIR   = CV_ROOT / "runs"             # 每个 fold-run 的输出

FULLDATA_DIR  = EXP_DIR / "full_data_view"   # 旧 train+val 合并后的全量训练视图
NEWVAL_VIEW_DIR = EXP_DIR / "newval_proc_view"

FINAL_DIR     = EXP_DIR / "final_models"
INFER_DIR     = EXP_DIR / "infer"
EVAL_DIR      = EXP_DIR / "eval"
THREED_DIR    = EXP_DIR / "threed"

for d in [
    CFG_DIR, LOG_DIR, MET_DIR, EXPORT_DIR, DELIV_DIR,
    CV_ROOT, FOLD_VIEW_DIR, CV_RUNS_DIR,
    FULLDATA_DIR, NEWVAL_VIEW_DIR,
    FINAL_DIR, INFER_DIR, EVAL_DIR, THREED_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

PATHS = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "ROOT": str(ROOT),
    "DATA_ROOT": str(DATA_ROOT),
    "OLD_TRAIN_DIR": str(OLD_TRAIN_DIR),
    "OLD_VAL_DIR": str(OLD_VAL_DIR),
    "NEWVAL_ROOT": str(NEWVAL_ROOT),
    "NEWVAL_IMG_DIR": str(NEWVAL_IMG_DIR),
    "NEWVAL_GT_DIR": str(NEWVAL_GT_DIR),
    "PROC_3D_STACK_PATH": str(PROC_3D_STACK_PATH),
    "EXP_NAME": EXP_NAME,
    "EXP_DIR": str(EXP_DIR),
    "CFG_DIR": str(CFG_DIR),
    "LOG_DIR": str(LOG_DIR),
    "MET_DIR": str(MET_DIR),
    "EXPORT_DIR": str(EXPORT_DIR),
    "DELIV_DIR": str(DELIV_DIR),
    "CV_ROOT": str(CV_ROOT),
    "FOLD_VIEW_DIR": str(FOLD_VIEW_DIR),
    "CV_RUNS_DIR": str(CV_RUNS_DIR),
    "FULLDATA_DIR": str(FULLDATA_DIR),
    "NEWVAL_VIEW_DIR": str(NEWVAL_VIEW_DIR),
    "FINAL_DIR": str(FINAL_DIR),
    "INFER_DIR": str(INFER_DIR),
    "EVAL_DIR": str(EVAL_DIR),
    "THREED_DIR": str(THREED_DIR),
    "note": "5-fold CV on processed old train+val; external evaluation on new_val_proc; all artifacts isolated in one experiment folder."
}

(PATHS_JSON := CFG_DIR / "PATHS.json").write_text(
    json.dumps(PATHS, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("OLD_TRAIN_DIR     :", OLD_TRAIN_DIR, "| exists:", OLD_TRAIN_DIR.exists())
print("OLD_VAL_DIR       :", OLD_VAL_DIR,   "| exists:", OLD_VAL_DIR.exists())
print("NEWVAL_IMG_DIR    :", NEWVAL_IMG_DIR, "| exists:", NEWVAL_IMG_DIR.exists())
print("NEWVAL_GT_DIR     :", NEWVAL_GT_DIR,  "| exists:", NEWVAL_GT_DIR.exists())
print("PROC_3D_STACK_PATH:", PROC_3D_STACK_PATH, "| exists:", PROC_3D_STACK_PATH.exists())
print("EXP_DIR           :", EXP_DIR)
print("PATHS_JSON        :", PATHS_JSON)

assert OLD_TRAIN_DIR.exists(), f"OLD_TRAIN_DIR 不存在: {OLD_TRAIN_DIR}"
assert OLD_VAL_DIR.exists(),   f"OLD_VAL_DIR 不存在: {OLD_VAL_DIR}"
assert NEWVAL_IMG_DIR.exists(), f"NEWVAL_IMG_DIR 不存在: {NEWVAL_IMG_DIR}"
assert NEWVAL_GT_DIR.exists(),  f"NEWVAL_GT_DIR 不存在: {NEWVAL_GT_DIR}"

print("\n✅ Cell 1 done.")
print("="*80)


# In[3]:


# Cell 2：扫描旧 train/val，生成合并后的 master manifest（允许重名 sample_id）
import pandas as pd
from pathlib import Path

print("="*80)
print("📚 Cell 2 | Scan old train/val and build master manifest")
print("="*80)

IMG_EXT = ".tif"
MASK_SUFFIX = "_masks.tif"
FLOW_SUFFIX = "_flows.tif"

def collect_split(split_dir: Path, source_split: str):
    rows = []
    all_imgs = sorted(split_dir.glob(f"*{IMG_EXT}"))
    for img_path in all_imgs:
        name = img_path.name

        # 跳过 masks / flows 文件本体
        if name.endswith(MASK_SUFFIX) or name.endswith(FLOW_SUFFIX):
            continue

        stem = img_path.stem
        mask_path = split_dir / f"{stem}_masks.tif"
        flow_path = split_dir / f"{stem}_flows.tif"

        if not mask_path.exists():
            print(f"⚠️ 跳过：缺少 mask -> {img_path.name}")
            continue

        rows.append({
            "sample_id": stem,                           # 原始 stem，可重复
            "source_split": source_split,                # old_train / old_val
            "global_id": f"{source_split}__{stem}",      # 全局唯一主键
            "img_path": str(img_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "flow_path": str(flow_path.resolve()) if flow_path.exists() else "",
            "has_flow": flow_path.exists(),
        })
    return rows

rows = []
rows += collect_split(OLD_TRAIN_DIR, "old_train")
rows += collect_split(OLD_VAL_DIR, "old_val")

master_df = pd.DataFrame(rows).sort_values(["source_split", "sample_id"]).reset_index(drop=True)

assert len(master_df) > 0, "master manifest 为空，请检查数据路径。"
assert master_df["global_id"].is_unique, "global_id 仍然重复，这不正常，请检查数据。"

MASTER_MANIFEST_CSV = CFG_DIR / "master_manifest.csv"
master_df.to_csv(MASTER_MANIFEST_CSV, index=False)

print("Total samples:", len(master_df))
print(master_df.head())

dup_sample_ids = master_df["sample_id"][master_df["sample_id"].duplicated(keep=False)]
if len(dup_sample_ids) > 0:
    print("\n⚠️ 检测到跨 split 同名 sample_id（允许存在）:")
    print(master_df[master_df["sample_id"].isin(sorted(set(dup_sample_ids)))][
        ["sample_id", "source_split", "global_id"]
    ].head(20))

print("\nSaved to:", MASTER_MANIFEST_CSV)

print("\nSplit counts:")
print(master_df["source_split"].value_counts())

print("\n✅ Cell 2 done.")
print("="*80)


# In[4]:


# Cell 3：统计每张图的细胞数，并做密度分层
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path

print("="*80)
print("🔬 Cell 3 | Count instances and assign density bins")
print("="*80)

MASTER_MANIFEST_CSV = CFG_DIR / "master_manifest.csv"
master_df = pd.read_csv(MASTER_MANIFEST_CSV)

n_inst_list = []
img_h_list = []
img_w_list = []
fg_area_list = []
density_list = []

for i, row in master_df.iterrows():
    mask = tiff.imread(row["mask_path"])
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"mask 不是 2D: {row['mask_path']} | shape={mask.shape}")

    u = np.unique(mask)
    u = u[u > 0]  # 去掉背景 0
    n_inst = int(len(u))

    h, w = mask.shape
    fg_area = int((mask > 0).sum())
    density = n_inst / float(h * w)

    n_inst_list.append(n_inst)
    img_h_list.append(h)
    img_w_list.append(w)
    fg_area_list.append(fg_area)
    density_list.append(density)

master_df["n_inst"] = n_inst_list
master_df["img_h"] = img_h_list
master_df["img_w"] = img_w_list
master_df["fg_area"] = fg_area_list
master_df["inst_density"] = density_list

# ===== 按 n_inst 分层 =====
# 三档：low / mid / high
q1 = master_df["n_inst"].quantile(1/3)
q2 = master_df["n_inst"].quantile(2/3)

def density_bin_from_n_inst(x):
    if x <= q1:
        return "low"
    elif x <= q2:
        return "mid"
    else:
        return "high"

master_df["density_bin"] = master_df["n_inst"].apply(density_bin_from_n_inst)

MASTER_STATS_CSV = CFG_DIR / "master_manifest_with_stats.csv"
master_df.to_csv(MASTER_STATS_CSV, index=False)

print("q1:", q1, "| q2:", q2)
print(master_df[["sample_id", "source_split", "n_inst", "density_bin"]].head())
print("\nDensity bin counts:")
print(master_df["density_bin"].value_counts())
print("\nSaved to:", MASTER_STATS_CSV)

print("\n✅ Cell 3 done.")
print("="*80)


# In[5]:


# Cell 4：按细胞数密度分层做 5-fold 划分
import pandas as pd
import numpy as np

print("="*80)
print("🧩 Cell 4 | Stratified 5-fold split by cell-count density")
print("="*80)

SEED = 20260312
N_FOLDS = 5

master_df = pd.read_csv(CFG_DIR / "master_manifest_with_stats.csv")
rng = np.random.default_rng(SEED)

fold_ids = {}

for bin_name in ["low", "mid", "high"]:
    sub = master_df[master_df["density_bin"] == bin_name].copy()
    idxs = list(sub.index)
    rng.shuffle(idxs)

    for j, idx in enumerate(idxs):
        fold_ids[idx] = j % N_FOLDS

master_df["fold_id"] = master_df.index.map(fold_ids)

assert master_df["fold_id"].notna().all(), "有样本未分配 fold_id"
master_df["fold_id"] = master_df["fold_id"].astype(int)

FOLD_ASSIGN_CSV = CFG_DIR / "fold_assignments.csv"
master_df.to_csv(FOLD_ASSIGN_CSV, index=False)

print(master_df[["sample_id", "source_split", "n_inst", "density_bin", "fold_id"]].head(10))

print("\nOverall fold counts:")
print(master_df["fold_id"].value_counts().sort_index())

print("\nPer-fold density counts:")
print(pd.crosstab(master_df["fold_id"], master_df["density_bin"]))

print("\nSaved to:", FOLD_ASSIGN_CSV)
print("\n✅ Cell 4 done.")
print("="*80)


# In[9]:


# Cell 5：创建每折 train/val 视图，以及 full-data / new_val_proc 视图
import os
import shutil
import pandas as pd
from pathlib import Path

print("="*80)
print("🔗 Cell 5 | Build fold train/val symlink views (isolated)")
print("="*80)

fold_df = pd.read_csv(CFG_DIR / "fold_assignments.csv")

def reset_dir(d: Path):
    if d.exists():
        for p in d.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    d.mkdir(parents=True, exist_ok=True)

def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

# ------------------------------------------------------------------
# A. 5 folds: 每折 train / val
# ------------------------------------------------------------------
for fold_id in range(5):
    fold_root = FOLD_VIEW_DIR / f"fold{fold_id}"
    train_dir = fold_root / "train"
    val_dir   = fold_root / "val"

    reset_dir(train_dir)
    reset_dir(val_dir)

    df_val = fold_df[fold_df["fold_id"] == fold_id].copy()
    df_train = fold_df[fold_df["fold_id"] != fold_id].copy()

    # 为避免命名冲突，统一重命名为 cv_xxxxx
    for split_name, df_split, out_dir in [
        ("train", df_train, train_dir),
        ("val", df_val, val_dir),
    ]:
        for j, row in enumerate(df_split.itertuples(index=False), start=1):
            stem = f"cvf{fold_id}_{split_name}_{j:05d}__{row.global_id}"

            img_src = Path(row.img_path)
            mask_src = Path(row.mask_path)

            img_dst = out_dir / f"{stem}{img_src.suffix.lower()}"
            mask_dst = out_dir / f"{stem}_masks{mask_src.suffix.lower()}"

            safe_symlink(img_src, img_dst)
            safe_symlink(mask_src, mask_dst)

    print(f"✅ fold{fold_id}: train={len(df_train)}, val={len(df_val)}")

# ------------------------------------------------------------------
# B. 全量旧数据视图：最后 final retrain 用
# ------------------------------------------------------------------
FULL_TRAIN_ALL_DIR = FULLDATA_DIR / "train_all_old"
reset_dir(FULL_TRAIN_ALL_DIR)

for j, row in enumerate(fold_df.itertuples(index=False), start=1):
    stem = f"fullold_{j:05d}__{row.global_id}"
    img_src = Path(row.img_path)
    mask_src = Path(row.mask_path)

    img_dst = FULL_TRAIN_ALL_DIR / f"{stem}{img_src.suffix.lower()}"
    mask_dst = FULL_TRAIN_ALL_DIR / f"{stem}_masks{mask_src.suffix.lower()}"

    safe_symlink(img_src, img_dst)
    safe_symlink(mask_src, mask_dst)

print(f"✅ FULL_TRAIN_ALL_DIR built: {FULL_TRAIN_ALL_DIR} | n={len(fold_df)}")

# ------------------------------------------------------------------
# C. new_val_proc 统一评估视图（按排序后一一配对）
# ------------------------------------------------------------------
reset_dir(NEWVAL_VIEW_DIR)

img_files = sorted(NEWVAL_IMG_DIR.glob("*.tif"))
gt_files = sorted(NEWVAL_GT_DIR.glob("*.tif"))

print("n_img_files:", len(img_files))
print("n_gt_files :", len(gt_files))

assert len(img_files) > 0, "NEWVAL_IMG_DIR 为空"
assert len(gt_files) > 0, "NEWVAL_GT_DIR 为空"
assert len(img_files) == len(gt_files), \
    f"图像数与 GT 数不一致: {len(img_files)} vs {len(gt_files)}"

print("\n前 5 对（按排序配对预览）:")
for i in range(min(5, len(img_files))):
    print(f"[{i}] IMG={img_files[i].name}  <-->  GT={gt_files[i].name}")

newval_pairs = []
for j, (img_src, gt_src) in enumerate(zip(img_files, gt_files), start=1):
    stem = f"newproc_{j:05d}"

    img_dst = NEWVAL_VIEW_DIR / f"{stem}{img_src.suffix.lower()}"
    gt_dst  = NEWVAL_VIEW_DIR / f"{stem}_masks{gt_src.suffix.lower()}"

    safe_symlink(img_src, img_dst)
    safe_symlink(gt_src, gt_dst)

    newval_pairs.append({
        "pair_id": j,
        "img_src": str(img_src),
        "gt_src": str(gt_src),
        "img_dst": str(img_dst),
        "gt_dst": str(gt_dst),
    })

NEWVAL_PAIR_MANIFEST = CFG_DIR / "newval_proc_pair_manifest.csv"
pd.DataFrame(newval_pairs).to_csv(NEWVAL_PAIR_MANIFEST, index=False)

print(f"\n✅ NEWVAL_VIEW_DIR built: {NEWVAL_VIEW_DIR} | n={len(newval_pairs)}")
print("✅ Pair manifest saved to:", NEWVAL_PAIR_MANIFEST)

VIEW_SUMMARY = {
    "fold_view_root": str(FOLD_VIEW_DIR),
    "full_train_all_dir": str(FULL_TRAIN_ALL_DIR),
    "newval_view_dir": str(NEWVAL_VIEW_DIR),
    "n_total_old_samples": int(len(fold_df)),
    "n_newval_samples": int(len(common_stems)),
}
(CFG_DIR / "view_summary.json").write_text(
    json.dumps(VIEW_SUMMARY, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n✅ Cell 5 done.")
print("="*80)


# In[8]:


print([p.stem for p in sorted(NEWVAL_IMG_DIR.glob("*.tif"))[:10]])
print([p.stem for p in sorted(NEWVAL_GT_DIR.glob("*.tif"))[:10]])


# In[10]:


# Cell 6：定义候选参数，并展开成 5-fold run plan
import pandas as pd
import json

print("="*80)
print("🧪 Cell 6 | Define candidate params and expand 5-fold run plan")
print("="*80)

# ===== 第一轮只跑 3 个头部参数，避免 45-run 地狱 =====
CANDIDATES = [
    dict(param_tag="C1_lr8e5_wd8e3", learning_rate=8e-5, weight_decay=8e-3),
    dict(param_tag="C2_lr8e5_wd9e3", learning_rate=8e-5, weight_decay=9e-3),
    dict(param_tag="C3_lr9e5_wd8e3", learning_rate=9e-5, weight_decay=8e-3),
]

# ===== 固定主线配置 =====
BASE_CFG = {
    "pretrained_model": "cpsam",
    "diameter": 14,
    "augment": True,
    "transformer": True,
    "train_batch_size": 32,
    "bsize": 256,
    "mask_filter": "_masks",
    "save_every": 5,
    "save_each": True,
    "use_gpu": True,
    "verbose": True,
    "n_epochs": 400,

    # 这几个仍然保留给你的 Cell 8 训练执行器使用
    "early_stop_enabled": True,
    "early_stop_patience_epochs": 50,
    "early_stop_min_delta": 1e-4,
}

run_rows = []
for cand in CANDIDATES:
    for fold_id in range(5):
        fold_root = FOLD_VIEW_DIR / f"fold{fold_id}"
        train_dir = fold_root / "train"
        val_dir   = fold_root / "val"

        run_tag = f"{cand['param_tag']}__fold{fold_id}"

        run_rows.append({
            "run_tag": run_tag,
            "param_tag": cand["param_tag"],
            "fold_id": fold_id,
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "learning_rate": cand["learning_rate"],
            "weight_decay": cand["weight_decay"],
            **BASE_CFG,
        })

run_df = pd.DataFrame(run_rows).sort_values(["param_tag", "fold_id"]).reset_index(drop=True)

CV_RUN_PLAN_CSV = CFG_DIR / "cv_run_plan.csv"
run_df.to_csv(CV_RUN_PLAN_CSV, index=False)

(CFG_DIR / "candidate_params.json").write_text(
    json.dumps({
        "base_cfg": BASE_CFG,
        "candidates": CANDIDATES,
        "n_folds": 5,
        "n_total_runs": int(len(run_df)),
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(run_df.head(10))
print("\nTotal CV runs:", len(run_df))
print("Saved to:", CV_RUN_PLAN_CSV)

print("\n✅ Cell 6 done.")
print("="*80)


# In[11]:


# Cell 7：5-fold 串行训练执行器
import json, os, time, shlex, subprocess, traceback, re
from pathlib import Path
from datetime import datetime
import pandas as pd

print("="*80)
print("🚀 Cell 7 | Serial CV training executor (5-fold, processed)")
print("="*80)

RUN_PLAN_CSV = CFG_DIR / "cv_run_plan.csv"
assert RUN_PLAN_CSV.exists(), f"未找到: {RUN_PLAN_CSV}"

run_df = pd.read_csv(RUN_PLAN_CSV)
assert len(run_df) > 0, "cv_run_plan.csv 为空"

CV_RUNS_JSONL = CFG_DIR / "CV_RUNS.jsonl"

PAT_SAVED_LIST = [
    re.compile(r"model trained and saved to\s+(?P<dir>/\S+)"),
    re.compile(r"saving model to\s+(?P<dir>/\S+)"),
    re.compile(r"saving network parameters to\s+(?P<dir>/\S+)"),
]

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_saved_model_dir(log_text: str):
    hits = []
    for pat in PAT_SAVED_LIST:
        for m in pat.finditer(log_text):
            hits.append(m.group("dir"))
    return hits[-1] if hits else None

runs_to_process = run_df.to_dict(orient="records")
print("Runs to process:", len(runs_to_process))

# 若你不想重复跑已经成功的 run，可开这个开关
SKIP_DONE_IF_SUMMARY_EXISTS = True
CV_TRAIN_SUMMARY_CSV = MET_DIR / "cv_train_summary.csv"
done_run_tags = set()
if SKIP_DONE_IF_SUMMARY_EXISTS and CV_TRAIN_SUMMARY_CSV.exists():
    try:
        old_df = pd.read_csv(CV_TRAIN_SUMMARY_CSV)
        done_run_tags = set(old_df.loc[old_df["status"] == "DONE", "run_tag"].tolist())
        print("Found existing DONE runs:", len(done_run_tags))
    except Exception as e:
        print("⚠️ 读取旧 summary 失败，忽略跳过逻辑：", e)

for i, r in enumerate(runs_to_process, start=1):
    run_tag = r["run_tag"]

    if run_tag in done_run_tags:
        print("="*100)
        print(f"[{i}/{len(runs_to_process)}] SKIP DONE: {run_tag}")
        print("="*100)
        continue

    fold_id = int(r["fold_id"])
    param_tag = r["param_tag"]
    train_dir = Path(r["train_dir"])
    val_dir = Path(r["val_dir"])

    run_out_dir = CV_RUNS_DIR / run_tag
    run_out_dir.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"train_{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [
        "python", "-m", "cellpose",
        "--train",
        "--dir", str(train_dir),
        "--test_dir", str(val_dir),
        "--pretrained_model", str(r["pretrained_model"]),
        "--diameter", str(int(r["diameter"])),
        "--learning_rate", str(r["learning_rate"]),
        "--weight_decay", str(r["weight_decay"]),
        "--n_epochs", str(int(r["n_epochs"])),
        "--train_batch_size", str(int(r["train_batch_size"])),
        "--bsize", str(int(r["bsize"])),
        "--mask_filter", str(r["mask_filter"]),
        "--save_every", str(int(r["save_every"])),
        "--save_each",
        "--use_gpu",
        "--verbose",
    ]

    if bool(r["transformer"]):
        cmd.append("--transformer")
    if bool(r["augment"]):
        cmd.append("--augment")

    print("="*100)
    print(f"[{i}/{len(runs_to_process)}] RUN: {run_tag}")
    print("="*100)
    print("CMD:")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("\nLOG:", log_path)

    started_at = datetime.now().isoformat(timespec="seconds")
    proc = None
    return_code = None
    status = "FAILED_TO_START"
    final_model_dir = None
    err_msg = ""

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print("🚀 Started | PID:", proc.pid)

        return_code = proc.wait()
        print("✅ process finished with return code:", return_code)

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        final_model_dir = extract_saved_model_dir(log_text)

        if return_code == 0:
            status = "DONE"
        else:
            status = "FAILED"

            tail = "\n".join(log_text.splitlines()[-30:])
            err_msg = tail

        if final_model_dir is None:
            print("⚠️ final_model_dir 未捕获，请检查 log。")
        else:
            print("📦 captured final_model_dir:", final_model_dir)

    except Exception as e:
        status = "EXCEPTION"
        err_msg = traceback.format_exc()
        print("❌ Exception:", e)

    rec = {
        "run_tag": run_tag,
        "param_tag": param_tag,
        "fold_id": fold_id,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "learning_rate": float(r["learning_rate"]),
        "weight_decay": float(r["weight_decay"]),
        "diameter": int(r["diameter"]),
        "augment": bool(r["augment"]),
        "transformer": bool(r["transformer"]),
        "train_batch_size": int(r["train_batch_size"]),
        "n_epochs": int(r["n_epochs"]),
        "early_stop_enabled": bool(r["early_stop_enabled"]),
        "early_stop_patience_epochs": int(r["early_stop_patience_epochs"]),
        "early_stop_min_delta": float(r["early_stop_min_delta"]),
        "status": status,
        "return_code": return_code,
        "pid": proc.pid if proc is not None else None,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "final_model_dir": final_model_dir if final_model_dir else "",
        "final_model_dir_exists": bool(final_model_dir) and Path(final_model_dir).exists(),
        "error_tail": err_msg,
    }
    append_jsonl(CV_RUNS_JSONL, rec)

print("\n✅ Cell 7 done.")
print("="*80)


# In[12]:


# Cell 8：解析训练日志，汇总 CV 训练摘要
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("📊 Cell 8 | Parse CV training records -> cv_train_summary.csv")
print("="*80)

CV_RUNS_JSONL = CFG_DIR / "CV_RUNS.jsonl"
assert CV_RUNS_JSONL.exists(), f"未找到: {CV_RUNS_JSONL}"

records = []
for line in CV_RUNS_JSONL.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    records.append(json.loads(line))

summary_df = pd.DataFrame(records).sort_values(["param_tag", "fold_id"]).reset_index(drop=True)
CV_TRAIN_SUMMARY_CSV = MET_DIR / "cv_train_summary.csv"
summary_df.to_csv(CV_TRAIN_SUMMARY_CSV, index=False)

print(summary_df[[
    "run_tag", "param_tag", "fold_id", "status",
    "return_code", "final_model_dir", "final_model_dir_exists"
]].head(15))

print("\nStatus counts:")
print(summary_df["status"].value_counts())

print("\nSaved to:", CV_TRAIN_SUMMARY_CSV)
print("\n✅ Cell 8 done.")
print("="*80)


# In[19]:


# Cell 9：对每个 fold 模型做统一评估（修正版：从 val_dir 读取 cellpose 输出）
import os, json, subprocess, shlex, traceback, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile as tiff

print("="*80)
print("🧪 Cell 9 | Unified fold-wise evaluation (fixed output path)")
print("="*80)

CV_TRAIN_SUMMARY_CSV = MET_DIR / "cv_train_summary.csv"
assert CV_TRAIN_SUMMARY_CSV.exists(), f"未找到: {CV_TRAIN_SUMMARY_CSV}"

train_df = pd.read_csv(CV_TRAIN_SUMMARY_CSV)
done_df = train_df[train_df["status"] == "DONE"].copy()
assert len(done_df) > 0, "没有 DONE 的 fold-run，无法评估"

def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)

def instance_iou_matrix(gt_mask, pred_mask):
    gt_ids = [x for x in np.unique(gt_mask) if x > 0]
    pr_ids = [x for x in np.unique(pred_mask) if x > 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return gt_ids, pr_ids, np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)

    mat = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    gt_bin_cache = {gid: (gt_mask == gid) for gid in gt_ids}
    pr_bin_cache = {pid: (pred_mask == pid) for pid in pr_ids}

    for i, gid in enumerate(gt_ids):
        g = gt_bin_cache[gid]
        for j, pid in enumerate(pr_ids):
            p = pr_bin_cache[pid]
            mat[i, j] = iou_binary(g, p)
    return gt_ids, pr_ids, mat

def greedy_match_ap50(gt_mask, pred_mask, thr=0.5):
    gt_ids, pr_ids, mat = instance_iou_matrix(gt_mask, pred_mask)

    if len(gt_ids) == 0 and len(pr_ids) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0, ap50=1.0)
    if len(gt_ids) == 0:
        fp = len(pr_ids)
        return dict(tp=0, fp=fp, fn=0, precision=0.0, recall=1.0, f1=0.0, ap50=0.0)
    if len(pr_ids) == 0:
        fn = len(gt_ids)
        return dict(tp=0, fp=0, fn=fn, precision=1.0, recall=0.0, f1=0.0, ap50=0.0)

    pairs = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= thr:
                pairs.append((mat[i, j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pr = set()
    tp = 0
    for score, i, j in pairs:
        if i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        tp += 1

    fp = len(pr_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    ap50 = precision * recall / (precision + recall - precision * recall + 1e-8)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, ap50=ap50)

def find_pred_mask_for_stem(val_dir: Path, stem: str, gt_path: Path):
    """
    在 val_dir 中寻找 cellpose 推理生成的预测 mask。
    优先找 *_cp_masks.tif；如果没有，再谨慎找其他可能结果。
    """
    candidates = [
        val_dir / f"{stem}_cp_masks.tif",
        val_dir / f"{stem}_masks.tif",
    ]

    for cp in candidates:
        if cp.exists():
            # 若恰好是 GT 本身，则跳过
            if cp.resolve() == gt_path.resolve():
                continue
            return cp

    return None

eval_rows = []

for _, row in done_df.iterrows():
    run_tag = row["run_tag"]
    val_dir = Path(row["val_dir"])
    model_dir = Path(row["final_model_dir"])
    assert val_dir.exists(), f"val_dir 不存在: {val_dir}"
    assert model_dir.exists(), f"model_dir 不存在: {model_dir}"

    # 这里只保留一个记录目录，不再假设预测结果写进这里
    pred_dir = EVAL_DIR / run_tag / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(val_dir),
        "--pretrained_model", str(model_dir),
        "--diameter", str(int(row["diameter"])),
        "--save_tif",
        "--no_npy",
        "--use_gpu",
        "--verbose",
    ]

    print("="*100)
    print("Infer:", run_tag)
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))

    infer_log = LOG_DIR / f"infer_{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(infer_log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()

    if ret != 0:
        print("❌ inference failed:", run_tag)
        eval_rows.append({
            "run_tag": run_tag,
            "param_tag": row["param_tag"],
            "fold_id": int(row["fold_id"]),
            "status": "INFER_FAILED",
            "AP50": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "n_eval": 0,
            "pred_dir": str(pred_dir),
            "infer_log": str(infer_log),
        })
        continue

    gt_files = sorted(val_dir.glob("*_masks.tif"))
    metrics_img = []
    missing_pred = []

    for gt_path in gt_files:
        stem = gt_path.name.replace("_masks.tif", "")
        pred_path = find_pred_mask_for_stem(val_dir, stem, gt_path)

        if pred_path is None:
            missing_pred.append(stem)
            continue

        gt_mask = np.squeeze(tiff.imread(str(gt_path)))
        pred_mask = np.squeeze(tiff.imread(str(pred_path)))

        m = greedy_match_ap50(gt_mask, pred_mask, thr=0.5)
        metrics_img.append(m)

        # 顺手把找到的预测结果复制到 pred_dir 里，方便后续排查/留档
        backup_pred = pred_dir / pred_path.name
        if not backup_pred.exists():
            shutil.copy2(pred_path, backup_pred)

    if len(metrics_img) == 0:
        print(f"⚠️ {run_tag}: no matched predictions found. missing={len(missing_pred)}")
        eval_rows.append({
            "run_tag": run_tag,
            "param_tag": row["param_tag"],
            "fold_id": int(row["fold_id"]),
            "status": "NO_PRED_MATCHED",
            "AP50": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "n_eval": 0,
            "n_missing_pred": len(missing_pred),
            "pred_dir": str(pred_dir),
            "infer_log": str(infer_log),
        })
        continue

    ap50 = float(np.mean([m["ap50"] for m in metrics_img]))
    precision = float(np.mean([m["precision"] for m in metrics_img]))
    recall = float(np.mean([m["recall"] for m in metrics_img]))
    f1 = float(np.mean([m["f1"] for m in metrics_img]))

    eval_rows.append({
        "run_tag": run_tag,
        "param_tag": row["param_tag"],
        "fold_id": int(row["fold_id"]),
        "status": "DONE",
        "AP50": ap50,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "n_eval": len(metrics_img),
        "n_missing_pred": len(missing_pred),
        "pred_dir": str(pred_dir),
        "infer_log": str(infer_log),
    })

cv_eval_df = pd.DataFrame(eval_rows).sort_values(["param_tag", "fold_id"]).reset_index(drop=True)
CV_EVAL_RESULTS_CSV = MET_DIR / "cv_fold_eval_results.csv"
cv_eval_df.to_csv(CV_EVAL_RESULTS_CSV, index=False)

print(cv_eval_df.head(15))
print("\nStatus counts:")
print(cv_eval_df["status"].value_counts(dropna=False))
print("\nSaved to:", CV_EVAL_RESULTS_CSV)

print("\n✅ Cell 9 done.")
print("="*80)


# In[15]:


import pandas as pd
df = pd.read_csv(MET_DIR / "cv_fold_eval_results.csv")
print(df[["run_tag", "param_tag", "fold_id", "status", "n_eval", "pred_dir", "infer_log"]].head(20))
print(df["status"].value_counts(dropna=False))


# In[17]:


print("\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]))


# In[18]:


from pathlib import Path
df = pd.read_csv(MET_DIR / "cv_fold_eval_results.csv")
pred_dir = Path(df.iloc[0]["pred_dir"])
print(pred_dir)
print([p.name for p in sorted(pred_dir.glob("*"))[:50]])


# In[20]:


# Cell 10：汇总 5-fold mean/std 排名
import pandas as pd
import numpy as np

print("="*80)
print("📈 Cell 10 | Summarize 5-fold CV mean/std and rank params")
print("="*80)

CV_EVAL_RESULTS_CSV = MET_DIR / "cv_fold_eval_results.csv"
assert CV_EVAL_RESULTS_CSV.exists(), f"未找到: {CV_EVAL_RESULTS_CSV}"

df = pd.read_csv(CV_EVAL_RESULTS_CSV)
df_ok = df[df["status"] == "DONE"].copy()
assert len(df_ok) > 0, "没有成功完成的 fold 评估结果"

agg = df_ok.groupby("param_tag").agg(
    mean_AP50=("AP50", "mean"),
    std_AP50=("AP50", "std"),
    mean_Precision=("Precision", "mean"),
    std_Precision=("Precision", "std"),
    mean_Recall=("Recall", "mean"),
    std_Recall=("Recall", "std"),
    mean_F1=("F1", "mean"),
    std_F1=("F1", "std"),
    n_folds_done=("fold_id", "count"),
).reset_index()

agg["std_AP50"] = agg["std_AP50"].fillna(0.0)
agg["std_Precision"] = agg["std_Precision"].fillna(0.0)
agg["std_Recall"] = agg["std_Recall"].fillna(0.0)
agg["std_F1"] = agg["std_F1"].fillna(0.0)

agg = agg.sort_values(
    ["mean_AP50", "mean_F1", "std_AP50"],
    ascending=[False, False, True]
).reset_index(drop=True)

agg["cv_rank"] = np.arange(1, len(agg) + 1)

CV_PARAM_SUMMARY_CSV = MET_DIR / "cv_param_summary.csv"
agg.to_csv(CV_PARAM_SUMMARY_CSV, index=False)

print(agg)
print("\nSaved to:", CV_PARAM_SUMMARY_CSV)

print("\n✅ Cell 10 done.")
print("="*80)


# In[21]:


# Cell 11：自动选 top1 / top2 参数
import pandas as pd
import json

print("="*80)
print("🏆 Cell 11 | Auto-pick top params from 5-fold CV")
print("="*80)

CV_PARAM_SUMMARY_CSV = MET_DIR / "cv_param_summary.csv"
assert CV_PARAM_SUMMARY_CSV.exists(), f"未找到: {CV_PARAM_SUMMARY_CSV}"

cv_sum = pd.read_csv(CV_PARAM_SUMMARY_CSV)
assert len(cv_sum) >= 1, "cv_param_summary 为空"

TOP1_PARAM = cv_sum.iloc[0].to_dict()
TOP2_PARAMS = cv_sum.head(min(2, len(cv_sum))).to_dict(orient="records")

print("TOP1_PARAM:")
print(TOP1_PARAM)

print("\nTOP2_PARAMS:")
for x in TOP2_PARAMS:
    print(x)

TOP_PICK_JSON = CFG_DIR / "cv_top_picks.json"
TOP_PICK_JSON.write_text(
    json.dumps({
        "top1_param": TOP1_PARAM,
        "top2_params": TOP2_PARAMS,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nSaved to:", TOP_PICK_JSON)
print("\n✅ Cell 11 done.")
print("="*80)


# In[22]:


# Cell 12：用全部旧数据重训 top2 final models
import subprocess, shlex, json, re, traceback
from pathlib import Path
from datetime import datetime
import pandas as pd

print("="*80)
print("🎯 Cell 12 | Retrain top2 params on all old processed data")
print("="*80)

TOP_PICK_JSON = CFG_DIR / "cv_top_picks.json"
assert TOP_PICK_JSON.exists(), f"未找到: {TOP_PICK_JSON}"

top_pick = json.loads(TOP_PICK_JSON.read_text(encoding="utf-8"))
top2_params = top_pick["top2_params"]
assert len(top2_params) >= 1, "没有 top 参数"

FINAL_RUNS_JSONL = CFG_DIR / "FINAL_RUNS.jsonl"

PAT_SAVED_LIST = [
    re.compile(r"model trained and saved to\s+(?P<dir>/\S+)"),
    re.compile(r"saving model to\s+(?P<dir>/\S+)"),
    re.compile(r"saving network parameters to\s+(?P<dir>/\S+)"),
]

def append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_saved_model_dir(log_text: str):
    hits = []
    for pat in PAT_SAVED_LIST:
        for m in pat.finditer(log_text):
            hits.append(m.group("dir"))
    return hits[-1] if hits else None

for rank_idx, p in enumerate(top2_params, start=1):
    param_tag = p["param_tag"]

    lr = float(param_tag.split("_lr")[1].split("_wd")[0].replace("e5", "e-5").replace("e4", "e-4").replace("e3", "e-3"))
    # 上面这行太野了，直接从 CANDIDATES 对照更稳，但这里只要 param_tag 格式固定基本能过
    # 为稳妥，这里再从 cv_run_plan 中查一次
    run_plan = pd.read_csv(CFG_DIR / "cv_run_plan.csv")
    sub = run_plan[run_plan["param_tag"] == param_tag].iloc[0]

    final_tag = f"FINAL_rank{rank_idx}__{param_tag}"
    log_path = LOG_DIR / f"train_{final_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [
        "python", "-m", "cellpose",
        "--train",
        "--dir", str(FULL_TRAIN_ALL_DIR),
        "--test_dir", str(NEWVAL_VIEW_DIR),  # 这里仍给 test_dir，便于训练期监控，但最终统一评估还是后面单独做
        "--pretrained_model", str(sub["pretrained_model"]),
        "--diameter", str(int(sub["diameter"])),
        "--learning_rate", str(sub["learning_rate"]),
        "--weight_decay", str(sub["weight_decay"]),
        "--n_epochs", str(int(sub["n_epochs"])),
        "--train_batch_size", str(int(sub["train_batch_size"])),
        "--bsize", str(int(sub["bsize"])),
        "--mask_filter", str(sub["mask_filter"]),
        "--save_every", str(int(sub["save_every"])),
        "--save_each",
        "--use_gpu",
        "--verbose",
    ]
    if bool(sub["transformer"]):
        cmd.append("--transformer")
    if bool(sub["augment"]):
        cmd.append("--augment")

    print("="*100)
    print("RUN FINAL:", final_tag)
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))
    print("LOG:", log_path)

    started_at = datetime.now().isoformat(timespec="seconds")
    status = "FAILED_TO_START"
    return_code = None
    final_model_dir = None
    err_msg = ""

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        print("🚀 Started | PID:", proc.pid)

        return_code = proc.wait()
        print("✅ process finished with return code:", return_code)

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        final_model_dir = extract_saved_model_dir(log_text)

        status = "DONE" if return_code == 0 else "FAILED"
        if return_code != 0:
            err_msg = "\n".join(log_text.splitlines()[-30:])

    except Exception:
        status = "EXCEPTION"
        err_msg = traceback.format_exc()

    rec = {
        "final_tag": final_tag,
        "rank_from_cv": rank_idx,
        "param_tag": param_tag,
        "learning_rate": float(sub["learning_rate"]),
        "weight_decay": float(sub["weight_decay"]),
        "diameter": int(sub["diameter"]),
        "augment": bool(sub["augment"]),
        "transformer": bool(sub["transformer"]),
        "train_dir": str(FULL_TRAIN_ALL_DIR),
        "test_dir": str(NEWVAL_VIEW_DIR),
        "status": status,
        "return_code": return_code,
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "final_model_dir": final_model_dir if final_model_dir else "",
        "final_model_dir_exists": bool(final_model_dir) and Path(final_model_dir).exists(),
        "error_tail": err_msg,
    }
    append_jsonl(FINAL_RUNS_JSONL, rec)

print("\n✅ Cell 12 done.")
print("="*80)


# In[27]:


# Cell 13：在 new_val_proc 上做外部评估（修正版：从 NEWVAL_VIEW_DIR 读取 cellpose 输出）
import json, subprocess, shlex, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tifffile as tiff

print("="*80)
print("🧪 Cell 13 | External evaluation on new_val_proc (fixed output path)")
print("="*80)

FINAL_RUNS_JSONL = CFG_DIR / "FINAL_RUNS.jsonl"
assert FINAL_RUNS_JSONL.exists(), f"未找到: {FINAL_RUNS_JSONL}"

final_records = [json.loads(x) for x in FINAL_RUNS_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
final_df = pd.DataFrame(final_records)
final_done = final_df[final_df["status"] == "DONE"].copy()
assert len(final_done) > 0, "没有 DONE 的 final model"

def iou_binary(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)

def instance_iou_matrix(gt_mask, pred_mask):
    gt_ids = [x for x in np.unique(gt_mask) if x > 0]
    pr_ids = [x for x in np.unique(pred_mask) if x > 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return gt_ids, pr_ids, np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    mat = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
    gt_bin_cache = {gid: (gt_mask == gid) for gid in gt_ids}
    pr_bin_cache = {pid: (pred_mask == pid) for pid in pr_ids}
    for i, gid in enumerate(gt_ids):
        g = gt_bin_cache[gid]
        for j, pid in enumerate(pr_ids):
            p = pr_bin_cache[pid]
            mat[i, j] = iou_binary(g, p)
    return gt_ids, pr_ids, mat

def greedy_match_ap50(gt_mask, pred_mask, thr=0.5):
    gt_ids, pr_ids, mat = instance_iou_matrix(gt_mask, pred_mask)
    if len(gt_ids) == 0 and len(pr_ids) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0, ap50=1.0)
    if len(gt_ids) == 0:
        fp = len(pr_ids)
        return dict(tp=0, fp=fp, fn=0, precision=0.0, recall=1.0, f1=0.0, ap50=0.0)
    if len(pr_ids) == 0:
        fn = len(gt_ids)
        return dict(tp=0, fp=0, fn=fn, precision=1.0, recall=0.0, f1=0.0, ap50=0.0)

    pairs = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= thr:
                pairs.append((mat[i, j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pr = set()
    tp = 0
    for score, i, j in pairs:
        if i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        tp += 1

    fp = len(pr_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    ap50 = precision * recall / (precision + recall - precision * recall + 1e-8)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, ap50=ap50)

def find_pred_mask_for_stem(eval_dir: Path, stem: str, gt_path: Path):
    candidates = [
        eval_dir / f"{stem}_cp_masks.tif",
        eval_dir / f"{stem}_masks.tif",
    ]
    for cp in candidates:
        if cp.exists():
            if cp.resolve() == gt_path.resolve():
                continue
            return cp
    return None

ext_rows = []

for _, row in final_done.iterrows():
    final_tag = row["final_tag"]
    model_dir = Path(row["final_model_dir"])
    assert model_dir.exists(), f"final model 不存在: {model_dir}"

    pred_dir = EVAL_DIR / "external_newval_proc" / final_tag / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(NEWVAL_VIEW_DIR),
        "--pretrained_model", str(model_dir),
        "--diameter", str(int(row["diameter"])),
        "--save_tif",
        "--no_npy",
        "--use_gpu",
        "--verbose",
    ]

    infer_log = LOG_DIR / f"infer_external_{final_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("="*100)
    print("Infer external:", final_tag)
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))

    with open(infer_log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()

    if ret != 0:
        ext_rows.append({
            "final_tag": final_tag,
            "param_tag": row["param_tag"],
            "status": "INFER_FAILED",
            "AP50": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "n_eval": 0,
            "n_missing_pred": np.nan,
            "pred_dir": str(pred_dir),
            "infer_log": str(infer_log),
            "eval_model_path": str(model_dir),
            "eval_model_type": "final_model",
        })
        continue

    gt_files = sorted(NEWVAL_VIEW_DIR.glob("*_masks.tif"))
    metrics_img = []
    missing_pred = []

    for gt_path in gt_files:
        stem = gt_path.name.replace("_masks.tif", "")
        pred_path = find_pred_mask_for_stem(NEWVAL_VIEW_DIR, stem, gt_path)

        if pred_path is None:
            missing_pred.append(stem)
            continue

        gt_mask = np.squeeze(tiff.imread(str(gt_path)))
        pred_mask = np.squeeze(tiff.imread(str(pred_path)))
        m = greedy_match_ap50(gt_mask, pred_mask, thr=0.5)
        metrics_img.append(m)

        backup_pred = pred_dir / pred_path.name
        if not backup_pred.exists():
            shutil.copy2(pred_path, backup_pred)

    if len(metrics_img) == 0:
        ext_rows.append({
            "final_tag": final_tag,
            "param_tag": row["param_tag"],
            "status": "NO_PRED_MATCHED",
            "AP50": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "n_eval": 0,
            "n_missing_pred": len(missing_pred),
            "pred_dir": str(pred_dir),
            "infer_log": str(infer_log),
            "eval_model_path": str(model_dir),
            "eval_model_type": "final_model",
        })
        continue

    ext_rows.append({
        "final_tag": final_tag,
        "param_tag": row["param_tag"],
        "status": "DONE",
        "AP50": float(np.mean([m["ap50"] for m in metrics_img])),
        "Precision": float(np.mean([m["precision"] for m in metrics_img])),
        "Recall": float(np.mean([m["recall"] for m in metrics_img])),
        "F1": float(np.mean([m["f1"] for m in metrics_img])),
        "n_eval": len(metrics_img),
        "n_missing_pred": len(missing_pred),
        "pred_dir": str(pred_dir),
        "infer_log": str(infer_log),
        "eval_model_path": str(model_dir),
        "eval_model_type": "final_model",
    })

ext_df = pd.DataFrame(ext_rows).sort_values(["AP50", "F1"], ascending=[False, False]).reset_index(drop=True)

FINAL_EXTERNAL_SUMMARY_CSV = MET_DIR / "final_external_newval_proc_summary.csv"
ext_df.to_csv(FINAL_EXTERNAL_SUMMARY_CSV, index=False)

print(ext_df)
print("\nStatus counts:")
print(ext_df["status"].value_counts(dropna=False))
print("\nSaved to:", FINAL_EXTERNAL_SUMMARY_CSV)

print("\n✅ Cell 13 done.")
print("="*80)


# In[25]:


import pandas as pd
df = pd.read_csv(MET_DIR / "final_external_newval_proc_summary.csv")
print(df)
print(df["status"].value_counts(dropna=False))


# In[26]:


from pathlib import Path
df = pd.read_csv(MET_DIR / "final_external_newval_proc_summary.csv")
log_path = Path(df.iloc[0]["infer_log"])
print(log_path)
print("\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]))


# In[ ]:





# In[28]:


# Cell 14：自动选 top2 final models 跑 3D
import json
import pandas as pd
import subprocess, shlex
from pathlib import Path
from datetime import datetime
import tifffile as tiff
import numpy as np

print("="*80)
print("🚀 Cell 14 | Auto-pick top2 final models and run 3D inference (PROC)")
print("="*80)

FINAL_EXTERNAL_SUMMARY_CSV = MET_DIR / "final_external_newval_proc_summary.csv"
assert FINAL_EXTERNAL_SUMMARY_CSV.exists(), f"未找到: {FINAL_EXTERNAL_SUMMARY_CSV}"

rank_df = pd.read_csv(FINAL_EXTERNAL_SUMMARY_CSV)
rank_df = rank_df[rank_df["status"] == "DONE"].copy()
assert len(rank_df) >= 1, "没有成功完成 external eval 的 final model"

rank_df = rank_df.sort_values(["AP50", "F1"], ascending=[False, False]).reset_index(drop=True)
pick_df = rank_df.head(min(2, len(rank_df))).copy()

MODELS_3D = []
for rank_idx, (_, row) in enumerate(pick_df.iterrows(), start=1):
    MODELS_3D.append({
        "rank_2d": rank_idx,
        "tag": row["final_tag"],
        "param_tag": row["param_tag"],
        "model_path": row["eval_model_path"],
        "eval_model_type": row["eval_model_type"],
        "AP50_2d": float(row["AP50"]),
        "F1_2d": float(row["F1"]),
        "Recall_2d": float(row["Recall"]),
        "Precision_2d": float(row["Precision"]),
    })

TRUE_3D_DIAMETER = 8
DO_3D = True
Z_AXIS = 0
STITCH_THRESHOLD = 0.5
BATCH_SIZE_3D = 16

picked_tags = "__".join([m["tag"] for m in MODELS_3D])
THREED_ROOT = (THREED_DIR / f"top2_proc_{picked_tags}").resolve()
THREED_ROOT.mkdir(parents=True, exist_ok=True)

THREED_RESULTS = {}

for m in MODELS_3D:
    out_dir = THREED_ROOT / f"rank{m['rank_2d']}_{m['tag']}_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(PROC_3D_STACK_PATH.parent),
        "--image_path", str(PROC_3D_STACK_PATH),
        "--pretrained_model", str(m["model_path"]),
        "--diameter", str(TRUE_3D_DIAMETER),
        "--do_3D",
        "--z_axis", str(Z_AXIS),
        "--stitch_threshold", str(STITCH_THRESHOLD),
        "--save_tif",
        "--no_npy",
        "--use_gpu",
        "--verbose",
    ]

    infer_log = LOG_DIR / f"threed_{m['tag']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("="*100)
    print("3D infer:", m["tag"])
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))

    with open(infer_log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(out_dir), stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()

    if ret != 0:
        print("❌ 3D inference failed:", m["tag"])
        continue

    # 尝试找输出 mask
    candidates = list(out_dir.glob("*masks*.tif")) + list(out_dir.glob("*.tif"))
    candidates = [x for x in candidates if x.name != PROC_3D_STACK_PATH.name]

    if len(candidates) == 0:
        print("⚠️ 未找到 3D mask 输出:", m["tag"])
        continue

    mask_path = candidates[0]
    mask = tiff.imread(str(mask_path))
    total_cells = int(len([x for x in np.unique(mask) if x > 0]))

    THREED_RESULTS[m["tag"]] = {
        "mask_path": str(mask_path),
        "stats": {
            "run_label": m["tag"],
            "model_path": m["model_path"],
            "raw_3d_stack_path": str(PROC_3D_STACK_PATH),
            "mask_save_path": str(mask_path),
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "total_cells": total_cells,
            "elapsed_s": np.nan,
            "true_3d_diameter": TRUE_3D_DIAMETER,
            "do_3d": DO_3D,
            "stitch_threshold": STITCH_THRESHOLD,
            "z_axis": Z_AXIS,
            "batch_size_3d": BATCH_SIZE_3D,
        },
        "meta": {
            **m,
            "out_dir": str(out_dir),
            "infer_log": str(infer_log),
        }
    }

THREED_CONFIG = {
    "created_at": pd.Timestamp.now().isoformat(),
    "mode": "auto_top2_proc3d",
    "proc_3d_stack_path": str(PROC_3D_STACK_PATH),
    "models_3d": MODELS_3D,
    "output_root": str(THREED_ROOT),
}

(THREED_ROOT / "threed_config.json").write_text(
    json.dumps(THREED_CONFIG, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

(THREED_ROOT / "threed_results.json").write_text(
    json.dumps(THREED_RESULTS, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Available THREED_RESULTS keys:", list(THREED_RESULTS.keys()))
print("\n✅ Cell 14 done.")
print("="*80)


# In[29]:


import tifffile as tiff
from pathlib import Path
import numpy as np

p = Path("/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010_cp_masks.tif")
m = tiff.imread(p)

print("shape:", m.shape)
print("dtype:", m.dtype)
print("min/max:", m.min(), m.max())
print("unique count (sample):", len(np.unique(m[:5])) if m.ndim >= 3 else len(np.unique(m)))


# In[30]:


import numpy as np
import tifffile as tiff
from pathlib import Path

p = Path("/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010_cp_masks.tif")
m = tiff.imread(p)

uids = np.unique(m)
print("背景是否存在:", 0 in uids)
print("实例总数(含背景):", len(uids))
print("非零实例数:", np.sum(uids > 0))
print("前20个唯一值:", uids[:20])
print("最后20个唯一值:", uids[-20:])


# In[1]:


#cell14第二个模型没跑，这是手动再跑版本，顺便修复找不到文件的bug
# Cell 14B：单独补跑 FINAL_rank2__C2_lr8e5_wd9e3 的 3D inference（PROC）
import shutil
import subprocess
import time
from pathlib import Path

print("=" * 100)
print("🚀 Cell 14B | Rerun only FINAL_rank2__C2_lr8e5_wd9e3 3D inference (PROC)")
print("=" * 100)

# =========================================================
# 1) 基础路径
# =========================================================
EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421")
THREED_DIR = EXP_DIR / "threed"
THREED_DIR.mkdir(parents=True, exist_ok=True)

PROC_3D_STACK_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif")
assert PROC_3D_STACK_PATH.exists(), f"❌ 3D stack 不存在: {PROC_3D_STACK_PATH}"

# rank2 模型路径（你这次中断的是这个）
MODEL_TAG = "FINAL_rank2__C2_lr8e5_wd9e3"
MODEL_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421/full_data_view/train_all_old/models/cellpose_1773323559.9886878")
assert MODEL_PATH.exists(), f"❌ 模型不存在: {MODEL_PATH}"

OUT_DIR = THREED_DIR / MODEL_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🎯 MODEL_TAG : {MODEL_TAG}")
print(f"🎯 MODEL_PATH: {MODEL_PATH}")
print(f"🎯 OUT_DIR   : {OUT_DIR}")
print(f"🎯 STACK     : {PROC_3D_STACK_PATH}")

# =========================================================
# 2) 输出文件定义
# =========================================================
src_dir = PROC_3D_STACK_PATH.parent
stem = PROC_3D_STACK_PATH.stem

# Cellpose 通常会把结果写回输入图像所在目录
expected_src_mask = src_dir / f"{stem}_cp_masks.tif"

# 归档到当前 run 目录的两个文件
archived_mask = OUT_DIR / f"{stem}_cp_masks.tif"
tagged_mask = OUT_DIR / f"{MODEL_TAG}_cp_masks.tif"

# 日志
ts = time.strftime("%Y%m%d_%H%M%S")
infer_log = OUT_DIR / f"infer_{MODEL_TAG}_{ts}.log"

# =========================================================
# 3) 如果已经归档过，就直接复用，不重复跑
# =========================================================
if archived_mask.exists():
    print(f"✅ 已存在归档结果，跳过重跑: {archived_mask}")
else:
    # -----------------------------------------------------
    # 3.1 如果输入目录已经有旧的 _cp_masks.tif，先删掉
    #     避免把上一次模型的结果误认成这一次的
    # -----------------------------------------------------
    if expected_src_mask.exists():
        print(f"🧹 删除输入目录中的旧 mask，避免串台: {expected_src_mask}")
        expected_src_mask.unlink()

    # -----------------------------------------------------
    # 3.2 运行 3D 推理
    # -----------------------------------------------------
    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(src_dir),
        "--image_path", str(PROC_3D_STACK_PATH),
        "--pretrained_model", str(MODEL_PATH),
        "--diameter", "8",
        "--do_3D",
        "--z_axis", "0",
        "--stitch_threshold", "0.5",
        "--save_tif",
        "--no_npy",
        "--use_gpu",
        "--verbose",
    ]

    print("-" * 100)
    print(f"3D infer: {MODEL_TAG}")
    print("CMD:", " ".join(cmd))
    print(f"LOG: {infer_log}")
    print("-" * 100)

    with open(infer_log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(OUT_DIR), stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()

    print(f"🔚 return code = {ret}")
    if ret != 0:
        print(f"❌ 3D inference failed: {MODEL_TAG}")
        print(f"请查看日志: {infer_log}")
        raise RuntimeError(f"3D inference failed: {MODEL_TAG}")

    # -----------------------------------------------------
    # 3.3 推理结束后，从输入目录寻找真实输出
    # -----------------------------------------------------
    candidates = []

    patterns = [
        f"{stem}_cp_masks.tif",
        f"{stem}*_cp_masks*.tif",
        f"{stem}*_masks*.tif",
        f"{stem}*seg*.tif",
    ]

    for pat in patterns:
        candidates.extend(sorted(src_dir.glob(pat)))

    # 去重
    uniq = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    candidates = uniq

    if len(candidates) == 0:
        print(f"⚠️ 未找到 3D mask 输出: {MODEL_TAG}")
        print(f"请检查日志: {infer_log}")
        raise FileNotFoundError(f"No 3D mask found for {MODEL_TAG}")

    # 优先选最标准的 _cp_masks.tif
    mask_path = None
    for p in candidates:
        if p.name == f"{stem}_cp_masks.tif":
            mask_path = p
            break
    if mask_path is None:
        mask_path = candidates[0]

    print(f"✅ 找到原始输出: {mask_path}")

    # -----------------------------------------------------
    # 3.4 复制归档到 run 目录
    # -----------------------------------------------------
    shutil.copy2(mask_path, archived_mask)
    shutil.copy2(mask_path, tagged_mask)

    print(f"📦 已归档到: {archived_mask}")
    print(f"📦 已另存副本: {tagged_mask}")

# =========================================================
# 4) 最终检查
# =========================================================
print("-" * 100)
print("Final check:")
print("archived_mask exists:", archived_mask.exists(), archived_mask)
print("tagged_mask   exists:", tagged_mask.exists(), tagged_mask)
print("log exists          :", infer_log.exists(), infer_log)
print("-" * 100)

# 可选：显示输出目录内容
for p in sorted(OUT_DIR.glob("*")):
    print(p.name)


# In[ ]:





# In[3]:


#补档
from pathlib import Path
import glob
import os

print("="*80)
print("🩹 Rebuild THREED_RESULTS from disk")
print("="*80)

THREED_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421/threed")
assert THREED_ROOT.exists(), f"❌ THREED_ROOT 不存在: {THREED_ROOT}"

THREED_RESULTS = {}

for subdir in sorted(THREED_ROOT.iterdir()):
    if not subdir.is_dir():
        continue
    
    tag = subdir.name
    
    # 1) 找 log
    log_candidates = sorted(subdir.glob("infer_*.log"))
    log_path = str(log_candidates[-1]) if log_candidates else None
    
    # 2) 找 mask
    # 优先找带 tag 的 mask，再退化到任意 *_cp_masks.tif
    tagged_mask_candidates = sorted(subdir.glob(f"{tag}_cp_masks.tif"))
    generic_mask_candidates = sorted(subdir.glob("*_cp_masks.tif"))
    
    mask_path = None
    archived_mask = None
    
    if tagged_mask_candidates:
        mask_path = str(tagged_mask_candidates[-1])
    
    # archived_mask 尽量找“不是 tagged 的那个”
    for p in generic_mask_candidates:
        if p.name != f"{tag}_cp_masks.tif":
            archived_mask = str(p)
            break
    
    # 如果没找到 tagged mask，就退化用任意 cp_masks
    if mask_path is None and generic_mask_candidates:
        mask_path = str(generic_mask_candidates[-1])
    
    THREED_RESULTS[tag] = {
        "model_tag": tag,
        "out_dir": str(subdir),
        "mask_path": mask_path,
        "archived_mask": archived_mask,
        "log_path": log_path,
    }

print(f"✅ 已重建 THREED_RESULTS, 共 {len(THREED_RESULTS)} 个条目")
for k, v in THREED_RESULTS.items():
    print("-"*80)
    print("tag         :", k)
    print("mask_path   :", v["mask_path"])
    print("archived_mask:", v["archived_mask"])
    print("log_path    :", v["log_path"])
    print("out_dir     :", v["out_dir"])


# In[5]:


from pathlib import Path
import pandas as pd

print("="*80)
print("🩹 Rebuild vars for Cell 15")
print("="*80)

# =========================
# 1) 实验根目录
# =========================
EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421")
THREED_ROOT = EXP_DIR / "threed"
MET_DIR = EXP_DIR / "metrics"
MET_DIR.mkdir(parents=True, exist_ok=True)

print(f"EXP_DIR   : {EXP_DIR}")
print(f"THREED_ROOT: {THREED_ROOT}")
print(f"MET_DIR   : {MET_DIR}")

# =========================
# 2) 重建 THREED_RESULTS
# 你这里要放 14 和 14B 对应的两个模型
# =========================
THREED_RESULTS = {
    # 这里是 14B 这个你已经确认成功的模型
    "FINAL_rank2__C2_lr8e5_wd9e3": {
        "model_tag": "FINAL_rank2__C2_lr8e5_wd9e3",
        "rank_2d": 2,
        "out_dir": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3"),
        "mask_path": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "FINAL_rank2__C2_lr8e5_wd9e3_cp_masks.tif"),
        "archived_mask": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "iccv_sbg_imNor_30_0.000010_cp_masks.tif"),
        "log_path": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "infer_FINAL_rank2__C2_lr8e5_wd9e3_20260314_194322.log"),
    },

    # 这里是 Cell 14 的另一个模型 —— 你把 tag 改成你实际那个目录名
    # 并把下面几个文件名按实际情况改掉
    "FINAL_rank1__YOUR_MODEL_TAG": {
        "model_tag": "FINAL_rank1__YOUR_MODEL_TAG",
        "rank_2d": 1,
        "out_dir": str(THREED_ROOT / "FINAL_rank1__YOUR_MODEL_TAG"),
        "mask_path": str(THREED_ROOT / "FINAL_rank1__YOUR_MODEL_TAG" / "FINAL_rank1__YOUR_MODEL_TAG_cp_masks.tif"),
        "archived_mask": str(THREED_ROOT / "FINAL_rank1__YOUR_MODEL_TAG" / "iccv_sbg_imNor_30_0.000010_cp_masks.tif"),
        "log_path": str(THREED_ROOT / "FINAL_rank1__YOUR_MODEL_TAG" / "infer_FINAL_rank1__YOUR_MODEL_TAG_xxxxxxxxxxxxx.log"),
    },
}

# =========================
# 3) 做存在性检查，避免 Cell 15 后面再炸
# =========================
for tag, meta in THREED_RESULTS.items():
    print("-"*80)
    print(f"TAG: {tag}")
    for k in ["out_dir", "mask_path", "archived_mask", "log_path"]:
        p = meta.get(k)
        exists = Path(p).exists() if p else False
        print(f"{k:14s}: {p} | exists={exists}")

print("-"*80)
print("✅ THREED_RESULTS / MET_DIR 已补回，可以继续跑 Cell 15")


# In[8]:


from pathlib import Path

THREED_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421/threed")

print("="*80)
print("📂 threed 子目录与文件")
print("="*80)

for subdir in sorted(THREED_ROOT.iterdir()):
    if subdir.is_dir():
        print(f"\n[DIR] {subdir.name}")
        for p in sorted(subdir.glob("*")):
            print("   ", p.name)


# In[9]:


from pathlib import Path
import pandas as pd

print("="*80)
print("🩹 Rebuild THREED_RESULTS / MET_DIR from real files")
print("="*80)

EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260312_proc_5fold_cv_20260312_131421")
THREED_ROOT = EXP_DIR / "threed"
MET_DIR = EXP_DIR / "metrics"
MET_DIR.mkdir(parents=True, exist_ok=True)

THREED_RESULTS = {
    "FINAL_rank1__C3_lr9e5_wd8e3": {
        "model_tag": "FINAL_rank1__C3_lr9e5_wd8e3",
        "rank_2d": 1,
        "out_dir": str(THREED_ROOT / "FINAL_rank1__C3_lr9e5_wd8e3"),
        # rank1 没有 tagged mask，就直接指向唯一存在的 archived mask
        "mask_path": str(THREED_ROOT / "FINAL_rank1__C3_lr9e5_wd8e3" / "iccv_sbg_imNor_30_0.000010_cp_masks.tif"),
        "archived_mask": str(THREED_ROOT / "FINAL_rank1__C3_lr9e5_wd8e3" / "iccv_sbg_imNor_30_0.000010_cp_masks.tif"),
        "log_path": None,
    },
    "FINAL_rank2__C2_lr8e5_wd9e3": {
        "model_tag": "FINAL_rank2__C2_lr8e5_wd9e3",
        "rank_2d": 2,
        "out_dir": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3"),
        "mask_path": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "FINAL_rank2__C2_lr8e5_wd9e3_cp_masks.tif"),
        "archived_mask": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "iccv_sbg_imNor_30_0.000010_cp_masks.tif"),
        "log_path": str(THREED_ROOT / "FINAL_rank2__C2_lr8e5_wd9e3" / "infer_FINAL_rank2__C2_lr8e5_wd9e3_20260314_194322.log"),
    },
}

for tag, meta in THREED_RESULTS.items():
    print("-"*80)
    print("tag         :", tag)
    print("rank_2d     :", meta["rank_2d"])
    print("mask_path   :", meta["mask_path"], "| exists =", Path(meta["mask_path"]).exists())
    print("archived    :", meta["archived_mask"], "| exists =", Path(meta["archived_mask"]).exists())
    print("log_path    :", meta["log_path"], "| exists =", (Path(meta["log_path"]).exists() if meta["log_path"] else False))
    print("out_dir     :", meta["out_dir"], "| exists =", Path(meta["out_dir"]).exists())

print("-"*80)
print("✅ THREED_RESULTS 和 MET_DIR 已补回")


# In[10]:


# Cell 15：3D 结果摘要
import pandas as pd

print("="*80)
print("📦 Cell 15 | Summarize 3D results")
print("="*80)

assert "THREED_RESULTS" in globals(), "未找到 THREED_RESULTS，请先跑 Cell 14"

rows = []
for tag, info in THREED_RESULTS.items():
    meta = info.get("meta", {})
    stats = info.get("stats", {})
    rows.append({
        "tag": tag,
        "rank_2d": meta.get("rank_2d"),
        "param_tag": meta.get("param_tag"),
        "AP50_2d": meta.get("AP50_2d"),
        "F1_2d": meta.get("F1_2d"),
        "Recall_2d": meta.get("Recall_2d"),
        "Precision_2d": meta.get("Precision_2d"),
        "total_cells_3d": stats.get("total_cells"),
        "mask_path": info.get("mask_path"),
        "out_dir": meta.get("out_dir"),
    })

threed_df = pd.DataFrame(rows).sort_values(["rank_2d"]).reset_index(drop=True)
THREED_SUMMARY_CSV = MET_DIR / "threed_summary.csv"
threed_df.to_csv(THREED_SUMMARY_CSV, index=False)

print(threed_df)
print("\nSaved to:", THREED_SUMMARY_CSV)

print("\n✅ Cell 15 done.")
print("="*80)


# In[ ]:





# In[12]:


for tag in THREED_RESULTS:
    if "meta" not in THREED_RESULTS[tag]:
        THREED_RESULTS[tag]["meta"] = {
            "rank_2d": THREED_RESULTS[tag].get("rank_2d"),
            "model_tag": THREED_RESULTS[tag].get("model_tag"),
            "out_dir": THREED_RESULTS[tag].get("out_dir"),
            "mask_path": THREED_RESULTS[tag].get("mask_path"),
            "archived_mask": THREED_RESULTS[tag].get("archived_mask"),
            "log_path": THREED_RESULTS[tag].get("log_path"),
        }

print("✅ 已为 THREED_RESULTS 补上 meta")
for tag, info in THREED_RESULTS.items():
    print(tag, "->", info["meta"])


# In[13]:


# Cell 16：3D 可视化对比（dict-style THREED_RESULTS, PROC）
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path

print("="*80)
print("🖼️ Cell 16 | Slice overlay comparison & local crop analysis for auto-picked top2")
print("="*80)

assert "THREED_RESULTS" in globals(), "未找到 THREED_RESULTS，请先运行 3D 推理 cell。"
assert isinstance(THREED_RESULTS, dict), f"当前 THREED_RESULTS 不是 dict，而是 {type(THREED_RESULTS)}"
assert len(THREED_RESULTS) >= 1, f"THREED_RESULTS 为空"

results_items = []
for tag, info in THREED_RESULTS.items():
    meta = info.get("meta", {})
    rank_2d = meta.get("rank_2d", 999)
    results_items.append((rank_2d, tag, info))
results_items = sorted(results_items, key=lambda x: x[0])

TAG_LEFT = results_items[0][1]
TAG_RIGHT = results_items[1][1] if len(results_items) >= 2 else results_items[0][1]

print("TAG_LEFT :", TAG_LEFT)
print("TAG_RIGHT:", TAG_RIGHT)

mask_dict = {tag: info["mask_path"] for _, tag, info in results_items}
print("Available tags:", list(mask_dict.keys()))

assert PROC_3D_STACK_PATH.exists(), f"PROC_3D_STACK_PATH 不存在: {PROC_3D_STACK_PATH}"

raw_stack = tiff.imread(str(PROC_3D_STACK_PATH))
mask_left = tiff.imread(mask_dict[TAG_LEFT])
mask_right = tiff.imread(mask_dict[TAG_RIGHT])

assert raw_stack.shape == mask_left.shape == mask_right.shape, \
    "raw_stack / mask_left / mask_right 形状不一致，请检查 3D 推理输出。"

Z_LAYER = raw_stack.shape[0] // 2
print("Using Z_LAYER =", Z_LAYER)

raw_slice = raw_stack[Z_LAYER]
left_slice = mask_left[Z_LAYER]
right_slice = mask_right[Z_LAYER]

raw_float = raw_slice.astype(np.float32)
lo, hi = np.percentile(raw_float, [1, 99])
raw_norm = np.clip((raw_float - lo) / (hi - lo + 1e-8), 0, 1)

def mask_to_boundary(mask2d):
    fg = (mask2d > 0).astype(np.uint8)
    up    = np.roll(fg, -1, axis=0)
    down  = np.roll(fg,  1, axis=0)
    left  = np.roll(fg, -1, axis=1)
    right = np.roll(fg,  1, axis=1)
    boundary = fg & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 0
    return boundary.astype(bool)

bd_left = mask_to_boundary(left_slice)
bd_right = mask_to_boundary(right_slice)

overlay_left = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
overlay_right = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
overlay_left[bd_left] = [1.0, 0.0, 0.0]
overlay_right[bd_right] = [0.0, 1.0, 0.0]

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(raw_norm, cmap="gray")
plt.title(f"Proc 3D raw | Z={Z_LAYER}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(overlay_left)
plt.title(f"{TAG_LEFT}\n2D rank={THREED_RESULTS[TAG_LEFT]['meta'].get('rank_2d', 'NA')}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_right)
plt.title(f"{TAG_RIGHT}\n2D rank={THREED_RESULTS[TAG_RIGHT]['meta'].get('rank_2d', 'NA')}")
plt.axis("off")

plt.tight_layout()
plt.show()

ys, xs = np.where((left_slice > 0) | (right_slice > 0))
if len(ys) == 0:
    yc, xc = raw_slice.shape[0] // 2, raw_slice.shape[1] // 2
else:
    yc, xc = int(np.median(ys)), int(np.median(xs))

CROP_HALF = 128
y1 = max(0, yc - CROP_HALF)
y2 = min(raw_slice.shape[0], yc + CROP_HALF)
x1 = max(0, xc - CROP_HALF)
x2 = min(raw_slice.shape[1], xc + CROP_HALF)

raw_crop = raw_norm[y1:y2, x1:x2]
left_crop = overlay_left[y1:y2, x1:x2]
right_crop = overlay_right[y1:y2, x1:x2]

print(f"Crop box: y=({y1},{y2}), x=({x1},{x2})")

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(raw_crop, cmap="gray")
plt.title("Proc 3D raw crop")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(left_crop)
plt.title(f"{TAG_LEFT} crop")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(right_crop)
plt.title(f"{TAG_RIGHT} crop")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("📌 Top2 summary")
print("="*80)
for tag in [TAG_LEFT, TAG_RIGHT]:
    meta = THREED_RESULTS[tag].get("meta", {})
    stats = THREED_RESULTS[tag].get("stats", {})
    print(f"\n[{tag}]")
    print("2D rank     :", meta.get("rank_2d"))
    print("param_tag   :", meta.get("param_tag"))
    print("AP50_2d     :", meta.get("AP50_2d"))
    print("F1_2d       :", meta.get("F1_2d"))
    print("Recall_2d   :", meta.get("Recall_2d"))
    print("Precision_2d:", meta.get("Precision_2d"))
    print("total_cells :", stats.get("total_cells"))
    print("mask_path   :", THREED_RESULTS[tag].get("mask_path"))


# In[ ]:




