#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 0：Imports + 环境自检（跑一次）
import os, sys, json, time, shlex, subprocess, socket
from pathlib import Path
from datetime import datetime

print("="*80)
print("🧪 Cell 0 | Environment sanity check")
print("="*80)

# ---- 基本环境信息 ----
print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])
print("CWD:", os.getcwd())
print("User:", os.environ.get("USER", "unknown"))
print("HOSTNAME:", socket.gethostname())

# ---- 集群节点提醒：避免在 login 节点跑训练 ----
host = socket.gethostname().lower()
if "login" in host or "a2n" in host or "log" in host:
    print("\n⚠️ 节点提醒：你当前看起来像在登录节点/登录机上。")
    print("   真正训练请务必在计算节点跑（salloc / srun / sbatch）。")
else:
    print("\n✅ 节点看起来不像登录节点（仍建议你确认是计算节点）。")

# ---- 子进程命令包装 ----
def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return 999, "", repr(e)

# ---- cellpose 可用性自检 ----
try:
    import cellpose
    print("\n✅ cellpose import OK | version(attr):", getattr(cellpose, "__version__", "unknown"))
except Exception as e:
    print("\n❌ cellpose import FAILED:", repr(e))
    print("=> ⚠️ 别往下跑了！先把 notebook kernel 切到 cpsm 环境再继续。")
    raise

# ---- metadata 版本自检 ----
try:
    from importlib.metadata import version
    print("cellpose_version(meta):", version("cellpose"))
    print("torch_version:", version("torch"))
except Exception as e:
    print("⚠️ metadata version read failed:", repr(e))

# ---- CUDA / GPU 自检 ----
code, out, err = _run(["nvidia-smi", "-L"])
if code == 0 and out:
    print("\n🟢 GPU detected (nvidia-smi -L):")
    print(out)
else:
    print("\n⚠️ 没找到 nvidia-smi 或当前无 GPU 可见。")
    if err:
        print("stderr:", err[:300])

# ---- Git 信息快照（可选）----
def get_git_snapshot():
    code, out, _ = _run(["git", "rev-parse", "--show-toplevel"])
    if code != 0 or not out:
        return None

    repo_root = out.strip()
    code, commit, _ = _run(["git", "rev-parse", "HEAD"])
    code2, status, _ = _run(["git", "status", "--porcelain"])
    return {
        "repo_root": repo_root,
        "commit": commit.strip() if commit else None,
        "dirty": True if status.strip() else False,
        "status_porcelain": status.splitlines()[:50],
    }

git_info = get_git_snapshot()
if git_info:
    print("\n📌 Git snapshot:")
    print("repo_root:", git_info["repo_root"])
    print("commit:", git_info["commit"])
    print("dirty:", git_info["dirty"])
else:
    print("\nℹ️ 当前目录不是 git repo（跳过 git snapshot）。")

print("\n✅ Cell 0 done.")
print("="*80)


# In[2]:


# Cell 1：路径常量（固定不动）+ 实验目录初始化
from pathlib import Path
from datetime import datetime
import os, json

print("="*80)
print("🧱 Cell 1 | Paths & experiment directory init")
print("="*80)

# ========= 1) 项目根目录 =========
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

# ========= 2) 数据集路径（这轮固定） =========
DATA_ROOT = (ROOT / "Cellpose2TrainDataset").resolve()

# 训练集：保持不变
TRAIN_DIR = (DATA_ROOT / "trainset_ft_os3").resolve()

# 新验证集：这轮训练期 test_dir 的来源
NEWVAL_ROOT = (DATA_ROOT / "new_val_proc").resolve()
VAL_IMG_DIR = (NEWVAL_ROOT / "images_sp_lcn").resolve()
VAL_GT_DIR  = (NEWVAL_ROOT / "ground").resolve()

# ========= 3) 实验目录 =========
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"exp_20260306_retrain_newval_{STAMP}"   # 你后续想改实验名字，就改这里
EXP_DIR = (ROOT / "runs" / EXP_NAME).resolve()

LOG_DIR     = EXP_DIR / "logs"
MET_DIR     = EXP_DIR / "metrics"
CFG_DIR     = EXP_DIR / "config"
INFER_DIR   = EXP_DIR / "infer"
EVAL_DIR    = EXP_DIR / "eval"
EXPORT_DIR  = EXP_DIR / "exports"
DELIV_DIR   = EXP_DIR / "delivery"

# 训练期专用的新验证集“视图目录”
VALVIEW_DIR = EXP_DIR / "valview_newproc"

for d in [LOG_DIR, MET_DIR, CFG_DIR, INFER_DIR, EVAL_DIR, EXPORT_DIR, DELIV_DIR, VALVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========= 4) 写一份路径索引 =========
PATH_INDEX = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "ROOT": str(ROOT),
    "DATA_ROOT": str(DATA_ROOT),
    "TRAIN_DIR": str(TRAIN_DIR),
    "NEWVAL_ROOT": str(NEWVAL_ROOT),
    "VAL_IMG_DIR": str(VAL_IMG_DIR),
    "VAL_GT_DIR": str(VAL_GT_DIR),
    "EXP_NAME": EXP_NAME,
    "EXP_DIR": str(EXP_DIR),
    "LOG_DIR": str(LOG_DIR),
    "MET_DIR": str(MET_DIR),
    "CFG_DIR": str(CFG_DIR),
    "INFER_DIR": str(INFER_DIR),
    "EVAL_DIR": str(EVAL_DIR),
    "EXPORT_DIR": str(EXPORT_DIR),
    "DELIV_DIR": str(DELIV_DIR),
    "VALVIEW_DIR": str(VALVIEW_DIR),
    "note": "Retrain sweep with new_val_proc used as training-time validation/test_dir.",
}
(PATH_INDEX_PATH := (CFG_DIR / "PATHS.json")).write_text(
    json.dumps(PATH_INDEX, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# ========= 5) 打印确认 =========
print("ROOT        :", ROOT)
print("DATA_ROOT   :", DATA_ROOT)
print("TRAIN_DIR   :", TRAIN_DIR,   "| exists:", TRAIN_DIR.exists())
print("NEWVAL_ROOT :", NEWVAL_ROOT, "| exists:", NEWVAL_ROOT.exists())
print("VAL_IMG_DIR :", VAL_IMG_DIR, "| exists:", VAL_IMG_DIR.exists())
print("VAL_GT_DIR  :", VAL_GT_DIR,  "| exists:", VAL_GT_DIR.exists())

print("\nEXP_DIR     :", EXP_DIR)
print("LOG_DIR     :", LOG_DIR)
print("MET_DIR     :", MET_DIR)
print("CFG_DIR     :", CFG_DIR)
print("INFER_DIR   :", INFER_DIR)
print("EVAL_DIR    :", EVAL_DIR)
print("EXPORT_DIR  :", EXPORT_DIR)
print("DELIV_DIR   :", DELIV_DIR)
print("VALVIEW_DIR :", VALVIEW_DIR)

print("\n📌 PATHS.json saved to:", PATH_INDEX_PATH)

# ========= 6) 基础断言 =========
assert ROOT.exists(), f"ROOT not found: {ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
assert TRAIN_DIR.exists(), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert NEWVAL_ROOT.exists(), f"NEWVAL_ROOT not found: {NEWVAL_ROOT}"
assert VAL_IMG_DIR.exists(), f"VAL_IMG_DIR not found: {VAL_IMG_DIR}"
assert VAL_GT_DIR.exists(), f"VAL_GT_DIR not found: {VAL_GT_DIR}"

print("\n✅ Cell 1 done.")
print("="*80)


# In[3]:


# Cell 2：把 new_val_proc 整理成训练期可直接使用的 test_dir（valview_newproc）
import re, json, shutil
from pathlib import Path

print("="*80)
print("🧩 Cell 2 | Build training-time val view from new_val_proc")
print("="*80)

# -----------------------------
# 1) 文件后缀配置
# -----------------------------
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]

# 末尾数字提取：例如
# image_sparse_ln_0000.tif      -> 0
# 20220317_gou_Dataset_00000.tif -> 0
NUM_TAIL = re.compile(r"(\d+)$")

def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

# -----------------------------
# 2) 扫描原始 new_val_proc
# -----------------------------
imgs_all = list_files(VAL_IMG_DIR, IMG_EXTS)
gts_all  = list_files(VAL_GT_DIR, GT_EXTS)

print("VAL_IMG_DIR:", VAL_IMG_DIR)
print("VAL_GT_DIR :", VAL_GT_DIR)
print("#images:", len(imgs_all), "| examples:", [p.name for p in imgs_all[:5]])
print("#gts   :", len(gts_all),  "| examples:", [p.name for p in gts_all[:5]])

assert len(imgs_all) > 0, f"VAL_IMG_DIR has no images: {VAL_IMG_DIR}"
assert len(gts_all) > 0,  f"VAL_GT_DIR has no masks: {VAL_GT_DIR}"

# -----------------------------
# 3) 建立编号映射
# -----------------------------
img_map = {}
for p in imgs_all:
    i = tail_id(p)
    if i is not None:
        img_map[i] = p

gt_map = {}
for p in gts_all:
    i = tail_id(p)
    if i is not None:
        gt_map[i] = p

common_ids = sorted(set(img_map.keys()) & set(gt_map.keys()))
missing_img_ids = sorted(set(gt_map.keys()) - set(img_map.keys()))
missing_gt_ids  = sorted(set(img_map.keys()) - set(gt_map.keys()))

print("\n✅ common matched ids:", len(common_ids))
print("First 10 matched pairs:")
for i in common_ids[:10]:
    print(f"  id={i:05d} | IMG={img_map[i].name} | GT={gt_map[i].name}")

print("\nExamples missing GT ids:", missing_gt_ids[:10])
print("Examples missing IMG ids:", missing_img_ids[:10])

assert len(common_ids) > 0, "No common image/GT ids found. Check filename numbering rule."

# -----------------------------
# 4) 清空 / 重建 VALVIEW_DIR
#    注意：这里只清理当前实验目录下的 valview，不碰原始数据
# -----------------------------
if VALVIEW_DIR.exists():
    # 保险起见，只删里面内容，不删目录本身
    for p in VALVIEW_DIR.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

VALVIEW_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 5) 建立统一命名的软链接视图
#    形式：
#      nv_00000.tif
#      nv_00000_masks.tif
# -----------------------------
manifest_rows = []

for i in common_ids:
    img_src = img_map[i]
    gt_src  = gt_map[i]

    stem = f"nv_{i:05d}"
    img_link = VALVIEW_DIR / f"{stem}{img_src.suffix.lower()}"
    gt_link  = VALVIEW_DIR / f"{stem}_masks{gt_src.suffix.lower()}"

    if img_link.exists() or img_link.is_symlink():
        img_link.unlink()
    if gt_link.exists() or gt_link.is_symlink():
        gt_link.unlink()

    img_link.symlink_to(img_src)
    gt_link.symlink_to(gt_src)

    manifest_rows.append({
        "id": i,
        "stem": stem,
        "img_src": str(img_src),
        "gt_src": str(gt_src),
        "img_link": str(img_link),
        "gt_link": str(gt_link),
    })

# -----------------------------
# 6) 写 manifest
# -----------------------------
val_manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_img_dir": str(VAL_IMG_DIR),
    "source_gt_dir": str(VAL_GT_DIR),
    "valview_dir": str(VALVIEW_DIR),
    "n_pairs": len(manifest_rows),
    "pairs": manifest_rows,
    "note": "This directory is a training-time test_dir view for cellpose CLI. Files are symlinks.",
}

VAL_MANIFEST_PATH = CFG_DIR / "valview_manifest.json"
VAL_MANIFEST_PATH.write_text(
    json.dumps(val_manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# -----------------------------
# 7) 最终检查
# -----------------------------
valview_files = sorted(VALVIEW_DIR.iterdir())
n_imgs = len([p for p in valview_files if p.is_file() and "_masks" not in p.stem])
n_gts  = len([p for p in valview_files if p.is_file() and "_masks" in p.stem])

print("\nVALVIEW_DIR:", VALVIEW_DIR)
print("Total files in valview:", len(valview_files))
print("Image links:", n_imgs)
print("Mask  links:", n_gts)
print("Manifest:", VAL_MANIFEST_PATH)

print("\nSample files in VALVIEW_DIR:")
for p in valview_files[:10]:
    print(" ", p.name, "->", os.readlink(p) if p.is_symlink() else "(not symlink)")

assert n_imgs == len(common_ids), f"Image link count mismatch: {n_imgs} vs {len(common_ids)}"
assert n_gts  == len(common_ids), f"Mask link count mismatch: {n_gts} vs {len(common_ids)}"

print("\n✅ Cell 2 done.")
print("="*80)


# In[4]:


# Cell 3：Sweep 参数表（这轮：训练集不变，test_dir 改成 new val view）
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("🧪 Cell 3 | Define sweep variants")
print("="*80)

# ============ 统一默认参数（这轮重训的共同底座）============
BASE = {
    # 训练集：保持不变
    "train_dir": str(TRAIN_DIR),

    # 训练期验证集：改用 new_val_proc 整理后的 valview
    "test_dir": str(VALVIEW_DIR),

    # mask 命名规则：valview 中 GT 已整理成 *_masks.xxx
    "mask_filter": "_masks",

    # 预训练模型
    "pretrained_model": "cpsam",

    # 保存 / 日志 / 运行行为
    "save_every": 5,
    "use_gpu": True,
    "verbose": True,

    # patch size
    "bsize": 256,
}
# =========================================================

# ============ 版本设计 ============
SWEEP = [
    # V0：基线复刻（相当于“旧体系”在新验证口径下的复跑）
    dict(
        tag="V0_baseline_repro",
        diameter=20,
        learning_rate=1e-4,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=100,
        augment=False,
    ),

    # V1：低 lr + 长训 + augment（偏抗过拟合）
    dict(
        tag="V1_lowLR_long_aug",
        diameter=20,
        learning_rate=5e-5,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=160,
        augment=True,
    ),

    # V2：原“大 batch”方案 OOM 后的稳定收敛替代版
    # 思路：既然 64/48 都炸，就回到 bs=32，但把 lr 再降一点，
    # 尽量保留“更稳、更保守更新”的对照意味。
    dict(
        tag="V2_stable_alt_aug",
        diameter=20,
        learning_rate=3e-5,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=120,
        augment=True,
        note="Engineering replacement of failed big-batch plan (OOM at bs=64/48).",
    ),

    # V3：低 wd（更敢切边界，但也可能 FP 上升）
    dict(
        tag="V3_lowWD_aug",
        diameter=20,
        learning_rate=1e-4,
        weight_decay=1e-3,
        train_batch_size=32,
        n_epochs=100,
        augment=True,
    ),

    # V4：diam 偏小（更倾向拆分粘连）
    dict(
        tag="V4_diam16_aug",
        diameter=16,
        learning_rate=1e-4,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=100,
        augment=True,
    ),

    # V5：diam 偏大（更倾向整体一致性）
    dict(
        tag="V5_diam24_aug",
        diameter=24,
        learning_rate=1e-4,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=100,
        augment=True,
    ),

    # V6：Transformer 对照（挂在更稳健的低 lr 长训风格上）
    dict(
        tag="V6_transformer_on",
        diameter=20,
        learning_rate=5e-5,
        weight_decay=1e-2,
        train_batch_size=32,
        n_epochs=160,
        augment=True,
        transformer=True,
    ),
]
# ==================================

print("BASE:")
print(json.dumps(BASE, indent=2, ensure_ascii=False))

print(f"\n✅ Defined {len(SWEEP)} sweep variants:")
for i, v in enumerate(SWEEP, 1):
    print(f"[{i}] {v['tag']}")
    print("   ",
          f"d={v['diameter']} | "
          f"lr={v['learning_rate']} | "
          f"wd={v['weight_decay']} | "
          f"bs={v['train_batch_size']} | "
          f"epochs={v['n_epochs']} | "
          f"augment={v.get('augment', False)} | "
          f"transformer={v.get('transformer', False)}")

print("\n✅ Cell 3 done.")
print("="*80)


# In[5]:


# Cell 4：批量 commit runs -> RUNS.jsonl + per-run config snapshot
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("📝 Cell 4 | Commit runs into RUNS.jsonl")
print("="*80)

RUNS = []
RUN_INDEX_PATH = CFG_DIR / "RUNS.jsonl"

# 如果你担心误重复运行这一格，可以保守一点：
# 若 RUNS.jsonl 已存在且非空，则先提示，不自动覆盖
if RUN_INDEX_PATH.exists() and RUN_INDEX_PATH.stat().st_size > 0:
    raise RuntimeError(
        f"RUNS.jsonl already exists and is non-empty:\n{RUN_INDEX_PATH}\n"
        "为避免重复提交 run，请新建实验目录后再运行，或手动确认后清空该文件。"
    )

RUN_INDEX_PATH.touch(exist_ok=True)

def commit_run(base: dict, variant: dict) -> dict:
    p = dict(base)
    p.update(variant)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{p['tag']}"
        f"_d{p['diameter']}"
        f"_lr{p['learning_rate']}"
        f"_wd{p['weight_decay']}"
        f"_bs{p['train_batch_size']}"
        f"_e{p['n_epochs']}"
        f"_{ts}"
    )

    ctx = {
        "run_name": run_name,
        "tag": p["tag"],
        "created_at": datetime.now().isoformat(timespec="seconds"),

        # 本 run 的完整参数
        "params": p,

        # 路径产物
        "log_path": str(LOG_DIR / f"train_{run_name}.log"),
        "pid_path": str(LOG_DIR / f"train_{run_name}.pid"),
        "metrics_path": str(MET_DIR / f"metrics_{run_name}.csv"),
        "cmd_path": str(CFG_DIR / f"cmd_{run_name}.txt"),
        "config_snapshot_path": str(CFG_DIR / f"config_{run_name}.json"),

        # 训练后回填
        "model_dir": None,

        # 状态标记（后续 cell 会更新）
        "status": "NOT_STARTED",
        "note": p.get("note", ""),
    }

    # 每个 run 单独 snapshot
    Path(ctx["config_snapshot_path"]).write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 追加写入总索引
    with open(RUN_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ctx, ensure_ascii=False) + "\n")

    return ctx

for v in SWEEP:
    ctx = commit_run(BASE, v)
    RUNS.append(ctx)

print(f"✅ 已生成 {len(RUNS)} 个 runs")
print("📌 RUNS index:", RUN_INDEX_PATH)

print("\n--- Runs preview ---")
for r in RUNS:
    p = r["params"]
    print(f"- {r['run_name']}")
    print("   ",
          f"tag={r['tag']} | "
          f"train_dir={p['train_dir']} | "
          f"test_dir={p['test_dir']} | "
          f"d={p['diameter']} | "
          f"lr={p['learning_rate']} | "
          f"wd={p['weight_decay']} | "
          f"bs={p['train_batch_size']} | "
          f"epochs={p['n_epochs']} | "
          f"augment={p.get('augment', False)} | "
          f"transformer={p.get('transformer', False)}")

# 额外写一份 sweep 总览
SWEEP_SUMMARY_PATH = CFG_DIR / "SWEEP_SUMMARY.json"
SWEEP_SUMMARY_PATH.write_text(
    json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(EXP_DIR),
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(VALVIEW_DIR),
        "n_runs": len(RUNS),
        "runs": RUNS,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n📌 Sweep summary saved to:", SWEEP_SUMMARY_PATH)
print("\n✅ Cell 4 done.")
print("="*80)


# In[ ]:





# In[ ]:


#如果断联运行5~8，然后运行9（会自动跳过已训练好的）


# In[7]:


# Cell 5：从 RUNS.jsonl 恢复上下文（断线/重连后优先跑这个）
import json
from pathlib import Path

print("="*80)
print("🔁 Cell 5 | Restore context from RUNS.jsonl")
print("="*80)

# ========= 这里改成你刚刚第 1 批生成的真实实验目录 =========
# 如果你当前 notebook 里还保留着 EXP_DIR 变量，这里会自动复用；
# 否则请手动把下面路径改成你的实验目录。
if "EXP_DIR" in globals():
    EXP = Path(EXP_DIR).resolve()
else:
    EXP = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/REPLACE_WITH_YOUR_EXP_DIR").resolve()

CFG_DIR = EXP / "config"
LOG_DIR = EXP / "logs"
MET_DIR = EXP / "metrics"
INFER_DIR = EXP / "infer"
EVAL_DIR = EXP / "eval"
EXPORT_DIR = EXP / "exports"
DELIV_DIR = EXP / "delivery"
VALVIEW_DIR = EXP / "valview_newproc"

PATHS_JSON = CFG_DIR / "PATHS.json"
RUNS_JSONL = CFG_DIR / "RUNS.jsonl"

assert EXP.exists(), f"EXP_DIR not found: {EXP}"
assert CFG_DIR.exists(), f"CFG_DIR not found: {CFG_DIR}"
assert PATHS_JSON.exists(), f"PATHS.json not found: {PATHS_JSON}"
assert RUNS_JSONL.exists(), f"RUNS.jsonl not found: {RUNS_JSONL}"

paths = json.loads(PATHS_JSON.read_text(encoding="utf-8"))
RUNS = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

ROOT = Path(paths["ROOT"]).resolve()
DATA_ROOT = Path(paths["DATA_ROOT"]).resolve()
TRAIN_DIR = Path(paths["TRAIN_DIR"]).resolve()
NEWVAL_ROOT = Path(paths["NEWVAL_ROOT"]).resolve()
VAL_IMG_DIR = Path(paths["VAL_IMG_DIR"]).resolve()
VAL_GT_DIR = Path(paths["VAL_GT_DIR"]).resolve()

print("EXP_DIR     :", EXP)
print("CFG_DIR     :", CFG_DIR)
print("LOG_DIR     :", LOG_DIR)
print("MET_DIR     :", MET_DIR)
print("VALVIEW_DIR :", VALVIEW_DIR)

print("\nTRAIN_DIR   :", TRAIN_DIR, "| exists:", TRAIN_DIR.exists())
print("NEWVAL_ROOT :", NEWVAL_ROOT, "| exists:", NEWVAL_ROOT.exists())
print("VAL_IMG_DIR :", VAL_IMG_DIR, "| exists:", VAL_IMG_DIR.exists())
print("VAL_GT_DIR  :", VAL_GT_DIR,  "| exists:", VAL_GT_DIR.exists())

print("\n✅ loaded RUNS:", len(RUNS))
if RUNS:
    print("First run:", RUNS[0]["run_name"])
    print("Last  run:", RUNS[-1]["run_name"])

print("\n✅ Cell 5 done.")
print("="*80)


# In[8]:


# Cell 6：训练前状态扫描（不修改，只看状态）
import json, subprocess, re
from pathlib import Path

print("="*80)
print("🩺 Cell 6 | Scan run statuses")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

PAT_SAVED = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")
PAT_OOM = re.compile(r"(OutOfMemoryError|CUDA out of memory)", re.IGNORECASE)
PAT_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")

def ps_info(pid: int) -> str:
    r = subprocess.run(
        ["ps", "-o", "pid,ppid,stat,etime,cmd", "-p", str(pid)],
        text=True, capture_output=True
    )
    return (r.stdout or r.stderr).strip()

def ps_stat(pid: int) -> str:
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True)
    return (r.stdout or "").strip()

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def tail_text(path: Path, n=40):
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""

def guess_saved_model_from_log(log_path: Path):
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in reversed(txt.splitlines()[-1500:]):
        m = PAT_SAVED.search(line)
        if m:
            return m.group("dir")
    return None

status_rows = []

for i, r in enumerate(runs, 1):
    run_name = r["run_name"]
    tag = r["tag"]

    pid_path = Path(r["pid_path"])
    log_path = Path(r["log_path"])
    snap_path = Path(r["config_snapshot_path"])

    snap = read_json(snap_path) or {}
    model_dir = snap.get("model_dir") or r.get("model_dir")
    model_ok = bool(model_dir) and Path(model_dir).exists()

    pid = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None

    running = False
    zombie = False
    psline = ""
    if pid is not None:
        st = ps_stat(pid)
        psline = ps_info(pid)
        if st:
            if "Z" in st:
                zombie = True
            else:
                running = True

    guessed_model = guess_saved_model_from_log(log_path)

    log_tail = tail_text(log_path, 30)
    has_oom = bool(PAT_OOM.search(log_tail))
    has_tb = bool(PAT_TRACEBACK.search(log_tail))
    log_exists = log_path.exists()

    if model_ok:
        status = "DONE"
    elif running:
        status = "RUNNING"
    elif (not log_exists) and (pid is None):
        status = "NOT_STARTED"
    elif zombie:
        status = "FAILED"
    elif has_oom or has_tb:
        status = "FAILED"
    elif guessed_model and Path(guessed_model).exists():
        status = "BROKEN_SNAPSHOT"
    else:
        status = "FAILED" if log_exists else "NOT_STARTED"

    status_rows.append({
        "idx": i,
        "tag": tag,
        "run_name": run_name,
        "status": status,
        "pid": pid,
        "log_exists": log_exists,
        "snapshot_exists": snap_path.exists(),
        "model_dir": model_dir,
        "model_ok": model_ok,
        "guessed_model": guessed_model,
        "has_oom": has_oom,
        "has_traceback": has_tb,
        "ps": psline,
    })

print(f"Total runs: {len(status_rows)}")
print("-"*120)
for row in status_rows:
    print(f"[{row['idx']}] {row['tag']:<22} | status={row['status']:<15} | pid={str(row['pid']):<8} | model_ok={row['model_ok']}")
    if row["status"] in ["FAILED", "BROKEN_SNAPSHOT", "RUNNING"]:
        print("   run_name      :", row["run_name"])
        print("   log_exists    :", row["log_exists"])
        print("   guessed_model :", row["guessed_model"])
        if row["has_oom"]:
            print("   ⚠️ detected OOM in log tail")
        if row["has_traceback"]:
            print("   ⚠️ detected traceback in log tail")
        if row["ps"]:
            print("   ps            :", row["ps"].replace("\n", " | "))

print("\n✅ Cell 6 done.")
print("="*80)


# In[9]:


# Cell 7：自动修复与清理（清 stale pid + 回填 model_dir + 更新 snapshot 状态）
import json, subprocess, re
from pathlib import Path

print("="*80)
print("🧹 Cell 7 | Auto-fix stale states and patch snapshots")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

PAT_SAVED = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")
PAT_OOM = re.compile(r"(OutOfMemoryError|CUDA out of memory)", re.IGNORECASE)
PAT_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")

def ps_stat(pid: int) -> str:
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True)
    return (r.stdout or "").strip()

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def guess_saved_model_from_log(log_path: Path):
    if not log_path.exists():
        return None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in reversed(txt.splitlines()[-1500:]):
        m = PAT_SAVED.search(line)
        if m:
            return m.group("dir")
    return None

def tail_text(path: Path, n=50):
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])

patched_model_dir = 0
removed_pid = 0
updated_status = 0

for r in runs:
    pid_path = Path(r["pid_path"])
    log_path = Path(r["log_path"])
    snap_path = Path(r["config_snapshot_path"])

    snap = read_json(snap_path)
    if snap is None:
        continue

    # ---------- 1) 清理 stale pid ----------
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            st = ps_stat(pid)
            if (not st) or ("Z" in st):
                pid_path.unlink(missing_ok=True)
                removed_pid += 1
                print("🧽 removed stale pid:", pid, "|", r["tag"])
        except Exception:
            pid_path.unlink(missing_ok=True)
            removed_pid += 1
            print("🧽 removed unreadable pid file for", r["tag"])

    # ---------- 2) 回填 model_dir ----------
    model_dir = snap.get("model_dir")
    if not model_dir:
        guessed = guess_saved_model_from_log(log_path)
        if guessed and Path(guessed).exists():
            snap["model_dir"] = guessed
            model_dir = guessed
            patched_model_dir += 1
            print("✅ patched model_dir for", r["tag"], "->", guessed)

    # ---------- 3) 更新 status ----------
    if model_dir and Path(model_dir).exists():
        new_status = "DONE"
    else:
        tail = tail_text(log_path, 50)
        if PAT_OOM.search(tail):
            new_status = "FAILED"
            snap["note"] = (snap.get("note", "") + " | failed with OOM").strip(" |")
        elif PAT_TRACEBACK.search(tail):
            new_status = "FAILED"
            snap["note"] = (snap.get("note", "") + " | failed with traceback").strip(" |")
        elif log_path.exists():
            new_status = "FAILED"
        else:
            new_status = "NOT_STARTED"

    if snap.get("status") != new_status:
        snap["status"] = new_status
        updated_status += 1

    write_json(snap_path, snap)

print("\nSummary:")
print("patched model_dir :", patched_model_dir)
print("removed stale pid :", removed_pid)
print("updated statuses  :", updated_status)
print("\n✅ Cell 7 done.")
print("="*80)


# In[10]:


# Cell 8：训练命令构建器（只定义函数，不开跑）
import os, json, shlex, re
from pathlib import Path

print("="*80)
print("🛠 Cell 8 | Training command builder")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
RUNS = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

PAT_SAVED = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def build_cmd(ctx: dict):
    p = ctx["params"]

    cmd = [
        "python", "-m", "cellpose",
        "--train",
        "--dir", str(p["train_dir"]),
        "--test_dir", str(p["test_dir"]),
        "--pretrained_model", str(p["pretrained_model"]),
        "--diameter", str(p["diameter"]),
        "--learning_rate", str(p["learning_rate"]),
        "--weight_decay", str(p["weight_decay"]),
        "--n_epochs", str(p["n_epochs"]),
        "--train_batch_size", str(p["train_batch_size"]),
        "--bsize", str(p["bsize"]),
        "--mask_filter", str(p["mask_filter"]),
        "--save_every", str(p["save_every"]),
    ]

    if p.get("transformer", False):
        cmd.append("--transformer")
    if p.get("augment", False):
        cmd.append("--augment")
    if p.get("use_gpu", True):
        cmd.append("--use_gpu")
    if p.get("verbose", True):
        cmd.append("--verbose")

    return cmd

def cmd_to_text(cmd):
    return " ".join(shlex.quote(x) for x in cmd)

def already_done(ctx: dict) -> bool:
    snap = read_json(Path(ctx["config_snapshot_path"])) or {}
    m = snap.get("model_dir")
    return bool(m) and Path(m).exists()

def capture_model_dir_from_log(log_path: Path):
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines[-1500:]):
        m = PAT_SAVED.search(line)
        if m:
            return m.group("dir")
    return None

# 预览前 2 条
print("Preview commands:")
for ctx in RUNS[:2]:
    cmd = build_cmd(ctx)
    print("-"*80)
    print(ctx["tag"])
    print(cmd_to_text(cmd))

print("\n✅ Cell 8 done.")
print("="*80)


# In[11]:


# Cell 9：串行训练执行器（主力 cell）
import os, time, json, shlex, subprocess, traceback
from pathlib import Path

print("="*80)
print("🚀 Cell 9 | Serial training executor")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
RUNS = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

# ========= 运行控制 =========
POLL_S = 60              # 每隔多少秒看一次进程
RUN_ONLY_TAGS = None     # 例如 ["V0_baseline_repro", "V4_diam16_aug"]；None 表示全跑
STOP_ON_FAILURE = False  # True=某个 run 失败就停；False=继续跑后面的
# ==========================

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def choose_runs(runs):
    if RUN_ONLY_TAGS is None:
        return runs
    picked = [r for r in runs if r["tag"] in RUN_ONLY_TAGS]
    print("Selected tags:", RUN_ONLY_TAGS)
    print("Selected runs:", len(picked))
    return picked

def mark_snapshot(ctx: dict, **kwargs):
    snap_path = Path(ctx["config_snapshot_path"])
    snap = read_json(snap_path) or {}
    snap.update(kwargs)
    write_json(snap_path, snap)

def tail_file(path: Path, n=20):
    if not path.exists():
        return "(no file)"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"

selected_runs = choose_runs(RUNS)
print("Runs to process:", len(selected_runs))

for idx, ctx in enumerate(selected_runs, 1):
    print("\n" + "="*100)
    print(f"[{idx}/{len(selected_runs)}] RUN: {ctx['run_name']}")
    print("="*100)

    # ---------- 已完成则跳过 ----------
    if already_done(ctx):
        print("✅ already done -> skip")
        mark_snapshot(ctx, status="DONE")
        continue

    cmd = build_cmd(ctx)
    cmd_text = cmd_to_text(cmd)

    cmd_path = Path(ctx["cmd_path"])
    log_path = Path(ctx["log_path"])
    pid_path = Path(ctx["pid_path"])
    snap_path = Path(ctx["config_snapshot_path"])

    cmd_path.write_text(cmd_text + "\n", encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("CMD:")
    print(cmd_text)
    print("\nLOG:", log_path)
    print("PID:", pid_path)

    # ---------- 环境变量：减少显存碎片 ----------
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        with open(log_path, "w", buffering=1, encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )

            pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
            mark_snapshot(
                ctx,
                status="RUNNING",
                launched_at=datetime.now().isoformat(timespec="seconds"),
                pid=proc.pid,
            )

            print("🚀 Started")
            print("   PID:", proc.pid)

            while True:
                ret = proc.poll()
                if ret is not None:
                    print("✅ process finished with return code:", ret)
                    break
                print(f"⏳ running... sleep {POLL_S}s")
                time.sleep(POLL_S)

        # ---------- 进程结束后清 pid ----------
        pid_path.unlink(missing_ok=True)

        # ---------- 回填 model_dir ----------
        guessed_model = capture_model_dir_from_log(log_path)
        if guessed_model and Path(guessed_model).exists():
            mark_snapshot(
                ctx,
                status="DONE",
                model_dir=guessed_model,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                return_code=ret,
            )
            print("🏁 model_dir:", guessed_model)
        else:
            # 失败 / 中断 / 未捕获
            note = "training ended but model_dir not captured; inspect log"
            log_tail = tail_file(log_path, 25)

            failed_note = note
            if "OutOfMemoryError" in log_tail or "CUDA out of memory" in log_tail:
                failed_note = "failed with OOM"
            elif "Traceback (most recent call last):" in log_tail:
                failed_note = "failed with traceback"

            mark_snapshot(
                ctx,
                status="FAILED",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                return_code=ret,
                note=failed_note,
            )

            print("⚠️ model_dir 未捕获。请看 log 尾巴：")
            print("-"*80)
            print(log_tail)
            print("-"*80)

            if STOP_ON_FAILURE:
                raise RuntimeError(f"Run failed and STOP_ON_FAILURE=True: {ctx['tag']}")

    except Exception as e:
        pid_path.unlink(missing_ok=True)
        mark_snapshot(
            ctx,
            status="FAILED",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            note=f"executor exception: {repr(e)}",
        )
        print("❌ Executor exception:", repr(e))
        print(traceback.format_exc())
        if STOP_ON_FAILURE:
            raise

print("\n🎉 Cell 9 finished: selected runs processed.")
print("="*80)


# In[12]:


# Cell 10：解析训练日志 -> summary_runs.csv
import json, re
from pathlib import Path
import pandas as pd

print("="*80)
print("📊 Cell 10 | Parse training logs -> summary_runs.csv")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
OUT_CSV = EXP / "summary_runs.csv"

runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

# 训练日志行格式示例：
# 2026-03-05 ... [INFO] 50, train_loss=0.7851, test_loss=0.8477, LR=0.000050, time ...
MET_RE = re.compile(
    r"\b(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)"
)

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

rows = []

for r in runs:
    snap = read_json(Path(r["config_snapshot_path"])) or {}
    model_dir = snap.get("model_dir")
    status = snap.get("status", r.get("status", "UNKNOWN"))

    log_path = Path(r["log_path"])
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    metrics = []
    for m in MET_RE.finditer(text):
        metrics.append({
            "epoch": int(m.group("epoch")),
            "train_loss": float(m.group("tr")),
            "test_loss": float(m.group("te")),
            "lr": float(m.group("lr")),
        })

    if metrics:
        dfm = pd.DataFrame(metrics).sort_values("epoch")
        best_idx = dfm["test_loss"].idxmin()
        best_epoch = int(dfm.loc[best_idx, "epoch"])
        best_test  = float(dfm.loc[best_idx, "test_loss"])
        best_train = float(dfm.loc[best_idx, "train_loss"])

        last_epoch = int(dfm.iloc[-1]["epoch"])
        last_train = float(dfm.iloc[-1]["train_loss"])
        last_test  = float(dfm.iloc[-1]["test_loss"])
        n_points = int(dfm.shape[0])
    else:
        best_epoch = best_test = best_train = None
        last_epoch = last_train = last_test = None
        n_points = 0

    p = r["params"]
    rows.append({
        "tag": r["tag"],
        "run_name": r["run_name"],
        "status": status,
        "model_dir": model_dir,
        "model_dir_exists": bool(model_dir) and Path(model_dir).exists(),

        "train_dir": p.get("train_dir"),
        "test_dir": p.get("test_dir"),
        "diameter": p.get("diameter"),
        "lr": p.get("learning_rate"),
        "weight_decay": p.get("weight_decay"),
        "train_batch_size": p.get("train_batch_size"),
        "n_epochs": p.get("n_epochs"),
        "augment": p.get("augment", False),
        "transformer": p.get("transformer", False),

        "n_logged_points": n_points,
        "best_epoch_by_test_loss": best_epoch,
        "best_train_loss_at_best_test": best_train,
        "best_test_loss": best_test,
        "last_epoch_logged": last_epoch,
        "last_train_loss": last_train,
        "last_test_loss": last_test,

        "log_path": str(log_path),
        "config_snapshot_path": r["config_snapshot_path"],
        "note": snap.get("note", r.get("note", "")),
    })

df = pd.DataFrame(rows).sort_values("tag")
df.to_csv(OUT_CSV, index=False)

print("✅ Wrote:", OUT_CSV)
pd.set_option("display.max_colwidth", 200)
display(df[[
    "tag", "status", "diameter", "augment", "transformer",
    "train_batch_size", "n_epochs",
    "best_epoch_by_test_loss", "best_test_loss", "last_test_loss",
    "model_dir"
]])

print("\n✅ Cell 10 done.")
print("="*80)


# In[13]:


# Cell 11：训练曲线可视化（单模型曲线 + 全部 test_loss 对比）
import json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("📈 Cell 11 | Plot training curves")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
CURVE_DIR = EXPORT_DIR / "train_curves"
CURVE_DIR.mkdir(parents=True, exist_ok=True)

runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

MET_RE = re.compile(
    r"\b(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)"
)

curve_map = {}

for r in runs:
    log_path = Path(r["log_path"])
    if not log_path.exists():
        print("⚠️ missing log:", log_path)
        continue

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = []
    for m in MET_RE.finditer(text):
        metrics.append({
            "epoch": int(m.group("epoch")),
            "train_loss": float(m.group("tr")),
            "test_loss": float(m.group("te")),
            "lr": float(m.group("lr")),
        })

    if not metrics:
        print("⚠️ no metric lines parsed for", r["tag"])
        continue

    dfm = pd.DataFrame(metrics).sort_values("epoch")
    curve_map[r["tag"]] = dfm

    # ---------- 单模型图 ----------
    plt.figure(figsize=(8, 5))
    plt.plot(dfm["epoch"], dfm["train_loss"], label="train_loss")
    plt.plot(dfm["epoch"], dfm["test_loss"], label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{r['tag']} | train/test loss")
    plt.legend()
    plt.tight_layout()

    out_png = CURVE_DIR / f"curve_{r['tag']}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("🖼 saved:", out_png)

# ---------- 总对比图：全部 test_loss ----------
if curve_map:
    plt.figure(figsize=(10, 6))
    for tag, dfm in curve_map.items():
        plt.plot(dfm["epoch"], dfm["test_loss"], label=tag)
    plt.xlabel("epoch")
    plt.ylabel("test_loss")
    plt.title("All runs | test_loss comparison")
    plt.legend()
    plt.tight_layout()

    out_png = CURVE_DIR / "all_runs_test_loss.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("🖼 saved:", out_png)

print("\nCURVE_DIR:", CURVE_DIR)
print("\n✅ Cell 11 done.")
print("="*80)


# In[14]:


# Cell 12：在 new_val_proc 上统一推理（所有成功模型）
import os, shlex, subprocess, time, json
from pathlib import Path

print("="*80)
print("🤖 Cell 12 | Unified inference on new_val_proc")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

# ========= 推理参数（统一赛场） =========
FLOW_TH = 0.4
CELLPROB_TH = 0.0
# ======================================

OUT_ROOT = (EXP / "battle_newval").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
RUN_OUT = OUT_ROOT / f"newval_{stamp}"
RUN_OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = RUN_OUT / "manifest.json"

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def sh(cmd, cwd=None):
    print("\n$", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout.strip()[:1500])
    if r.stderr.strip():
        print("stderr:", r.stderr.strip()[:1500])
    return r.returncode, r

# ---------- 收集可用模型 ----------
models = []
for r in runs:
    snap = read_json(Path(r["config_snapshot_path"])) or {}
    model_dir = snap.get("model_dir")
    status = snap.get("status", "UNKNOWN")

    if not model_dir or not Path(model_dir).exists():
        print(f"⚠️ skip {r['tag']} (no valid model_dir)")
        continue
    if status != "DONE":
        print(f"⚠️ skip {r['tag']} (status={status})")
        continue

    models.append({
        "tag": r["tag"],
        "run_name": r["run_name"],
        "model_dir": model_dir,
        "diameter": int(r["params"]["diameter"]),
    })

assert len(models) > 0, "No usable trained models found."

print("\n✅ Models to infer:")
for m in models:
    print("-", m["tag"], "| d=", m["diameter"], "|", m["model_dir"])

assert VAL_IMG_DIR.exists(), f"VAL_IMG_DIR not found: {VAL_IMG_DIR}"
assert VAL_GT_DIR.exists(),  f"VAL_GT_DIR not found: {VAL_GT_DIR}"

pred_dirs = {}
results = []

for m in models:
    tag = m["tag"]
    model_dir = m["model_dir"]
    diam = int(m["diameter"])

    savedir = RUN_OUT / f"pred_{tag}"
    savedir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(VAL_IMG_DIR),
        "--pretrained_model", str(model_dir),
        "--diameter", str(diam),
        "--flow_threshold", str(FLOW_TH),
        "--cellprob_threshold", str(CELLPROB_TH),
        "--use_gpu",
        "--save_tif",
        "--no_npy",
        "--savedir", str(savedir),
    ]

    rc, proc = sh(cmd, cwd=str(ROOT))
    print(f"✅ {tag} done (rc={rc}) -> {savedir}")

    pred_dirs[tag] = str(savedir)
    results.append({
        "tag": tag,
        "model_dir": model_dir,
        "diameter": diam,
        "rc": rc,
        "savedir": str(savedir),
        "stderr_tail": (proc.stderr or "")[-800:],
    })

manifest = {
    "time": stamp,
    "exp_dir": str(EXP),
    "run_out": str(RUN_OUT),
    "img_dir": str(VAL_IMG_DIR),
    "gt_dir": str(VAL_GT_DIR),
    "flow_th": FLOW_TH,
    "cellprob_th": CELLPROB_TH,
    "models": models,
    "pred_dirs": pred_dirs,
    "results": results,
}
MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n🎉 Inference outputs:", RUN_OUT)
print("📌 manifest:", MANIFEST)
print("\n✅ Cell 12 done.")
print("="*80)


# In[15]:


# Cell 13：Visual Battle Board（GT vs 多模型 Pred）
import re
import json
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("🖼 Cell 13 | Visual battle boards on new_val_proc")
print("="*80)

# ---------- 自动从上一个推理 cell 读 RUN_OUT ----------
assert "RUN_OUT" in globals(), "RUN_OUT 未定义：请先运行 Cell 12。"
RUN_OUT = Path(RUN_OUT).resolve()
MANIFEST = RUN_OUT / "manifest.json"
assert MANIFEST.exists(), f"manifest not found: {MANIFEST}"

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
IMG_DIR = Path(manifest["img_dir"]).resolve()
GT_DIR  = Path(manifest["gt_dir"]).resolve()
pred_dirs = {k: Path(v) for k, v in manifest["pred_dirs"].items()}
models = manifest["models"]

# ---------- 你想可视化多少张 ----------
N_VIS = 8

# ---------- 读取图片 + 归一化 ----------
def load_img(p: Path):
    x = tiff.imread(str(p))
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got {x.shape} for {p}")
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

def load_mask(p: Path):
    if p is None or (not Path(p).exists()):
        return None
    x = tiff.imread(str(p))
    x = np.squeeze(x)
    return x.astype(np.int32)

def mask_outline(m):
    if m is None:
        return None
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
    edge = (m != up) | (m != dn) | (m != lf) | (m != rt)
    edge &= (m > 0)
    return edge

# ---------- new_val 配对规则：按 stem 末尾数字 ----------
NUM_TAIL = re.compile(r"(\d+)$")
def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

imgs = list_files(IMG_DIR, IMG_EXTS)
gts  = list_files(GT_DIR,  GT_EXTS)

img_map = {}
for p in imgs:
    i = tail_id(p)
    if i is not None:
        img_map[i] = p

gt_map = {}
for p in gts:
    i = tail_id(p)
    if i is not None:
        gt_map[i] = p

common_ids = sorted(set(img_map.keys()) & set(gt_map.keys()))
assert len(common_ids) > 0, "No matched image<->GT pairs by tail number."

vis_ids = common_ids[:N_VIS]
print("✅ matched pairs:", len(common_ids))
print("🎯 visual sample ids:", vis_ids)

def find_pred_mask(pred_root: Path, img_path: Path):
    stem = img_path.stem
    cands = [
        pred_root / f"{stem}_cp_masks.tif",
        pred_root / f"{stem}_masks.tif",
        pred_root / f"{stem}_cp_masks.tiff",
        pred_root / f"{stem}_masks.tiff",
    ]
    for p in cands:
        if p.exists():
            return p
    glob_c = sorted(pred_root.glob(f"{stem}*masks*.tif"))
    if glob_c:
        return glob_c[0]
    glob_c = sorted(pred_root.glob(f"{stem}*masks*.tiff"))
    return glob_c[0] if glob_c else None

BOARD_DIR = RUN_OUT / "boards"
BOARD_DIR.mkdir(parents=True, exist_ok=True)

ncol = len(models)
assert ncol > 0, "No models in manifest."

for i in vis_ids:
    img_path = img_map[i]
    gt_path  = gt_map[i]

    img = load_img(img_path)
    gt = load_mask(gt_path)
    gt_edge = mask_outline(gt)

    fig, axes = plt.subplots(2, ncol, figsize=(4*ncol, 8))
    if ncol == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, m in enumerate(models):
        tag = m["tag"]
        pred_root = pred_dirs.get(tag)

        pred_path = find_pred_mask(pred_root, img_path) if pred_root else None
        pred = load_mask(pred_path) if pred_path else None
        pred_edge = mask_outline(pred)

        # Row 0: GT
        ax = axes[0, j]
        ax.imshow(img, cmap="gray")
        if gt_edge is not None:
            ax.imshow(gt_edge, alpha=0.8)
        ax.set_title(f"{tag} | GT")
        ax.axis("off")

        # Row 1: Pred
        ax = axes[1, j]
        ax.imshow(img, cmap="gray")
        if pred_edge is not None:
            ax.imshow(pred_edge, alpha=0.8)
        ax.set_title(f"{tag} | Pred")
        ax.axis("off")

        if pred_path is None:
            print(f"⚠️ pred not found: {tag} | sample={img_path.name} | pred_root={pred_root}")

    fig.suptitle(f"ID={i} | IMG={img_path.name} | GT={gt_path.name}", fontsize=14)
    out_png = BOARD_DIR / f"board_{img_path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("🖼 saved:", out_png)

print("\n🎉 Boards saved in:", BOARD_DIR)
print("\n✅ Cell 13 done.")
print("="*80)


# In[16]:


# Cell 14：Evaluate all models on new_val_proc (AP50 / Precision / Recall / F1)
import re, json
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from cellpose import metrics
from IPython.display import display

print("="*80)
print("🏆 Cell 14 | Instance-level evaluation on new_val_proc")
print("="*80)

assert "RUN_OUT" in globals(), "RUN_OUT 未定义：请先运行 Cell 12。"
RUN_OUT = Path(RUN_OUT).resolve()
assert RUN_OUT.exists(), f"RUN_OUT 不存在：{RUN_OUT}"

manifest_path = RUN_OUT / "manifest.json"
assert manifest_path.exists(), f"找不到 manifest.json：{manifest_path}"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

IMG_DIR = Path(manifest["img_dir"]).resolve()
GT_DIR  = Path(manifest["gt_dir"]).resolve()
FLOW_TH = manifest.get("flow_th", None)
CELLPROB_TH = manifest.get("cellprob_th", None)

print("RUN_OUT :", RUN_OUT)
print("IMG_DIR :", IMG_DIR, "exists=", IMG_DIR.exists())
print("GT_DIR  :", GT_DIR,  "exists=", GT_DIR.exists())
assert IMG_DIR.exists() and GT_DIR.exists()

# =========================
# 1) new_val 配对规则：按文件名末尾数字编号
# =========================
NUM_TAIL = re.compile(r"(\d+)$")

def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

imgs = list_files(IMG_DIR, IMG_EXTS)
gts  = list_files(GT_DIR,  GT_EXTS)
assert len(imgs) > 0 and len(gts) > 0, "IMG_DIR 或 GT_DIR 为空"

img_map = {}
for p in imgs:
    i = tail_id(p)
    if i is not None:
        img_map[i] = p

gt_map = {}
for p in gts:
    i = tail_id(p)
    if i is not None:
        gt_map[i] = p

common_ids = sorted(set(img_map.keys()) & set(gt_map.keys()))
assert len(common_ids) > 0, "图片与GT无法按末尾数字对齐（common_ids=0）"

print("✅ matched pairs:", len(common_ids), " (show first 5)")
for i in common_ids[:5]:
    print(" ", img_map[i].name, "<->", gt_map[i].name)

# =========================
# 2) 找预测 mask：在 pred_<tag>/ 下找 <img_stem>_*masks*.tif
# =========================
def find_pred_mask(pred_root: Path, img_path: Path):
    pred_root = Path(pred_root)
    stem = img_path.stem
    cands = [
        pred_root / f"{stem}_cp_masks.tif",
        pred_root / f"{stem}_masks.tif",
        pred_root / f"{stem}_cp_masks.tiff",
        pred_root / f"{stem}_masks.tiff",
    ]
    for p in cands:
        if p.exists():
            return p
    glob_c = sorted(pred_root.glob(f"{stem}*masks*.tif"))
    if glob_c:
        return glob_c[0]
    glob_c = sorted(pred_root.glob(f"{stem}*masks*.tiff"))
    return glob_c[0] if glob_c else None

def read_mask(path: Path):
    x = tiff.imread(str(path))
    x = np.squeeze(x)
    return x.astype(np.int32)

# =========================
# 3) 指标：AP50 + Precision/Recall/F1（实例级）
# =========================
def compute_ap50_prec_rec_f1(gt_mask: np.ndarray, pred_mask: np.ndarray):
    ap, tp, fp, fn = metrics.average_precision(gt_mask, pred_mask, threshold=[0.5])
    ap50 = float(ap[0])
    tp0, fp0, fn0 = float(tp[0]), float(fp[0]), float(fn[0])

    precision = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    recall    = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ap50, precision, recall, f1

# =========================
# 4) 扫描 RUN_OUT 下所有 pred_*，逐模型评估
# =========================
pred_dirs = sorted([p for p in RUN_OUT.glob("pred_*") if p.is_dir()])
assert len(pred_dirs) > 0, f"RUN_OUT 下没找到 pred_*：{RUN_OUT}"

diam_map = {}
for m in manifest.get("models", []):
    if isinstance(m, dict) and "tag" in m:
        if "diameter" in m and m["diameter"] is not None:
            diam_map[m["tag"]] = int(m["diameter"])

rows = []

for pred_root in pred_dirs:
    tag = pred_root.name.replace("pred_", "", 1)

    per_img = []
    n_missing = 0

    for i in common_ids:
        img_path = img_map[i]
        gt_path  = gt_map[i]

        pred_path = find_pred_mask(pred_root, img_path)
        if pred_path is None:
            n_missing += 1
            continue

        gt = read_mask(gt_path)
        pr = read_mask(pred_path)

        ap50, prec, rec, f1 = compute_ap50_prec_rec_f1(gt, pr)
        per_img.append((ap50, prec, rec, f1))

    if len(per_img) == 0:
        rows.append({
            "model_tag": tag,
            "diameter": diam_map.get(tag, None),
            "AP50": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "n_eval": 0,
            "n_pairs_total": len(common_ids),
            "n_missing_pred": n_missing,
            "flow_th": FLOW_TH,
            "cellprob_th": CELLPROB_TH,
            "pred_dir": str(pred_root),
        })
        continue

    arr = np.array(per_img, dtype=float)
    rows.append({
        "model_tag": tag,
        "diameter": diam_map.get(tag, None),
        "AP50": float(arr[:, 0].mean()),
        "Precision": float(arr[:, 1].mean()),
        "Recall": float(arr[:, 2].mean()),
        "F1": float(arr[:, 3].mean()),
        "n_eval": int(arr.shape[0]),
        "n_pairs_total": len(common_ids),
        "n_missing_pred": n_missing,
        "flow_th": FLOW_TH,
        "cellprob_th": CELLPROB_TH,
        "pred_dir": str(pred_root),
    })

df_eval = pd.DataFrame(rows)
df_eval = df_eval.sort_values(["AP50", "F1"], ascending=False).reset_index(drop=True)

out_csv = (RUN_OUT / "eval_newval_metrics.csv").resolve()
df_eval.to_csv(out_csv, index=False)

print("\n🏆 New-val ranking:")
display(df_eval[[
    "model_tag", "diameter", "AP50", "Precision", "Recall", "F1",
    "n_eval", "n_missing_pred"
]])
print("\n✅ saved:", out_csv)

print("\n✅ Cell 14 done.")
print("="*80)


# In[17]:


# Cell 15：融合训练汇总 + 最终实例级评估 -> final_merged_ranking.csv
import pandas as pd
from pathlib import Path
from IPython.display import display

print("="*80)
print("🧾 Cell 15 | Merge training summary with final evaluation")
print("="*80)

summary_csv = EXP / "summary_runs.csv"
eval_csv = RUN_OUT / "eval_newval_metrics.csv"

assert summary_csv.exists(), f"summary_runs.csv 不存在：{summary_csv}"
assert eval_csv.exists(), f"eval_newval_metrics.csv 不存在：{eval_csv}"

df_sum = pd.read_csv(summary_csv)
df_eval = pd.read_csv(eval_csv)

# 左边 tag，右边 model_tag
df_merged = df_sum.merge(
    df_eval,
    how="left",
    left_on="tag",
    right_on="model_tag"
)

# 更顺手的排序：最终还是看 AP50/F1
if "AP50" in df_merged.columns and "F1" in df_merged.columns:
    df_merged = df_merged.sort_values(["AP50", "F1"], ascending=False, na_position="last").reset_index(drop=True)

out_csv = EXP / "final_merged_ranking.csv"
df_merged.to_csv(out_csv, index=False)

show_cols = [
    "tag", "status",
    "diameter_x", "augment", "transformer",
    "train_batch_size", "n_epochs",
    "best_epoch_by_test_loss", "best_test_loss",
    "AP50", "Precision", "Recall", "F1",
    "model_dir"
]

# 某些列名 merge 后会变化，做兼容处理
show_cols = [c for c in show_cols if c in df_merged.columns]

print("✅ saved:", out_csv)
display(df_merged[show_cols])

print("\n✅ Cell 15 done.")
print("="*80)


# In[18]:


# Cell 16：打包交付 zip
import os, zipfile
from pathlib import Path
from datetime import datetime

print("="*80)
print("📦 Cell 16 | Bundle key outputs into zip")
print("="*80)

assert "RUN_OUT" in globals(), "RUN_OUT 未定义：请先运行 Cell 12。"
RUN_OUT = Path(RUN_OUT).resolve()
assert RUN_OUT.exists(), f"RUN_OUT 不存在：{RUN_OUT}"

zip_dir = RUN_OUT.parent / "zips"
zip_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = (zip_dir / f"{RUN_OUT.name}__bundle_{stamp}.zip").resolve()

def add_file(zf: zipfile.ZipFile, path: Path, arcroot: Path):
    rel = path.relative_to(arcroot)
    zf.write(path, rel.as_posix())

def add_dir(zf: zipfile.ZipFile, d: Path, arcroot: Path, patterns=None):
    if not d.exists():
        return 0
    n = 0
    if patterns:
        for pat in patterns:
            for p in d.rglob(pat):
                if p.is_file():
                    add_file(zf, p, arcroot)
                    n += 1
    else:
        for p in d.rglob("*"):
            if p.is_file():
                add_file(zf, p, arcroot)
                n += 1
    return n

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    n_added = 0

    # ---------- RUN_OUT 内关键内容 ----------
    for fname in ["manifest.json", "eval_newval_metrics.csv"]:
        p = RUN_OUT / fname
        if p.exists():
            add_file(zf, p, RUN_OUT)
            n_added += 1

    boards_dir = RUN_OUT / "boards"
    n_added += add_dir(zf, boards_dir, RUN_OUT, patterns=["*.png"])

    for d in sorted(RUN_OUT.glob("pred_*")):
        if d.is_dir():
            n_added += add_dir(zf, d, RUN_OUT)

    # ---------- EXP 根目录下的关键表 ----------
    extra_files = [
        EXP / "summary_runs.csv",
        EXP / "final_merged_ranking.csv",
        CFG_DIR / "PATHS.json",
        CFG_DIR / "RUNS.jsonl",
        CFG_DIR / "SWEEP_SUMMARY.json",
    ]
    # 这些文件放到 zip 的 exp_meta/ 下
    for p in extra_files:
        if p.exists():
            arcname = Path("exp_meta") / p.name
            zf.write(p, arcname.as_posix())
            n_added += 1

    # ---------- 打一份目录树 ----------
    tree_txt = RUN_OUT / "BUNDLE_TREE.txt"
    lines = []
    for p in sorted(RUN_OUT.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(RUN_OUT).as_posix()
        try:
            sz = p.stat().st_size
        except Exception:
            sz = -1
        lines.append(f"{sz:>12}  {rel}")
    tree_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    add_file(zf, tree_txt, RUN_OUT)
    n_added += 1

print("✅ Bundled files:", n_added)
print("📦 ZIP saved to:", zip_path)

print("\n✅ Cell 16 done.")
print("="*80)


# In[ ]:




