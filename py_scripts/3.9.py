#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 0：环境自检
import os, sys, json, time, shlex, subprocess, socket, re
from pathlib import Path
from datetime import datetime

print("="*80)
print("🧪 Cell 0 | Environment sanity check")
print("="*80)

print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])
print("CWD:", os.getcwd())
print("User:", os.environ.get("USER", "unknown"))
print("HOSTNAME:", socket.gethostname())

host = socket.gethostname().lower()
if "login" in host or "a2n" in host or "log" in host:
    print("\n⚠️ 当前看起来像登录节点，训练前请确认你在计算节点。")
else:
    print("\n✅ 节点看起来不像登录节点。")

def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return 999, "", repr(e)

try:
    import cellpose
    print("\n✅ cellpose import OK | version(attr):", getattr(cellpose, "__version__", "unknown"))
except Exception as e:
    print("\n❌ cellpose import FAILED:", repr(e))
    raise

try:
    from importlib.metadata import version
    print("cellpose_version(meta):", version("cellpose"))
    print("torch_version:", version("torch"))
except Exception as e:
    print("⚠️ metadata version read failed:", repr(e))

code, out, err = _run(["nvidia-smi", "-L"])
if code == 0 and out:
    print("\n🟢 GPU detected:")
    print(out)
else:
    print("\n⚠️ nvidia-smi not found or no GPU visible.")
    if err:
        print("stderr:", err[:300])

print("\n✅ Cell 0 done.")
print("="*80)


# In[2]:


# Cell 1：路径初始化
from pathlib import Path
from datetime import datetime
import json

print("="*80)
print("🧱 Cell 1 | Paths & experiment directory init")
print("="*80)

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
DATA_ROOT = (ROOT / "Cellpose2TrainDataset").resolve()

TRAIN_DIR = (DATA_ROOT / "trainset_ft_os3").resolve()

NEWVAL_ROOT = (DATA_ROOT / "new_val_proc").resolve()
VAL_IMG_DIR = (NEWVAL_ROOT / "images_sp_lcn").resolve()
VAL_GT_DIR  = (NEWVAL_ROOT / "ground").resolve()

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

EXP_NAME = f"exp_20260309_grid9_trans_wd_epochs_{STAMP}"

EXP_DIR = (ROOT / "runs" / EXP_NAME).resolve()

LOG_DIR     = EXP_DIR / "logs"
MET_DIR     = EXP_DIR / "metrics"
CFG_DIR     = EXP_DIR / "config"
INFER_DIR   = EXP_DIR / "infer"
EVAL_DIR    = EXP_DIR / "eval"
EXPORT_DIR  = EXP_DIR / "exports"
DELIV_DIR   = EXP_DIR / "delivery"
VALVIEW_DIR = EXP_DIR / "valview_newproc"

for d in [LOG_DIR, MET_DIR, CFG_DIR, INFER_DIR, EVAL_DIR, EXPORT_DIR, DELIV_DIR, VALVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    "note": "3x3 confirmation sweep around d=14, transformer=True, lr=1e-4, varying weight_decay and n_epochs.",
}
(PATH_INDEX_PATH := (CFG_DIR / "PATHS.json")).write_text(
    json.dumps(PATH_INDEX, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("ROOT       :", ROOT)
print("TRAIN_DIR  :", TRAIN_DIR, "| exists:", TRAIN_DIR.exists())
print("VAL_IMG_DIR:", VAL_IMG_DIR, "| exists:", VAL_IMG_DIR.exists())
print("VAL_GT_DIR :", VAL_GT_DIR,  "| exists:", VAL_GT_DIR.exists())
print("EXP_DIR    :", EXP_DIR)
print("VALVIEW_DIR:", VALVIEW_DIR)
print("PATHS.json :", PATH_INDEX_PATH)

assert ROOT.exists()
assert TRAIN_DIR.exists()
assert VAL_IMG_DIR.exists()
assert VAL_GT_DIR.exists()

print("\n✅ Cell 1 done.")
print("="*80)


# In[3]:


# Cell 2：构建训练期 test_dir 视图（valview_newproc）
import re, json, shutil, os
from pathlib import Path
from datetime import datetime

print("="*80)
print("🧩 Cell 2 | Build training-time val view from new_val_proc")
print("="*80)

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]
NUM_TAIL = re.compile(r"(\d+)$")

def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

imgs_all = list_files(VAL_IMG_DIR, IMG_EXTS)
gts_all  = list_files(VAL_GT_DIR, GT_EXTS)

img_map = {tail_id(p): p for p in imgs_all if tail_id(p) is not None}
gt_map  = {tail_id(p): p for p in gts_all if tail_id(p) is not None}
common_ids = sorted(set(img_map.keys()) & set(gt_map.keys()))
assert len(common_ids) > 0, "No matched image/GT ids found."

if VALVIEW_DIR.exists():
    for p in VALVIEW_DIR.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

VALVIEW_DIR.mkdir(parents=True, exist_ok=True)

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

VAL_MANIFEST_PATH = CFG_DIR / "valview_manifest.json"
VAL_MANIFEST_PATH.write_text(
    json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_img_dir": str(VAL_IMG_DIR),
        "source_gt_dir": str(VAL_GT_DIR),
        "valview_dir": str(VALVIEW_DIR),
        "n_pairs": len(manifest_rows),
        "pairs": manifest_rows,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("✅ common matched ids:", len(common_ids))
print("VALVIEW_DIR:", VALVIEW_DIR)
print("Manifest:", VAL_MANIFEST_PATH)
for p in sorted(VALVIEW_DIR.iterdir())[:10]:
    print(" ", p.name, "->", os.readlink(p) if p.is_symlink() else "(not symlink)")

print("\n✅ Cell 2 done.")
print("="*80)


# In[4]:


# Cell 3：定义 3x3 确认性 sweep（wd × epochs）
import json
from pathlib import Path

print("="*80)
print("🧪 Cell 3 | Define 3x3 transformer confirmation sweep")
print("="*80)

BEST_CELLPROB_TH = -0.5
BEST_FLOW_TH = 0.6

BASE = {
    "train_dir": str(TRAIN_DIR),
    "test_dir": str(VALVIEW_DIR),
    "mask_filter": "_masks",
    "pretrained_model": "cpsam",
    "save_every": 5,
    "use_gpu": True,
    "verbose": True,
    "bsize": 256,
    "augment": True,
    "train_batch_size": 32,
    "transformer": True,
    "diameter": 14,
    "learning_rate": 1e-4,
}

SWEEP = [
    dict(tag="G00_wd8e3_e120",  weight_decay=8e-3,  n_epochs=120),
    dict(tag="G01_wd8e3_e140",  weight_decay=8e-3,  n_epochs=140),
    dict(tag="G02_wd8e3_e160",  weight_decay=8e-3,  n_epochs=160),

    dict(tag="G10_wd1e2_e120",  weight_decay=1e-2,  n_epochs=120),
    dict(tag="G11_wd1e2_e140",  weight_decay=1e-2,  n_epochs=140),
    dict(tag="G12_wd1e2_e160",  weight_decay=1e-2,  n_epochs=160),

    dict(tag="G20_wd2e2_e120",  weight_decay=2e-2,  n_epochs=120),
    dict(tag="G21_wd2e2_e140",  weight_decay=2e-2,  n_epochs=140),
    dict(tag="G22_wd2e2_e160",  weight_decay=2e-2,  n_epochs=160),
]

print("BEST_CELLPROB_TH:", BEST_CELLPROB_TH)
print("BEST_FLOW_TH    :", BEST_FLOW_TH)
print("Sweep size:", len(SWEEP))
for v in SWEEP:
    print(v)

print("\n✅ Cell 3 done.")
print("="*80)


# In[5]:


# Cell 4：生成 RUNS.jsonl（增强版：预留 best/final checkpoint 字段）
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("📝 Cell 4 | Commit refinement runs into RUNS.jsonl (best-checkpoint ready)")
print("="*80)

RUNS = []
RUN_INDEX_PATH = CFG_DIR / "RUNS.jsonl"

if RUN_INDEX_PATH.exists() and RUN_INDEX_PATH.stat().st_size > 0:
    raise RuntimeError(
        f"RUNS.jsonl already exists and is non-empty:\n{RUN_INDEX_PATH}\n"
        "为避免重复提交 run，请新建实验目录后再运行。"
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
        "params": p,
        "log_path": str(LOG_DIR / f"train_{run_name}.log"),
        "pid_path": str(LOG_DIR / f"train_{run_name}.pid"),
        "metrics_path": str(MET_DIR / f"metrics_{run_name}.csv"),
        "cmd_path": str(CFG_DIR / f"cmd_{run_name}.txt"),
        "config_snapshot_path": str(CFG_DIR / f"config_{run_name}.json"),

        "status": "NOT_STARTED",
        "model_dir": None,
        "final_model_path": None,
        "best_model_path": None,
        "best_epoch_by_test_loss": None,
        "best_test_loss": None,
        "last_epoch_logged": None,
        "last_test_loss": None,
        "best_model_found": False,
        "eval_model_path": None,
        "eval_model_type": None,
        "available_checkpoints": [],
        "checkpoint_strategy": {
            "save_every": p.get("save_every", None),
            "save_each": True,
            "epoch_index_base": 0,
            "note": "best_epoch_by_test_loss and last_epoch_logged are 0-based indices from training logs."
        },

        "note": p.get("note", ""),
    }

    Path(ctx["config_snapshot_path"]).write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    with open(RUN_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ctx, ensure_ascii=False) + "\n")

    return ctx

for v in SWEEP:
    ctx = commit_run(BASE, v)
    RUNS.append(ctx)

SWEEP_SUMMARY_PATH = CFG_DIR / "SWEEP_SUMMARY.json"
SWEEP_SUMMARY_PATH.write_text(
    json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "exp_dir": str(EXP_DIR),
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(VALVIEW_DIR),
        "best_cellprob_threshold": BEST_CELLPROB_TH,
        "best_flow_threshold": BEST_FLOW_TH,
        "n_runs": len(RUNS),
        "runs": RUNS,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"✅ 已生成 {len(RUNS)} 个 runs")
print("RUNS index:", RUN_INDEX_PATH)
print("SWEEP summary:", SWEEP_SUMMARY_PATH)
for r in RUNS:
    print("-", r["run_name"])

print("\n✅ Cell 4 done.")
print("="*80)


# In[6]:


# Cell 5：串行训练执行器（增强版：保存并登记阶段 checkpoint）
import json, os, time, shlex, subprocess, traceback, re
from pathlib import Path
from datetime import datetime

print("="*80)
print("🚀 Cell 5 | Serial training executor (best-checkpoint ready)")
print("="*80)

PATHS_JSON = CFG_DIR / "PATHS.json"
RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
assert PATHS_JSON.exists()
assert RUNS_JSONL.exists()

RUNS = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

PAT_SAVED_LIST = [
    re.compile(r"model trained and saved to\s+(?P<dir>/\S+)"),
    re.compile(r"saving model to\s+(?P<dir>/\S+)"),
    re.compile(r"saving network parameters to\s+(?P<dir>/\S+)"),
]

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
        "--save_each",
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
    final_model_path = snap.get("final_model_path") or snap.get("model_dir")
    return bool(final_model_path) and Path(final_model_path).exists()

def capture_final_model_from_log(log_path: Path):
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in reversed(lines[-3000:]):
        for pat in PAT_SAVED_LIST:
            m = pat.search(line)
            if m:
                return m.group("dir")
    return None

def scan_available_checkpoints(final_model_path: Path):
    out = []
    if final_model_path is None:
        return out
    final_model_path = Path(final_model_path)
    parent = final_model_path.parent
    stem = final_model_path.name

    if not parent.exists():
        return out

    if final_model_path.exists():
        out.append(str(final_model_path.resolve()))

    cands = sorted(parent.glob(f"{stem}_epoch_*"))
    for p in cands:
        if p.exists():
            out.append(str(p.resolve()))

    return out

def mark_snapshot(ctx: dict, **kwargs):
    snap_path = Path(ctx["config_snapshot_path"])
    snap = read_json(snap_path) or {}
    snap.update(kwargs)
    write_json(snap_path, snap)

POLL_S = 60
RUN_ONLY_TAGS = None   # 先试跑可写 ["G11_wd1e2_e140"]
STOP_ON_FAILURE = False

selected_runs = [r for r in RUNS if (RUN_ONLY_TAGS is None or r["tag"] in RUN_ONLY_TAGS)]
print("Runs to process:", len(selected_runs))

for idx, ctx in enumerate(selected_runs, 1):
    print("\n" + "="*100)
    print(f"[{idx}/{len(selected_runs)}] RUN: {ctx['run_name']}")
    print("="*100)

    if already_done(ctx):
        print("✅ already done -> skip")
        mark_snapshot(ctx, status="DONE")
        continue

    cmd = build_cmd(ctx)
    cmd_text = cmd_to_text(cmd)

    cmd_path = Path(ctx["cmd_path"])
    log_path = Path(ctx["log_path"])
    pid_path = Path(ctx["pid_path"])

    cmd_path.write_text(cmd_text + "\n", encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("CMD:")
    print(cmd_text)
    print("\nLOG:", log_path)

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

            print("🚀 Started | PID:", proc.pid)

            while True:
                ret = proc.poll()
                if ret is not None:
                    print("✅ process finished with return code:", ret)
                    break
                print(f"⏳ running... sleep {POLL_S}s")
                time.sleep(POLL_S)

        pid_path.unlink(missing_ok=True)

        final_model = capture_final_model_from_log(log_path)
        if final_model and Path(final_model).exists():
            ckpts = scan_available_checkpoints(Path(final_model))
            mark_snapshot(
                ctx,
                status="DONE",
                model_dir=final_model,
                final_model_path=final_model,
                finished_at=datetime.now().isoformat(timespec="seconds"),
                return_code=ret,
                available_checkpoints=ckpts,
                checkpoint_strategy={
                    "save_every": ctx["params"].get("save_every", None),
                    "save_each": True,
                    "epoch_index_base": 0,
                    "note": "0-based epoch index from logs; intermediate checkpoints saved with _epoch_XXXX suffix."
                }
            )
            print("🏁 final_model_path:", final_model)
            print("🧩 checkpoints found:", len(ckpts))
            for p in ckpts[:8]:
                print("   ", p)
        else:
            mark_snapshot(
                ctx,
                status="FAILED",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                return_code=ret,
                note="training ended but final_model_path not captured; inspect log",
            )
            print("⚠️ final_model_path 未捕获，请检查 log。")
            if STOP_ON_FAILURE:
                raise RuntimeError(f"Run failed: {ctx['tag']}")

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

print("\n🎉 Cell 5 finished.")
print("="*80)


# In[7]:


# Cell 6：训练汇总表（增强版：自动选择 best checkpoint）
import json, re
from pathlib import Path
import pandas as pd

print("="*80)
print("📊 Cell 6 | Parse training logs -> summary_runs.csv (best-checkpoint enabled)")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
OUT_CSV = EXP_DIR / "summary_runs.csv"

runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

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

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def resolve_best_model_path(final_model_path: str, best_epoch: int):
    if final_model_path is None:
        return None, "no_final_model"

    fmp = Path(final_model_path)
    if not fmp.exists():
        return None, "final_model_missing"

    ckpt = Path(str(fmp) + f"_epoch_{int(best_epoch):04d}")
    if ckpt.exists():
        return str(ckpt.resolve()), "best_ckpt"

    return str(fmp.resolve()), "final_model_fallback"

rows = []
for r in runs:
    snap_path = Path(r["config_snapshot_path"])
    snap = read_json(snap_path) or {}

    final_model_path = snap.get("final_model_path") or snap.get("model_dir")
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

    best_epoch = best_test = last_epoch = last_test = None
    final_epoch_logged = False
    best_model_path = None
    best_model_source = "unresolved"
    best_model_found = False

    if metrics:
        dfm = pd.DataFrame(metrics).sort_values("epoch")
        best_idx = dfm["test_loss"].idxmin()

        best_epoch = int(dfm.loc[best_idx, "epoch"])
        best_test  = float(dfm.loc[best_idx, "test_loss"])
        last_epoch = int(dfm.iloc[-1]["epoch"])
        last_test  = float(dfm.iloc[-1]["test_loss"])

        n_epochs_cfg = int(r["params"].get("n_epochs", 0))
        final_epoch_logged = (last_epoch == (n_epochs_cfg - 1))

        best_model_path, best_model_source = resolve_best_model_path(final_model_path, best_epoch)
        best_model_found = (best_model_source == "best_ckpt")

    snap.update({
        "best_epoch_by_test_loss": best_epoch,
        "best_test_loss": best_test,
        "last_epoch_logged": last_epoch,
        "last_test_loss": last_test,
        "best_model_path": best_model_path,
        "best_model_found": best_model_found,
        "best_model_source": best_model_source,
        "final_model_path": final_model_path,
        "epoch_index_base": 0,
        "final_epoch_logged": final_epoch_logged,
    })
    write_json(snap_path, snap)

    p = r["params"]
    rows.append({
        "tag": r["tag"],
        "run_name": r["run_name"],
        "status": status,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "best_model_found": best_model_found,
        "best_model_source": best_model_source,
        "diameter": p.get("diameter"),
        "lr": p.get("learning_rate"),
        "weight_decay": p.get("weight_decay"),
        "train_batch_size": p.get("train_batch_size"),
        "n_epochs": p.get("n_epochs"),
        "augment": p.get("augment", False),
        "transformer": p.get("transformer", False),
        "best_epoch_by_test_loss": best_epoch,
        "best_test_loss": best_test,
        "last_epoch_logged": last_epoch,
        "last_test_loss": last_test,
        "epoch_index_base": 0,
        "final_epoch_logged": final_epoch_logged,
        "log_path": str(log_path),
    })

df = pd.DataFrame(rows).sort_values("tag")
df.to_csv(OUT_CSV, index=False)

print("✅ Wrote:", OUT_CSV)
print("\n⚠️ 注意：best_epoch_by_test_loss / last_epoch_logged 都是 0-based epoch index。")
print("    所以 n_epochs=100/120/140/160 时，如果日志按每10轮打印，看到最后是 90/110/130/150 是正常的。")
display(df)

print("\n✅ Cell 6 done.")
print("="*80)


# In[8]:


# Cell 7：统一推理（增强版：优先使用 best checkpoint）
import os, shlex, subprocess, time, json
from pathlib import Path

print("="*80)
print("🤖 Cell 7 | Unified inference with fixed best thresholds (prefer best checkpoint)")
print("="*80)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

OUT_ROOT = (EXP_DIR / "battle_refine").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
RUN_OUT = OUT_ROOT / f"refine_{stamp}"
RUN_OUT.mkdir(parents=True, exist_ok=True)

FLOW_TH = BEST_FLOW_TH
CELLPROB_TH = BEST_CELLPROB_TH

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def sh(cmd, cwd=None):
    print("\n$", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout.strip()[:1500])
    if r.stderr.strip():
        print("stderr:", r.stderr.strip()[:1500])
    return r.returncode, r

models = []
for r in runs:
    snap = read_json(Path(r["config_snapshot_path"])) or {}
    status = snap.get("status", "UNKNOWN")
    if status != "DONE":
        continue

    best_model_path = snap.get("best_model_path")
    final_model_path = snap.get("final_model_path") or snap.get("model_dir")

    chosen_model_path = None
    chosen_model_type = None

    if best_model_path and Path(best_model_path).exists():
        chosen_model_path = best_model_path
        chosen_model_type = "best_ckpt"
    elif final_model_path and Path(final_model_path).exists():
        chosen_model_path = final_model_path
        chosen_model_type = "final_model_fallback"
    else:
        continue

    snap["eval_model_path"] = chosen_model_path
    snap["eval_model_type"] = chosen_model_type
    write_json(Path(r["config_snapshot_path"]), snap)

    models.append({
        "tag": r["tag"],
        "run_name": r["run_name"],
        "eval_model_path": chosen_model_path,
        "eval_model_type": chosen_model_type,
        "diameter": int(r["params"]["diameter"]),
        "transformer": bool(r["params"].get("transformer", False)),
        "best_epoch_by_test_loss": snap.get("best_epoch_by_test_loss"),
    })

assert len(models) > 0, "No trained models / usable checkpoints found."

pred_dirs = {}
results = []

for m in models:
    tag = m["tag"]
    model_path = m["eval_model_path"]
    diam = int(m["diameter"])

    savedir = RUN_OUT / f"pred_{tag}"
    savedir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(VAL_IMG_DIR),
        "--pretrained_model", str(model_path),
        "--diameter", str(diam),
        "--flow_threshold", str(FLOW_TH),
        "--cellprob_threshold", str(CELLPROB_TH),
        "--use_gpu",
        "--save_tif",
        "--no_npy",
        "--savedir", str(savedir),
    ]

    rc, proc = sh(cmd, cwd=str(ROOT))
    pred_dirs[tag] = str(savedir)
    results.append({
        "tag": tag,
        "eval_model_path": model_path,
        "eval_model_type": m["eval_model_type"],
        "best_epoch_by_test_loss": m["best_epoch_by_test_loss"],
        "diameter": diam,
        "transformer": m["transformer"],
        "rc": rc,
        "savedir": str(savedir),
    })
    print(f"✅ {tag} done ->", savedir, "|", m["eval_model_type"])

MANIFEST = RUN_OUT / "manifest.json"
MANIFEST.write_text(
    json.dumps({
        "time": stamp,
        "exp_dir": str(EXP_DIR),
        "run_out": str(RUN_OUT),
        "img_dir": str(VAL_IMG_DIR),
        "gt_dir": str(VAL_GT_DIR),
        "flow_th": FLOW_TH,
        "cellprob_th": CELLPROB_TH,
        "models": models,
        "pred_dirs": pred_dirs,
        "results": results,
        "note": "Inference prefers best_model_path; falls back to final_model_path if best checkpoint is unavailable."
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("RUN_OUT:", RUN_OUT)
print("MANIFEST:", MANIFEST)
print("\n✅ Cell 7 done.")
print("="*80)


# In[9]:


# Cell 8：实例级评估
import re, json
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from cellpose import metrics

print("="*80)
print("🏁 Cell 8 | Evaluate refinement sweep")
print("="*80)

assert "RUN_OUT" in globals(), "RUN_OUT 未定义，请先跑 Cell 7。"
RUN_OUT = Path(RUN_OUT).resolve()
MANIFEST = RUN_OUT / "manifest.json"
assert MANIFEST.exists()

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
IMG_DIR = Path(manifest["img_dir"]).resolve()
GT_DIR  = Path(manifest["gt_dir"]).resolve()

NUM_TAIL = re.compile(r"(\d+)$")
def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

imgs = list_files(IMG_DIR, [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
gts  = list_files(GT_DIR,  [".tif", ".tiff", ".png"])
img_map = {tail_id(p): p for p in imgs if tail_id(p) is not None}
gt_map  = {tail_id(p): p for p in gts  if tail_id(p) is not None}
common_ids = sorted(set(img_map.keys()) & set(gt_map.keys()))
assert len(common_ids) > 0

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

def read_mask(path: Path):
    x = tiff.imread(str(path))
    x = np.squeeze(x)
    return x.astype(np.int32)

def compute_ap50_prec_rec_f1(gt_mask, pred_mask):
    ap, tp, fp, fn = metrics.average_precision(gt_mask, pred_mask, threshold=[0.5])
    ap50 = float(ap[0])
    tp0, fp0, fn0 = float(tp[0]), float(fp[0]), float(fn[0])
    precision = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    recall    = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ap50, precision, recall, f1

rows = []
pred_dirs = sorted([p for p in RUN_OUT.glob("pred_*") if p.is_dir()])

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
        per_img.append(compute_ap50_prec_rec_f1(gt, pr))

    if len(per_img) == 0:
        rows.append({
            "model_tag": tag,
            "AP50": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan,
            "n_eval": 0, "n_missing_pred": n_missing,
            "pred_dir": str(pred_root),
        })
    else:
        arr = np.array(per_img, dtype=float)
        rows.append({
            "model_tag": tag,
            "AP50": float(arr[:,0].mean()),
            "Precision": float(arr[:,1].mean()),
            "Recall": float(arr[:,2].mean()),
            "F1": float(arr[:,3].mean()),
            "n_eval": int(arr.shape[0]),
            "n_missing_pred": n_missing,
            "pred_dir": str(pred_root),
        })

df_eval = pd.DataFrame(rows).sort_values(["AP50", "F1"], ascending=False).reset_index(drop=True)
EVAL_CSV = RUN_OUT / "eval_refine_metrics.csv"
df_eval.to_csv(EVAL_CSV, index=False)

print("✅ saved:", EVAL_CSV)
display(df_eval)
print("\n✅ Cell 8 done.")
print("="*80)


# In[10]:


# Cell 9：融合总榜（增强版：标明评估到底使用 best 还是 final）
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("🧾 Cell 9 | Merge refinement summary with final evaluation (best-checkpoint aware)")
print("="*80)

summary_csv = EXP_DIR / "summary_runs.csv"
eval_csv = RUN_OUT / "eval_refine_metrics.csv"
runs_jsonl = CFG_DIR / "RUNS.jsonl"

assert summary_csv.exists(), f"summary_runs.csv 不存在：{summary_csv}"
assert eval_csv.exists(), f"eval_refine_metrics.csv 不存在：{eval_csv}"
assert runs_jsonl.exists(), f"RUNS.jsonl 不存在：{runs_jsonl}"

df_sum = pd.read_csv(summary_csv)
df_eval = pd.read_csv(eval_csv)

runs = [json.loads(l) for l in runs_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]

extra_rows = []
for r in runs:
    snap_path = Path(r["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
    extra_rows.append({
        "tag": r["tag"],
        "final_model_path": snap.get("final_model_path"),
        "best_model_path": snap.get("best_model_path"),
        "best_model_found": snap.get("best_model_found", False),
        "best_model_source": snap.get("best_model_source"),
        "best_epoch_by_test_loss": snap.get("best_epoch_by_test_loss"),
        "best_test_loss": snap.get("best_test_loss"),
        "last_epoch_logged": snap.get("last_epoch_logged"),
        "last_test_loss": snap.get("last_test_loss"),
        "eval_model_path": snap.get("eval_model_path"),
        "eval_model_type": snap.get("eval_model_type"),
    })

df_extra = pd.DataFrame(extra_rows)

base_cols = [c for c in df_sum.columns if c not in df_extra.columns or c == "tag"]
df_sum2 = df_sum[base_cols].merge(df_extra, how="left", on="tag")

df_merged = df_sum2.merge(df_eval, how="left", left_on="tag", right_on="model_tag")
df_merged = df_merged.sort_values(["AP50", "F1"], ascending=False, na_position="last").reset_index(drop=True)

out_csv = EXP_DIR / "final_refine_merged_ranking.csv"
df_merged.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_merged)

print("\n✅ Cell 9 done.")
print("="*80)


# In[11]:


# Cell 10：打包所有关键结果
import os, zipfile
from pathlib import Path
from datetime import datetime

print("="*80)
print("📦 Cell 10 | Bundle refinement experiment")
print("="*80)

zip_dir = EXP_DIR / "zips"
zip_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = (zip_dir / f"{EXP_NAME}__bundle_{stamp}.zip").resolve()

def add_file(zf: zipfile.ZipFile, path: Path, arcname: str):
    zf.write(path, arcname)

def add_dir(zf: zipfile.ZipFile, d: Path, prefix: str):
    if not d.exists():
        return 0
    n = 0
    for p in d.rglob("*"):
        if p.is_file():
            rel = p.relative_to(d)
            zf.write(p, str(Path(prefix) / rel))
            n += 1
    return n

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    n_added = 0

    n_added += add_dir(zf, CFG_DIR, "config")
    n_added += add_dir(zf, LOG_DIR, "logs")
    n_added += add_dir(zf, MET_DIR, "metrics")

    if "RUN_OUT" in globals() and Path(RUN_OUT).exists():
        n_added += add_dir(zf, Path(RUN_OUT), "refine_outputs")

    for p in [
        EXP_DIR / "summary_runs.csv",
        EXP_DIR / "final_refine_merged_ranking.csv",
    ]:
        if p.exists():
            add_file(zf, p, f"summary/{p.name}")
            n_added += 1

print("✅ Bundled files:", n_added)
print("📦 ZIP saved to:", zip_path)
print("\n✅ Cell 10 done.")
print("="*80)


# In[16]:


# Cell 11：从当前总榜自动锁定冠军模型，并配置 3D 路径（仅冠军模型，不跑 baseline）
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("🏆 Cell 11 | Load champion model from merged ranking & setup 3D paths (champion only)")
print("="*80)

# ========= 1. 读取当前 2D 总榜 =========
RANK_CSV = EXP_DIR / "final_refine_merged_ranking.csv"
assert RANK_CSV.exists(), f"未找到总榜文件: {RANK_CSV}"

df_rank = pd.read_csv(RANK_CSV)
assert len(df_rank) > 0, "总榜为空，无法选择冠军模型。"

champ = df_rank.iloc[0].to_dict()

CHAMP_TAG = str(champ["tag"])
CHAMP_RUN_NAME = str(champ["run_name"])
CHAMP_MODEL_PATH = str(champ["eval_model_path"])
CHAMP_MODEL_TYPE = str(champ["eval_model_type"])
CHAMP_DIAMETER_2D = int(champ["diameter"])

assert Path(CHAMP_MODEL_PATH).exists(), f"冠军模型路径不存在: {CHAMP_MODEL_PATH}"

print("✅ 当前冠军模型已锁定：")
print("CHAMP_TAG       :", CHAMP_TAG)
print("CHAMP_RUN_NAME  :", CHAMP_RUN_NAME)
print("CHAMP_MODEL_PATH:", CHAMP_MODEL_PATH)
print("CHAMP_MODEL_TYPE:", CHAMP_MODEL_TYPE)
print("2D diameter     :", CHAMP_DIAMETER_2D)

# ========= 2. 3D 数据路径配置 =========
RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()

assert RAW_3D_STACK_PATH.exists(), f"3D 原始数据不存在: {RAW_3D_STACK_PATH}"

# 3D 推理参数
TRUE_3D_DIAMETER = 8
DO_3D = True
Z_AXIS = 0
STITCH_THRESHOLD = 0.5
BATCH_SIZE_3D = 16

# ========= 3. 输出目录 =========
THREED_ROOT = (EXP_DIR / "3d_fullbrain").resolve()
THREED_ROOT.mkdir(parents=True, exist_ok=True)

CHAMP_3D_DIR = THREED_ROOT / f"Champion_{CHAMP_TAG}"
CHAMP_3D_DIR.mkdir(parents=True, exist_ok=True)

THREED_CONFIG = {
    "created_at": pd.Timestamp.now().isoformat(),
    "mode": "champion_only",
    "rank_csv": str(RANK_CSV),
    "champion": {
        "tag": CHAMP_TAG,
        "run_name": CHAMP_RUN_NAME,
        "eval_model_path": CHAMP_MODEL_PATH,
        "eval_model_type": CHAMP_MODEL_TYPE,
        "diameter_2d": CHAMP_DIAMETER_2D,
    },
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "true_3d_diameter": TRUE_3D_DIAMETER,
    "do_3d": DO_3D,
    "z_axis": Z_AXIS,
    "stitch_threshold": STITCH_THRESHOLD,
    "batch_size_3d": BATCH_SIZE_3D,
    "output_root": str(THREED_ROOT),
    "champion_dir": str(CHAMP_3D_DIR),
    "note": "Champion model is selected automatically from final_refine_merged_ranking.csv top-1 row. Baseline inference is skipped."
}

THREED_CONFIG_PATH = THREED_ROOT / "3d_run_config.json"
THREED_CONFIG_PATH.write_text(
    json.dumps(THREED_CONFIG, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nRAW_3D_STACK_PATH:", RAW_3D_STACK_PATH)
print("THREED_ROOT      :", THREED_ROOT)
print("CHAMP_3D_DIR     :", CHAMP_3D_DIR)
print("CONFIG saved to  :", THREED_CONFIG_PATH)
print("\n✅ Cell 11 done.")
print("="*80)


# In[17]:


# Cell 12：仅运行当前冠军模型 3D 推理（跳过 baseline）
import time
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from cellpose import models, io

print("="*80)
print("🚀 Cell 12 | Run 3D inference: champion only (skip baseline)")
print("="*80)

io.logger_setup()

def run_3d_inference_once(model_path, out_dir: Path, run_label: str):
    """
    对完整 3D stack 跑一次推理。
    如果结果已存在，则直接复用，不重复跑。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_save_path = out_dir / "full_brain_3d_masks.tif"
    stats_save_path = out_dir / "stats_3d.json"

    # 如果已经存在结果，直接复用
    if mask_save_path.exists() and stats_save_path.exists():
        print(f"✅ [{run_label}] 已存在结果，直接复用：{mask_save_path}")
        stats = json.loads(stats_save_path.read_text(encoding="utf-8"))
        return mask_save_path, stats

    print(f"\n🚀 开始 3D 推理: {run_label}")
    print("Model     :", model_path)
    print("Output dir:", out_dir)

    t0 = time.time()

    stack = tiff.imread(str(RAW_3D_STACK_PATH))
    print("3D stack shape:", stack.shape, "| dtype:", stack.dtype)

    model = models.CellposeModel(
        gpu=True,
        pretrained_model=model_path
    )

    masks, flows, styles = model.eval(
        stack,
        diameter=TRUE_3D_DIAMETER,
        do_3D=DO_3D,
        stitch_threshold=STITCH_THRESHOLD,
        z_axis=Z_AXIS,
        batch_size=BATCH_SIZE_3D,
        progress=True
    )

    masks = np.asarray(masks)
    tiff.imwrite(str(mask_save_path), masks.astype(np.uint32))

    total_cells = int(len(np.unique(masks)) - 1)
    elapsed_s = float(time.time() - t0)

    stats = {
        "run_label": run_label,
        "model_path": str(model_path),
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_save_path": str(mask_save_path),
        "shape": list(masks.shape),
        "dtype": str(masks.dtype),
        "total_cells": total_cells,
        "elapsed_s": elapsed_s,
        "true_3d_diameter": TRUE_3D_DIAMETER,
        "do_3d": DO_3D,
        "stitch_threshold": STITCH_THRESHOLD,
        "z_axis": Z_AXIS,
        "batch_size_3d": BATCH_SIZE_3D,
    }

    stats_save_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"✅ [{run_label}] 推理完成")
    print("Mask saved :", mask_save_path)
    print("Total cells:", total_cells)
    print("Elapsed(s) :", round(elapsed_s, 2))

    return mask_save_path, stats


# ========= 仅跑 champion =========
champ_mask_path, champ_stats = run_3d_inference_once(
    model_path=CHAMP_MODEL_PATH,
    out_dir=CHAMP_3D_DIR,
    run_label=f"Champion_{CHAMP_TAG}"
)

# ========= 保存本次 3D 运行摘要 =========
summary = {
    "mode": "champion_only",
    "champion": champ_stats,
    "champion_tag": CHAMP_TAG,
    "champion_run_name": CHAMP_RUN_NAME,
    "champion_model_type": CHAMP_MODEL_TYPE,
}

THREED_SUMMARY_PATH = THREED_ROOT / "3d_champion_summary.json"
THREED_SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n📌 3D summary saved to:", THREED_SUMMARY_PATH)
print("Champion total cells :", champ_stats["total_cells"])
print("Champion mask path   :", champ_mask_path)

print("\n✅ Cell 12 done.")
print("="*80)


# In[18]:


# Cell 13：Champion-only 3D 总细胞数 + 体积分布图 + Z 轴密度图
import json
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

print("="*80)
print("📊 Cell 13 | Champion-only 3D statistics: cell counts, volume histogram, z-density")
print("="*80)

assert Path(champ_mask_path).exists(), "champion 3D mask 不存在，请先跑 Cell 12。"

ANALYSIS_DIR = (THREED_ROOT / "analysis").resolve()
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

mask_champ = tiff.imread(str(champ_mask_path))
print("mask_champ.shape:", mask_champ.shape, "| dtype:", mask_champ.dtype)

# ========= 1. 总细胞数 =========
champ_total_cells = int(len(np.unique(mask_champ)) - 1)

count_df = pd.DataFrame([
    {
        "model": f"Champion_{CHAMP_TAG}",
        "total_cells": champ_total_cells
    }
])
count_csv = ANALYSIS_DIR / "3d_total_cell_counts.csv"
count_df.to_csv(count_csv, index=False)

print("\n✅ 3D total cell counts:")
display(count_df)

# ========= 2. 体积分布 =========
_, counts_champ = np.unique(mask_champ, return_counts=True)
vols_champ = counts_champ[1:]  # 去掉背景 0

vol_df = pd.DataFrame({
    "model": [f"Champion_{CHAMP_TAG}"] * len(vols_champ),
    "volume_voxels": vols_champ
})
vol_csv = ANALYSIS_DIR / "3d_instance_volumes_long.csv"
vol_df.to_csv(vol_csv, index=False)

# 一些摘要统计，后面写论文/PPT很好用
vol_summary = {
    "model": f"Champion_{CHAMP_TAG}",
    "n_instances": int(len(vols_champ)),
    "mean_volume": float(np.mean(vols_champ)) if len(vols_champ) > 0 else 0.0,
    "median_volume": float(np.median(vols_champ)) if len(vols_champ) > 0 else 0.0,
    "std_volume": float(np.std(vols_champ)) if len(vols_champ) > 0 else 0.0,
    "min_volume": int(np.min(vols_champ)) if len(vols_champ) > 0 else 0,
    "max_volume": int(np.max(vols_champ)) if len(vols_champ) > 0 else 0,
    "p5_volume": float(np.percentile(vols_champ, 5)) if len(vols_champ) > 0 else 0.0,
    "p95_volume": float(np.percentile(vols_champ, 95)) if len(vols_champ) > 0 else 0.0,
}
vol_summary_path = ANALYSIS_DIR / "3d_volume_summary.json"
vol_summary_path.write_text(json.dumps(vol_summary, indent=2, ensure_ascii=False), encoding="utf-8")

plt.figure(figsize=(12, 7))
plt.hist(
    vols_champ, bins=60, range=(0, 1000),
    histtype='step', linewidth=2.5, alpha=0.95,
    label=f'Champion ({CHAMP_TAG})'
)

plt.axvline(
    np.mean(vols_champ),
    linestyle='dashed',
    linewidth=2,
    label=f'Mean: {np.mean(vols_champ):.1f}'
)

plt.title(f"3D Cell Volume Distribution: Champion ({CHAMP_TAG})", fontsize=16)
plt.xlabel("Cell Volume (voxels)", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.3)

vol_fig = ANALYSIS_DIR / "3d_volume_distribution_champion.png"
plt.savefig(vol_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Volume histogram saved:", vol_fig)

# ========= 3. Z 轴密度图 =========
z_counts_champ = []
for z in range(mask_champ.shape[0]):
    z_counts_champ.append(int(len(np.unique(mask_champ[z])) - 1))

z_df = pd.DataFrame({
    "z_index": np.arange(len(z_counts_champ)),
    "champion_cells": z_counts_champ,
})
z_csv = ANALYSIS_DIR / "3d_z_axis_density.csv"
z_df.to_csv(z_csv, index=False)

z_summary = {
    "model": f"Champion_{CHAMP_TAG}",
    "n_slices": int(len(z_counts_champ)),
    "mean_cells_per_slice": float(np.mean(z_counts_champ)) if len(z_counts_champ) > 0 else 0.0,
    "max_cells_per_slice": int(np.max(z_counts_champ)) if len(z_counts_champ) > 0 else 0,
    "argmax_z": int(np.argmax(z_counts_champ)) if len(z_counts_champ) > 0 else -1,
}
z_summary_path = ANALYSIS_DIR / "3d_z_density_summary.json"
z_summary_path.write_text(json.dumps(z_summary, indent=2, ensure_ascii=False), encoding="utf-8")

plt.figure(figsize=(12, 6))
plt.plot(
    z_df["z_index"],
    z_df["champion_cells"],
    linewidth=2,
    label=f"Champion ({CHAMP_TAG})"
)
plt.fill_between(
    z_df["z_index"],
    z_df["champion_cells"],
    alpha=0.15
)

plt.title(f"Whole-brain Cell Density along Z-axis: Champion ({CHAMP_TAG})", fontsize=16)
plt.xlabel("Z-slice index", fontsize=13)
plt.ylabel("Number of cells in slice", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=11)

z_fig = ANALYSIS_DIR / "3d_z_axis_density_champion.png"
plt.savefig(z_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Z-density plot saved:", z_fig)
print("✅ Counts CSV saved    :", count_csv)
print("✅ Volume CSV saved    :", vol_csv)
print("✅ Z-density CSV saved :", z_csv)
print("✅ Volume summary JSON :", vol_summary_path)
print("✅ Z summary JSON      :", z_summary_path)

print("\n✅ Cell 13 done.")
print("="*80)


# In[19]:


# Cell 14：Champion-only 典型切片 overlay + 局部高密区域 crop
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("🖼️ Cell 14 | Champion-only slice overlay & local crop analysis")
print("="*80)

VIS_DIR = (THREED_ROOT / "visuals").resolve()
VIS_DIR.mkdir(parents=True, exist_ok=True)

assert Path(champ_mask_path).exists(), "champion 3D mask 不存在，请先跑 Cell 12。"
assert Path(RAW_3D_STACK_PATH).exists(), "RAW_3D_STACK_PATH 不存在。"

raw_stack = tiff.imread(str(RAW_3D_STACK_PATH))
mask_champ = tiff.imread(str(champ_mask_path))

def norm01(img, p_low=1, p_high=99):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

# ========= 1. 自动选择中间切片 =========
Z_LAYER = raw_stack.shape[0] // 2

img_z = raw_stack[Z_LAYER]
mask_champ_z = mask_champ[Z_LAYER]
img_z_norm = norm01(img_z)

print("Using Z_LAYER:", Z_LAYER)
print("Slice shape   :", img_z.shape)

# ========= 2. 全貌 overlay =========
plt.figure(figsize=(12, 11))
plt.imshow(img_z_norm, cmap="gray")
plt.imshow(
    np.ma.masked_where(mask_champ_z == 0, mask_champ_z),
    cmap="autumn",
    alpha=0.5
)
plt.title(
    f"Champion ({CHAMP_TAG}) Overlay\nCells in Z={Z_LAYER}: {len(np.unique(mask_champ_z)) - 1}",
    fontsize=18,
    fontweight='bold'
)
plt.axis("off")

overlay_fig = VIS_DIR / f"champion_slice_overlay_z{Z_LAYER}.png"
plt.savefig(overlay_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Overlay figure saved:", overlay_fig)

# ========= 3. 局部 crop（默认居中区域） =========
H, W = img_z.shape
H_START, H_END = int(H * 0.25), int(H * 0.75)
W_START, W_END = int(W * 0.25), int(W * 0.75)

img_crop = img_z[H_START:H_END, W_START:W_END]
champ_crop = mask_champ_z[H_START:H_END, W_START:W_END]
img_crop_norm = norm01(img_crop)

plt.figure(figsize=(12, 10))
plt.imshow(img_crop_norm, cmap="gray")
plt.imshow(
    np.ma.masked_where(champ_crop == 0, champ_crop),
    cmap="autumn",
    alpha=0.6
)
plt.title(
    f"Champion Crop ({CHAMP_TAG})\n"
    f"Cells in crop: {len(np.unique(champ_crop)) - 1}\n"
    f"H[{H_START}:{H_END}], W[{W_START}:{W_END}]",
    fontsize=18,
    fontweight='bold'
)
plt.axis("off")

crop_fig = VIS_DIR / f"champion_slice_crop_overlay_z{Z_LAYER}.png"
plt.savefig(crop_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Crop figure saved:", crop_fig)

# ========= 4. 保存 crop 参数，防止以后忘了自己裁了哪里 =========
crop_meta = {
    "z_layer": int(Z_LAYER),
    "h_start": int(H_START),
    "h_end": int(H_END),
    "w_start": int(W_START),
    "w_end": int(W_END),
    "champion_tag": CHAMP_TAG,
    "champion_mask_path": str(champ_mask_path),
}
crop_meta_path = VIS_DIR / "crop_metadata.json"
crop_meta_path.write_text(json.dumps(crop_meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("✅ Crop metadata saved:", crop_meta_path)
print("\n✅ Cell 14 done.")
print("="*80)


# In[20]:


# Cell 15：Champion-only 3D 结果总打包与摘要导出
import os
import json
import zipfile
from pathlib import Path
from datetime import datetime

print("="*80)
print("📦 Cell 15 | Bundle champion-only 3D fullbrain results")
print("="*80)

ZIP_DIR = THREED_ROOT / "zips"
ZIP_DIR.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = ZIP_DIR / f"3d_fullbrain_champion_{CHAMP_TAG}_{stamp}.zip"

def add_dir(zf: zipfile.ZipFile, d: Path, prefix: str):
    n = 0
    if not d.exists():
        return n
    for p in d.rglob("*"):
        if p.is_file():
            rel = p.relative_to(d)
            zf.write(p, arcname=str(Path(prefix) / rel))
            n += 1
    return n

def add_file(zf: zipfile.ZipFile, p: Path, arcname: str):
    if p.exists():
        zf.write(p, arcname=arcname)
        return 1
    return 0

n_added = 0
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    n_added += add_file(zf, THREED_CONFIG_PATH, "config/3d_run_config.json")
    n_added += add_file(zf, THREED_SUMMARY_PATH, "summary/3d_champion_summary.json")
    n_added += add_file(zf, RANK_CSV, "summary/final_refine_merged_ranking.csv")

    n_added += add_dir(zf, CHAMP_3D_DIR, f"results/Champion_{CHAMP_TAG}")
    n_added += add_dir(zf, ANALYSIS_DIR, "analysis")
    n_added += add_dir(zf, VIS_DIR, "visuals")

print("✅ Added files:", n_added)
print("📦 ZIP saved to:", zip_path)
print("\n✅ Cell 15 done.")
print("="*80)


# In[ ]:




