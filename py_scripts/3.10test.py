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
EXP_NAME = f"exp_20260310_lr_wd_6runs_400ep_{STAMP}"
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
    "note": "6-run 400-epoch sweep on transformer mainline: weight_decay x learning_rate. Best checkpoint aware.",
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


# Cell 3：定义 6 个版本（wd × lr, 400 epochs）
import json
from pathlib import Path

print("="*80)
print("🧪 Cell 3 | Define 6-run 400-epoch sweep")
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
    "n_epochs": 400,
}

SWEEP = [
    dict(tag="L00_wd8e3_lr5e5",   weight_decay=8e-3, learning_rate=5e-5),
    dict(tag="L01_wd8e3_lr1e4",   weight_decay=8e-3, learning_rate=1e-4),
    dict(tag="L02_wd8e3_lr15e5",  weight_decay=8e-3, learning_rate=1.5e-4),

    dict(tag="L10_wd1e2_lr5e5",   weight_decay=1e-2, learning_rate=5e-5),
    dict(tag="L11_wd1e2_lr1e4",   weight_decay=1e-2, learning_rate=1e-4),
    dict(tag="L12_wd1e2_lr15e5",  weight_decay=1e-2, learning_rate=1.5e-4),
]

print("BEST_CELLPROB_TH:", BEST_CELLPROB_TH)
print("BEST_FLOW_TH    :", BEST_FLOW_TH)
print("Sweep size:", len(SWEEP))
for v in SWEEP:
    print(v)

print("\n✅ Cell 3 done.")
print("="*80)


# In[5]:


# Cell 4：生成 RUNS.jsonl（best-checkpoint ready）
import json
from pathlib import Path
from datetime import datetime

print("="*80)
print("📝 Cell 4 | Commit runs into RUNS.jsonl")
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


# Cell 5：串行训练执行器（400轮，best-checkpoint ready）
import json, os, time, shlex, subprocess, traceback, re
from pathlib import Path
from datetime import datetime

print("="*80)
print("🚀 Cell 5 | Serial training executor (400 epochs, best-checkpoint ready)")
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
    for line in reversed(lines[-5000:]):
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
RUN_ONLY_TAGS = None   # 如需只跑部分，可写 ["L01_wd8e3_lr1e4"]
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
            for p in ckpts[:10]:
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


# Cell 6：训练汇总表（自动选择 best checkpoint）
import json, re
from pathlib import Path
import pandas as pd

print("="*80)
print("📊 Cell 6 | Parse training logs -> summary_runs.csv")
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
display(df)

print("\n✅ Cell 6 done.")
print("="*80)


# In[8]:


# Cell 7：画每个版本的 loss 曲线 + 总对比图
import json, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

print("="*80)
print("📈 Cell 7 | Plot per-run loss curves and combined comparison")
print("="*80)

PLOT_DIR = (EXPORT_DIR / "loss_curves").resolve()
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_JSONL = CFG_DIR / "RUNS.jsonl"
runs = [json.loads(l) for l in RUNS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

MET_RE = re.compile(
    r"\b(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)"
)

all_curves = []

for r in runs:
    tag = r["tag"]
    log_path = Path(r["log_path"])
    if not log_path.exists():
        print(f"⚠️ log missing: {tag}")
        continue

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in MET_RE.finditer(text):
        rows.append({
            "epoch": int(m.group("epoch")),
            "train_loss": float(m.group("tr")),
            "test_loss": float(m.group("te")),
            "lr": float(m.group("lr")),
            "tag": tag,
        })

    if not rows:
        print(f"⚠️ no parsed metrics: {tag}")
        continue

    dfm = pd.DataFrame(rows).sort_values("epoch")
    all_curves.append(dfm)

    # 单独画每个版本
    plt.figure(figsize=(10, 5))
    plt.plot(dfm["epoch"], dfm["train_loss"], marker="o", markersize=3, linewidth=1.5, label="Train Loss")
    plt.plot(dfm["epoch"], dfm["test_loss"], marker="s", markersize=3, linewidth=1.5, label="Test Loss")

    best_idx = dfm["test_loss"].idxmin()
    best_ep = int(dfm.loc[best_idx, "epoch"])
    best_te = float(dfm.loc[best_idx, "test_loss"])
    plt.axvline(best_ep, linestyle="--", linewidth=1.5, label=f"Best Ep={best_ep}")

    plt.title(f"Loss Curve | {tag}", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out_png = PLOT_DIR / f"loss_curve_{tag}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ saved: {out_png.name}")

assert len(all_curves) > 0, "没有可用的 loss 曲线数据。"

df_all = pd.concat(all_curves, ignore_index=True)
df_all.to_csv(PLOT_DIR / "all_loss_curves_long.csv", index=False)

# 全部 train loss 对比
plt.figure(figsize=(12, 7))
for tag, g in df_all.groupby("tag"):
    plt.plot(g["epoch"], g["train_loss"], linewidth=1.6, label=tag)
plt.title("All Runs | Train Loss Comparison", fontsize=15)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=9)
plt.savefig(PLOT_DIR / "all_runs_train_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# 全部 test loss 对比
plt.figure(figsize=(12, 7))
for tag, g in df_all.groupby("tag"):
    plt.plot(g["epoch"], g["test_loss"], linewidth=1.6, label=tag)
plt.title("All Runs | Test Loss Comparison", fontsize=15)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=9)
plt.savefig(PLOT_DIR / "all_runs_test_loss.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Combined plots saved to:", PLOT_DIR)
print("\n✅ Cell 7 done.")
print("="*80)


# In[9]:


# Cell 8：统一推理（优先使用 best checkpoint）
import os, shlex, subprocess, time, json
from pathlib import Path

print("="*80)
print("🤖 Cell 8 | Unified inference with fixed thresholds (prefer best checkpoint)")
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
print("\n✅ Cell 8 done.")
print("="*80)


# In[10]:


# Cell 9：实例级评估
import re, json
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from cellpose import metrics

print("="*80)
print("🏁 Cell 9 | Evaluate 6-run sweep")
print("="*80)

assert "RUN_OUT" in globals(), "RUN_OUT 未定义，请先跑 Cell 8。"
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
EVAL_CSV = RUN_OUT / "eval_metrics.csv"
df_eval.to_csv(EVAL_CSV, index=False)

print("✅ saved:", EVAL_CSV)
display(df_eval)
print("\n✅ Cell 9 done.")
print("="*80)


# In[11]:


# Cell 10：融合总榜（标明 best / final）
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("🧾 Cell 10 | Merge summary with final evaluation")
print("="*80)

summary_csv = EXP_DIR / "summary_runs.csv"
eval_csv = RUN_OUT / "eval_metrics.csv"
runs_jsonl = CFG_DIR / "RUNS.jsonl"

assert summary_csv.exists(), f"summary_runs.csv 不存在：{summary_csv}"
assert eval_csv.exists(), f"eval_metrics.csv 不存在：{eval_csv}"
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

out_csv = EXP_DIR / "final_merged_ranking.csv"
df_merged.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_merged)

print("\n✅ Cell 10 done.")
print("="*80)


# In[12]:


# Cell 11：打包所有关键结果
import os, zipfile
from pathlib import Path
from datetime import datetime

print("="*80)
print("📦 Cell 11 | Bundle experiment")
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
    n_added += add_dir(zf, EXPORT_DIR, "exports")

    if "RUN_OUT" in globals() and Path(RUN_OUT).exists():
        n_added += add_dir(zf, Path(RUN_OUT), "inference_eval")

    for p in [
        EXP_DIR / "summary_runs.csv",
        EXP_DIR / "final_merged_ranking.csv",
    ]:
        if p.exists():
            add_file(zf, p, f"summary/{p.name}")
            n_added += 1

print("✅ Bundled files:", n_added)
print("📦 ZIP saved to:", zip_path)
print("\n✅ Cell 11 done.")
print("="*80)


# In[ ]:




