#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#原版数据集9个版本对比


# In[1]:


# Cell 0：环境自检
import os, sys, json, time, shlex, subprocess, socket, re, signal
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

TRAIN_DIR = (DATA_ROOT / "trainset").resolve()

NEWVAL_ROOT = (DATA_ROOT / "new_val_proc").resolve()
VAL_IMG_DIR = (NEWVAL_ROOT / "images_sp_lcn").resolve()
VAL_GT_DIR  = (NEWVAL_ROOT / "ground").resolve()

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"exp_20260311_ori_train_9runs_400ep_es50_{STAMP}"
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
    "note": "Ultra-small refinement on transformer mainline. 400-epoch cap + early stopping patience 50 + best-checkpoint aware.",
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


# Cell 3：定义 9 个版本（original train local refinement, 400 cap + ES50）
import json
from pathlib import Path

print("="*80)
print("🧪 Cell 3 | Define 9-run local refinement sweep on original trainset")
print("="*80)

BEST_CELLPROB_TH = -0.5
BEST_FLOW_TH = 0.6

# 训练策略
N_EPOCHS = 400
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE_EPOCHS = 50
EARLY_STOP_MIN_DELTA = 1e-4

BASE = {
    "train_dir": str(TRAIN_DIR),
    "test_dir": str(VALVIEW_DIR),
    "mask_filter": "_masks",
    "pretrained_model": "cpsam",
    "save_every": 5,
    "save_each": True,
    "use_gpu": True,
    "verbose": True,
    "bsize": 256,
    "augment": True,
    "train_batch_size": 32,
    "transformer": True,
    "diameter": 14,
    "n_epochs": N_EPOCHS,

    # early stopping 配置（由 Cell 5 执行器负责，不是 cellpose CLI 原生参数）
    "early_stop_enabled": EARLY_STOP_ENABLED,
    "early_stop_patience_epochs": EARLY_STOP_PATIENCE_EPOCHS,
    "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
}

SWEEP = [
    # lr = 7e-5
    dict(tag="P00_lr7e5_wd7e3", learning_rate=7e-5, weight_decay=7e-3),
    dict(tag="P01_lr7e5_wd8e3", learning_rate=7e-5, weight_decay=8e-3),
    dict(tag="P02_lr7e5_wd9e3", learning_rate=7e-5, weight_decay=9e-3),

    # lr = 8e-5
    dict(tag="P10_lr8e5_wd7e3", learning_rate=8e-5, weight_decay=7e-3),
    dict(tag="P11_lr8e5_wd8e3", learning_rate=8e-5, weight_decay=8e-3),  # 当前冠军复现点
    dict(tag="P12_lr8e5_wd9e3", learning_rate=8e-5, weight_decay=9e-3),

    # lr = 9e-5
    dict(tag="P20_lr9e5_wd7e3", learning_rate=9e-5, weight_decay=7e-3),
    dict(tag="P21_lr9e5_wd8e3", learning_rate=9e-5, weight_decay=8e-3),
    dict(tag="P22_lr9e5_wd9e3", learning_rate=9e-5, weight_decay=9e-3),
]

print("BEST_CELLPROB_TH:", BEST_CELLPROB_TH)
print("BEST_FLOW_TH    :", BEST_FLOW_TH)
print("N_EPOCHS        :", N_EPOCHS)
print("EARLY_STOP      :", EARLY_STOP_ENABLED, "| patience =", EARLY_STOP_PATIENCE_EPOCHS)
print("TRAIN_DIR       :", TRAIN_DIR)
print("VALVIEW_DIR     :", VALVIEW_DIR)

print("\nPlanned runs:")
for v in SWEEP:
    print(
        f" - {v['tag']}"
        f" | lr={v['learning_rate']}"
        f" | wd={v['weight_decay']}"
        f" | d={BASE['diameter']}"
        f" | trans={BASE['transformer']}"
    )

(CFG_DIR / "sweep_preview.json").write_text(
    json.dumps({"base": BASE, "sweep": SWEEP}, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n✅ Sweep preview saved:", CFG_DIR / "sweep_preview.json")
print("✅ Cell 3 done.")
print("="*80)


# In[5]:


# Cell 4：生成 RUNS.jsonl（best-checkpoint ready + early-stop aware）
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
        "best_model_source": None,
        "eval_model_path": None,
        "eval_model_type": None,
        "available_checkpoints": [],

        "early_stop_triggered": False,
        "early_stop_reason": None,
        "early_stop_epoch": None,
        "stale_epochs_when_stopped": None,

        "checkpoint_strategy": {
            "save_every": p.get("save_every", None),
            "save_each": p.get("save_each", True),
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


# Cell 5：串行训练执行器（400上限 + early stopping + best-checkpoint ready）
import json, os, time, shlex, subprocess, traceback, re, signal, csv
from pathlib import Path
from datetime import datetime

print("="*80)
print("🚀 Cell 5 | Serial training executor (400 cap + ES50 + best-checkpoint ready)")
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

MET_RE = re.compile(
    r"\b(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)",
    re.IGNORECASE
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
    status = snap.get("status", "")
    return bool(final_model_path) and Path(final_model_path).exists() and status == "DONE"

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

def ensure_metrics_csv(metrics_path: Path):
    if metrics_path.exists():
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["time","epoch","train_loss","test_loss","lr","raw_line"]
        )
        w.writeheader()

def parse_new_metrics_from_log(log_path: Path, state: dict):
    if not log_path.exists():
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(state["offset"])
        new_text = f.read()
        state["offset"] = f.tell()

    rows = []
    if not new_text:
        return rows

    for line in new_text.splitlines():
        m = MET_RE.search(line.strip())
        if not m:
            continue
        ep = int(m.group("epoch"))
        if ep in state["seen_epochs"]:
            continue
        state["seen_epochs"].add(ep)
        rows.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "epoch": ep,
            "train_loss": float(m.group("tr")),
            "test_loss": float(m.group("te")),
            "lr": float(m.group("lr")),
            "raw_line": line.strip()[:500],
        })
    return rows

def append_metrics_csv(metrics_path: Path, rows):
    if not rows:
        return
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["time","epoch","train_loss","test_loss","lr","raw_line"]
        )
        for r in rows:
            w.writerow(r)

def kill_process_gracefully(proc, grace_s=8):
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < grace_s:
        ret = proc.poll()
        if ret is not None:
            return
        time.sleep(0.5)

    try:
        proc.kill()
    except Exception:
        pass

POLL_S = 30
RUN_ONLY_TAGS = None
STOP_ON_FAILURE = False

selected_runs = [r for r in RUNS if (RUN_ONLY_TAGS is None or r["tag"] in RUN_ONLY_TAGS)]
print("Runs to process:", len(selected_runs))

for idx, ctx in enumerate(selected_runs, 1):
    print("\n" + "="*100)
    print(f"[{idx}/{len(selected_runs)}] RUN: {ctx['run_name']}")
    print("="*100)

    if already_done(ctx):
        print("✅ already done -> skip")
        continue

    cmd = build_cmd(ctx)
    cmd_text = cmd_to_text(cmd)

    cmd_path = Path(ctx["cmd_path"])
    log_path = Path(ctx["log_path"])
    pid_path = Path(ctx["pid_path"])
    metrics_path = Path(ctx["metrics_path"])
    snap_path = Path(ctx["config_snapshot_path"])

    ensure_metrics_csv(metrics_path)
    cmd_path.write_text(cmd_text + "\n", encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("CMD:")
    print(cmd_text)
    print("\nLOG:", log_path)

    p = ctx["params"]
    es_enabled = bool(p.get("early_stop_enabled", False))
    es_patience = int(p.get("early_stop_patience_epochs", 0))
    es_min_delta = float(p.get("early_stop_min_delta", 0.0))

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser_state = {"offset": 0, "seen_epochs": set()}
    best_test_loss = None
    best_epoch = None
    last_epoch = None
    last_test = None
    stale_epochs = 0
    early_stop_triggered = False
    early_stop_reason = None
    early_stop_epoch = None

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

            snap = read_json(snap_path) or {}
            snap.update({
                "status": "RUNNING",
                "launched_at": datetime.now().isoformat(timespec="seconds"),
                "pid": proc.pid,
                "early_stop_triggered": False,
                "early_stop_reason": None,
                "early_stop_epoch": None,
                "stale_epochs_when_stopped": None,
            })
            write_json(snap_path, snap)

            print("🚀 Started | PID:", proc.pid)

            while True:
                ret = proc.poll()

                new_rows = parse_new_metrics_from_log(log_path, parser_state)
                append_metrics_csv(metrics_path, new_rows)

                if new_rows:
                    for row in new_rows:
                        ep = int(row["epoch"])
                        te = float(row["test_loss"])

                        last_epoch = ep
                        last_test = te

                        improved = False
                        if best_test_loss is None:
                            improved = True
                        elif te < (best_test_loss - es_min_delta):
                            improved = True

                        if improved:
                            best_test_loss = te
                            best_epoch = ep
                            stale_epochs = 0
                        else:
                            if best_epoch is not None:
                                stale_epochs = ep - best_epoch
                            else:
                                stale_epochs += 1

                    snap = read_json(snap_path) or {}
                    snap.update({
                        "best_epoch_by_test_loss": best_epoch,
                        "best_test_loss": best_test_loss,
                        "last_epoch_logged": last_epoch,
                        "last_test_loss": last_test,
                    })
                    write_json(snap_path, snap)

                    print(f"📌 latest ep={last_epoch} | test={last_test:.6f} | best_ep={best_epoch} | best_test={best_test_loss:.6f} | stale={stale_epochs}")

                    if es_enabled and best_epoch is not None and stale_epochs >= es_patience and ret is None:
                        early_stop_triggered = True
                        early_stop_epoch = last_epoch
                        early_stop_reason = (
                            f"no test-loss improvement for {stale_epochs} epochs "
                            f"(patience={es_patience}, min_delta={es_min_delta})"
                        )
                        print("🛑 Early stopping triggered:", early_stop_reason)
                        kill_process_gracefully(proc, grace_s=8)
                        ret = proc.poll()

                if ret is not None:
                    print("✅ process finished with return code:", ret)
                    break

                time.sleep(POLL_S)

        pid_path.unlink(missing_ok=True)

        final_model = capture_final_model_from_log(log_path)
        ckpts = scan_available_checkpoints(Path(final_model)) if final_model else []

        best_model_path = None
        best_model_source = None
        best_model_found = False
        if final_model and best_epoch is not None and Path(final_model).exists():
            best_model_path, best_model_source = resolve_best_model_path(final_model, best_epoch)
            best_model_found = (best_model_source == "best_ckpt")

        snap = read_json(snap_path) or {}
        if final_model and Path(final_model).exists():
            snap.update({
                "status": "DONE",
                "model_dir": final_model,
                "final_model_path": final_model,
                "best_model_path": best_model_path,
                "best_model_found": best_model_found,
                "best_model_source": best_model_source,
                "best_epoch_by_test_loss": best_epoch,
                "best_test_loss": best_test_loss,
                "last_epoch_logged": last_epoch,
                "last_test_loss": last_test,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "return_code": ret,
                "available_checkpoints": ckpts,
                "early_stop_triggered": early_stop_triggered,
                "early_stop_reason": early_stop_reason,
                "early_stop_epoch": early_stop_epoch,
                "stale_epochs_when_stopped": stale_epochs if early_stop_triggered else None,
                "checkpoint_strategy": {
                    "save_every": p.get("save_every", None),
                    "save_each": p.get("save_each", True),
                    "epoch_index_base": 0,
                    "note": "0-based epoch index from logs; intermediate checkpoints saved with _epoch_XXXX suffix."
                }
            })
            write_json(snap_path, snap)

            print("🏁 final_model_path:", final_model)
            print("🧩 checkpoints found:", len(ckpts))
            print("🎯 best_model_path  :", best_model_path)
            print("🛑 early_stop       :", early_stop_triggered, "|", early_stop_reason)
        else:
            snap.update({
                "status": "FAILED",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "return_code": ret,
                "note": "training ended but final_model_path not captured; inspect log",
                "best_epoch_by_test_loss": best_epoch,
                "best_test_loss": best_test_loss,
                "last_epoch_logged": last_epoch,
                "last_test_loss": last_test,
                "early_stop_triggered": early_stop_triggered,
                "early_stop_reason": early_stop_reason,
                "early_stop_epoch": early_stop_epoch,
                "stale_epochs_when_stopped": stale_epochs if early_stop_triggered else None,
            })
            write_json(snap_path, snap)

            print("⚠️ final_model_path 未捕获，请检查 log。")
            if STOP_ON_FAILURE:
                raise RuntimeError(f"Run failed: {ctx['tag']}")

    except Exception as e:
        pid_path.unlink(missing_ok=True)
        snap = read_json(snap_path) or {}
        snap.update({
            "status": "FAILED",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "note": f"executor exception: {repr(e)}",
        })
        write_json(snap_path, snap)

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
    best_model_path = snap.get("best_model_path")
    best_model_source = snap.get("best_model_source", "unresolved")
    best_model_found = snap.get("best_model_found", False)

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
        "early_stop_enabled": p.get("early_stop_enabled", False),
        "early_stop_patience_epochs": p.get("early_stop_patience_epochs"),
        "best_epoch_by_test_loss": best_epoch,
        "best_test_loss": best_test,
        "last_epoch_logged": last_epoch,
        "last_test_loss": last_test,
        "epoch_index_base": 0,
        "final_epoch_logged": final_epoch_logged,
        "early_stop_triggered": snap.get("early_stop_triggered", False),
        "early_stop_reason": snap.get("early_stop_reason"),
        "early_stop_epoch": snap.get("early_stop_epoch"),
        "stale_epochs_when_stopped": snap.get("stale_epochs_when_stopped"),
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
    snap_path = Path(r["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}

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

    plt.figure(figsize=(10, 5))
    plt.plot(dfm["epoch"], dfm["train_loss"], marker="o", markersize=3, linewidth=1.5, label="Train Loss")
    plt.plot(dfm["epoch"], dfm["test_loss"], marker="s", markersize=3, linewidth=1.5, label="Test Loss")

    best_idx = dfm["test_loss"].idxmin()
    best_ep = int(dfm.loc[best_idx, "epoch"])
    best_te = float(dfm.loc[best_idx, "test_loss"])
    plt.axvline(best_ep, linestyle="--", linewidth=1.5, label=f"Best Ep={best_ep}")

    if snap.get("early_stop_triggered", False) and snap.get("early_stop_epoch") is not None:
        plt.axvline(int(snap["early_stop_epoch"]), linestyle=":", linewidth=1.5, label=f"ES @ Ep={snap['early_stop_epoch']}")

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


# Cell 10：融合总榜（标明 best / final / early-stop）
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
        "early_stop_triggered": snap.get("early_stop_triggered", False),
        "early_stop_reason": snap.get("early_stop_reason"),
        "early_stop_epoch": snap.get("early_stop_epoch"),
        "stale_epochs_when_stopped": snap.get("stale_epochs_when_stopped"),
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


# In[13]:


# Cell 12：自动生成本轮参数表与结果解读摘要
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*80)
print("🧠 Cell 12 | Auto-generate experiment summary & report-friendly tables")
print("="*80)

RANK_CSV = EXP_DIR / "final_merged_ranking.csv"
SUMMARY_CSV = EXP_DIR / "summary_runs.csv"

assert RANK_CSV.exists(), f"未找到 final_merged_ranking.csv: {RANK_CSV}"
assert SUMMARY_CSV.exists(), f"未找到 summary_runs.csv: {SUMMARY_CSV}"

df = pd.read_csv(RANK_CSV)
assert len(df) > 0, "final_merged_ranking.csv 为空。"

REPORT_DIR = (EXP_DIR / "report_summary").resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. 生成汇报友好的精简表
# =========================
report_cols = [
    "tag",
    "lr",
    "weight_decay",
    "n_epochs",
    "best_epoch_by_test_loss",
    "best_test_loss",
    "last_epoch_logged",
    "last_test_loss",
    "early_stop_triggered",
    "early_stop_epoch",
    "eval_model_type",
    "AP50",
    "Precision",
    "Recall",
    "F1",
    "n_eval",
    "n_missing_pred",
]

report_cols = [c for c in report_cols if c in df.columns]
df_report = df[report_cols].copy()

# 排序：先按 AP50/F1
df_report = df_report.sort_values(["AP50", "F1"], ascending=False, na_position="last").reset_index(drop=True)

report_csv = REPORT_DIR / "ranking_report_table.csv"
df_report.to_csv(report_csv, index=False)

print("✅ 汇报精简表已保存:", report_csv)
display(df_report)

# =========================
# 2. 自动提取关键结论
# =========================
top1 = df.iloc[0].to_dict()
top2 = df.iloc[1].to_dict() if len(df) >= 2 else None
top3 = df.iloc[2].to_dict() if len(df) >= 3 else None

top1_tag = top1.get("tag")
top1_ap50 = float(top1.get("AP50", float("nan")))
top1_f1 = float(top1.get("F1", float("nan")))
top1_recall = float(top1.get("Recall", float("nan")))
top1_prec = float(top1.get("Precision", float("nan")))
top1_lr = top1.get("lr")
top1_wd = top1.get("weight_decay")
top1_eval_type = top1.get("eval_model_type")
top1_best_ep = top1.get("best_epoch_by_test_loss")
top1_es = bool(top1.get("early_stop_triggered", False))

delta_top12_ap50 = None
delta_top12_f1 = None
if top2 is not None:
    delta_top12_ap50 = float(top1["AP50"]) - float(top2["AP50"])
    delta_top12_f1 = float(top1["F1"]) - float(top2["F1"])

# early stop 统计
n_total = len(df)
n_es = int(df["early_stop_triggered"].fillna(False).astype(bool).sum()) if "early_stop_triggered" in df.columns else 0

# 统计参数趋势
group_lr = None
group_wd = None
if "lr" in df.columns and "AP50" in df.columns:
    group_lr = df.groupby("lr", dropna=False)[["AP50", "F1", "Recall", "Precision"]].mean().reset_index()
if "weight_decay" in df.columns and "AP50" in df.columns:
    group_wd = df.groupby("weight_decay", dropna=False)[["AP50", "F1", "Recall", "Precision"]].mean().reset_index()

if group_lr is not None:
    group_lr_csv = REPORT_DIR / "groupby_lr_mean_metrics.csv"
    group_lr.to_csv(group_lr_csv, index=False)
    print("✅ lr 分组均值表:", group_lr_csv)
    display(group_lr)

if group_wd is not None:
    group_wd_csv = REPORT_DIR / "groupby_wd_mean_metrics.csv"
    group_wd.to_csv(group_wd_csv, index=False)
    print("✅ wd 分组均值表:", group_wd_csv)
    display(group_wd)

# =========================
# 3. 自动生成文字摘要
# =========================
lines = []
lines.append(f"实验目录：{EXP_DIR}")
lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
lines.append("")
lines.append("【总体结论】")
lines.append(
    f"本轮共评估 {n_total} 个模型，当前排名第 1 的模型为 {top1_tag}，"
    f"其超参数为 lr={top1_lr}, weight_decay={top1_wd}，"
    f"最终指标为 AP50={top1_ap50:.6f}, Precision={top1_prec:.6f}, Recall={top1_recall:.6f}, F1={top1_f1:.6f}。"
)

if top2 is not None:
    lines.append(
        f"与第 2 名相比，冠军模型 AP50 领先 {delta_top12_ap50:.6f}，"
        f"F1 领先 {delta_top12_f1:.6f}。"
    )

lines.append(
    f"冠军模型的统一评估使用的是 {top1_eval_type}，最佳 checkpoint 对应 epoch={top1_best_ep}。"
)

if top1_es:
    lines.append("冠军模型训练过程中触发了 early stopping，说明其最优性能已在训练后期前出现。")
else:
    lines.append("冠军模型训练过程中未触发 early stopping，说明其在设定训练窗口内仍保持可训练状态。")

lines.append("")
lines.append("【early stopping 情况】")
lines.append(f"本轮共有 {n_es}/{n_total} 个模型触发 early stopping。")

if n_es > 0:
    lines.append("这说明 400 epoch 作为上限是合理的，但并非所有模型都需要完整训练到最后。")
else:
    lines.append("这说明在当前 400 epoch 上限和 patience 设置下，模型大多仍能跑完整个训练窗口。")

lines.append("")
lines.append("【头部模型建议】")
head_tags = df["tag"].head(min(3, len(df))).tolist()
lines.append(f"建议优先关注头部模型：{', '.join(head_tags)}。")

if len(df) >= 2:
    lines.append("由于前 2~3 名之间的差距通常较小，建议至少保留前 2 名作为后续 3D 或更高层级验证候选。")

lines.append("")
lines.append("【参数趋势粗读】")

if group_lr is not None and len(group_lr) > 0:
    best_lr_row = group_lr.sort_values("AP50", ascending=False).iloc[0].to_dict()
    lines.append(
        f"按 learning rate 分组后，平均 AP50 最优的 lr 为 {best_lr_row['lr']}，"
        f"其均值 AP50={best_lr_row['AP50']:.6f}。"
    )

if group_wd is not None and len(group_wd) > 0:
    best_wd_row = group_wd.sort_values("AP50", ascending=False).iloc[0].to_dict()
    lines.append(
        f"按 weight decay 分组后，平均 AP50 最优的 wd 为 {best_wd_row['weight_decay']}，"
        f"其均值 AP50={best_wd_row['AP50']:.6f}。"
    )

lines.append("最终冠军仍应以统一实例级评估结果为准，而不是仅依据训练/测试 loss 曲线判断。")

summary_txt = "\n".join(lines)
summary_txt_path = REPORT_DIR / "auto_summary.txt"
summary_txt_path.write_text(summary_txt, encoding="utf-8")

print("\n" + "="*80)
print("📌 自动摘要如下：\n")
print(summary_txt)
print("="*80)

print("✅ 自动摘要已保存:", summary_txt_path)

# =========================
# 4. 生成一个更适合 PPT/汇报的头部表
# =========================
ppt_cols = [
    "tag", "lr", "weight_decay",
    "best_epoch_by_test_loss",
    "early_stop_triggered",
    "AP50", "Precision", "Recall", "F1"
]
ppt_cols = [c for c in ppt_cols if c in df.columns]

df_top = df[ppt_cols].head(min(5, len(df))).copy()
ppt_csv = REPORT_DIR / "top_models_for_ppt.csv"
df_top.to_csv(ppt_csv, index=False)

print("✅ PPT 用 top 表已保存:", ppt_csv)
display(df_top)

# =========================
# 5. 存一个 json 方便后续程序继续读
# =========================
summary_json = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "exp_dir": str(EXP_DIR),
    "n_models": int(n_total),
    "n_early_stop": int(n_es),
    "top1": {
        "tag": top1_tag,
        "lr": top1_lr,
        "weight_decay": top1_wd,
        "AP50": top1_ap50,
        "Precision": top1_prec,
        "Recall": top1_recall,
        "F1": top1_f1,
        "eval_model_type": top1_eval_type,
        "best_epoch_by_test_loss": top1_best_ep,
        "early_stop_triggered": top1_es,
    },
    "top2": None if top2 is None else {
        "tag": top2.get("tag"),
        "AP50": float(top2.get("AP50", float("nan"))),
        "F1": float(top2.get("F1", float("nan"))),
    },
    "top3": None if top3 is None else {
        "tag": top3.get("tag"),
        "AP50": float(top3.get("AP50", float("nan"))),
        "F1": float(top3.get("F1", float("nan"))),
    },
    "delta_top12_ap50": delta_top12_ap50,
    "delta_top12_f1": delta_top12_f1,
    "head_tags": head_tags,
}
summary_json_path = REPORT_DIR / "auto_summary.json"
summary_json_path.write_text(
    json.dumps(summary_json, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("✅ JSON 摘要已保存:", summary_json_path)
print("\n✅ Cell 12 done.")
print("="*80)


# In[14]:


# Cell 13：自动锁定总榜前2名，并配置 3D 路径
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("🏷️ Cell 13 | Auto-pick top 2 models and setup 3D paths")
print("="*80)

RANK_CSV = EXP_DIR / "final_merged_ranking.csv"
assert RANK_CSV.exists(), f"未找到总榜文件: {RANK_CSV}"

df_rank = pd.read_csv(RANK_CSV)
assert len(df_rank) >= 2, "总榜不足 2 个模型，无法选 top2。"

# ===== 排名规则：优先 AP50，再看 F1 =====
df_rank = df_rank.sort_values(
    ["AP50", "F1"],
    ascending=False,
    na_position="last"
).reset_index(drop=True)

# ===== 自动取前2名 =====
df_pick = df_rank.head(2).copy()

# 可选：如果你想只允许某类模型进入 3D，比如必须 eval_model_path 存在
assert df_pick["eval_model_path"].notna().all(), "前2名里存在空的 eval_model_path"
assert df_pick["tag"].nunique() == 2, "前2名 tag 不唯一，请检查总榜"

MODELS_3D = []
for rank_idx, (_, row) in enumerate(df_pick.iterrows(), start=1):
    model_path = str(row["eval_model_path"])
    assert Path(model_path).exists(), f"模型路径不存在: {model_path}"

    MODELS_3D.append({
        "rank_2d": rank_idx,
        "tag": str(row["tag"]),
        "run_name": str(row["run_name"]),
        "model_path": model_path,
        "eval_model_type": str(row["eval_model_type"]),
        "diameter_2d": int(row["diameter"]),
        "AP50_2d": float(row["AP50"]),
        "F1_2d": float(row["F1"]),
        "Recall_2d": float(row["Recall"]),
        "Precision_2d": float(row["Precision"]),
    })

print("✅ 本次自动进入 3D 的两个模型：")
for m in MODELS_3D:
    print("-"*60)
    print("2D rank        :", m["rank_2d"])
    print("tag            :", m["tag"])
    print("run_name       :", m["run_name"])
    print("model_path     :", m["model_path"])
    print("eval_model_type:", m["eval_model_type"])
    print("AP50_2d        :", m["AP50_2d"])
    print("F1_2d          :", m["F1_2d"])
    print("Recall_2d      :", m["Recall_2d"])
    print("Precision_2d   :", m["Precision_2d"])

RAW_3D_STACK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_proc/iccv_sbg_imNor_30_0.000010.tif"
).resolve()
assert RAW_3D_STACK_PATH.exists(), f"3D 原始数据不存在: {RAW_3D_STACK_PATH}"

# ===== 3D 参数 =====
TRUE_3D_DIAMETER = 8
DO_3D = True
Z_AXIS = 0
STITCH_THRESHOLD = 0.5
BATCH_SIZE_3D = 16

# 输出目录也改成自动 top2 命名
picked_tags = "__".join([m["tag"] for m in MODELS_3D])
THREED_ROOT = (EXP_DIR / f"3d_top2_{picked_tags}").resolve()
THREED_ROOT.mkdir(parents=True, exist_ok=True)

for m in MODELS_3D:
    out_dir = THREED_ROOT / f"rank{m['rank_2d']}_{m['tag']}_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    m["out_dir"] = str(out_dir)

THREED_CONFIG = {
    "created_at": pd.Timestamp.now().isoformat(),
    "mode": "auto_top2_3d",
    "rank_csv": str(RANK_CSV),
    "selection_rule": "sort by AP50 desc, then F1 desc, pick top 2",
    "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
    "true_3d_diameter": TRUE_3D_DIAMETER,
    "do_3d": DO_3D,
    "z_axis": Z_AXIS,
    "stitch_threshold": STITCH_THRESHOLD,
    "batch_size_3d": BATCH_SIZE_3D,
    "models_3d": MODELS_3D,
    "output_root": str(THREED_ROOT),
    "note": "Automatically pick top-2 models from final_merged_ranking.csv and run 3D inference."
}

THREED_CONFIG_PATH = THREED_ROOT / "3d_auto_top2_config.json"
THREED_CONFIG_PATH.write_text(
    json.dumps(THREED_CONFIG, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nRAW_3D_STACK_PATH:", RAW_3D_STACK_PATH)
print("THREED_ROOT      :", THREED_ROOT)
print("CONFIG saved to  :", THREED_CONFIG_PATH)

print("\n✅ Cell 13 done.")
print("="*80)


# In[15]:


# Cell 14：运行 R20 / R21 的 3D 推理（自动跳过已有结果）
import time
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from cellpose import models, io

print("="*80)
print("🚀 Cell 14 | Run 3D inference for R20 and R21 (skip baseline)")
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


THREED_RESULTS = {}

for m in MODELS_3D:
    tag = m["tag"]
    out_dir = Path(m["out_dir"])
    mask_path, stats = run_3d_inference_once(
        model_path=m["model_path"],
        out_dir=out_dir,
        run_label=tag
    )
    THREED_RESULTS[tag] = {
        "mask_path": str(mask_path),
        "stats": stats,
        "meta": m
    }

THREED_SUMMARY_PATH = THREED_ROOT / "3d_dual_summary.json"
THREED_SUMMARY_PATH.write_text(
    json.dumps({
        "mode": "dual_model_3d_no_baseline",
        "results": THREED_RESULTS,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n📌 3D summary saved to:", THREED_SUMMARY_PATH)
for tag, item in THREED_RESULTS.items():
    print(f"{tag} total cells :", item['stats']['total_cells'])

print("\n✅ Cell 14 done.")
print("="*80)


# In[16]:


# Cell 15：总细胞数 + 体积分布 + Z 轴密度对比
import json
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

print("="*80)
print("📊 Cell 15 | 3D dual-model statistics: counts, volume histogram, z-density")
print("="*80)

ANALYSIS_DIR = (THREED_ROOT / "analysis").resolve()
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

assert "THREED_RESULTS" in globals(), "请先跑 Cell 14。"

masks = {}
for tag, item in THREED_RESULTS.items():
    mask_path = Path(item["mask_path"])
    assert mask_path.exists(), f"{tag} mask 不存在: {mask_path}"
    masks[tag] = tiff.imread(str(mask_path))
    print(f"{tag}.shape:", masks[tag].shape, "| dtype:", masks[tag].dtype)

# ========= 1. 总细胞数 =========
count_rows = []
for tag, mask in masks.items():
    total_cells = int(len(np.unique(mask)) - 1)
    count_rows.append({
        "model": tag,
        "total_cells": total_cells
    })

count_df = pd.DataFrame(count_rows).sort_values("model").reset_index(drop=True)
count_csv = ANALYSIS_DIR / "3d_total_cell_counts.csv"
count_df.to_csv(count_csv, index=False)

print("\n✅ 3D total cell counts:")
display(count_df)

# ========= 2. 体积分布 =========
vol_rows = []
vol_summary_rows = []

for tag, mask in masks.items():
    _, counts = np.unique(mask, return_counts=True)
    vols = counts[1:]  # 去掉背景 0

    for v in vols:
        vol_rows.append({
            "model": tag,
            "volume_voxels": int(v)
        })

    vol_summary_rows.append({
        "model": tag,
        "n_instances": int(len(vols)),
        "mean_volume": float(np.mean(vols)) if len(vols) > 0 else 0.0,
        "median_volume": float(np.median(vols)) if len(vols) > 0 else 0.0,
        "std_volume": float(np.std(vols)) if len(vols) > 0 else 0.0,
        "min_volume": int(np.min(vols)) if len(vols) > 0 else 0,
        "max_volume": int(np.max(vols)) if len(vols) > 0 else 0,
        "p5_volume": float(np.percentile(vols, 5)) if len(vols) > 0 else 0.0,
        "p95_volume": float(np.percentile(vols, 95)) if len(vols) > 0 else 0.0,
    })

vol_df = pd.DataFrame(vol_rows)
vol_csv = ANALYSIS_DIR / "3d_instance_volumes_long.csv"
vol_df.to_csv(vol_csv, index=False)

vol_summary_df = pd.DataFrame(vol_summary_rows).sort_values("model").reset_index(drop=True)
vol_summary_csv = ANALYSIS_DIR / "3d_volume_summary.csv"
vol_summary_df.to_csv(vol_summary_csv, index=False)

plt.figure(figsize=(12, 7))
for tag in sorted(masks.keys()):
    vols = vol_df.loc[vol_df["model"] == tag, "volume_voxels"].values
    plt.hist(
        vols,
        bins=60,
        range=(0, 1000),
        histtype='step',
        linewidth=2.5,
        alpha=0.95,
        label=f'{tag}'
    )
    if len(vols) > 0:
        plt.axvline(
            np.mean(vols),
            linestyle='dashed',
            linewidth=2,
            label=f'{tag} mean: {np.mean(vols):.1f}'
        )

plt.title("3D Cell Volume Distribution: R20 vs R21", fontsize=16)
plt.xlabel("Cell Volume (voxels)", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)

vol_fig = ANALYSIS_DIR / "3d_volume_distribution_r20_vs_r21.png"
plt.savefig(vol_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Volume histogram saved:", vol_fig)

# ========= 3. Z 轴密度图 =========
z_df = None
z_rows = []

for tag, mask in masks.items():
    z_counts = [int(len(np.unique(mask[z])) - 1) for z in range(mask.shape[0])]
    for z_idx, cnt in enumerate(z_counts):
        z_rows.append({
            "model": tag,
            "z_index": z_idx,
            "cells_in_slice": cnt
        })

z_long_df = pd.DataFrame(z_rows)
z_csv = ANALYSIS_DIR / "3d_z_axis_density_long.csv"
z_long_df.to_csv(z_csv, index=False)

plt.figure(figsize=(12, 6))
for tag in sorted(masks.keys()):
    g = z_long_df[z_long_df["model"] == tag].sort_values("z_index")
    plt.plot(
        g["z_index"],
        g["cells_in_slice"],
        linewidth=2,
        label=tag
    )
    plt.fill_between(
        g["z_index"],
        g["cells_in_slice"],
        alpha=0.15
    )

plt.title("Whole-brain Cell Density along Z-axis: R20 vs R21", fontsize=16)
plt.xlabel("Z-slice index", fontsize=13)
plt.ylabel("Number of cells in slice", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=11)

z_fig = ANALYSIS_DIR / "3d_z_axis_density_r20_vs_r21.png"
plt.savefig(z_fig, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Z-density plot saved:", z_fig)
print("✅ Counts CSV saved    :", count_csv)
print("✅ Volume CSV saved    :", vol_csv)
print("✅ Volume summary CSV  :", vol_summary_csv)
print("✅ Z-density CSV saved :", z_csv)

print("\n✅ Cell 15 done.")
print("="*80)


# In[18]:


# Cell 16：Slice overlay comparison & local crop analysis for auto-picked top2 (dict-style THREED_RESULTS)
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from pathlib import Path

print("="*80)
print("🖼️ Cell 16 | Slice overlay comparison & local crop analysis for auto-picked top2")
print("="*80)

# ========= 0. 解析 THREED_RESULTS（当前是 dict[tag] -> result） =========
assert "THREED_RESULTS" in globals(), "未找到 THREED_RESULTS，请先运行 3D 推理 cell。"
assert isinstance(THREED_RESULTS, dict), f"当前 THREED_RESULTS 不是 dict，而是 {type(THREED_RESULTS)}"
assert len(THREED_RESULTS) >= 2, f"THREED_RESULTS 数量不足 2，当前只有 {len(THREED_RESULTS)} 个"

# 按 2D rank 排序，保证 left/right 顺序稳定
results_items = []
for tag, info in THREED_RESULTS.items():
    meta = info.get("meta", {})
    rank_2d = meta.get("rank_2d", 999)
    results_items.append((rank_2d, tag, info))

results_items = sorted(results_items, key=lambda x: x[0])

TAG_LEFT  = results_items[0][1]
TAG_RIGHT = results_items[1][1]

print("TAG_LEFT :", TAG_LEFT)
print("TAG_RIGHT:", TAG_RIGHT)

# ========= 1. 构建 mask_dict =========
mask_dict = {
    tag: info["mask_path"]
    for _, tag, info in results_items
}

print("Available tags:", list(mask_dict.keys()))
assert TAG_LEFT in mask_dict and TAG_RIGHT in mask_dict, \
    f"缺少指定 tag，请检查 mask_dict keys: {list(mask_dict.keys())}"

# ========= 2. 读取原始 3D 图像 =========
RAW_STACK_PATH = None
if "RAW_3D_STACK_PATH" in globals():
    RAW_STACK_PATH = Path(RAW_3D_STACK_PATH)
elif "THREED_CONFIG" in globals() and isinstance(THREED_CONFIG, dict) and "raw_3d_stack_path" in THREED_CONFIG:
    RAW_STACK_PATH = Path(THREED_CONFIG["raw_3d_stack_path"])
else:
    RAW_STACK_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/fish7/data/complete_raw/fish7 double 3d_C1.tif")

assert RAW_STACK_PATH.exists(), f"RAW_STACK_PATH 不存在: {RAW_STACK_PATH}"

raw_stack = tiff.imread(str(RAW_STACK_PATH))
mask_left = tiff.imread(mask_dict[TAG_LEFT])
mask_right = tiff.imread(mask_dict[TAG_RIGHT])

print("raw_stack.shape :", raw_stack.shape)
print(f"{TAG_LEFT}.shape :", mask_left.shape)
print(f"{TAG_RIGHT}.shape:", mask_right.shape)

assert raw_stack.shape == mask_left.shape == mask_right.shape, \
    "raw_stack / mask_left / mask_right 形状不一致，请检查 3D 推理输出。"

# ========= 3. 自动选择中间切片 =========
Z_LAYER = raw_stack.shape[0] // 2
print("Using Z_LAYER =", Z_LAYER)

raw_slice = raw_stack[Z_LAYER]
left_slice = mask_left[Z_LAYER]
right_slice = mask_right[Z_LAYER]

# ========= 4. 简单归一化原图，方便显示 =========
raw_float = raw_slice.astype(np.float32)
lo, hi = np.percentile(raw_float, [1, 99])
raw_norm = np.clip((raw_float - lo) / (hi - lo + 1e-8), 0, 1)

# ========= 5. 生成二值轮廓辅助显示 =========
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

# ========= 6. 叠加显示 =========
overlay_left = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)
overlay_right = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

# 左图：红色边界
overlay_left[bd_left] = [1.0, 0.0, 0.0]

# 右图：绿色边界
overlay_right[bd_right] = [0.0, 1.0, 0.0]

# ========= 7. 全图对比 =========
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(raw_norm, cmap="gray")
plt.title(f"Raw image | Z={Z_LAYER}")
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

# ========= 8. 自动找一个中心 crop =========
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
plt.title("Raw crop")
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

# ========= 9. 打印两个模型的 2D / 3D 统计 =========
print("\n" + "="*80)
print("📌 Top2 summary")
print("="*80)

for tag in [TAG_LEFT, TAG_RIGHT]:
    meta = THREED_RESULTS[tag].get("meta", {})
    stats = THREED_RESULTS[tag].get("stats", {})
    print(f"\n[{tag}]")
    print("2D rank     :", meta.get("rank_2d"))
    print("AP50_2d     :", meta.get("AP50_2d"))
    print("F1_2d       :", meta.get("F1_2d"))
    print("Recall_2d   :", meta.get("Recall_2d"))
    print("Precision_2d:", meta.get("Precision_2d"))
    print("total_cells :", stats.get("total_cells"))
    print("elapsed_s   :", stats.get("elapsed_s"))
    print("mask_path   :", THREED_RESULTS[tag].get("mask_path"))


# In[19]:


# Cell 17：打包 3D 结果
import json
import zipfile
from pathlib import Path
from datetime import datetime

print("="*80)
print("📦 Cell 17 | Bundle R20/R21 3D fullbrain results")
print("="*80)

ZIP_DIR = THREED_ROOT / "zips"
ZIP_DIR.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = ZIP_DIR / f"3d_r20_r21_bundle_{stamp}.zip"

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
    n_added += add_file(zf, THREED_CONFIG_PATH, "config/3d_dual_run_config.json")
    n_added += add_file(zf, THREED_SUMMARY_PATH, "summary/3d_dual_summary.json")
    n_added += add_file(zf, RANK_CSV, "summary/final_merged_ranking.csv")

    for m in MODELS_3D:
        out_dir = Path(m["out_dir"])
        n_added += add_dir(zf, out_dir, f"results/{m['tag']}_3d")

    n_added += add_dir(zf, ANALYSIS_DIR, "analysis")
    n_added += add_dir(zf, VIS_DIR, "visuals")

print("✅ Added files:", n_added)
print("📦 ZIP saved to:", zip_path)
print("\n✅ Cell 17 done.")
print("="*80)


# In[1]:


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


# In[1]:


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


# In[5]:


from pathlib import Path

runs_root = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs")
cand = sorted(runs_root.glob("exp_20260311_ori_train_9runs_400ep_es50_*"))

print("匹配到的目录：")
for p in cand:
    print(p)


# In[11]:


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


# In[2]:


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


# In[3]:


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
# Cell New-2：合并多轮 sweep，选前2名跑全脑 3D
# ==========================================
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from cellpose import models, io

print("=" * 100)
print("🚀 Cell New-2 | Merge sweep summaries and run full-brain 3D for top-2")
print("=" * 100)

io.logger_setup()

# --------------------------------------------------
# 0) 路径
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

BASELINE_MASK_PATH = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/fish7/inference_results/3D_Comparison_Native/Baseline_cpsam/full_brain_3d_masks.tif"
).resolve()

BASELINE_NAME = "Baseline_cpsam"

assert ROOT.exists()
assert EXP_DIR.exists()
assert CFG_DIR.exists()
assert RAW_3D_STACK_PATH.exists()
assert BASELINE_MASK_PATH.exists()

# --------------------------------------------------
# 1) 汇总表路径
# --------------------------------------------------
OLD_SUMMARY_CSV = (
    EXP_DIR / "sweep_3d_compare" / f"summary_3d_sweep_{TARGET_TAG}.csv"
).resolve()

NEW_SUMMARY_CSV = (
    EXP_DIR / "sweep_3d_compare" / "sweep_cp_v2_a1p5_2p5_d7_9_cp-1_1" / f"summary_3d_sweep_{TARGET_TAG}.csv"
).resolve()

print("OLD_SUMMARY_CSV:", OLD_SUMMARY_CSV)
print("NEW_SUMMARY_CSV:", NEW_SUMMARY_CSV)

assert OLD_SUMMARY_CSV.exists(), f"旧 summary 不存在: {OLD_SUMMARY_CSV}"
assert NEW_SUMMARY_CSV.exists(), f"新 summary 不存在: {NEW_SUMMARY_CSV}"

# --------------------------------------------------
# 2) 工具函数
# --------------------------------------------------
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
# 3) 找模型
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

assert selected_model_path is not None, f"没找到 {TARGET_TAG} 的模型"

print("✅ selected_model_path   :", selected_model_path)
print("✅ selected_snapshot_path:", selected_snapshot_path)

# --------------------------------------------------
# 4) 合并旧/新 summary
# --------------------------------------------------
old_df = pd.read_csv(OLD_SUMMARY_CSV)
new_df = pd.read_csv(NEW_SUMMARY_CSV)

if "cellprob_threshold" not in old_df.columns:
    old_df["cellprob_threshold"] = 0.0
if "stitch_threshold" not in old_df.columns:
    old_df["stitch_threshold"] = 0.0

if "cellprob_threshold" not in new_df.columns:
    new_df["cellprob_threshold"] = 0.0
if "stitch_threshold" not in new_df.columns:
    new_df["stitch_threshold"] = 0.0

old_df["source_summary"] = str(OLD_SUMMARY_CSV)
new_df["source_summary"] = str(NEW_SUMMARY_CSV)

merged_df = pd.concat([old_df, new_df], ignore_index=True)

# 去重：同一参数只保留一条
merged_df = merged_df.drop_duplicates(
    subset=["anisotropy", "diameter", "cellprob_threshold", "stitch_threshold"],
    keep="first"
).copy()

merged_df = merged_df.sort_values(
    ["large_obj_ratio_ge_2x_median", "total_cells", "median_volume"],
    ascending=[True, False, True]
).reset_index(drop=True)

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
MERGE_ROOT.mkdir(parents=True, exist_ok=True)

merged_csv = MERGE_ROOT / f"merged_sweep_ranking_{TARGET_TAG}.csv"
merged_df.to_csv(merged_csv, index=False)

print("✅ merged ranking saved:", merged_csv)

display_cols = [
    "anisotropy", "diameter", "cellprob_threshold", "stitch_threshold",
    "total_cells", "median_volume", "p90_volume", "max_volume",
    "large_obj_ratio_ge_2x_median", "source_summary"
]
display(merged_df[display_cols].head(10))

# --------------------------------------------------
# 5) 选前2名
# --------------------------------------------------
top2_df = merged_df.head(2).copy()
assert len(top2_df) >= 1, "没有可用参数组合"

top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
top2_df.to_csv(top2_csv, index=False)

print("✅ top2 params saved:", top2_csv)
display(top2_df[display_cols])

# --------------------------------------------------
# 6) 读取 full brain / baseline
# --------------------------------------------------
stack = tiff.imread(str(RAW_3D_STACK_PATH))
baseline_masks = tiff.imread(str(BASELINE_MASK_PATH))

assert stack.ndim == 3
assert baseline_masks.shape == stack.shape

Z_VIS = 100

# --------------------------------------------------
# 7) 加载模型
# --------------------------------------------------
model = models.CellposeModel(
    gpu=True,
    pretrained_model=selected_model_path
)
print("✅ model loaded")

# --------------------------------------------------
# 8) 跑前2名 full brain 3D
# --------------------------------------------------
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()
FIG_ROOT = (FULL_ROOT / "figures").resolve()

MASK_ROOT.mkdir(parents=True, exist_ok=True)
STAT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_ROOT.mkdir(parents=True, exist_ok=True)

summary_rows = []

for rank_idx, (_, row) in enumerate(top2_df.iterrows(), 1):
    anisotropy = float(row["anisotropy"])
    diameter = float(row["diameter"])
    cellprob = float(row["cellprob_threshold"])
    stitch = float(row["stitch_threshold"])

    run_name = f"FINAL_rank{rank_idx}__" + run_name_from_params(
        TARGET_TAG, anisotropy, diameter, cellprob, stitch
    )

    out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
    out_stats_path = STAT_ROOT / f"{run_name}__stats.json"
    out_fig_path = FIG_ROOT / f"{run_name}__compare_z{Z_VIS}.png"

    print("\n" + "=" * 100)
    print(f"🚀 Running full brain rank#{rank_idx}")
    print(f"anisotropy={anisotropy}, diameter={diameter}, cellprob={cellprob}, stitch={stitch}")
    print("=" * 100)

    if out_mask_path.exists():
        print("✅ 已存在，直接复用:", out_mask_path)
        masks = tiff.imread(str(out_mask_path))
        elapsed_s = np.nan
    else:
        t0 = time.time()
        masks, _, _ = model.eval(
            stack,
            diameter=diameter,
            do_3D=True,
            stitch_threshold=stitch,
            z_axis=0,
            batch_size=16,
            anisotropy=anisotropy,
            cellprob_threshold=cellprob,
            progress=True
        )
        masks = np.asarray(masks)
        elapsed_s = float(time.time() - t0)
        tiff.imwrite(str(out_mask_path), masks.astype(np.uint32))
        print(f"✅ full brain 3D 完成，耗时 {elapsed_s/60:.2f} min")

    stats = summarize_mask(masks)

    raw_mid = stack[Z_VIS]
    base_mid = baseline_masks[Z_VIS]
    this_mid = masks[Z_VIS]

    raw_norm = normalize_img(raw_mid)
    base_overlay = overlay_boundary(raw_mid, base_mid, color="red")
    this_overlay = overlay_boundary(raw_mid, this_mid, color="green")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(raw_norm, cmap="gray")
    axes[0].set_title(f"Raw | z={Z_VIS}")
    axes[0].axis("off")

    axes[1].imshow(base_overlay)
    axes[1].set_title("Baseline_cpsam")
    axes[1].axis("off")

    axes[2].imshow(this_overlay)
    axes[2].set_title(
        f"rank#{rank_idx}\n"
        f"a={anisotropy}, d={diameter}, cp={cellprob}, s={stitch}\n"
        f"cells={stats['total_cells']} | medV={stats['median_volume']:.1f} | "
        f"largeR={stats['large_obj_ratio_ge_2x_median']:.3f}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    stats_record = {
        "rank_idx": rank_idx,
        "target_tag": TARGET_TAG,
        "selected_model_path": selected_model_path,
        "selected_snapshot_path": selected_snapshot_path,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "baseline_mask_path": str(BASELINE_MASK_PATH),
        "mask_path": str(out_mask_path),
        "fig_path": str(out_fig_path),
        "shape": list(masks.shape),
        "elapsed_s": None if pd.isna(elapsed_s) else elapsed_s,
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

    summary_rows.append({
        "rank_idx": rank_idx,
        "run_name": run_name,
        "anisotropy": anisotropy,
        "diameter": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "elapsed_s": elapsed_s,
        "mask_path": str(out_mask_path),
        "fig_path": str(out_fig_path),
        **stats
    })

full_summary_df = pd.DataFrame(summary_rows)
full_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"
full_summary_df.to_csv(full_summary_csv, index=False)

print("\n✅ full brain top2 summary saved:", full_summary_csv)
display(full_summary_df)


# In[1]:


# ==========================================
# Cell Rescue-1：检查进程 / 已生成文件 / 恢复现场
# ==========================================
from pathlib import Path
import subprocess
import pandas as pd
import json

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()
TARGET_TAG = "P21_lr9e5_wd8e3"

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
FULL_ROOT = (MERGE_ROOT / TARGET_TAG).resolve()
MASK_ROOT = (FULL_ROOT / "masks").resolve()
STAT_ROOT = (FULL_ROOT / "stats").resolve()
FIG_ROOT = (FULL_ROOT / "figures").resolve()

merged_csv = MERGE_ROOT / f"merged_sweep_ranking_{TARGET_TAG}.csv"
top2_csv = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"
full_summary_csv = MERGE_ROOT / f"full_brain_top2_summary_{TARGET_TAG}.csv"

print("=" * 100)
print("🔎 Rescue check")
print("=" * 100)
print("MERGE_ROOT       :", MERGE_ROOT)
print("FULL_ROOT        :", FULL_ROOT)
print("MASK_ROOT exists :", MASK_ROOT.exists(), MASK_ROOT)
print("STAT_ROOT exists :", STAT_ROOT.exists(), STAT_ROOT)
print("FIG_ROOT exists  :", FIG_ROOT.exists(), FIG_ROOT)
print("merged_csv exists:", merged_csv.exists(), merged_csv)
print("top2_csv exists  :", top2_csv.exists(), top2_csv)
print("full_summary exists:", full_summary_csv.exists(), full_summary_csv)

print("\n" + "=" * 100)
print("🧠 Python / cellpose related processes")
print("=" * 100)
cmd = r"ps -ef | grep -E 'jupyter|ipykernel|python|cellpose' | grep -v grep"
ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(ret.stdout if ret.stdout.strip() else "(没有发现相关进程)")

print("\n" + "=" * 100)
print("📂 Existing full-brain outputs")
print("=" * 100)
if MASK_ROOT.exists():
    masks = sorted(MASK_ROOT.glob("*.tif"))
    for p in masks:
        print("MASK:", p.name, "| size:", p.stat().st_size)
else:
    print("(MASK_ROOT 不存在)")

if STAT_ROOT.exists():
    stats = sorted(STAT_ROOT.glob("*.json"))
    for p in stats:
        print("STAT:", p.name, "| size:", p.stat().st_size)
else:
    print("(STAT_ROOT 不存在)")

if FIG_ROOT.exists():
    figs = sorted(FIG_ROOT.glob("*.png"))
    for p in figs:
        print("FIG :", p.name, "| size:", p.stat().st_size)
else:
    print("(FIG_ROOT 不存在)")

print("\n" + "=" * 100)
print("🏁 top2 params preview")
print("=" * 100)
if top2_csv.exists():
    df = pd.read_csv(top2_csv)
    display(df)
else:
    print("(top2_csv 不存在)")


# In[1]:


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


# In[5]:


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





# In[3]:


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


# In[6]:


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


# In[7]:


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
BATCH_SIZE_3D = 4   # 如果还炸，再改成 2

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


# In[1]:


import subprocess
print(subprocess.run(["which", "sbatch"], capture_output=True, text=True).stdout)


# In[ ]:





# In[8]:


from pathlib import Path
import subprocess
import textwrap
from datetime import datetime

# =========================================================
# 0) 路径配置：按你的项目改
# =========================================================
PROJECT_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

SCRIPT_DIR = (EXP_DIR / "auto_submit_scripts").resolve()
LOG_DIR = (EXP_DIR / "slurm_logs").resolve()

SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

py_script_path = SCRIPT_DIR / "baseline_stepA_cpsam_rank1.py"
sbatch_path = SCRIPT_DIR / "baseline_stepA_cpsam_rank1.sbatch"

# =========================================================
# 1) 生成真正执行任务的 Python 脚本
# =========================================================
py_code = textwrap.dedent(r'''
    # ==========================================
    # Auto-generated by Jupyter:
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
    BATCH_SIZE_3D = 4

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
    print("\\n[Step 1/4] 读取 rank1 参数")
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
    out_running_path = STAT_ROOT / f"{run_name}__RUNNING.json"

    print("\\n" + "=" * 100)
    print("Baseline(cpsam) with rank1 params")
    print("mask exists:", out_mask_path.exists(), out_mask_path)
    print("=" * 100)

    # 先写一个 running 记录
    running_record = {
        "status": "RUNNING",
        "model_name": "cpsam",
        "target_tag": TARGET_TAG,
        "raw_3d_stack_path": str(RAW_3D_STACK_PATH),
        "mask_path": str(out_mask_path),
        "anisotropy": anisotropy,
        "diameter_3d": diameter,
        "cellprob_threshold": cellprob,
        "stitch_threshold": stitch,
        "step": "A_save_mask_only"
    }
    out_running_path.write_text(
        json.dumps(running_record, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

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

        tmp_mask_path = out_mask_path.with_suffix(".tmp.tif")
        print("写出 baseline mask tif 到临时文件:", tmp_mask_path)

        if np.issubdtype(masks.dtype, np.integer):
            tiff.imwrite(str(tmp_mask_path), masks)
        else:
            tiff.imwrite(str(tmp_mask_path), masks.astype(np.uint32))

        tmp_mask_path.replace(out_mask_path)
        print("✅ baseline mask tif saved:", out_mask_path)

        del flows, styles
        safe_cuda_cleanup()

    minimal_record = {
        "status": "DONE",
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

    print("\\n✅ StepA summary saved:", stepA_summary_csv)

    # 如果运行成功，把 RUNNING 标记删掉
    if out_running_path.exists():
        out_running_path.unlink()

    print("✅ Job finished successfully.")
''').strip() + "\n"

py_script_path.write_text(py_code, encoding="utf-8")
print(f"✅ Python 脚本已写出: {py_script_path}")

# =========================================================
# 2) 生成 sbatch 脚本
# =========================================================
sbatch_code = textwrap.dedent(f'''\
    #!/bin/bash
    #SBATCH -J cpsam_baseA
    #SBATCH -p GPUA800
    #SBATCH --gres=gpu:1
    #SBATCH -c 8
    #SBATCH --mem=192G
    #SBATCH -t 12:00:00
    #SBATCH -o {LOG_DIR}/cpsam_baseA_%j.out
    #SBATCH -e {LOG_DIR}/cpsam_baseA_%j.err

    source ~/.bashrc
    micromamba activate cpsm

    cd {PROJECT_ROOT}
    python {py_script_path}
''')

sbatch_path.write_text(sbatch_code, encoding="utf-8")
print(f"✅ sbatch 脚本已写出: {sbatch_path}")

# =========================================================
# 3) 提交 sbatch
# =========================================================
res = subprocess.run(
    ["sbatch", str(sbatch_path)],
    capture_output=True,
    text=True
)

print("=" * 100)
print("sbatch return code:", res.returncode)
print("stdout:")
print(res.stdout)
print("stderr:")
print(res.stderr)
print("=" * 100)

if res.returncode != 0:
    raise RuntimeError("❌ sbatch 提交失败，请检查 stderr")

print("✅ 已成功提交作业。")


# In[6]:


# ==========================================
# Monitor Cell | 一键监控 Slurm 作业 + 日志 + 输出文件
# ==========================================
from pathlib import Path
import subprocess
import os
import time
from datetime import datetime

# --------------------------------------------------
# 0) 你要改的配置
# --------------------------------------------------
JOB_ID = "1318473"   # ← 改成你的 job id

EXP_DIR = Path(
    "/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260311_ori_train_9runs_400ep_es50_20260311_200828"
).resolve()

TARGET_TAG = "P21_lr9e5_wd8e3"
RUN_ONLY_RANK = 1

MERGE_ROOT = (EXP_DIR / "full3d_top2_from_merged_sweeps").resolve()
BASELINE_ROOT = (MERGE_ROOT / "Baseline_cpsam_rank1params").resolve()
MASK_ROOT = (BASELINE_ROOT / "masks").resolve()
STAT_ROOT = (BASELINE_ROOT / "stats").resolve()
LOG_DIR = (EXP_DIR / "slurm_logs").resolve()

TOP2_CSV = MERGE_ROOT / f"top2_params_{TARGET_TAG}.csv"

SHOW_TAIL_LINES = 80   # 显示日志最后多少行


# --------------------------------------------------
# 1) 小工具函数
# --------------------------------------------------
def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": repr(e),
        }

def safe_read_text(path: Path, max_chars=None):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            return text[-max_chars:]
        return text
    except Exception as e:
        return f"[读取失败] {e}"

def tail_file(path: Path, n=50):
    if not path.exists():
        return f"[文件不存在] {path}"
    cmd = ["tail", "-n", str(n), str(path)]
    res = run_cmd(cmd)
    if res["ok"]:
        return res["stdout"]
    return f"[tail失败]\n{res['stderr']}"

def fmt_mtime(path: Path):
    if not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

def fmt_size(path: Path):
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.2f} {units[i]}"

def run_name_from_params(tag, anisotropy, diameter, cellprob, stitch):
    a = str(anisotropy).replace(".", "p")
    d = str(diameter).replace(".", "p")
    cp = str(cellprob).replace(".", "p").replace("-", "m")
    s = str(stitch).replace(".", "p")
    return f"{tag}__a{a}__d{d}__cp{cp}__s{s}"


# --------------------------------------------------
# 2) 根据 top2_csv 自动推导输出文件名
# --------------------------------------------------
out_mask_path = None
running_json_path = None
final_json_path = None

if TOP2_CSV.exists():
    try:
        import pandas as pd
        top2_df = pd.read_csv(TOP2_CSV)
        assert len(top2_df) >= RUN_ONLY_RANK, f"top2_csv 行数不足，RUN_ONLY_RANK={RUN_ONLY_RANK}"

        row = top2_df.iloc[RUN_ONLY_RANK - 1]
        anisotropy = float(row["anisotropy"])
        diameter = float(row["diameter"])
        cellprob = float(row["cellprob_threshold"])
        stitch = float(row["stitch_threshold"])

        run_name = "BASELINE_cpsam__rank1params__" + run_name_from_params(
            TARGET_TAG, anisotropy, diameter, cellprob, stitch
        )

        out_mask_path = MASK_ROOT / f"{run_name}__full_brain_3d_masks.tif"
        running_json_path = STAT_ROOT / f"{run_name}__RUNNING.json"
        final_json_path = STAT_ROOT / f"{run_name}__stepA_minimal.json"

    except Exception as e:
        print("❌ 解析 top2_csv 失败：", e)
else:
    print("⚠️ TOP2_CSV 不存在：", TOP2_CSV)


# --------------------------------------------------
# 3) 查找本 job 的日志文件
# --------------------------------------------------
out_log = LOG_DIR / f"cpsam_baseA_{JOB_ID}.out"
err_log = LOG_DIR / f"cpsam_baseA_{JOB_ID}.err"


# --------------------------------------------------
# 4) 输出监控信息
# --------------------------------------------------
print("=" * 120)
print("📡 Slurm Job Monitor")
print("=" * 120)
print("当前时间         :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("JOB_ID          :", JOB_ID)
print("EXP_DIR         :", EXP_DIR)
print("LOG_DIR         :", LOG_DIR)
print("TOP2_CSV exists :", TOP2_CSV.exists(), TOP2_CSV)
print()

# ---------------------------
# 4.1 squeue
# ---------------------------
print("-" * 120)
print("1) squeue")
print("-" * 120)
res = run_cmd(["squeue", "-j", str(JOB_ID)])
if res["stdout"]:
    print(res["stdout"])
else:
    print("[无输出]")
    if res["stderr"]:
        print("stderr:", res["stderr"])
print()

# ---------------------------
# 4.2 sacct
# ---------------------------
print("-" * 120)
print("2) sacct")
print("-" * 120)
res = run_cmd(["sacct", "-j", str(JOB_ID), "--format=JobID,JobName,Partition,AllocCPUS,State,Elapsed,MaxRSS,ExitCode"])
if res["stdout"]:
    print(res["stdout"])
else:
    print("[无输出]")
    if res["stderr"]:
        print("stderr:", res["stderr"])
print()

# ---------------------------
# 4.3 日志文件状态
# ---------------------------
print("-" * 120)
print("3) 日志文件状态")
print("-" * 120)
for p in [out_log, err_log]:
    print(f"{p.name}")
    print("  exists :", p.exists())
    print("  size   :", fmt_size(p))
    print("  mtime  :", fmt_mtime(p))
print()

# ---------------------------
# 4.4 最新 .out
# ---------------------------
print("-" * 120)
print(f"4) 最新 .out 最后 {SHOW_TAIL_LINES} 行")
print("-" * 120)
print(tail_file(out_log, SHOW_TAIL_LINES))
print()

# ---------------------------
# 4.5 最新 .err
# ---------------------------
print("-" * 120)
print(f"5) 最新 .err 最后 {SHOW_TAIL_LINES} 行")
print("-" * 120)
print(tail_file(err_log, SHOW_TAIL_LINES))
print()

# ---------------------------
# 4.6 当前 GPU 占用情况
# ---------------------------
print("-" * 120)
print("6) 当前 GPU 占用情况 (nvidia-smi)")
print("-" * 120)

res = run_cmd([
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits"
])

if res["ok"] and res["stdout"]:
    lines = [x.strip() for x in res["stdout"].splitlines() if x.strip()]
    print("index | name | mem_total_MB | mem_used_MB | mem_free_MB | gpu_util_% | temp_C")
    for line in lines:
        print(line)
else:
    print("[无法获取 GPU 信息]")
    if res["stderr"]:
        print("stderr:", res["stderr"])
print()

# ---------------------------
# 4.7 当前 GPU 进程占用情况
# ---------------------------
print("-" * 120)
print("7) 当前 GPU 进程占用情况")
print("-" * 120)

res = run_cmd([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits"
])

if res["ok"] and res["stdout"]:
    lines = [x.strip() for x in res["stdout"].splitlines() if x.strip()]
    if len(lines) == 0:
        print("[当前没有 compute 进程占用 GPU]")
    else:
        print("gpu_uuid | pid | process_name | used_memory_MB")
        for line in lines:
            print(line)
else:
    print("[无法获取 GPU 进程信息，可能当前节点无 GPU / 无权限 / 没有 compute 进程]")
    if res["stderr"]:
        print("stderr:", res["stderr"])
print()

# ---------------------------
# 4.6 输出文件状态
# ---------------------------
print("-" * 120)
print("6) 输出文件状态")
print("-" * 120)

if out_mask_path is None:
    print("⚠️ 由于前面未成功解析 run_name，暂时无法自动定位输出文件。")
else:
    check_paths = [
        ("MASK tif", out_mask_path),
        ("RUNNING json", running_json_path),
        ("FINAL minimal json", final_json_path),
    ]
    for name, p in check_paths:
        print(f"{name}")
        print("  exists :", p.exists())
        print("  size   :", fmt_size(p))
        print("  mtime  :", fmt_mtime(p))
        print("  path   :", p)
        print()

# ---------------------------
# 4.7 粗略健康判断
# ---------------------------
print("-" * 120)
print("7) 粗略判断")
print("-" * 120)

job_running = False
res_sq = run_cmd(["squeue", "-j", str(JOB_ID)])
if str(JOB_ID) in res_sq["stdout"]:
    job_running = True

mask_exists = (out_mask_path is not None and out_mask_path.exists())
running_exists = (running_json_path is not None and running_json_path.exists())
final_json_exists = (final_json_path is not None and final_json_path.exists())

if job_running and not mask_exists:
    print("🟡 作业还在队列/运行中，最终 mask 还没出来。大概率还在跑。")
elif job_running and mask_exists:
    print("🟢 作业还在运行，同时 mask 已存在。可能正在收尾，或结果已提前写出。")
elif (not job_running) and mask_exists and final_json_exists:
    print("✅ 作业大概率已正常完成：队列里没有，但最终 mask 和 final json 都已存在。")
elif (not job_running) and running_exists and (not final_json_exists):
    print("🔴 队列里没了，但 RUNNING.json 还在、final json 不在。像是中途中断了。")
elif (not job_running) and (not mask_exists):
    print("🔴 队列里没了，而且 mask 也没看到。大概率失败或根本没成功启动。")
else:
    print("🟠 状态不够明确，建议结合 sacct + out/err 日志一起判断。")


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




