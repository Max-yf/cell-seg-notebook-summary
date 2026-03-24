#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    print("   训练请务必在计算节点跑（salloc/srun/sbatch）。")
else:
    print("\n✅ 节点看起来不像登录节点（仍建议你确认是计算节点）。")

# ---- cellpose 可用性自检 ----
try:
    import cellpose
    print("\n✅ cellpose import OK | version:", getattr(cellpose, "__version__", "unknown"))
except Exception as e:
    print("\n❌ cellpose import FAILED:", repr(e))
    print("=> ⚠️ 别往下跑了！先把 notebook kernel 切到 cpsm 环境再继续。")
    raise

# ---- CUDA / GPU 自检（不依赖 torch 也能看个大概）----
def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return 999, "", repr(e)

code, out, err = _run(["nvidia-smi", "-L"])
if code == 0 and out:
    print("\n🟢 GPU detected (nvidia-smi -L):")
    print(out)
else:
    print("\n⚠️ 没找到 nvidia-smi 或当前无 GPU 可见。")
    if err:
        print("stderr:", err[:300])

# ---- Git 信息快照（可选，但很利于复现）----
def get_git_snapshot():
    # 如果当前目录不在 git repo，会返回 None
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

from importlib.metadata import version
print("cellpose_version(meta):", version("cellpose"))
print("torch_version:", version("torch"))


# In[3]:


# Cell 1：路径常量（固定不动）+ 实验目录初始化
from pathlib import Path
from datetime import datetime
import os, json

print("="*80)
print("🧱 Cell 1 | Paths & experiment directory init")
print("="*80)

# 你的项目根目录（按你现在工程习惯）
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

# 这次重新开一个“多版本大炼丹”实验目录（自动时间戳，避免覆盖）
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"exp_20260305_h100_sweep_{STAMP}"   # 你想改名字就在这改
EXP_DIR = (ROOT / "runs" / EXP_NAME).resolve()

# 标准输出目录（后续所有东西都往这里收）
LOG_DIR     = EXP_DIR / "logs"
MET_DIR     = EXP_DIR / "metrics"
CFG_DIR     = EXP_DIR / "config"
INFER_DIR   = EXP_DIR / "infer"
EVAL_DIR    = EXP_DIR / "eval"
EXPORT_DIR  = EXP_DIR / "exports"
DELIV_DIR   = EXP_DIR / "delivery"

for d in [LOG_DIR, MET_DIR, CFG_DIR, INFER_DIR, EVAL_DIR, EXPORT_DIR, DELIV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 把关键路径写一份 json，断线/重连也能瞬间找回来
RUN_INDEX = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "ROOT": str(ROOT),
    "EXP_DIR": str(EXP_DIR),
    "LOG_DIR": str(LOG_DIR),
    "MET_DIR": str(MET_DIR),
    "CFG_DIR": str(CFG_DIR),
    "INFER_DIR": str(INFER_DIR),
    "EVAL_DIR": str(EVAL_DIR),
    "EXPORT_DIR": str(EXPORT_DIR),
    "DELIV_DIR": str(DELIV_DIR),
    "note": "This experiment is designed for multi-run H100 sweep with robust logging.",
}
(PATH_INDEX := (CFG_DIR / "PATHS.json")).write_text(
    json.dumps(RUN_INDEX, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("ROOT:", ROOT)
print("EXP_DIR:", EXP_DIR)
print("✅ dirs ready:")
print(" - LOG_DIR    :", LOG_DIR)
print(" - MET_DIR    :", MET_DIR)
print(" - CFG_DIR    :", CFG_DIR)
print(" - INFER_DIR  :", INFER_DIR)
print(" - EVAL_DIR   :", EVAL_DIR)
print(" - EXPORT_DIR :", EXPORT_DIR)
print(" - DELIV_DIR  :", DELIV_DIR)
print("\n📌 PATHS.json saved to:", PATH_INDEX)

# ⚠️ 数据集根目录（先在这里统一声明，后面 Cell 2 只引用）
DATA_ROOT = (ROOT / "Cellpose2TrainDataset").resolve()
TRAIN_DIR = (DATA_ROOT / "trainset_ft_os3").resolve()
VAL_DIR   = (DATA_ROOT / "valset").resolve()

print("\n📂 Dataset paths:")
print(" - DATA_ROOT:", DATA_ROOT)
print(" - TRAIN_DIR:", TRAIN_DIR, "| exists:", TRAIN_DIR.exists())
print(" - VAL_DIR  :", VAL_DIR,   "| exists:", VAL_DIR.exists())

assert ROOT.exists(), f"ROOT not found: {ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
assert TRAIN_DIR.exists(), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert VAL_DIR.exists(), f"VAL_DIR not found: {VAL_DIR}"

print("\n✅ Cell 1 done.")
print("="*80)


# In[4]:


# Cell 2：多版本 Sweep 参数表 + 批量 Commit（一次性生成 6 个 run）
import json
from pathlib import Path
from datetime import datetime

# ============ 统一默认参数（你只改这里的大方向）============
BASE = {
    "train_dir": str(TRAIN_DIR),
    "val_dir":   str(VAL_DIR),
    "mask_filter": "_masks",

    "pretrained_model": "cpsam",

    # 训练与保存策略（抗断线：save_every 小一点）
    "save_every": 5,
    "use_gpu": True,
    "verbose": True,

    # patch size（保持你现有 pipeline）
    "bsize": 256,
}
# ============================================================

# ============ 6 个版本：性格明显不同（第一轮筛 Top2）===========
SWEEP = [
    # V0：基线复刻（稳）
    dict(tag="V0_baseline_repro", diameter=20, learning_rate=1e-4, weight_decay=1e-2, train_batch_size=32, n_epochs=100, augment=False),

    # V1：低 lr + 长训 + augment（抗过拟合）
    dict(tag="V1_lowLR_long_aug", diameter=20, learning_rate=5e-5, weight_decay=1e-2, train_batch_size=32, n_epochs=160, augment=True),

    # V2：中 batch（稳定收敛，64爆炸了，别用）
    dict(tag="V2_bigBatch_aug", diameter=20, learning_rate=1e-4, weight_decay=1e-2, train_batch_size=48, n_epochs=100, augment=True),

    # V3：小 wd（更敢切边界，但可能 FP↑）
    dict(tag="V3_lowWD_aug", diameter=20, learning_rate=1e-4, weight_decay=1e-3, train_batch_size=32, n_epochs=100, augment=True),

    # V4：diam 偏小（更倾向拆分粘连）
    dict(tag="V4_diam16_aug", diameter=16, learning_rate=1e-4, weight_decay=1e-2, train_batch_size=32, n_epochs=100, augment=True),

    # V5：diam 偏大（更倾向整体一致性）
    dict(tag="V5_diam24_aug", diameter=24, learning_rate=1e-4, weight_decay=1e-2, train_batch_size=32, n_epochs=100, augment=True),

    # V6：Transformer 对照（在“最抗过拟合”的 V1 上打开 transformer）
    # 目的：把结构开关的影响测干净，避免被 lr/wd/batch 这些混杂因素污染
    dict(tag="V6_transformer_on", diameter=20, learning_rate=5e-5, weight_decay=1e-2, train_batch_size=32, n_epochs=160, augment=True, transformer=True),
]
# ============================================================

RUNS = []  # 本轮所有 run 的上下文都在这（后面 Cell 3 用）
RUN_INDEX_PATH = CFG_DIR / "RUNS.jsonl"  # 每个 run 一行 json（断线也不怕）
RUN_INDEX_PATH.touch(exist_ok=True)

def commit_run(base: dict, variant: dict) -> dict:
    p = dict(base)
    p.update(variant)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{p['tag']}_d{p['diameter']}_lr{p['learning_rate']}_wd{p['weight_decay']}_bs{p['train_batch_size']}_e{p['n_epochs']}_{ts}"

    ctx = {
        "run_name": run_name,
        "tag": p["tag"],
        "params": p,
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "log_path": str(LOG_DIR / f"train_{run_name}.log"),
        "pid_path": str(LOG_DIR / f"train_{run_name}.pid"),
        "metrics_path": str(MET_DIR / f"metrics_{run_name}.csv"),
        "cmd_path": str(CFG_DIR / f"cmd_{run_name}.txt"),
        "config_snapshot_path": str(CFG_DIR / f"config_{run_name}.json"),
    }

    # 保存 config snapshot
    Path(ctx["config_snapshot_path"]).write_text(
        json.dumps(ctx, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 追加写入总索引（jsonl）
    with open(RUN_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ctx, ensure_ascii=False) + "\n")

    return ctx

for v in SWEEP:
    ctx = commit_run(BASE, v)
    RUNS.append(ctx)

print(f"✅ 已生成 {len(RUNS)} 个 runs")
print("📌 RUNS index:", RUN_INDEX_PATH)
print("\n--- Runs preview (name | tag | epochs | batch | lr | wd | diam | augment) ---")
for r in RUNS:
    p = r["params"]
    print(f"- {r['run_name']}")
    print(f"  tag={r['tag']} | e={p['n_epochs']} | bs={p['train_batch_size']} | lr={p['learning_rate']} | wd={p['weight_decay']} | d={p['diameter']} | augment={p.get('augment', False)}")


# In[ ]:





# In[2]:


#网卡了运行这个，（如果cell3提示先运行cell2的话）
import json
from pathlib import Path

# ✅ 改成你实际的 exp 目录（你日志里那个）
EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_120333")
LOG_DIR = EXP_DIR / "logs"
CFG_DIR = EXP_DIR / "cfg"
MET_DIR = EXP_DIR / "metrics"

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

RUN_INDEX_PATH = CFG_DIR / "RUNS.jsonl"
assert RUN_INDEX_PATH.exists(), f"RUNS.jsonl 不存在：{RUN_INDEX_PATH}"

RUNS = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

print("✅ 恢复完成")
print("EXP_DIR:", EXP_DIR)
print("RUNS:", len(RUNS))
print("First run:", RUNS[0]["run_name"] if RUNS else None)


# In[3]:


import subprocess
from pathlib import Path

def ps_stat(pid: int) -> str:
    r = subprocess.run(["ps","-o","pid,ppid,stat,etime,cmd","-p",str(pid)],
                       text=True, capture_output=True)
    return (r.stdout or r.stderr).strip()

alive = []
for r in RUNS:
    p = Path(r["pid_path"])
    if not p.exists():
        continue
    try:
        pid = int(p.read_text().strip())
    except Exception:
        continue
    out = ps_stat(pid)
    if out and "Z" not in out:   # Z = zombie
        alive.append((r["run_name"], pid, out))

print("Alive non-zombie processes:", len(alive))
for name, pid, out in alive[:10]:
    print("\n---", name, "PID=", pid, "---")
    print(out)
    


# In[4]:


from pathlib import Path

RUNS_ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs")
candidates = sorted(RUNS_ROOT.glob("**/RUNS.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

print("Found RUNS.jsonl:", len(candidates))
for p in candidates[:20]:
    print(p, "| mtime:", p.stat().st_mtime)


# In[5]:


import json
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
CFG_DIR = RUN_INDEX_PATH.parent
EXP_DIR = CFG_DIR.parent
LOG_DIR = EXP_DIR / "logs"
MET_DIR = EXP_DIR / "metrics"
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

RUNS = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

print("✅ RUNS 恢复完成:", len(RUNS))
print("EXP_DIR:", EXP_DIR)
print("First run:", RUNS[0]["run_name"] if RUNS else None)
print("Last  run:", RUNS[-1]["run_name"] if RUNS else None)


# In[6]:


from pathlib import Path
import subprocess, re

# 1) 统计 run.log 里完成次数
runlog = Path("/gpfs/share/home/2306391536/.cellpose/run.log")
done_lines = subprocess.run(["grep","-n","model trained and saved to", str(runlog)], text=True, capture_output=True).stdout.strip().splitlines()
print("✅ run.log 中完成记录条数:", len([l for l in done_lines if l.strip()]))
print("last 5:")
for l in done_lines[-5:]:
    print(" ", l)

# 2) 列出 train_dir 下 models 最新的几个（如果所有 run 同一个 train_dir）
train_dir = Path(RUNS[0]["params"]["train_dir"])
models_dir = train_dir / "models"
print("\ntrain_dir:", train_dir)
print("models_dir exists:", models_dir.exists())
if models_dir.exists():
    print("latest models:")
    for p in sorted(models_dir.glob("cellpose_*"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        print(" ", p)


# In[7]:


import json, subprocess, re
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
CFG_DIR = RUN_INDEX_PATH.parent
EXP_DIR = CFG_DIR.parent
LOG_DIR = EXP_DIR / "logs"

runs = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

def ps_info(pid: int) -> str:
    r = subprocess.run(["ps","-o","pid,ppid,stat,etime,cmd","-p",str(pid)], text=True, capture_output=True)
    return (r.stdout or r.stderr).strip()

def tail(path: Path, n=10) -> str:
    if not path.exists(): return "(no file)"
    lines = path.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"

def read_json(path: Path):
    if not path.exists(): return None
    try: return json.loads(path.read_text(errors="ignore"))
    except: return None

print("EXP_DIR:", EXP_DIR)
print("Total runs:", len(runs))
print("="*90)

for i, r in enumerate(runs, 1):
    run_name = r["run_name"]
    pid_path = Path(r["pid_path"])
    log_path = Path(r["log_path"])
    snap_path = Path(r["config_snapshot_path"])
    cellpose_slice = LOG_DIR / f"cellpose_{run_name}.log"

    pid = None
    if pid_path.exists():
        try: pid = int(pid_path.read_text().strip())
        except: pid = None

    snap = read_json(snap_path) or {}
    model_dir = snap.get("model_dir") or r.get("model_dir")
    model_ok = bool(model_dir) and Path(model_dir).exists()

    print(f"\n[{i}] {run_name}")
    print("  pid_file:", "YES" if pid_path.exists() else "NO")
    if pid is not None:
        info = ps_info(pid)
        print("  pid:", pid)
        print("  ps :", info.replace("\n"," | "))
        if " Z" in (" "+info+" "):
            print("  ⚠️ zombie detected")
    else:
        print("  pid:", None)

    print("  snapshot:", "YES" if snap_path.exists() else "NO")
    print("  model_dir:", model_dir)
    print("  model_dir exists:", model_ok)

    print("  wrapper_log:", str(log_path) if log_path.exists() else "(no)")
    print("  wrapper_log tail:")
    print("    " + tail(log_path, 6).replace("\n", "\n    "))

    print("  cellpose_slice_log:", str(cellpose_slice) if cellpose_slice.exists() else "(no)")


# In[8]:


import json, re
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
runs = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

PAT = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")
fixed = 0

for r in runs:
    snap_path = Path(r["config_snapshot_path"])
    log_path  = Path(r["log_path"])
    if not snap_path.exists() or not log_path.exists():
        continue

    snap = json.loads(snap_path.read_text(errors="ignore"))
    if snap.get("model_dir"):  # 已有就跳过
        continue

    txt = log_path.read_text(errors="ignore")
    m = None
    for line in reversed(txt.splitlines()[-400:]):  # 只看末尾更快
        mm = PAT.search(line)
        if mm:
            m = mm.group("dir")
            break
    if m and Path(m).exists():
        snap["model_dir"] = m
        snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
        fixed += 1
        print("✅ patched model_dir for", r["run_name"], "->", m)

print("Done. patched:", fixed)


# In[9]:


import json, subprocess
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
runs = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

def ps_stat(pid: int) -> str:
    r = subprocess.run(["ps","-o","stat=","-p",str(pid)], text=True, capture_output=True)
    return (r.stdout or "").strip()

cleaned = 0
for r in runs:
    pid_path = Path(r["pid_path"])
    if not pid_path.exists(): 
        continue
    try:
        pid = int(pid_path.read_text().strip())
    except:
        continue
    st = ps_stat(pid)
    if (not st) or ("Z" in st):
        pid_path.unlink(missing_ok=True)
        cleaned += 1
        print("🧹 removed stale pid_file for", r["run_name"], "pid=", pid, "stat=", st)

print("Done. cleaned:", cleaned)


# In[10]:


import json
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
CFG_DIR = RUN_INDEX_PATH.parent
EXP_DIR = CFG_DIR.parent
LOG_DIR = EXP_DIR / "logs"
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

RUNS = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

print("✅ loaded RUNS:", len(RUNS))
print("EXP_DIR:", EXP_DIR)
print("V2 run:", [r["run_name"] for r in RUNS if r["tag"].startswith("V2")][0])


# In[11]:


import json
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
runs = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]

changed = 0
for r in runs:
    if not r["tag"].startswith("V2"):
        continue

    old = r["params"]["train_batch_size"]
    r["params"]["train_batch_size"] = 48

    # 同步改 snapshot
    snap_path = Path(r["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(errors="ignore"))
    snap["params"]["train_batch_size"] = 48
    snap["note"] = f"patched train_batch_size {old} -> 48 (retry after OOM)"
    snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    # 清理旧 pid（避免误判）
    Path(r["pid_path"]).unlink(missing_ok=True)

    changed += 1
    print(f"✅ V2 patched: bs {old} -> 48")
    print("   snapshot:", snap_path)

# 写回 RUNS.jsonl（每行一个 json）
RUN_INDEX_PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in runs) + "\n", encoding="utf-8")

print("Done. changed entries:", changed)


# In[12]:


get_ipython().system('nvidia-smi')


# In[ ]:


import json, time, shlex, subprocess, os, re
from pathlib import Path

RUN_INDEX_PATH = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/config/RUNS.jsonl")
CFG_DIR = RUN_INDEX_PATH.parent
EXP_DIR = CFG_DIR.parent
LOG_DIR = EXP_DIR / "logs"
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()

RUNS = [json.loads(l) for l in RUN_INDEX_PATH.read_text().splitlines() if l.strip()]
POLL_S = 60

PAT_SAVED = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")

def already_done(ctx: dict) -> bool:
    sp = Path(ctx["config_snapshot_path"])
    if not sp.exists(): 
        return False
    try:
        snap = json.loads(sp.read_text(errors="ignore"))
        m = snap.get("model_dir")
        return bool(m) and Path(m).exists()
    except Exception:
        return False

def build_cmd(ctx: dict):
    p = ctx["params"]
    cmd = [
        "python","-m","cellpose",
        "--train",
        "--dir", p["train_dir"],
        "--test_dir", p["val_dir"],
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
    if p.get("transformer", False): cmd.append("--transformer")
    if p.get("augment", False): cmd.append("--augment")
    if p.get("use_gpu", True): cmd.append("--use_gpu")
    if p.get("verbose", True): cmd.append("--verbose")
    return cmd

def capture_model_dir_from_log(log_path: Path):
    if not log_path.exists(): 
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    for line in reversed(lines[-1200:]):
        m = PAT_SAVED.search(line)
        if m:
            return m.group("dir")
    return None

def run_one(ctx: dict):
    cmd = build_cmd(ctx)
    cmd_text = " ".join(shlex.quote(x) for x in cmd)
    Path(ctx["cmd_path"]).write_text(cmd_text + "\n", encoding="utf-8")

    log_path = Path(ctx["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)

    # 减少碎片（对你这种“贴顶 OOM”很关键）
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True, env=env)
    Path(ctx["pid_path"]).write_text(str(proc.pid) + "\n", encoding="utf-8")

    print("🚀 Started:", ctx["run_name"])
    print("   PID:", proc.pid)
    print("   bs:", ctx["params"]["train_batch_size"])
    print("   LOG:", log_path)

    while True:
        ret = proc.poll()
        if ret is not None:
            print("✅ finished return:", ret)
            break
        print(f"⏳ running... (sleep {POLL_S}s)")
        time.sleep(POLL_S)

    # 清理 pid
    Path(ctx["pid_path"]).unlink(missing_ok=True)

    # 写回 model_dir（如果成功）
    snap_path = Path(ctx["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(errors="ignore"))
    if not snap.get("model_dir"):
        mdir = capture_model_dir_from_log(log_path)
        if mdir and Path(mdir).exists():
            snap["model_dir"] = mdir
            snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
            print("🏁 model_dir:", mdir)
        else:
            print("⚠️ model_dir 未捕获（大概率失败/中断）。去看 log 尾巴：")
            print("   ", log_path)

# 从 V2 开始
start = False
for ctx in RUNS:
    if ctx["tag"].startswith("V2"):
        start = True
    if not start:
        continue

    print("\n" + "-"*80)
    print("RUN:", ctx["run_name"])

    if already_done(ctx):
        print("✅ already done -> skip")
        continue

    run_one(ctx)

print("\n🎉 从 V2 起的串行训练流程已执行完（或遇到失败停止）。")


# In[14]:


get_ipython().system('ps -fp 272580')
get_ipython().system('nvidia-smi')


# In[15]:


get_ipython().system('tail -n 80 /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/logs/train_V6_transformer_on_d20_lr5e-05_wd0.01_bs32_e160_20260305_134233.log')


# In[16]:


get_ipython().system('grep -n "model trained and saved to" /gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/logs/train_V6_transformer_on_d20_lr5e-05_wd0.01_bs32_e160_20260305_134233.log | tail -n 5')


# In[17]:


import json, re
from pathlib import Path

EXP = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228")
RUNS_JSONL = EXP / "config" / "RUNS.jsonl"

runs = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]
v6 = [r for r in runs if r["tag"].startswith("V6")][0]

model_dir = "/gpfs/share/home/2306391536/projects/cell_seg/Cellpose2TrainDataset/trainset_ft_os3/models/cellpose_1772694824.1367216"

snap_path = Path(v6["config_snapshot_path"])
snap = json.loads(snap_path.read_text(errors="ignore"))
snap["model_dir"] = model_dir
snap_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

print("✅ patched V6 model_dir into snapshot:")
print("run:", v6["run_name"])
print("model_dir:", model_dir)


# In[18]:


import json, re
from pathlib import Path

EXP = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228")
RUNS_JSONL = EXP / "config" / "RUNS.jsonl"
LOG_DIR = EXP / "logs"

runs = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]

PAT = re.compile(r"model trained and saved to\s+(?P<dir>/\S+)")

def guess_from_log(log_path: Path):
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    for line in reversed(lines[-1500:]):
        m = PAT.search(line)
        if m:
            return m.group("dir")
    return None

print("EXP:", EXP)
print("="*110)

for r in runs:
    tag = r["tag"]
    snap_path = Path(r["config_snapshot_path"])
    log_path = Path(r["log_path"])

    snap = {}
    if snap_path.exists():
        try:
            snap = json.loads(snap_path.read_text(errors="ignore"))
        except Exception:
            snap = {}

    model_dir = snap.get("model_dir")
    ok = bool(model_dir) and Path(model_dir).exists()

    print(f"\n[{tag}] {r['run_name']}")
    print("  snapshot:", "YES" if snap_path.exists() else "NO", "|", snap_path)
    print("  model_dir in snapshot:", model_dir)
    print("  model_dir exists:", ok)

    if not ok:
        g = guess_from_log(log_path)
        print("  ↪ guess from train log:", g)
        if g:
            print("    (exists:", Path(g).exists(), ")")
    else:
        # 给你一个安心：顺便显示该 models 目录名
        print("  ✅ ok")

print("\nDone.")


# In[19]:


import json, re
from pathlib import Path
import pandas as pd

EXP = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228")
RUNS_JSONL = EXP / "config" / "RUNS.jsonl"
OUT_CSV = EXP / "summary_runs.csv"

runs = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]

# 解析 cellpose 训练日志里这种行：
# 2026-03-05 ... [INFO] 50, train_loss=0.7851, test_loss=0.8477, LR=0.000050, time ...
MET_RE = re.compile(
    r"\b(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)"
)

rows = []
for r in runs:
    snap = json.loads(Path(r["config_snapshot_path"]).read_text(errors="ignore"))
    model_dir = snap.get("model_dir")
    if not model_dir or not Path(model_dir).exists():
        continue  # 跳过 V2/缺席

    log_path = Path(r["log_path"])
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
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
        last_epoch = int(dfm.iloc[-1]["epoch"])
        last_test  = float(dfm.iloc[-1]["test_loss"])
    else:
        best_epoch = best_test = last_epoch = last_test = None

    p = r["params"]
    rows.append({
        "tag": r["tag"],
        "run_name": r["run_name"],
        "model_dir": model_dir,
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
        "log_path": str(log_path),
    })

df = pd.DataFrame(rows).sort_values("tag")
df.to_csv(OUT_CSV, index=False)
print("✅ Wrote:", OUT_CSV)
display(df[[
    "tag","diameter","augment","transformer","train_batch_size","n_epochs",
    "best_epoch_by_test_loss","best_test_loss","last_test_loss","model_dir"
]])


# In[21]:


#弥补了上面表格最后……看不到的情况

import pandas as pd
from pathlib import Path

csv_path = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228/summary_runs.csv")
df = pd.read_csv(csv_path)
pd.set_option("display.max_colwidth", 200)
display(df[["tag","diameter","augment","transformer","best_epoch_by_test_loss","best_test_loss","model_dir"]])


# In[ ]:


# Cell: Visual Battle Board (GT vs Pred) for multiple models on same samples
import os, json, shlex, subprocess, textwrap, re
from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# =========================
# 0) 你需要填的最少信息
# =========================
EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228").resolve()
RUNS_JSONL = EXP_DIR / "config" / "RUNS.jsonl"

VAL_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/Cellpose2TrainDataset/valset").resolve()

# 你想肉眼看的样本（填文件名，不含路径；可留空让它自动挑前 N 张）
SAMPLE_NAMES = []  # e.g. ["img_0001.tif", "img_0037.tif"]
N_AUTO_SAMPLES = 3

# 推理参数（先统一，确保“同一场地比武”）
FLOW_TH = 0.4
CELLPROB_TH = 0.0

# 输出目录：每次运行都生成一个新文件夹
OUT_DIR = (EXP_DIR / "battleboard_vis").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读 GT 的后缀规则（你现在用的是 mask_filter="_masks"）
GT_MASK_SUFFIX = "_masks"  # 会匹配 *_masks.tif / *_masks.png 等

# 图像后缀候选（按你的数据实际情况增减）
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

# =========================
# 1) 帮助函数
# =========================
def sh(cmd, cwd=None):
    """run shell command; print for traceability"""
    print("\n$", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout.strip()[:2000])
    if r.stderr.strip():
        print("stderr:", r.stderr.strip()[:2000])
    return r.returncode

def list_images(val_dir: Path):
    files = []
    for ext in IMG_EXTS:
        files += sorted(val_dir.glob(f"*{ext}"))
    # 过滤掉 mask 文件（含 _masks）
    files = [p for p in files if GT_MASK_SUFFIX not in p.stem]
    return sorted(files)

def find_gt_mask(img_path: Path):
    # 尝试同目录下常见 GT mask 命名：<stem>_masks.<ext>
    for ext in [".tif", ".tiff", ".png", ".npy"]:
        p = img_path.with_name(img_path.stem + GT_MASK_SUFFIX + ext)
        if p.exists():
            return p
    # 兜底：扫一遍同目录里含 stem+_masks 的文件
    cand = list(img_path.parent.glob(img_path.stem + GT_MASK_SUFFIX + ".*"))
    return cand[0] if cand else None

def read_mask_any(path: Path):
    if path is None or (not path.exists()):
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    return tiff.imread(path) if path.suffix.lower() in [".tif", ".tiff"] else None

def mask_to_outline(mask: np.ndarray):
    """simple outline: boundary pixels where neighbors differ"""
    if mask is None:
        return None
    m = mask.astype(np.int32)
    # 4-neighborhood differences
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:,1:] = m[:,:-1]
    rt = np.zeros_like(m); rt[:,:-1] = m[:,1:]
    edge = (m != up) | (m != dn) | (m != lf) | (m != rt)
    edge &= (m > 0)
    return edge

def norm01(img):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

def load_image(img_path: Path):
    x = tiff.imread(img_path)
    # 如果是多通道/多维，尽量挤到 2D
    if x.ndim > 2:
        x = x.squeeze()
    if x.ndim != 2:
        raise ValueError(f"Image {img_path} has shape {x.shape}, expected 2D after squeeze.")
    return x

def run_cellpose_infer_on_dir(val_dir: Path, model_dir: str, diameter: int, savedir: Path):
    savedir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(val_dir),
        "--pretrained_model", str(model_dir),
        "--diameter", str(diameter),
        "--flow_threshold", str(FLOW_TH),
        "--cellprob_threshold", str(CELLPROB_TH),
        "--use_gpu",
        "--save_tif",
        "--no_npy",
        "--savedir", str(savedir),
        "--output_name", ""  # 输出名保持默认文件名（更好对齐）
    ]
    return sh(cmd)

def find_pred_mask(savedir: Path, img_path: Path):
    # cellpose 默认 mask tif 后缀常见：*_cp_masks.tif
    # 但不同版本/参数可能略不同，这里做几种候选
    stem = img_path.stem
    cands = [
        savedir / f"{stem}_cp_masks.tif",
        savedir / f"{stem}_masks.tif",
        savedir / f"{stem}_cp_masks.tiff",
        savedir / f"{stem}_masks.tiff",
    ]
    for p in cands:
        if p.exists():
            return p
    # 兜底：找包含 stem 且包含 "masks" 的 tif
    glob_cand = sorted(savedir.glob(f"{stem}*masks*.tif"))
    return glob_cand[0] if glob_cand else None

# =========================
# 2) 读取本次 sweep 的模型列表（跳过 V2）
# =========================
runs = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]

models = []
for r in runs:
    tag = r["tag"]
    snap = json.loads(Path(r["config_snapshot_path"]).read_text(errors="ignore"))
    model_dir = snap.get("model_dir")
    if not model_dir or not Path(model_dir).exists():
        print(f"⚠️ skip {tag} (no model_dir)")
        continue
    if tag.startswith("V2"):
        print(f"⚠️ skip {tag} (failed/OOM)")
        continue
    models.append({
        "tag": tag,
        "run_name": r["run_name"],
        "model_dir": model_dir,
        "diameter": int(r["params"]["diameter"]),
    })

print("\n✅ Models for visual battle:")
for m in models:
    print(f"- {m['tag']} | d={m['diameter']} | {m['model_dir']}")

assert len(models) > 0, "No usable models found."

# =========================
# 3) 选择要看的样本
# =========================
imgs = list_images(VAL_DIR)
assert len(imgs) > 0, f"No images found in {VAL_DIR}"

if SAMPLE_NAMES:
    chosen = []
    name2path = {p.name: p for p in imgs}
    for n in SAMPLE_NAMES:
        if n in name2path:
            chosen.append(name2path[n])
        else:
            print("⚠️ sample not found:", n)
    chosen = chosen[:N_AUTO_SAMPLES]
else:
    chosen = imgs[:N_AUTO_SAMPLES]

print("\n🎯 Chosen samples:")
for p in chosen:
    print("-", p.name)

# =========================
# 4) 对每个模型跑推理（同一 val_dir，输出到独立 savedir）
# =========================
stamp = time.strftime("%Y%m%d_%H%M%S")
battle_dir = OUT_DIR / f"battle_{stamp}"
battle_dir.mkdir(parents=True, exist_ok=True)

pred_roots = {}
for m in models:
    savedir = battle_dir / f"pred_{m['tag']}"
    rc = run_cellpose_infer_on_dir(VAL_DIR, m["model_dir"], m["diameter"], savedir)
    pred_roots[m["tag"]] = savedir
    if rc != 0:
        print(f"❌ Inference failed for {m['tag']} (return {rc}). Continue anyway.")

print("\n✅ Inference done. Outputs under:", battle_dir)

# =========================
# 5) 生成可视化对比图（每个样本一张）
#    layout: 2 rows (GT / Pred) x N models columns
# =========================
for img_path in chosen:
    img = load_image(img_path)
    img_n = norm01(img)

    gt_path = find_gt_mask(img_path)
    gt_mask = read_mask_any(gt_path)
    gt_edge = mask_to_outline(gt_mask) if gt_mask is not None else None

    ncol = len(models)
    fig, axes = plt.subplots(2, ncol, figsize=(4*ncol, 8))
    if ncol == 1:
        axes = np.array(axes).reshape(2,1)

    for j, m in enumerate(models):
        tag = m["tag"]
        pred_path = find_pred_mask(pred_roots[tag], img_path)
        pred_mask = read_mask_any(pred_path) if pred_path else None
        pred_edge = mask_to_outline(pred_mask) if pred_mask is not None else None

        # Row 0: GT overlay
        ax = axes[0, j]
        ax.imshow(img_n, cmap="gray")
        if gt_edge is not None:
            ax.imshow(gt_edge, alpha=0.8)  # outline as overlay
        ax.set_title(f"{tag} | GT outline")
        ax.axis("off")

        # Row 1: Pred overlay
        ax = axes[1, j]
        ax.imshow(img_n, cmap="gray")
        if pred_edge is not None:
            ax.imshow(pred_edge, alpha=0.8)
        ax.set_title(f"{tag} | Pred outline")
        ax.axis("off")

        # 同时单独保存每个模型的 pred mask 路径信息
        if pred_path is None:
            print(f"⚠️ pred mask not found for {tag} sample {img_path.name} in {pred_roots[tag]}")

    fig.suptitle(f"Sample: {img_path.name} | GT={gt_path.name if gt_path else 'None'}", fontsize=16)
    out_png = battle_dir / f"board_{img_path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("🖼 saved:", out_png)

print("\n🎉 Visual battle boards saved to:", battle_dir)
print("Tip: open the board_*.png files and pick winners by eyeballing boundary quality / FP / FN.")


# In[2]:


from pathlib import Path
import re

ROOT_NEWVAL = Path("/gpfs/share/home/2306391536/projects/cell_seg/Cellpose2TrainDataset/new_val_proc").resolve()
IMG_DIR = ROOT_NEWVAL / "images_sp_lcn"
GT_DIR  = ROOT_NEWVAL / "ground"

print("ROOT_NEWVAL:", ROOT_NEWVAL)
print("IMG_DIR exists:", IMG_DIR.exists(), IMG_DIR)
print("GT_DIR  exists:", GT_DIR.exists(),  GT_DIR)

# 收集图片
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
imgs = []
for ext in IMG_EXTS:
    imgs += sorted(IMG_DIR.glob(f"*{ext}"))
print("\n#images:", len(imgs))
print("examples:", [p.name for p in imgs[:5]])

# 收集 GT masks（也可能是 tif/tiff/png）
gts = []
for ext in [".tif", ".tiff", ".png"]:
    gts += sorted(GT_DIR.glob(f"*{ext}"))
print("\n#gt masks:", len(gts))
print("examples:", [p.name for p in gts[:5]])

# 用“stem 归一化”尝试对齐（把 _masks / _mask / -mask 这种尾巴去掉）
def norm_stem(stem: str) -> str:
    s = stem
    s = re.sub(r"(_masks?|_mask|[-_]gt|[-_]label|[-_]ann|[-_]seg)$", "", s, flags=re.IGNORECASE)
    return s

img_map = {norm_stem(p.stem): p for p in imgs}
gt_map  = {norm_stem(p.stem): p for p in gts}

keys_img = set(img_map.keys())
keys_gt  = set(gt_map.keys())
common = sorted(keys_img & keys_gt)

print("\nCommon matched stems:", len(common))
print("First 10 matches:")
for k in common[:10]:
    print(" ", k, "->", img_map[k].name, "|", gt_map[k].name)

missing_gt = sorted(keys_img - keys_gt)[:10]
missing_img = sorted(keys_gt - keys_img)[:10]
print("\nExamples missing GT for image stems:", missing_gt)
print("Examples missing image for GT stems:", missing_img)


# In[3]:


import json
from pathlib import Path

EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228").resolve()
RUNS_JSONL = EXP_DIR / "config" / "RUNS.jsonl"

runs = [json.loads(l) for l in RUNS_JSONL.read_text().splitlines() if l.strip()]

models = []
for r in runs:
    tag = r["tag"]
    snap = json.loads(Path(r["config_snapshot_path"]).read_text(errors="ignore"))
    model_dir = snap.get("model_dir")
    if not model_dir or not Path(model_dir).exists():
        continue
    if tag.startswith("V2"):
        continue
    models.append({
        "tag": tag,
        "model_dir": model_dir,
        "diameter": int(r["params"]["diameter"]),
    })

print("✅ Models loaded:", len(models))
for m in models:
    print("-", m["tag"], "| d=", m["diameter"], "|", m["model_dir"])


# In[6]:


# Cell: Multi-model inference on new_val_proc (debug-first, no valset pollution)
# - 输入：IMG_DIR 里的图片（new_val_proc/images_sp_lcn）
# - GT_DIR 仅用于 debug 选样（确保抽到的图有对应 GT），本脚本不做评估
# - 输出：EXP_DIR/battle_newval/newval_<stamp>/{tiny_images/, pred_<tag>/, manifest.json}

import os, shlex, subprocess, time, json, re
from pathlib import Path

# =========================
# 0) 必填：路径 + 运行开关
# =========================
EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_134228").resolve()

ROOT_NEWVAL = Path("/gpfs/share/home/2306391536/projects/cell_seg/Cellpose2TrainDataset/new_val_proc").resolve()
IMG_DIR = (ROOT_NEWVAL / "images_sp_lcn").resolve()
GT_DIR  = (ROOT_NEWVAL / "ground").resolve()

# 先小样本测试：跑 N 张；确认OK后改成 None 跑全量
N_DEBUG = None  # None 表示全量

# 推理参数（统一对比）
FLOW_TH = 0.4
CELLPROB_TH = 0.0

# 你要推理的模型列表：要求每个元素至少有 tag/model_dir/diameter
# 你前面那个 “从 RUNS.jsonl + snapshot 读 models” 的 cell 先跑出来 models
assert "models" in globals() and len(models) > 0, "models 未定义：请先生成 models 列表（tag/model_dir/diameter）"

# =========================
# 1) 输出目录（每次运行一个新文件夹）
# =========================
OUT_ROOT = (EXP_DIR / "battle_newval").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
RUN_OUT = OUT_ROOT / f"newval_{stamp}"
RUN_OUT.mkdir(parents=True, exist_ok=True)

# 记录本次运行信息
MANIFEST = RUN_OUT / "manifest.json"

print("EXP_DIR   :", EXP_DIR)
print("RUN_OUT   :", RUN_OUT)
print("IMG_DIR   :", IMG_DIR, "exists=", IMG_DIR.exists())
print("GT_DIR    :", GT_DIR,  "exists=", GT_DIR.exists())
assert IMG_DIR.exists(), f"IMG_DIR 不存在：{IMG_DIR}"
assert GT_DIR.exists(),  f"GT_DIR 不存在：{GT_DIR}"

# =========================
# 2) helper: shell runner
# =========================
def sh(cmd, cwd=None):
    print("\n$", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout.strip()[:1500])
    if r.stderr.strip():
        print("stderr:", r.stderr.strip()[:1500])
    return r.returncode, r

# =========================
# 3) debug 选样：按“末尾数字编号”匹配 IMG 与 GT
#    因为你实际文件名是：
#      IMG: image_sparse_ln_0000.tif
#      GT : 20220317_gou_Dataset_00000.tif
# =========================
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]

NUM_TAIL = re.compile(r"(\d+)$")  # 抓 stem 末尾连续数字

def tail_id(p: Path):
    m = NUM_TAIL.search(p.stem)
    return int(m.group(1)) if m else None

def list_files(d: Path, exts):
    out = []
    for ext in exts:
        out += sorted(d.glob(f"*{ext}"))
    return out

imgs_all = list_files(IMG_DIR, IMG_EXTS)
gts_all  = list_files(GT_DIR, GT_EXTS)

print("\n#images:", len(imgs_all), "examples:", [p.name for p in imgs_all[:5]])
print("#gt    :", len(gts_all),  "examples:", [p.name for p in gts_all[:5]])
assert len(imgs_all) > 0, f"IMG_DIR 没有可用图片：{IMG_DIR}"
assert len(gts_all) > 0,  f"GT_DIR 没有可用 GT：{GT_DIR}"

# GT 建索引：id -> path（如果重复，保留最后一个）
gt_map = {}
for p in gts_all:
    i = tail_id(p)
    if i is not None:
        gt_map[i] = p

# =========================
# 4) 只想对少量图推理：建 tiny_images 软链接目录（不复制数据）
# =========================
if N_DEBUG is not None:
    tiny_dir = RUN_OUT / "tiny_images"
    tiny_dir.mkdir(parents=True, exist_ok=True)

    chosen = []
    pairs = []
    for p in imgs_all:
        i = tail_id(p)
        if i is not None and i in gt_map:
            chosen.append(p)
            pairs.append((p, gt_map[i]))
        if len(chosen) >= N_DEBUG:
            break

    print("\n🎯 debug pairs (IMG <-> GT):")
    for im, gt in pairs:
        print(" ", im.name, "<->", gt.name)

    if len(chosen) == 0:
        raise RuntimeError(
            "没选到任何 debug 样本：说明 IMG/GT 无法按末尾数字对齐。\n"
            "请检查两边编号是否同一套（是否一个从0开始一个从1开始），或者文件名末尾不是编号。"
        )

    # 建软链接（只链接图片，推理不需要 GT）
    for p in chosen:
        link = tiny_dir / p.name
        if not link.exists():
            link.symlink_to(p)

    INFER_DIR = tiny_dir
else:
    INFER_DIR = IMG_DIR

print("\n🧪 INFER_DIR:", INFER_DIR)
print("INFER_DIR n_files:", len(list(INFER_DIR.iterdir())))

# =========================
# 5) 跑推理：每个模型输出到独立 pred_<tag>（不污染输入目录）
# =========================
pred_dirs = {}
results = []

for m in models:
    tag = m["tag"]
    model_dir = m["model_dir"]
    diam = int(m["diameter"])

    # 健壮性检查
    if not model_dir:
        print(f"⚠️ skip {tag}: empty model_dir")
        continue
    if model_dir != "cpsam" and (not Path(model_dir).exists()):
        print(f"⚠️ skip {tag}: model_dir not exists -> {model_dir}")
        continue

    savedir = RUN_OUT / f"pred_{tag}"
    savedir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(INFER_DIR),
        "--pretrained_model", str(model_dir),
        "--diameter", str(diam),
        "--flow_threshold", str(FLOW_TH),
        "--cellprob_threshold", str(CELLPROB_TH),
        "--use_gpu",
        "--save_tif",
        "--no_npy",
        "--savedir", str(savedir),
        # "--verbose",  # 需要日志再打开
    ]

    rc, proc = sh(cmd, cwd=str(EXP_DIR))
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

# =========================
# 6) 写 manifest（便于断线恢复 + 追溯）
# =========================
manifest = {
    "time": stamp,
    "exp_dir": str(EXP_DIR),
    "run_out": str(RUN_OUT),
    "img_dir": str(IMG_DIR),
    "gt_dir": str(GT_DIR),
    "infer_dir": str(INFER_DIR),
    "n_debug": N_DEBUG,
    "flow_th": FLOW_TH,
    "cellprob_th": CELLPROB_TH,
    "models": models,
    "pred_dirs": pred_dirs,
    "results": results,
}
MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n🎉 Inference outputs:", RUN_OUT)
print("📌 manifest:", MANIFEST)
print("Tip: 预测 masks 通常在 RUN_OUT/pred_<tag> 里，文件名形如 *_cp_masks.tif")


# In[7]:


# Cell: Visual Battle Boards on new_val_proc (GT vs multi-model Pred) — tail-number pairing edition
# 依赖：你已经跑完 “new_val 推理脚本”，并且当前环境里有：
# - IMG_DIR, GT_DIR, INFER_DIR, RUN_OUT, N_DEBUG
# - models (list of dict: {"tag","model_dir","diameter"...})
# - pred_dirs (dict: tag -> savedir(str or Path))

import re
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0) 读取图片（2D）+ 归一化
# =========================
def load_img(p: Path):
    x = tiff.imread(str(p))
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, got {x.shape} for {p}")
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

# 读取 mask（tif）
def load_mask(p: Path):
    if p is None or (not Path(p).exists()):
        return None
    x = tiff.imread(str(p))
    x = np.squeeze(x)
    return x.astype(np.int32)

# mask -> outline（边界）
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

# =========================
# 1) new_val 的对齐规则：按文件名末尾数字编号配对
#    IMG: image_sparse_ln_0000.tif
#    GT : 20220317_gou_Dataset_00000.tif
# =========================
NUM_TAIL = re.compile(r"(\d+)$")

def tail_id_from_stem(stem: str):
    m = NUM_TAIL.search(stem)
    return int(m.group(1)) if m else None

# =========================
# 2) 建映射：id(int) -> path
# =========================
IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
GT_EXTS  = [".tif", ".tiff", ".png"]

imgs = []
for ext in IMG_EXTS:
    imgs += sorted(Path(IMG_DIR).glob(f"*{ext}"))

gts = []
for ext in GT_EXTS:
    gts += sorted(Path(GT_DIR).glob(f"*{ext}"))

img_map = {}
for p in imgs:
    i = tail_id_from_stem(p.stem)
    if i is not None:
        img_map[i] = p

gt_map = {}
for p in gts:
    i = tail_id_from_stem(p.stem)
    if i is not None:
        gt_map[i] = p

common = sorted(set(img_map.keys()) & set(gt_map.keys()))
assert len(common) > 0, "No matched image<->GT pairs by tail number. Check naming / directory."

print("✅ Matched IMG<->GT pairs:", len(common))
print("examples:", [(img_map[i].name, gt_map[i].name) for i in common[:5]])

# =========================
# 3) 选要可视化的样本：
#    debug 模式下就用 INFER_DIR(tiny_images) 里的那几张；
#    全量时默认画前 8 张（你可自行改）
# =========================
vis_files = sorted([p for p in Path(INFER_DIR).iterdir() if p.is_file()])
vis_ids = []
for p in vis_files:
    i = tail_id_from_stem(p.stem)
    if i is not None and (i in img_map) and (i in gt_map):
        vis_ids.append(i)

if N_DEBUG is not None:
    vis_ids = vis_ids[:N_DEBUG]
else:
    vis_ids = vis_ids[:8]

assert len(vis_ids) > 0, "No visual samples found from INFER_DIR. Check INFER_DIR content."

print("🎯 Visual sample ids:", vis_ids)

# =========================
# 4) 预测 mask 寻找规则
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
    # 兜底搜索
    glob_c = sorted(pred_root.glob(f"{stem}*masks*.tif"))
    return glob_c[0] if glob_c else None

# =========================
# 5) 输出目录：RUN_OUT/boards
# =========================
BOARD_DIR = Path(RUN_OUT) / "boards"
BOARD_DIR.mkdir(parents=True, exist_ok=True)

# pred_dirs 可能是 str，统一转 Path
pred_dirs_path = {k: Path(v) for k, v in pred_dirs.items()}

# =========================
# 6) 画板：每个样本一张
#    layout: 2 rows (GT / Pred) x N models columns
# =========================
ncol = len(models)
assert ncol > 0, "models is empty."

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
        pred_root = pred_dirs_path.get(tag)

        pred_path = find_pred_mask(pred_root, img_path) if pred_root else None
        pred = load_mask(pred_path) if pred_path else None
        pred_edge = mask_outline(pred)

        # Row 0: GT outline
        ax = axes[0, j]
        ax.imshow(img, cmap="gray")
        if gt_edge is not None:
            ax.imshow(gt_edge, alpha=0.8)
        ax.set_title(f"{tag} | GT")
        ax.axis("off")

        # Row 1: Pred outline
        ax = axes[1, j]
        ax.imshow(img, cmap="gray")
        if pred_edge is not None:
            ax.imshow(pred_edge, alpha=0.8)
        ax.set_title(f"{tag} | Pred")
        ax.axis("off")

        if pred_path is None:
            print(f"⚠️ pred not found: {tag} sample={img_path.name} in {pred_root}")

    fig.suptitle(f"ID={i} | IMG={img_path.name} | GT={gt_path.name}", fontsize=14)
    out_png = BOARD_DIR / f"board_{img_path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("🖼 saved:", out_png)

print("\n🎉 Boards saved in:", BOARD_DIR)
print("Tip: 打开 board_*.png 直接肉眼比：边界贴合/漏检(FN)/误检(FP) 哪家最离谱，一眼就知道。")


# In[8]:


import os, zipfile, json
from pathlib import Path
from datetime import datetime

# 你现在的 RUN_OUT 应该已经在上一个脚本里定义了
assert "RUN_OUT" in globals(), "RUN_OUT 未定义：请在推理脚本跑完后再运行本 cell"
run_out = Path(RUN_OUT).resolve()
assert run_out.exists(), f"RUN_OUT 不存在：{run_out}"

# 打包输出位置（放在 battle_newval 下的 zip 目录里）
zip_dir = run_out.parent / "zips"
zip_dir.mkdir(parents=True, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = (zip_dir / f"{run_out.name}__bundle_{stamp}.zip").resolve()

def add_file(zf: zipfile.ZipFile, path: Path, arcroot: Path):
    """add a file to zip with relative path from arcroot"""
    rel = path.relative_to(arcroot)
    zf.write(path, rel.as_posix())

def add_dir(zf: zipfile.ZipFile, d: Path, arcroot: Path, patterns=None):
    """add directory recursively; optional patterns list like ['*.png','*.json']"""
    if not d.exists():
        return 0
    n = 0
    if patterns:
        for pat in patterns:
            for p in d.rglob(pat):
                if p.is_file():
                    add_file(zf, p, arcroot); n += 1
    else:
        for p in d.rglob("*"):
            if p.is_file():
                add_file(zf, p, arcroot); n += 1
    return n

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    n_added = 0

    # 1) manifest（最关键）
    manifest = run_out / "manifest.json"
    if manifest.exists():
        add_file(zf, manifest, run_out); n_added += 1

    # 2) boards（可视化对比图）
    boards_dir = run_out / "boards"
    n_added += add_dir(zf, boards_dir, run_out, patterns=["*.png"])

    # 3) pred_*（预测结果：masks/flows 等都收）
    for d in sorted(run_out.glob("pred_*")):
        if d.is_dir():
            n_added += add_dir(zf, d, run_out)

    # 4) （可选）把本次 run 的目录结构写一份清单，方便你交付/复现
    tree_txt = run_out / "BUNDLE_TREE.txt"
    lines = []
    for p in sorted(run_out.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(run_out).as_posix()
        try:
            sz = p.stat().st_size
        except:
            sz = -1
        lines.append(f"{sz:>12}  {rel}")
    tree_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    add_file(zf, tree_txt, run_out); n_added += 1

print("✅ Bundled files:", n_added)
print("📦 ZIP saved to:", zip_path)


# In[9]:


# Cell: Evaluate all models on new_val_proc (AP50 / Precision / Recall / F1)
# 输出：RUN_OUT/eval_newval_metrics.csv + 表格 display
import re, json
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from cellpose import metrics
from IPython.display import display

# =========================
# 0) 你只需要保证这几个变量有：RUN_OUT
# =========================
assert "RUN_OUT" in globals(), "RUN_OUT 未定义：请先跑完 new_val 推理脚本（它会打印 RUN_OUT）"
RUN_OUT = Path(RUN_OUT).resolve()
assert RUN_OUT.exists(), f"RUN_OUT 不存在：{RUN_OUT}"

# 如果你没在环境里留 IMG_DIR/GT_DIR，也没关系：从 manifest 里读
manifest_path = RUN_OUT / "manifest.json"
assert manifest_path.exists(), f"找不到 manifest.json：{manifest_path}（请确认你用的是我给你的推理脚本输出）"
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
    # 兜底：包含 stem 且包含 masks 的 tif
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
    # average_precision 返回: ap, tp, fp, fn（数组，按阈值）
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

# 尝试从 manifest 里拿 diameter（更完整）
diam_map = {}
for m in manifest.get("models", []):
    if isinstance(m, dict) and "tag" in m:
        if "diameter" in m and m["diameter"] is not None:
            diam_map[m["tag"]] = int(m["diameter"])

rows = []
missing_detail = []

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
            "AP50": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan,
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
        "AP50": float(arr[:,0].mean()),
        "Precision": float(arr[:,1].mean()),
        "Recall": float(arr[:,2].mean()),
        "F1": float(arr[:,3].mean()),
        "n_eval": int(arr.shape[0]),
        "n_pairs_total": len(common_ids),
        "n_missing_pred": n_missing,
        "flow_th": FLOW_TH,
        "cellprob_th": CELLPROB_TH,
        "pred_dir": str(pred_root),
    })

df = pd.DataFrame(rows)

# 排序：先看 AP50，再看 F1（你也可以按自己口味改）
df = df.sort_values(["AP50", "F1"], ascending=False).reset_index(drop=True)

out_csv = (RUN_OUT / "eval_newval_metrics.csv").resolve()
df.to_csv(out_csv, index=False)

print("\n🏆 New-val ranking (top 10):")
display(df.head(10)[["model_tag","diameter","AP50","Precision","Recall","F1","n_eval","n_missing_pred"]])
print("\n✅ saved:", out_csv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# Cell 3 (REWRITE): 串行训练调度器（防僵尸 + 独立日志 + 自动抓 model_dir）
import os, shlex, subprocess, time, json
from pathlib import Path

# ============ 你已有的上下文检查 ============
assert "RUNS" in globals() and len(RUNS) > 0, "RUNS 不存在，请先跑 Cell 2。"
assert "ROOT" in globals(), "ROOT 不存在。"
assert "LOG_DIR" in globals() and "CFG_DIR" in globals(), "LOG_DIR/CFG_DIR 不存在。"

CELLPOSE_RUNLOG = (Path.home() / ".cellpose" / "run.log").resolve()
CELLPOSE_RUNLOG.parent.mkdir(parents=True, exist_ok=True)
CELLPOSE_RUNLOG.touch(exist_ok=True)

# ============ 工具函数 ============
def _ps_stat(pid: int) -> str:
    """返回 ps stat（含 Z=僵尸）；失败返回空字符串"""
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], text=True, capture_output=True)
    return (r.stdout or "").strip()

def _pid_alive(pid: int) -> bool:
    """严格存活判断：不存在/僵尸 都算 False"""
    stat = _ps_stat(pid)
    if not stat:
        return False
    if "Z" in stat:
        return False
    return True

def build_cellpose_cmd(ctx: dict):
    p = ctx["params"]
    cmd = [
        "python", "-m", "cellpose",
        "--train",
        "--dir", p["train_dir"],
        "--test_dir", p["val_dir"],
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

def _runlog_pos() -> int:
    """当前 run.log 文件大小（字节），用于切片"""
    try:
        return CELLPOSE_RUNLOG.stat().st_size
    except Exception:
        return 0

def _read_runlog_slice(start_pos: int) -> str:
    """从 start_pos 起读 ~/.cellpose/run.log 的新增内容"""
    try:
        with open(CELLPOSE_RUNLOG, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_pos)
            return f.read()
    except Exception as e:
        return f"[WARN] failed to read run.log slice: {e}\n"

def _extract_model_dir(runlog_text: str) -> str | None:
    """从该 run 的日志切片中抓模型保存目录"""
    # 最可靠的是最后那句：model trained and saved to <dir>
    key = "model trained and saved to "
    for line in reversed(runlog_text.splitlines()):
        if key in line:
            return line.split(key, 1)[1].strip()
    # 退而求其次：saving model to <dir>
    key2 = "saving model to "
    for line in reversed(runlog_text.splitlines()):
        if key2 in line:
            return line.split(key2, 1)[1].strip()
    return None

import re

MODEL_TRAINED_RE = re.compile(r"model trained and saved to\s+(?P<path>/\S+)", re.I)
SAVING_MODEL_RE  = re.compile(r"saving model to\s+(?P<path>/\S+)", re.I)

def _extract_model_dir_from_text(text: str) -> str | None:
    # 优先：trained and saved
    for line in reversed(text.splitlines()):
        m = MODEL_TRAINED_RE.search(line)
        if m: return m.group("path").strip()
    # 其次：saving model to
    for line in reversed(text.splitlines()):
        m = SAVING_MODEL_RE.search(line)
        if m: return m.group("path").strip()
    return None

def _update_ctx_snapshot(ctx: dict):
    """把 ctx 写回它自己的 config_snapshot_path（覆盖更新）"""
    p = Path(ctx["config_snapshot_path"])
    p.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")

def start_one(ctx: dict):
    """启动一个 run：若已有 pid 且仍活着则复用；否则重启"""
    pid_path = Path(ctx["pid_path"])
    cmd_path = Path(ctx["cmd_path"])
    wrapper_log_path = Path(ctx["log_path"])  # 你的 wrapper log（可留作启动记录）

    # 如果 pid 仍活着（且不是僵尸），就不重复启动
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            if _pid_alive(pid):
                print(f"⚠️ 已在运行（复用）：{ctx['run_name']} (PID={pid}, stat={_ps_stat(pid)})")
                return None, pid, None   # 复用模式：不切 run.log
        except Exception:
            pass

    cmd = build_cellpose_cmd(ctx)
    cmd_text = " ".join(shlex.quote(x) for x in cmd)
    cmd_path.write_text(cmd_text + "\n", encoding="utf-8")

    wrapper_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 记录启动前 run.log 的偏移，用于切片该 run 的真实训练日志
    start_pos = _runlog_pos()

    # 启动训练进程（stdout/stderr 仍然重定向到 wrapper log，方便你看命令是否炸）
    log_f = open(wrapper_log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")

    print(f"🚀 Started: {ctx['run_name']}")
    print(f"   PID: {proc.pid}")
    print(f"   CMD: {cmd_text}")
    print(f"   Wrapper LOG: {wrapper_log_path}")
    print(f"   Cellpose run.log slice start byte: {start_pos}")
    return proc, proc.pid, start_pos

def wait_one(proc: subprocess.Popen | None, pid: int, poll_s: int = 60):
    """等待训练结束；如果 proc=None（复用外部 pid），就轮询 stat 直到不活"""
    if proc is not None:
        # 这是最干净的：wait 会把子进程收尸 -> 不产生 zombie
        while True:
            ret = proc.poll()
            if ret is not None:
                return ret
            print(f"⏳ PID={pid} running... (sleep {poll_s}s)")
            time.sleep(poll_s)
    else:
        # 复用已有 PID 的兜底：轮询 stat（能识别僵尸）
        while _pid_alive(pid):
            print(f"⏳ PID={pid} running... (sleep {poll_s}s) stat={_ps_stat(pid)}")
            time.sleep(poll_s)
        return None  # 无法获得 return code（非本 kernel 启动的子进程）

def finalize_one(ctx: dict, start_pos: int | None):
    runlog_text = ""
    cellpose_log_path = Path(LOG_DIR) / f"cellpose_{ctx['run_name']}.log"

    if start_pos is not None:
        runlog_text = _read_runlog_slice(start_pos)
        cellpose_log_path.write_text(runlog_text, encoding="utf-8", errors="ignore")
        ctx["cellpose_runlog_path"] = str(cellpose_log_path)
        ctx["cellpose_runlog_start_byte"] = start_pos
    else:
        # 复用模式：不写切片日志（避免误导），但保留字段提示
        ctx["cellpose_runlog_path"] = None
        ctx["cellpose_runlog_start_byte"] = None

    # 抓 model_dir：优先 runlog_text（若有），否则 fallback 到 wrapper log
    model_dir = _extract_model_dir_from_text(runlog_text) if runlog_text else None
    if not model_dir:
        try:
            wrapper_text = Path(ctx["log_path"]).read_text(errors="ignore")
            model_dir = _extract_model_dir_from_text(wrapper_text)
        except Exception:
            model_dir = None

    ctx["model_dir"] = model_dir
    _update_ctx_snapshot(ctx)

    if start_pos is not None:
        print(f"🧾 Saved cellpose log slice -> {cellpose_log_path}")
    else:
        print("🧾 Reused PID: skip cellpose run.log slice (avoid partial slice)")

    if model_dir:
        print(f"🏁 model_dir captured -> {model_dir}")
    else:
        print("⚠️ 没抓到 model_dir：可能训练尚未保存/崩溃/日志格式变化。")

        
# ============ 主逻辑：串行跑完 RUNS ============
print("="*80)
print(f"🧨 Serial sweep launch (REWRITE) | total runs: {len(RUNS)}")
print("策略：一次只跑一个；每个 run 单独保存真实训练日志；自动抓 model_dir")
print("="*80)

for i, ctx in enumerate(RUNS, 1):
    print("\n" + "-"*80)
    print(f"[{i}/{len(RUNS)}] launching {ctx['run_name']}")

    proc, pid, start_pos = start_one(ctx)

    # 小延迟让 cellpose 把“开头几行”写进 run.log
    time.sleep(2)

    # 等待结束（会正确 wait() -> 不产生僵尸）
    ret = wait_one(proc, pid, poll_s=60)
    if ret is None:
        print(f"✅ PID={pid} finished. return=UNKNOWN (reused external pid)")
    else:
        print(f"✅ PID={pid} finished. return={ret}")

    # 收集该 run 的 run.log 切片 + 抓 model_dir
    finalize_one(ctx, start_pos)

print("\n🎉 全部 runs 串行完成（或已全部复用运行完成）。")
print("下一步：批量 model_dir -> valset 推理网格 -> 总榜排名（我可以继续给你写）。")


# In[ ]:





# In[ ]:





# In[ ]:


# Cell 4：断网恢复 & 全局体检（RUNS 恢复 + 每个 run 状态一览）
import os, json, re, time
from pathlib import Path
from datetime import datetime

EXP_DIR = Path("/gpfs/share/home/2306391536/projects/cell_seg/runs/exp_20260305_h100_sweep_20260305_120333").resolve()
ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
CFG_DIR = EXP_DIR / "config"
LOG_DIR = EXP_DIR / "logs"
MET_DIR = EXP_DIR / "metrics"
INFER_DIR = EXP_DIR / "infer"
EVAL_DIR = EXP_DIR / "eval"
EXPORT_DIR = EXP_DIR / "exports"

for d in [CFG_DIR, LOG_DIR, MET_DIR, INFER_DIR, EVAL_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_INDEX_PATH = CFG_DIR / "RUNS.jsonl"
assert RUN_INDEX_PATH.exists(), f"没找到 {RUN_INDEX_PATH}，你是不是还没跑 Cell2？"

RUNS = [json.loads(x) for x in RUN_INDEX_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
print("✅ Loaded RUNS:", len(RUNS))

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

LOSS_LINE = re.compile(r"(?P<epoch>\d+)\s*,\s*train_loss=(?P<tr>[0-9]*\.?[0-9]+)\s*,\s*test_loss=(?P<te>[0-9]*\.?[0-9]+)\s*,\s*LR=(?P<lr>[0-9]*\.?[0-9]+)", re.I)
SAVE_LINE = re.compile(r"saving model to\s+(?P<path>/\S+)", re.I)

def summarize_run(ctx: dict):
    snap_path = Path(ctx["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
    # log 优先用 snapshot 里的（如果你存了），否则退回 ctx
    log_path = Path(snap.get("cellpose_runlog_path") or snap.get("log_path") or ctx["log_path"])
    pid_path = Path(ctx["pid_path"])
    status = {"run_name": ctx["run_name"], "tag": ctx.get("tag", "?")}
    status["model_dir"] = snap.get("model_dir")  # ✅ 关键：从 snapshot 拿

    pid = None
    if pid_path.exists():
        try: pid = int(pid_path.read_text().strip())
        except: pid = None

    status["pid"] = pid
    status["alive"] = _pid_alive(pid) if pid else False
    status["log_exists"] = log_path.exists()
    status["log_size_mb"] = round(log_path.stat().st_size/1024/1024, 2) if log_path.exists() else 0.0

    # 最新 loss 行 & 是否出现 saving model to
    last_loss = None
    last_save = None
    if log_path.exists():
        lines = log_path.read_text(errors="ignore").splitlines()
        for ln in reversed(lines):
            if last_save is None:
                m = SAVE_LINE.search(ln)
                if m: last_save = m.group("path")
            if last_loss is None:
                m = LOSS_LINE.search(ln)
                if m:
                    last_loss = {
                        "epoch": int(m.group("epoch")),
                        "train_loss": float(m.group("tr")),
                        "test_loss": float(m.group("te")),
                        "lr": float(m.group("lr")),
                    }
            if last_loss and last_save:
                break

    status["last_loss"] = last_loss
    # status["model_dir"] 已经从 snapshot 来了
    # last_save 你可以留作“log 里有没有出现过”的辅助字段（可选）
    status["model_dir_from_log"] = last_save
    return status

print("\n" + "="*90)
print("📋 GLOBAL STATUS (断网回来就看这个)")
print("="*90)

rows = []
for ctx in RUNS:
    s = summarize_run(ctx)
    rows.append(s)
    ll = s["last_loss"]
    ll_str = f"ep{ll['epoch']} tr{ll['train_loss']:.4f} te{ll['test_loss']:.4f} lr{ll['lr']:.2e}" if ll else "—"
    md = (s["model_dir"][-55:] if s["model_dir"] else "—")
    print(f"- {s['tag']:<18} | alive={str(s['alive']):<5} | pid={str(s['pid']):<8} | logMB={s['log_size_mb']:<6} | last={ll_str} | model={md}")

print("\n✅ Cell 4 done.")


# In[ ]:


# Cell 5：批量锁定 model_dir（从 log 抓 saving model to），并写入 index
import json, re
from pathlib import Path

SAVE_LINE = re.compile(r"saving model to\s+(?P<path>/\S+)", re.I)

MODEL_INDEX_PATH = (CFG_DIR / "MODEL_INDEX.json").resolve()
model_index = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "exp_dir": str(EXP_DIR),
    "models": []
}

def find_model_dir(log_path: Path):
    if not log_path.exists(): 
        return None
    lines = log_path.read_text(errors="ignore").splitlines()
    cand = []
    for ln in lines:
        m = SAVE_LINE.search(ln)
        if m:
            cand.append(m.group("path"))
    return cand[-1] if cand else None

updated = 0
for ctx in RUNS:
    snap_path = Path(ctx["config_snapshot_path"])
    snap = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else {}
    mdir = snap.get("model_dir")   # ✅ 直接读 snapshot

ctx["model_dir"] = mdir  # 这行可留，方便后面 cell 用 ctx

    model_index["models"].append({
        "run_name": ctx["run_name"],
        "tag": ctx.get("tag"),
        "params": ctx.get("params", {}),
        "log_path": ctx["log_path"],
        "metrics_path": ctx["metrics_path"],
        "model_dir": mdir,
        "config_snapshot_path": ctx["config_snapshot_path"],
    })

MODEL_INDEX_PATH.write_text(json.dumps(model_index, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"✅ Found model_dir for {updated}/{len(RUNS)} runs")
print("📌 MODEL_INDEX:", MODEL_INDEX_PATH)


# In[ ]:


# Cell 6：批量推理网格（--savedir 归档到 INFER_DIR，不污染 valset）
import shlex, subprocess, itertools, json
from pathlib import Path

# ✅ 直接使用 Cell1/Cell4 的全局 VAL_DIR（你应该在更早的 cell 里定义）
assert "VAL_DIR" in globals(), "VAL_DIR 没定义：请在 Cell1 固定写死 valset 路径"
VAL_DIR = Path(VAL_DIR).resolve()
assert VAL_DIR.exists()

# ===== 网格配置（你想缩小就改这里）=====
DIAM_LIST = [16, 20, 24]
CELLPROB_LIST = [-1.0, 0.0, 1.0]
FLOW_LIST = [0.2, 0.4, 0.6]
MIN_SIZE_LIST = [0, 10]
# =====================================

def run_infer_one(model_label: str, pretrained_model: str, out_dir: Path, diameter: int, cellprob: float, flow: float, min_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "cellpose",
        "--dir", str(VAL_DIR),
        "--pretrained_model", str(pretrained_model),
        "--diameter", str(diameter),
        "--cellprob_threshold", str(cellprob),
        "--flow_threshold", str(flow),
        "--min_size", str(min_size),
        "--use_gpu",
        "--save_tif",
        "--no_npy",
        "--savedir", str(out_dir),
    ]
    print("▶", model_label, "|", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        # 把尾巴打印出来就够定位
        print("❌ infer failed:", model_label)
        print((r.stderr or r.stdout)[-2000:])
        raise RuntimeError("Inference crashed")
    return True

# 生成一个 “infer_manifest.jsonl”，每个组合一行，便于断线恢复
MANIFEST = (INFER_DIR / "infer_manifest.jsonl").resolve()

def already_done(out_dir: Path):
    # 简单判定：存在至少 10 张 masks 就认为完成（valset 55 张的话肯定够）
    return len(list(out_dir.glob("*_cp_masks.tif"))) >= 10

jobs = []

# 0) baseline cpsam 也纳入
jobs.append({
    "model_label": "BASELINE_cpsam",
    "pretrained_model": "cpsam",
})

# 1) 每个 finetuned model
for m in model_index["models"]:
    if not m.get("model_dir"):
        continue
    jobs.append({
        "model_label": m["tag"],
        "pretrained_model": m["model_dir"],
    })

print("✅ Models to infer:", len(jobs))

# 主循环
n_total = 0
n_skipped = 0
with open(MANIFEST, "a", encoding="utf-8") as f:
    for jb in jobs:
        for (d, cp, fl, ms) in itertools.product(DIAM_LIST, CELLPROB_LIST, FLOW_LIST, MIN_SIZE_LIST):
            tag = f"d{d}_cp{cp}_fl{fl}_ms{ms}"
            out_dir = (INFER_DIR / jb["model_label"] / "valset" / f"masks_{tag}").resolve()

            rec = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "model_label": jb["model_label"],
                "pretrained_model": jb["pretrained_model"],
                "diameter": d, "cellprob": cp, "flow": fl, "min_size": ms,
                "out_dir": str(out_dir),
            }

            if already_done(out_dir):
                n_skipped += 1
                continue

            run_infer_one(jb["model_label"], jb["pretrained_model"], out_dir, d, cp, fl, ms)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_total += 1

print(f"🎉 Inference jobs finished: {n_total} (skipped existing: {n_skipped})")
print("📌 MANIFEST:", MANIFEST)


# In[ ]:


from pathlib import Path
import random

# 1) 确认 VAL_ROOT 里有没有“原图”和“GT masks”
VAL_ROOT = Path(VAL_DIR)   # 你现在 Cell7 就是这么干的
imgs = sorted([p for p in VAL_ROOT.glob("*.tif") if "_masks" not in p.name and "_flows" not in p.name])
gts  = sorted(list(VAL_ROOT.glob("*_masks.tif")))

print("VAL_ROOT =", VAL_ROOT)
print("n_images =", len(imgs))
print("n_gt_masks =", len(gts))
assert len(imgs) > 0, "VAL_ROOT 里没有原图 tif（不含 _masks/_flows）"
assert len(gts) > 0, "VAL_ROOT 里没有 *_masks.tif（GT 不在这里）"

# 随机抽 1 张，检查它的 GT 是否存在
img = random.choice(imgs)
name = img.stem
gt_path = VAL_ROOT / f"{name}_masks.tif"
print("Sample image:", img.name)
print("Expected GT :", gt_path.name, "exists=", gt_path.exists())
assert gt_path.exists(), "这张图的 GT 不在 VAL_ROOT：说明 VAL_ROOT 指错目录 or GT 命名不一致"

# 2) 确认推理输出里有没有 *_cp_masks.tif
# 随便找一个 pred_dir（拿第一个模型的第一个 masks_* 目录）
pred_candidates = sorted((INFER_DIR).glob("*/*/masks_*"))
assert len(pred_candidates) > 0, "INFER_DIR 下没找到 */valset/masks_* 目录：说明 Cell6 还没跑出结果 or 目录结构不同"
pred_dir = pred_candidates[0]
pred_path = pred_dir / f"{name}_cp_masks.tif"

print("Sample pred_dir:", pred_dir)
print("Expected Pred :", pred_path.name, "exists=", pred_path.exists())
assert pred_path.exists(), "预测文件名/路径不匹配：Cell6 没产出 *_cp_masks.tif 或者你的 pred_dir 结构不对"

print("✅ 结论：Cell7 需要的 GT 路径 & Pred 文件名 都对得上，可以跑评估。")


# In[ ]:


# Cell 7：批量评估总榜（输出 summary_ranking.csv）
import os
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from cellpose import metrics

VAL_ROOT = VAL_DIR  # 原图+GT 都在这里
assert VAL_ROOT.exists()

def compute_all_metrics(gt_mask, pred_mask):
    ap, tp, fp, fn = metrics.average_precision(gt_mask, pred_mask, threshold=[0.5])
    ap50 = ap[0]
    tp_val, fp_val, fn_val = tp[0], fp[0], fn[0]
    precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
    recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0
    inter = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (gt_bool.sum() + pred_bool.sum()) if (gt_bool.sum() + pred_bool.sum()) > 0 else 0.0
    return ap50, precision, recall, f1, dice, iou

# valset 图片名列表
img_list = [p for p in VAL_ROOT.glob("*.tif") if "_masks" not in p.name and "_flows" not in p.name]
assert len(img_list) > 0, "VAL_ROOT 没找到原图 tif"

def eval_one_dir(pred_dir: Path):
    records = []
    for img_path in img_list:
        name = img_path.stem
        gt_path = VAL_ROOT / f"{name}_masks.tif"
        pred_path = pred_dir / f"{name}_cp_masks.tif"
        if not (gt_path.exists() and pred_path.exists()):
            continue
        gt = tiff.imread(str(gt_path))
        pr = tiff.imread(str(pred_path))

        ap50, precision, recall, f1, dice, iou = compute_all_metrics(gt, pr)

        gt_count = len(np.unique(gt)) - 1
        pr_count = len(np.unique(pr)) - 1
        cnt_err = abs(pr_count - gt_count) / gt_count if gt_count > 0 else 0.0

        records.append((ap50, precision, recall, f1, dice, iou, cnt_err))
    if not records:
        return None
    arr = np.array(records, dtype=float)
    return {
        "n_images": len(records),
        "AP50": arr[:,0].mean(),
        "Precision": arr[:,1].mean(),
        "Recall": arr[:,2].mean(),
        "F1": arr[:,3].mean(),
        "Dice": arr[:,4].mean(),
        "IoU": arr[:,5].mean(),
        "CntErr": arr[:,6].mean(),
    }

rows = []
# 扫描所有推理输出
for model_label_dir in INFER_DIR.iterdir():
    if not model_label_dir.is_dir():
        continue
    valset_dir = model_label_dir / "valset"
    if not valset_dir.exists():
        continue
    for masks_dir in sorted(valset_dir.glob("masks_*")):
        s = eval_one_dir(masks_dir)
        if s is None:
            continue
        # 从目录名解析参数 tag
        tag = masks_dir.name.replace("masks_", "")
        rows.append({
            "model_label": model_label_dir.name,
            "infer_tag": tag,
            "pred_dir": str(masks_dir),
            **s
        })

df = pd.DataFrame(rows)
assert not df.empty, "没评估到任何结果：请先跑 Cell6 推理"

# 排名策略：先看 AP50，再看 Recall，再看 CntErr（越低越好）
df["Score"] = df["AP50"] + 0.25*df["Recall"] - 0.10*df["CntErr"]
df = df.sort_values(["Score","AP50","Recall"], ascending=False).reset_index(drop=True)

OUT = (EVAL_DIR / "summary_ranking.csv").resolve()
df.to_csv(OUT, index=False)

print("🏆 Top 10 combos:")
display(df.head(10)[["model_label","infer_tag","AP50","Recall","Precision","F1","Dice","CntErr","n_images","Score"]])
print("\n✅ saved:", OUT)


# In[ ]:


# Cell 8：TopK 交通灯诊断图导出（Top3 × {Worst/Median/Best}）
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from pathlib import Path
from cellpose import metrics

rank_csv = (EVAL_DIR / "summary_ranking.csv").resolve()
df = pd.read_csv(rank_csv)
TOPK = 3
top = df.head(TOPK).copy()

OUT_DIR = (EXPORT_DIR / "topk_diagnostics").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm01(img, p_low=1, p_high=99):
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

def get_tp_fp_fn_masks(gt_mask, pred_mask, iou_threshold=0.5):
    iou = metrics._intersection_over_union(gt_mask, pred_mask)[1:, 1:]
    tp_pred_ids, fp_pred_ids, tp_gt_ids, fn_gt_ids = [], [], [], []
    if iou.size > 0:
        best_gt_ious = iou.max(axis=0)
        best_gt_indices = iou.argmax(axis=0) + 1
        for pred_idx, max_i in enumerate(best_gt_ious):
            p_id = pred_idx + 1
            if max_i > iou_threshold:
                tp_pred_ids.append(p_id)
                tp_gt_ids.append(best_gt_indices[pred_idx])
            else:
                fp_pred_ids.append(p_id)
        all_gt_ids = set(np.unique(gt_mask)) - {0}
        fn_gt_ids = list(all_gt_ids - set(tp_gt_ids))
    else:
        fp_pred_ids = list(np.unique(pred_mask)[1:])
        fn_gt_ids = list(np.unique(gt_mask)[1:])
    mask_tp = np.isin(pred_mask, tp_pred_ids)
    mask_fp = np.isin(pred_mask, fp_pred_ids)
    mask_fn = np.isin(gt_mask, fn_gt_ids)
    return mask_tp, mask_fp, mask_fn

def ap50(gt, pr):
    ap, _, _, _ = metrics.average_precision(gt, pr, threshold=[0.5])
    return float(ap[0])

# 图片列表
img_list = [p for p in VAL_ROOT.glob("*.tif") if "_masks" not in p.name and "_flows" not in p.name]

for _, row in top.iterrows():
    model_label = row["model_label"]
    infer_tag = row["infer_tag"]
    pred_dir = Path(row["pred_dir"])

    # 逐图打分，挑 worst/median/best
    per = []
    for img_path in img_list:
        name = img_path.stem
        gt_path = VAL_ROOT / f"{name}_masks.tif"
        pr_path = pred_dir / f"{name}_cp_masks.tif"
        if not (gt_path.exists() and pr_path.exists()):
            continue
        gt = tiff.imread(str(gt_path))
        pr = tiff.imread(str(pr_path))
        per.append((name, ap50(gt, pr)))
    if not per:
        continue
    per.sort(key=lambda x: x[1])
    picks = [("Worst", per[0][0]), ("Median", per[len(per)//2][0]), ("Best", per[-1][0])]

    for label, name in picks:
        img = tiff.imread(str(VAL_ROOT / f"{name}.tif"))
        gt = tiff.imread(str(VAL_ROOT / f"{name}_masks.tif"))
        pr = tiff.imread(str(pred_dir / f"{name}_cp_masks.tif"))
        if img.ndim > 2: img = img.squeeze()
        if img.ndim > 2: img = img[0]
        img01 = norm01(img)

        tp, fp, fn = get_tp_fp_fn_masks(gt, pr)
        fn_outline = binary_dilation(fn) ^ fn
        score = ap50(gt, pr)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img01, cmap="gray")
        axes[0].imshow(np.ma.masked_where(gt==0, gt), cmap="nipy_spectral", alpha=0.35)
        axes[0].set_title(f"GT | n={len(np.unique(gt))-1}")
        axes[0].axis("off")

        axes[1].imshow(img01, cmap="gray")
        axes[1].imshow(np.ma.masked_where(pr==0, pr), cmap="nipy_spectral", alpha=0.35)
        axes[1].set_title(f"Pred | n={len(np.unique(pr))-1}")
        axes[1].axis("off")

        axes[2].imshow(img01, cmap="gray")
        axes[2].imshow(np.ma.masked_where(~tp, tp), cmap="Greens", alpha=0.5)
        axes[2].imshow(np.ma.masked_where(~fp, fp), cmap="Reds", alpha=0.6)
        axes[2].imshow(np.ma.masked_where(~fn_outline, fn_outline), cmap="autumn", alpha=1.0)
        axes[2].set_title(f"TP/FP/FN | AP50={score:.3f}")
        axes[2].axis("off")

        plt.suptitle(f"{model_label} | {infer_tag} | {label} | {name}", fontsize=14, y=1.02)
        plt.tight_layout()

        out = OUT_DIR / f"{model_label}__{infer_tag}__{label}__{name}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("✅ saved:", out.name)

print("\n🎉 TopK diagnostics saved to:", OUT_DIR)


# In[ ]:




