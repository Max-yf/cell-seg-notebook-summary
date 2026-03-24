#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1：配置实验目录与输出目录
from pathlib import Path
from datetime import datetime
import json
import os

print("="*80)
print("🧱 Cell 1 | Configure experiment list and output root")
print("="*80)

ROOT = Path("/gpfs/share/home/2306391536/projects/cell_seg").resolve()
RUNS_ROOT = (ROOT / "runs").resolve()

EXP_NAMES = [
    "exp_20260306_retrain_newval_20260306_144900",
    "exp_20260306_v4_finetune_20260306_194113",
    "exp_20260307_refine_20260307_132357",
    "exp_20260307_V2_refine_20260307_153314",
    "exp_20260307_V2_trans_confirm_20260307_172946",
    "exp_20260309_grid9_trans_wd_epochs_20260310_135800",
    "exp_20260310_lr_wd_6runs_400ep_20260310_180121",
    "exp_20260311_ultrasmall_refine_6runs_400ep_es50_20260311_115944",
]

EXP_DIRS = [(RUNS_ROOT / x).resolve() for x in EXP_NAMES]

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIT_ROOT = (ROOT / "runs" / f"loss_audit_all_experiments_{STAMP}").resolve()

FIG_DIR = AUDIT_ROOT / "figures"
TABLE_DIR = AUDIT_ROOT / "tables"
CACHE_DIR = AUDIT_ROOT / "cache"

for d in [AUDIT_ROOT, FIG_DIR, TABLE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

cfg = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "root": str(ROOT),
    "runs_root": str(RUNS_ROOT),
    "audit_root": str(AUDIT_ROOT),
    "experiments": [str(x) for x in EXP_DIRS],
}
(AUDIT_ROOT / "audit_config.json").write_text(
    json.dumps(cfg, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("ROOT      :", ROOT)
print("RUNS_ROOT :", RUNS_ROOT)
print("AUDIT_ROOT:", AUDIT_ROOT)
print("\nExperiments:")
for i, p in enumerate(EXP_DIRS, 1):
    print(f"[{i}] {p} | exists={p.exists()}")

assert ROOT.exists(), f"ROOT not found: {ROOT}"
assert RUNS_ROOT.exists(), f"RUNS_ROOT not found: {RUNS_ROOT}"

print("\n✅ Cell 1 done.")
print("="*80)


# In[2]:


# Cell 2：扫描实验资产
from pathlib import Path
import json
import pandas as pd

print("="*80)
print("🧭 Cell 2 | Scan experiment assets")
print("="*80)

rows = []

for exp_dir in EXP_DIRS:
    exp_name = exp_dir.name
    cfg_dir = exp_dir / "config"
    log_dir = exp_dir / "logs"
    export_dir = exp_dir / "exports"

    runs_jsonl = cfg_dir / "RUNS.jsonl"
    paths_json = cfg_dir / "PATHS.json"
    summary_csv = exp_dir / "summary_runs.csv"
    final_rank_csv = exp_dir / "final_merged_ranking.csv"

    n_runs = None
    if runs_jsonl.exists():
        try:
            runs = [json.loads(l) for l in runs_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
            n_runs = len(runs)
        except Exception:
            n_runs = None

    n_logs = len(list(log_dir.glob("*.log"))) if log_dir.exists() else 0

    rows.append({
        "experiment": exp_name,
        "exp_dir": str(exp_dir),
        "exists": exp_dir.exists(),
        "has_config_dir": cfg_dir.exists(),
        "has_log_dir": log_dir.exists(),
        "has_export_dir": export_dir.exists(),
        "has_runs_jsonl": runs_jsonl.exists(),
        "has_paths_json": paths_json.exists(),
        "has_summary_runs_csv": summary_csv.exists(),
        "has_final_merged_ranking_csv": final_rank_csv.exists(),
        "n_runs": n_runs,
        "n_log_files": n_logs,
    })

df_assets = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
assets_csv = TABLE_DIR / "experiment_assets_inventory.csv"
df_assets.to_csv(assets_csv, index=False)

print("✅ saved:", assets_csv)
display(df_assets)

print("\n✅ Cell 2 done.")
print("="*80)


# In[3]:


# Cell 3：统一解析日志 -> long table + run summary
import json, re
from pathlib import Path
import pandas as pd
import numpy as np

print("="*80)
print("📦 Cell 3 | Parse all logs into long table and run summary")
print("="*80)

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

long_rows = []
summary_rows = []

for exp_dir in EXP_DIRS:
    exp_name = exp_dir.name
    runs_jsonl = exp_dir / "config" / "RUNS.jsonl"

    if not runs_jsonl.exists():
        print(f"⚠️ skip {exp_name}: RUNS.jsonl missing")
        continue

    runs = [json.loads(l) for l in runs_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"📁 {exp_name} | runs={len(runs)}")

    for r in runs:
        tag = r.get("tag")
        run_name = r.get("run_name")
        params = r.get("params", {})
        log_path = Path(r["log_path"])

        snap = read_json(Path(r["config_snapshot_path"])) or {}
        status = snap.get("status", r.get("status", "UNKNOWN"))
        model_dir = snap.get("model_dir") or snap.get("final_model_path") or r.get("model_dir")

        if not log_path.exists():
            summary_rows.append({
                "experiment": exp_name,
                "tag": tag,
                "run_name": run_name,
                "status": status,
                "log_path": str(log_path),
                "log_exists": False,
                "model_dir": model_dir,
                "model_dir_exists": bool(model_dir) and Path(model_dir).exists(),
                "diameter": params.get("diameter"),
                "lr_cfg": params.get("learning_rate"),
                "weight_decay": params.get("weight_decay"),
                "train_batch_size": params.get("train_batch_size"),
                "n_epochs_cfg": params.get("n_epochs"),
                "augment": params.get("augment", False),
                "transformer": params.get("transformer", False),
                "n_logged_points": 0,
                "best_epoch_by_test_loss": np.nan,
                "best_train_loss_at_best_test": np.nan,
                "best_test_loss": np.nan,
                "last_epoch_logged": np.nan,
                "last_train_loss": np.nan,
                "last_test_loss": np.nan,
            })
            print(f"   ⚠️ missing log: {tag}")
            continue

        text = log_path.read_text(encoding="utf-8", errors="ignore")

        metrics = []
        for m in MET_RE.finditer(text):
            row = {
                "experiment": exp_name,
                "tag": tag,
                "run_name": run_name,
                "epoch": int(m.group("epoch")),
                "train_loss": float(m.group("tr")),
                "test_loss": float(m.group("te")),
                "lr_logged": float(m.group("lr")),
                "log_path": str(log_path),
            }
            metrics.append(row)
            long_rows.append(row)

        if len(metrics) == 0:
            summary_rows.append({
                "experiment": exp_name,
                "tag": tag,
                "run_name": run_name,
                "status": status,
                "log_path": str(log_path),
                "log_exists": True,
                "model_dir": model_dir,
                "model_dir_exists": bool(model_dir) and Path(model_dir).exists(),
                "diameter": params.get("diameter"),
                "lr_cfg": params.get("learning_rate"),
                "weight_decay": params.get("weight_decay"),
                "train_batch_size": params.get("train_batch_size"),
                "n_epochs_cfg": params.get("n_epochs"),
                "augment": params.get("augment", False),
                "transformer": params.get("transformer", False),
                "n_logged_points": 0,
                "best_epoch_by_test_loss": np.nan,
                "best_train_loss_at_best_test": np.nan,
                "best_test_loss": np.nan,
                "last_epoch_logged": np.nan,
                "last_train_loss": np.nan,
                "last_test_loss": np.nan,
            })
            print(f"   ⚠️ no parsed metrics: {tag}")
            continue

        dfm = pd.DataFrame(metrics).sort_values("epoch").reset_index(drop=True)
        best_idx = dfm["test_loss"].idxmin()

        summary_rows.append({
            "experiment": exp_name,
            "tag": tag,
            "run_name": run_name,
            "status": status,
            "log_path": str(log_path),
            "log_exists": True,
            "model_dir": model_dir,
            "model_dir_exists": bool(model_dir) and Path(model_dir).exists(),
            "diameter": params.get("diameter"),
            "lr_cfg": params.get("learning_rate"),
            "weight_decay": params.get("weight_decay"),
            "train_batch_size": params.get("train_batch_size"),
            "n_epochs_cfg": params.get("n_epochs"),
            "augment": params.get("augment", False),
            "transformer": params.get("transformer", False),
            "n_logged_points": int(dfm.shape[0]),
            "best_epoch_by_test_loss": int(dfm.loc[best_idx, "epoch"]),
            "best_train_loss_at_best_test": float(dfm.loc[best_idx, "train_loss"]),
            "best_test_loss": float(dfm.loc[best_idx, "test_loss"]),
            "last_epoch_logged": int(dfm.iloc[-1]["epoch"]),
            "last_train_loss": float(dfm.iloc[-1]["train_loss"]),
            "last_test_loss": float(dfm.iloc[-1]["test_loss"]),
        })

df_long = pd.DataFrame(long_rows)
df_summary = pd.DataFrame(summary_rows)

if len(df_long) > 0:
    df_long = df_long.sort_values(["experiment", "tag", "epoch"]).reset_index(drop=True)
df_summary = df_summary.sort_values(["experiment", "tag"]).reset_index(drop=True)

long_csv = TABLE_DIR / "all_runs_loss_long.csv"
summary_csv = TABLE_DIR / "all_runs_loss_summary.csv"

df_long.to_csv(long_csv, index=False)
df_summary.to_csv(summary_csv, index=False)

print("✅ saved:", long_csv)
print("✅ saved:", summary_csv)

print("\nLong table preview:")
display(df_long.head(10))

print("\nSummary preview:")
display(df_summary.head(20))

print("\n✅ Cell 3 done.")
print("="*80)


# In[9]:


# Cell 4：每个实验绘制 all-runs train/test loss 总图（保存 + 直接显示）
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("📈 Cell 4 | Plot per-experiment all-runs train/test loss (save + show)")
print("="*80)

assert "df_long" in globals(), "请先运行 Cell 3"

PER_EXP_FIG_DIR = FIG_DIR / "per_experiment_all_runs"
PER_EXP_FIG_DIR.mkdir(parents=True, exist_ok=True)

experiments = sorted(df_long["experiment"].dropna().unique().tolist())

for exp_name in experiments:
    dfe = df_long[df_long["experiment"] == exp_name].copy()
    if dfe.empty:
        continue

    out_dir = PER_EXP_FIG_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print(f"🧪 Experiment: {exp_name}")
    print("="*100)

    # ---------- Train loss ----------
    plt.figure(figsize=(12, 7))
    for tag, g in dfe.groupby("tag"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["train_loss"], linewidth=1.8, label=tag)
    plt.title(f"{exp_name} | All runs train loss", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_png = out_dir / "all_runs_train_loss.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("🖼 saved:", out_png)
    plt.show()
    plt.close()

    # ---------- Test loss ----------
    plt.figure(figsize=(12, 7))
    for tag, g in dfe.groupby("tag"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["test_loss"], linewidth=1.8, label=tag)
    plt.title(f"{exp_name} | All runs test loss", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_png = out_dir / "all_runs_test_loss.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("🖼 saved:", out_png)
    plt.show()
    plt.close()

print("\n✅ Cell 4 done.")
print("="*80)


# In[10]:


# Cell 5：每个实验绘制 top-k run 的 test loss 精简图（保存 + 直接显示）
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("🥇 Cell 5 | Plot per-experiment top-k test loss (save + show)")
print("="*80)

assert "df_long" in globals() and "df_summary" in globals(), "请先运行 Cell 3"

TOPK = 5

TOPK_FIG_DIR = FIG_DIR / "per_experiment_topk"
TOPK_FIG_DIR.mkdir(parents=True, exist_ok=True)

for exp_name in sorted(df_summary["experiment"].dropna().unique()):
    dfs = df_summary[df_summary["experiment"] == exp_name].copy()
    dfl = df_long[df_long["experiment"] == exp_name].copy()

    dfs = dfs.dropna(subset=["best_test_loss"]).sort_values("best_test_loss", ascending=True).head(TOPK)
    top_tags = dfs["tag"].tolist()

    if len(top_tags) == 0:
        print(f"⚠️ skip {exp_name}: no valid top tags")
        continue

    out_dir = TOPK_FIG_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-"*100)
    print(f"🏁 Experiment: {exp_name} | Top-{TOPK} test loss")
    print("Top tags:", top_tags)
    print("-"*100)

    plt.figure(figsize=(12, 7))
    for tag in top_tags:
        g = dfl[dfl["tag"] == tag].sort_values("epoch")
        if g.empty:
            continue
        best_epoch = int(dfs.loc[dfs["tag"] == tag, "best_epoch_by_test_loss"].iloc[0])
        best_loss = float(dfs.loc[dfs["tag"] == tag, "best_test_loss"].iloc[0])

        plt.plot(
            g["epoch"], g["test_loss"],
            linewidth=2.2,
            label=f"{tag} | best={best_loss:.4f}@{best_epoch}"
        )
        plt.axvline(best_epoch, linestyle="--", linewidth=1.0, alpha=0.5)

    plt.title(f"{exp_name} | Top-{TOPK} runs test loss", fontsize=15)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_png = out_dir / f"top{TOPK}_runs_test_loss.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("🖼 saved:", out_png)
    plt.show()
    plt.close()

print("\n✅ Cell 5 done.")
print("="*80)


# In[6]:


# Cell 6：每个 run 单独 train/test 双曲线
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("🔬 Cell 6 | Plot single-run train/test curves")
print("="*80)

assert "df_long" in globals() and "df_summary" in globals(), "请先运行 Cell 3"

SINGLE_FIG_DIR = FIG_DIR / "single_run_curves"
SINGLE_FIG_DIR.mkdir(parents=True, exist_ok=True)

for _, row in df_summary.iterrows():
    exp_name = row["experiment"]
    tag = row["tag"]

    g = df_long[(df_long["experiment"] == exp_name) & (df_long["tag"] == tag)].sort_values("epoch")
    if g.empty:
        continue

    out_dir = SINGLE_FIG_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_epoch = row["best_epoch_by_test_loss"]
    best_test = row["best_test_loss"]

    plt.figure(figsize=(10, 5.5))
    plt.plot(g["epoch"], g["train_loss"], linewidth=1.8, label="Train Loss")
    plt.plot(g["epoch"], g["test_loss"], linewidth=1.8, label="Test Loss")

    if pd.notna(best_epoch):
        plt.axvline(float(best_epoch), linestyle="--", linewidth=1.3,
                    label=f"Best epoch={int(best_epoch)}, best test={float(best_test):.4f}")

    plt.title(f"{exp_name} | {tag}", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)

    out_png = out_dir / f"{tag}_train_test_curve.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()

print("✅ single-run plots saved to:", SINGLE_FIG_DIR)
print("\n✅ Cell 6 done.")
print("="*80)


# In[7]:


# Cell 7：按实验拆分导出 loss summary 表
from pathlib import Path

print("="*80)
print("🧾 Cell 7 | Export per-experiment loss summary tables")
print("="*80)

assert "df_summary" in globals(), "请先运行 Cell 3"

PER_EXP_TABLE_DIR = TABLE_DIR / "per_experiment_loss_summary"
PER_EXP_TABLE_DIR.mkdir(parents=True, exist_ok=True)

for exp_name in sorted(df_summary["experiment"].dropna().unique()):
    dfs = df_summary[df_summary["experiment"] == exp_name].copy()
    out_csv = PER_EXP_TABLE_DIR / f"{exp_name}__loss_summary.csv"
    dfs.to_csv(out_csv, index=False)
    print("✅ saved:", out_csv)

print("\n✅ Cell 7 done.")
print("="*80)


# In[11]:


# Cell 8：跨实验总览图（保存 + 关键图直接显示）
import matplotlib.pyplot as plt

print("="*80)
print("🌍 Cell 8 | Cross-experiment overview plots (save + selective show)")
print("="*80)

assert "df_long" in globals() and "df_summary" in globals(), "请先运行 Cell 3"

CROSS_DIR = FIG_DIR / "cross_experiment"
CROSS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 图 1：所有 run 的 test loss 大乱斗（保存，不默认 show） ----------
plt.figure(figsize=(16, 9))
for (exp_name, tag), g in df_long.groupby(["experiment", "tag"]):
    g = g.sort_values("epoch")
    plt.plot(g["epoch"], g["test_loss"], linewidth=1.2, alpha=0.9, label=f"{exp_name} | {tag}")

plt.title("All experiments | all runs test loss", fontsize=16)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=6, ncol=2)
plt.tight_layout()

out_png = CROSS_DIR / "all_experiments_all_runs_test_loss.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("🖼 saved:", out_png)
plt.close()

# ---------- 图 2：每个实验 top1 的 test loss（保存 + 直接显示） ----------
top1_rows = (
    df_summary
    .dropna(subset=["best_test_loss"])
    .sort_values(["experiment", "best_test_loss"], ascending=[True, True])
    .groupby("experiment", as_index=False)
    .first()
)

plt.figure(figsize=(14, 8))
for _, row in top1_rows.iterrows():
    exp_name = row["experiment"]
    tag = row["tag"]
    g = df_long[(df_long["experiment"] == exp_name) & (df_long["tag"] == tag)].sort_values("epoch")
    if g.empty:
        continue
    plt.plot(
        g["epoch"], g["test_loss"],
        linewidth=2.2,
        label=f"{exp_name} | {tag} | best={row['best_test_loss']:.4f}"
    )

plt.title("All experiments | top-1 run test loss", fontsize=16)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

out_png = CROSS_DIR / "all_experiments_top1_test_loss.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("🖼 saved:", out_png)
plt.show()
plt.close()

print("\n✅ Cell 8 done.")
print("="*80)


# In[12]:


# Cell 9：扫描各实验的评估文件
from pathlib import Path
import pandas as pd

print("="*80)
print("🧭 Cell 9 | Scan evaluation files across experiments")
print("="*80)

def find_eval_candidates(exp_dir: Path):
    cands = []

    # 1) 最优先：final_merged_ranking.csv
    p = exp_dir / "final_merged_ranking.csv"
    if p.exists():
        cands.append(("final_merged_ranking", p))

    # 2) 直接在实验目录下找 eval_metrics.csv
    for p in exp_dir.rglob("eval_metrics.csv"):
        cands.append(("eval_metrics", p))

    # 3) report summary 里的表
    for p in exp_dir.rglob("ranking_report_table.csv"):
        cands.append(("ranking_report_table", p))

    # 去重
    seen = set()
    uniq = []
    for kind, p in cands:
        key = str(p.resolve())
        if key not in seen:
            uniq.append((kind, p))
            seen.add(key)
    return uniq

rows = []
for exp_dir in EXP_DIRS:
    exp_name = exp_dir.name
    cands = find_eval_candidates(exp_dir)

    if len(cands) == 0:
        rows.append({
            "experiment": exp_name,
            "exp_dir": str(exp_dir),
            "has_eval_candidate": False,
            "kind": None,
            "path": None,
        })
    else:
        for kind, p in cands:
            rows.append({
                "experiment": exp_name,
                "exp_dir": str(exp_dir),
                "has_eval_candidate": True,
                "kind": kind,
                "path": str(p),
            })

df_eval_assets = pd.DataFrame(rows).sort_values(["experiment", "kind"], na_position="last").reset_index(drop=True)
out_csv = TABLE_DIR / "evaluation_assets_inventory.csv"
df_eval_assets.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_eval_assets)

print("\n✅ Cell 9 done.")
print("="*80)


# In[13]:


# Cell 10：统一读取评估结果 -> 标准化 eval summary
import pandas as pd
from pathlib import Path
import numpy as np

print("="*80)
print("📥 Cell 10 | Load and normalize evaluation tables")
print("="*80)

def pick_best_eval_file(exp_dir: Path):
    """
    优先级：
    1. final_merged_ranking.csv
    2. 任意 eval_metrics.csv
    3. ranking_report_table.csv
    """
    p1 = exp_dir / "final_merged_ranking.csv"
    if p1.exists():
        return "final_merged_ranking", p1

    evals = sorted(exp_dir.rglob("eval_metrics.csv"))
    if len(evals) > 0:
        return "eval_metrics", evals[0]

    reports = sorted(exp_dir.rglob("ranking_report_table.csv"))
    if len(reports) > 0:
        return "ranking_report_table", reports[0]

    return None, None

def normalize_eval_df(df_raw: pd.DataFrame, exp_name: str, source_kind: str, source_path: Path):
    df = df_raw.copy()

    # -------- tag 字段兼容 --------
    if "tag" in df.columns:
        df["tag_std"] = df["tag"].astype(str)
    elif "model_tag" in df.columns:
        df["tag_std"] = df["model_tag"].astype(str)
    else:
        df["tag_std"] = np.nan

    # -------- 标准字段补齐 --------
    wanted_cols = [
        "AP50", "Precision", "Recall", "F1",
        "n_eval", "n_missing_pred",
        "eval_model_type", "eval_model_path",
        "best_epoch_by_test_loss", "best_test_loss",
        "last_epoch_logged", "last_test_loss",
        "lr", "weight_decay", "n_epochs",
    ]

    for c in wanted_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 有些表可能没有 n_epochs，但有 n_epochs_cfg；这里只做轻兼容
    if "n_epochs" not in df.columns and "n_epochs_cfg" in df.columns:
        df["n_epochs"] = df["n_epochs_cfg"]

    keep = [
        "tag_std",
        "AP50", "Precision", "Recall", "F1",
        "n_eval", "n_missing_pred",
        "eval_model_type", "eval_model_path",
        "best_epoch_by_test_loss", "best_test_loss",
        "last_epoch_logged", "last_test_loss",
        "lr", "weight_decay", "n_epochs",
    ]

    out = df[keep].copy()
    out.insert(0, "experiment", exp_name)
    out.insert(1, "eval_source_kind", source_kind)
    out.insert(2, "eval_source_path", str(source_path))
    return out

eval_tables = []

for exp_dir in EXP_DIRS:
    exp_name = exp_dir.name
    source_kind, source_path = pick_best_eval_file(exp_dir)

    if source_path is None:
        print(f"⚠️ no eval file found: {exp_name}")
        continue

    try:
        df_raw = pd.read_csv(source_path)
        df_norm = normalize_eval_df(df_raw, exp_name, source_kind, source_path)
        eval_tables.append(df_norm)
        print(f"✅ loaded eval: {exp_name} | {source_kind} | rows={len(df_norm)}")
    except Exception as e:
        print(f"❌ failed to read eval file: {exp_name} | {source_path} | {repr(e)}")

if len(eval_tables) > 0:
    df_eval_summary = pd.concat(eval_tables, ignore_index=True)
    df_eval_summary = df_eval_summary.sort_values(["experiment", "tag_std"]).reset_index(drop=True)
else:
    df_eval_summary = pd.DataFrame(columns=[
        "experiment", "eval_source_kind", "eval_source_path", "tag_std",
        "AP50", "Precision", "Recall", "F1",
        "n_eval", "n_missing_pred",
        "eval_model_type", "eval_model_path",
        "best_epoch_by_test_loss", "best_test_loss",
        "last_epoch_logged", "last_test_loss",
        "lr", "weight_decay", "n_epochs",
    ])

out_csv = TABLE_DIR / "all_runs_eval_summary.csv"
df_eval_summary.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_eval_summary.head(30))

print("\n✅ Cell 10 done.")
print("="*80)


# In[14]:


# Cell 11：合并 loss summary + eval summary
import pandas as pd
import numpy as np

print("="*80)
print("🔗 Cell 11 | Merge loss summary with evaluation summary")
print("="*80)

assert "df_summary" in globals(), "请先运行 loss 部分的 Cell 3"
assert "df_eval_summary" in globals(), "请先运行 Cell 10"

df_loss_merge = df_summary.copy()
df_loss_merge["tag_std"] = df_loss_merge["tag"].astype(str)

# 避免重复列冲突：只保留 eval 侧更有意义的那部分
eval_keep = [
    "experiment", "tag_std",
    "AP50", "Precision", "Recall", "F1",
    "n_eval", "n_missing_pred",
    "eval_model_type", "eval_model_path",
    "eval_source_kind", "eval_source_path",
]

df_eval_merge = df_eval_summary[eval_keep].copy()

df_merged = df_loss_merge.merge(
    df_eval_merge,
    how="left",
    on=["experiment", "tag_std"]
)

# 一些便于排序/查看的新字段
df_merged["has_eval"] = df_merged["AP50"].notna()
df_merged["loss_rank_in_exp"] = (
    df_merged.groupby("experiment")["best_test_loss"]
    .rank(method="min", ascending=True)
)
df_merged["ap50_rank_in_exp"] = (
    df_merged.groupby("experiment")["AP50"]
    .rank(method="min", ascending=False)
)

out_csv = TABLE_DIR / "all_runs_loss_eval_merged.csv"
df_merged.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_merged.head(30))

print("\n有评估结果的 run 数量：", int(df_merged["has_eval"].sum()), "/", len(df_merged))
print("\n✅ Cell 11 done.")
print("="*80)


# In[15]:


# Cell 12：按实验导出 merged summary，并显示每实验 top 表
from pathlib import Path
import pandas as pd

print("="*80)
print("🧾 Cell 12 | Export per-experiment merged summary tables")
print("="*80)

assert "df_merged" in globals(), "请先运行 Cell 11"

MERGED_PER_EXP_DIR = TABLE_DIR / "per_experiment_loss_eval_merged"
MERGED_PER_EXP_DIR.mkdir(parents=True, exist_ok=True)

for exp_name in sorted(df_merged["experiment"].dropna().unique()):
    dfe = df_merged[df_merged["experiment"] == exp_name].copy()

    out_csv = MERGED_PER_EXP_DIR / f"{exp_name}__loss_eval_merged.csv"
    dfe.to_csv(out_csv, index=False)

    print("\n" + "="*100)
    print(f"🧪 Experiment: {exp_name}")
    print("✅ saved:", out_csv)

    show_cols = [
        "tag", "best_epoch_by_test_loss", "best_test_loss",
        "AP50", "Precision", "Recall", "F1",
        "loss_rank_in_exp", "ap50_rank_in_exp",
        "eval_model_type"
    ]
    show_cols = [c for c in show_cols if c in dfe.columns]

    dfe_show = dfe[show_cols].copy()

    # 优先按 AP50 排，有评估的更适合这么看；没有评估就按 loss
    if dfe_show["AP50"].notna().any():
        dfe_show = dfe_show.sort_values(["AP50", "F1", "best_test_loss"], ascending=[False, False, True])
    else:
        dfe_show = dfe_show.sort_values(["best_test_loss"], ascending=True)

    display(dfe_show.head(10))

print("\n✅ Cell 12 done.")
print("="*80)


# In[16]:


# Cell 13：绘制 best_test_loss vs AP50 散点图
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("🎯 Cell 13 | Plot best_test_loss vs AP50")
print("="*80)

assert "df_merged" in globals(), "请先运行 Cell 11"

SCATTER_DIR = FIG_DIR / "loss_vs_eval"
SCATTER_DIR.mkdir(parents=True, exist_ok=True)

df_plot = df_merged.dropna(subset=["best_test_loss", "AP50"]).copy()

if len(df_plot) == 0:
    print("⚠️ 没有同时包含 best_test_loss 和 AP50 的数据，跳过作图。")
else:
    # ---------- 全部实验总图 ----------
    plt.figure(figsize=(10, 7))
    for exp_name, g in df_plot.groupby("experiment"):
        plt.scatter(g["best_test_loss"], g["AP50"], s=55, alpha=0.85, label=exp_name)

    plt.xlabel("Best Test Loss")
    plt.ylabel("AP50")
    plt.title("All experiments | Best Test Loss vs AP50")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_png = SCATTER_DIR / "all_experiments_best_test_loss_vs_AP50.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print("🖼 saved:", out_png)
    plt.show()
    plt.close()

    # ---------- 每个实验单独图 ----------
    for exp_name in sorted(df_plot["experiment"].unique()):
        g = df_plot[df_plot["experiment"] == exp_name].copy()
        if g.empty:
            continue

        plt.figure(figsize=(8.5, 6.5))
        plt.scatter(g["best_test_loss"], g["AP50"], s=65, alpha=0.9)

        for _, row in g.iterrows():
            plt.annotate(
                row["tag"],
                (row["best_test_loss"], row["AP50"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8
            )

        plt.xlabel("Best Test Loss")
        plt.ylabel("AP50")
        plt.title(f"{exp_name} | Best Test Loss vs AP50")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_png = SCATTER_DIR / f"{exp_name}__best_test_loss_vs_AP50.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        print("🖼 saved:", out_png)
        plt.show()
        plt.close()

print("\n✅ Cell 13 done.")
print("="*80)


# In[17]:


# Cell 14：loss 排名 vs AP50 排名 对照表
import pandas as pd
from pathlib import Path

print("="*80)
print("🏆 Cell 14 | Compare loss rank and AP50 rank within each experiment")
print("="*80)

assert "df_merged" in globals(), "请先运行 Cell 11"

rank_rows = []

for exp_name in sorted(df_merged["experiment"].dropna().unique()):
    dfe = df_merged[df_merged["experiment"] == exp_name].copy()
    dfe = dfe.dropna(subset=["best_test_loss"])

    if dfe.empty:
        continue

    for _, row in dfe.iterrows():
        rank_rows.append({
            "experiment": exp_name,
            "tag": row["tag"],
            "best_test_loss": row["best_test_loss"],
            "AP50": row["AP50"],
            "F1": row["F1"],
            "loss_rank_in_exp": row["loss_rank_in_exp"],
            "ap50_rank_in_exp": row["ap50_rank_in_exp"],
            "rank_gap_ap50_minus_loss": (
                row["ap50_rank_in_exp"] - row["loss_rank_in_exp"]
                if pd.notna(row["ap50_rank_in_exp"]) and pd.notna(row["loss_rank_in_exp"])
                else pd.NA
            )
        })

df_rank_compare = pd.DataFrame(rank_rows).sort_values(
    ["experiment", "loss_rank_in_exp"], ascending=[True, True]
).reset_index(drop=True)

out_csv = TABLE_DIR / "loss_rank_vs_ap50_rank.csv"
df_rank_compare.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_rank_compare.head(50))

print("\n✅ Cell 14 done.")
print("="*80)


# In[18]:


# Cell 15：提取每实验的 loss 冠军 / AP50 冠军
import pandas as pd

print("="*80)
print("🥇 Cell 15 | Per-experiment champions by loss and AP50")
print("="*80)

assert "df_merged" in globals(), "请先运行 Cell 11"

rows = []

for exp_name in sorted(df_merged["experiment"].dropna().unique()):
    dfe = df_merged[df_merged["experiment"] == exp_name].copy()

    # loss champion
    dfl = dfe.dropna(subset=["best_test_loss"]).sort_values("best_test_loss", ascending=True)
    loss_champ = dfl.iloc[0] if len(dfl) > 0 else None

    # ap50 champion
    dfa = dfe.dropna(subset=["AP50"]).sort_values(["AP50", "F1"], ascending=[False, False])
    ap50_champ = dfa.iloc[0] if len(dfa) > 0 else None

    rows.append({
        "experiment": exp_name,
        "loss_champion_tag": None if loss_champ is None else loss_champ["tag"],
        "loss_champion_best_test_loss": None if loss_champ is None else loss_champ["best_test_loss"],
        "loss_champion_AP50": None if loss_champ is None else loss_champ.get("AP50", None),

        "ap50_champion_tag": None if ap50_champ is None else ap50_champ["tag"],
        "ap50_champion_best_test_loss": None if ap50_champ is None else ap50_champ.get("best_test_loss", None),
        "ap50_champion_AP50": None if ap50_champ is None else ap50_champ["AP50"],
        "ap50_champion_F1": None if ap50_champ is None else ap50_champ["F1"],

        "same_champion": (
            None if (loss_champ is None or ap50_champ is None)
            else (str(loss_champ["tag"]) == str(ap50_champ["tag"]))
        )
    })

df_champions = pd.DataFrame(rows)
out_csv = TABLE_DIR / "per_experiment_champions_loss_vs_ap50.csv"
df_champions.to_csv(out_csv, index=False)

print("✅ saved:", out_csv)
display(df_champions)

print("\n✅ Cell 15 done.")
print("="*80)


# In[20]:


# Cell：取消 pandas 列内容截断，完整显示路径
import pandas as pd

print("="*80)
print("🔍 Pandas display settings | show full paths")
print("="*80)

pd.set_option("display.max_colwidth", None)   # 不截断单元格内容
pd.set_option("display.max_columns", None)    # 显示所有列
pd.set_option("display.width", 2000)          # 增大总显示宽度
pd.set_option("display.expand_frame_repr", False)  # 尽量别自动换行拆表

print("✅ Pandas display options updated.")


# In[21]:


# Cell 16：汇总所有版本的参数设置
import pandas as pd
from pathlib import Path

print("="*80)
print("🧾 Cell 16 | Export all runs parameter settings")
print("="*80)

assert "df_summary" in globals(), "请先运行前面的 loss audit Cell 3，确保 df_summary 已生成。"

PARAM_DIR = TABLE_DIR / "all_parameter_settings"
PARAM_DIR.mkdir(parents=True, exist_ok=True)

# 挑出最核心的参数列
param_cols = [
    "experiment",
    "tag",
    "run_name",
    "status",
    "diameter",
    "lr_cfg",
    "weight_decay",
    "train_batch_size",
    "n_epochs_cfg",
    "augment",
    "transformer",
    "model_dir",
    "model_dir_exists",
    "n_logged_points",
    "best_epoch_by_test_loss",
    "best_test_loss",
    "last_epoch_logged",
    "last_test_loss",
    "log_path",
]

param_cols = [c for c in param_cols if c in df_summary.columns]

df_params = df_summary[param_cols].copy()

# 重命名一下，表头更顺眼
rename_map = {
    "lr_cfg": "learning_rate",
    "n_epochs_cfg": "n_epochs",
}
df_params = df_params.rename(columns=rename_map)

# 排序
sort_cols = [c for c in ["experiment", "tag"] if c in df_params.columns]
if sort_cols:
    df_params = df_params.sort_values(sort_cols).reset_index(drop=True)

# 保存总表
all_csv = PARAM_DIR / "all_runs_parameter_settings.csv"
df_params.to_csv(all_csv, index=False)

print("✅ 全部参数总表已保存:", all_csv)
display(df_params)

# 再按实验各存一份
for exp_name in sorted(df_params["experiment"].dropna().unique()):
    dfe = df_params[df_params["experiment"] == exp_name].copy()
    out_csv = PARAM_DIR / f"{exp_name}__parameter_settings.csv"
    dfe.to_csv(out_csv, index=False)
    print("✅ saved:", out_csv)

print("\n✅ Cell 16 done.")
print("="*80)


# In[ ]:




