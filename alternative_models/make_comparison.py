"""Generate 05_compare/comparison.ipynb: aggregates all metrics_*.json into
comparison tables + plots, and evaluates the historical production model
(best_model.pth) on the clean test split as a reference.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent

CELLS = [
    nbf.v4.new_markdown_cell("""\
# Comparaison finale

Ce notebook agrège les métriques sauvegardées par les 8 notebooks
d'expérience (`results/metrics_*.json`) et produit :
- `results/comparison.csv` — précision, taille, latence de chaque modèle ;
- `results/results_summary.csv` — robustesse (baisse de précision moyenne) ;
- les graphiques comparatifs dans `plots/comparisons/`.

Il évalue aussi le **modèle de production actuel** (`models/best_model.pth`,
entraîné avec l'ancien protocole augmenter-puis-split) sur le jeu de test
propre, à titre de référence historique — non comparable directement
aux nouvelles expériences (fuite de données possible dans l'ancien protocole).
"""),
    nbf.v4.new_code_cell("""\
import sys
from pathlib import Path

NB_DIR = Path.cwd()
ROOT = NB_DIR.parent
for p in [str(ROOT), str(ROOT / "common")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import numpy as np
import pandas as pd

from common.paths import RESULTS_DIR, PLOTS_DIR, MODELS_DIR, REPO_ROOT
from common.data import load_dataset, normalized_inputs
from common.train import full_metrics
"""),
    nbf.v4.new_markdown_cell("""\
## 1. Agrégation des métriques"""),
    nbf.v4.new_code_cell("""\
rows = []
rob_rows = []
for f in sorted(RESULTS_DIR.glob("metrics_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    m, ex = d["metrics"], d.get("extra", {})
    q = ex.get("quantized", {})
    name = Path(d["name"]).stem if "name" in d else f.stem.replace("metrics_", "")
    rows.append({
        "model": name,
        "params": ex.get("params"),
        "test_acc": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "best_val_acc": ex.get("best_val_acc"),
        "epochs": ex.get("epochs"),
        "torch_latency_ms": ex.get("torch_latency_ms"),
        "onnx_fp32_size_mb": q.get("fp32", {}).get("size_mb"),
        "onnx_fp32_latency_ms": q.get("fp32", {}).get("latency_ms"),
        "onnx_fp32_acc": q.get("fp32", {}).get("test_acc"),
        "onnx_fp16_size_mb": q.get("fp16", {}).get("size_mb"),
        "onnx_int8_size_mb": q.get("int8", {}).get("size_mb"),
        "onnx_int8_latency_ms": q.get("int8", {}).get("latency_ms"),
        "onnx_int8_acc": q.get("int8", {}).get("test_acc"),
    })
    rob = ex.get("robustness", {})
    base = rob.get("base_accuracy", m["accuracy"])
    drops = [v["drop"] for k, v in rob.items() if isinstance(v, dict) and "drop" in v]
    rob_rows.append({
        "model": name,
        "base_acc": base,
        "mean_drop": float(np.mean(drops)) if drops else None,
        "max_drop": float(np.max(drops)) if drops else None,
        "noise_0.05_acc": rob.get("noise_0.05", {}).get("accuracy"),
        "rotation_20_acc": rob.get("rotation_20", {}).get("accuracy"),
        "translation_0.10_acc": rob.get("translation_0.10", {}).get("accuracy"),
        "scale_1.10_acc": rob.get("scale_1.10", {}).get("accuracy"),
    })

df = pd.DataFrame(rows).sort_values("test_acc", ascending=False).reset_index(drop=True)
rdf = pd.DataFrame(rob_rows).sort_values("mean_drop").reset_index(drop=True)
df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
rdf.to_csv(RESULTS_DIR / "results_summary.csv", index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(df.to_string(index=False))
"""),
    nbf.v4.new_markdown_cell("""\
## 2. Évaluation du modèle de production actuel (référence historique)"""),
    nbf.v4.new_code_cell("""\
legacy_path = REPO_ROOT / "models" / "best_model.pth"
if legacy_path.exists():
    import torch
    from src.models.classifier import TSLClassifier

    ds = load_dataset()
    inp = normalized_inputs(ds, with_geo=False)
    X_test = torch.from_numpy(inp["test"]["features"]).float()
    y_test = torch.from_numpy(inp["test"]["labels"])

    model = TSLClassifier()
    state = torch.load(legacy_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
    lm = full_metrics(y_test.numpy(), logits.argmax(1).numpy(), num_classes=57)
    print(f"legacy best_model.pth on clean test: acc={lm['accuracy']:.4f} "
          f"macro_f1={lm['macro_f1']:.4f}")
    legacy = pd.DataFrame([{
        "model": "legacy_best_model", "params": sum(p.numel() for p in model.parameters()),
        "test_acc": lm["accuracy"], "macro_f1": lm["macro_f1"],
    }])
    legacy.to_csv(RESULTS_DIR / "legacy_reference.csv", index=False)
else:
    print("models/best_model.pth not found — skipped")
"""),
    nbf.v4.new_markdown_cell("""\
## 3. Graphiques comparatifs"""),
    nbf.v4.new_code_cell("""\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMP = PLOTS_DIR / "comparisons"
COMP.mkdir(parents=True, exist_ok=True)

df2 = df.copy()
df2["params_k"] = df2["params"] / 1000

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df2["params_k"], df2["test_acc"], s=90)
for _, r in df2.iterrows():
    ax.annotate(r["model"], (r["params_k"], r["test_acc"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("params (k, log scale)")
ax.set_ylabel("test accuracy")
ax.set_title("Accuracy vs size")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(COMP / "acc_vs_params.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df2["torch_latency_ms"], df2["test_acc"], s=90)
for _, r in df2.iterrows():
    ax.annotate(r["model"], (r["torch_latency_ms"], r["test_acc"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("torch latency (ms)")
ax.set_ylabel("test accuracy")
ax.set_title("Accuracy vs latency")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(COMP / "acc_vs_latency.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(rdf))
ax.barh(y, rdf["mean_drop"], color="steelblue")
ax.set_yticks(y, rdf["model"])
ax.set_xlabel("mean accuracy drop under perturbations")
ax.set_title("Robustness (lower is better)")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(COMP / "robustness_drop.png", dpi=150)

fig, ax = plt.subplots(figsize=(10, 5))
labels = ["noise .05", "rotation 20", "translation .10", "scale 1.10"]
x = np.arange(len(labels))
w = 0.13
for i, r in rdf.iterrows():
    vals = [r["noise_0.05_acc"], r["rotation_20_acc"],
            r["translation_0.10_acc"], r["scale_1.10_acc"]]
    ax.bar(x + i * w, vals, w, label=r["model"])
ax.set_xticks(x + w * len(rdf) / 2, labels)
ax.set_ylabel("accuracy under perturbation")
ax.legend(fontsize=7)
ax.set_title("Accuracy under strong perturbations")
fig.tight_layout()
fig.savefig(COMP / "perturbation_bars.png", dpi=150)
print("plots saved in", COMP)
"""),
    nbf.v4.new_markdown_cell("""\
## 4. Conclusions

Voir `README.md` pour l'analyse complète et la recommandation finale.""")
]

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
path = ROOT / "05_compare" / "comparison.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(path))
print("wrote", path)