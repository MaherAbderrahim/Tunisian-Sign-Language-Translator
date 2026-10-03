"""Generate the experiment notebooks for alternative_models/.

Each notebook is a real, executable pipeline:
    setup -> data -> model -> train -> evaluate -> robustness -> mobile -> conclusions
Run `python make_notebooks.py` then `jupyter nbconvert --execute` each one.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent

SETUP_MD = """\
# {title}

**Objectif :** {objective}

{description}

**Protocole expérimental (commun à toutes les expériences) :**
- Dataset : 7 664 images du dataset W7CH (TSL), landmarks MediaPipe (21 points × 2 mains).
- Split stratifié 80/10/10 (seed 42) sur les échantillons ORIGINAUX, **avant**
  toute augmentation → aucune fuite de données entre train et validation/test.
- Augmentation (rotation ±20°, échelle 0.85-1.15, translation ±0.1, bruit 0.02)
  appliquée **uniquement** sur le train, équilibré à 200 échantillons/classe.
- Entraînement : Adam (lr 1e-3, wd 1e-4), CrossEntropy + label smoothing 0.1,
  ReduceLROnPlateau, early stopping sur val_acc, seeds fixes.
- 56 classes effectives sur 57 (les images `metro` ne sont pas détectées par
  MediaPipe), tête de classification à 57 sorties (comme le modèle actuel).
"""

SETUP_CODE = """\
import sys
from pathlib import Path

NB_DIR = Path.cwd()
ROOT = NB_DIR
while not (ROOT / "common").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
for p in [str(ROOT), str(ROOT / "common")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from common.paths import RESULTS_DIR, MODELS_DIR, PLOTS_DIR, REPO_ROOT
from common.data import load_dataset, normalized_inputs
from common.models import (
    count_params, set_seed, BaselineMLP, LandmarkTransformer, HandGCN, HandGAT,
)
from common.train import (
    Trainer, predict, full_metrics, save_metrics,
    plot_training_curves, plot_confusion, run_torch_benchmark,
)
from common import export as ex
from common.robustness import robustness_summary

set_seed(42)

def load_class_names():
    with open(REPO_ROOT / "data" / "processed" / "class_names.txt", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

class_names = load_class_names()
print("torch", torch.__version__)
"""

DATA_MD = """\
## 1. Chargement des données

Le dataset préparé (split + augmentation, cf. README) est chargé. Le modèle
reçoit {input_desc}.
"""

DATA_CODE = """\
dataset = load_dataset()
{data_code}
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)
print("train:", tuple(X_train.shape), "| val:", tuple(X_val.shape), "| test:", tuple(X_test.shape))
"""

MODEL_MD = """\
## 2. Architecture

{arch_desc}

Nombre de paramètres : mesuré ci-dessous.
"""

MODEL_CODE = """\
model = {model_expr}
print(f"params: {{count_params(model):,}}")
"""

TRAIN_MD = """\
## 3. Entraînement

Mêmes hyperparamètres que le protocole commun (Adam lr 1e-3, wd 1e-4,
label smoothing 0.1, early stopping patience 20, max {max_epochs} epochs).
"""

TRAIN_CODE = """\
trainer = Trainer(model, patience=20)
history = trainer.fit(train_loader, val_loader, max_epochs={max_epochs}, early_stop=True)
trainer.restore_best()

plot_training_curves(history, PLOTS_DIR / "training_curves" / "{name}_curves.png",
                     title="{title}")
print("best val acc:", trainer.best_val_acc)
"""

EVAL_MD = """\
## 4. Évaluation sur le jeu de test

Métriques complètes : accuracy, precision/recall/F1 (macro et pondérés),
matrice de confusion, performance par classe.
"""

EVAL_CODE = """\
y_pred, y_true = predict(model, test_loader)
metrics = full_metrics(y_true, y_pred, num_classes=57)
print(f"test accuracy: {{metrics['accuracy']:.4f}}")
print(f"macro F1: {{metrics['macro_f1']:.4f}} | weighted F1: {{metrics['f1']:.4f}}")
print(f"macro precision: {{metrics['macro_precision']:.4f}} | macro recall: {{metrics['macro_recall']:.4f}}")

cm = np.array(metrics["confusion_matrix"])
plot_confusion(cm, class_names, PLOTS_DIR / "confusion_matrices" / "{name}_cm.png",
               title="{title} — confusion matrix (test)")
np.save(RESULTS_DIR / "confusion_{name}.npy", cm)

import pandas as pd
per = pd.DataFrame({{
    "class": class_names,
    "precision": metrics["per_class_precision"],
    "recall": metrics["per_class_recall"],
    "f1": metrics["per_class_f1"],
    "support": metrics["per_class_support"],
}})
per.to_csv(RESULTS_DIR / "per_class_{name}.csv", index=False)
print(per.sort_values("f1").head(8))
"""

ROB_MD = """\
## 5. Robustesse

Le modèle est évalué sous perturbations des landmarks appliquées **avant**
normalisation (dans l'espace image) : bruit gaussien, rotation, translation,
changement d'échelle — 5 essais par perturbation, moyenne ± écart-type.
"""

ROB_CODE = """\
rob = robustness_summary(model, dataset["test"]["img_landmarks"],
                         dataset["test"]["labels"], base_acc=metrics["accuracy"],
                         forward_fn={forward_fn})
"""

MOBILE_MD = """\
## 6. Mesures mobiles

- Nombre de paramètres et taille du checkpoint PyTorch.
- Export ONNX (FP32), vérification, latence ONNX Runtime CPU (batch=1).
- Versions FP16 et INT8 (quantification dynamique) : taille, latence,
  dégradation d'accuracy sur le jeu de test.
"""

MOBILE_CODE = """\
sample = X_test[:1].float()
lat_torch = run_torch_benchmark(model, sample)
print(f"torch latency: {{lat_torch:.3f}} ms/sample")

onnx_path = ex.export_onnx(model, sample, "{name}")
fp16 = ex.quantize_fp16(onnx_path, "{name}")
int8 = ex.quantize_int8(onnx_path, "{name}")

quantized = {{}}
for tag, p in [("fp32", onnx_path), ("fp16", fp16), ("int8", int8)]:
    preds = ex.ort_predict(p, X_test.numpy())
    quantized[tag] = {{
        "size_mb": ex.file_size_mb(p),
        "latency_ms": ex.ort_latency(p, sample.numpy()),
        "test_acc": float((preds == y_test.numpy()).mean()),
    }}
    print(f"  {{tag}}: size={{quantized[tag]['size_mb']:.2f}} MB "
          f"latency={{quantized[tag]['latency_ms']:.3f}} ms "
          f"acc={{quantized[tag]['test_acc']:.4f}}")

extra = {{
    "robustness": rob,
    "epochs": trainer.epochs_run,
    "params": count_params(model),
    "best_val_acc": trainer.best_val_acc,
    "torch_latency_ms": lat_torch,
    "quantized": quantized,
}}
save_metrics("{name}", metrics, extra=extra)

torch_path = ex.save_torch(model, "{name}")
print("saved:", torch_path, f"({{ex.file_size_mb(torch_path):.2f}} MB)")
"""

CONC_MD = """\
## 7. Conclusions

{conclusions}
"""

# forward functions ---------------------------------------------------------
FLAT_FORWARD = """\
from common.data import wrist_normalize, flatten
def forward_fn(model, raw, batch_size=512):
    model.eval()
    feats = flatten(wrist_normalize(raw))
    outs = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            x = torch.from_numpy(feats[i:i + batch_size]).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)
"""

TOKEN_FORWARD = """\
def forward_fn(model, raw, batch_size=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(raw), batch_size):
            x = torch.from_numpy(raw[i:i + batch_size]).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)
"""

GEO_FORWARD = """\
def forward_fn(model, raw, batch_size=512):
    # Perturbed raw landmarks -> wrist-normalized + geometric features (282-D)
    model.eval()
    from common.data import wrist_normalize, flatten, geometric_features
    feats = np.concatenate([flatten(wrist_normalize(raw)),
                            geometric_features(raw).reshape(len(raw), -1)], axis=1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            x = torch.from_numpy(feats[i:i + batch_size]).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)
"""

JOINT_FORWARD = """\
def forward_fn(model, raw, batch_size=512):
    # Perturbed raw landmarks -> per-joint features (x,y,z + bone vectors)
    model.eval()
    from common.data import joint_features
    jf = joint_features(raw)
    outs = []
    with torch.no_grad():
        for i in range(0, len(jf), batch_size):
            x = torch.from_numpy(jf[i:i + batch_size]).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)
"""

EMBED_FORWARD = """\
def forward_fn(model, raw, batch_size=128):
    # Re-embed the perturbed image landmarks through the frozen embedder
    # (world landmarks and handedness stay fixed) before predicting.
    model.eval()
    world = dataset["test"]["world_landmarks"]
    hand = dataset["test"]["handedness"]
    nh = dataset["test"]["n_hands"]
    outs = []
    with torch.no_grad():
        for i in range(0, len(raw), batch_size):
            emb = embedder.embed_batch(raw[i:i + batch_size],
                                       world[i:i + batch_size],
                                       hand[i:i + batch_size],
                                       nh[i:i + batch_size])
            x = torch.from_numpy(emb).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)
"""


def make_notebook(path: Path, title: str, objective: str, description: str,
                  input_desc: str, data_code: str, model_expr: str,
                  arch_desc: str, max_epochs: int, name: str,
                  forward_code: str, conclusions: str):
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            SETUP_MD.format(title=title, objective=objective, description=description)),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_markdown_cell(DATA_MD.format(input_desc=input_desc)),
        nbf.v4.new_code_cell(DATA_CODE.format(data_code=data_code)),
        nbf.v4.new_markdown_cell(MODEL_MD.format(arch_desc=arch_desc)),
        nbf.v4.new_code_cell(MODEL_CODE.format(model_expr=model_expr)),
        nbf.v4.new_markdown_cell(TRAIN_MD.format(max_epochs=max_epochs)),
        nbf.v4.new_code_cell(TRAIN_CODE.format(max_epochs=max_epochs, name=name, title=title)),
        nbf.v4.new_markdown_cell(EVAL_MD),
        nbf.v4.new_code_cell(EVAL_CODE.format(name=name, title=title)),
        nbf.v4.new_markdown_cell(ROB_MD),
        nbf.v4.new_code_cell(forward_code + ROB_CODE.format(forward_fn="forward_fn")),
        nbf.v4.new_markdown_cell(MOBILE_MD),
        nbf.v4.new_code_cell(MOBILE_CODE.format(name=name)),
        nbf.v4.new_markdown_cell(CONC_MD.format(conclusions=conclusions)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nb["cells"] = cells
    nbf.write(nb, str(path))
    print(f"wrote {path}")


RAW_DATA = """\
X_train = torch.from_numpy(dataset["train"]["img_landmarks"]).float()
y_train = torch.from_numpy(dataset["train"]["labels"])
X_val   = torch.from_numpy(dataset["val"]["img_landmarks"]).float()
y_val   = torch.from_numpy(dataset["val"]["labels"])
X_test  = torch.from_numpy(dataset["test"]["img_landmarks"]).float()
y_test  = torch.from_numpy(dataset["test"]["labels"])
"""


if __name__ == "__main__":
    # ---------------------------------------------------------------- baseline
    make_notebook(
        path=ROOT / "00_baseline" / "baseline_mlp.ipynb",
        title="Baseline — MLP résiduel actuel",
        objective="Reproduire le modèle de production actuel (TSLClassifier) sur le "
                  "protocole propre (sans fuite train/test) pour servir de baseline.",
        description="""Le modèle actuel est un MLP résiduel profond (126 → 512 → 512 → 256 → 256 → 128 → 64 → 57)
avec BatchNorm, ReLU, Dropout 0.2, init Kaiming. Il consomme les 126 features normalisées
(2 mains × 21 landmarks × 3 coordonnées, relatives au poignet et normalisées par la taille de la main).""",
        input_desc="vecteur normalisé de 126 features (2 × 21 × 3, relatif poignet + taille main)",
        data_code="""inp = normalized_inputs(dataset, with_geo=False)
X_train, y_train = torch.from_numpy(inp["train"]["features"]).float(), torch.from_numpy(inp["train"]["labels"])
X_val, y_val     = torch.from_numpy(inp["val"]["features"]).float(),   torch.from_numpy(inp["val"]["labels"])
X_test, y_test   = torch.from_numpy(inp["test"]["features"]).float(),  torch.from_numpy(inp["test"]["labels"])""",
        model_expr="BaselineMLP(num_classes=57, input_size=126)",
        arch_desc="""MLP résiduel de production (src/models/classifier.py::TSLClassifier) :
input 126 → 512 (+ skip 512) → 256 (+ skip 256) → 128 → 64 → 57.
BatchNorm + ReLU + Dropout(0.2) à chaque bloc, label smoothing 0.1.""",
        max_epochs=120, name="baseline_mlp", forward_code=FLAT_FORWARD,
        conclusions="""Le MLP actuel sert de référence. On mesure ici sa performance réelle sur un protocole
sans fuite de données (le protocole historique appliquait l'augmentation avant le split,
ce qui pouvait faire apparaître des copies augmentées du même échantillon dans train ET test).
Toute amélioration des architectures alternatives devra être mesurée relativement à cette baseline.""",
    )

    # -------------------------------------------------------------- pretrained
    make_notebook(
        path=ROOT / "01_pretrained" / "pretrained_landmark_model.ipynb",
        title="Alternative 1 — Transfer Learning : gesture embedder MediaPipe",
        objective="Utiliser le modèle pré-entraîné MediaPipe Gesture Embedding (entraîné sur HaGRID, "
                  "~30k+ images de mains) comme extracteur de features gelé, et n'entraîner qu'une tête "
                  "de classification sur les embeddings 256-D (2 × 128-D).",
        description="""Le **gesture embedding model** de MediaPipe (Uboweja et al., 2023, arXiv:2309.10858)
est un petit modèle TFLite qui transforme les landmarks d'une main (x,y,z image + world landmarks
+ handedness) en un embedding de 128 dimensions. Il est distribué officiellement par Google
(`gesture_embedder.tflite`, licence Apache 2.0) et c'est la représentation que Google recommande
pour les tâches de reconnaissance de gestes custom (pipeline Model Maker).
Il est exécuté ici **gelé** avec le runtime LiteRT (ai-edge-litert) : seule la tête 256 → 57 est entraînée.""",
        input_desc="embeddings 256-D = concat(embedding(main 1), embedding(main 2))",
        data_code="""from common.embedder import GestureEmbedder, extract_embeddings, EmbedderHeadMLP
from common.paths import DATA_DIR
embedder = GestureEmbedder()
emb = extract_embeddings(dataset, embedder, cache_path=DATA_DIR / "embeddings.npz")
X_train = torch.from_numpy(emb["train"]["embeddings"]).float()
y_train = torch.from_numpy(emb["train"]["labels"])
X_val   = torch.from_numpy(emb["val"]["embeddings"]).float()
y_val   = torch.from_numpy(emb["val"]["labels"])
X_test  = torch.from_numpy(emb["test"]["embeddings"]).float()
y_test  = torch.from_numpy(emb["test"]["labels"])""",
        model_expr="EmbedderHeadMLP(num_classes=57, input_dim=256, hidden=128)",
        arch_desc="""Tête MLP sur embeddings gelés : 256 → 128 → 128 → 57 (BatchNorm + ReLU + Dropout 0.2).
L'embedder n'est PAS entraîné (runtime TFLite gelé) — c'est un pur transfer learning.""",
        max_epochs=80, name="pretrained_embedder", forward_code=EMBED_FORWARD,
        conclusions="""Le transfer learning via l'embedder pré-entraîné de MediaPipe teste l'hypothèse :
« une représentation apprise sur des dizaines de milliers de mains (HaGRID) est-elle plus
discriminante que nos features normalisées maison ? ». Attention : pour ce modèle, les tests de
robustesse ne perturbent PAS l'embedder (il est hors du chemin de gradient) mais bien les
landmarks d'entrée — le forward_fn recalcule les features normalisées, tandis que l'embedding
utilisé par le modèle a été calculé sur les landmarks d'origine.""",
    )

    # ------------------------------------------------------------- transformer
    make_notebook(
        path=ROOT / "02_transformer" / "transformer_landmarks.ipynb",
        title="Alternative 2 — Transformer sur landmarks",
        objective="Déterminer si l'attention (mécanisme Transformer) apporte un gain par rapport au MLP "
                  "lorsque les 42 articulations (2 mains × 21) sont traitées comme des tokens structurés.",
        description="""Chaque articulation est un **token** de dimension 3 (x, y, z), augmenté d'embeddings
appris pour la position anatomique (21 articulations) et la main (gauche/droite). Un token [CLS]
agrège la séquence à travers un encodeur Transformer (2 couches, d_model=64, 4 têtes).
Les données sont des images statiques (pas de séquence temporelle) : on ne fabrique donc
PAS de séquence temporelle artificielle — l'attention opère sur la structure spatiale de la main.""",
        input_desc="42 tokens (2 × 21 articulations) × 3 coordonnées + embeddings appris",
        data_code=RAW_DATA,
        model_expr="LandmarkTransformer(num_classes=57, d_model=64, nhead=4, num_layers=2, dim_ff=128)",
        arch_desc="""Transformer encoder (norm_first) : 42 tokens (x,y,z) → Linear(3→64) + embedding
articulation + embedding main + token [CLS] + positionnel appris → 2 couches d'attention
(4 têtes, FFN 128) → tête 64 → 57. ~80k paramètres.""",
        max_epochs=60, name="transformer", forward_code=TOKEN_FORWARD,
        conclusions="""Question centrale : l'attention entre articulations apporte-t-elle quelque chose
de plus que les connexions implicites du MLP ? Avec un petit dataset statique, un Transformer
risque de sous-performer (moins d'inductive bias que le MLP/GCN, besoin de plus de données),
mais il peut capturer des relations globales entre doigts distants. Les résultats décideront.""",
    )

    # -------------------------------------------------------------------- gcn
    make_notebook(
        path=ROOT / "03_gnn" / "gnn_landmarks.ipynb",
        title="Alternative 3 — GCN sur la topologie de la main",
        objective="Exploiter explicitement la structure anatomique de la main : les 21 landmarks sont des "
                  "nœuds d'un graphe dont les arêtes suivent les os (poignet, doigts, phalanges, paume).",
        description="""**GCN** (Kipf & Welling) : convolution de graphe avec adjacence normalisée
symétriquement D^-1/2 A D^-1/2, 2 blocs résiduels à 64 dimensions.
Les deux mains partagent les poids ; le pooling (mean + max, masqué pour la main absente)
concatène les deux mains avant la tête de classification 57 classes.""",
        input_desc="graphe de 21 nœuds × 2 mains, features par nœud : (x, y, z)",
        data_code=RAW_DATA,
        model_expr="HandGCN(num_classes=57, hidden=64, num_layers=2)",
        arch_desc="""GCN : 21 nœuds, arêtes = connexions anatomiques MediaPipe (23 arêtes + self-loop),
adjacence normalisée symétrique, 2 couches (64D) avec résidus + BatchNorm, pooling mean+max
par main (masqué), tête 128 → 57. ~45k paramètres.""",
        max_epochs=100, name="gcn", forward_code=TOKEN_FORWARD,
        conclusions="""Le GCN force le modèle à n'agréger l'information qu'à travers la structure osseuse
de la main — un a priori anatomique fort. S'il s'avère au moins aussi bon que le MLP avec
~20x moins de paramètres, c'est un candidat sérieux pour le mobile.""",
    )

    # -------------------------------------------------------------------- gat
    make_notebook(
        path=ROOT / "03_gnn" / "gat_landmarks.ipynb",
        title="Alternative 3b — GAT (Graph Attention Network)",
        objective="Comparer la variante attentionnelle du GNN : au lieu de l'adjacence normalisée fixe, "
                  "chaque nœud apprend des poids d'attention sur ses voisins anatomiques.",
        description="""Le GAT remplace l'agrégation fixe du GCN par une attention paire-à-paire
(GATv1, 4 têtes) restreinte aux mêmes arêtes anatomiques. Structure de pooling et tête
identiques au GCN pour une comparaison directe GCN vs GAT.""",
        input_desc="graphe de 21 nœuds × 2 mains, features par nœud : (x, y, z)",
        data_code=RAW_DATA,
        model_expr="HandGAT(num_classes=57, hidden=64, num_layers=2, n_heads=4)",
        arch_desc="""GAT : 2 couches d'attention (4 têtes × 16D), arêtes anatomiques uniquement,
BatchNorm + ReLU, pooling mean+max masqué par main, tête 128 → 57. ~53k paramètres.""",
        max_epochs=100, name="gat", forward_code=TOKEN_FORWARD,
        conclusions="""Le GAT apprend si certaines connexions anatomiques sont plus importantes que
d'autres pour discriminer les signes. Comparé au GCN, il ajoute de la flexibilité au prix de
légèrement plus de paramètres.""",
    )

    # ---------------------------------------------------------------- ablation
    GEO_DATA = """\
inp = normalized_inputs(dataset, with_geo=True)
X_train, y_train = torch.from_numpy(inp["train"]["features"]).float(), torch.from_numpy(inp["train"]["labels"])
X_val, y_val     = torch.from_numpy(inp["val"]["features"]).float(),   torch.from_numpy(inp["val"]["labels"])
X_test, y_test   = torch.from_numpy(inp["test"]["features"]).float(),  torch.from_numpy(inp["test"]["labels"])
"""

    make_notebook(
        path=ROOT / "04_ablation" / "mlp_geo.ipynb",
        title="Ablation — MLP + features géométriques",
        objective="Mesurer l'impact des features géométriques (vecteurs osseux, angles articulaires, "
                  "normale de paume) sur le MLP baseline.",
        description="""Entrée = 126 coordonnées normalisées + 156 features géométriques (60 vecteurs
osseux unitaires, 15 cos(angles) articulaires, 3 normale de paume × 2 mains) = 282 dimensions.
Résultat à comparer à la baseline (126 dimensions seules).""",
        input_desc="282 features = 126 coordonnées normalisées + 156 features géométriques",
        data_code=GEO_DATA,
        model_expr="BaselineMLP(num_classes=57, input_size=282)",
        arch_desc="MLP résiduel identique à la baseline (mêmes blocs, même dropout), entrée 282.",
        max_epochs=120, name="mlp_geo", forward_code=GEO_FORWARD,
        conclusions="""Comparaison directe baseline vs baseline+geo : si le gain est significatif, les
features géométriques encodent de l'information que le MLP ne peut pas extraire seul des
coordonnées (invariance d'échelle/translation des angles et vecteurs unitaires).""",
    )

    GEO_JOINT_DATA = """\
from common.data import joint_features
jt_train = joint_features(dataset["train"]["img_landmarks"])
jt_val   = joint_features(dataset["val"]["img_landmarks"])
jt_test  = joint_features(dataset["test"]["img_landmarks"])
X_train, y_train = torch.from_numpy(jt_train).float(), torch.from_numpy(dataset["train"]["labels"])
X_val, y_val     = torch.from_numpy(jt_val).float(),   torch.from_numpy(dataset["val"]["labels"])
X_test, y_test   = torch.from_numpy(jt_test).float(),  torch.from_numpy(dataset["test"]["labels"])
"""

    make_notebook(
        path=ROOT / "04_ablation" / "transformer_geo.ipynb",
        title="Ablation — Transformer + features par articulation",
        objective="Mesurer l'impact des vecteurs osseux par articulation sur le Transformer.",
        description="""Chaque token passe de 3 dims (x,y,z) à 9 dims : xyz + vecteur osseux entrant
(unité) + vecteur osseux sortant (unité). Le reste de l'architecture est identique.""",
        input_desc="42 tokens × 9 dims (xyz + os entrant + os sortant)",
        data_code=GEO_JOINT_DATA,
        model_expr="LandmarkTransformer(num_classes=57, d_model=64, nhead=4, num_layers=2, dim_ff=128, token_dim=9)",
        arch_desc="Transformer identique au notebook 02, avec tokens à 9 dimensions.",
        max_epochs=60, name="transformer_geo", forward_code=JOINT_FORWARD,
        conclusions="""Comparaison transformer vs transformer+geo : les vecteurs osseux unitaires sont
invariants à l'échelle — ils peuvent aider le Transformer qui reçoit des coordonnées absolues.""",
    )

    make_notebook(
        path=ROOT / "04_ablation" / "gcn_geo.ipynb",
        title="Ablation — GCN + features par articulation",
        objective="Mesurer l'impact des vecteurs osseux par nœud sur le GCN.",
        description="""Chaque nœud du graphe passe de 3 dims (x,y,z) à 9 dims : xyz + vecteur osseux
entrant + vecteur osseux sortant. Même topologie, mêmes hyperparamètres.""",
        input_desc="graphe de 21 nœuds × 2 mains, features par nœud : 9 dims",
        data_code=GEO_JOINT_DATA,
        model_expr="HandGCN(num_classes=57, hidden=64, num_layers=2, input_dim=9)",
        arch_desc="GCN identique au notebook 03, entrées à 9 dimensions par nœud.",
        max_epochs=100, name="gcn_geo", forward_code=JOINT_FORWARD,
        conclusions="""Comparaison gcn vs gcn+geo : le GCN agrège déjà les vecteurs via les arêtes —
les vecteurs osseux explicites peuvent être redondants (les coordonnées suffisent à les
calculer) ou au contraire simplifier l'apprentissage.""",
    )

    print("notebooks written")