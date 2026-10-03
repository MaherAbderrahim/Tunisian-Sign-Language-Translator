# Alternative Models — Tunisian Sign Language (TSL) Recognition

Comparaison expérimentale de 4 familles d'architectures pour la reconnaissance
de 57 signes de la langue des signes tunisienne à partir des **landmarks
MediaPipe** (21 points × 2 mains, pas de pixels RGB), avec un objectif
**mobile** : précision, robustesse, taille et latence mesurées réellement.

Tout est exécuté dans les notebooks de ce dossier (résultats reproductibles).

---

## A. Objectif

Le pipeline actuel du projet (`src/`) reconnaît les signes avec un MLP résiduel
(`TSLClassifier`, ~903 k paramètres, `models/best_model.pth`). Ce dossier
répond à trois questions :

1. **Une architecture alternative fait-elle mieux** sur un protocole
   d'évaluation propre ?
2. **Le modèle actuel tient-il ses ~99 % annoncés** lorsqu'on élimine la fuite
   de données du protocole historique ?
3. **Quel modèle recommander pour le mobile** (taille, latence, quantification
   ONNX FP16/INT8) ?

## B. Données et protocole (important)

- 7 664 images du dataset W7CH (TSL) ; landmarks extraits avec MediaPipe
  (Landmarker 21 points, image-normalisés **et** world en mètres + handedness).
  56 classes sur 57 sont couvertes (les images `metro` ne sont pas détectées).
- **Split stratifié 80/10/10 (seed 42) avant toute augmentation** :
  les copies augmentées d'une image d'entraînement ne peuvent jamais fuiter
  dans validation/test (le protocole historique augmentait avant le split —
  fuite possible, cf. section G).
- **Augmentation train uniquement** : rotation ±20°, échelle 0.85–1.15,
  translation ±0.1, bruit 0.02 (espace image, world landmarks aussi),
  équilibrage à 200 échantillons/classe → train 13 282 / val 767 / test 767.
- **Les mêmes échantillons alimentent tous les modèles** (comparaison juste).
- Entraînement : Adam (lr 1e-3, wd 1e-4), CrossEntropy + label smoothing 0.1,
  ReduceLROnPlateau, early stopping sur val_acc, seeds fixes.

## C. Architectures comparées

| Famille | Modèle | Entrée | Params |
|---|---|---|---|
| **Baseline** | MLP résiduel (production) | 126 features normalisées (poignet + taille main) | 903 097 |
| **Alt 1 — Transfer** | Gesture embedder MediaPipe (gelé, HaGRID) + tête MLP | 256-D (2 × 128-D embeddings TFLite) | 57 273 |
| **Alt 2 — Transformer** | Encoder 2 couches, d=64, 4 têtes, token [CLS] | 42 tokens × 3 (x,y,z) | 79 481 |
| **Alt 3 — GNN** | GCN (adjacence anatomique normalisée) | graphe 21 nœuds × 2 mains | 45 049 |
| **Alt 3b — GNN** | GAT (attention sur arêtes anatomiques) | idem | 53 497 |
| **Ablations** | MLP/Transformer/GCN + features géométriques | + 156 geo / + 6 par articulation | — |

## D. Résultats — précision sur le test propre (767 échantillons)

| Modèle | Test acc | Macro F1 | Epochs | Early stop |
|---|---|---|---|---|
| **baseline_mlp** (production) | **0.9817** | 0.9802 | 120 | non |
| mlp_geo | 0.9804 | 0.9720 | 109 | oui |
| **gcn_geo** | 0.9687 | 0.9680 | 96 | oui |
| pretrained_embedder | 0.9335 | 0.9275 | 80 | non |
| transformer_geo | 0.9309 | 0.9320 | 60 | non |
| transformer | 0.9205 | 0.9196 | 60 | non |
| gcn | 0.7392 | 0.7166 | 53 | oui |
| gat | 0.6858 | 0.6666 | 35 | oui |

**Lectures :**
- Le **MLP actuel reste le plus précis** sur un protocole propre (98.2 %).
- **Aucune alternative ne bat la baseline** en précision pure.
- Les **features géométriques transforment le GCN** (+23 pts : 0.739 → 0.969)
  et aident légèrement le Transformer (+1 pt), mais ne changent rien au MLP :
  les coordonnées normalisées suffisent au MLP, pas aux graphes (qui reçoivent
  des coordonnées absolues).
- Le GAT sous-performe (attention sur graphe trop petite / instable).

## E. Robustesse (perturbations des landmarks, espace image)

Baisse de précision moyenne sur 10 perturbations × 5 essais (bruit 0.01/0.02/0.05,
rotation 5/10/20°, translation 0.05/0.10, échelle 0.90/1.10) :

| Modèle | Acc base | Drop moyen | Drop max | Acc sous rotation 20° |
|---|---|---|---|---|
| **transformer** | 0.9205 | **0.038** | 0.164 | 0.836 |
| transformer_geo | 0.9309 | 0.041 | 0.306 | 0.843 |
| **pretrained_embedder** | 0.9335 | 0.079 | 0.433 | 0.764 |
| gat | 0.6858 | 0.084 | 0.281 | 0.571 |
| baseline_mlp | 0.9817 | 0.095 | 0.420 | 0.857 |
| gcn_geo | 0.9687 | 0.098 | 0.486 | 0.842 |
| gcn | 0.7392 | 0.106 | 0.330 | 0.587 |
| mlp_geo | 0.9804 | 0.113 | 0.564 | 0.849 |

**Lectures :** le Transformer est le plus **résilient aux perturbations**
(attention globale = moins sensible aux déplacements locaux), l'embedder
pré-entraîné est **invariant à translation/échelle** (drop 0.000) mais fragile
au bruit ; le bruit est le pire ennemi de tous les modèles (0.05 → −30 à −50 pts).

## F. Mobile (mesures réelles, ONNX Runtime CPU, batch=1)

| Modèle | Params | ONNX fp32 | ONNX fp16 (poids) | ONNX int8 (dyn.) | Latence torch | Latence ORT int8 | Acc int8 |
|---|---|---|---|---|---|---|---|
| baseline_mlp | 903 k | 3.47 MB | 1.74 MB | 0.92 MB | 0.30 ms | 0.09 ms | **0.9831** |
| mlp_geo | 983 k | 3.77 MB | 1.89 MB | 0.99 MB | 0.32 ms | 0.08 ms | 0.9804 |
| **gcn_geo** | 45 k | 0.18 MB | 0.10 MB | **0.06 MB** | 0.27 ms | 0.06 ms | 0.9687 |
| pretrained_embedder | 57 k | 0.22 MB | 0.11 MB | 0.06 MB | **0.07 ms** | 0.02 ms | 0.9361 |
| transformer_geo | 80 k | 0.34 MB | 0.18 MB | 0.15 MB | 0.54 ms | 0.18 ms | 0.9335 |
| transformer | 79 k | 0.34 MB | 0.19 MB | 0.15 MB | 0.63 ms | 0.18 ms | 0.9179 |

Note : la conversion **FP16 est effectuée au niveau des poids uniquement**
(calcul en FP32) — la conversion plein-graphe d'onnxconverter-common produit
des graphes invalides pour GCN/GAT. La quantification INT8 dynamique
(onnxruntime) conserve la précision (Δ ≤ 0.013).

**TFLite** : l'export TFLite de nos modèles nécessite TensorFlow (non installé
sur cette machine — la conversion était bloquée après 15 min de timeout).
Chemin documenté : ONNX → `onnx2tf` ou `tensorflow.lite.TFLiteConverter` ;
`ai-edge-litert` est déjà installé et fait tourner le `gesture_embedder.tflite`
(latence mesurée : 0.019 ms/main) sur le pipeline Alt 1.

## G. Le modèle de production actuel, évalué proprement

Le `best_model.pth` (entraîné avec l'ancien protocole *augmenter-puis-split*,
~99 % annoncés) obtient **0.5919 acc / 0.5165 macro-F1 sur le test propre**
(`05_compare/comparison.ipynb`). La chute confirme la **fuite de données** de
l'ancien protocole : des copies augmentées du même échantillon se retrouvaient
dans le train ET le test. La baseline retrainée proprement (98.2 %) valide
l'architecture.

## H. Reproduction

```bash
# 1. Préparation des données (split propre + augmentation)
python -c "from common.data import load_dataset; load_dataset(force_prepare=True)"

# 2. Génération et exécution des notebooks
python make_notebooks.py
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  00_baseline/baseline_mlp.ipynb \
  01_pretrained/pretrained_landmark_model.ipynb \
  02_transformer/transformer_landmarks.ipynb \
  03_gnn/gnn_landmarks.ipynb 03_gnn/gat_landmarks.ipynb \
  04_ablation/mlp_geo.ipynb 04_ablation/transformer_geo.ipynb \
  04_ablation/gcn_geo.ipynb

# 3. Comparaison
python make_comparison.py
jupyter nbconvert --to notebook --execute --inplace 05_compare/comparison.ipynb
```

Sorties : `results/comparison.csv`, `results/results_summary.csv`,
`results/metrics_*.json`, `models/*.onnx` (fp32/fp16/int8), `plots/`.

## I. Conclusion et recommandation

| Critère | Meilleur modèle |
|---|---|
| Précision | **baseline_mlp** (0.9817) |
| Précision/taille (ratio) | **gcn_geo** (0.9687 pour 45 k params, 0.06 MB int8) |
| Robustesse | **transformer** (drop moyen 0.038) |
| Latence | **pretrained_embedder** (0.07 ms torch, 0.02 ms ORT) |
| Léger + précis mobile | **gcn_geo** (0.06 MB, 0.9687, acc int8 inchangée) |

**Recommandation finale :**
1. **Si l'on garde le pipeline actuel** (126 features + MLP) : le retrainer sur
   le protocole propre suffit — 98.2 % réels, et la version INT8 (0.92 MB)
   ne dégrade rien. C'est le choix le plus sûr pour l'application existante.
2. **Si le mobile domine** : **GCN + features géométriques** (282-D → graphe
   21 nœuds) est le meilleur compromis — 20× moins de paramètres que le MLP,
   0.06 MB en INT8, 0.9687 de précision, latence 0.06 ms, et l'export ONNX
   (donc TFLite via onnx2tf) est trivial (ops simples, pas de fusion native).
3. **L'embedder MediaPipe** (transfer learning) est attractif pour sa vitesse
   et son invariance translation/échelle, mais 4.8 pts en dessous de la
   baseline — à réserver aux cas où l'on préfère un modèle 100 % officiel
   Google sans entraînement d'extracteur.
4. Le **Transformer** n'apporte pas de gain en précision sur ce dataset
   statique ; il est néanmoins le plus robuste aux perturbations — piste
   intéressante si l'application rencontre du bruit de caméra.