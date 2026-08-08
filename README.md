# 🇹🇳 Tunisian Sign Language Translator (TSL)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Un système qui traduit la langue des signes tunisienne (TSL) en texte.**
Montre un signe à la caméra → il est reconnu → traduit en **tunisien, français, anglais et arabe** — 100 % hors ligne.

## ✨ Comment ça marche ? (en 30 secondes)

```
Caméra / Image / Vidéo
        │
        ▼
MediaPipe détecte les mains (21 points par main)
        │
        ▼
Modèle de deep learning reconnaît le signe (57 signes TSL)
        │
        ▼
Dictionnaire + traduction → tunisien, français, anglais, arabe
        │
        ▼
Résultat à l'écran / via l'API
```

---

## 🚀 Démarrage rapide

```bash
git clone https://github.com/MaherAbderrahim/Tunisian-Sign-Language-Translator.git
cd Tunisian-Sign-Language-Translator

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac

pip install -r requirements.txt
```

---

## 🖥️ Comment l'utiliser ?

### 1. L'interface web (la plus simple) → recommandée

```bash
python -m streamlit run web/app.py
```

Ouvre **http://localhost:8501** dans ton navigateur. 5 onglets :

| Onglet | Que fait-il ? |
|---|---|
| **Drop Image** | Traduire une image de signe |
| **Upload Video** | Reconnaître les signes en continu dans une vidéo |
| **Camera** | Traduction en direct depuis la webcam |
| **Text Translator** | Traduire du texte en plusieurs langues |
| **Dictionary** | Consulter le dictionnaire des signes |

### 2. L'API REST (pour les développeurs)

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Documentation interactive : **http://localhost:8000/docs**
- Vérifier que tout fonctionne : **http://localhost:8000/health** → `{"status":"healthy"}`

### 3. La ligne de commande

```bash
python demo.py                      # petite démo
python -m pytest tests/test_all.py -v   # exécute tous les tests
```

---

## 🔌 Principales routes de l'API

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/translate/frame` | Reconnaître un signe dans une image (upload) |
| `POST` | `/translate/text` | Traduire un mot TSL saisi |
| `POST` | `/translate/video` | Reconnaissance continue sur une vidéo |
| `POST` | `/translate/multi` | Texte → plusieurs langues (hors ligne) |
| `GET` | `/translate/languages` | Langues disponibles |
| `GET` | `/dictionary/labels` | Tous les signes TSL |
| `GET` | `/dictionary/words` | Tous les mots tunisiens |
| `GET` | `/dictionary/categories` | Catégories de signes |
| `GET` | `/dictionary/{label}` | Traduction d'un signe (ex. `/dictionary/labes`) |
| `GET` / `DELETE` | `/history` | Historique des traductions |

### Exemple : reconnaître un signe sur une image

```bash
curl -X POST http://localhost:8000/translate/frame -F "file=@test_images/labes.jpg"
```

Réponse :
```json
{
  "tsl": "labes",
  "tunisian": "labes",
  "french": "bien",
  "english": "well",
  "arabic": "مخيبش",
  "arabic_msa": "بخير",
  "confidence": 0.93
}
```

---

## 🗂️ Structure du projet

```
Tunisian-Sign-Language-Translator/
├── api/              → serveur REST (FastAPI)
├── web/              → interface web (Streamlit)
├── src/
│   ├── models/       → détection des mains, réseau de neurones, entraînement
│   ├── languages/    → dictionnaire TSL → tunisien / français / anglais / arabe
│   └── translation/  → le traducteur principal
├── models/           → modèles pré-entraînés (best_model.pth, MediaPipe)
├── scripts/          → téléchargement et préparation des données
├── tests/            → tests automatisés
├── test_images/      → images d'exemple pour tester
├── docs/             → documentation du projet
├── notebooks/        → notebooks d'exploration de données
├── demo.py           → démo en ligne de commande
└── requirements.txt
```

---

## 🎓 Entraîner le modèle from scratch

```bash
python scripts/download_data.py    # télécharge le dataset TSL
python scripts/prepare_data.py     # prépare les données
python -m src.models.train         # entraîne le modèle
```

**Architecture :** le réseau prend les 126 points de repère des mains (21 points × 2 mains), passe par 5 couches de neurones et sort le signe parmi 57 possibles.

| Metrix | Valeur |
|---|---|
| Exactitude (top-1) | ~95 % |
| Latence | < 100 ms |
| FPS | > 15 |

**Dataset utilisé :** [W7CH/First-ever-Tunisian-Sign-Language-Dataset](https://github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset) — 4 423 images, 57 signes, 7 personnes (licence CC0).

---

## 🐳 Docker (optionnel)

```bash
docker compose up
```

Construit l'API + l'interface web dans un seul conteneur.

---

## ❓ FAQ

**Le modèle ne se charge pas / `best_model.pth` introuvable ?**
Les poids (`.pth`) dépassent la limite de GitHub et ne sont pas versionnés. Relance l'entraînement pour les régénérer.

**Ça marche sans internet ?**
Oui, la traduction utilise Argos Translate, 100 % locale. Seul le premier téléchargement des paquets de langues nécessite le réseau.

**Quels signes sont supportés ?**
57 signes en 5 catégories : *Demandes* (24), *Famille* (12), *Destinations* (8), *Jours* (7), *Transport* (6). La liste complète est dans `/dictionary/labels`.

---

## 📜 Licence

MIT — libre d'utilisation. Voir [LICENSE](LICENSE).

## 🙏 Remerciements

- [W7CH/First-ever-Tunisian-Sign-Language-Dataset](https://github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset) pour le dataset TSL
- [MediaPipe](https://mediapipe.dev/) pour la détection des mains
- [Argos Translate](https://www.argosopentech.com/) pour la traduction hors ligne