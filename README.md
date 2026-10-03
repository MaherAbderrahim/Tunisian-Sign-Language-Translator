# 🇹🇳 Tunisian Sign Language Translator (TSL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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

**3 façons de lancer l'application :**

| | Commande | Ce que ça lance |
|---|---|---|
| **1. `start.py` (recommandé)** | `python start.py` | API (:8000) **+** interface web statique (:3000) — 2 services en une commande |
| 2. Streamlit | `python -m streamlit run web/app.py` | Interface Streamlit (:8501) |
| 3. API seule | `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` | API REST (:8000) |

L'interface statique (**http://localhost:3000**) inclut la **caméra temps réel** (via WebSocket) et le **mode orthographes (finger spelling)**.

### 1. L'interface web Streamlit

```bash
python -m streamlit run web/app.py
```

Ouvre **http://localhost:8501** dans ton navigateur. 5 onglets :

| Onglet | Que fait-il ? |
|---|---|
| **Drop Image** | Traduire une image de signe |
| **Upload Video** | Reconnaître les signes en continu dans une vidéo |
| **Camera** | Reconnaître un signe sur une **photo** prise avec la webcam (pas de flux en direct) |
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
python demo.py                          # petite démo
python -m pytest tests/ -v              # tous les tests (124 ; pytest requis : pip install pytest)
```

---

## 🔌 Principales routes de l'API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Informations sur l'API |
| `GET` | `/health` | État de l'API |
| `POST` | `/translate/frame` | Reconnaître un signe dans une image (upload) |
| `POST` | `/translate/text` | Traduire un mot TSL saisi |
| `POST` | `/translate/multi` | Texte → plusieurs langues (hors ligne) |
| `GET` | `/translate/languages` | Langues disponibles |
| `POST` | `/translate/video` | Reconnaissance continue sur une vidéo |
| `GET` | `/dictionary/labels` | Tous les signes TSL |
| `GET` | `/dictionary/words` | Tous les mots tunisiens |
| `GET` | `/dictionary/categories` | Catégories de signes |
| `GET` | `/dictionary/{label}` | Traduction d'un signe (ex. `/dictionary/labes`) |
| `GET` / `DELETE` | `/history` | Historique des traductions |
| `GET` | `/alphabet/labels` | Labels de l'alphabet (finger spelling) |
| `GET` | `/alphabet/{label}` | Traduction d'une lettre |
| `GET` / `POST` | `/llm/config` | Consulter / configurer le correcteur LLM |
| `POST` | `/llm/correct` | Corriger un texte avec le LLM |
| `GET` / `POST` | `/settings/inference` | Seuil de confiance / fenêtre de lissage |
| `GET` | `/mode` | Mode actuel (sign / spell) |
| `POST` | `/mode/toggle` | Basculer signe ↔ orthographe |
| `POST` | `/spell/flush` | Valider la phrase orthographiée |
| `WS` | `/ws/camera` | Caméra temps réel (frames base64 → traductions) |

> Les routes d'état (`/history`, `/mode`, `/translate/frame`, …) acceptent
> l'en-tête optionnel `X-Session-Id` pour isoler l'historique et le mode par
> client. Chaque WebSocket reçoit automatiquement sa propre session.

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
├── api/                 → serveur REST (FastAPI)
├── web/
│   ├── app.py           → interface Streamlit (:8501)
│   ├── index.html       → interface statique (:3000)
│   ├── app.js           → logique de l'interface statique (caméra, WebSocket)
│   └── style.css        → styles de l'interface statique
├── src/
│   ├── config/          → paramètres (settings.py)
│   ├── models/          → détection des mains, réseau de neurones, entraînement
│   ├── languages/       → dictionnaire TSL → tunisien / français / anglais / arabe
│   └── translation/     → le traducteur principal
├── models/              → modèles pré-entraînés (best_model.pth, MediaPipe)
├── data/                → données (dataset W7CH, landmarks)
├── scripts/             → téléchargement et préparation des données
├── tests/               → tests automatisés (124 tests)
├── manual_tests/        → scénarios et résultats des tests manuels
├── alternative_models/  → modèles alternatifs et comparaison (MLP, Transformer, GNN)
├── test_images/         → images d'exemple pour tester
├── docs/                → documentation du projet
├── notebooks/           → notebooks d'exploration de données
├── start.py             → lance API (:8000) + interface statique (:3000)
├── setup.py             → packaging du projet
├── demo.py              → démo en ligne de commande
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

Lance **deux conteneurs** (deux services) : `api` (API FastAPI, port 8000) et `web` (interface Streamlit, port 8501).

---

## ⚙️ Variables d'environnement

Copie `.env.example` vers `.env`, adapte les valeurs, puis redémarre :

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux / Mac
```

| Variable | Rôle | Défaut |
|---|---|---|
| `LLM_PROVIDER` | Fournisseur LLM pour la correction | `openrouter` |
| `LLM_API_KEY` | Clé API du LLM (optionnelle) | *(vide)* |
| `LLM_MODEL` | Modèle LLM utilisé | `meta-llama/llama-3.1-8b-instruct:free` |
| `LLM_BASE_URL` | URL de base alternative du fournisseur | *(vide)* |
| `LLM_TEMPERATURE` | Température de génération | `0.3` |
| `LLM_MAX_TOKENS` | Nombre maximal de tokens générés | `256` |
| `LLM_TIMEOUT` | Délai maximal d'appel LLM (secondes) | `10` |
| `LLM_ALLOW_CUSTOM_BASE_URL` | Autorise les endpoints LLM arbitraires (garde anti-SSRF) | `false` |
| `LLM_ALLOWED_HOSTS` | Hôtes LLM autorisés (séparés par virgules, alternative au flag) | *(vide)* |
| `TSL_ADMIN_TOKEN` | Si défini, `POST /llm/config` exige l'en-tête `X-Admin-Token` | *(vide)* |
| `TSL_HOST` | Hôte du serveur API | `0.0.0.0` |
| `TSL_PORT` | Port du serveur API | `8000` |
| `TSL_CORS_ORIGINS` | Origines CORS autorisées (séparées par virgules) | `http://localhost:3000,http://localhost:8501` |
| `WEB_PORT` | Port de l'interface web statique | `3000` |

---

## ❓ FAQ

**Le modèle ne se charge pas / `best_model.pth` introuvable ?**
Certains poids sont versionnés dans le dépôt, d'autres non :

- **Versionnés** (présents dès le `git clone`) : `models/best_model.pth` (modèle 57 signes, ~10 Mo), `models/continuous/synthetic_*.pth` et `models/hand_landmarker.task`.
- **Non versionnés** (à régénérer) : les modèles d'alphabet `models/alphabet/*.pth` (finger spelling) et tous les jeux de données `.npy`. Régénère-les avec `python scripts/setup_all.py` ou les scripts d'entraînement (`python -m src.models.train`, `python scripts/train_alphabet_models.py`).

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