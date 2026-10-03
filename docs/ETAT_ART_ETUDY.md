# État de l'Art et Étude Comparative - Traducteur TSL vers Dialecte Tunisien

## 1. Analyse des Ressources Existantes

### 1.1 Dataset Tunisien (W7CH/First-ever-Tunisian-Sign-Language-Dataset)

| Caractéristique | Détail |
|-----------------|--------|
| **Auteurs** | Wassim Chakroun, Rihab Jerbi (SUP'COM) |
| **Date** | Janvier 2023 |
| **Taille** | 4423 images |
| **Signes** | 57 signes standard |
| **Personnes** | 7 performers |
| **Résolution** | 224×224 pixels |
| **Licence** | CC0-1.0 (domaine public) |

**Catégories:**
- Demandes: 24 signes (1400 images)
- Destinations: 8 signes (608 images)
- Famille: 12 signes (1748 images)
- Jours: 7 signes (365 images)
- Transport: 6 signes (302 images)

**Points forts:**
- Premier dataset officiel de TSL
- Nommage en dialecte tunisien
- Collaboration avec l'ATILS
- Licence permissive

**Limitations:**
- Taille réduite (4423 images pour 57 classes)
- Seulement des images statiques (pas de vidéo)
- Variabilité limitée (7 personnes)

---

### 1.2 Dataset ArASL/Kaggle (zArASL_Database_54K)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | ArASL / zArASL Database 54K |
| **Taille** | 54,049 images |
| **Signes** | 32 alphabets/lettres arabes |
| **Performers** | 40+ personnes |
| **Résolution** | 64×64 pixels |
| **Licence** | Research use |

**Points forts:**
- Grand volume de données
- Diversité des performers
- Dataset de référence pour ArSL
- Disponible sur HuggingFace

**Limitations:**
- Seulement l'alphabet (32 signes)
- Pas de mots ou phrases
- Images en niveaux de gris
- Résolution faible (64×64)

---

### 1.3 Framework sign-language-translator

| Caractéristique | Détail |
|-----------------|--------|
| **URL** | github.com/sign-language-translator/sign-language-translator |
| **Stars** | 360+ |
| **Langages supportés** | Anglais, Ourdou, Hindi |
| **Langue des signes** | Pakistan Sign Language (PK-SL) |
| **Signes** | 776 signes |
| **Architecture** | Modulaire, extensible |

**Fonctionnalités:**
- Texte vers Signes (ConcatenativeSynthesis)
- Extraction de landmarks (MediaPipe)
- Language models (N-gram, Transformer)
- CLI et API Python

**Support Arabe: ❌ NON**
- Seuls Urdu, Anglais, Hindi sont supportés
- Aucun support pour l'arabe ou le tunisien

---

## 2. État de l'Art - Reconnaissance ArSL

### 2.1 Modèles Performants

| Référence | Architecture | Dataset | Accuracy |
|-----------|--------------|---------|----------|
| Balat et al. (2025) | EfficientNet-B2 | ArSL2018 | 99.48% |
| Mansour et al. (2025) | ResNet50ViT | ArSL2018 | 99.86% |
| Alnabih et al. (2024) | Vision Transformer | ArSL2018 | 99.30% |
| YOLO-Based (2024) | YOLO + Attention | ArSL | 99.00% |

### 2.2 Technologies Clés

1. **MediaPipe Hands** - Extraction de landmarks (21 points par main)
2. **CNN + LSTM** - Features spatiales + temporelles
3. **Vision Transformers** - Modélisation globale
4. **YOLO** - Détection temps réel

---

## 3. Justification des Choix Techniques

### 3.1 Choix du Framework: sign-language-translator

**Pourquoi ce framework?**

1. **Architecture modulaire** - Extensible à toute langue
2. **Code mature** - 360+ stars, documentation complète
3. **MediaPipe intégré** - Extraction de landmarks optimisée
4. **Pipeline complet** - Du texte à la vidéo et inversement
5. **Licence Apache** - Utilisation commerciale possible

**Ce qu'il faut ajouter:**
- Classe `TunisianDialect` (TextLanguage)
- Classe `TunisianSignLanguage` (SignLanguage)
- Dictionnaire de mapping TSL

### 3.2 Choix du Dataset Principal: TSL Dataset (W7CH)

**Pourquoi ce dataset?**

1. **Spécifique à la Tunisie** - 57 signes TunSL
2. **Noms en dialecte tunisien** - Mapping direct
3. **Licence CC0** - Aucune restriction
4. **Format compatible** - Images 224×224

**Stratégie d'augmentation:**
- Utiliser ArASL pour le transfer learning
- Data augmentation (rotation, flip, brightness)
- Collecte complémentaire si nécessaire

### 3.3 Pipeline Technique

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CAMÉRA ──► MEDIAPIPE ──► LANDMARKS ──► CLASSIFIEUR         │
│     │           │             │              │               │
│     │      Détection     Extraction    CNN/MLP             │
│     │      mains        21 points      │                   │
│     │                                  │                   │
│     │                          ┌───────▼───────┐           │
│     │                          │   TSL LABEL   │           │
│     │                          └───────┬───────┘           │
│     │                                  │                   │
│     │                          ┌───────▼───────┐           │
│     │                          │   DIALECTE    │           │
│     │                          │   TUNISIEN    │           │
│     │                          └───────┬───────┘           │
│     │                                  │                   │
│     │                          ┌───────▼───────┐           │
│     │                          │   FRANÇAIS    │           │
│     │                          │   ANGLAIS     │           │
│     │                          │   ARABE       │           │
│     │                          └───────────────┘           │
│     │                                                      │
│  INTERFACE (Web/Mobile)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Architecture Proposée

### 4.1 Structure du Projet

> **Proposition (non implémentée)** — arborescence initialement prévue lors de l'étude.
> Elle ne correspond pas au dépôt réel : voir la section « Structure du projet » du `README.md`.

```
tsl-translator/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── setup.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── raw/                    # Données brutes
│   │   ├── tsl_dataset/        # Dataset W7CH
│   │   └── arasl/              # Dataset ArASL (transfer learning)
│   ├── processed/              # Données prétraitées
│   └── augmented/              # Données augmentées
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                   # Pipeline de données
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Prétraitement images
│   │   ├── augment.py          # Data augmentation
│   │   └── dataset.py          # Dataset PyTorch
│   │
│   ├── models/                 # Modèles ML
│   │   ├── __init__.py
│   │   ├── classifier.py       # Classifieur CNN/MLP
│   │   ├── landmark_extractor.py  # MediaPipe wrapper
│   │   └── train.py            # Script d'entraînement
│   │
│   ├── languages/              # Traitement linguistique
│   │   ├── __init__.py
│   │   ├── tunisian_dialect.py # TextLanguage pour tunisien
│   │   ├── tsl.py             # SignLanguage pour TSL
│   │   └── dictionary.py      # Dictionnaire mapping
│   │
│   ├── translation/            # Traduction
│   │   ├── __init__.py
│   │   └── translator.py       # Pipeline complet
│   │
│   └── utils/                  # Utilitaires
│       ├── __init__.py
│       ├── visualization.py
│       └── metrics.py
│
├── api/                        # API REST
│   ├── __init__.py
│   ├── main.py
│   └── endpoints.py
│
├── web/                        # Interface web
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                      # Tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
└── scripts/                    # Scripts utilitaires
    ├── download_data.py
    ├── prepare_data.py
    └── train_model.py
```

### 4.2 Modèle de Classification

```python
# Architecture proposée
class TSLClassifier(nn.Module):
    """
    Classifieur pour la Langue des Signes Tunisienne
    - Backbone: EfficientNet-B0 (transfer learning)
    - Head: MLP personnalisé
    - Input: Landmarks MediaPipe (21 points × 3 = 63 features)
    """
    def __init__(self, num_classes=57):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

### 4.3 Pipeline de Traduction

```python
class TSLTranslator:
    """
    Pipeline complet: Caméra → Texte Tunisien → Français/Anglais/Arabe
    """
    def __init__(self):
        self.landmark_extractor = MediaPipeLandmarksModel()
        self.classifier = TSLClassifier.load('best_model.pth')
        self.dictionary = load_tsl_dictionary()
    
    def translate_frame(self, frame):
        # 1. Extraire les landmarks
        landmarks = self.landmark_extractor.embed(frame)
        
        # 2. Classifier le signe
        tsl_label = self.classifier.predict(landmarks)
        
        # 3. Convertir en dialecte tunisien
        tunisian = self.dictionary[tsl_label]
        
        # 4. Traduire vers d'autres langues
        french = translate_to_french(tunisian)
        english = translate_to_english(tunisian)
        arabic = translate_to_arabic(tunisian)
        
        return {
            'tsl': tsl_label,
            'tunisian': tunisian,
            'french': french,
            'english': english,
            'arabic': arabic
        }
```

---

## 5. Plan d'Implémentation

### Phase 1: Setup et Données (Semaine 1)
- [ ] Cloner le framework sign-language-translator
- [ ] Télécharger le dataset TSL (W7CH)
- [ ] Préparer l'environnement Python
- [ ] Explorer et nettoyer les données

### Phase 2: Modèle de Base (Semaine 2)
- [ ] Implémenter l'extraction de landmarks MediaPipe
- [ ] Créer le dataset PyTorch
- [ ] Entraîner un classifieur baseline (MLP)
- [ ] Évaluer les métriques

### Phase 3: Modèle Optimisé (Semaine 3)
- [ ] Fine-tuner EfficientNet-B0
- [ ] Data augmentation
- [ ] Optimisation hyperparamètres
- [ ] Sélection du meilleur modèle

### Phase 4: Intégration SLT (Semaine 4)
- [ ] Créer `TunisianDialect` (TextLanguage)
- [ ] Créer `TunisianSignLanguage` (SignLanguage)
- [ ] Mapper les signes TSL
- [ ] Tester la traduction

### Phase 5: API et Interface (Semaine 5)
- [ ] Créer l'API REST (FastAPI)
- [ ] Développer l'interface web (Streamlit)
- [ ] Intégrer la caméra temps réel
- [ ] Tests end-to-end

### Phase 6: Documentation (Semaine 6)
- [ ] README complet
- [ ] Documentation API
- [ ] Notebooks d'exemple
- [x] Dockerisation (Dockerfile + docker-compose.yml présents dans le dépôt)

---

## 6. Métriques d'Évaluation

| Métrique | Objectif | Mesure |
|----------|----------|--------|
| Accuracy top-1 | >95% | Classification |
| Latence | <100ms | Temps réel |
| FPS | >15 | Fluidité |
| F1-score | >0.93 | Par classe |

---

## 7. Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Dataset trop petit | Élevé | Transfer learning + augmentation |
| Variabilité performers | Moyen | Collecte complémentaire |
| Fond d'image | Moyen | Segmentation main |
| Performance temps réel | Élevé | Optimisation modèle |
| Dialecte variable | Moyen | Dictionnaire standardisé |

---

## 8. Conclusion

L'approche proposée combine:
1. **Dataset existant** (TSL W7CH) comme source principale
2. **Framework modulaire** (sign-language-translator) pour l'extensibilité
3. **Transfer learning** depuis ArASL pour améliorer la robustesse
4. **MediaPipe** pour l'extraction de features efficace
5. **Architecture modulaire** permettant l'ajout de nouvelles langues

Cette approche minimise le développement tout en maximisant la réutilisation de code existant.
