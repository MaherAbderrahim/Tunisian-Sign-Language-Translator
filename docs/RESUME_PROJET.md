# TSL Translator - Résumé du Projet

## 1. Analyse des Ressources

### Dataset Tunisien (W7CH)
- **4423 images**, 57 signes TunSL
- 7 performers, images 224×224
- Licence CC0 (domaine public)
- Catégories: Demandes (24), Destinations (8), Famille (12), Jours (7), Transport (6)

### Framework sign-language-translator
- 360+ stars, architecture modulaire
- **Supporte: Urdu, Anglais, Hindi** (pas l'arabe)
- Langue des signes: Pakistan Sign Language (776 signes)
- MediaPipe intégré pour l'extraction de landmarks

### Dataset ArASL (Kaggle)
- 54,049 images, 32 alphabets arabes
- Utile pour le transfer learning

## 2. Support Arabe

**Le framework sign-language-translator NE supporte PAS l'arabe.**
- Text languages: Urdu, English, Hindi
- Sign languages: Pakistan Sign Language uniquement

**Notre solution:** Implémenter une classe `TSLDictionary` personnalisée pour mapper les 57 signes TSL vers le dialecte tunisien et autres langues.

## 3. Architecture Implémentée

```
tsl-translator/
├── src/
│   ├── config/settings.py          # Configuration
│   ├── languages/dictionary.py     # Dictionnaire TSL (57 signes)
│   ├── models/
│   │   ├── classifier.py           # MLP classifier (126 → 57 classes)
│   │   ├── landmark_extractor.py   # MediaPipe wrapper
│   │   └── train.py                # Script d'entraînement
│   └── translation/translator.py   # Pipeline complet
├── api/main.py                     # API FastAPI
├── web/app.py                      # Interface Streamlit
├── tests/test_all.py               # Tests unitaires
├── demo.py                         # Script de démonstration
├── requirements.txt                # Dépendances
├── Dockerfile                      # Docker
├── docker-compose.yml              # Orchestration
└── README.md                       # Documentation complète
```

## 4. Pipeline de Traduction

```
Caméra → MediaPipe Hands → 21 landmarks × 2 mains → MLP Classifier → 57 classes
                                                                      ↓
                                              TSL Label → Tunisien → Français/Anglais/Arabe
```

## 5. Utilisation

```bash
# Installation
pip install -r requirements.txt

# Démonstration
python demo.py

# Interface web
streamlit run web/app.py

# API
uvicorn api.main:app --reload
```

## 6. Prochaines Étapes

1. **Entraîner le modèle** sur le dataset TSL réel
2. **Ajouter plus de signes** (collecte complémentaire)
3. **Reconnaissance continue** (gestes dynamiques)
4. **Application mobile** (Android/iOS)
5. **Support multi-mains** avancé

## 7. Performances Attendues

| Métrique | Objectif |
|----------|----------|
| Accuracy top-1 | >95% |
| Latence | <100ms |
| FPS | >15 |
| F1-score | >0.93 |
