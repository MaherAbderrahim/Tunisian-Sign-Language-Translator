# TEST FINAL END-TO-END

## Objectif
Tester l'intégralité du système de traduction du langage des signes tunisien, depuis l'enregistrement vidéo sur téléphone jusqu'à la phrase finale corrigée par le LLM.

## Prérequis
- Application lancée (`RUN_TESTS.ps1`)
- Caméra ou vidéo prête
- Navigateur ouvert sur `http://localhost:3000`
- Modèle MediaPipe chargé
- Moteur d'inférence actif
- LLM configuré (si disponible)

---

## Phase 1 : Caméra temps réel (5 minutes)

Test 5 signes en temps réel via la caméra.

### Signes à tester

| # | Signe | Mot tunisien | Attendu |
|---|-------|--------------|---------|
| 1 | bonjour | 3aslema | bonjour |
| 2 | oui | ey | oui |
| 3 | non | le | non |
| 4 | maison | dar | maison |
| 5 | famille | 3ayla | famille |

### Procédure
1. Activer la caméra en cliquant sur **Start Camera**
2. Pour chaque signe :
   - Se placer face à la caméra
   - Effectuer le signe de manière claire
   - Observer le badge de confiance (`confidence`)
   - Vérifier que la traduction affichée correspond au signe attendu
3. Noter les résultats dans le tableau ci-dessous

### Résultats Phase 1

| # | Signe | Confiance | Traduction | Correct | Notes |
|---|-------|-----------|------------|---------|-------|
| 1 | bonjour | | | ☐ | |
| 2 | oui | | | ☐ | |
| 3 | non | | | ☐ | |
| 4 | maison | | | ☐ | |
| 5 | famille | | | ☐ | |

---

## Phase 2 : Upload vidéo (10 minutes)

Enregistrer 3 vidéos sur téléphone et les uploader.

### Vidéos à préparer

| # | Contenu | Signes attendus |
|---|---------|-----------------|
| 1 | Signe unique | bienvenue |
| 2 | Deux signes | bonjour + bien |
| 3 | Trois signes | famille + pere + mere |

### Procédure
1. Cliquer sur **Upload Video**
2. Sélectionner la vidéo enregistrée sur le téléphone
3. Attendre le traitement complet (upload → MediaPipe → inférence → assemblage)
4. Vérifier les keypoints extraits
5. Vérifier la séquence de signes détectée
6. Vérifier la phrase assemblée
7. Noter les résultats

### Résultats Phase 2

| # | Vidéo | Signes détectés | Phrase assemblée | Correct | Notes |
|---|-------|-----------------|------------------|---------|-------|
| 1 | Single sign | | | ☐ | |
| 2 | Two signs | | | ☐ | |
| 3 | Three signs | | | ☐ | |

---

## Phase 3 : Fingerspelling (10 minutes)

Tester le mode alphabet (épelling).

### Mots à épeler

| # | Mot | Séquence attendue |
|---|-----|-------------------|
| 1 | dar | d-a-r |
| 2 | om | o-m |
| 3 | oui | o-u-i |

### Procédure
1. Basculer en mode **SPELL** via le bouton de toggle
2. Pour chaque mot :
   - Former chaque lettre de l'alphabet avec la main
   - Observer la détection de chaque lettre
   - Une fois les lettres saisies, appuyer sur **Flush** ou attendre l'auto-flush
   - Vérifier le mot assemblé
3. Noter les résultats

### Résultats Phase 3

| # | Mot | Lettres détectées | Mot assemblé | Correct | Notes |
|---|-----|-------------------|--------------|---------|-------|
| 1 | dar | | | ☐ | |
| 2 | om | | | ☐ | |
| 3 | oui | | | ☐ | |

---

## Phase 4 : Mode mixte (10 minutes)

Alterner entre les modes SIGN et SPELL dans une même session.

### Procédure
1. **Étape 1** : Mode **SIGN** → Effectuer le signe "bonjour"
2. **Étape 2** : Basculer en mode **SPELL** → Épeler "dar" (d-a-r)
3. **Étape 3** : Basculer en mode **SIGN** → Effectuer le signe "merci" (se7a)
4. Vérifier la sortie complète concaténée
5. Noter les résultats

### Résultats Phase 4

| Étape | Mode | Action | Résultat | Correct | Notes |
|-------|------|--------|----------|---------|-------|
| 1 | SIGN | bonjour | | ☐ | |
| 2 | SPELL | d-a-r | | ☐ | |
| 3 | SIGN | merci | | ☐ | |
| **Sortie complète** | | | | ☐ | |

---

## Phase 5 : Correction LLM (5 minutes)

Vérifier que le LLM corrige et améliore la sortie.

### Procédure
1. Vérifier que le LLM est configuré et actif
2. Effectuer un signe (ex. "bonjour")
3. Observer le champ **"Corrected"** dans la sortie
4. Comparer la traduction brute avec la correction LLM
5. Noter les résultats

### Résultats Phase 5

| # | Signe | Traduction brute | Correction LLM | Amélioration | Notes |
|---|-------|------------------|----------------|--------------|-------|
| 1 | bonjour | | | ☐ | |

---

## Phase 6 : Test de stress (5 minutes)

Soumettre le système à des conditions difficiles.

### Scénarios de stress

| # | Scénario | Description |
|---|----------|-------------|
| 1 | Signes rapides | Enchaîner 4-5 signes rapidement sans pause |
| 2 | Main partielle | Montrer seulement une partie de la main |
| 3 | Main hors caméra | Laisser la main sortir du cadre puis revenir |
| 4 | Switches rapides | Alterner rapidement entre SIGN et SPELL |

### Procédure
1. Pour chaque scénario, effectuer l'action décrite
2. Observer la stabilité du système
3. Vérifier qu'aucune erreur critique ne se produit
4. Noter les résultats

### Résultats Phase 6

| # | Scénario | Comportement attendu | Résultat réel | Pass | Notes |
|---|----------|---------------------|---------------|------|-------|
| 1 | Signes rapides | Détection stable de chaque signe | | ☐ | |
| 2 | Main partielle | Gestion gracieuse (pas de crash) | | ☐ | |
| 3 | Main hors caméra | Re-détection rapide au retour | | ☐ | |
| 4 | Switches rapides | Basculer proprement entre modes | | ☐ | |

---

## Bilan

| Phase | Tests | Pass | Fail | Notes |
|-------|-------|------|------|-------|
| 1. Caméra | 5 | | | |
| 2. Vidéo | 3 | | | |
| 3. Fingerspelling | 3 | | | |
| 4. Mode mixte | 1 | | | |
| 5. LLM | 1 | | | |
| 6. Stress | 4 | | | |
| **TOTAL** | **17** | | | |

---

## Verdict final

☐ **TOUT PASSE** — Le système est fonctionnel et prêt pour la production.

☐ **PROBLEMES MINEURS** — Quelques erreurs mineures mais le système fonctionne globalement. Des ajustements sont nécessaires.

☐ **PROBLEMES MAJEURS** — Le système ne fonctionne pas correctement. Des corrections importantes sont requises avant toute mise en production.

---

### Signes supportés dans ce test

> **ATTENTION** : Ce test utilise UNIQUEMENT les signes suivants qui sont supportés par le modèle :
> `bonjour` (3aslema), `oui` (ey), `non` (le), `maison` (dar), `famille` (3ayla),
> `bienvenue`, `bien`, `pere`, `mere`, `merci` (se7a)
>
> **NE PAS UTILISER** : `ta3raf`, `karhba`, `metro`, `train` — ces signes ne sont pas supportés par le modèle actuel.
