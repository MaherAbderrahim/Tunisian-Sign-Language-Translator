# 01 — Single Sign Tests

> 10 tests. Each verifies that a single isolated sign is recognized correctly.
> Signs are drawn from multiple categories to ensure broad coverage.

---

## Test 1

| Field | Value |
|-------|-------|
| **Test #** | 1 |
| **Sign label** | `3aslema` |
| **French** | bonjour |
| **Arabic** | اسلامة |
| **Gesture** | Open palm raised toward the camera with fingers together, held at chest height — the standard greeting wave. |
| **Expected result** | `tsl: "3aslema"`, `french: "bonjour"`, `english: "hello"`, `confidence > 0.7` |

---

## Test 2

| Field | Value |
|-------|-------|
| **Test #** | 2 |
| **Sign label** | `oui` |
| **French** | oui |
| **Arabic** | ايـه |
| **Gesture** | Single decisive nod of the head, or a thumbs-up held in front of the chest. |
| **Expected result** | `tsl: "oui"`, `french: "oui"`, `english: "yes"`, `confidence > 0.7` |

---

## Test 3

| Field | Value |
|-------|-------|
| **Test #** | 3 |
| **Sign label** | `non` |
| **French** | non |
| **Arabic** | لا |
| **Gesture** | Index finger wagged side to side in front of the chest. |
| **Expected result** | `tsl: "non"`, `french: "non"`, `english: "no"`, `confidence > 0.7` |

---

## Test 4

| Field | Value |
|-------|-------|
| **Test #** | 4 |
| **Sign label** | `bou` |
| **French** | père |
| **Arabic** | ابو |
| **Gesture** | Fist placed on the forehead, simulating a salute — the TSL sign for father. |
| **Expected result** | `tsl: "bou"`, `french: "père"`, `english: "father"`, `confidence > 0.7` |

---

## Test 5

| Field | Value |
|-------|-------|
| **Test #** | 5 |
| **Sign label** | `om` |
| **French** | mère |
| **Arabic** | ام |
| **Gesture** | Both hands cradling an imaginary baby at chest level. |
| **Expected result** | `tsl: "om"`, `french: "mère"`, `english: "mother"`, `confidence > 0.7` |

---

## Test 6

| Field | Value |
|-------|-------|
| **Test #** | 6 |
| **Sign label** | `sbitar` |
| **French** | hôpital |
| **Arabic** | سبيطار |
| **Gesture** | Two fingers (index + middle) pressed against the opposite wrist like taking a pulse. |
| **Expected result** | `tsl: "sbitar"`, `french: "hôpital"`, `english: "hospital"`, `confidence > 0.7` |

---

## Test 7

| Field | Value |
|-------|-------|
| **Test #** | 7 |
| **Sign label** | `taxi` |
| **French** | taxi |
| **Arabic** | طاكسي |
| **Gesture** | Hand extended forward, thumb up, making a hitchhiking motion. |
| **Expected result** | `tsl: "taxi"`, `french: "taxi"`, `english: "taxi"`, `confidence > 0.7` |

---

## Test 8

| Field | Value |
|-------|-------|
| **Test #** | 8 |
| **Sign label** | `jom3a` |
| **French** | vendredi |
| **Arabic** | الجمعة |
| **Gesture** | Hands brought together in a prayer-like gesture at chest level — TSL sign for Friday. |
| **Expected result** | `tsl: "jom3a"`, `french: "vendredi"`, `english: "Friday"`, `confidence > 0.7` |

---

## Test 9

| Field | Value |
|-------|-------|
| **Test #** | 9 |
| **Sign label** | `mar7ba` |
| **French** | bienvenue |
| **Arabic** | مرحبا |
| **Gesture** | Both arms swept open wide toward the camera as if inviting someone in. |
| **Expected result** | `tsl: "mar7ba"`, `french: "bienvenue"`, `english: "welcome"`, `confidence > 0.7` |

---

## Test 10

| Field | Value |
|-------|-------|
| **Test #** | 10 |
| **Sign label** | `cv` |
| **French** | cv |
| **Arabic** | سي في |
| **Gesture** | Both hands flat, palms facing each other, moved downward as if presenting a document. |
| **Expected result** | `tsl: "cv"`, `french: "cv"`, `english: "resume"`, `confidence > 0.7` |
