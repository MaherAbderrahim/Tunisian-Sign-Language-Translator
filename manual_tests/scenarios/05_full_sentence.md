# 05 — Full Sentence Tests

> 5 tests of natural multi-sign sentences.
> Each test specifies the sign sequence, the expected Tunisian meaning, a natural French translation,
> and the expected raw output before/after LLM correction.

---

## Test 1 — Greeting Sequence

| Field | Value |
|-------|-------|
| **Sign sequence** | `3aslema` → `labes` |
| **Tunisian meaning** | aslema, labes |
| **French natural** | Bonjour, comment allez-vous ? |
| **Expected raw output** | `"bonjour and bien"` |
| **Expected corrected (LLM)** | `"Bonjour, comment allez-vous ?"` |

---

## Test 2 — Family Will Come

| Field | Value |
|-------|-------|
| **Sign sequence** | `om` → `thnin` |
| **Tunisian meaning** | om, thnin |
| **French natural** | Maman viendra lundi. |
| **Expected raw output** | `"mother will come on Monday"` |
| **Expected corrected (LLM)** | `"Maman viendra lundi."` |

---

## Test 3 — Take Transport to Destination

| Field | Value |
|-------|-------|
| **Sign sequence** | `taxi` → `sbitar` |
| **Tunisian meaning** | taxi, sbitar |
| **French natural** | Prenez un taxi pour l'hôpital. |
| **Expected raw output** | `"Take taxi to hospital"` |
| **Expected corrected (LLM)** | `"Prenez un taxi pour l'hôpital."` |

---

## Test 4 — Demand Request

| Field | Value |
|-------|-------|
| **Sign sequence** | `t7eb` → `ta9ra` → `oui` |
| **Tunisian meaning** | t7eb, ta9ra, oui |
| **French natural** | Tu veux lire, oui ? |
| **Expected raw output** | `"tu veux and tu lis and oui"` |
| **Expected corrected (LLM)** | `"Tu veux lire, oui ?"` |

---

## Test 5 — Multi-sign Sentence

| Field | Value |
|-------|-------|
| **Sign sequence** | `bou` → `sbitar` |
| **Tunisian meaning** | bou, sbitar |
| **French natural** | Papa va à l'hôpital. |
| **Expected raw output** | `"father goes to hospital"` |
| **Expected corrected (LLM)** | `"Papa va à l'hôpital."` |
