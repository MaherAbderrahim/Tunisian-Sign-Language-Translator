# 02 — Multiple Sign Sequence Tests

> 5 tests. Each combines 2–3 signs into a meaningful sequence.
> The SentenceBuilder should stitch them into a readable sentence using category-based templates.

---

## Test 1 — Greeting + Affirmation

| Field | Value |
|-------|-------|
| **Test #** | 1 |
| **Sequence** | `3aslema` → `oui` |
| **Category combo** | Demandes → Demandes |
| **Expected combined meaning** | Hello, yes |
| **Expected raw output** | `"bonjour and oui"` |

---

## Test 2 — Family member + Day

| Field | Value |
|-------|-------|
| **Test #** | 2 |
| **Sequence** | `om` → `thnin` |
| **Category combo** | Famille → Jours |
| **Expected combined meaning** | Mother comes on Monday |
| **Expected raw output** | `"mother will come on Monday"` (template: `family_then_day`) |

---

## Test 3 — Transport + Destination

| Field | Value |
|-------|-------|
| **Test #** | 3 |
| **Sequence** | `taxi` → `sbitar` |
| **Category combo** | Transport → Destinations |
| **Expected combined meaning** | Take a taxi to the hospital |
| **Expected raw output** | `"Take taxi to hospital"` (template: `transport_then_dest`) |

---

## Test 4 — Three-sign: Day + Demand + Affirmation

| Field | Value |
|-------|-------|
| **Test #** | 4 |
| **Sequence** | `lyoum` → `jom3a` → `oui` |
| **Category combo** | Demandes → Jours → Demandes |
| **Expected combined meaning** | Today is Friday, yes |
| **Expected raw output** | `"aujourd'hui and vendredi and oui"` |

---

## Test 5 — Three-sign: Family + Destination

| Field | Value |
|-------|-------|
| **Test #** | 5 |
| **Sequence** | `3ayla` → `dar` |
| **Category combo** | Famille → Destinations |
| **Expected combined meaning** | The family goes to the house |
| **Expected raw output** | `"family will go to house"` (template: `family_then_dest`) |
