# 04 — Mixed Sign + Fingerspelling Tests

> 5 tests combining whole-sign recognition with fingerspelled letters.
> These test the system's ability to handle sign→spell→sign transitions.
> Use the `cv` sign to enter SPELL mode (the mode controller treats certain signs as mode toggles),
> then fingerspell, then return to SIGN mode with a known sign.

---

## Test 1 — Sign + Letters + Sign

| Field | Value |
|-------|-------|
| **Test #** | 1 |
| **Sequence** | `oui` → (switch to spell) → alef → ba → (switch to sign) → `non` |
| **Expected meaning** | Yes, then letters a-b, then no |
| **Expected raw output** | Sign mode: `tsl: "oui"`. Spell mode: `letter: "alef"`, `letter: "ba"`, assembled word: `اب`. Sign mode: `tsl: "non"` |

---

## Test 2 — Sign + Letters + Sign

| Field | Value |
|-------|-------|
| **Test #** | 2 |
| **Sequence** | `mar7ba` → (spell) → sin → mim → (sign) → `labes` |
| **Expected meaning** | Welcome, then letters s-m (forming sm), then well |
| **Expected raw output** | `tsl: "mar7ba"`, then spell output `asm`, then `tsl: "labes"` |

---

## Test 3 — Sign + Letters + Sign

| Field | Value |
|-------|-------|
| **Test #** | 3 |
| **Sequence** | `3aslema` → (spell) → kaf → mim → (sign) → `oui` |
| **Expected meaning** | Hello, then letters k-m, then yes |
| **Expected raw output** | `tsl: "3aslema"`, spell: `كم`, `tsl: "oui"` |

---

## Test 4 — Sign + Letters + Sign

| Field | Value |
|-------|-------|
| **Test #** | 4 |
| **Sequence** | `t7eb` → (spell) → ba → ba → (sign) → `non` |
| **Expected meaning** | You want, then letters b-b, then no |
| **Expected raw output** | `tsl: "t7eb"`, spell: `بب`, `tsl: "non"` |

---

## Test 5 — Sign + Letters + Sign

| Field | Value |
|-------|-------|
| **Test #** | 5 |
| **Sequence** | `demande` → (spell) → dal → mim → (sign) → `oui` |
| **Expected meaning** | Request, then letters d-m, then yes |
| **Expected raw output** | `tsl: "demande"`, spell: `دم`, `tsl: "oui"` |
