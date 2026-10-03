# 03 — Fingerspelling Tests

> Tests the 31-class Arabic alphabet fingerspelling system.
> Each letter must be held stable for the model to register it.
> The SpellAssembler accumulates letters into words, using `space` as a word separator.

---

## Part A — Single Letters (5 tests)

Each test: perform one letter gesture, hold for ~1 second.

| # | Letter label | Arabic | Latin | Gesture description | Expected raw output |
|---|-------------|--------|-------|---------------------|---------------------|
| 1 | `alef` | أ | a | Index finger pointed straight up, hand held at chest height | `letter: "alef"`, `arabic: "أ"` |
| 2 | `ba` | ب | b | Flat hand, palm down, fingers together, held horizontally | `letter: "ba"`, `arabic: "ب"` |
| 3 | `mim` | م | m | Fist with thumb tucked between index and middle finger | `letter: "mim"`, `arabic: "م"` |
| 4 | `ayn` | ع | 3 | Thumb and index finger forming a circle (OK gesture), other fingers extended | `letter: "ayn"`, `arabic: "ع"` |
| 5 | `kaf` | ك | k | Open hand, fingers spread, palm facing camera, held at shoulder height | `letter: "kaf"`, `arabic: "ك"` |

---

## Part B — Short Words (5 tests)

Each test: spell the word letter-by-letter, pausing briefly (~0.5s) between letters.
After the last letter, lower your hand (or use the `finish` gesture) to signal end of word.

### Test 6 — Word: "salam" (سلام = peace)

| Field | Value |
|-------|-------|
| **Letters to spell** | sin → alef → lam → alef → mim |
| **Latin** | s - a - l - a - m |
| **Expected assembled word** | `سلام` |
| **Expected raw output** | After each letter: `letter_added`. After `finish` or gap: `sentence: "سلام"` |

### Test 7 — Word: "kitab" (كتاب = book)

| Field | Value |
|-------|-------|
| **Letters to spell** | kaf → ta → alef → ba |
| **Latin** | k - t - a - b |
| **Expected assembled word** | `كتاب` |
| **Expected raw output** | `sentence: "كتاب"` |

### Test 8 — Word: "dar" (دار = house — same as the sign `dar`)

| Field | Value |
|-------|-------|
| **Letters to spell** | dal → alef → ra |
| **Latin** | d - a - r |
| **Expected assembled word** | `دار` |
| **Expected raw output** | `sentence: "دار"` |

### Test 9 — Word: "nam" (نام = sleep)

| Field | Value |
|-------|-------|
| **Letters to spell** | nun → alef → mim |
| **Latin** | n - a - m |
| **Expected assembled word** | `نام` |
| **Expected raw output** | `sentence: "نام"` |

### Test 10 — Word: "kalb" (كلب = dog)

| Field | Value |
|-------|-------|
| **Letters to spell** | kaf → alef → lam → ba |
| **Latin** | k - a - l - b |
| **Expected assembled word** | `كلب` |
| **Expected raw output** | `sentence: "كلب"` |
