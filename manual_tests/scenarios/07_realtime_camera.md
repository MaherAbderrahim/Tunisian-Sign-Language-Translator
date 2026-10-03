# 07 — Real-Time Camera Tests

> Instructions for testing the live camera translation pipeline.
> The system reads from the webcam, classifies each frame, and displays results in real time.

---

## Setup Steps

1. Connect a USB webcam or use the built-in laptop camera.
2. Run the translator:
   ```
   python demo.py
   ```
   Or via the web interface:
   ```
   python web/app.py
   ```
3. A window named `TSL Translator` should open showing the camera feed with overlaid landmarks and recognized signs.
4. Press `s` to toggle between SIGN and SPELL mode.
5. Press `q` to quit.

---

## Real-Time Test Scenarios

### Scenario 1 — Greeting on Camera

| Field | Value |
|-------|-------|
| **What to do** | Stand 1m from camera. Perform `3aslema` (bonjour) — open palm raised. |
| **What to observe** | TSL label `3aslema` appears on the overlay. French shows `bonjour`. Confidence bar fills up. |
| **Success criteria** | Sign detected within 2 seconds. Confidence > 0.7. No rejected frames during the gesture. |

### Scenario 2 — Rapid Sign Switching

| Field | Value |
|-------|-------|
| **What to do** | Perform `oui`, immediately switch to `non`, then back to `oui`. |
| **What to observe** | Each sign is recognized separately: `oui` → `non` → `oui`. No cross-contamination. |
| **Success criteria** | All three signs appear in the overlay in the correct order. Temporal smoothing does not merge distinct signs. |

### Scenario 3 — Spell Mode Test

| Field | Value |
|-------|-------|
| **What to do** | Press `s` to enter SPELL mode. Spell the letter `alef` (أ), then `ba` (ب), then `mim` (م). Press `s` again to return to SIGN mode. |
| **What to observe** | Spell overlay shows each letter as it is recognized. Current word builds up: `أ` → `أب` → `أبم`. |
| **Success criteria** | All three letters recognized. Word accumulation visible. No mode-switching glitches when returning to SIGN mode. |

### Scenario 4 — Sequence: Transport + Destination

| Field | Value |
|-------|-------|
| **What to do** | Perform `taxi`, pause 1s with hands down, then perform `sbitar`. |
| **What to observe** | Signs recognized: `taxi` → `sbitar`. Sentence assembled: `"Take taxi to hospital"`. |
| **Success criteria** | Both signs detected. Template `transport_then_dest` applied. Gap between signs triggers sentence finalization. |

### Scenario 5 — Extended Session Stability

| Field | Value |
|-------|-------|
| **What to do** | Perform 5 signs in sequence with 1s pauses: `3aslema` → `labes` → `oui` → `lyoum` → `mar7ba`. |
| **What to observe** | All 5 signs recognized. History bounded (does not grow unbounded). No memory leaks or UI freezes. |
| **Success criteria** | All 5 signs appear in the overlay. At least one sentence is assembled from the sequence. UI remains responsive throughout. |

---

## What to Observe

- **Landmark overlay**: Green skeleton drawn on hands in the video feed.
- **TSL label**: Shown at the top of the overlay (e.g., `TSL: 3aslema`).
- **Tunisian/French/English**: Shown below the TSL label.
- **Confidence bar**: Green bar indicating model confidence per frame.
- **Mode indicator**: `MODE: SIGN` or `MODE: SPELL` at the top.

---

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Hand detection rate | > 90% of frames with visible hands |
| Sign recognition confidence | > 0.7 for accepted predictions |
| Temporal smoothing | Stable label held for 4+ consecutive frames before emission |
| Sentence assembly | Correct template applied for category combos |
| Frame rate | > 15 FPS on CPU (30 FPS preferred) |
| Latency | < 200ms from frame capture to overlay update |
| Memory usage | Stable over 5-minute session (< 500MB) |
