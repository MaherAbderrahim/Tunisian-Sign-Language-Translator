# 06 — Video Upload Tests

> Instructions for testing the system with pre-recorded video files.
> The system processes video frame-by-frame, accumulates signs, and outputs finalized sentences.

---

## How to Record

1. **Resolution**: 640x480 or higher (720p preferred).
2. **Lighting**: Bright, even lighting on the signer. Avoid backlighting.
3. **Background**: Plain, non-cluttered background. Avoid patterns that confuse MediaPipe.
4. **Framing**: Upper body visible from waist to above head. Hands must be clearly visible at chest-to-shoulder height.
5. **Duration**: Keep each sign gesture to 1–2 seconds. Pause 0.5s between signs.
6. **Camera**: Front-facing (webcam). Record at 30fps if possible.
7. **File format**: `.mp4` or `.avi`.

---

## What to Include in the Video

- Start with hands at rest (no sign) for 1 second — establishes baseline.
- Perform each sign clearly, one at a time.
- Pause (hands down or at rest) between signs for 0.5–1 second.
- End with hands at rest for 1 second.

---

## Expected Behavior

1. System detects hands via MediaPipe landmark extraction.
2. Each stable sign is recognized and buffered by the `StreamingRecognizer`.
3. On a long enough gap (no hands), the `SentenceBuilder` flushes accumulated signs into a sentence.
4. The LLM corrector (if enabled) refines the raw output.

---

## Video Test Scenarios

### Scenario 1 — Single Sign Video

| Field | Value |
|-------|-------|
| **File name** | `test_01_hello.mp4` |
| **Content** | Perform `3aslema` (bonjour), hold 2 seconds, lower hands. |
| **Expected recognized signs** | `["3aslema"]` |
| **Expected sentence** | `"They say: hello"` or `"bonjour"` depending on output_lang |

### Scenario 2 — Two-Sign Sequence Video

| Field | Value |
|-------|-------|
| **File name** | `test_02_taxi_hospital.mp4` |
| **Content** | Perform `taxi`, pause 0.5s, perform `sbitar`, lower hands. |
| **Expected recognized signs** | `["taxi", "sbitar"]` |
| **Expected sentence** | `"Take taxi to hospital"` (transport_then_dest template) |

### Scenario 3 — Multi-sign Sentence Video

| Field | Value |
|-------|-------|
| **File name** | `test_03_mom_monday.mp4` |
| **Content** | Perform `om`, pause 0.5s, perform `thnin`, lower hands. |
| **Expected recognized signs** | `["om", "thnin"]` |
| **Expected sentence** | `"mother will come on Monday"` (family_then_day template) |

---

## Verification Checklist

- [ ] Video plays without errors in the system
- [ ] MediaPipe hand detection works (landmarks visible on frames)
- [ ] Signs are recognized in the correct order
- [ ] Sentence is assembled correctly from the template
- [ ] No unsupported signs (ta3raf, karhba, metro, train) appear in output
- [ ] Confidence scores are above threshold (0.7+) for each sign
- [ ] LLM correction (if enabled) improves readability
