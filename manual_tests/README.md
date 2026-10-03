# Manual Testing Guide — Tunisian Sign Language Translator

Welcome to the test kit for the Tunisian Sign Language (TSL) Translator. This guide will walk you through everything you need to know, even if you have zero experience with sign language.

---

## 1. What Is This Project?

This app translates Tunisian Sign Language into text using a camera. A person signs in front of the camera, and the system tries to recognize the sign and display the meaning on screen. You are helping us test how well it works.

---

## 2. Prerequisites

Before you begin, make sure you have:

- **Python 3.12 or later** installed on your computer
- **A working webcam** (built-in or external)
- **A phone** to record yourself or someone else signing
- **Internet connection** (for first-time setup only)
- **A quiet room** with decent lighting

---

## 3. Quick Start

There are two ways to launch the app:

**Option A — Use the PowerShell script (recommended):**
1. Open PowerShell in the project folder
2. Run: `.\RUN_TESTS.ps1`

**Option B — Use Python directly:**
1. Open a terminal in the project folder
2. Run: `python start.py`

If the app does not start, check the [Troubleshooting](#10-troubleshooting) section below.

---

## 4. How to Record Videos

This is the most important part. Follow these rules carefully so the app can recognize signs correctly.

### Camera Position
- Place the camera **at arm's length** from your body
- Frame yourself from the **chest up**
- Keep the camera **steady** — use a tripod or prop it against something

### Lighting
- **Face the light source** (window, lamp)
- **Avoid backlight** — do not stand in front of a bright window
- The light should be **even on your face and hands**

### Background
- Use a **plain, solid-colored background** (white, gray, or blue wall works well)
- The background should **contrast with your skin tone**
- Avoid busy patterns, posters, or moving objects behind you

### Hands
- Make sure both hands are **visible** in the frame
- **No gloves** — the app needs to see your skin
- **Remove jewelry** — rings, bracelets, and watches can confuse the system

### Duration
- Each sign should last **1 to 3 seconds**
- Do not rush — move slowly and deliberately

### Starting Position
- Begin each video with your hands **at rest** at your sides or in front of your body
- Hold this position for **1 second** before signing

### Performing the Sign
- Execute the sign **clearly and fully**
- Move your hands **deliberately**, not too fast
- Make sure the **full motion** is captured on camera

### Ending Position
- After the sign, **return your hands to the rest position**
- Hold for **1 second** before stopping the recording

### Separating Signs
- If you record multiple signs in one video, **pause 1–2 seconds** between each sign
- This helps the system detect where one sign ends and the next begins

---

## 5. How to Transfer Videos

Once you have recorded your test videos, transfer them to the computer:

- **USB cable** — Connect your phone and drag files to the project folder
- **Email** — Send the videos to yourself and download on the computer
- **Cloud storage** — Upload to Google Drive or OneDrive and download on the computer

Keep your test videos in a folder of your choice (there is no dedicated `test_videos` folder) — you will upload them to the app from the browser during testing. The step-by-step scenario guides live in `manual_tests\scenarios\`.

---

## 6. How to Use the App

1. Launch the app using one of the methods in [Quick Start](#3-quick-start)
2. Your **browser opens automatically** (http://localhost:3000) — the app runs in the browser, not in a desktop window
3. Click **Start Camera** and allow camera access when prompted
4. When you are ready, **perform a sign** in front of the camera
5. The app will process the sign and display the result on screen
6. Repeat for each sign you want to test
7. When done, **close the browser tab** (or stop the launcher with `Ctrl+C` in the terminal)

---

## 7. How to Interpret Results

After each sign, the app displays several pieces of information:

- **Translation output** — The word or phrase the app thinks you signed
- **Confidence** — A percentage showing how sure the app is (e.g., 87% confident)
- **Corrected** — If the app self-corrects its output, this shows the final result after correction

### Confidence Thresholds
- **80% and above** — High confidence, likely correct
- **60%–79%** — Medium confidence, may be wrong
- **Below 60%** — Low confidence, likely incorrect

---

## 8. Test Scenarios Overview

There are 7 test scenarios to complete:

| # | Scenario | What You Are Testing |
|---|----------|---------------------|
| 1 | Single common signs | Basic vocabulary (greetings, simple words) |
| 2 | Multi-sign phrases | Sequences of 2–3 signs together |
| 3 | Speed variation | Signs performed at slow, normal, and fast pace |
| 4 | Lighting conditions | Bright, dim, and uneven lighting |
| 5 | Background variation | Plain vs. cluttered backgrounds |
| 6 | Hand accessories | Signs with and without rings/bracelets |
| 7 | Unsupported signs | Signs the system is not trained on (expected to fail) |

---

## 9. Results Tracking

For each scenario, fill in the results template:

1. Open `manual_tests\results\MANUAL_TEST_RESULTS.md`
2. For each sign you test, record:
   - The sign you performed
   - The translation output from the app
   - The confidence percentage
   - Whether the result was correct (Yes/No)
   - Any notes (e.g., "app struggled with fast motion")
3. Save the file when finished
4. Share it with the team

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| App does not start | Make sure Python 3.12+ is installed and on your PATH |
| Camera not detected | Check that no other app is using the camera. Restart the app. |
| Poor recognition | Improve lighting, use a plain background, slow down your signs |
| App is slow | Close other programs. Make sure your computer meets the minimum requirements. |
| Wrong translations | Review the [recording guidelines](#4-how-to-record-videos). Retake the video. |
| Error messages | Copy the full error message and report it to the team |

---

## 11. Limitations

Be aware of these current limitations:

- **4 unsupported signs** are included in the test scenarios — these are expected to fail and help us measure the system's behavior on unknown input
- Confidence scores are **not always reliable** — a high score does not guarantee correctness
- The system works best with **one person signing at a time**
- **Rapid consecutive signing** may cause missed or merged detections
- **Extreme angles** (too high, too low, too far to the side) reduce accuracy

---

Thank you for helping us test the Tunisian Sign Language Translator. Your feedback makes this tool better for everyone.
