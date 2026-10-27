# Gesture Control (v1)

Control your Windows PC with hand gestures in front of a webcam.
No model training, no dataset, no GPU — it runs on CPU with a webcam.

## Gestures

| What you do | What happens |
|---|---|
| Index finger only, move it | Mouse cursor follows (smoothed, so it does not jitter) |
| Thumb + index pinch | Click (fires once per pinch, not while you hold it) |
| Index + middle finger, move up/down | Scroll up / down |
| Open palm, swipe left / right | Previous / next slide (left / right arrow keys) |
| No hand visible | Nothing happens — no stray clicks, no crash |

A preview window shows your webcam feed with hand landmarks and the
currently detected gesture written on it, so you can see what the app
understood. Press `q` in the preview window to quit. Moving the mouse
to a screen corner triggers pyautogui's fail-safe and stops control.

## Honest note: what is and is not mine here

Hand tracking uses **MediaPipe's pre-trained hand model** — I did not
train it, and this project does not claim otherwise. The work in this
project is the gesture classification logic (`gestures.py`: which finger
poses and motions map to which action, with pinch edge detection,
scroll thresholds, and a swipe cooldown), the control system (cursor
smoothing, mirrored mapping, fail-safe), and the app itself
(`gesture_control.py`).

## Run it (Windows PC)

```bash
pip install -r requirements.txt
python gesture_control.py
```

Run it from inside the project folder (not as a lone file from
elsewhere), because `gesture_control.py` imports `gestures.py` from the
same folder.

## How it works

- `gesture_control.py` — opens the webcam, runs MediaPipe Hands on each
  frame, feeds the 21 hand landmarks to the classifier, and turns the
  result into mouse/keyboard actions with pyautogui.
- `gestures.py` — pure landmark-to-gesture logic with no camera or
  display dependency, so it can be unit-tested with synthetic
  landmarks (one test per gesture plus a no-hand case).

## Troubleshooting

- `ERROR: No webcam found` — connect a webcam and close any other app
  using it (browser, Teams, etc.), then run again.
- `pip` reports no matching distribution for mediapipe — mediapipe
  publishes wheels for specific Python versions. If your Python is very
  new (e.g. 3.14), install Python 3.11 or 3.12 from python.org and use
  that for this project.
- Tracking is jumpy — use good frontal lighting, keep your hand fully
  in frame, and hold it roughly upright. Only one hand is tracked.
- The app does nothing when your hand leaves the frame — that is by
  design. Bring your hand back and it resumes.

## Limitations (v1)

- One hand only; a second hand is ignored.
- Gestures assume a roughly upright hand.
- Swipe needs a deliberate sideways motion and has a 1-second cooldown,
  so one swipe changes exactly one slide.
- Everything runs locally on your PC. Nothing is recorded or sent
  anywhere.
