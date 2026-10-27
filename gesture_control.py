"""Webcam gesture control (v1).

Show your hand to the webcam:
- Index finger only        -> move the mouse cursor (smoothed)
- Thumb-index pinch        -> click (once per pinch, not while held)
- Two fingers, move up/down -> scroll
- Open palm, swipe left/right -> previous / next slide (arrow keys)

Press 'q' in the preview window to quit. Moving the mouse to a screen
corner triggers pyautogui's fail-safe and stops control immediately.

Honest note: hand tracking is MediaPipe's pre-trained model, not a model
trained here. The work in this project is the gesture classification
(gestures.py), the control mapping, and this app.

pyautogui is imported only after the webcam opens, so a missing webcam
reports a clean error instead of an import traceback.
"""

import sys

import cv2
import mediapipe as mp

from gestures import GestureClassifier

SMOOTHING = 0.4  # higher = snappier cursor, lower = steadier cursor


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _fail("No webcam found. Connect a webcam, close any other app "
              "using it, then run again.")

    try:
        import pyautogui
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        cap.release()
        _fail(f"Could not start mouse control: pyautogui failed to import "
              f"({exc}). On Windows this needs a normal desktop session.")

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    draw = mp.solutions.drawing_utils
    classifier = GestureClassifier()

    smooth_x = smooth_y = None
    print("Gesture control running. Show your hand to the webcam.")
    print("Index = move | pinch = click | two fingers = scroll | "
          "palm swipe = slides | 'q' = quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: Could not read a frame from the webcam. "
                      "Exiting.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            label = "No hand"
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = [(p.x, p.y) for p in hand.landmark]
                out = classifier.classify(lm)
                gesture = out["gesture"]
                label = gesture
                if gesture == "move":
                    nx, ny = out["point"]
                    # Mirrored x so the cursor follows your hand naturally.
                    tx, ty = (1.0 - nx) * screen_w, ny * screen_h
                    if smooth_x is None:
                        smooth_x, smooth_y = tx, ty
                    else:
                        smooth_x += SMOOTHING * (tx - smooth_x)
                        smooth_y += SMOOTHING * (ty - smooth_y)
                    pyautogui.moveTo(smooth_x, smooth_y)
                elif gesture == "click" and out.get("edge"):
                    pyautogui.click()
                elif gesture == "scroll" and out.get("amount"):
                    pyautogui.scroll(out["amount"])
                elif gesture == "swipe_left":
                    pyautogui.press("left")
                    label = "swipe_left (previous slide)"
                elif gesture == "swipe_right":
                    pyautogui.press("right")
                    label = "swipe_right (next slide)"
                draw.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                classifier.classify(None)

            # Flip the annotated frame so the preview feels like a mirror.
            # Landmarks flip along with the image, so they stay aligned;
            # the text label is drawn after the flip so it reads normally.
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Gesture control - press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting.")
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
