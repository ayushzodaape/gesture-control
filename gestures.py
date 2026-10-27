"""Landmark-to-gesture classification (pure Python, no camera needed).

Input: a list of 21 (x, y) normalized landmarks in MediaPipe hand order,
or None when no hand is visible. Output: a dict describing the gesture.

Conventions (MediaPipe image coordinates, y grows downward):
- A finger counts as "up" (extended) when its tip is above its PIP joint.
  This assumes a roughly upright hand, which is the normal webcam case.
- Thumb-index pinch: distance between thumb tip (4) and index tip (8).
- Scroll uses the average height of the index and middle fingertips.
- Swipe uses the wrist (0) horizontal position.

The classifier is deliberately stateful for motion gestures: scroll needs
the previous frame's finger height, swipe needs the previous wrist
position plus a cooldown so one swipe means one slide change, and click
fires only on the rising edge of a pinch so holding a pinch does not
spam clicks. Passing None (no hand) resets all motion state.
"""

import time

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

PINCH_THRESHOLD = 0.05      # normalized distance, thumb tip <-> index tip
SCROLL_THRESHOLD = 0.015    # normalized vertical movement per frame
SWIPE_THRESHOLD = 0.08      # normalized horizontal wrist movement
SWIPE_COOLDOWN = 1.0        # seconds between slide changes


def _finger_up(lm, tip, pip):
    """True when the finger is extended (tip above its PIP joint)."""
    return lm[tip][1] < lm[pip][1]


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class GestureClassifier:
    """Turns one frame of hand landmarks into a gesture dict.

    Result shapes (gesture is always present):
      {"gesture": "move", "point": (x, y)}   index fingertip, normalized
      {"gesture": "click", "edge": bool}     edge=True only on pinch start
      {"gesture": "scroll", "amount": int}  >0 = up, <0 = down, 0 = held still
      {"gesture": "swipe_left"} / {"gesture": "swipe_right"}
      {"gesture": "palm"}                    open palm held, no swipe motion
      {"gesture": "none"}                    no hand, fist, or unknown pose
    """

    def __init__(self, scroll_threshold=SCROLL_THRESHOLD,
                 swipe_threshold=SWIPE_THRESHOLD,
                 swipe_cooldown=SWIPE_COOLDOWN,
                 time_fn=None):
        self.scroll_threshold = scroll_threshold
        self.swipe_threshold = swipe_threshold
        self.swipe_cooldown = swipe_cooldown
        self._time_fn = time_fn or time.monotonic
        self._prev_mid_y = None
        self._prev_wrist_x = None
        self._last_swipe_time = -float("inf")
        self._pinching = False

    def _reset_motion(self):
        self._prev_mid_y = None
        self._prev_wrist_x = None

    def classify(self, landmarks):
        if landmarks is None:
            self._reset_motion()
            self._pinching = False
            return {"gesture": "none"}

        lm = landmarks
        index_up = _finger_up(lm, INDEX_TIP, INDEX_PIP)
        middle_up = _finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
        ring_up = _finger_up(lm, RING_TIP, RING_PIP)
        pinky_up = _finger_up(lm, PINKY_TIP, PINKY_PIP)

        # 1. Thumb-index pinch -> click (rising edge only).
        if index_up and _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < PINCH_THRESHOLD:
            edge = not self._pinching
            self._pinching = True
            self._reset_motion()
            return {"gesture": "click", "edge": edge}
        self._pinching = False

        # 2. Index + middle up -> scroll from vertical finger movement.
        if index_up and middle_up and not ring_up and not pinky_up:
            mid_y = (lm[INDEX_TIP][1] + lm[MIDDLE_TIP][1]) / 2.0
            amount = 0
            if self._prev_mid_y is not None:
                dy = self._prev_mid_y - mid_y  # >0 when fingers move up
                if abs(dy) >= self.scroll_threshold:
                    amount = max(1, min(10, int(abs(dy) * 40)))
                    amount = amount if dy > 0 else -amount
            self._prev_mid_y = mid_y
            self._prev_wrist_x = None
            return {"gesture": "scroll", "amount": amount}

        # 3. Open palm -> horizontal swipe (with cooldown), else "palm".
        if index_up and middle_up and ring_up and pinky_up:
            now = self._time_fn()
            if now - self._last_swipe_time < self.swipe_cooldown:
                # Cooling down after a swipe: disarm so the next swipe
                # needs fresh movement, not leftover motion.
                self._prev_wrist_x = None
                self._prev_mid_y = None
                return {"gesture": "palm"}
            wrist_x = lm[WRIST][0]
            if self._prev_wrist_x is not None:
                dx = wrist_x - self._prev_wrist_x
                if dx >= self.swipe_threshold:
                    self._last_swipe_time = now
                    self._prev_wrist_x = None
                    self._prev_mid_y = None
                    return {"gesture": "swipe_right"}
                if dx <= -self.swipe_threshold:
                    self._last_swipe_time = now
                    self._prev_wrist_x = None
                    self._prev_mid_y = None
                    return {"gesture": "swipe_left"}
            self._prev_wrist_x = wrist_x
            self._prev_mid_y = None
            return {"gesture": "palm"}

        # 4. Index finger only -> move cursor to the fingertip.
        if index_up and not middle_up and not ring_up and not pinky_up:
            self._reset_motion()
            return {"gesture": "move", "point": lm[INDEX_TIP]}

        # Anything else (e.g. a fist) -> do nothing.
        self._reset_motion()
        return {"gesture": "none"}
