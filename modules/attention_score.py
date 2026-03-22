import time
import json
import os
from datetime import datetime

# Nudge levels based on distraction duration
NUDGE_LEVELS = {
    5:  1,   # 5 sec  → level 1 (border turns orange)
    10: 2,   # 10 sec → level 2 (popup message)
    20: 3,   # 20 sec → level 3 (sound nudge)
    30: 4,   # 30 sec → level 4 (pause + breathing)
}

SESSION_DIR = "data/sessions"


class AttentionScoreEngine:
    """
    Aggregates signals from all modules and computes:
    - A rolling Focus Score (0–100)
    - Nudge level based on continuous distraction time
    - Session log saved locally as JSON
    """

    def __init__(self):
        self.score              = 100.0
        self.distracted_since   = None   # timestamp when distraction started
        self.distraction_secs   = 0.0
        self.nudge_level        = 0
        self.session_start      = time.time()
        self.score_history      = []     # list of (timestamp, score)
        self.event_log          = []     # list of distraction events
        self._last_score_time   = time.time()
        self._focused_streak    = 0.0    # seconds of continuous focus
        os.makedirs(SESSION_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Main update — call every frame
    # ------------------------------------------------------------------
    def update(self, head_distracted: bool, gaze_distracted: bool,
               drowsy: bool, distractors: list) -> dict:
        """
        Takes boolean signals from each module, updates the score.
        Returns current state dict for the UI.
        """
        now = time.time()

        # Any distraction signal
        any_distracted = head_distracted or gaze_distracted or drowsy or bool(distractors)

        # Track continuous distraction time
        if any_distracted:
            if self.distracted_since is None:
                self.distracted_since = now
            self.distraction_secs = now - self.distracted_since
            self._focused_streak  = 0.0
        else:
            if self.distracted_since is not None:
                # Log the distraction event
                self.event_log.append({
                    "start":    self.distracted_since,
                    "duration": self.distraction_secs,
                    "causes": {
                        "head":        head_distracted,
                        "gaze":        gaze_distracted,
                        "drowsy":      drowsy,
                        "distractor":  bool(distractors),
                    }
                })
            self.distracted_since = None
            self.distraction_secs = 0.0
            self._focused_streak += (now - self._last_score_time)

        # Compute nudge level
        self.nudge_level = 0
        for secs, level in sorted(NUDGE_LEVELS.items(), reverse=True):
            if self.distraction_secs >= secs:
                self.nudge_level = level
                break

        # Update focus score every second
        elapsed = now - self._last_score_time
        if elapsed >= 1.0:
            if any_distracted:
                # Decay faster for longer distractions
                decay = 2.0 + (self.distraction_secs / 10.0)
                self.score = max(0.0, self.score - decay * elapsed)
            else:
                # Slowly recover when focused
                self.score = min(100.0, self.score + 1.5 * elapsed)

            self.score_history.append({
                "t":     now - self.session_start,
                "score": round(self.score, 1)
            })
            self._last_score_time = now

        return self.get_state(head_distracted, gaze_distracted, drowsy, distractors)

    # ------------------------------------------------------------------
    def get_state(self, head_distracted, gaze_distracted, drowsy, distractors):
        return {
            "score":            round(self.score, 1),
            "nudge_level":      self.nudge_level,
            "distraction_secs": round(self.distraction_secs, 1),
            "head_distracted":  head_distracted,
            "gaze_distracted":  gaze_distracted,
            "drowsy":           drowsy,
            "distractors":      distractors,
            "session_elapsed":  round(time.time() - self.session_start, 0),
            "focused_streak":   round(self._focused_streak, 0),
        }

    # ------------------------------------------------------------------
    def save_session(self):
        """Save the session log to a local JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = os.path.join(SESSION_DIR, f"session_{timestamp}.json")

        data = {
            "session_start":   datetime.fromtimestamp(self.session_start).isoformat(),
            "session_end":     datetime.now().isoformat(),
            "duration_secs":   round(time.time() - self.session_start, 0),
            "final_score":     round(self.score, 1),
            "score_history":   self.score_history,
            "distraction_events": self.event_log,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Session saved to {path}")
        return path

    # ------------------------------------------------------------------
    def get_nudge_message(self):
        messages = {
            0: "",
            1: "Stay focused!",
            2: "Hey, you drifted 👀",
            3: "Come back to your work!",
            4: "Take a deep breath. Refocus.",
        }
        return messages.get(self.nudge_level, "")

    def get_score_color(self):
        """Returns a color string based on current score."""
        if self.score >= 70:
            return "green"
        elif self.score >= 40:
            return "orange"
        else:
            return "red"
