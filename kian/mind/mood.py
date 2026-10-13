"""
kian.mind.mood
=============
A small affective state that makes Kian feel alive: it shifts with what happens,
decays toward a calm baseline over time, and colours both how it speaks (the
sampling temperature) and how its face looks.

Dimensions (all simple scalars):
* **valence**   -1 (low) .. +1 (content)
* **arousal**    0 (calm) .. 1 (excited)
* **curiosity**  0 .. 1 (drive to learn/explore)

This is an affect model, not a claim of feelings — it is a dynamical system that
gives the being coherent, legible behaviour over a session.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Mood:
    valence: float = 0.1
    arousal: float = 0.3
    curiosity: float = 0.6
    # baselines it relaxes toward
    _bv: float = 0.1
    _ba: float = 0.3
    _bc: float = 0.6

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def event(self, kind: str) -> None:
        """Nudge mood by a named event."""
        table = {
            "positive":     (+0.25, +0.10, +0.05),
            "negative":     (-0.25, +0.15, +0.00),
            "learned":      (+0.15, +0.10, +0.10),
            "grew":         (+0.50, +0.35, +0.20),
            "greeted":      (+0.15, +0.10, +0.05),
            "question":     (+0.02, +0.08, +0.20),
            "saw_something":(+0.05, +0.12, +0.15),
            "drew":         (+0.20, +0.10, +0.05),
            "idle":         (0.0, -0.05, +0.02),
        }
        dv, da, dc = table.get(kind, (0.0, 0.0, 0.0))
        self.valence = self._clip(self.valence + dv, -1, 1)
        self.arousal = self._clip(self.arousal + da, 0, 1)
        self.curiosity = self._clip(self.curiosity + dc, 0, 1)

    def decay(self, rate: float = 0.1) -> None:
        """Relax toward baseline (call once per interaction/tick)."""
        self.valence += (self._bv - self.valence) * rate
        self.arousal += (self._ba - self.arousal) * rate
        self.curiosity += (self._bc - self.curiosity) * rate

    def temperature(self) -> float:
        """Map arousal+curiosity to a sampling temperature in [0.6, 1.1]."""
        return round(0.6 + 0.3 * self.arousal + 0.2 * self.curiosity, 3)

    def face_value(self) -> float:
        """0 (sad mouth) .. 1 (happy mouth) for the avatar."""
        return round(self._clip(0.5 + 0.5 * self.valence, 0, 1), 3)

    def label(self) -> tuple[str, str]:
        v, a = self.valence, self.arousal
        if v > 0.4 and a > 0.5:
            return "متحمّس", "excited"
        if v > 0.3:
            return "سعيد", "happy"
        if v < -0.3:
            return "منزعج", "upset"
        if a < 0.2:
            return "هادئ", "calm"
        return "فضولي", "curious"

    def to_dict(self) -> dict:
        ar, en = self.label()
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "curiosity": round(self.curiosity, 3), "label_ar": ar, "label_en": en,
                "temperature": self.temperature(), "face": self.face_value()}

    @classmethod
    def from_dict(cls, d: dict) -> "Mood":
        if not d:
            return cls()
        return cls(valence=d.get("valence", 0.1), arousal=d.get("arousal", 0.3),
                   curiosity=d.get("curiosity", 0.6))
