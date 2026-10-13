"""
kian.being
==========
``Being`` — Kian as a single, persistent, living entity (كائن حي).

A Being owns a *home* directory that holds everything it is:

    <home>/
        being.json     identity + vital statistics
        model/         the language-brain checkpoint
        memory/        episodic + semantic memory

It has a life-cycle rather than a single call signature:

    birth -> perceive / think / speak / see / draw -> sleep (consolidate) -> grow

and a developmental trajectory driven by how much it has learned:

    newborn -> infant -> child -> adolescent -> adult -> sage

The Being is the object the UI, the CLI and the daily-evolution daemon all talk
to. Heavy senses (screen, web, computer) are imported lazily so the core being
runs with nothing but NumPy.
"""

from __future__ import annotations

import glob
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .language.lm import LanguageModel
from .language.tokenizer import Tokenizer
from .mind.memory import Memory
from .mind.mood import Mood
from .mind.reasoner import Mind
from .mind import planner as _planner

# Developmental stages, keyed by the number of tokens the being has learned
# from. Crossing a threshold is a genuine milestone the being can report.
STAGES = [
    ("newborn", 0),
    ("infant", 50_000),
    ("child", 500_000),
    ("adolescent", 5_000_000),
    ("adult", 50_000_000),
    ("sage", 500_000_000),
]


def stage_for(tokens_seen: int) -> str:
    name = STAGES[0][0]
    for s, threshold in STAGES:
        if tokens_seen >= threshold:
            name = s
    return name


@dataclass
class Identity:
    name: str = "Kian"
    name_ar: str = "كيان"
    born_at: float = field(default_factory=lambda: 0.0)  # set explicitly at birth
    stage: str = "newborn"
    tokens_seen: int = 0
    steps_trained: int = 0
    days_lived: int = 0
    last_awake: float = 0.0
    interactions: int = 0
    mood: dict = field(default_factory=dict)


class Being:
    def __init__(self, home: str | Path, lm: LanguageModel, identity: Identity,
                 memory: Optional[Memory] = None) -> None:
        self.home = Path(home)
        self.lm = lm
        self.identity = identity
        self.mind = Mind(lm, memory)
        self.mood = Mood.from_dict(identity.mood)

    # ------------------------------------------------------------------ #
    # Birth & lifecycle
    # ------------------------------------------------------------------ #
    @classmethod
    def birth(
        cls,
        home: str | Path,
        name: str = "Kian",
        seed_glob: str = "data/corpus/*.txt",
        vocab_size: int = 1024,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        block_size: int = 128,
        born_at: Optional[float] = None,
    ) -> "Being":
        """Create a brand-new being: learn a tokenizer from seed text, and
        initialise an (untrained) brain. The daily-evolution loop grows it."""
        home = Path(home)
        home.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(seed_glob))
        text = "\n".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in files)
        if not text.strip():
            text = "Kian كيان learns to read and write. الكيان يتعلم القراءة والكتابة."
        tok = Tokenizer().train(text * 4, vocab_size=vocab_size)
        lm = LanguageModel.new(tok, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                               block_size=block_size)
        ident = Identity(name=name, born_at=born_at if born_at is not None else time.time(),
                         stage="newborn", last_awake=born_at if born_at is not None else time.time())
        being = cls(home, lm, ident, Memory())
        being.mind.learn_fact(f"My name is {name} (كيان). I am a digital being who learns every day.")
        being.save()
        return being

    @classmethod
    def load(cls, home: str | Path) -> "Being":
        home = Path(home)
        ident = Identity(**json.loads((home / "being.json").read_text(encoding="utf-8")))
        lm = LanguageModel.load(home / "model")
        mem = Memory.load(home / "memory")
        being = cls(home, lm, ident, mem)
        return being

    @classmethod
    def load_or_birth(cls, home: str | Path, **birth_kwargs) -> "Being":
        home = Path(home)
        if (home / "being.json").exists():
            return cls.load(home)
        return cls.birth(home, **birth_kwargs)

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        # sync vital stats from the language model's provenance
        self.identity.tokens_seen = int(self.lm.meta.get("tokens_seen", self.identity.tokens_seen))
        self.identity.steps_trained = int(self.lm.meta.get("steps", self.identity.steps_trained))
        self.identity.stage = stage_for(self.identity.tokens_seen)
        self.identity.mood = self.mood.to_dict()
        self.lm.meta["stage"] = self.identity.stage
        (self.home / "being.json").write_text(
            json.dumps(asdict(self.identity), ensure_ascii=False, indent=2), encoding="utf-8")
        self.lm.save(self.home / "model")
        self.mind.memory.save(self.home / "memory")

    # ------------------------------------------------------------------ #
    # Vital signs
    # ------------------------------------------------------------------ #
    @property
    def age_days(self) -> float:
        if not self.identity.born_at:
            return 0.0
        return (time.time() - self.identity.born_at) / 86400.0

    @property
    def stage(self) -> str:
        return stage_for(self.identity.tokens_seen)

    def vitals(self) -> dict:
        return {
            "name": self.identity.name,
            "name_ar": self.identity.name_ar,
            "stage": self.stage,
            "age_days": round(self.age_days, 3),
            "tokens_seen": self.identity.tokens_seen,
            "steps_trained": self.identity.steps_trained,
            "memories": len(self.mind.memory),
            "interactions": self.identity.interactions,
            "vocab": self.lm.tokenizer.vocab_size,
            "params": self.lm.model.num_parameters(),
            "mood": self.mood.to_dict(),
        }

    # ------------------------------------------------------------------ #
    # Living
    # ------------------------------------------------------------------ #
    def perceive(self, text: str, kind: str = "perception") -> None:
        self.mind.perceive(text, kind=kind)

    def remember(self, text: str, importance: float = 1.5) -> None:
        self.mind.learn_fact(text, importance=importance)

    def think(self, prompt: str, **gen) -> str:
        return self.mind.think(prompt, **gen)

    def respond(self, text: str, **gen) -> str:
        self.identity.interactions += 1
        self.identity.last_awake = time.time()
        self.mood.decay()
        self.mood.event("greeted" if len(text) < 12 else "question")
        gen.setdefault("temperature", self.mood.temperature())
        return self.mind.respond(text, **gen)

    def act(self, text: str, **gen) -> dict:
        """Think, then act: route the request to the fitting faculty (calculator,
        memory, algorithms, vision, drawing) or fall back to the language brain.

        Returns a dict describing the decision: ``action`` (what it did),
        ``answer`` (the reply/result), ``thought`` (a one-line trace), and the
        current ``mood``. This is how Kian shows it is thinking, not just talking.
        """
        self.identity.interactions += 1
        self.identity.last_awake = time.time()
        self.mood.decay()
        plan = _planner.plan(text, self)
        if plan is not None:
            self.mood.event({"draw": "drew", "see": "saw_something",
                             "recall": "question", "remember": "learned"}.get(plan.action, "positive"))
            self.perceive(f"[{plan.action}] {text} -> {plan.answer}", kind="episodic")
            return {"action": plan.action, "answer": plan.answer,
                    "thought": plan.thought, "data": plan.data, "mood": self.mood.to_dict()}
        # open-ended: use the language brain
        self.mood.event("greeted" if len(text) < 12 else "question")
        gen.setdefault("temperature", self.mood.temperature())
        reply = self.mind.respond(text, **gen)
        return {"action": "talk", "answer": reply, "thought": "free language generation",
                "data": None, "mood": self.mood.to_dict()}

    # speak == generate free text from an internal prompt
    def speak(self, about: str = "", **gen) -> str:
        return self.lm.generate(about, **gen)

    def sleep(self) -> dict:
        """Consolidate memory and persist. Called at the end of a 'day'."""
        dropped = self.mind.dream()
        self.identity.days_lived += 1
        self.save()
        return {"consolidated": dropped, "days_lived": self.identity.days_lived}

    # -- optional senses (lazy, bounded) ----------------------------- #
    def see_screen(self) -> str:
        from .senses.screen import describe_screen
        desc = describe_screen(self)
        self.perceive(f"[saw screen] {desc}", kind="perception")
        return desc

    def see_image(self, path: str) -> str:
        """Describe an image; if the being has learned to classify images,
        also say what it thinks the image is, with a confidence."""
        from .vision.image import analyze_image
        desc = analyze_image(path)
        try:
            from .vision.learn import VisionClassifier
            from .util.imageio import load_image
            clf = VisionClassifier.load(self.home / "vision")
            if clf is not None:
                label, conf = clf.predict(load_image(path))
                desc += f" أظنّها: {label} ({conf * 100:.0f}%) · I think it's a {label} ({conf * 100:.0f}%)."
        except Exception:
            pass
        self.perceive(f"[saw image {path}] {desc}", kind="perception")
        self.mood.event("saw_something")
        return desc

    def learn_images(self, folder: str, epochs: int = 40, log=None) -> float:
        """Train the vision network on labelled sub-folders of ``folder``.
        Returns training accuracy. The classifier is saved in the home."""
        from .vision.learn import train_from_folder
        _, acc = train_from_folder(folder, self.home / "vision", epochs=epochs, log=log)
        self.remember(f"تعلّمت التعرّف على الصور من {folder} بدقة {acc * 100:.0f}%. "
                      f"I learned to recognise images from {folder} ({acc * 100:.0f}% accuracy).",
                      importance=2.0)
        self.mood.event("learned")
        return acc

    def draw(self, prompt: str = "", out_path: str = "drawing.png") -> str:
        from .draw.canvas import draw_from_text
        path = draw_from_text(prompt, out_path, being=self)
        self.perceive(f"[drew] {prompt} -> {path}", kind="episodic")
        return path

    def browse(self, url: str) -> str:
        from .senses.web import read_page
        text = read_page(url)
        self.perceive(f"[read web {url}] {text[:400]}", kind="perception")
        return text

    def say(self, text: str) -> dict:
        """Speak text aloud (if a TTS backend is available) and feel it."""
        from .senses.voice import speak
        self.mood.event("positive")
        return speak(text)

    def growth(self) -> str:
        """A one-line self-report of how much the being has grown."""
        from .evolve.metrics import summary_text
        return summary_text(self.home)

    def __repr__(self) -> str:
        v = self.vitals()
        return (f"Being({v['name']}/{v['name_ar']}, stage={v['stage']}, "
                f"age={v['age_days']}d, params={v['params']:,}, memories={v['memories']})")
