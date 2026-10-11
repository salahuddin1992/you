"""
kian.evolve.curriculum
======================
A developmental curriculum: Kian learns the way a child does — letters first,
then words, then sentences, then meaning.

The curriculum reads the being's current developmental stage and returns
(a) training hyper-parameters appropriate to that stage and (b) stage-matched
"drill" text that is mixed into the real corpus. A newborn drills the alphabet
and counting; a child reads whole sentences. This shapes *what* each day's
training emphasises, so growth is gradual and grounded rather than random.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ARABIC_LETTERS = "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي"
ENGLISH_LETTERS = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
ARABIC_NUMS = "صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة"
ENGLISH_NUMS = "zero one two three four five six seven eight nine ten"
SIMPLE_WORDS_AR = ["ماء", "نور", "بيت", "شمس", "قمر", "كتاب", "قلم", "باب", "يد", "عين",
                   "قلب", "علم", "حب", "سلام", "ورد", "بحر", "جبل", "نجم", "طفل", "أم"]
SIMPLE_WORDS_EN = ["water", "light", "home", "sun", "moon", "book", "pen", "door", "hand",
                   "eye", "heart", "mind", "love", "peace", "rose", "sea", "star", "child"]


@dataclass
class StagePlan:
    stage: str
    block_size: int       # context length to train with this stage
    steps: int            # steps to run this day
    lr: float
    drill_weight: int     # how many times to repeat drills in the mix


STAGE_PLANS: Dict[str, StagePlan] = {
    "newborn":    StagePlan("newborn", 32, 300, 3e-3, 6),
    "infant":     StagePlan("infant", 48, 300, 2e-3, 4),
    "child":      StagePlan("child", 64, 350, 1.5e-3, 2),
    "adolescent": StagePlan("adolescent", 96, 400, 1e-3, 1),
    "adult":      StagePlan("adult", 128, 400, 7e-4, 0),
    "sage":       StagePlan("sage", 128, 400, 5e-4, 0),
}


def plan_for(stage: str, block_cap: int) -> StagePlan:
    p = STAGE_PLANS.get(stage, STAGE_PLANS["child"])
    # never exceed the model's actual block_size
    return StagePlan(p.stage, min(p.block_size, block_cap), p.steps, p.lr, p.drill_weight)


def drills(stage: str) -> str:
    """Stage-appropriate synthetic practice text."""
    letters = f"{ARABIC_LETTERS}\n{ENGLISH_LETTERS}\n"
    nums = f"{ARABIC_NUMS}\n{ENGLISH_NUMS}\n"
    if stage == "newborn":
        words = " ".join(SIMPLE_WORDS_AR[:8] + SIMPLE_WORDS_EN[:8])
        return (letters * 3) + (nums * 2) + words + "\n"
    if stage == "infant":
        pairs_ar = " ".join(f"{w} {w}" for w in SIMPLE_WORDS_AR)
        pairs_en = " ".join(f"the {w}" for w in SIMPLE_WORDS_EN)
        return letters + nums + pairs_ar + "\n" + pairs_en + "\n"
    if stage == "child":
        s_ar = " ".join(f"هذا {w}." for w in SIMPLE_WORDS_AR[:10])
        s_en = " ".join(f"this is a {w}." for w in SIMPLE_WORDS_EN[:10])
        return s_ar + "\n" + s_en + "\n"
    return ""  # adolescent+ learn from real text only


def build_mix(corpus_text: str, memory_text: str, stage: str) -> str:
    """Combine real corpus, the being's own memories, and stage drills."""
    plan = STAGE_PLANS.get(stage, STAGE_PLANS["child"])
    parts = [corpus_text]
    if memory_text.strip():
        parts.append(memory_text)
    d = drills(stage)
    if d and plan.drill_weight:
        parts.append((d + "\n") * plan.drill_weight)
    return "\n".join(p for p in parts if p.strip())
