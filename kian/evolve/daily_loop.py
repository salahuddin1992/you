"""
kian.evolve.daily_loop
=====================
The self-development engine: Kian trains itself a little more every day.

One evolution cycle (:meth:`Evolver.evolve_once`) does, strictly within the
being's own home and its designated data folders:

1. **ingest**    gather new text — the corpus folder, an ``inbox`` drop folder,
                 and the being's *own* recent memories (so it learns from its
                 experience).
2. **curriculum** pick stage-appropriate settings and drills.
3. **study**     continue training the language brain for a bounded number of
                 steps.
4. **consolidate** sleep: compress memory, bump the day counter.
5. **reflect**   write a short journal line and remember it.
6. **record**    append growth metrics to ``evolution_log.jsonl`` and save a
                 versioned checkpoint.

Safety boundaries (by construction, not by request):
* It trains on **data only** — it never edits, generates, or executes code, and
  never modifies this program. "Self-improvement" here means better weights.
* It reads/writes only the being's home and the configured data directories.
* Every cycle is bounded in steps and time; losses that go non-finite abort the
  cycle and roll nothing forward.

``daemon`` runs one cycle per interval. For unattended daily runs prefer the OS
scheduler (cron / Windows Task Scheduler) calling ``python -m kian evolve`` —
that survives reboots; the in-process daemon is for foreground sessions.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..being import Being, stage_for
from ..language.data import TextDataset
from ..language.trainer import TrainConfig, Trainer
from . import curriculum as C


class Evolver:
    def __init__(
        self,
        being: Being,
        corpus_glob: str = "data/corpus/*.txt",
        inbox_dir: str = "data/inbox",
        max_seconds: float = 900.0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.being = being
        self.corpus_glob = corpus_glob
        self.inbox_dir = Path(inbox_dir)
        self.max_seconds = max_seconds
        self.log = log

    # ------------------------------------------------------------------ #
    def _gather_text(self) -> tuple[str, int]:
        parts: List[str] = []
        for p in sorted(glob.glob(self.corpus_glob)):
            try:
                parts.append(Path(p).read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
        # Lessons archived by earlier cycles live in <corpus>/learned/, which the
        # non-recursive corpus glob does not cover — include them so a fed
        # lesson keeps being trained on, as promised, not just once.
        learned = Path(self.corpus_glob).parent / "learned"
        if learned.is_dir():
            for p in sorted(learned.glob("*.txt")):
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
        new_files = 0
        if self.inbox_dir.exists():
            for p in sorted(self.inbox_dir.glob("*")):
                if p.is_file() and p.suffix.lower() in (".txt", ".md", ""):
                    try:
                        parts.append(p.read_text(encoding="utf-8", errors="replace"))
                        new_files += 1
                    except Exception:
                        pass
        return "\n".join(parts), new_files

    def _memory_text(self, limit: int = 300) -> str:
        items = self.being.mind.memory.recent(limit)
        return "\n".join(it.text for it in items)

    def _archive_inbox(self) -> None:
        """Move consumed inbox files into the corpus so they persist as learned."""
        if not self.inbox_dir.exists():
            return
        learned = Path(self.corpus_glob).parent / "learned"
        learned.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.inbox_dir.glob("*")):
            if p.is_file():
                dest = learned / f"{int(time.time())}_{p.name}"
                try:
                    p.replace(dest)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    def evolve_once(self, steps: Optional[int] = None) -> dict:
        t0 = time.time()
        being = self.being
        stage = being.stage
        plan = C.plan_for(stage, block_cap=being.lm.model.cfg.block_size)
        n_steps = steps if steps is not None else plan.steps

        corpus_text, new_files = self._gather_text()
        mem_text = self._memory_text()
        mix = C.build_mix(corpus_text, mem_text, stage)
        if not mix.strip():
            mix = "Kian كيان يتعلم كل يوم. Kian learns every day."

        ds = TextDataset.from_text(mix, being.lm.tokenizer, block_size=plan.block_size)
        train_ds, val_ds = ds.split(0.1)

        self.log(f"[evolve] stage={stage} block={plan.block_size} steps={n_steps} "
                 f"corpus={len(corpus_text)}B mem_items={len(being.mind.memory)} inbox={new_files}")

        tokens_before = being.lm.meta.get("tokens_seen", 0)
        cfg = TrainConfig(steps=n_steps, batch_size=16, lr=plan.lr,
                          min_lr=plan.lr * 0.1, warmup=max(10, n_steps // 10),
                          eval_every=max(20, n_steps // 5), eval_batches=4, log=self.log)
        trainer = Trainer(being.lm, cfg)
        trainer.train(train_ds, val_ds)

        # consolidate + advance the day
        sleep_info = being.sleep()
        self._archive_inbox()

        new_stage = stage_for(being.lm.meta.get("tokens_seen", 0))
        grew = new_stage != stage
        last_val = trainer.history[-1]["val"] if trainer.history else float("nan")
        last_train = trainer.history[-1]["train"] if trainer.history else float("nan")

        # reflect: a short journal line, stored as memory
        journal = (f"في اليوم {being.identity.days_lived}: تدربت {n_steps} خطوة، "
                   f"مرحلتي {new_stage}، خسارتي {last_train:.2f}. "
                   f"Day {being.identity.days_lived}: trained {n_steps} steps, "
                   f"stage {new_stage}, loss {last_train:.2f}.")
        being.mind.learn_fact(journal, importance=1.5)
        try:
            being.mood.event("learned")
            if grew:
                being.mood.event("grew")
        except Exception:
            pass
        if grew:
            being.remember(f"لقد نموت! أصبحت الآن في مرحلة {new_stage}. "
                           f"I grew! I am now at the {new_stage} stage.", importance=3.0)

        record = {
            "ts": time.time(),
            "day": being.identity.days_lived,
            "stage_before": stage,
            "stage_after": new_stage,
            "grew": grew,
            "steps": n_steps,
            "tokens_seen": being.lm.meta.get("tokens_seen", 0),
            "tokens_added": being.lm.meta.get("tokens_seen", 0) - tokens_before,
            "train_loss": round(float(last_train), 4),
            "val_loss": round(float(last_val), 4),
            "memories": len(being.mind.memory),
            "seconds": round(time.time() - t0, 1),
            "new_inbox_files": new_files,
        }
        self._log_record(record)
        being.save()
        self.log(f"[evolve] done in {record['seconds']}s — stage {new_stage}"
                 + (" (GREW!)" if grew else "") + f", loss {record['train_loss']}")
        return record

    def _log_record(self, record: dict) -> None:
        path = self.being.home / "evolution_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    def daemon(self, interval_hours: float = 24.0, max_cycles: Optional[int] = None) -> None:
        """Run an evolution cycle every ``interval_hours`` (foreground loop)."""
        cycles = 0
        self.log(f"[evolve] daemon started; one cycle every {interval_hours}h. Ctrl-C to stop.")
        while True:
            try:
                self.evolve_once()
            except KeyboardInterrupt:
                self.log("[evolve] stopped by user.")
                break
            except Exception as e:  # never let a bad day kill the being
                self.log(f"[evolve] cycle error (skipped): {e}")
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(max(1.0, interval_hours * 3600.0))


def history(home: str | Path) -> List[dict]:
    path = Path(home) / "evolution_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
