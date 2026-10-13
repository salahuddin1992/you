"""Tests for v0.4: perplexity + growth summary, voice fallback, and feed."""
import json

import numpy as np
import pytest


def _tiny_being(home):
    from kian.being import Being
    return Being.birth(home, vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=32)


def test_perplexity_is_positive(tmp_path):
    from kian.evolve.metrics import perplexity
    b = _tiny_being(tmp_path / "b")
    ppl = perplexity(b.lm, "الذكاء الصناعي يتعلم. Kian learns every day.")
    assert ppl > 1.0 and np.isfinite(ppl)


def test_growth_summary_parses_log(tmp_path):
    from kian.evolve.metrics import growth_summary, summary_text
    home = tmp_path / "b"
    _tiny_being(home)
    # synthesise an evolution log
    rows = [
        {"day": 1, "train_loss": 6.0, "stage_after": "newborn", "tokens_seen": 10000, "grew": False},
        {"day": 2, "train_loss": 4.5, "stage_after": "infant", "tokens_seen": 60000, "grew": True},
        {"day": 3, "train_loss": 3.8, "stage_after": "infant", "tokens_seen": 110000, "grew": False},
    ]
    with open(home / "evolution_log.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    g = growth_summary(home)
    assert g["days"] == 3
    assert g["improved"] is True
    assert g["stage_path"] == ["newborn", "infant"]
    assert g["total_grow_events"] == 1
    assert "3" in summary_text(home)


def test_growth_summary_empty(tmp_path):
    from kian.evolve.metrics import growth_summary
    _tiny_being(tmp_path / "b")
    assert growth_summary(tmp_path / "b")["days"] == 0


def test_voice_fallback_never_raises():
    from kian.senses.voice import speak, listen
    r = speak("مرحبا test")
    assert isinstance(r, dict) and "spoken" in r and r["text"] == "مرحبا test"
    r2 = speak("")
    assert r2["spoken"] is False
    # listen with no backend/mic must degrade gracefully
    l = listen()
    assert isinstance(l, dict) and "heard" in l


def test_feed_file_and_text(tmp_path):
    from kian.evolve.feed import feed_text, feed_file, feed
    inbox = tmp_path / "inbox"
    p = feed_text("درس تجريبي lesson one", inbox=inbox, name="t1")
    assert (inbox / "" ).parent.exists()
    assert "lesson one" in open(p, encoding="utf-8").read()
    src = tmp_path / "src.txt"
    src.write_text("محتوى الملف file body", encoding="utf-8")
    p2 = feed(str(src), inbox=inbox)
    assert "file body" in open(p2, encoding="utf-8").read()
    # two lessons now waiting
    assert len(list(inbox.glob("*.txt"))) == 2


def test_feed_rejects_missing_file(tmp_path):
    from kian.evolve.feed import feed_file
    with pytest.raises(FileNotFoundError):
        feed_file(tmp_path / "nope.txt", inbox=tmp_path / "inbox")
