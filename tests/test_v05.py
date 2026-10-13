"""Tests for v0.5: vision learning from labelled folders, classifier
persistence and integration into Being.see_image, and gradient accumulation."""
import numpy as np
import pytest

from kian.draw.canvas import Canvas
from kian.util.imageio import save_png


def _make_shape_folder(root, n_per=12, size=40):
    """Synthesise a labelled image folder: circles vs squares."""
    rng = np.random.default_rng(0)
    for label in ("circle", "square"):
        d = root / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per):
            c = Canvas(size, size, bg=(15, 15, 15))
            col = [int(x) for x in rng.integers(120, 255, 3)]
            cx, cy = rng.integers(12, size - 12, 2)
            if label == "circle":
                c.circle(cx, cy, rng.integers(6, 10), col)
            else:
                r = rng.integers(6, 10)
                c.rect(cx - r, cy - r, cx + r, cy + r, col)
            save_png(c.array(), d / f"{label}_{i}.png")


def test_vision_learns_from_folder_and_persists(tmp_path):
    from kian.vision.learn import train_from_folder, VisionClassifier, load_folder_dataset
    root = tmp_path / "images"
    _make_shape_folder(root)
    X, y, labels = load_folder_dataset(root)
    assert X.shape[1:] == (3, 32, 32) and labels == ["circle", "square"]
    clf, acc = train_from_folder(root, tmp_path / "vision", epochs=25)
    assert acc > 0.85
    # reload and predict
    clf2 = VisionClassifier.load(tmp_path / "vision")
    assert clf2 is not None and clf2.labels == labels
    from kian.util.imageio import load_image
    img = load_image(root / "circle" / "circle_0.png")
    label, conf = clf2.predict(img)
    assert label in labels and 0.0 <= conf <= 1.0


def test_vision_load_missing_returns_none(tmp_path):
    from kian.vision.learn import VisionClassifier
    assert VisionClassifier.load(tmp_path / "nothing") is None


def test_being_learn_images_and_see(tmp_path):
    from kian.being import Being
    root = tmp_path / "images"
    _make_shape_folder(root, n_per=10)
    b = Being.birth(tmp_path / "b", vocab_size=320, n_layer=1, n_head=2, n_embd=32, block_size=32)
    acc = b.learn_images(str(root), epochs=20)
    assert acc > 0.8
    desc = b.see_image(str(root / "square" / "square_1.png"))
    assert "I think it's a" in desc
    # survives save/reload of the being (classifier lives in home/vision)
    b.save()
    b2 = Being.load(tmp_path / "b")
    assert "I think it's a" in b2.see_image(str(root / "circle" / "circle_2.png"))


def test_gradient_accumulation_matches_mean_gradient():
    """accum_steps=k with batch b must equal one update on the k micro-batches'
    mean gradient — verified by comparing parameter deltas to a manual run."""
    from kian.language.tokenizer import Tokenizer
    from kian.language.lm import LanguageModel
    from kian.language.data import TextDataset
    from kian.language.trainer import Trainer, TrainConfig
    from kian.core.optim import clip_grad_norm_
    np.random.seed(0)
    tok = Tokenizer().train("abc def ghi jkl mno " * 50, vocab_size=300)
    text = "abc def ghi jkl mno " * 80
    ds = TextDataset.from_text(text, tok, block_size=16)

    # run A: trainer with accum=2 for 2 steps (one optimizer update)
    lm_a = LanguageModel.new(tok, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    before = [p.data.copy() for p in lm_a.model.parameters()]
    tr = Trainer(lm_a, TrainConfig(steps=2, batch_size=4, lr=1e-2, min_lr=1e-3, warmup=1,
                                   accum_steps=2, grad_clip=1e9, eval_every=1000, log=None))
    tr.train(ds)
    after_a = [p.data.copy() for p in lm_a.model.parameters()]

    # run B: same init + same batches, manual mean-gradient single update
    lm_b = LanguageModel.new(tok, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    for p, b0 in zip(lm_b.model.parameters(), before):
        p.data = b0.copy()
    rng = np.random.default_rng(1337)
    from kian.core.optim import AdamW, CosineWarmup
    opt = AdamW(lm_b.model.parameters(), lr=1e-2, weight_decay=0.01)
    sched = CosineWarmup(opt, 1e-2, 1, 2, 1e-3)
    lm_b.model.train(); lm_b.model.zero_grad()
    for _ in range(2):
        sched.step()
        x, y = ds.batch(4, rng)
        (lm_b.model.loss(x, y) * 0.5).backward()
    clip_grad_norm_(lm_b.model.parameters(), 1e9)
    opt.step()
    after_b = [p.data for p in lm_b.model.parameters()]
    for a, b in zip(after_a, after_b):
        assert np.allclose(a, b, atol=1e-6)
