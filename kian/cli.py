"""
kian.cli
========
Command-line entry point for Kian. Run ``python -m kian <command>``.

Commands
--------
  birth    create a new being in a home directory
  status   print the being's vital signs
  chat     interactive conversation (REPL)
  train    one supervised training pass over the corpus
  evolve   run a self-development cycle (``--daemon`` to loop daily)
  draw     generate a drawing from a text prompt
  face     render the being's form (face)
  see      describe the screen, or an image with ``--image``
  browse   read a web page
  live     launch the local web UI (the animated face)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_HOME = os.environ.get("KIAN_HOME", "kian_home")


def _load_or_birth(home: str):
    from .being import Being
    return Being.load_or_birth(home)


def cmd_birth(args) -> None:
    from .being import Being
    being = Being.birth(args.home, name=args.name, vocab_size=args.vocab,
                        n_layer=args.layers, n_head=args.heads, n_embd=args.embd,
                        block_size=args.block)
    print(f"🌱 Born: {being}")
    print(f"   home: {Path(args.home).resolve()}")
    print("   Next: `python -m kian --home {h} evolve` to let it start learning."
          .format(h=args.home))


def cmd_status(args) -> None:
    being = _load_or_birth(args.home)
    import json
    v = being.vitals()
    print(json.dumps(v, ensure_ascii=False, indent=2))
    from .evolve.daily_loop import history
    h = history(args.home)
    if h:
        print(f"\nevolution history: {len(h)} day(s); last loss "
              f"{h[-1].get('train_loss')}, stage {h[-1].get('stage_after')}")


def cmd_chat(args) -> None:
    being = _load_or_birth(args.home)
    print(f"💬 {being.identity.name} ({being.identity.name_ar}) — stage {being.stage}. "
          "Type 'exit' to leave, 'save' to persist.\n")
    while True:
        try:
            text = input("you » ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("exit", "quit", "خروج"):
            break
        if text == "save":
            being.save(); print("   (saved)"); continue
        out = being.act(text, max_new_tokens=args.tokens, temperature=args.temp)
        tag = "" if out["action"] == "talk" else f"  \033[2m[{out['action']}: {out['thought']}]\033[0m"
        print(f"{being.identity.name} » {out['answer']}{tag}\n")
    being.save()
    print("   (saved. goodbye)")


def cmd_ask(args) -> None:
    being = _load_or_birth(args.home)
    out = being.act(args.text, max_new_tokens=args.tokens)
    print(f"[{out['action']}] {out['answer']}")
    if out["action"] != "talk":
        print(f"  ↳ {out['thought']}")
    being.save()


def cmd_train(args) -> None:
    from .language.data import TextDataset
    from .language.trainer import TrainConfig, Trainer
    being = _load_or_birth(args.home)
    import glob
    text = "\n".join(Path(p).read_text(encoding="utf-8", errors="replace")
                     for p in sorted(glob.glob(args.corpus)))
    ds = TextDataset.from_text(text or "Kian learns.", being.lm.tokenizer,
                               block_size=being.lm.model.cfg.block_size)
    tr, va = ds.split(0.1)
    Trainer(being.lm, TrainConfig(steps=args.steps, batch_size=args.batch, lr=args.lr)).train(tr, va)
    being.save()
    print(f"✅ trained {args.steps} steps. {being}")


def cmd_evolve(args) -> None:
    from .evolve.daily_loop import Evolver
    being = _load_or_birth(args.home)
    ev = Evolver(being, corpus_glob=args.corpus, inbox_dir=args.inbox, max_seconds=args.max_seconds)
    if args.daemon:
        ev.daemon(interval_hours=args.interval, max_cycles=args.cycles)
    else:
        rec = ev.evolve_once(steps=args.steps)
        import json
        print(json.dumps(rec, ensure_ascii=False, indent=2))


def cmd_draw(args) -> None:
    being = _load_or_birth(args.home) if args.use_being else None
    from .draw.canvas import draw_from_text
    path = draw_from_text(args.prompt, args.out, being=being)
    print(f"🎨 drew -> {path}")


def cmd_face(args) -> None:
    being = _load_or_birth(args.home)
    from .draw.canvas import draw_face
    path = draw_face(being, args.out, mood=args.mood)
    print(f"🫧 form -> {path}  (stage {being.stage})")


def cmd_see(args) -> None:
    if args.image:
        being = _load_or_birth(args.home)
        print(being.see_image(args.image))
        being.save()
    else:
        from .senses.screen import describe_screen
        print(describe_screen())


def cmd_learn_images(args) -> None:
    being = _load_or_birth(args.home)
    print(f"👁️  learning images from {args.folder} ...")
    acc = being.learn_images(args.folder, epochs=args.epochs, log=print)
    being.save()
    print(f"✅ vision trained: {acc * 100:.1f}% accuracy. Saved to {being.home / 'vision'}")


def cmd_browse(args) -> None:
    from .senses.web import read_page
    print(read_page(args.url, max_chars=args.chars))


def cmd_say(args) -> None:
    being = _load_or_birth(args.home)
    res = being.say(args.text)
    being.save()
    if res.get("spoken"):
        print(f"🔊 ({res['backend']}) {res['text']}")
    else:
        print(f"🔇 {res['text']}\n   {res.get('note', '')}")


def cmd_feed(args) -> None:
    from .evolve.feed import feed
    path = feed(args.target, inbox=args.inbox)
    print(f"🍃 fed into inbox -> {path}")
    print("   Kian will learn it on the next `evolve` cycle.")


def cmd_metrics(args) -> None:
    from .evolve.metrics import summary_text
    print(summary_text(args.home))


def cmd_live(args) -> None:
    from .app.server import run_server
    run_server(home=args.home, host=args.host, port=args.port, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kian", description="Kian (كيان) — a from-scratch digital being.")
    p.add_argument("--home", default=DEFAULT_HOME, help="being home directory")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("birth", help="create a new being")
    b.add_argument("--name", default="Kian")
    b.add_argument("--vocab", type=int, default=1024)
    b.add_argument("--layers", type=int, default=4)
    b.add_argument("--heads", type=int, default=4)
    b.add_argument("--embd", type=int, default=128)
    b.add_argument("--block", type=int, default=128)
    b.set_defaults(func=cmd_birth)

    s = sub.add_parser("status", help="print vital signs"); s.set_defaults(func=cmd_status)

    c = sub.add_parser("chat", help="interactive conversation")
    c.add_argument("--tokens", type=int, default=120); c.add_argument("--temp", type=float, default=0.8)
    c.set_defaults(func=cmd_chat)

    a = sub.add_parser("ask", help="one-shot: think and act on a request")
    a.add_argument("text"); a.add_argument("--tokens", type=int, default=100)
    a.set_defaults(func=cmd_ask)

    t = sub.add_parser("train", help="one training pass over the corpus")
    t.add_argument("--corpus", default="data/corpus/*.txt")
    t.add_argument("--steps", type=int, default=500); t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4); t.set_defaults(func=cmd_train)

    e = sub.add_parser("evolve", help="self-development cycle")
    e.add_argument("--corpus", default="data/corpus/*.txt")
    e.add_argument("--inbox", default="data/inbox")
    e.add_argument("--steps", type=int, default=None)
    e.add_argument("--daemon", action="store_true")
    e.add_argument("--interval", type=float, default=24.0)
    e.add_argument("--cycles", type=int, default=None)
    e.add_argument("--max-seconds", dest="max_seconds", type=float, default=900.0)
    e.set_defaults(func=cmd_evolve)

    d = sub.add_parser("draw", help="generate a drawing")
    d.add_argument("prompt"); d.add_argument("--out", default="drawing.png")
    d.add_argument("--use-being", action="store_true")
    d.set_defaults(func=cmd_draw)

    f = sub.add_parser("face", help="render the being's form")
    f.add_argument("--out", default="face.png"); f.add_argument("--mood", type=float, default=0.6)
    f.set_defaults(func=cmd_face)

    v = sub.add_parser("see", help="describe the screen or an image")
    v.add_argument("--image", default=None); v.set_defaults(func=cmd_see)

    li = sub.add_parser("learn-images", help="train vision on labelled image sub-folders")
    li.add_argument("folder"); li.add_argument("--epochs", type=int, default=40)
    li.set_defaults(func=cmd_learn_images)

    w = sub.add_parser("browse", help="read a web page")
    w.add_argument("url"); w.add_argument("--chars", type=int, default=4000)
    w.set_defaults(func=cmd_browse)

    sy = sub.add_parser("say", help="speak text aloud (needs a TTS backend)")
    sy.add_argument("text"); sy.set_defaults(func=cmd_say)

    fd = sub.add_parser("feed", help="feed a file or URL into the inbox to learn")
    fd.add_argument("target"); fd.add_argument("--inbox", default="data/inbox")
    fd.set_defaults(func=cmd_feed)

    mt = sub.add_parser("metrics", help="report the being's growth")
    mt.set_defaults(func=cmd_metrics)

    l = sub.add_parser("live", help="launch the web UI (the face)")
    l.add_argument("--host", default="127.0.0.1"); l.add_argument("--port", type=int, default=7860)
    l.add_argument("--no-browser", action="store_true"); l.set_defaults(func=cmd_live)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n(interrupted)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
