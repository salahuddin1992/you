# Kian — Architecture

This document explains how Kian is built, layer by layer, and why. The guiding
constraint: **everything that makes Kian intelligent is implemented in this
repository and runs on NumPy.** No deep-learning framework is used at any layer.

## 1. The math brain — `kian/core`

### `tensor.py` — reverse-mode autodiff
A `Tensor` wraps an `np.ndarray` and records the operation that produced it. Each
op installs a `_backward` closure implementing its vector-Jacobian product.
`Tensor.backward()` topologically sorts the graph and runs the closures in
reverse. Broadcasting gradients are summed back to the original shape by
`_unbroadcast` — the single most error-prone part of any autodiff engine, and
the reason the gradient checks in `tests/test_autograd.py` exist.

Supported ops include `+ - * / @ ** `, `sum/mean/var/max`, `exp/log/sqrt`,
`relu/tanh/sigmoid/gelu/softmax`, a fused numerically-stable `cross_entropy`,
reshape/transpose/swapaxes/getitem (gather), and `cat/stack`. Dtype is preserved
so the engine can be gradient-checked in float64 while training runs in float32.

### `nn.py` — layers
`Module` collects parameters by attribute discovery and provides train/eval mode
and a plain `state_dict`. Layers: `Linear` (He init), `Embedding`, `LayerNorm`,
`Dropout` (inverted), `Sequential`.

### `optim.py` — optimisers
`SGD` (momentum), `Adam`/`AdamW` (decoupled weight decay), `clip_grad_norm_`,
and a `CosineWarmup` schedule.

## 2. Language — `kian/language`

- **`tokenizer.py`** — byte-level BPE trained from data. Because it starts from
  raw UTF-8 bytes, it covers every language with no per-script config and
  round-trips losslessly (byte fallback for unseen characters).
- **`model.py`** — a GPT-style decoder: token + learned positional embeddings,
  pre-norm blocks of causal multi-head attention and a GELU MLP, final LayerNorm,
  and a (weight-tied) output head. Includes temperature/top-k sampling.
- **`data.py` / `trainer.py` / `lm.py`** — batching, the AdamW+cosine training
  loop with validation and gradient-health logging, and a high-level
  `LanguageModel` that binds tokenizer+model and defines the checkpoint format.

## 3. Vision — `kian/vision`

- **`image.py`** — classical perception: from-scratch k-means for dominant
  colours, Sobel edges, flood-fill region counting, bilingual colour naming and
  descriptions.
- **`conv.py`** — a trainable CNN. The only new primitive is `conv2d`, written as
  one custom autodiff op (im2col forward, col2im for the input gradient, matmul
  for the weight gradient). Everything above it (ReLU, global average pool, the
  classifier) reuses verified ops, so the vision net trains by backprop.

## 4. Drawing — `kian/draw`

`canvas.py` is a pure-NumPy raster engine (anti-aliased circles, rings, rects,
triangles, lines, gradients) saved as PNG via the dependency-free codec in
`kian/util/imageio.py`. `draw_from_text` turns a prompt into a deterministic
generative composition; `draw_face` renders the being's "form".

## 5. Mind — `kian/mind`

- **`memory.py`** — persistent episodic + semantic memory. Recall is by meaning:
  every item carries a vector from the model's embeddings, and recall blends
  cosine similarity with mild recency/importance priors. Consolidation drops the
  least useful items when over capacity.
- **`reasoner.py`** — the `Mind` runs perceive → recall → think → reflect: it
  embeds text with the model, retrieves related memories, builds a working-memory
  context, lets the model continue it, and stores the exchange.

## 6. Self-evolution — `kian/evolve`

- **`curriculum.py`** — maps developmental stage to training settings and
  stage-appropriate drills (alphabet/counting for a newborn, sentences later).
- **`daily_loop.py`** — one cycle: ingest corpus + inbox + own memories, pick a
  curriculum, train for a bounded number of steps, consolidate memory, write a
  reflective journal, and append growth metrics. It trains on **data only** —
  never modifying code or executing arbitrary commands — within the being's home.

## 7. Senses — `kian/senses`

- **`policy.py`** — the `SafetyPolicy`: capability switches, a non-configurable
  hard blocklist (data destruction, security disablement, system tampering), a
  default-deny confirmation callback, and a rate limit.
- **`computer.py`** — the bounded action interface (open URL, type, click, run
  allow-listed commands with `shell=False`), every call gated and audited.
- **`screen.py` / `web.py`** — read-only perception: screenshot description and
  HTML-to-text web reading.

## 8. The being & its form — `kian/being.py`, `kian/app`

`Being` is the unified entity: identity, vital signs, developmental stage, and a
life-cycle (birth → perceive/think/speak/see/draw → sleep/consolidate → grow),
persisted to a home directory. `app/server.py` + `templates/index.html` serve a
local animated face that reacts to the being's state.

## 9. Algorithms & thinking (v0.2) — `kian/algorithms`, `kian/mind`

- **`algorithms/`** — a from-scratch, tested library the being can use as tools:
  `sort` (quick/merge/heap/insertion), `search` (binary, BFS, DFS, Dijkstra,
  A*), `graph` (toposort, components, cycles, Kruskal MST), `dp` (edit distance,
  LCS, knapsack, LIS, coin change), `numbers` (gcd/lcm, sieve, factorisation,
  modpow, extended Euclid), and `ml` (k-means++, kNN, linear/logistic
  regression, PCA) — classic ML complementing the neural brain.
- **`mind/tools.py`** — a safe arithmetic calculator: it parses to an AST and
  walks a whitelist (numbers, operators, a fixed function set, pi/e). It never
  uses `eval`/`exec` and rejects names, attributes, calls to anything
  un-whitelisted, comprehensions and lambdas. Arabic-Indic digits are
  normalised. `extract_expression` pulls an expression out of a sentence.
- **`mind/mood.py`** — a 3-scalar affect state (valence/arousal/curiosity) that
  reacts to events, decays to baseline, and maps to sampling temperature and the
  face's expression.
- **`mind/planner.py`** — the intent router. `Being.act` calls it to decide
  between calculator, memory store/recall, algorithm tools, vision, drawing, or
  the language model, returning the action taken and a one-line thought trace.

## Verification

`tests/` promotes the engineering checks into a suite: float64 gradient checks
for the autodiff engine and `conv2d`, Transformer overfitting, multilingual
tokenizer round-trips, PNG round-trip, memory persistence, and the safety gate.
