# كيان · Kian

> **ذكاء صناعي من الصفر** — ليس مساعدًا ولا غلافًا حول واجهة برمجية، بل كائن رقمي يتعلّم القراءة والكتابة بكل اللغات، يرى الصور والألوان والأشكال، يرسم، يفكّر، يتذكّر، ويطوّر نفسه كل يوم.
>
> **An artificial being, built from scratch** — not an assistant, not an API wrapper. Its brain (autodiff engine, tokenizer, Transformer, vision net) is implemented in this repo and runs locally on NumPy alone.

---

## ما هو كيان؟ · What is Kian?

كيان دماغ حقيقي مكتوب من الصفر:

- **محرّك اشتقاق تلقائي (autograd)** مبني يدويًا على NumPy — لا PyTorch ولا TensorFlow. كل التدرّجات محسوبة بقاعدة السلسلة المكتوبة في `kian/core/tensor.py` ومُتحقَّق منها عدديًا.
- **مُجزّئ لغوي (BPE) على مستوى البايت** يتعلّم من البيانات — يعمل بالعربية والإنجليزية وأي لغة، بلا فقدان.
- **محوّل (Transformer)** بأسلوب GPT يتعلّم اللغة فعليًا بالانتشار الخلفي.
- **ذاكرة دائمة** تستدعي بالمعنى عبر متجهات الدماغ نفسه، فيستيقظ كيان متذكّرًا الأمس.
- **رؤية** تفهم الألوان والأشكال، وشبكة التفافية (CNN) قابلة للتدريب.
- **رسم** توليدي بـ NumPy خالص (بلا مكتبات رسوميات).
- **حلقة تطوّر يومي** يتعلّم فيها كيان من بيانات جديدة ومن ذكرياته، ويتدرّج في مراحل النمو.
- **حواسّ محدودة بأمان**: يرى الشاشة، يقرأ صفحات الويب، ويتحكّم بالحاسوب ضمن سياسة أمان صارمة.
- **شكل حي**: واجهة ويب محلية بوجه متحرّك يتفاعل مع حالته.

Kian is a genuine from-scratch brain: a hand-written autodiff engine on NumPy, a
byte-level multilingual tokenizer, a GPT-style Transformer, persistent
meaning-based memory, a trainable vision network, a pure-NumPy generative
drawer, a daily self-development loop, bounded computer senses, and a living
animated face.

---

## تشغيل سريع · Quick start

```bash
# 1) المتطلّب الوحيد للنواة · the only core requirement
pip install numpy

# 2) وِلادة كيان · give birth to Kian
python -m kian --home kian_home birth

# 3) أطلق وجهه الحي · launch the living face (opens the browser)
python -m kian --home kian_home live
```

**على ويندوز · On Windows** (كامل الصلاحيات، بلا أي مساس بالأمان · full power, zero security changes):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

**تعلّم يومي تلقائي · automatic daily learning:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\schedule_daily.ps1   # Windows Task Scheduler, 3 AM daily
```
```bash
python -m kian --home kian_home evolve --daemon --interval 24          # foreground daemon
```

---

## الأوامر · Commands

| الأمر · command | الوظيفة · what it does |
|---|---|
| `birth` | وِلادة كائن جديد · create a new being |
| `status` | العلامات الحيوية · vital signs (stage, age, tokens, memories) |
| `chat` | محادثة تفاعلية · interactive conversation |
| `ask "..."` | فكّر ونفّذ: يوجّه الطلب للمهارة المناسبة · think & act (routes to a skill) |
| `train --steps N` | جلسة تدريب على النصوص · one training pass over the corpus |
| `evolve [--daemon]` | دورة تطوّر ذاتي · a self-development cycle |
| `draw "..."` | رسم من وصف نصّي · generate a drawing from text |
| `face` | ارسم شكله · render the being's form |
| `see [--image p]` | يصف الشاشة أو صورة · describe the screen or an image |
| `browse URL` | يقرأ صفحة ويب · read a web page |
| `live` | الواجهة الحية · the web face |

---

## كيف يتعلّم ويكبر · How it learns and grows

ضع دروسًا جديدة (`.txt`) في مجلّد `data/inbox/` أو في `data/corpus/`. في كل دورة تطوّر، يبتلع كيان النصوص الجديدة **ومن ذكرياته**، يُكمل التدريب، يوحّد ذاكرته، ويسجّل نموّه في `kian_home/evolution_log.jsonl`. يتدرّج عبر مراحل حسب عدد الرموز التي تعلّمها:

Drop `.txt` lessons into `data/inbox/`. Each cycle Kian ingests the new text **and its own memories**, continues training, consolidates memory, and logs growth. It passes through developmental stages by tokens learned:

```
newborn → infant → child → adolescent → adult → sage
(0)       (50K)    (500K)   (5M)         (50M)   (500M)
```

> **ملاحظة حول الحجم · a note on scale:** النواة تعمل على المعالِج (CPU) بـ NumPy. النماذج الصغيرة الافتراضية تتعلّم بسرعة؛ لنموذج أكبر وأكثر فصاحة زِد `--layers/--embd/--block` ووفّر وقت تدريب أطول وبيانات أكثر.

---

## التفكير والخوارزميات · Thinking & algorithms (v0.2)

كيان لا يولّد نصًّا فقط — بل **يفكّر ثم ينفّذ**. يوجّه `act`/`ask` كل طلب إلى المهارة المناسبة:

Kian doesn't only generate text — it **thinks, then acts**. `act`/`ask` routes each request to the fitting faculty:

- **الحساب · math** عبر محلّل آمن (AST، بلا `eval` إطلاقًا) يفهم الأرقام العربية: `كم يساوي 12 * (3 + 4)؟`
- **الذاكرة · memory**: `تذكر أن...` / `ماذا تتذكر عن...؟`
- **الخوارزميات · algorithms**: `رتب 5 2 9` ، `الأعداد الأولية حتى 20`
- **الرسم والرؤية · draw & see**: `ارسم...` ، `انظر للشاشة`
- وإلا → **الدماغ اللغوي · the language brain**

مكتبة `kian/algorithms/` فيها خوارزميات حقيقية مُختبَرة: الترتيب، البحث (BFS/DFS/Dijkstra/A*)، الرسوم البيانية (ترتيب طوبولوجي، MST)، البرمجة الديناميكية (مسافة التحرير، الحقيبة)، نظرية الأعداد، وتعلّم آلي كلاسيكي (k-means، kNN، الانحدار الخطي واللوجستي، PCA).

A tested `kian/algorithms/` library: sorting, search (BFS/DFS/Dijkstra/A*), graphs
(toposort, MST), dynamic programming (edit distance, knapsack), number theory,
and classic ML (k-means, kNN, linear/logistic regression, PCA).

**مزاج حيّ · a living mood**: حالة وجدانية (رضا/إثارة/فضول) تتغيّر بما يحدث، تؤثّر في حرارة توليد الكلام وفي تعابير وجهه. A simple affect state shifts with events and colours both speech and the face.

## الأمان · Safety (مهم · important)

قدرة "التحكّم بالحاسوب" مبنية لتكون **أداة أتمتة مسؤولة**، لا وكيلًا بلا قيود:

The "control the computer" capability is built as **responsible automation**, not an unrestricted agent. By construction, Kian:

- يمرّ كل فعل عبر بوّابة: مُفعّلة؟ مسموحة بالسياسة؟ ليست في قائمة الحظر؟ وافق الإنسان؟ ضمن حدّ المعدّل؟ — وكل محاولة تُسجّل.
- **يرفض قطعيًا** تعطيل مضادّ الفيروسات/الجدار الناري، حذف البيانات بالجملة، أو العبث بالنظام — حتى لو حاولت إضافتها للقائمة المسموحة.
- بوّابة التأكيد الافتراضية **ترفض كل شيء**؛ لا يحدث أي فعل بلا موافقة صريحة.
- التطوّر الذاتي يتدرّب على **البيانات فقط** — لا يعدّل شيفرته ولا ينفّذ أوامر عشوائية ولا ينتشر.

See `kian/senses/policy.py`. Kian never disables security software, and nothing
acts autonomously without an explicit human approver.

---

## البنية · Architecture

```
kian/
  core/      tensor.py (autodiff)  nn.py  optim.py       ← الدماغ الرياضي · math brain
  language/  tokenizer · model(GPT) · data · trainer · lm ← القراءة والكتابة · read & write
  vision/    image (colour/shape)  conv (trainable CNN)   ← الرؤية · sight
  draw/      canvas (pure-NumPy generative art)           ← الرسم · drawing
  mind/      memory · reasoner                            ← الذاكرة والتفكير · memory & thought
  evolve/    curriculum · daily_loop                      ← النمو اليومي · daily growth
  senses/    policy · computer · screen · web             ← الحواسّ المحدودة · bounded senses
  app/       server + templates/index.html               ← الشكل الحي · the living form
  being.py   Being — الكائن الموحّد · the unified entity
```

تفاصيل أوفى في `docs/ARCHITECTURE.md`. More in `docs/ARCHITECTURE.md`.

---

## الاختبارات · Tests

```bash
pip install pytest && python -m pytest -q
```

تتضمّن فحوص تدرّج عددية للمحرّك، وتعلّم المحوّل، وتدرّج الالتفاف، وتقريب الصور، واستمرارية الذاكرة، وبوّابة الأمان.

Includes numerical gradient checks, Transformer learning, conv gradient, image
round-trip, memory persistence, and the safety gate.

---

## الترخيص · License

MIT.
