# VQA pipeline design (v2)

The design of record for the AIC Q&A task. Built on the July-2026
evidence in note [17](17-vqa-research-2026-07.md) — including its
second sweep (§4b) — and on one strategic observation the literature
does not exploit: **generic long-video QA systems must find evidence at
question time; a competition team owns the corpus for weeks before the
round.** Every idea below moves work from the 5-minute round into
unlimited offline time, or converts an unreliable VLM judgment into a
deterministic lookup or a tool measurement. v2 supersedes the v1
locate->read->vote sketch (kept as the spine of Track B below).

## 0. Design theses

1. **Answers live in text first.** This is a *news* corpus:
   NewsVideoQA/ViTexQA (note 17 §4b) show on-screen text and speech
   carry most news answers, and Vortex won "outstanding QA" at AIC 2025
   on retrieval + metadata, not on a video reasoner. Our Chronicle
   already stores that text per shot.
2. **VLMs must not count, read tiny text, or guess timelines.**
   CountQA/PushupBench prove counting fails; a detector-counter (SAM
   3.1 default) counts; OCR reads; the Chronicle's timeline aggregates.
   The VLM routes,
   adjudicates, and handles the open-ended remainder.
3. **The round is an expected-value game, not a benchmark.** DRES
   allows resubmission and rewards speed; the optimal policy is a race
   between a fast cheap track and a slow careful track with the
   operator arbitrating — not a single best-effort answer.

## 1. Contract (unchanged)

Vietnamese question anchored by a scene description; submit **video id
+ frame + short textual answer**, judged on all three, ~5 minutes,
resubmission allowed (note 01 §2).

## 2. Offline: the Answer Ledger (new stage, the flagship)

Extend the Chronicle with a fourth extractor pass producing one typed
**factoid record per shot** — the pre-computed answer space:

```
ledger.jsonl, one row per shot:
  people_count      int|null     "person" instances from the detector-
                                 counter backend on the shot's keyframes
  salient_objects   [{concept, count}]   the backend over a config concept
                                          list (vehicles, animals, flags, ...)
  dominant_colors   [{object, color}]    VLM lite pass on keyframes
  screen_text       (already in Chronicle OCR; reading-order sorted,
                     the ViTextVQA token-order lesson)
  spoken_entities   [names/places/orgs]  flattened from the Chronicle's
                     ChronicleRecord.entities (persons/orgs/places), now
                     populated by a language-general NER stage
  scene_time        day|night|indoor|outdoor   from captions
```

**`spoken_entities` reuses the Chronicle's existing (but so-far
deferred) `ChronicleRecord.entities` field** rather than a parallel
store: a new resumable + multi-GPU NER stage (mirroring the ASR/OCR
stages) fills `entities`, the assembler joins it, and the ledger
flattens `persons+orgs+places` into `spoken_entities`. The NER backend
is **config-selectable and language-general** (note 17 §4b — the corpus
is bilingual, so no VN-only tool): `llm` (an EndpointPool call, highest
accuracy, needs a server) or `gliner` (local GLiNER-multi, zero-shot,
multilingual, CPU, Apache-2.0), with `auto` preferring `llm` when an
endpoint is configured and falling back to `gliner`.

The count/segmentation fields come from a **config-selectable
detector-counter backend** (note 17 §4b), not one hardcoded model —
default **SAM 3.1** (masks + count + the zoom crop in one gated HF
dependency; halved VRAM vs SAM 3 so a T4 can host it), with **CountVid**
(video-native unique-instance counting, no double-count),
**CountGD++** (dense still frames), and **RT-Counter** (real-time
online) as challengers chosen by the phase-4 ablation.

Then two derived indexes:

- **Factoid index**: the ledger rows joined into the existing text
  stack (sparse + dense) so "đội nào vô địch", "áo màu đỏ", "3 ca sĩ"
  match ledger text directly.
- **Timeline per video**: the ordered scene_time / entity / count
  sequence — the substrate for cumulative questions ("bao nhiêu đêm
  trôi qua...", the note-01 example), which NO single-moment reader can
  answer.

Cost containment: the ledger pass is resumable like every stage, runs
on the same data-parallel machinery, and every field is optional — a
profile can enable only `screen_text`+`spoken_entities` (free: derived
from existing Chronicle) and skip the detector-counter backend
entirely. Evidence hooks: HiMu computes
exactly these expert signals *online per question*; we bake them
offline once. Deep Video Discovery's "multi-granular video database"
becomes queryable at the *fact* granularity, not just the shot.

## 3. Online: the Answer Compiler and the two-track race

### 3.1 QA compile (Cortex QA mode) -> an answer *program*

The Cortex classifies the question into an archetype and emits a typed
plan — not just clauses:

| Archetype | Program |
| --- | --- |
| count ("bao nhiêu...") | locate moments -> detector-counter backend (SAM 3.1 default) on zoomed frames -> count instances -> VLM adjudicates occlusion/borderline only |
| read-text ("chữ/biển/tên trên màn hình...") | literal-index lookup first (the answer is usually already OCR'd); VLM crop-read as fallback |
| entity ("ai/đội nào/ở đâu...") | spoken_entities + OCR lookup first (news *says and captions* names); visual read as fallback |
| attribute ("màu gì/loại gì...") | ledger colors -> VLM read with spatial crop |
| cumulative/temporal ("mấy lần/mấy đêm... cho đến...") | timeline scan over the ledger up to the anchor moment; never a single-moment read |
| open/other | plain VLM read (Track B) |

Fallback on any compile failure: archetype `other`, whole question as
locator — the v1 behaviour, nothing breaks.

### 3.2 Track A — lookup (sub-second)

Run the locator through the existing KIS fusion AND the question terms
through the factoid/literal indexes. Where both agree on a moment, the
ledger row often *already contains the answer field*. The console
shows these instantly as "answer cards with provenance" (the OCR line,
the ledger count, the ASR sentence — the operator can eyeball-verify
in seconds). For many news questions the round ends here.

### 3.3 Track A' — answer-first reverse retrieval (our Imagine->Match twist)

When the locator is weak but the answer space is enumerable, invert
the direction: the LLM *hypothesises* candidate answers ("which team
won the final" -> plausible team names from spoken_entities of
final-related shots), and each hypothesis is searched as a query in
the OCR/ASR/factoid indexes — the answer that retrieves a coherent
moment wins, and that moment is the frame we submit. This is the
Imagine->Match idea (note 07) transplanted from pixels to answers:
generate the target, then match it against the corpus. No paper in
note 17 does this for interactive VQA.

### 3.4 Track B — grounded read (the v1 spine, seconds to a minute)

For top-M candidate moments: per-candidate VLM read over an evidence
pack (keyframes centred on the moment ± window + shot ASR/OCR/caption,
OCR in reading order), JSON reply {answerable, answer,
evidence_timestamp_ms, confidence}. Reuses the vlm_verify endpoint
machinery verbatim; per-candidate calls (never listwise — the local
context-overflow lesson).

### 3.5 Verification stack (cheap, ordered)

1. **Type validation**: a count must parse as a number, a colour must
   normalise to the Vietnamese colour lexicon — malformed reads are
   demoted before any model votes.
2. **Cross-candidate agreement** (LongVidSearch): identical normalised
   answers from independently retrieved moments pool their scores.
   Normalisation is Vietnamese-aware: digits <-> number words ("3" ==
   "ba"), diacritic-insensitive, unit-stripped.
3. **Cross-track agreement**: Track A and Track B agreeing is the
   strongest signal in the system and auto-promotes the answer to the
   submit box.
4. **One disconfirmation call** (CoVe/CASHEW-shaped, budget-aware per
   2510.14913): "does this evidence contradict answer X — yes/no+why"
   against the runner-up moment's pack. Cheaper than more samples, and
   it catches the plausible-answer-wrong-moment failure that
   three-way DRES judging punishes.

### 3.6 Submit policy (EV, not accuracy-max)

Show the best card the moment it exists; the operator submits early
and cheap (Track A) because a DRES reject costs less than 60 seconds
of waiting — then Track B and the verification stack refine in the
background and the next answer group is one keystroke away (the KIS-C
narrowing pattern). The QPP hint extends to QA: high-agreement ->
"submit now", low -> "wait for Track B / zoom".

### 3.7 Escalations (operator-triggered, hard timeouts, degrade-to-unchanged)

- **Iterative perception zoom** (VTTS, note 17 §4b): re-read with
  temporally denser frames AND spatially cropped to the detector-
  counter mask/box of the concept in question — look again, don't think
  longer.
- **Ledger drill-down**: open the raw ledger/OCR/ASR rows of the
  candidate shot in the console (instant, no model).
- **Last-resort long read**: ship the whole suspect *video* (not
  corpus) to a long-context reader on the API profile.

### 3.8 Streaming delivery (service.streaming, additive)

The submit policy in §3.6 — "show the best card the moment it exists" — is only
real if the transport hands the operator Track A *before* Track B finishes. The
one-shot `/api/qa` returns everything at once, so the reply is gated by the
slowest stage (Track B reads / the count detector). The additive `/api/qa/stream`
and `/api/search/stream` routes fix that: same answer, incremental delivery.

- **What ships when.** `/api/qa/stream` emits `track_a` (plan + cards) first,
  then `reverse`, `count`, and `reads` (voted groups) as each finishes, then the
  settled `hint`. `/api/search/stream` emits the fused ranking as a read-only
  `results_preview`, then the reranked `results`. Each line is one NDJSON event
  keyed by `event`; a stage that does not fire simply never emits.
- **Same answer, different delivery.** Every stage delegates to the same helper
  the one-shot route uses (run off the event loop via `run_in_threadpool`), and
  the route tests assert the streamed `answer_groups` / `results` are
  byte-identical to the one-shot route's. Streaming is a transport, not a second
  code path.
- **Sequential, not concurrent.** Stages emit in a fixed order rather than
  as-finished. The guaranteed win — first byte = Track A / fused rank for *any*
  consumer, not just the console — needs no concurrency; overlapping the two
  non-contending stages (local reverse ∥ networked Track B) is a measured
  follow-up, deliberately not taken because yielding across an `anyio` cancel
  scope is fragile on client disconnect (see `implementation-notes.md`,
  2026-07-19).
- **Sessions stay authoritative.** A KIS-C session omits the fused preview:
  narrowing / `result_version` must happen exactly once (in the shared `_respond`
  tail), so a session search streams only the one authoritative `results` event.
- **Off by default, degrade-clean.** `service.streaming: false` does not register
  the `/stream` routes; the console tries them and falls back to the one-shot
  route on 404, so the operator UI works either way.

### 3.9 Post-audit wiring notes (2026-07-19)

The full-pipeline audit (implementation-notes, 2026-07-19) wired the pieces
this note promised but the code had left dangling, so the design above now
matches the code exactly:

- **The factoid index is queried.** The QA locator's text clauses
  (entity_terms / ocr_literals / asr_phrases) and every reverse-retrieval
  hypothesis probe the Answer Ledger factoid channel at ``vqa.factoid_weight``
  (default 1.0). KIS dispatch never touches it — proven by tests — so the §2
  "pre-computed answer space" is live for QA without any KIS ranking change.
- **The concentration gate discriminates.** Gate A's peakiness is computed
  over the top ``reverse.gate_top_m`` hits (default 8): over a full RRF tail
  max/median is ~2.6 for ANY hit list, which silently waved every hypothesis
  through the "double gate" of §3.3.
- **The vote's agreement threshold surfaces.** A group with
  ``vote_min_agree``+ agreeing moments carries ``confident: true`` and the
  console marks it "(agreed)" — the §3.4 cross-candidate signal an operator
  can act on before the EV hint.
- **The EV budget is spent.** ``ev_hint.max_early_submits`` counts submit_now
  hints per question (a new question resets it), the §3.6 budget.
- **The count floor and crop pad are config.** ``count.min_mask_confidence``
  filters detections at query time (it was previously documented but unread);
  ``count.crop_pad_frac`` replaces a hardcoded 0.15.

## 4. Config sketch

```yaml
paths:
  hf_token_env: HF_TOKEN     # NAME of the env var holding the HF access
                             # token for gated weights (SAM 3.1)
  hf_token: null             # OR the token value inline for convenience
                             # — if set, GITIGNORE this file, never push
                             # it (committed profiles keep this null)
chronicle:
  ledger:                    # offline Answer Ledger pass
    enabled: false
    fields: [screen_text, spoken_entities]   # free tier; add
             # people_count, salient_objects, dominant_colors, scene_time
    entity_backend: auto     # auto | llm | gliner  (auto = llm if an
                             # entity endpoint is configured, else gliner)
    entity:                  # the gliner local backend (language-general)
      model_id: urchade/gliner_multi-v2.1   # multilingual, Apache-2.0
      device: auto
      labels: [person, organization, location]   # zero-shot NER labels
      threshold: 0.5
    concepts: [person, car, flag, ...]       # detector prompt list
    counter:                 # only if count/mask fields on
      backend: sam3.1        # sam3.1 | sam3 | countvid | countgd++ | rt-counter
      model_id: ...
      device: auto
      max_batch: ...         # cap for a 16 GB T4 (fp16 batch OOMs high)
vqa:
  enabled: false             # off = the KIS system, bit-identical
  archetypes: [count, read_text, entity, attribute, cumulative, other]
  top_m: 8
  frames_per_candidate: 4
  zoom_window_ms: 10000
  vote_min_agree: 2
  disconfirm: true           # the single verification call
  reverse_retrieval: true    # Track A' on weak-locator questions
  timeout_s: 30.0
  # reader + adjudicator endpoints: MultiEndpointFields, so the local
  # 2B drafts and (optionally) a stronger API model verifies (cascade).
```

## 5. Profiles

| Profile | Reader / tools | Notes |
| --- | --- | --- |
| t4-colab-api | Gemini reader + free-tier ledger | Track A carries most weight; API verifies |
| t4-colab-local | Qwen3.5-2B reader; ledger text fields only | counting via SAM 3.1 if installed (fits the T4 with a capped batch), else archetype degrades to read |
| GPU server | Qwen3-VL-8B reader + full ledger incl. SAM 3.1 | the target competition rig; CountVid/CountGD++ selectable per the phase-4 ablation |

Reader and detector-counter are both **config-selectable backends**,
not architecture: Gemini / Qwen3-VL / Qwen3.5 for the reader (note 17
§5), and SAM 3.1 / SAM 3 / CountVid / CountGD++ / RT-Counter for the
counter (note 17 §4b) — a preferred default never removes the others.
Gated weights (SAM 3.1) need an HF access token: name an env var via
`paths.hf_token_env`, or set `paths.hf_token` inline for convenience
in a **gitignored** config that is never pushed.

## 6. Evaluation

Fixture `kind: qa` cases carry question, accepted answers (list),
truth window, **and archetype** — the generator's QA mode writes all
four from the sampled keyframes. Metrics: `answer_accuracy`
(VN-normalised match), `moment_hit`, `qa_correct` (= DRES), each **also
reported per archetype** — the routing table (which archetype trusts
which track/model) is then data, not opinion. Ablations, one YAML
each: ledger off (v1 behaviour), reverse_retrieval off, disconfirm
off, detector-counter count (SAM 3.1) vs VLM count and SAM 3.1 vs
CountVid (the note-17 §4b claim, measured on our
own corpus), local vs API reader per archetype.
