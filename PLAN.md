# Build plan — patient advocate agent

**Thesis:** a patient-side agent that surfaces the questions the patient should be
asking — grounded in their chart, deterministic care-gap computation, and abductive
clue-elevation from their own life context.

**The agent never interrupts the clinician.** It listens to the whole visit, watches
which of its concerns the doctor addresses unprompted, and hands the patient only what
is left over at the end. A device that chirps mid-encounter derails a focused history
and would never be deployed; a short list handed over at the door is exactly the moment
that already exists in care (the after-visit summary).

**So the hard problem is not "call an LLM." It is *what survives*.** A visit generates
a handful of candidate concerns across ~90 utterances. Most get addressed by a good
clinician without prompting. Everything below exists to deliver **the questions that
genuinely were not answered**, instead of everything that was raised.

Measured across all 25 encounters (see Eval): **6 proposed -> 4 after the conversation
-> 3 delivered**, medians. The conversation itself closes 37% of candidates.

---

## Three phases, one engine

| Phase | What happens | Output |
|---|---|---|
| **Before** | generators run on chart + wearable | prep questions the patient brings in |
| **During** | agent listens silently; marks candidates *resolved* as the doctor covers them | nothing shown — this is the suppression work |
| **After** | surviving candidates ranked, grounded, phrased | the delivered short list |

The *During* phase produces no output. That is the point, and it is what makes the
*After* list short.

---

## File layout

```
patient_advocate/
  stream.py      Utterance type + replay adapter + whisper adapter
  chart.py       cohort loader + FHIR/wearable accessors
  detect.py      deterministic candidate generators   <- item 2
  resolve.py     resolution tracking + suppression
  deliver.py     end-of-visit grounding, ranking, phrasing
  agent.py       orchestration                        <- item 1
  eval.py        all-25 run, metrics, suppression ON/OFF  <- item 4
  ui.py          rich live display                    <- item 3
  make_wearable.py
```

---

## The one abstraction that matters

Everything downstream consumes `Utterance`. The agent cannot tell replay from live audio.

```python
@dataclass
class Utterance:
    idx: int
    speaker: str      # DR | PT | NURSE | FAMILY
    text: str
    t: float          # seconds from start
```

Two adapters, same output type:
- `replay(transcript, speed=8.0)` — split on newline, parse `SPK:`, sleep(dur/speed)
- `whisper_stream(wav)` — segments from faster_whisper; recover speaker by matching
  segment start-time against the known TTS utterance boundaries (Whisper has no
  diarization; in production Abridge's ASR supplies it)

Build `replay` first. Add `whisper_stream` last, only if ahead of schedule.

**Because nothing is delivered mid-stream, replay can run fast (8–20x).** A 10-minute
visit compresses to under a minute of demo. This is a direct benefit of end-of-visit
delivery.

---

## Item 2 — detection: LLM-primary, deterministic spine (`detect.py`, 1.0h)

**Measured on all 25 encounters before committing to this design** (`gap_compare.py`,
Opus 4.8, $2.01, results in `data/gap_comparison.xlsx`):

| | Deterministic rules | Opus 4.8 |
|---|---|---|
| Total gaps | 45 | **198** |
| Mean / encounter | 1.8 | **7.92** |
| Encounters with 0 gaps | **3** | 0 |

On Elias the model found *both* deterministic gaps under different names and added
seven more. Three were hand-verified against raw FHIR and all three were real: anemia
on B12 with no CBC drawn, resting tachycardia 100/min never explained, and
**hydrochlorothiazide prescribed twice at one encounter** (the only true duplicate in
the cohort). It also produced the OSA insight — *"morning occipital headaches...
without considering obstructive sleep apnea"* — from the transcript alone, with no
wearable data.

So: **LLM-primary for breadth.** Deterministic keeps two jobs it is strictly better at:

1. **Exact computation and thresholds** — eGFR, cutoffs, arithmetic. The LLM was
   explicitly forbidden from computing values it was not given, and that stays. This is
   the 14%-recall finding; it has not changed.
2. **A stable taxonomy.** The unconstrained LLM run produced **185 distinct type names
   for 198 gaps** (`abnormal_finding_not_addressed`, `abnormal_lab_followup`,
   `abnormal_creatinine_not_addressed` = one concept, three labels). That breaks dedup,
   aggregation, and the eval. **Fix: constrain `type` to an `enum`** in the JSON schema —
   the 10 deterministic types plus ~10 more (cancer screening, prenatal supplementation,
   medication duplication, unmonitored condition, uninvestigated symptom, social barrier).
   LLM breadth, stable vocabulary.

Each candidate carries **structured evidence**, which is what makes groundedness
checkable later.

```python
@dataclass
class Candidate:
    id: str
    topic: str            # for resolution matching
    kind: str             # gap | score | drug_disease | clue | wearable
    trigger: str          # human-readable why
    evidence: list[dict]  # [{source, field, value, date}]  <- must be verifiable
    priority: float
    resolved_by: str|None # utterance idx + span, set during the visit
    verifiable_from_record: bool   # False -> deliver as a QUESTION, never an assertion
    bucket: str           # visit | health_maintenance
```

### `verifiable_from_record` — the largest finding from the ground-truth review

Reviewer marked all 52 high-severity LLM gaps: **28 confirmed valuable, 24
maybe, 0 rejected as wrong.** No hallucinated gaps. But the 24 decompose:

| Reviewer note | n | What it is |
|---|---|---|
| "is there anything in history that can explain" | **17** | unverifiable from one encounter |
| "low value" | 6 | real, but not worth asking — Gate C's job |
| "risk vs benefit in 80Y" | 1 | clinical judgment |

The 17 are structural: the model asserts *not addressed* when it can only see this
encounter. `longitudinal_summary` gives `Procedure: 45` as a **count with no labels**,
so prior screening is invisible. Shipping that tells patients they are overdue for
things they had done last year — the failure Abridge clinicians will spot fastest.

**Do not drop them — reframe them.** An unverifiable gap becomes a question, not a claim:

> ~~"You are overdue for cervical cancer screening."~~
> **"Am I up to date on my cervical cancer screening?"**

That is a genuinely good patient question *because* neither patient nor agent can check,
but the clinician has the chart open. Honest about the uncertainty, still surfaces it.

### `bucket` — visit vs health maintenance

Screening and preventive items are a different category from questions arising out of
what happened today, and they are **PCP scope**. Two consequences:

- **Suppress `health_maintenance` entirely** when the encounter is not primary care —
  specialist visits (psychiatry), hospice, SNF. Uses the same care-context triple as
  scope routing.
- **Separate budget.** Health maintenance must not compete with visit questions for
  `K = 3`. Render as its own shorter list (max 2), below the visit questions.

### Decided against: synthesising longitudinal history

Tempting fix for the 17, but rejected. Authoring the screening history *and* the
detector that reads it is circular — we would decide the answer, then "discover" it
(the same trap as authoring the 271 benefit lines that `coverage_awareness` reads).
Synthea would give genuine history, but those patients have no transcripts, so they
cannot be demo cases. And the verification-question reframe is **better product
behaviour even with a full record**: real charts are incomplete (care elsewhere,
outside imaging), so "am I up to date?" stays the right ask. ~45-60 min saved.

Generators (each 10–20 lines; do not gold-plate):

| # | Generator | Fires when |
|---|---|---|
| 1 | `bp_control` | BP >=140/90 -> stage 2 |
| 2 | `egfr` | CKD-EPI 2021 from creatinine+age+sex; flag <60, or diuretic started with no documented eGFR |
| 3 | `ascvd_inputs_missing` | cardiometabolic condition set but no lipid panel -> risk not computable |
| 4 | `statin_gap` | condition set implies statin consideration + no statin in med labels |
| 5 | `a1c_threshold` | 5.7–6.4 prediabetes, >=6.5 diabetes |
| 6 | `phq2_escalation` | PHQ-2 >=3 -> PHQ-9 required (hard rule) |
| 7 | `drug_disease` | e.g. thiazide + prediabetes (HCTZ raises glucose) |
| 8 | `wearable_delta` | sleep / RHR / steps shift beyond threshold in window |
| 9 | `transcript_clue` | lifestyle/exposure cue in patient's own words (sofa bed, shift work, new med) |
| 10 | `cost_adherence_risk` | prior cost/coverage disruption + new prescriptions today + no written cost contingency |

Rule: **generators propose, the model never invents.** The LLM only selects and phrases.

### Build steps — `cost_adherence_risk` (~20 lines, the second demo case)

Four small functions. The nuance in step 3 is what keeps it from being naive.

1. **`cost_barrier_evidence(rec) -> list[dict]`** — collect proof the patient has a
   cost/coverage problem, from two sources:
   - *transcript regex*: `insurance switched|stopped refilling|ran out of|couldn't
     afford|left it on the counter|\$\d+` near pharmacy/refill context
   - *structured*: PRAPARE `Primary insurance` in {Medicaid, None/uninsured};
     condition labels containing `Unemployed`, `Housing unsatisfactory`
   Each hit becomes an evidence dict with source + verbatim span, so groundedness
   stays deterministically checkable.
   **Do not require Medicaid/uninsured status** — Julius is on private insurance and
   his risk comes from the transcript disclosure. Gate on evidence, not category.

2. **`new_prescriptions(rec) -> list[str]`** — `MedicationRequest` resources on this
   encounter. This is the *exposure*: more new scripts = higher abandonment risk.

3. **`cost_contingency_present(...) -> bool`** — the important distinction.
   Do **not** test whether cost was *mentioned*. Julius's doctor did mention it
   ("all cheap generics, a few dollars a month") — a naive keyword trigger would
   wrongly mark it resolved. Test whether a **contingency** exists: an instruction for
   what to do *if* the price differs (call back, alternative, assistance program,
   90-day supply, pharmacy price check). Reassurance is not a plan.

4. **Assemble the Candidate.** Priority scales with:
   `prior_abandonment_event (bool) x n_new_prescriptions x (not contingency_present)`.

Pass the patient's own verbatim quote into the phrasing call so the question echoes
their language rather than clinical register.

---

## Item 1 — listen, resolve, deliver (`resolve.py` + `deliver.py` + `agent.py`, 2.5h)

### During the visit — `resolve.py` (deterministic, free, no LLM)

For each incoming utterance, test every open candidate for **resolution**:

- **addressed**: the clinician stated a plan, order, or explanation covering the topic
- **partially addressed**: topic raised but no actionable plan -> stays open, priority
  drops (Julius's cost case lives here — reassurance without contingency)
- **untouched**: never came up -> stays open at full priority

Match on topic keywords plus a plan-verb test (`start`, `order`, `refer`, `recheck`,
`we'll`, `I'm putting in`). Record `resolved_by` = utterance index + the verbatim span,
so the eval can show *why* something was suppressed.

**This is the whole product.** A good clinician resolves most candidates unprompted;
the residue is what the patient actually needs.

### At the end — `deliver.py`

Triggered on stream end, or on a closing cue (`any questions`, `see you`, `follow up`).

0. **Goals-of-care guard (first, deterministic).** This cohort has 2 hospice and 3 SNF
   admissions. Suggesting colorectal screening to an end-stage colon cancer patient on
   hospice is the output that sinks a demo in front of clinicians. Gate on
   `Encounter.class` + hospice/SNF `serviceProvider` and drop all preventive and
   referral candidates outright. Cheap; say it out loud to judges.
1. **Filter** — drop resolved candidates.
2. **Gate B — grounding** (one batched LLM call over all survivors): each question must
   cite a specific datum or it dies. Batched, so it is one call, not N.
3. **Scope routing** — see below. Folds into the same call as ranking/phrasing.
4. **Gate C — budget**: hard cap `K = 3`, strict priority ordering.
5. **Phrase** — patient's voice, first person, <=25 words.

### Scope routing — the selection axis

Ranking by a hand-tuned priority float is arbitrary. The real axis is **can the
clinician in front of the patient actually resolve this?** Care context comes from a
triple available on every encounter: `Encounter.class` (AMB/IMP/HH), `visit_type`
(SNOMED), and `serviceProvider.display` (e.g. "PRIMARY CARE MEDICINE AND PEDIATRICS",
"FIRST PSYCHIATRIC PLANNERS", "NEW ENGLAND HOSPICE II"). No `Practitioner` resource is
in the bundle, so there is no NPI/specialty lookup — the triple is the signal.

Scope classification is *knowledge* work, so it rides in the existing ranking/phrasing
LLM call. **No extra calls.**

| Scope | Archetype | Elias example |
|---|---|---|
| in scope | ask now | thiazide/glucose, lipid panel, missing CBC, duplicate HCTZ, HTN intensity |
| warrants referral | ask for a referral | OSA -> sleep specialist |
| barrier | ask about access | Julius's pharmacy cost |
| unverifiable | ask to confirm | "Am I up to date on my screening?" |

Delivered output is **two lists**: visit questions (`K = 3`) and, only when the
encounter is primary care, health maintenance (max 2).

**Do not over-refer.** Scope is not binary — PCPs routinely order home sleep apnea
tests. Default to in-scope; route to referral only when the gap genuinely needs
specialist capability. "Get a referral for your lipid panel" is the failure mode.

### Coverage is SUBTEXT, never the question

**A doctor does not know the patient's copay.** Asking them "is this covered?" is
unanswerable, wastes visit time, and reads as naive to any clinician judging this.

So every delivered item separates the two:

```python
@dataclass
class DeliveredItem:
    question: str          # spoken to the clinician - clinical only
    evidence: list[dict]   # what it is grounded in
    scope: str             # in_scope | referral | barrier
    context: str | None    # patient-facing subtext - NEVER asked aloud
```

Rendered:

> **Ask:** "Could my headaches be sleep apnea — should I see a sleep specialist?"
> *Your plan: specialist visit $40 copay, no referral required, in-network.*

The context line answers the patient's unspoken question ("can I afford to say yes if
they suggest it?") without making the clinician field an insurance query. Coverage comes
from the mock 271 via a deterministic benefit-line lookup.

**271 scope note:** an X12 271 returns *medical* benefits — office-visit copay,
deductible, OOP max. Pharmacy benefits are a separate PBM carve-out and drug-level cost
needs an NCPDP real-time benefit check, **not** a 271. So the 271 is correct for
referral cost and **must not** be used to answer Julius's pharmacy question.

### LLM usage — now trivially cheap

End-of-visit delivery removes all latency pressure. **Two calls per encounter**, both
after the conversation ends:

- **Gate B (grounding, batched):** `claude-opus-4-8`, `output_config={"effort":"low"}`,
  `thinking={"type":"disabled"}`, json_schema output.
- **Phrasing:** `claude-opus-4-8`, adaptive thinking.

No streaming inference, no per-utterance calls, no latency risk during the demo.
This is a real simplification over mid-visit delivery — say so.

### Deliberate non-goal
No urgent-interrupt path. If something were genuinely time-critical you would want one,
and that is the obvious v2 — but interrupting is the thing clinicians reject, so the
default must be silence. Good Q&A answer; do not build it.

---

## Item 4 — eval (`eval.py`, 0.75h) — BUILD THIS BEFORE THE UI

Runs headless over all 25 encounters. Metrics:

| Metric | How | Why it matters |
|---|---|---|
| candidates proposed vs delivered | count | the headline: 6 proposed -> 3 delivered (medians) |
| **suppression accuracy** | for each suppressed candidate, does `resolved_by` actually contain a plan for that topic? spot-check a sample | proves suppression is real, not lossy |
| **groundedness rate** | **deterministic** — verify each delivered question's evidence refs resolve to a real field/value | no judge noise (but see the caveat below) |
| questions / encounter | count | median 3, range 0–5 |

**MEASURED, all 25 encounters** (`run_full`, ~6 min, ~75 Sonnet calls):

| stage | median | range | total |
|---|---|---|---|
| proposed | 6 | 4–10 | 155 |
| after resolve | 4 | 0–7 | 97 |
| **delivered — visit** | **2** | 0–3 | 56 |
| **delivered — health maintenance** | **1** | 0–2 | 21 |
| **delivered — total** | **3** | 0–5 | 77 |

Stage by stage, in `deliver()`'s actual order (guard -> filter -> **ground** ->
split -> **budget**), totals across the 25 encounters:

| stage | remaining | dropped |
|---|---|---|
| proposed | 155 | — |
| after resolve | 97 | **58 (37%)** |
| after goals-of-care guard | 97 | 0 |
| after **Gate B** (groundedness) | 93 | **4** |
| after bucket scoping | 92 | 1 |
| after **Gate C** (K budget) | 75 | **17** |

**The budget does most of the filtering, not the LLM gate.** Gate B drops only 4 of
97 — and reading them, it behaves as an evidence-PACKAGING check rather than a
clinical-validity one: 2 of the 4 concerns are plausibly real but their evidence array
didn't carry the specific datum the trigger claimed. One encounter delivers nothing,
and it is a genuine zero (every candidate resolved during the visit), not a failure.
Invariant violations: 0.

**The money chart: suppression ON vs OFF.** Same run, flip resolution tracking.
OFF delivers a median of 6 per visit, most already answered during the visit — which is
precisely the useless product. ON delivers 3.
Cheap, honest, within-run A/B; almost nobody at a hackathon ships one.

**Caveat, state it before a judge does:** the OFF arm is not an independent measurement —
it equals `proposed` at every encounter, because OFF simply skips `resolve()`. The
ablation therefore measures the resolver only, and says nothing about Gates B and C.
Likewise the 1.000 groundedness rate is near-tautological: Gate B is an LLM check that
drops ungrounded candidates, and the deterministic rate then scores its survivors. It
confirms the two agree; it is not independent evidence that the output is grounded.

Groundedness is checkable **deterministically** because candidates carry structured
evidence refs. That is the whole reason for the `evidence` field.

---

## Item 3 — UI (`ui.py`, 0.75h)

`rich` Live layout, three zones:
- **left**: utterances streaming past (fast)
- **right top**: open candidates, greying out and striking through as they resolve —
  *this is the visible suppression, and it is the best thing in the demo*
- **right bottom**: at end of stream, the 3 delivered questions with evidence lines

Terminal, not web — faster to build, screen-records well, no web stack.

---

## Timebox (submissions due 5:00 PM)

| Time | Work |
|---|---|
| 10:30–11:00 | repo + `chart.py` accessors, load cohort + wearable |
| 11:00–12:00 | `detect.py` — the 10 generators (item 2) |
| 12:00–12:30 | `stream.py` replay + candidate state |
| 12:30–13:00 | lunch / buffer |
| 13:00–14:30 | `resolve.py` + `deliver.py` + `agent.py` (item 1) |
| 14:30–15:15 | `eval.py` + suppression ON/OFF ablation (item 4) |
| 15:15–16:00 | `ui.py` (item 3) |
| 16:00–16:30 | wearable rewrite + TTS excerpt |
| 16:30–16:50 | record 1-min video, submit |

**There is no slack. Cut order if behind:**
1. **Wearable** — cut first. Opus found the OSA insight from the transcript alone, so
   the wearable is corroboration, not the source. Frees ~35 min.
2. Whisper loop drops entirely — demo the replay
3. UI degrades to coloured stdout (costs nothing in judging)
4. **Never cut the eval, the scope routing, or the goals-of-care guard.** The eval is
   the differentiator; the other two are what make the output clinically credible.

Eval is scheduled before UI on purpose: it runs headless, so if the day collapses
you want eval + stdout, not a pretty UI with no numbers.

---

## The two demo cases

Two patients, two **structurally different** generators, one engine. That contrast is
the proof the system generalizes rather than being tuned to a single record.

### Hero — Elias Wisozk (record 6), the wearable-clue path
54M, new BP 141/100, morning occipital headaches, BMI 30.2, A1c 6.28, started on HCTZ.
- he says it himself: sleeps on his sister's sofa bed, *"I blamed the sofa bed, honestly"*
- wearable corroborates: sleep 6.8 -> 5.4h, awakenings 1.1 -> 3.1, resting HR 72 -> 76
- the doctor started an antihypertensive and never asked about sleep
- poor sleep / undiagnosed apnea is a genuine reversible contributor to new hypertension
- **survives to delivery** because the topic never came up at all
- **question:** "Could my blood pressure and the headaches be about how badly I've slept
  since I moved in with my sister?"

### Second — Julius Renner (record 12), the cost-adherence path
36M, unemployed, unsatisfactory housing, hypertension diagnosed previously and
**never treated**.
- verbatim: *"I ran out of everything months ago and stopped refilling when the
  insurance switched over to my wife's plan"*
- verbatim: *"The pharmacy said it would be forty bucks and I just... left it on the counter"*
- today he receives **four** prescriptions (lisinopril, amlodipine, HCTZ, acetaminophen)
- the doctor offers reassurance ("cheap generics") but no contingency
- **survives to delivery as *partially addressed*** — the strongest possible
  demonstration that resolution tracking is not keyword matching
- **question:** "Last time I left the pharmacy because it was forty dollars. What will
  these four cost me on my wife's plan, and what do I do if the counter price isn't
  what you expect?"

Cost-driven non-adherence is one of the largest real drivers of uncontrolled
hypertension, so this question plausibly decides whether the treatment happens at all.

---

## Demo script (3 min)

1. **20s** — the problem: patients don't know what to ask; the chart knows things
   they don't, and they know things the chart doesn't. And no one wants an AI
   interrupting their doctor.
2. **60s** — live: Elias's visit replaying at speed. **Candidates light up on the right
   and strike through as the doctor addresses them.** At the end, 3 questions remain.
   Land on the sleep one.
3. **30s** — same engine, different patient: Julius. His cost concern survives as
   *partially addressed* — the doctor said "cheap generics" but never said what to do
   if the counter price is wrong, and that is exactly what failed him last time.
4. **35s** — how: an LLM pass proposes over the record, resolution tracking suppresses,
   every delivered question traces to a datum. 6 proposed, 3 delivered (medians over 25).
5. **25s** — the eval: suppression ON vs OFF across 25 encounters, groundedness rate.
6. **10s** — scale: same engine over a population -> "what did patients like me
   wish they'd asked."

## Say out loud, unprompted
- the agent never interrupts the clinician — that is a design decision, not a limitation
- the wearable is synthetic, generated by us
- any cohort statistic is synthetic, illustrative of the mechanism, not evidence
- Whisper has no diarization; speaker mapping is by timestamp
