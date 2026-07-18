# Synthetic Ambient FHIR Encounters

A fully synthetic clinical dataset for **The Future of Agentic AI in
Healthcare** hackathon: **25 clinical encounters from 25 synthetic
patients** (one encounter per patient), each pairing an ambient conversation
transcript with the resulting clinical note, an after-visit summary, the
patient's chart background, and structured FHIR R4 context.

## Files

| File | What it is |
|------|------------|
| `synthetic-ambient-fhir-25.jsonl` | Canonical dataset — one JSON record per line |
| `synthetic-ambient-fhir-25.json` | The same 25 records as a single JSON array |
| `schema.json` | JSON Schema for one record |
| `summary.json` | Index of all encounters (dates, visit titles, word counts) |
| `index.html` | Interactive guide — browse every encounter's transcript, note, AVS, and FHIR context in any browser (no server needed) |

The full package ships as `synthetic-ambient-fhir-25.zip`.

## Record structure

Every record has the same fields:

- `id` — `"<patient_id>::<encounter_id>"`.
- `metadata` — encounter date, a human-readable `visit_title` (e.g.
  "Annual physical — diabetes follow-up"), the coded `visit_type`, status,
  and per-resource counts.
- `patient_context` — FHIR `Patient` resource plus a chart background
  summary (active condition and medication labels, per-type resource
  counts for the patient's full record).
- `encounter_fhir` — the FHIR `Encounter` resource and `related_resources`
  (standard FHIR R4 resources recorded at this visit, grouped by type:
  conditions, observations, procedures, medications, reports, immunizations).
- `transcript` — word-for-word clinician–patient conversation,
  speaker-labeled (`DR:`, `PT:`, `NURSE:`, `FAMILY:`).
- `note` — the clinical note for the visit in SOAP-style markdown.
- `after_visit_summary` + `after_visit_summary_provenance` — patient-facing
  summary extracted from the note's Assessment & Plan.

## Quickstart

```python
import json

records = [json.loads(l) for l in open("synthetic-ambient-fhir-25.jsonl")]

r = records[0]
print(r["metadata"]["visit_title"], r["metadata"]["date"])
print(r["transcript"][:500])   # ambient conversation
print(r["note"][:500])         # clinical note
print(r["encounter_fhir"]["related_resources"].keys())  # FHIR at this visit
```

```bash
jq -r '.metadata | "\(.date[:10])  \(.visit_title)"' synthetic-ambient-fhir-25.jsonl
```

## Ideas to build on

- Ambient documentation: transcript → note / AVS generation and evaluation.
- Chart-aware reasoning: combine the transcript with `patient_context` and
  the FHIR `related_resources` to ground agentic workflows.
- Patient communication: turn notes into safer, clearer patient-facing
  summaries and compare against the included AVS.
- The cohort spans ages ~20–85 and settings from wellness exams and
  prenatal intakes to hospital, skilled-nursing, and hospice admissions.

## Data notes

- **Everything is synthetic.** Patients are simulated (Synthea); transcripts
  and notes are LLM-generated, grounded strictly in each encounter's
  structured record. No real patient data is present.
- The `after_visit_summary` is extracted deterministically from the note's
  Assessment & Plan and has **not** been clinically reviewed — treat it as a
  baseline to improve on, not ground truth.
- Clinical content follows simulation modules; it is realistic in shape but
  is not a validated clinical benchmark.

Provided by Abridge for use during the hackathon.
