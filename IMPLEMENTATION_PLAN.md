# OpenParla — implementation plan

## Goal

Build an auditable analysis pipeline over the `unicamp-dl/PublicHearingBR`
dataset that surfaces two distinct kinds of evidence:

1. **Party-alignment divergence:** a politician's statement is materially
   distant from the political tendency assigned to their party.
2. **Self-contradiction:** two statements by the same politician concern the
   same policy topic and express incompatible positions.

The two scenarios are separate outputs: the first compares one statement with
the party reference; the second compares two statements made by the same
politician. Neither output is proof without review of the text and context.

The system must use **word/sentence embeddings** as its primary semantic
mechanism. Results are candidates for human review, not claims of fact about a
politician or party.

## Current starting point

The substantive implementation is on `origin/lorenna`, not on `main`:

- `Leo/preprocessamento_por_estado.ipynb` downloads and partitions
  PublicHearingBR data by state.
- `Lorenna/publichearingbr_ideology/preparar_dados.py` aggregates statements
  by deputy and assigns a reference ideology to each party.
- `ideology_classifier.py` compares statement embeddings with centroids built
  from ideology anchors.
- `contradiction_detector.py` filters statement pairs by embedding similarity,
  then scores candidate pairs with multilingual NLI.
- `pipeline.py` writes classified statements and detected contradiction pairs.

Before feature work, promote this work into an integration branch or merge it
into `main` with its full history intact. Do not copy generated data manually.

## Principles

- Preserve the original statement, speaker, party, hearing/session ID, topic,
  source dataset, model version, and thresholds for every result.
- Keep **party alignment** and **self-contradiction** separate. A statement
  unlike its party is not necessarily a personal contradiction.
- Use embeddings to compare semantic meaning; never rely on keyword matching
  as the primary decision rule.
- Report confidence and evidence, and require review for high-impact results.
- Separate reproducible source code from raw downloads, model caches, and
  generated reports.

## Target data flow

```text
PublicHearingBR LDS/NLI
        |
        v
normalize speakers + statements + source metadata
        |
        +--> party reference map + party/statute evidence
        |             |
        v             v
embedding model --> statement vectors --> ideology/topic representations
        |                                  |
        |                                  +--> party-alignment candidates
        |
        +--> same-politician, same-topic pairs --> contradiction candidates
                                                   |
                                                   v
                                      JSON/CSV review report + summary metrics
```

## Work plan

### 1. Make the project reproducible

1. Merge or base a new integration branch on `origin/lorenna`.
2. Replace the empty README with:
   - project purpose and non-claim disclaimer;
   - data provenance and download instructions;
   - environment setup and commands;
   - expected outputs and review workflow.
3. Add a pinned dependency file or lockfile. Include `requests`, which is
   imported by `preparar_dados.py` but absent from the current requirements.
4. Add a `.gitignore` for model caches, virtual environments, temporary files,
   and newly generated reports. Decide explicitly whether large source/output
   datasets remain versioned or move to Git LFS/reproducible download steps.
5. Convert the preprocessing notebook's reusable logic into importable Python
   modules and leave the notebook as an exploratory companion only.

**Acceptance:** a clean clone can prepare a small sample and run tests with
documented commands.

### 2. Normalize the PublicHearingBR input

Create a canonical statement record, one record per speaker statement:

```json
{
  "statement_id": "lds:1:marcel-van-hattem:2",
  "source_dataset": "LDS",
  "session_id": 1,
  "topic": "...",
  "speaker_name": "...",
  "speaker_key": "normalized stable name",
  "office": "Deputado",
  "party": "NOVO",
  "state": "RS",
  "text": "..."
}
```

Improve identity handling before aggregation:

- retain the raw name and cargo alongside normalized values;
- avoid grouping two people solely because their normalized names match;
- record missing or ambiguous party/state extraction rather than guessing;
- deduplicate only exact duplicates, preserving their source references;
- distinguish quoted speech or journalistic paraphrase from a direct quote
  whenever the dataset provides that signal.

**Acceptance:** validation reports counts of records, speakers, missing party,
missing state, ambiguous identities, and statements per source dataset.

### 3. Establish transparent party reference positions

The existing party-to-label dictionary is useful as a baseline, but it must be
versioned and reviewable.

1. Move it to `config/party_positions.json` with fields such as:

```json
{
  "party": "PL",
  "reference_label": "direita",
  "reference_order": 4,
  "valid_from": "2024-01-01",
  "source_note": "Research-defined reference; review required"
}
```

2. Use the statute PDFs as evidence material, but do not assume every statute
   contains a left-right position. Extract relevant policy passages and retain
   page/file references for human review.
3. Version the map and record its version in every analysis run.
4. Support historical party changes, mergers, and renamed parties where the
   dataset period requires them.

**Acceptance:** every party-alignment result identifies the exact reference-map
version and the reference label used.

### 4. Improve embedding-based ideology classification

Keep the current centroid approach, but make its outputs suitable for review.

1. Continue using a multilingual sentence embedding model (the current
   `intfloat/multilingual-e5-large-instruct` is an appropriate baseline).
2. Embed each statement once, in batches, with normalized vectors. Cache
   vectors keyed by statement ID, model name, and preprocessing version.
3. Store the full similarity vector to every ideology centroid, the winning
   label, top-two margin, and absolute score.
4. Expand anchors with manually reviewed, domain-specific parliamentary
   statements. Keep anchors balanced by topic and label; do not use party name
   alone as an ideology anchor.
5. Add a topic-aware layer: classify or cluster policy domains (economy,
   environment, security, rights, institutional questions, etc.) with
   embeddings. This prevents a single broad left/right score from masking
   issue-specific divergence.
6. Calibrate margin and absolute-similarity thresholds on a manually labeled
   validation set. Keep `neutra` for procedural/non-substantive statements and
   uncertain cases.

**Acceptance:** a held-out labeled sample reports per-label precision, recall,
F1, confusion matrix, abstention rate, and calibration thresholds.

### 5. Define party-alignment divergence rigorously

Treat party alignment as a spectrum distance, not simply an exact-label match.

Use an ordered scale:

```text
esquerda = 0 | centro-esquerda = 1 | centro = 2 |
centro-direita = 3 | direita = 4
```

For each non-neutral statement:

- `distance = abs(statement_order - party_reference_order)`;
- `0`: aligned;
- `1`: adjacent / weak divergence, report only in aggregate;
- `>= 2`: material divergence candidate, subject to confidence threshold;
- `neutra` or low-confidence: not evaluated, never counted as aligned.

Generate both statement-level and politician-level measures:

- number of evaluated statements;
- distribution of labels;
- material divergence count and rate;
- median/mean ideology distance;
- issue-specific divergence rate;
- examples ranked by confidence and distance.

**Acceptance:** the output distinguishes `aligned`, `adjacent`,
`divergence_candidate`, and `insufficient_evidence` without collapsing them
into a single binary claim.

### 6. Improve same-politician contradiction detection

The current two-stage design is sound and should remain embedding-led:

1. Embed statements with a multilingual semantic model.
2. Compare only pairs from the same normalized politician identity.
3. Use cosine similarity and/or embedding-based topic clusters to retain pairs
   that plausibly address the same policy issue.
4. Use bidirectional multilingual NLI on those candidate pairs to estimate
   incompatibility. Retain both directional scores.
5. Add a temporal condition: only label a pair as a possible reversal when
   statement dates/session ordering are known. Otherwise call it an
   `incompatible_statement_pair`, not a later reversal.
6. Exclude or downgrade procedural remarks, third-party quotations, and highly
   truncated/paraphrased statements.
7. Group duplicate or near-duplicate detected pairs into one review case so a
   repeated statement does not inflate results.

Output fields should include:

```json
{
  "case_id": "...",
  "speaker": "...",
  "party": "...",
  "topic_cluster": "...",
  "statement_a": {"text": "...", "session_id": 1, "date": null},
  "statement_b": {"text": "...", "session_id": 39, "date": null},
  "topic_similarity": 0.82,
  "nli_contradiction_a_to_b": 0.91,
  "nli_contradiction_b_to_a": 0.87,
  "status": "review_required"
}
```

**Acceptance:** detected pairs are sampled and manually adjudicated; report
precision at the chosen score threshold, plus the false-positive categories.

### 7. Build review-first reports

Produce machine-readable JSON plus analyst-friendly CSV/Markdown reports.

- `party_alignment_candidates.jsonl/csv`: one evaluated statement per row.
- `self_contradiction_candidates.jsonl/csv`: one deduplicated candidate pair
  per row.
- `run_metadata.json`: model IDs, anchors/reference-map versions, thresholds,
  input hashes, record counts, and timestamps.
- `summary.md`: totals by party, politician, state, topic, confidence band, and
  examples with source session IDs.

Each candidate should expose the original wording and rationale, not just an
opaque score. A reviewer must be able to mark a candidate `confirmed`,
`rejected`, `unclear`, or `needs_context`; those judgments become the
validation dataset for future threshold calibration.

**Acceptance:** an analyst can trace any reported conclusion back to its
original dataset record, text, and configuration.

### 8. Test and evaluate

1. Preserve existing smoke tests, but convert them to `pytest` tests and avoid
   test scripts that write into the repository root.
2. Add unit tests for party normalization, identity aggregation, spectrum
   distance, neutral/uncertain handling, and report schemas.
3. Add deterministic tests with fixed fake embeddings/NLI outputs.
4. Maintain a small, manually adjudicated Portuguese evaluation fixture
   containing:
   - aligned and divergent party examples;
   - actual contradictions, compatible same-topic pairs, unrelated pairs;
   - procedural and quoted-speech false-positive cases.
5. Gate model/threshold changes on measured evaluation deltas rather than
   anecdotal examples.

**Acceptance:** CI runs formatting, unit tests, schema validation, and the
small evaluation suite without downloading full models or datasets.

## Delivery sequence

1. Integrate the Lorenna branch and establish project documentation/config.
2. Refactor data preparation into a canonical statement dataset.
3. Version party positions and anchor sets.
4. Batch/cached embedding inference and ideology-confidence outputs.
5. Party-alignment report with spectrum-distance rules.
6. Topic-filtered, bidirectional-NLI contradiction report.
7. Manual review set, threshold calibration, and evaluation metrics.
8. Publish reproducible reports for the full corpus.

## Decisions to make before full-corpus claims

- Which source(s) define party positions, and who approves changes to that
  map?
- What period does each party position apply to?
- Is a one-step spectrum difference a reportable divergence or only an
  aggregate statistic?
- What confidence thresholds are acceptable after manual validation?
- Which reviewer(s) adjudicate candidates, and how is disagreement resolved?

Until these decisions and a validation pass exist, label outputs as
**embedding-based candidates for review**, never confirmed political
contradictions or ideological positions.
