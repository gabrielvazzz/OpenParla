# Review of `lorenna-treino`

## Purpose and scope

OpenParla processes the PublicHearingBR corpus to identify two kinds of cases for
human review: statements that may diverge from an assigned party position, and
pairs of statements by the same deputy that may be incompatible. It uses
sentence embeddings for ideology/topic representations and a multilingual NLI
model for contradiction scoring. This review covers the current
`lorenna-treino` worktree, including its generated results.

## Final judgment

This is a useful **candidate-generation prototype**, but it is not a validated
classifier of political ideology or contradiction. The code contains sensible
traceability and abstention safeguards, and the core cosine/vector operations
are correct. However, the experiment does not establish that fine-tuning helps
the target tasks, and the current outputs are not reliable enough to support
claims about deputies or parties. They should remain review candidates, never
findings of fact.

The reported similarity to a naive GPT-4o approach is consistent with the main
result of this review: there is no demonstrated gain from training. Because the
two approaches were not compared on the same held-out, human-labeled set, that
similarity is not yet a quantitative result either.

## Main findings

### Critical: there is no valid task evaluation

- The 200 sampled speech annotations have no topic or stance labels, and all
  461 pair annotations mark `contradicao` as false. Therefore contradiction
  precision/recall and ideology/topic quality cannot be measured
  (`Lorenna/experimentos/dados_anotados/anotacoes_falas.json:10-11`,
  `Lorenna/experimentos/dados_anotados/anotacoes_pares.json:3-10`).
- Both experiment reports contain corpus and candidate counts, but no quality
  metrics (`Lorenna/saidas_experimentos/relatorio_experimento.json:243-247`,
  `Lorenna/saidas_experimentos/relatorio_embedding_ajustado.json:243-247`).
  This does not meet the repository's own held-out evaluation criteria
  (`IMPLEMENTATION_PLAN.md:168-173`, `IMPLEMENTATION_PLAN.md:242-243`).

### Critical: training is not aligned with the claimed task

- Fine-tuning uses all 884 summary/transcript pairs, no train/validation split,
  no contradiction labels, and `MultipleNegativesRankingLoss`
  (`Lorenna/experimentos/experimento_hibrido.py:293-307`). This trains retrieval
  between a generated summary and its transcript context; it does not train
  same-topic retrieval, ideology classification, or contradiction detection.
- Production compares summary-to-summary embeddings and delegates the final
  decision to a separately pretrained NLI model
  (`Lorenna/contradiction_detector.py:128-155`). The trained embedding only
  changes the optional topic filter (`Lorenna/pipeline.py:105-110`,
  `Lorenna/pipeline.py:332-335`).
- Training contexts are truncated to 128 tokens
  (`Lorenna/modelo/README.md:348-362`), and the model weights are excluded from
  Git (`.gitignore:2`). A clean checkout cannot reproduce the adjusted-model
  result.

### Critical: the comparison is circular and internally confounded

- Annotation pairs are selected as the top three TF-IDF pairs per deputy
  (`Lorenna/experimentos/experimento_hibrido.py:136-173`). Embeddings are then
  judged only on their overlap with that TF-IDF-selected set: 345/461 base and
  354/461 adjusted pairs. Unique embedding candidates are not judged.
- The threshold is the median score of the same records being evaluated, and a
  high semantic similarity is used directly as a prediction of contradiction
  (`Lorenna/experimentos/experimento_hibrido.py:232-245`,
  `Lorenna/experimentos/experimento_hibrido.py:279-286`). Similarity can find the
  same topic, but cannot determine whether two positions conflict.
- The before/after report says unchanged party logic moved from zero to two
  results (`Lorenna/saidas_experimentos/comparacao_resultados_2.json:4-19`). The
  classified-speech files are identical, so this change cannot be attributed
  to the topic embedding.

### High: current results do not pass a basic content check

- None of the four current incompatible-pair candidates is a clear
  contradiction. They include two compatible statements supporting a
  negotiated two-state solution, two unrelated procedural remarks, two
  complementary disaster-fund sources, and escalating actions against ENEL
  (`Lorenna/saida_novo/pares_potencialmente_incompativeis.json:10-23`,
  `:47-60`, `:84-97`, `:121-134`).
- The three adjusted-model candidates have the same problem; for example,
  criticism of a previous health administration and praise for the current one
  are mutually compatible
  (`Lorenna/saidas_experimentos/contradicoes_falas_2.json:28-49`). High NLI
  scores here are not calibrated probabilities of a real political
  contradiction.
- The two party-divergence results depend on broad editorial assumptions. One
  even classifies criticism of restrictive environmental policy as
  center-left (`Lorenna/saida_novo/candidatos_divergencia_partidaria.json:8-24`).

## Bias, fairness, and grounding in real data

The defensible statement is:

> The analysis is grounded in a public corpus of real Brazilian congressional
> hearings. It retains only machine-generated opinion summaries that the source
> dataset manually marked as supported by nearby transcript excerpts, preserves
> session/context evidence, abstains on most statements, and marks outputs for
> human review. These safeguards improve traceability and reduce unsupported
> claims; they do not establish representativeness, ideological neutrality, or
> equal error rates across groups.

Important limits to that statement are:

- The retained text is a generated third-person summary, not necessarily the
  speaker's literal wording (`Lorenna/preparar_dados.py:332-400`). The pipeline
  scores that summary; transcript excerpts are attached only for later review
  (`Lorenna/pipeline.py:119-126`, `Lorenna/pipeline.py:232-263`).
- The final corpus has 884 statements from 219 deputies and 176 sessions, but
  it is uneven: PT, PL, and PSOL provide 409/884 statements; 134/884 statements
  have no party and 148/884 have no state. No sampling or weighting establishes
  national, regional, party, topic, gender, or racial representativeness.
- Ideology anchors and party positions are editorial and explicitly unvalidated
  (`Lorenna/anchors.py:10-17`, `Lorenna/party_alignment.py:12-15`). Only selected
  parties receive issue-specific adjustments (`Lorenna/party_alignment.py:55-72`),
  while centrist parties are excluded from divergence detection
  (`Lorenna/party_alignment.py:226-228`). This creates unequal exposure to being
  flagged.
- There is no annotation protocol, second annotator, agreement measure, or
  error-rate breakdown by party, state, topic, or demographic group. Fairness
  has therefore not been measured.

## Correctness and engineering assessment

- The main vector calculations are correct: embedding vectors and TF-IDF rows
  are normalized before dot products, ideology centroids are normalized, and
  the current NLI path evaluates real text pairs in both directions
  (`Lorenna/ideology_classifier.py:113-125`,
  `Lorenna/contradiction_detector.py:69-97`,
  `Lorenna/experimentos/experimento_hibrido.py:169-173`). No evident tensor-shape
  or cosine-sign error explains the poor results.
- Identity aggregation is unsafe for longitudinal political claims: people are
  merged solely by normalized name, and the first party/state is generally kept
  (`Lorenna/preparar_dados.py:356-377`). Namesakes and affiliation changes can
  therefore combine or misattribute statements.
- Reproducibility is incomplete: dependencies and remote model/data revisions
  are not pinned, the trained weights are absent, and outputs contain no input
  hashes or run metadata. Several alternative implementations and stale output
  schemas also remain in the tree.
- Tests cover plumbing with deterministic fake encoders and NLI, not real model
  quality (`Lorenna/tests/test_logic_smoke.py:1-7`). The available non-model
  preparation, enrichment, report, and compile checks pass; the full suite could
  not run in the present environment because `pytest`, NumPy, and model
  dependencies are not installed.

## Minimum needed before stronger claims

1. Build a stratified, held-out set with real positive and negative
   contradictions and ideology/topic labels, at least two annotators, and
   disagreement reporting.
2. Split by deputy and session before training; compare base embedding, adjusted
   embedding, and GPT-4o on exactly the same set using precision, recall,
   precision-at-k, abstention, and false-positive categories.
3. Either train on an objective tied to same-topic contradiction retrieval or
   describe the current model narrowly as summary-to-transcript retrieval
   tuning. Pin and publish every artifact/revision needed to reproduce it.
4. Source and time-bound party positions, remove asymmetric treatment, and
   report coverage and error rates by party, state, topic, and other groups for
   which responsible labels are available.
