"""Detecta contradições entre falas do mesmo deputado.

O detector combina três sinais:

1. similaridade temática para não comparar assuntos diferentes;
2. reversão no mesmo eixo de política pública, quando disponível;
3. NLI com premissa/hipótese passadas como par real ao tokenizer.

A decisão usa a média geométrica das DUAS direções do NLI. Usar o máximo,
como na implementação anterior, gerava falsos positivos quando apenas uma
ordem era interpretada incorretamente pelo modelo; usar o mínimo, por outro
lado, descartava contradições reais por pequenas assimetrias do NLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from ideology_classifier import ClassificationResult


TOPIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
TOPIC_SIM_THRESHOLD = 0.62
CONTRADICTION_THRESHOLD = 0.70
POLICY_REVERSAL_STRENGTH = 0.25


@dataclass
class ContradictionPair:
    opiniao_a: str
    opiniao_b: str
    indice_a: int
    indice_b: int
    topic_similarity: float
    contradiction_score: float
    contradiction_a_b: float
    contradiction_b_a: float
    policy_issue: str | None
    policy_reversal: bool


class ContradictionDetector:
    def __init__(
        self,
        topic_model_name: str = TOPIC_MODEL_NAME,
        nli_model_name: str = NLI_MODEL_NAME,
        topic_sim_threshold: float = TOPIC_SIM_THRESHOLD,
        contradiction_threshold: float = CONTRADICTION_THRESHOLD,
        device: str | None = None,
        batch_size: int = 8,
    ):
        self.topic_model = SentenceTransformer(topic_model_name, device=device)
        self.nli = pipeline(
            "text-classification",
            model=nli_model_name,
            top_k=None,
            device=0 if device == "cuda" else -1,
        )
        self.topic_sim_threshold = topic_sim_threshold
        self.contradiction_threshold = contradiction_threshold
        self.batch_size = batch_size

    def _topic_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        embs = self.topic_model.encode(texts, normalize_embeddings=True)
        return embs @ embs.T

    @staticmethod
    def _contradiction_from_output(raw) -> float:
        while raw and isinstance(raw[0], list):
            raw = raw[0]
        scores = {entry["label"].lower(): entry["score"] for entry in raw}
        return float(scores.get("contradiction", 0.0))

    def _bidirectional_scores(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[float, float]]:
        if not pairs:
            return []

        inputs = []
        for text_a, text_b in pairs:
            inputs.append({"text": text_a, "text_pair": text_b})
            inputs.append({"text": text_b, "text_pair": text_a})

        raw_outputs = self.nli(
            inputs,
            truncation=True,
            batch_size=self.batch_size,
        )
        scores = [self._contradiction_from_output(raw) for raw in raw_outputs]
        return [(scores[i], scores[i + 1]) for i in range(0, len(scores), 2)]

    @staticmethod
    def _is_policy_reversal(
        first: ClassificationResult,
        second: ClassificationResult,
    ) -> tuple[bool, str | None]:
        same_issue = first.issue is not None and first.issue == second.issue
        if not same_issue:
            return False, None
        if first.ideology_score is None or second.ideology_score is None:
            return False, first.issue

        strong = (
            first.stance_strength >= POLICY_REVERSAL_STRENGTH
            and second.stance_strength >= POLICY_REVERSAL_STRENGTH
        )
        opposite = first.ideology_score * second.ideology_score < 0
        return strong and opposite, first.issue

    def find_contradictions(
        self,
        opinioes: list[str],
        classifications: list[ClassificationResult] | None = None,
    ) -> list[ContradictionPair]:
        n = len(opinioes)
        if n < 2:
            return []
        if classifications is not None and len(classifications) != n:
            raise ValueError("classifications deve ter o mesmo tamanho de opinioes")

        sim_matrix = self._topic_similarity_matrix(opinioes)
        candidates = []

        for i, j in combinations(range(n), 2):
            topic_similarity = float(sim_matrix[i, j])
            policy_reversal = False
            policy_issue = None
            if classifications is not None:
                policy_reversal, policy_issue = self._is_policy_reversal(
                    classifications[i], classifications[j]
                )

            if topic_similarity < self.topic_sim_threshold and not policy_reversal:
                continue
            candidates.append(
                (i, j, topic_similarity, policy_reversal, policy_issue)
            )

        text_pairs = [(opinioes[i], opinioes[j]) for i, j, *_ in candidates]
        nli_scores = self._bidirectional_scores(text_pairs)
        found = []

        for candidate, (score_a_b, score_b_a) in zip(candidates, nli_scores):
            i, j, topic_similarity, policy_reversal, policy_issue = candidate
            # A média geométrica pune fortemente uma direção baixa sem exigir
            # que as duas probabilidades sejam idênticas.
            score = float(np.sqrt(score_a_b * score_b_a))
            if score < self.contradiction_threshold:
                continue

            found.append(
                ContradictionPair(
                    opiniao_a=opinioes[i],
                    opiniao_b=opinioes[j],
                    indice_a=i,
                    indice_b=j,
                    topic_similarity=topic_similarity,
                    contradiction_score=score,
                    contradiction_a_b=score_a_b,
                    contradiction_b_a=score_b_a,
                    policy_issue=policy_issue,
                    policy_reversal=policy_reversal,
                )
            )
        return found
