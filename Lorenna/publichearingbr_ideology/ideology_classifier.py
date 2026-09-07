"""Classificação contrastiva da posição política de uma fala.

O E5 produz cossenos absolutos muito altos e próximos entre centróides
ideológicos (tipicamente 0.95--0.98 neste corpus). Por isso, este módulo não
decide mais pelo maior cosseno entre rótulos genéricos. Ele constrói eixos de
posturas opostas sobre a mesma pauta, por exemplo:

    aborto: legalizar (-1) <----------------------> proibir (+1)

A projeção da fala no eixo é normalizada para que os próprios polos valham
aproximadamente -1 e +1. Antes de classificar, há um filtro de evidência: a
pauta vencedora deve ser mais semelhante à fala do que os exemplos neutros e
também se destacar das demais pautas. Falas procedurais continuam "neutra".

Os limiares são pontos de partida e precisam ser validados em amostra
rotulada. Os campos de auditoria no resultado permitem fazer essa calibração.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from anchors import IDEOLOGY_ANCHORS, POLICY_STANCE_ANCHORS


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
INSTRUCTION = (
    "Represent the concrete policy stance expressed in a parliamentary "
    "statement. Focus on what the speaker supports or opposes, including "
    "negation; statements on the same issue with opposite positions should "
    "be distinguishable"
)

# A pauta precisa superar o centróide procedural/neutro por esta margem.
POLICY_EVIDENCE_THRESHOLD = 0.015
# A pauta mais próxima precisa se destacar da segunda colocada.
TOPIC_MARGIN_THRESHOLD = 0.010
# Força mínima usada pelos detectores de contradição, não para neutralidade.
STANCE_STRENGTH_THRESHOLD = 0.25
NEUTRAL_LABEL = "neutra"

LABEL_POSITIONS = {
    "esquerda": -1.0,
    "centro-esquerda": -0.5,
    "centro": 0.0,
    "centro-direita": 0.5,
    "direita": 1.0,
}


def _format(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


@dataclass
class ClassificationResult:
    label: str
    scores: dict[str, float]
    margin: float
    top1: float
    ideology_score: float | None
    issue: str | None
    issue_similarity: float
    policy_evidence: float
    topic_margin: float
    stance_strength: float

    @property
    def has_policy_stance(self) -> bool:
        return self.label != NEUTRAL_LABEL and self.ideology_score is not None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "margin": round(self.margin, 4),
            "top1": round(self.top1, 4),
            "ideology_score": (
                round(self.ideology_score, 4)
                if self.ideology_score is not None
                else None
            ),
            "issue": self.issue,
            "issue_similarity": round(self.issue_similarity, 4),
            "policy_evidence": round(self.policy_evidence, 4),
            "topic_margin": round(self.topic_margin, 4),
            "stance_strength": round(self.stance_strength, 4),
        }


class IdeologyClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        anchors: dict[str, list[str]] | None = None,
        policy_anchors: dict[str, dict[str, list[str]]] | None = None,
        policy_evidence_threshold: float = POLICY_EVIDENCE_THRESHOLD,
        topic_margin_threshold: float = TOPIC_MARGIN_THRESHOLD,
        device: str | None = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.anchors = anchors or IDEOLOGY_ANCHORS
        self.policy_anchors = policy_anchors or POLICY_STANCE_ANCHORS
        self.policy_evidence_threshold = policy_evidence_threshold
        self.topic_margin_threshold = topic_margin_threshold
        self.issue_poles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.neutral_centroid: np.ndarray
        self._build_references()

    def _encode(self, texts: list[str]) -> np.ndarray:
        formatted = [_format(t) for t in texts]
        return self.model.encode(
            formatted,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    @staticmethod
    def _normalized_centroid(embeddings: np.ndarray) -> np.ndarray:
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else centroid

    def _build_references(self) -> None:
        neutral_examples = self.anchors.get(NEUTRAL_LABEL, [])
        if not neutral_examples:
            raise ValueError("É necessário fornecer exemplos-âncora da classe neutra.")

        all_examples = list(neutral_examples)
        ranges = {}
        for issue, poles in self.policy_anchors.items():
            left_start = len(all_examples)
            all_examples.extend(poles["left"])
            right_start = len(all_examples)
            all_examples.extend(poles["right"])
            ranges[issue] = (left_start, right_start, len(all_examples))

        embeddings = self._encode(all_examples)
        self.neutral_centroid = self._normalized_centroid(
            embeddings[:len(neutral_examples)]
        )

        for issue, (left_start, right_start, end) in ranges.items():
            left = self._normalized_centroid(embeddings[left_start:right_start])
            right = self._normalized_centroid(embeddings[right_start:end])
            self.issue_poles[issue] = (left, right)

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score <= -0.55:
            return "esquerda"
        if score <= -0.15:
            return "centro-esquerda"
        if score < 0.15:
            return "centro"
        if score < 0.55:
            return "centro-direita"
        return "direita"

    @staticmethod
    def _label_scores(score: float) -> dict[str, float]:
        # Afinidade na escala, não cosseno bruto. Cada rótulo vale 1 em sua
        # posição e decai linearmente conforme a distância no espectro.
        return {
            label: max(0.0, 1.0 - abs(score - position))
            for label, position in LABEL_POSITIONS.items()
        }

    def _classify_embedding(self, embedding: np.ndarray) -> ClassificationResult:
        issue_candidates = []
        for issue, (left, right) in self.issue_poles.items():
            left_similarity = float(np.dot(embedding, left))
            right_similarity = float(np.dot(embedding, right))
            issue_similarity = max(left_similarity, right_similarity)

            axis = right - left
            half_squared_span = 0.5 * float(np.dot(axis, axis))
            score = (
                (right_similarity - left_similarity) / half_squared_span
                if half_squared_span > 0
                else 0.0
            )
            issue_candidates.append((issue_similarity, issue, score))

        issue_candidates.sort(reverse=True)
        issue_similarity, issue, raw_score = issue_candidates[0]
        second_similarity = (
            issue_candidates[1][0] if len(issue_candidates) > 1 else 0.0
        )
        topic_margin = issue_similarity - second_similarity
        neutral_similarity = float(np.dot(embedding, self.neutral_centroid))
        policy_evidence = issue_similarity - neutral_similarity

        # Limita extrapolações extremas sem destruir a direção do eixo.
        ideology_score = float(np.clip(raw_score, -1.0, 1.0))
        scores = self._label_scores(ideology_score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top1 = ranked[0]
        margin = top1 - ranked[1][1]

        has_policy_evidence = (
            policy_evidence >= self.policy_evidence_threshold
            and topic_margin >= self.topic_margin_threshold
        )
        final_label = top_label if has_policy_evidence else NEUTRAL_LABEL

        return ClassificationResult(
            label=final_label,
            scores=scores,
            margin=margin,
            top1=top1,
            ideology_score=ideology_score if has_policy_evidence else None,
            issue=issue if has_policy_evidence else None,
            issue_similarity=issue_similarity,
            policy_evidence=policy_evidence,
            topic_margin=topic_margin,
            stance_strength=abs(ideology_score) if has_policy_evidence else 0.0,
        )

    def classify(self, text: str) -> ClassificationResult:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        if not texts:
            return []

        valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        results = [self._neutral_result() for _ in texts]
        if not valid_indices:
            return results

        embeddings = self._encode([texts[i] for i in valid_indices])
        for index, embedding in zip(valid_indices, embeddings):
            results[index] = self._classify_embedding(embedding)
        return results

    @staticmethod
    def _neutral_result() -> ClassificationResult:
        return ClassificationResult(
            label=NEUTRAL_LABEL,
            scores={},
            margin=0.0,
            top1=0.0,
            ideology_score=None,
            issue=None,
            issue_similarity=0.0,
            policy_evidence=0.0,
            topic_margin=0.0,
            stance_strength=0.0,
        )
