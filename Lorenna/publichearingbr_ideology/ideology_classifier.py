"""
ideology_classifier.py

Classifica a orientação política de uma fala parlamentar via
embeddings instruction-tuned (multilingual-e5-large-instruct) e
similaridade de cosseno a centróides de exemplos-âncora por classe.

O mecanismo central — instrução prefixada ao texto + decisão por
similaridade de cosseno — segue a abordagem usada pela equipe Munibuc
no Touché 2025 com o NV-Embed-v2 para classificar orientação de
discurso parlamentar.

O que é extensão própria, não validada na literatura original:
- centróide de MÚLTIPLOS exemplos-âncora por classe, em vez de
  comparar contra um único rótulo-palavra ("esquerda"/"direita");
- a margem contínua (top1 - top2) como proxy de confiança/escala,
  usada junto com um piso de similaridade absoluta para decidir
  quando não há sinal suficiente e a fala deve ser rotulada "neutra".

Trate os thresholds (MARGIN_THRESHOLD, MIN_ABS_SIMILARITY) como ponto
de partida: calibre-os com um conjunto de falas rotuladas manualmente
antes de usar os resultados como afirmação factual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from anchors import IDEOLOGY_ANCHORS


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# Instrução usada tanto para as falas quanto para os exemplos-âncora,
# de forma que ambos sejam projetados no MESMO subespaço "orientação
# política" do embedding. Convenção E5-instruct:
#   "Instruct: {tarefa}\nQuery: {texto}"
# A instrução é em inglês porque o model card do e5-instruct reporta
# melhor desempenho com instruções em inglês mesmo para texto em
# outros idiomas — isso é uma escolha empírica do modelo, não minha;
# vale testar também a versão em PT-BR no seu conjunto de validação.
INSTRUCTION = (
    "Given a statement made by a member of parliament, identify the "
    "underlying political ideology (left-right spectrum) reflected by "
    "the substantive policy position expressed, based on the stance "
    "taken on economic, social and institutional issues"
)

# Margem mínima entre a 1ª e a 2ª classe mais similar. Abaixo disso,
# consideramos que não há sinal ideológico claro o suficiente.
MARGIN_THRESHOLD = 0.015

# Piso de similaridade absoluta para a classe vencedora. Embeddings do
# e5 são normalizados e tendem a ter similaridades de cosseno numa
# faixa mais alta que o usual (~0.75-0.90) — calibre este valor com
# exemplos reais do seu corpus.
MIN_ABS_SIMILARITY = 0.78

NEUTRAL_LABEL = "neutra"


def _format(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


@dataclass
class ClassificationResult:
    label: str
    scores: dict[str, float]
    margin: float
    top1: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "margin": round(self.margin, 4),
            "top1": round(self.top1, 4),
        }


class IdeologyClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        anchors: dict[str, list[str]] | None = None,
        margin_threshold: float = MARGIN_THRESHOLD,
        min_abs_similarity: float = MIN_ABS_SIMILARITY,
        device: str | None = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.anchors = anchors or IDEOLOGY_ANCHORS
        self.margin_threshold = margin_threshold
        self.min_abs_similarity = min_abs_similarity
        self.centroids: dict[str, np.ndarray] = {}
        self._build_centroids()

    # -- construção dos centróides ------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        formatted = [_format(t) for t in texts]
        return self.model.encode(
            formatted, normalize_embeddings=True, convert_to_numpy=True
        )

    def _build_centroids(self) -> None:
        for label, examples in self.anchors.items():
            embs = self._encode(examples)
            centroid = embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.centroids[label] = centroid

    # -- classificação ---------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                label=NEUTRAL_LABEL, scores={}, margin=0.0, top1=0.0
            )

        emb = self._encode([text])[0]
        scores = {
            label: float(np.dot(emb, centroid))
            for label, centroid in self.centroids.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1 - top2

        if top_label != NEUTRAL_LABEL and (
            margin < self.margin_threshold or top1 < self.min_abs_similarity
        ):
            final_label = NEUTRAL_LABEL
        else:
            final_label = top_label

        return ClassificationResult(
            label=final_label, scores=scores, margin=margin, top1=top1
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        return [self.classify(t) for t in texts]
