"""
contradiction_detector.py

Detecta contradições entre falas de um mesmo deputado: pares de
opiniões que tratam do mesmo tema, mas assumem posições opostas
(ex.: a favor do aborto em uma fala, contra em outra).

Isto é uma heurística de duas etapas, de engenharia própria (não vem
de um paper específico):

1. Filtro de tópico: só comparamos pares de falas cuja similaridade
   temática (embedding genérico "sentence-transformers", SEM a
   instrução de ideologia) supere um limiar. Isso evita rodar o
   modelo de NLI — que é mais caro e menos preciso em textos longos e
   heterogêneos — em pares de falas sobre assuntos completamente
   diferentes, e reduz falsos positivos (uma fala sobre economia e
   outra sobre segurança pública não são "contraditórias", são só
   sobre temas diferentes).
2. Para os pares que passam no filtro de tópico, rodamos um modelo de
   NLI multilíngue (entailment / neutro / contradição) nas duas
   direções (A como premissa e B como hipótese, e vice-versa) e usamos
   o maior score de "contradiction" das duas direções, já que a ordem
   pode afetar o resultado do NLI.

Trate os limiares (TOPIC_SIM_THRESHOLD, CONTRADICTION_THRESHOLD) como
ponto de partida — ajuste-os observando exemplos reais do seu corpus.
Um "contradiction" alto do NLI indica incompatibilidade textual, não
necessariamente uma contradição de posição política real (ex.: o
deputado pode estar citando a opinião de outra pessoa) — recomendo
revisão manual dos pares reportados antes de tirar conclusões.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


TOPIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Similaridade temática mínima para considerar que duas falas tratam
# do "mesmo assunto" e, portanto, vale a pena checar contradição.
TOPIC_SIM_THRESHOLD = 0.55

# Score mínimo de "contradiction" do NLI (máximo entre as duas
# direções) para reportar o par como contraditório.
CONTRADICTION_THRESHOLD = 0.70


@dataclass
class ContradictionPair:
    opiniao_a: str
    opiniao_b: str
    indice_a: int
    indice_b: int
    topic_similarity: float
    contradiction_score: float


class ContradictionDetector:
    def __init__(
        self,
        topic_model_name: str = TOPIC_MODEL_NAME,
        nli_model_name: str = NLI_MODEL_NAME,
        topic_sim_threshold: float = TOPIC_SIM_THRESHOLD,
        contradiction_threshold: float = CONTRADICTION_THRESHOLD,
        device: str | None = None,
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

    # -- similaridade temática -------------------------------------------

    def _topic_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        embs = self.topic_model.encode(texts, normalize_embeddings=True)
        return embs @ embs.T

    # -- NLI ---------------------------------------------------------------

    def _contradiction_score(self, premise: str, hypothesis: str) -> float:
        raw = self.nli(f"{premise}</s></s>{hypothesis}", truncation=True)
        # A pipeline com top_k=None retorna uma lista de dicts
        # {"label": ..., "score": ...} para cada classe; dependendo da
        # versão do transformers, pode vir aninhada em outra lista.
        entries = raw[0] if isinstance(raw[0], list) else raw
        scores = {e["label"].lower(): e["score"] for e in entries}
        return scores.get("contradiction", 0.0)

    # -- API pública -------------------------------------------------------

    def find_contradictions(self, opinioes: list[str]) -> list[ContradictionPair]:
        n = len(opinioes)
        if n < 2:
            return []

        sim_matrix = self._topic_similarity_matrix(opinioes)
        found: list[ContradictionPair] = []

        for i, j in combinations(range(n), 2):
            topic_sim = float(sim_matrix[i, j])
            if topic_sim < self.topic_sim_threshold:
                continue

            score_ij = self._contradiction_score(opinioes[i], opinioes[j])
            score_ji = self._contradiction_score(opinioes[j], opinioes[i])
            score = max(score_ij, score_ji)

            if score >= self.contradiction_threshold:
                found.append(
                    ContradictionPair(
                        opiniao_a=opinioes[i],
                        opiniao_b=opinioes[j],
                        indice_a=i,
                        indice_b=j,
                        topic_similarity=topic_sim,
                        contradiction_score=score,
                    )
                )

        return found
