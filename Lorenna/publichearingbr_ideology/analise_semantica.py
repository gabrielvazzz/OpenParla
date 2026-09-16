"""Classificação e detecção de contradição guiadas por embeddings.

Embeddings são a primeira etapa em ambos os casos: ideologia é determinada por
similaridade com centróides semânticos e pares só chegam ao NLI se forem
semanticamente próximos. O NLI apenas refina os candidatos, evitando comparar
todo par de falas sem relação temática.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline as criar_pipeline_nli


MODELO_IDEOLOGIA = "intfloat/multilingual-e5-large-instruct"
MODELO_TOPICO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODELO_NLI = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
INSTRUCAO = (
    "Given a Brazilian parliamentary statement, identify its political "
    "orientation on a left-right spectrum from the substantive policy stance."
)

# Conjunto inicial: deve crescer apenas com exemplos revisados por pessoas.
ANCORAS = {
    "esquerda": [
        "Defendo ampliar os direitos trabalhistas e fortalecer os sindicatos.",
        "O Estado deve garantir saúde e educação públicas universais, financiadas por tributação progressiva.",
        "Defendo reforma agrária, políticas afirmativas e proteção rigorosa de comunidades indígenas.",
    ],
    "centro-esquerda": [
        "Defendo um Estado regulador forte e parcerias privadas quando beneficiarem a população.",
        "Apoio transferência de renda e investimento público com responsabilidade social.",
        "Defendo metas ambientais rígidas negociadas em uma transição justa.",
    ],
    "centro": [
        "É preciso equilíbrio fiscal sem abandonar investimentos sociais essenciais.",
        "Cada proposta deve ser avaliada pelo mérito técnico e pelo diálogo entre setores.",
        "Defendo um Estado eficiente, nem mínimo nem inchado.",
    ],
    "centro-direita": [
        "Defendo responsabilidade fiscal, abertura comercial e uma rede básica de proteção social.",
        "Apoio concessões e parcerias público-privadas para melhorar a infraestrutura.",
        "O agronegócio deve ser um vetor de desenvolvimento com regras ambientais previsíveis.",
    ],
    "direita": [
        "Defendo privatizações, livre comércio e menor interferência do Estado na economia.",
        "Defendo redução de impostos, desburocratização e liberdade econômica.",
        "Sou contrário ao aborto e defendo penas mais rigorosas contra a criminalidade.",
    ],
    "neutra": [
        "Passo a palavra ao próximo orador desta audiência pública.",
        "A sessão será suspensa para verificação de quórum.",
        "Registro que o relatório foi protocolado na mesa diretora.",
    ],
}


def formatar(texto: str) -> str:
    return f"Instruct: {INSTRUCAO}\nQuery: {texto}"


@dataclass
class ResultadoIdeologia:
    rotulo: str
    scores: dict[str, float]
    margem: float
    confianca: float

    def serializar(self) -> dict:
        return {
            "rotulo": self.rotulo,
            "confianca": round(self.confianca, 4),
            "margem": round(self.margem, 4),
            "scores": {chave: round(valor, 4) for chave, valor in self.scores.items()},
        }


class ClassificadorIdeologia:
    def __init__(self, modelo: str = MODELO_IDEOLOGIA, margem_minima: float = 0.015,
                 similaridade_minima: float = 0.78, device: str | None = None):
        self.modelo_nome = modelo
        self.modelo = SentenceTransformer(modelo, device=device)
        self.margem_minima = margem_minima
        self.similaridade_minima = similaridade_minima
        self.centroides = self._construir_centroides()

    def _codificar(self, textos: list[str]) -> np.ndarray:
        return self.modelo.encode([formatar(t) for t in textos], normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)

    def _construir_centroides(self) -> dict[str, np.ndarray]:
        centroides = {}
        for rotulo, exemplos in ANCORAS.items():
            centroide = self._codificar(exemplos).mean(axis=0)
            centroides[rotulo] = centroide / np.linalg.norm(centroide)
        return centroides

    def classificar_lote(self, textos: list[str]) -> list[ResultadoIdeologia]:
        if not textos:
            return []
        embeddings = self._codificar(textos)
        resultados = []
        for embedding in embeddings:
            scores = {rotulo: float(np.dot(embedding, centroide))
                      for rotulo, centroide in self.centroides.items()}
            ordenados = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            melhor, score_1 = ordenados[0]
            score_2 = ordenados[1][1]
            margem = score_1 - score_2
            rotulo = melhor
            # Fala incerta é neutra; uma fala explicitamente procedural também
            # pode vencer a âncora neutra e continua neutra.
            if melhor != "neutra" and (margem < self.margem_minima or score_1 < self.similaridade_minima):
                rotulo = "neutra"
            resultados.append(ResultadoIdeologia(rotulo, scores, margem, score_1))
        return resultados


class DetectorContradicoes:
    def __init__(self, modelo_topico: str = MODELO_TOPICO, modelo_nli: str = MODELO_NLI,
                 similaridade_topico_minima: float = 0.58,
                 contradicao_minima: float = 0.72, device: str | None = None):
        self.modelo_topico_nome = modelo_topico
        self.modelo_nli_nome = modelo_nli
        self.modelo_topico = SentenceTransformer(modelo_topico, device=device)
        self.similaridade_topico_minima = similaridade_topico_minima
        self.contradicao_minima = contradicao_minima
        self.nli = criar_pipeline_nli("text-classification", model=modelo_nli, top_k=None,
                                      device=0 if device == "cuda" else -1)

    def _score_contradicao(self, premissa: str, hipotese: str) -> float:
        saida = self.nli(f"{premissa}</s></s>{hipotese}", truncation=True)
        entradas = saida[0] if saida and isinstance(saida[0], list) else saida
        for entrada in entradas:
            if "contradiction" in entrada["label"].lower():
                return float(entrada["score"])
        return 0.0

    def encontrar(self, falas: list[dict]) -> list[dict]:
        """Retorna somente pares semanticamente próximos e incompatíveis."""
        if len(falas) < 2:
            return []
        textos = [fala["texto"] for fala in falas]
        vetores = self.modelo_topico.encode(textos, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False)
        casos = []
        for indice_a, indice_b in combinations(range(len(falas)), 2):
            similaridade = float(np.dot(vetores[indice_a], vetores[indice_b]))
            if similaridade < self.similaridade_topico_minima:
                continue
            score_ab = self._score_contradicao(textos[indice_a], textos[indice_b])
            score_ba = self._score_contradicao(textos[indice_b], textos[indice_a])
            score = max(score_ab, score_ba)
            if score >= self.contradicao_minima:
                casos.append({
                    "fala_a": falas[indice_a],
                    "fala_b": falas[indice_b],
                    "similaridade_topica": round(similaridade, 4),
                    "score_contradicao": round(score, 4),
                    "score_a_para_b": round(score_ab, 4),
                    "score_b_para_a": round(score_ba, 4),
                    "status": "revisao_necessaria",
                })
        return casos
