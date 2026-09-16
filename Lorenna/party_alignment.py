"""Detecta oposição entre uma fala e a orientação atribuída ao partido.

Dois níveis de severidade, ambos calculados SEM modelos generativos:

- "contradicao": a fala e o partido ficam em lados opostos do mesmo eixo,
  com distância grande no espectro (critério conservador);
- "tensao": divergência menor (lados opostos, distância menor) capturada
  quando a posição do partido é avaliada PAUTA POR PAUTA. Isso permite pegar
  casos como um parlamentar do REPUBLICANOS criticando gastos com armas ou um
  do PL defendendo novo tributo, sem que seja um conflito ideológico total.

A posição por pauta vem de PARTY_ISSUE_ADJUSTMENTS sobre uma posição global
estimada por partido (PARTY_GLOBAL_POSITION). Assim como PARTIDO_IDEOLOGIA em
preparar_dados.py, é ponto de partida para revisão, não posição oficial.
"""

from __future__ import annotations

from dataclasses import dataclass

from ideology_classifier import (
    ClassificationResult,
    IdeologyClassifier,
    LABEL_POSITIONS,
)

MIN_SPEECH_STRENGTH = 0.25
MIN_POSITION_DISTANCE = 0.75
TENSION_DISTANCE = 0.50
TENSION_SPEECH_STRENGTH = 0.15

# ---------------------------------------------------------------------------
# Posição estimada por partido, na mesma escala do ideology_score (-1..+1).
# É a base: pautas específicas podem sobrescrever em PARTY_ISSUE_ADJUSTMENTS.
# ---------------------------------------------------------------------------

PARTY_GLOBAL_POSITION: dict[str, float] = {
    # Esquerda
    "PT": -1.0, "PCdoB": -1.0, "PSOL": -1.0, "PSTU": -1.0, "PCB": -1.0,
    "UP": -1.0,
    # Centro-esquerda
    "PDT": -0.5, "PSB": -0.5, "PV": -0.5, "REDE": -0.5, "CIDADANIA": -0.5,
    # Centro
    "MDB": 0.0, "PSD": 0.0, "PSDB": 0.0, "AVANTE": 0.0,
    "SOLIDARIEDADE": 0.0, "PODE": 0.0, "DC": 0.0, "PMB": 0.0, "PMN": 0.0,
    "AGIR": 0.0,
    # Centro-direita
    "UNIÃO": 0.5, "PRD": 0.75,
    # Direita
    "PL": 1.0, "PP": 0.75, "REPUBLICANOS": 0.75, "NOVO": 1.0,
    "PSC": 1.0, "PATRIOTA": 1.0, "DEM": 1.0, "PSL": 1.0, "PTB": 1.0,
    "PRTB": 1.0, "PRP": 1.0, "PROS": 1.0,
}

# Desvios conhecidos da posição global em pautas específicas. Edite aqui para
# calibrar com o comportamento real de cada partido nas suas pautas.
PARTY_ISSUE_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "PT": {"meio_ambiente": -0.5, "drogas": -0.5, "armas": -0.5},
    "PCdoB": {"aborto": -0.5, "direitos_lgbt": -0.5, "drogas": -0.5},
    "PSOL": {"meio_ambiente": -1.0, "armas": -1.0, "drogas": -1.0},
    "PV": {"meio_ambiente": -1.0, "direitos_lgbt": -0.5, "armas": -0.5},
    "CIDADANIA": {"aborto": -0.5, "drogas": -0.5, "direitos_lgbt": -0.5},
    "NOVO": {"meio_ambiente": 0.0, "drogas": -0.5, "direitos_lgbt": -0.5},
    "UNIÃO": {"aborto": 0.5, "armas": 0.5, "direitos_lgbt": 0.5},
    "PL": {"meio_ambiente": 0.5, "aborto": 1.0, "armas": 1.0, "drogas": 1.0,
           "cotas": 1.0, "direitos_lgbt": 1.0, "reforma_agraria": 1.0},
    "PP": {"meio_ambiente": 0.5, "aborto": 1.0, "armas": 0.75, "drogas": 0.75},
    "REPUBLICANOS": {"meio_ambiente": 0.5, "aborto": 1.0, "armas": 1.0,
                     "drogas": 1.0, "cotas": 1.0, "direitos_lgbt": 1.0},
    "PSDB": {"meio_ambiente": 0.0, "aborto": 0.0, "armas": 0.25,
             "drogas": 0.25, "tributacao": 0.5},
}


def party_position_for_issue(
    partido: str | None, issue: str | None
) -> float | None:
    """Posição do partido (escala -1..+1) na pauta, com fallback global."""
    if not partido:
        return None
    base = PARTY_GLOBAL_POSITION.get(partido)
    if base is None:
        return None
    if issue:
        ajuste = PARTY_ISSUE_ADJUSTMENTS.get(partido, {})
        if issue in ajuste:
            return ajuste[issue]
    return base


def party_global_label(partido: str | None) -> str | None:
    """Rótulo estável do partido, independente da pauta analisada."""
    score = PARTY_GLOBAL_POSITION.get(partido or "")
    return IdeologyClassifier._label_from_score(score) if score is not None else None


@dataclass
class PartyContradiction:
    indice: int
    fala: str
    sessao_id: int | None
    assunto: str | None
    posicionamento_partido: str
    posicionamento_fala: str
    score_partido: float
    score_fala: float
    distancia: float
    pauta: str
    evidencia_pauta: float
    severidade: str = "contradicao"
    origem: str = "classificador"

    def to_dict(self) -> dict:
        return {
            "indice": self.indice,
            "fala": self.fala,
            "sessao_id": self.sessao_id,
            "assunto": self.assunto,
            "severidade": self.severidade,
            "origem": self.origem,
            # O rótulo do partido é sempre sua posição global. O score mantém
            # o eventual ajuste específico da pauta usado na comparação.
            "posicionamento_partido": self.posicionamento_partido,
            "posicionamento_partido_pauta": self.pauta,
            "posicionamento_fala": self.posicionamento_fala,
            "score_partido": round(self.score_partido, 4),
            "score_fala": round(self.score_fala, 4),
            "distancia": round(self.distancia, 4),
            "pauta": self.pauta,
            "evidencia_pauta": round(self.evidencia_pauta, 4),
        }


def _montar_resultado(
    indice: int,
    opiniao,
    texto: str,
    resultado: ClassificationResult,
    score_partido: float,
    posicao_label: str,
    severidade: str,
) -> PartyContradiction:
    sessao_id = opiniao.get("sessao_id") if isinstance(opiniao, dict) else None
    assunto = opiniao.get("assunto") if isinstance(opiniao, dict) else None
    return PartyContradiction(
        indice=indice,
        fala=texto,
        sessao_id=sessao_id,
        assunto=assunto,
        posicionamento_partido=posicao_label,
        posicionamento_fala=resultado.label,
        score_partido=score_partido,
        score_fala=resultado.ideology_score,
        distancia=abs(score_partido - resultado.ideology_score),
        pauta=resultado.issue,
        evidencia_pauta=resultado.policy_evidence,
        severidade=severidade,
    )


def find_party_contradictions(
    party_label: str | None,
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    min_speech_strength: float = MIN_SPEECH_STRENGTH,
    min_position_distance: float = MIN_POSITION_DISTANCE,
) -> list[PartyContradiction]:
    """API legada: compara com a posição GLOBAL do partido (rótulo).

    Retorna somente oposições de lado fortes ("contradicao").
    """
    party_score = LABEL_POSITIONS.get(party_label or "")
    if party_score is None or party_score == 0:
        return []

    found = []
    for indice, (opiniao, texto, resultado) in enumerate(
        zip(opinioes, textos, resultados)
    ):
        if resultado.ideology_score is None or resultado.issue is None:
            continue
        if resultado.stance_strength < min_speech_strength:
            continue

        distance = abs(party_score - resultado.ideology_score)
        opposite_sides = party_score * resultado.ideology_score < 0
        if not opposite_sides or distance < min_position_distance:
            continue

        found.append(
            _montar_resultado(
                indice, opiniao, texto, resultado,
                score_partido=party_score,
                posicao_label=party_label,
                severidade="contradicao",
            )
        )
    return found


def find_party_divergences(
    partido: str | None,
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    min_speech_strength: float = MIN_SPEECH_STRENGTH,
    min_position_distance: float = MIN_POSITION_DISTANCE,
    tension_distance: float = TENSION_DISTANCE,
    tension_speech_strength: float = TENSION_SPEECH_STRENGTH,
) -> list[PartyContradiction]:
    """Detecta contradições (fortes) E tensões (leves) partido-fala.

    Usa a posição PAUTA POR PAUTA do partido. Partidos sem mapa (desconhecidos)
    ou de centro não geram resultado — centro não tem polo oposto definido.
    """
    found: list[PartyContradiction] = []

    for indice, (opiniao, texto, resultado) in enumerate(
        zip(opinioes, textos, resultados)
    ):
        speech_score = resultado.ideology_score
        if speech_score is None or resultado.issue is None:
            continue

        party_score = party_position_for_issue(partido, resultado.issue)
        if party_score is None or party_score == 0:
            continue

        distance = abs(party_score - speech_score)
        opposite_sides = party_score * speech_score < 0
        if not opposite_sides:
            continue

        label = party_global_label(partido)
        if label is None:
            continue

        if (
            resultado.stance_strength >= min_speech_strength
            and distance >= min_position_distance
        ):
            severidade = "contradicao"
        elif (
            resultado.stance_strength >= tension_speech_strength
            and distance >= tension_distance
        ):
            severidade = "tensao"
        else:
            continue

        found.append(
            _montar_resultado(
                indice, opiniao, texto, resultado,
                score_partido=party_score,
                posicao_label=label,
                severidade=severidade,
            )
        )
    return found
