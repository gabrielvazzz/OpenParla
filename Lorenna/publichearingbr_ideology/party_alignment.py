"""Detecta oposição entre uma fala e a orientação atribuída ao partido."""

from __future__ import annotations

from dataclasses import dataclass

from ideology_classifier import ClassificationResult, LABEL_POSITIONS


MIN_SPEECH_STRENGTH = 0.25
MIN_POSITION_DISTANCE = 0.75


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

    def to_dict(self) -> dict:
        return {
            "indice": self.indice,
            "fala": self.fala,
            "sessao_id": self.sessao_id,
            "assunto": self.assunto,
            "posicionamento_partido": self.posicionamento_partido,
            "posicionamento_fala": self.posicionamento_fala,
            "score_partido": round(self.score_partido, 4),
            "score_fala": round(self.score_fala, 4),
            "distancia": round(self.distancia, 4),
            "pauta": self.pauta,
            "evidencia_pauta": round(self.evidencia_pauta, 4),
        }


def find_party_contradictions(
    party_label: str | None,
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    min_speech_strength: float = MIN_SPEECH_STRENGTH,
    min_position_distance: float = MIN_POSITION_DISTANCE,
) -> list[PartyContradiction]:
    """Retorna somente oposições de lado, não meras diferenças de intensidade.

    Um partido de centro não gera contradição automática porque não há um polo
    oposto bem definido. Falas neutras ou sem pauta política reconhecida também
    não são reportadas.
    """
    party_score = LABEL_POSITIONS.get(party_label or "")
    if party_score is None or party_score == 0:
        return []

    found = []
    for indice, (opiniao, texto, resultado) in enumerate(
        zip(opinioes, textos, resultados)
    ):
        speech_score = resultado.ideology_score
        if speech_score is None or resultado.issue is None:
            continue
        if resultado.stance_strength < min_speech_strength:
            continue

        distance = abs(party_score - speech_score)
        opposite_sides = party_score * speech_score < 0
        if not opposite_sides or distance < min_position_distance:
            continue

        sessao_id = opiniao.get("sessao_id") if isinstance(opiniao, dict) else None
        assunto = opiniao.get("assunto") if isinstance(opiniao, dict) else None
        found.append(
            PartyContradiction(
                indice=indice,
                fala=texto,
                sessao_id=sessao_id,
                assunto=assunto,
                posicionamento_partido=party_label,
                posicionamento_fala=resultado.label,
                score_partido=party_score,
                score_fala=speech_score,
                distancia=distance,
                pauta=resultado.issue,
                evidencia_pauta=resultado.policy_evidence,
            )
        )
    return found
