"""Camada léxica determinística de INDÍCIOS de posicionamento.

O classificador (e5) frequentemente devolve "neutra" para falas do
PublicHearingBR, porque são resumos narrativos em 3ª pessoa sobre temas
muito específicos (FGTS, tarifas de energia, apostas, artes marciais...)
que não se encaixam nos 13 eixos de pauta das âncoras. Sem issue/polo, os
detectores de partido e de tensão transversal silenciam.

Este módulo preenche esse buraco com regras lexicais curtas e auditáveis:
palavras/frames que indicam pauta + lado do espectro (ex.: "se gasta tanto
com armas" -> armas/esquerda; "defendeu a taxação das apostas" ->
tributacao/esquerda). Resultado é sempre sinal FRACO ("tensao"), nunca
"contradicao" — o rol exato de casos (Márcio Marinho, Mauricio do Vôlei,
Danilo Forte, Ivan Valente) é verificado nos testes com os textos reais.

As regras NÃO são exaustivas: pegar os padrões que os casos borderline
conhecidos exigem, sem disparar em falas procedurais. Calibre adicionando/
removendo entradas em _RULES.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace

from ideology_classifier import ClassificationResult, IdeologyClassifier
from party_alignment import (
    PartyContradiction,
    party_global_label,
    party_position_for_issue,
)
from padroes_politicos import (
    CrossIssueTension,
    CROSS_ISSUE_MIN_POSITION,
    CROSS_ISSUE_MIN_STRENGTH,
)


@dataclass
class Indicio:
    issue: str
    lado: int  # -1 = esquerda, +1 = direita
    forca: float
    descricao: str

    @property
    def score(self) -> float:
        return self.lado * self.forca


# (regex, issue, lado, forca, descricao). Ordem importa: primeiro match ganha.
_RULES: list[tuple[re.Pattern, str, int, float, str]] = [
    (
        re.compile(
            r"(?:criticou|critica).{0,100}(?:suspens[ãa]o|restriç(?:ão|ões)"
            r"|proibiç(?:ão|ões)).{0,140}(?:arma|armamento|porte|posse|aquisiç)",
            re.IGNORECASE,
        ),
        "armas", +1, 0.4, "crítica a restrição de acesso a armas",
    ),
    (
        re.compile(
            r"(?:se gasta|gastamos|gasta|criticou|contra).{0,60}"
            r"(?:armas|armamento|armando|cidadão armado)",
            re.IGNORECASE,
        ),
        "armas", -1, 0.4, "crítica a gastos com armas",
    ),
    (
        re.compile(
            r"(?:defendo|favor).{0,30}(?:armamento|porte de armas|armas)",
            re.IGNORECASE,
        ),
        "armas", +1, 0.4, "defesa do armamento",
    ),
    (
        re.compile(
            r"(?:defendeu a taxação|a favor da taxação|favorável à taxação"
            r"|publicidade.{0,40}apostas)",
            re.IGNORECASE,
        ),
        "tributacao", -1, 0.4, "defesa de nova tributação",
    ),
    (
        re.compile(
            r"(?:aumento de tributos|aumento de impostos|tributos|impostos)"
            r".{0,120}(?:impopular|não têm a simpatia|rejeit)"
            r"|(?:impopular|não têm a simpatia).{0,120}"
            r"(?:aumento de tributos|tributos|impostos)",
            re.IGNORECASE,
        ),
        "tributacao", +1, 0.4, "resistência a aumento de tributos",
    ),
    (
        re.compile(
            r"(?:Tarifa Social|transferência de renda|subsídio"
            r"|benefícios sociais|programas sociais)",
            re.IGNORECASE,
        ),
        "programas_sociais", -1, 0.4, "defesa de subsídio social",
    ),
    (
        re.compile(
            r"capacidade estratégica da Petrobras|aposta na produção"
            r" de petróleo|autossuficiência.{0,40}petróleo",
            re.IGNORECASE,
        ),
        "meio_ambiente", +1, 0.4, "defesa da produção petrolífera",
    ),
    (
        re.compile(
            r"(?:equilibrar|equilíbrio).{0,60}(?:sustentabilidade|meio ambiente"
            r"|ambiental)|sustentabilidade ambiental",
            re.IGNORECASE,
        ),
        "meio_ambiente", -1, 0.3, "moderação entre produção e ambiente",
    ),
]


def indicio_da_fala(texto: str) -> Indicio | None:
    """Primeira regra que casar; None se a fala não tem indício mapeado."""
    for regex, issue, lado, forca, desc in _RULES:
        if regex.search(texto):
            return Indicio(issue=issue, lado=lado, forca=forca, descricao=desc)
    return None


def reconciliar_resultado_com_indicio(
    resultado: ClassificationResult, indicio: Indicio | None
) -> ClassificationResult:
    """Corrige uma inversão do modelo quando uma regra específica a contradiz."""
    if (
        indicio is None
        or resultado.issue != indicio.issue
        or resultado.ideology_score is None
        or resultado.ideology_score * indicio.score >= 0
    ):
        return resultado

    score = indicio.score
    scores = IdeologyClassifier._label_scores(score)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return replace(
        resultado,
        label=ranked[0][0],
        scores=scores,
        margin=ranked[0][1] - ranked[1][1],
        top1=ranked[0][1],
        ideology_score=score,
        stance_strength=abs(score),
    )


def party_divergences_from_indicios(
    partido: str | None,
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    indicios: list[Indicio | None],
) -> list[PartyContradiction]:
    """Oposição partido-fala via indício léxico, capada em "tensao".

    Só considera falas cujo indício não foi coberto pelo classificador
    (issue do e5 ausente ou diferente da pauta do indício).
    """
    found: list[PartyContradiction] = []
    for indice, (opiniao, texto, resultado, ind) in enumerate(
        zip(opinioes, textos, resultados, indicios)
    ):
        if ind is None:
            continue
        if resultado.issue is not None and resultado.issue == ind.issue:
            continue

        party_score = party_position_for_issue(partido, ind.issue)
        if party_score is None or party_score == 0:
            continue
        score_fala = ind.score
        if party_score * score_fala >= 0:
            continue  # mesmo lado do partido

        sessao_id = opiniao.get("sessao_id") if isinstance(opiniao, dict) else None
        assunto = opiniao.get("assunto") if isinstance(opiniao, dict) else None
        label = party_global_label(partido)
        if label is None:
            continue
        found.append(
            PartyContradiction(
                indice=indice,
                fala=texto,
                sessao_id=sessao_id,
                assunto=assunto,
                posicionamento_partido=label,
                posicionamento_fala=resultado.label,
                score_partido=party_score,
                score_fala=score_fala,
                distancia=abs(party_score - score_fala),
                pauta=ind.issue,
                evidencia_pauta=1.0,
                severidade="tensao",
                origem="indicio_lexical",
            )
        )
    return found


def cross_issue_tensions_from_indicios(
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    indicios: list[Indicio | None],
    min_position: float = CROSS_ISSUE_MIN_POSITION,
    min_strength: float = CROSS_ISSUE_MIN_STRENGTH,
) -> list[CrossIssueTension]:
    """Tensão transversal usando indícios quando o classificador não viu pauta."""
    por_pauta: dict[str, dict] = defaultdict(
        lambda: {"scores": [], "forcas": [], "indices": []}
    )
    for indice, (opiniao, resultado, ind) in enumerate(
        zip(opinioes, resultados, indicios)
    ):
        if ind is None:
            continue
        if resultado.issue is not None and resultado.issue == ind.issue:
            continue
        info = por_pauta[ind.issue]
        info["scores"].append(ind.score)
        info["forcas"].append(ind.forca)
        info["indices"].append(indice)

    esquerda, direita = [], []
    for pauta, info in por_pauta.items():
        n = len(info["scores"])
        if n == 0:
            continue
        media_score = sum(info["scores"]) / n
        media_forca = sum(info["forcas"]) / n
        if media_forca < min_strength:
            continue
        if media_score <= -min_position:
            esquerda.append((media_score, pauta, media_forca, info["indices"]))
        elif media_score >= min_position:
            direita.append((media_score, pauta, media_forca, info["indices"]))

    tensoes = [
        CrossIssueTension(
            pauta_esquerda=pauta_e,
            pauta_direita=pauta_d,
            score_esquerda=score_e,
            score_direita=score_d,
            forca_esquerda=forca_e,
            forca_direita=forca_d,
            falas_esquerda=idx_e,
            falas_direita=idx_d,
        )
        for score_e, pauta_e, forca_e, idx_e in esquerda
        for score_d, pauta_d, forca_d, idx_d in direita
    ]
    tensoes.sort(
        key=lambda t: abs(t.score_esquerda - t.score_direita), reverse=True
    )
    return tensoes
