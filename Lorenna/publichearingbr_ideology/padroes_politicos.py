"""Padrões de posicionamento além da comparação fala vs. partido.

Duas extensões da detecção básica (contradição fala-fala e oposição
fala-partido), ambas SEM modelos generativos:

1. Tensão transversal (coerência cruzada entre pautas): o mesmo deputado
   acumula posições fortes em lados OPOSTOS em pautas DIFERENTES (ex.:
   defende gasto social ampliado numa fala e, em outra, critica o mesmo
   tipo de aporte no orçamento). Dentro de uma única pauta isso seria
   contradição direta; como as falas estão em pautas distintas, o
   ContradictionDetector não as compara — este módulo faz a comparação
   cruzada e retorna um sinal leve ("tensao").

2. Fala defensiva (negação de rótulo): padrão léxico "não somos/sou/são +
   rótulo" (ex.: "não somos antivacinas"). Sinaliza que o parlamentar
   gasta energia NEGANDO uma acusação em vez de declarar posição — útil
   para não tratar a mera negação como posição firme no eixo ideológico.

Ambos retornam apenas sinal fraco ("tensao"); decisões de contradição
forte continuam a cargo dos detectores conservadores já existentes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from ideology_classifier import ClassificationResult


# ---------------------------------------------------------------------------
# 1) Tensão transversal: posições fortes e opostas em pautas diferentes
# ---------------------------------------------------------------------------

# |média do score| mínimo para considerar a pauta "posicionada".
CROSS_ISSUE_MIN_POSITION = 0.25
# Força mínima média para não flagrar ruído no classificador.
CROSS_ISSUE_MIN_STRENGTH = 0.15


@dataclass
class CrossIssueTension:
    pauta_esquerda: str
    pauta_direita: str
    score_esquerda: float
    score_direita: float
    forca_esquerda: float
    forca_direita: float
    falas_esquerda: list[int]
    falas_direita: list[int]
    severidade: str = "tensao"

    def to_dict(self) -> dict:
        return {
            "pauta_esquerda": self.pauta_esquerda,
            "pauta_direita": self.pauta_direita,
            "score_esquerda": round(self.score_esquerda, 4),
            "score_direita": round(self.score_direita, 4),
            "forca_esquerda": round(self.forca_esquerda, 4),
            "forca_direita": round(self.forca_direita, 4),
            "indices_falas_esquerda": self.falas_esquerda,
            "indices_falas_direita": self.falas_direita,
            "severidade": self.severidade,
        }


def find_cross_issue_tensions(
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    min_position: float = CROSS_ISSUE_MIN_POSITION,
    min_strength: float = CROSS_ISSUE_MIN_STRENGTH,
) -> list[CrossIssueTension]:
    """Agrupa falas por pauta e sinaliza deputados com polos opostos
    em pautas DIFERENTES (posição alta de esquerda e de direita)."""
    por_pauta: dict[str, dict] = defaultdict(
        lambda: {"scores": [], "forcas": [], "indices": []}
    )
    for indice, (texto, resultado) in enumerate(zip(textos, resultados)):
        if resultado.ideology_score is None or resultado.issue is None:
            continue
        info = por_pauta[resultado.issue]
        info["scores"].append(resultado.ideology_score)
        info["forcas"].append(resultado.stance_strength)
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


# ---------------------------------------------------------------------------
# 2) Fala defensiva: negação de rótulo ("não somos antivacinas")
# ---------------------------------------------------------------------------

# Rótulos/posturas que, quando negados, indicam defesa em vez de posição.
# Mantido pequeno de propósito para evitar falso-positivos; ordene do mais
# longo para o mais curto (alternância preferencial do regex).
TERMOS_NEGADOS: tuple[str, ...] = (
    "a favor", "anti-vacina", "antivacina", "favorável", "favoravel",
    "antivax",
    "contra", "comunista", "fascista", "racista", "homofóbico",
    "homofobico", "machista", "nazista", "golpista", "corrupto",
    "corrupta", "extremista", "radical", "negacionista", "terrorista",
    "populista", "sectário", "sectario", "autoritário", "autoritario",
    "ditador", "bandido", "ladrão", "ladrao",
)

PADRAO_NEGACAO = re.compile(
    r"\bnão\s+(?:(?:somos|sou|é|são|fomos|seria|seriam)\s+)?"
    r"(?P<aspas>['\"“”«`-]*)"
    r"(?P<termo>"
    + "|".join(re.escape(t) + r"(?:s)?" for t in TERMOS_NEGADOS)
    + r")\b",
    re.IGNORECASE,
)


@dataclass
class DefensiveSpeech:
    indice: int
    fala: str
    sessao_id: int | None
    assunto: str | None
    termo_negado: str
    expressao: str
    pauta: str | None
    posicao_fala: str | None
    severidade: str = "tensao"

    def to_dict(self) -> dict:
        return {
            "indice": self.indice,
            "fala": self.fala,
            "sessao_id": self.sessao_id,
            "assunto": self.assunto,
            "termo_negado": self.termo_negado,
            "expressao": self.expressao,
            "pauta": self.pauta,
            "posicao_fala": self.posicao_fala,
            "severidade": self.severidade,
        }


def detect_defensive_speech(
    opinioes: list,
    textos: list[str],
    resultados: list[ClassificationResult],
    textos_contexto: list[str] | None = None,
) -> list[DefensiveSpeech]:
    """Sinaliza falas com o padrão "não +(somos/sou/...) + rótulo".

    ``textos_contexto`` permite examinar a transcrição sem substituir a fala
    resumida que será exibida no relatório.
    """
    encontrados: list[DefensiveSpeech] = []
    fontes = textos_contexto if textos_contexto is not None else textos
    for indice, (opiniao, texto, contexto, resultado) in enumerate(
        zip(opinioes, textos, fontes, resultados)
    ):
        for m in PADRAO_NEGACAO.finditer(contexto):
            sessao_id = (
                opiniao.get("sessao_id") if isinstance(opiniao, dict) else None
            )
            assunto = opiniao.get("assunto") if isinstance(opiniao, dict) else None
            encontrados.append(
                DefensiveSpeech(
                    indice=indice,
                    fala=texto,
                    sessao_id=sessao_id,
                    assunto=assunto,
                    termo_negado=m.group("termo").lower(),
                    expressao=m.group(0),
                    pauta=resultado.issue,
                    posicao_fala=resultado.label,
                )
            )
            break
    return encontrados
