"""Referência editorial, versionada, de posição ideológica dos partidos.

Não representa uma classificação oficial. A referência deve ser revisada antes
de uma análise pública e cada resultado grava a versão usada.
"""

from __future__ import annotations

import re
import unicodedata

VERSAO_MAPA_PARTIDOS = "2026-09-inicial"
FONTE_MAPA_PARTIDOS = (
    "Referência pública inicial, revisável; conferir programas/estatutos "
    "partidários antes de divulgar resultados."
)
ESCALA_IDEOLOGICA = {
    "esquerda": 0,
    "centro-esquerda": 1,
    "centro": 2,
    "centro-direita": 3,
    "direita": 4,
}

PARTIDO_POSICIONAMENTO = {
    "PT": "esquerda", "PCdoB": "esquerda", "PSOL": "esquerda",
    "PSTU": "esquerda", "PCB": "esquerda", "UP": "esquerda",
    "PDT": "centro-esquerda", "PSB": "centro-esquerda",
    "PV": "centro-esquerda", "REDE": "centro-esquerda",
    "CIDADANIA": "centro-esquerda",
    "MDB": "centro", "PSD": "centro", "PSDB": "centro",
    "AVANTE": "centro", "SOLIDARIEDADE": "centro", "PODE": "centro",
    "DC": "centro", "PMB": "centro", "PMN": "centro", "AGIR": "centro",
    "UNIÃO": "centro-direita", "PRD": "centro-direita",
    "PL": "direita", "PP": "direita", "REPUBLICANOS": "direita",
    "NOVO": "direita", "PSC": "direita", "PATRIOTA": "direita",
    "DEM": "direita", "PSL": "direita", "PTB": "direita",
    "PRTB": "direita", "PRP": "direita", "PROS": "direita",
}

SINONIMOS_PARTIDO = {
    "uniao": "UNIÃO", "uniao brasil": "UNIÃO", "podemos": "PODE",
    "pode": "PODE", "novo": "NOVO", "psol": "PSOL", "pcdob": "PCdoB",
    "republicanos": "REPUBLICANOS", "cidadania": "CIDADANIA",
    "solidariedade": "SOLIDARIEDADE", "patriota": "PATRIOTA",
    "movimento democratico brasileiro": "MDB",
}


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto or "")
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip().lower()


def partido_canonico(valor: str | None) -> str | None:
    if not valor:
        return None
    chave = normalizar(valor)
    if chave in SINONIMOS_PARTIDO:
        return SINONIMOS_PARTIDO[chave]
    for partido in PARTIDO_POSICIONAMENTO:
        if normalizar(partido) == chave:
            return partido
    return None
