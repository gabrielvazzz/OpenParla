"""Enriquece opiniões com os trechos REAIS da transcrição da audiência.

As opiniões do `metadados` do LDS são resumos em 3ª pessoa gerados a
partir da notícia — não é a fala literal. Este módulo faz o caminho
inverso: dado o texto de uma opinião, localiza na `transcricao` da mesma
sessão os blocos de fala mais próximos lexicamente (equivalente à ideia
de `chunks_proximos` do NLI) e devolve até `k` deles.

O casamento é determinístico e auditável:
1. divide a transcrição em blocos (separados por linhas em branco);
2. normaliza tokens (minúsculas, sem acento, sem stopwords);
3. pontua cada bloco pela sobreposição de termos com a opinião, com leve
   penalidade por tamanho do bloco;
4. devolve os `k` melhores blocos, na ordem em que aparecem no texto.

Não exige modelo de linguagem. A qualidade depende da sobreposição
lexical (paráfrases muito distantes da fala podem não casar) — use junto
do NLI para casos de dúvida.
"""

from __future__ import annotations

import re
import unicodedata

K_TRECHOS = 4

_STOPWORDS = set("""
a o e é de do da dos das um uma umas uns em no na nos nas com por para
que se não mais ao aos à às ou como mas porém seu sua seus suas este
esta estes estas isso isto aquilo ele ela eles elas eu tu voce o sr sra
dr dra presidente senhor senhora sr senhora deputado deputada senador
vereador relativamente acerca conforme durante entre sobre sob desde até
também ainda já pois qual quais quanto quanta todos todas cada outro
outra outros outras muito muita muitos muitas pouco pouca poucos poucas
""".split())


def _normalizar_tokens(texto: str) -> list[str]:
    """Minúsculas, sem acentos, apenas tokens alfanuméricos de 3+ letras."""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    tokens = re.findall(r"[a-z0-9]+", texto.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def dividir_trechos(transcricao: str) -> list[str]:
    """Blocos de fala separados por linhas em branco, sem vazios."""
    blocos = [b.strip() for b in re.split(r"\n\s*\n", transcricao) if b.strip()]
    return blocos


def trechos_proximos(
    opiniao: str,
    blocos_tok: list[tuple[str, list[str]]],
    k: int = K_TRECHOS,
) -> list[str]:
    """Top-k blocos mais similares à opinião (sobreposição de termos)."""
    termos = set(_normalizar_tokens(opiniao))
    if not termos:
        return []

    pontuados: list[tuple[float, str]] = []
    for bloco, toks in blocos_tok:
        conjunto = set(toks)
        inter = len(termos & conjunto)
        if inter == 0:
            continue
        penalidade = 0.5 + len(conjunto) / 200.0
        pontuados.append((inter / penalidade, bloco))

    pontuados.sort(key=lambda x: x[0], reverse=True)
    return [bloco for _, bloco in pontuados[:k]]


def enriquecer(
    deputados: list[dict],
    sessoes_por_id: dict[int, str],
    k: int = K_TRECHOS,
) -> list[dict]:
    """Adiciona `trechos_transcricao` a cada opinião do deputado."""
    cache: dict[int, list[tuple[str, list[str]]]] = {}

    def trechos_de(texto: str, sessao_id) -> list[str]:
        if not texto or sessao_id is None:
            return []
        transcricao = sessoes_por_id.get(sessao_id)
        if not transcricao:
            return []
        if sessao_id not in cache:
            cache[sessao_id] = [
                (bloco, _normalizar_tokens(bloco))
                for bloco in dividir_trechos(transcricao)
            ]
        return trechos_proximos(texto, cache[sessao_id], k=k)

    enriquecidos = []
    for dep in deputados:
        opinioes = []
        for opiniao in dep.get("opinioes", []):
            novo = dict(opiniao)
            novo["trechos_transcricao"] = trechos_de(
                opiniao.get("opiniao", ""), opiniao.get("sessao_id")
            )
            opinioes.append(novo)
        novo_dep = dict(dep)
        novo_dep["opinioes"] = opinioes
        enriquecidos.append(novo_dep)
    return enriquecidos


def resumo_enriquecimento(deputados: list[dict]) -> tuple[int, int]:
    """(opiniões com trechos encontrados, total de opiniões)."""
    total = com_trechos = 0
    for dep in deputados:
        for opiniao in dep.get("opinioes", []):
            total += 1
            if opiniao.get("trechos_transcricao"):
                com_trechos += 1
    return com_trechos, total