"""
preparar_dados.py

Baixa o dataset público unicamp-dl/PublicHearingBR do Hugging Face,
lê APENAS a coluna "metadados" de cada sessão e gera um JSON no formato
que o pipeline espera (publichearingbr_ideology/pipeline.py):

    {
        "nome": "Filipe Barros",
        "cargo": "Deputado",
        "estado": "Paraná",
        "partido": "PL",
        "opinioes": [
            {"opiniao": "...", "sessao_id": 1, "assunto": "..."},
            {"opiniao": "...", "sessao_id": 39, "assunto": "..."}
        ],
        "posicionamento_politico_partido": "direita",
        "posicionamento_politico_fala": null
    }

Cada `metadados.envolvidos[i]` vira um registro de deputado. Deputados
que aparecem em várias sessões são agregados em um único registro, com
todas as opiniões deles somadas na lista "opinioes" junto com o id da
sessão e o assunto de onde cada fala veio — é isso que permite a
detecção de contradições "entre todas as falas de um mesmo deputado"
com rastreabilidade para a sessão de origem.

Partido e estado são extraídos do campo "cargo" (padrão "(SIGLA-UF)",
ex.: "Deputado (PL-AM)"). O posicionamento de partido vem de um mapa
partido → orientação (PARTIDO_IDEOLOGIA), editável no começo do
arquivo — é ponto de partida, não classificação oficial.

Como o dataset só existe como dois arquivos .jsonl (e a API `datasets`
do HF falha por schema misto LDS/NLI), baixamos o .jsonl diretamente
com `requests`. Se o arquivo já existir localmente, ele é reutilizado.

Uso:
    python preparar_dados.py \
        [--saida deputados.json] \
        [--arquivo-local PUBLICHEARINGBR_LDS.jsonl] \
        [--incluir-nao-deputados]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuração do dataset
# ---------------------------------------------------------------------------

REPO_ID = "unicamp-dl/PublicHearingBR"
ARQUIVO_DATASET = "PublicHearingBR_LDS.jsonl"
URL_DATASET = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ARQUIVO_DATASET}"
)

# Caminhos já existentes no repositório que podem conter o arquivo baixado.
CAMINHOS_LOCAIS_PADRAO = [
    Path("data") / ARQUIVO_DATASET,
    Path("../Leo/data") / ARQUIVO_DATASET,
    Path("../Leo/data") / "PublicHearingBR_LDS.jsonl",
]

# ---------------------------------------------------------------------------
# UFs e nomes por extenso
# ---------------------------------------------------------------------------

UFS = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}

# ---------------------------------------------------------------------------
# Partidos: sinônimos de grafia -> nome canônico
# ---------------------------------------------------------------------------

SINONIMOS_PARTIDO = {
    "uniao": "UNIÃO",
    "unión": "UNIÃO",
    "pode": "PODE",
    "podemos": "PODE",
    "podes": "PODE",
    "novo": "NOVO",
    "psol": "PSOL",
    "cidadania": "CIDADANIA",
    "republicanos": "REPUBLICANOS",
    "solidariedade": "SOLIDARIEDADE",
    "patriota": "PATRIOTA",
    "prd": "PRD",
    "psdb": "PSDB",
    "psb": "PSB",
    "psd": "PSD",
    "pt": "PT",
    "pl": "PL",
    "pp": "PP",
    "pv": "PV",
    "pcdob": "PCdoB",
    "pdt": "PDT",
    "mdb": "MDB",
    "dem": "DEM",
    "psl": "PSL",
    "pcb": "PCB",
    "pstu": "PSTU",
    "up": "UP",
    "rede": "REDE",
    "psc": "PSC",
    "prtb": "PRTB",
    "ptb": "PTB",
    "ptdo": "PTdoB",
    "avante": "AVANTE",
    "dc": "DC",
    "agir": "AGIR",
    "pmb": "PMB",
    "pmn": "PMN",
    "prp": "PRP",
    "pros": "PROS",
    "pr": "PR",
}

# ---------------------------------------------------------------------------
# Mapa partido -> orientação no espectro (mesmos rótulos usados pelo
# IdeologyClassifier: esquerda, centro-esquerda, centro, centro-direita,
# direita). Este é um ponto de partida razoável para o cenário político
# brasileiro — ajuste/complete conforme sua necessidade.
# ---------------------------------------------------------------------------

PARTIDO_IDEOLOGIA: dict[str, str] = {
    # Esquerda
    "PT": "esquerda", "PCdoB": "esquerda", "PSOL": "esquerda",
    "PSTU": "esquerda", "PCB": "esquerda", "UP": "esquerda",
    # Centro-esquerda
    "PDT": "centro-esquerda", "PSB": "centro-esquerda",
    "PV": "centro-esquerda", "REDE": "centro-esquerda",
    "CIDADANIA": "centro-esquerda",
    # Centro
    "MDB": "centro", "PSD": "centro", "PSDB": "centro",
    "AVANTE": "centro", "SOLIDARIEDADE": "centro", "PODE": "centro",
    "DC": "centro", "PMB": "centro", "PMN": "centro",
    "AGIR": "centro",
    # Centro-direita
    "UNIÃO": "centro-direita", "PRD": "centro-direita",
    # Direita
    "PL": "direita", "PP": "direita", "REPUBLICANOS": "direita",
    "NOVO": "direita", "PSC": "direita", "PATRIOTA": "direita",
    "DEM": "direita", "PSL": "direita", "PTB": "direita",
    "PRTB": "direita", "PRP": "direita", "PROS": "direita",
}

PARTIDO_DESCONHECIDO = "desconhecido"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Remove acentos, normaliza caixa e espaços (usado p/ agrupar nomes)."""
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip().lower()


def baixar_arquivo(url: str, destino: Path, chunk_size: int = 1 << 16) -> Path:
    """Baixa o arquivo com barra de progresso simples."""
    print(f"Baixando {url}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        baixado = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    print(f"\r  {baixado / total:.1%} ({baixado / 1e6:.1f} MB)",
                          end="", flush=True)
    print("\r  100% — concluído.              ")
    return destino


def obter_arquivo_local_ou_baixar(arquivo_local: str | None) -> Path:
    """Prioriza arquivo local se existir; senão, baixa do HuggingFace."""
    if arquivo_local:
        path = Path(arquivo_local)
        if path.exists():
            print(f"[ok] Usando arquivo local: {path}")
            return path
        print(f"[!] {path} não existe; baixando do HuggingFace.")

    for cand in CAMINHOS_LOCAIS_PADRAO:
        if cand.exists():
            print(f"[ok] Usando arquivo local: {cand.resolve()}")
            return cand

    destino = Path("data") / ARQUIVO_DATASET
    return baixar_arquivo(URL_DATASET, destino)


# ---------------------------------------------------------------------------
# Parsing do campo "cargo"
# ---------------------------------------------------------------------------

PADRAO_PARENTESES = re.compile(r"\(([^()]*)\)")

# Ex.: "Deputado (PL-AM)" -> partido "PL", UF "AM".
#     "Deputado (União-CE)" -> partido "União" (-> UNIÃO), UF "CE".
# Também "Deputado distrital (Psol)" -> partido "Psol", sem UF.


def extrair_partido_estado(cargo: str) -> tuple[str | None, str | None]:
    """Retorna (partido, uf) extraídos de um cargo tipo '(Partido-UF)'."""
    for grupo in PADRAO_PARENTESES.findall(cargo):
        partes = [p.strip() for p in grupo.split("-")]
        if len(partes) == 2 and partes[1] in UFS:
            partido = _canonico(partes[0])
            if partido:
                return partido, partes[1]
        # Caso "(Partido)" sem UF
        partido = _canonico(grupo)
        if partido:
            return partido, None
    return None, None


def _canonico(nome: str) -> str | None:
    if not nome:
        return None
    chave = normalizar(nome)
    return SINONIMOS_PARTIDO.get(chave)


def eh_deputado(cargo: str) -> bool:
    return "deputad" in cargo.lower()


# ---------------------------------------------------------------------------
# Leitura do dataset (coluna "metadados")
# ---------------------------------------------------------------------------

def carregar_sessoes(caminho: Path) -> list[dict]:
    sessoes = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            sessoes.append(json.loads(linha))
    return sessoes


def processar_metadados(
    sessoes: list[dict],
    incluir_nao_deputados: bool = False,
) -> list[dict]:
    """
    Percorre apenas a coluna "metadados" de cada sessão e monta a lista
    de deputados agregados (nome -> todos as suas opiniões com sessão).
    """
    # chave de agregação (nome normalizado) -> dados do registro
    agrupado: dict[str, dict] = {}

    for sessao in sessoes:
        sessa_id = sessao.get("id")
        metadados = sessao.get("metadados") or {}
        assunto = metadados.get("assunto", "")
        envolvidos = metadados.get("envolvidos", [])

        for env in envolvidos:
            cargo = env.get("cargo", "") or ""
            nome = (env.get("nome", "") or "").strip()

            if not nome:
                continue
            if not incluir_nao_deputados and not eh_deputado(cargo):
                continue

            chave = normalizar(nome)
            reg = agrupado.get(chave)

            if reg is None:
                partido, uf = extrair_partido_estado(cargo)
                reg = {
                    "nome": nome,
                    "cargo": cargo,
                    "estado": UFS.get(uf, ""),
                    "partido": partido or "",
                    "opinioes": [],
                }
                agrupado[chave] = reg

            opinioes = env.get("opinioes", []) or []
            for opiniao in opinioes:
                if not opiniao or not opiniao.strip():
                    continue
                reg["opinioes"].append(
                    {
                        "opiniao": opiniao,
                        "sessao_id": sessa_id,
                        "assunto": assunto,
                    }
                )

    # Monta o JSON final no formato do pipeline.
    deputados = []
    for reg in agrupado.values():
        # Remove opiniões exatamente duplicadas (mantém a primeira).
        unicas, vistas = [], set()
        for opiniao_obj in reg["opinioes"]:
            texto = opiniao_obj["opiniao"]
            if texto not in vistas:
                vistas.add(texto)
                unicas.append(opiniao_obj)

        partido = reg["partido"]
        ideologia = PARTIDO_IDEOLOGIA.get(partido)

        deputados.append(
            {
                "nome": reg["nome"],
                "cargo": reg["cargo"],
                "estado": reg["estado"],
                "partido": partido,
                "opinioes": unicas,
                # Campo preenchido pelo pipeline depois que roda:
                "posicionamento_politico_partido": ideologia or PARTIDO_DESCONHECIDO,
                "posicionamento_politico_fala": None,
            }
        )

    return deputados


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa o PublicHearingBR (HF) e gera o JSON de entrada "
                    "do pipeline de posicionamento político."
    )
    parser.add_argument("--saida", default="deputados.json")
    parser.add_argument(
        "--arquivo-local",
        default=None,
        help="Caminho para o .jsonl já baixado (usa em vez de baixar).",
    )
    parser.add_argument(
        "--incluir-nao-deputados",
        action="store_true",
        help="Inclui também ministros, jornalistas e outros envolvidos "
             "(por padrão, mantém apenas quem tem 'Deputad' no cargo).",
    )
    args = parser.parse_args()

    caminho = obter_arquivo_local_ou_baixar(args.arquivo_local)

    print(f"Carregando sessões de {caminho} ...")
    sessoes = carregar_sessoes(caminho)
    print(f"[ok] {len(sessoes)} sessões carregadas.")

    deputados = processar_metadados(sessoes, args.incluir_nao_deputados)

    n_opinioes = sum(len(d["opinioes"]) for d in deputados)
    n_sem_partido = sum(1 for d in deputados if not d["partido"])
    n_sem_ideologia = sum(
        1 for d in deputados
        if d["posicionamento_politico_partido"] == PARTIDO_DESCONHECIDO
    )

    print(f"[ok] {len(deputados)} deputados, {n_opinioes} opiniões no total.")
    print(f"[ok] {n_sem_partido} sem partido extraído do cargo; "
          f"{n_sem_ideologia} sem ideologia de partido mapeada.")

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump(deputados, f, ensure_ascii=False, indent=2)

    print(f"[ok] Salvo em {args.saida}")


if __name__ == "__main__":
    main()