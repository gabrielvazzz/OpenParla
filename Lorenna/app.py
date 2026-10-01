from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DIRETORIO_SAIDAS = BASE_DIR / "saidas"
MODELO_TOPICO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class OutputRepository:

    def __init__(self, diretorio_saidas: Path = DIRETORIO_SAIDAS):
        self.diretorio_saidas = diretorio_saidas
        self._opinioes: list[dict] | None = None

    def arquivos(self) -> list[str]:
        if not self.diretorio_saidas.is_dir():
            return []
        return sorted(path.name for path in self.diretorio_saidas.glob("*.json"))

    def carregar(self, arquivo: str):
        if arquivo not in self.arquivos():
            raise KeyError(arquivo)
        with (self.diretorio_saidas / arquivo).open(encoding="utf-8") as file:
            return json.load(file)

    def opinioes(self) -> list[dict]:
        if self._opinioes is not None:
            return self._opinioes

        arquivo = (
            "deputados_classificados.json"
            if "deputados_classificados.json" in self.arquivos()
            else "deputados.json"
        )
        try:
            deputados = self.carregar(arquivo)
        except KeyError as error:
            raise FileNotFoundError("deputados.json não encontrado em saidas") from error

        opinioes = []
        for deputado_indice, deputado in enumerate(deputados):
            classificacoes = deputado.get("posicionamento_politico_fala_detalhe", [])
            for fala_indice, opiniao in enumerate(deputado.get("opinioes", [])):
                if isinstance(opiniao, dict):
                    texto = opiniao.get("opiniao", "")
                    sessao_id = opiniao.get("sessao_id")
                    assunto = opiniao.get("assunto")
                    trechos = opiniao.get("trechos_transcricao", [])
                else:
                    texto, sessao_id, assunto, trechos = opiniao, None, None, []
                if not isinstance(texto, str) or not texto.strip():
                    continue

                classificacao = (
                    classificacoes[fala_indice]
                    if fala_indice < len(classificacoes)
                    else None
                )
                opinioes.append(
                    {
                        "id": f"{deputado_indice}:{fala_indice}",
                        "deputado": deputado.get("nome"),
                        "partido": deputado.get("partido"),
                        "estado": deputado.get("estado"),
                        "fala": texto,
                        "sessao_id": sessao_id,
                        "assunto": assunto,
                        "classificacao": classificacao,
                        "trechos_transcricao": trechos,
                    }
                )
        self._opinioes = opinioes
        return opinioes

    def opiniao_por_id(self, opinion_id: str) -> dict:
        for opiniao in self.opinioes():
            if opiniao["id"] == opinion_id:
                return opiniao
        raise KeyError(opinion_id)


class SimilarityIndex:
    """Índice em memória carregado sob demanda para a visualização espacial."""

    def __init__(self, repository: OutputRepository, modelo: str = MODELO_TOPICO):
        self.repository = repository
        self.modelo_nome = modelo
        self._embeddings: np.ndarray | None = None

    def _vetores(self) -> np.ndarray:
        if self._embeddings is None:
            modelo = SentenceTransformer(self.modelo_nome)
            textos = [opinion["fala"] for opinion in self.repository.opinioes()]
            self._embeddings = modelo.encode(textos, normalize_embeddings=True)
        return self._embeddings

    def proximas(self, opinion_id: str, limite: int) -> list[dict]:
        opinioes = self.repository.opinioes()
        indice = next(
            (index for index, opinion in enumerate(opinioes) if opinion["id"] == opinion_id),
            None,
        )
        if indice is None:
            raise KeyError(opinion_id)

        similaridades = self._vetores() @ self._vetores()[indice]
        indices = np.argsort(-similaridades)
        proximas = []
        for vizinho_indice in indices:
            if int(vizinho_indice) == indice:
                continue
            opiniao = opinioes[int(vizinho_indice)]
            proximas.append(
                {
                    "opiniao": opiniao,
                    "similaridade": round(float(similaridades[vizinho_indice]), 4),
                }
            )
            if len(proximas) == limite:
                break
        return proximas


def _resumo_opiniao(opiniao: dict) -> dict:
    return {
        key: value
        for key, value in opiniao.items()
        if key != "trechos_transcricao"
    }


def create_app(
    repository: OutputRepository | None = None,
    similarity_index: SimilarityIndex | None = None,
) -> FastAPI:
    repository = repository or OutputRepository()
    similarity_index = similarity_index or SimilarityIndex(repository)
    app = FastAPI(title="OpenParla API", version="1.0.0")

    @app.get("/")
    def raiz():
        return {
            "mensagem": "OpenParla API",
            "rotas": ["/dados", "/dados/{arquivo}", "/opinioes", "/opinioes/{id}", "/opinioes/{id}/proximas"],
        }

    @app.get("/dados")
    def listar_dados():
        return {"arquivos": repository.arquivos()}

    @app.get("/dados/{arquivo}")
    def dados(arquivo: str):
        try:
            return repository.carregar(arquivo)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Arquivo não encontrado") from error

    @app.get("/opinioes")
    def listar_opinioes(
        offset: int = Query(0, ge=0),
        limite: int = Query(1000, ge=1, le=2000),
    ):
        opinioes = repository.opinioes()
        itens = [_resumo_opiniao(opiniao) for opiniao in opinioes[offset:offset + limite]]
        return {"total": len(opinioes), "offset": offset, "limite": limite, "itens": itens}

    @app.get("/opinioes/{opinion_id}/proximas")
    def opinioes_proximas(
        opinion_id: str,
        limite: int = Query(10, ge=1, le=50),
    ):
        try:
            origem = _resumo_opiniao(repository.opiniao_por_id(opinion_id))
            proximas = similarity_index.proximas(opinion_id, limite)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Opinião não encontrada") from error
        proximas = [
            {
                "opiniao": _resumo_opiniao(item["opiniao"]),
                "similaridade": item["similaridade"],
            }
            for item in proximas
        ]
        return {"origem": origem, "vizinhas": proximas}

    @app.get("/opinioes/{opinion_id}")
    def opiniao(opinion_id: str):
        try:
            return repository.opiniao_por_id(opinion_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Opinião não encontrada") from error

    return app


app = create_app()
