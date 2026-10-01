"""
pipeline.py

Script principal: lê o JSON de deputados, classifica a orientação de cada
fala e gera relatórios separados para (1) oposição entre fala e partido
(com severidade "contradicao"/"tensao" avaliada pauta por pauta),
(2) contradição entre falas do mesmo deputado, (3) tensão transversal
entre pautas no mesmo deputado e (4) falas defensivas (negação de rótulo).
Todos os resultados são candidatos para revisão humana, não conclusões factuais.

ATENÇÃO — decisão de estrutura de dados: no seu exemplo, "opinioes" é
uma lista, mas "posicionamento_politico_fala" aparecia como um único
campo (None). Como uma fala pode ser sobre um tema e outra sobre
outro, decidi tratar "posicionamento_politico_fala" como uma LISTA
paralela a "opinioes" (mesmo índice = mesma fala). Se no seu uso real
cada deputado sempre tem exatamente uma fala, a lista terá só um
elemento e o comportamento é equivalente a um campo único — mas ajuste
aqui se sua intenção for outra.

Uso:
    python pipeline.py \
        --input Lorenna/saidas/deputados.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ideology_classifier import IdeologyClassifier
from contradiction_detector import ContradictionDetector
from party_alignment import find_party_divergences
from padroes_politicos import (
    find_cross_issue_tensions,
    detect_defensive_speech,
)
from indicios import (
    indicio_da_fala,
    party_divergences_from_indicios,
    reconciliar_resultado_com_indicio,
    cross_issue_tensions_from_indicios,
)


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolver_dispositivo(dispositivo: str) -> str:
    """Resolve ``auto`` para CUDA quando ela estiver disponível."""
    if dispositivo == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda foi solicitado, mas CUDA não está disponível")
    return dispositivo


def limitar_falas_distribuidas(
    deputados: list[dict], max_falas: int | None
) -> list[dict]:
    """Limita falas sem concentrar a amostra nos primeiros deputados.

    A cota é distribuída em rodadas entre todos os deputados com falas. Para
    cada deputado escolhido, as falas são espaçadas no seu conjunto original,
    preservando cobertura do período em vez de selecionar apenas o início.
    """
    if max_falas is None:
        return deputados
    if max_falas < 1:
        raise ValueError("max_falas deve ser maior que zero")

    tamanhos = [len(dep.get("opinioes", [])) for dep in deputados]
    if max_falas >= sum(tamanhos):
        return deputados

    cotas = [0] * len(deputados)
    restantes = max_falas
    while restantes:
        houve_alocacao = False
        for indice, tamanho in enumerate(tamanhos):
            if restantes == 0:
                break
            if cotas[indice] < tamanho:
                cotas[indice] += 1
                restantes -= 1
                houve_alocacao = True
        if not houve_alocacao:
            break

    amostra = []
    for dep, cota, tamanho in zip(deputados, cotas, tamanhos):
        if not cota:
            continue
        opinioes = dep["opinioes"]
        if cota == tamanho:
            selecionadas = opinioes
        elif cota == 1:
            selecionadas = [opinioes[tamanho // 2]]
        else:
            indices = [round(i * (tamanho - 1) / (cota - 1)) for i in range(cota)]
            selecionadas = [opinioes[indice] for indice in indices]

        dep_amostrado = dict(dep)
        dep_amostrado["opinioes"] = selecionadas
        amostra.append(dep_amostrado)

    return amostra


def criar_reporter(progress_log_path: str | None):
    """Retorna uma função que escreve progresso em stdout e, opcionalmente, em arquivo."""
    arquivo = None
    if progress_log_path:
        destino = Path(progress_log_path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        arquivo = destino.open("w", encoding="utf-8")

    def reportar(mensagem: str) -> None:
        linha = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}"
        print(linha, flush=True)
        if arquivo:
            print(linha, file=arquivo, flush=True)

    return reportar, arquivo


BASE_DIR = Path(__file__).resolve().parent
DIRETORIO_SAIDAS = BASE_DIR / "saidas"


def save_json(data, path: str) -> None:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def texto_da_opiniao(opiniao) -> str:
    """Aceita opiniao como objeto {"opiniao": ...} ou como string pura."""
    if isinstance(opiniao, dict):
        return opiniao.get("opiniao", "")
    return opiniao


def sessao_da_opiniao(opiniao):
    """Retorna o id da sessão de origem (se a opiniao for objeto)."""
    if isinstance(opiniao, dict):
        return opiniao.get("sessao_id")
    return None


def assunto_da_opiniao(opiniao):
    if isinstance(opiniao, dict):
        return opiniao.get("assunto")
    return None


def trechos_da_opiniao(opiniao):
    """Trechos literais da transcrição casados com a opinião (enriquecimento)."""
    if isinstance(opiniao, dict):
        return opiniao.get("trechos_transcricao", [])
    return []


def run(
    input_path: str,
    output_classificado_path: str,
    output_contradicoes_path: str,
    output_contradicoes_partido_path: str = str(DIRETORIO_SAIDAS / "contradicoes_partido.json"),
    output_tensoes_transversais_path: str = str(DIRETORIO_SAIDAS / "tensoes_transversais.json"),
    output_falas_defensivas_path: str = str(DIRETORIO_SAIDAS / "falas_defensivas.json"),
    topic_model: str | None = None,
    max_deputados: int | None = None,
    max_falas: int | None = None,
    device: str = "auto",
    nli_batch_size: int = 8,
    progress_log_path: str | None = None,
) -> None:
    if nli_batch_size < 1:
        raise ValueError("nli_batch_size deve ser maior que zero")

    reportar, arquivo_log = criar_reporter(progress_log_path)
    inicio = time.monotonic()
    try:
        deputados = load_json(input_path)
        if max_deputados is not None:
            deputados = deputados[:max_deputados]
            reportar(
                f"[info] Rodando apenas com {len(deputados)} deputado(s) "
                "(--max-deputados)."
            )
        total_original = sum(len(dep.get("opinioes", [])) for dep in deputados)
        deputados = limitar_falas_distribuidas(deputados, max_falas)
        total_falas = sum(len(dep.get("opinioes", [])) for dep in deputados)
        if max_falas is not None and total_falas < total_original:
            reportar(
                f"[info] Amostra distribuída: {total_falas}/{total_original} falas "
                f"(--max-falas={max_falas})."
            )

        dispositivo = resolver_dispositivo(device)
        reportar(
            f"[info] Inicializando modelos em {dispositivo}; "
            f"NLI batch size={nli_batch_size}."
        )
        classifier = IdeologyClassifier(device=dispositivo)
        detector = (
            ContradictionDetector(
                topic_model_name=topic_model,
                device=dispositivo,
                batch_size=nli_batch_size,
            )
            if topic_model
            else ContradictionDetector(
                device=dispositivo,
                batch_size=nli_batch_size,
            )
        )

        pares_incompativeis_output = []
        divergencias_partido_output = []
        tensoes_transversais_output = []
        falas_defensivas_output = []

        preparados = []
        todos_textos = []
        for dep in deputados:
            opinioes = dep.get("opinioes", [])
            textos = [texto_da_opiniao(o) for o in opinioes]
            preparados.append((dep, opinioes, textos))
            todos_textos.extend(textos)

        reportar(f"[progresso] Classificando {len(todos_textos)} falas.")
        # Uma única chamada em lote evita centenas de invocações pequenas do E5.
        todos_resultados = classifier.classify_batch(todos_textos)
        reportar("[progresso] Classificação concluída; iniciando detectores por deputado.")
        cursor = 0

        for numero_dep, (dep, opinioes, textos) in enumerate(preparados, start=1):
            nome = dep.get("nome") or "sem nome"
            reportar(
                f"[progresso] {numero_dep}/{len(preparados)} — {nome}: "
                f"analisando {len(textos)} falas."
            )
            resultados = todos_resultados[cursor:cursor + len(textos)]
            cursor += len(textos)

            indicios = [indicio_da_fala(t) for t in textos]
            resultados = [
                reconciliar_resultado_com_indicio(resultado, indicio)
                for resultado, indicio in zip(resultados, indicios)
            ]

            # Classifica cada fala individualmente.
            dep["posicionamento_politico_fala"] = [r.label for r in resultados]
            # Scores brutos por fala, guardados para auditoria/calibração
            # posterior dos thresholds — remova este campo se não precisar.
            dep["posicionamento_politico_fala_detalhe"] = [
                r.to_dict() for r in resultados
            ]

            divergencias = find_party_divergences(
                dep.get("partido"),
                opinioes,
                textos,
                resultados,
            )
            # Indícios léxicos cobrem falas que o e5 deixou "neutra" (resumos
            # narrativos sem pauta nas âncoras). Sempre sinal fraco ("tensao").
            divergencias += party_divergences_from_indicios(
                dep.get("partido"), opinioes, textos, resultados, indicios
            )
            if divergencias:
                divergencias_partido_output.append(
                    {
                        "nome": dep.get("nome"),
                        "partido": dep.get("partido"),
                        "estado": dep.get("estado"),
                        "posicionamento_politico_partido": dep.get(
                            "posicionamento_politico_partido"
                        ),
                        "candidatos_divergencia_partidaria": [
                            {
                                **p.to_dict(),
                                "review_required": True,
                                "tipo": "candidate_party_divergence",
                                "trechos_transcricao": trechos_da_opiniao(
                                    opinioes[p.indice]
                                ),
                            }
                            for p in divergencias
                        ],
                    }
                )

            # Tensão transversal: posições fortes e opostas em pautas diferentes.
            tensoes = find_cross_issue_tensions(opinioes, textos, resultados)
            if not tensoes:
                tensoes = cross_issue_tensions_from_indicios(
                    opinioes, textos, resultados, indicios
                )
            if tensoes:
                tensoes_transversais_output.append(
                    {
                        "nome": dep.get("nome"),
                        "partido": dep.get("partido"),
                        "estado": dep.get("estado"),
                        "tensoes": [
                            {
                                **t.to_dict(),
                                "review_required": True,
                                "tipo": "candidate_cross_issue_tension",
                            }
                            for t in tensoes
                        ],
                    }
                )

            # Fala defensiva: negação de rótulo ("não somos antivacinas").
            textos_contexto = [
                "\n".join([texto, *trechos_da_opiniao(opiniao)])
                for texto, opiniao in zip(textos, opinioes)
            ]
            defensivas = detect_defensive_speech(
                opinioes, textos, resultados, textos_contexto
            )
            if defensivas:
                falas_defensivas_output.append(
                    {
                        "nome": dep.get("nome"),
                        "partido": dep.get("partido"),
                        "estado": dep.get("estado"),
                        "falas": [
                            {
                                **d.to_dict(),
                                "review_required": True,
                                "tipo": "candidate_defensive_speech",
                                "trechos_transcricao": trechos_da_opiniao(
                                    opinioes[d.indice]
                                ),
                            }
                            for d in defensivas
                        ],
                    }
                )

            # Detecta contradições entre as falas do mesmo deputado.
            pares = detector.find_contradictions(textos, resultados)
            if pares:
                pares_incompativeis_output.append(
                    {
                        "nome": dep.get("nome"),
                        "partido": dep.get("partido"),
                        "estado": dep.get("estado"),
                        "pares_potencialmente_incompativeis": [
                            {
                                "review_required": True,
                                "tipo": "candidate_incompatible_pair",
                                "fala_a": p.opiniao_a,
                                "fala_b": p.opiniao_b,
                                "indice_a": p.indice_a,
                                "indice_b": p.indice_b,
                                "similaridade_tematica": round(p.topic_similarity, 4),
                                "score_contradicao": round(p.contradiction_score, 4),
                                "score_a_para_b": round(p.contradiction_a_b, 4),
                                "score_b_para_a": round(p.contradiction_b_a, 4),
                                "pauta_politica": p.policy_issue,
                                "reversao_politica": p.policy_reversal,
                                "sessao_id_a": sessao_da_opiniao(opinioes[p.indice_a]),
                                "sessao_id_b": sessao_da_opiniao(opinioes[p.indice_b]),
                                "assunto_a": assunto_da_opiniao(opinioes[p.indice_a]),
                                "assunto_b": assunto_da_opiniao(opinioes[p.indice_b]),
                                "trechos_transcricao_a": trechos_da_opiniao(
                                    opinioes[p.indice_a]
                                ),
                                "trechos_transcricao_b": trechos_da_opiniao(
                                    opinioes[p.indice_b]
                                ),
                            }
                            for p in pares
                        ],
                    }
                )
            reportar(
                f"[progresso] {numero_dep}/{len(preparados)} — {nome}: concluído "
                f"({len(pares)} par(es) candidato(s)); "
                f"{time.monotonic() - inicio:.0f}s decorridos."
            )

        save_json(deputados, output_classificado_path)
        save_json(pares_incompativeis_output, output_contradicoes_path)
        save_json(divergencias_partido_output, output_contradicoes_partido_path)
        save_json(tensoes_transversais_output, output_tensoes_transversais_path)
        save_json(falas_defensivas_output, output_falas_defensivas_path)

        n_divergencias = sum(
            len(c["candidatos_divergencia_partidaria"])
            for c in divergencias_partido_output
        )
        n_contradicao = sum(
            1
            for c in divergencias_partido_output
            for d in c["candidatos_divergencia_partidaria"]
            if d.get("severidade") == "contradicao"
        )

        reportar(f"[OK] {len(deputados)} deputados processados.")
        reportar(f"[OK] Classificações salvas em: {output_classificado_path}")
        reportar(
            f"[OK] {len(pares_incompativeis_output)} deputado(s) com pares "
            f"potencialmente incompatíveis para revisão. Salvo em: {output_contradicoes_path}"
        )
        reportar(
            f"[OK] {len(divergencias_partido_output)} deputado(s) com possível divergência "
            f"fala-partido ({n_divergencias} divergência(s), das quais "
            f"{n_contradicao} forte(s)). Salvo em: {output_contradicoes_partido_path}"
        )
        reportar(
            f"[OK] {len(tensoes_transversais_output)} deputado(s) com tensão "
            f"transversal entre pautas. Salvo em: {output_tensoes_transversais_path}"
        )
        reportar(
            f"[OK] {len(falas_defensivas_output)} deputado(s) com falas "
            f"defensivas. Salvo em: {output_falas_defensivas_path}"
        )
    finally:
        if arquivo_log:
            arquivo_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DIRETORIO_SAIDAS / "deputados.json"))
    parser.add_argument(
        "--output-classificado",
        default=str(DIRETORIO_SAIDAS / "deputados_classificados.json"),
    )
    parser.add_argument(
        "--output-contradicoes",
        default=str(DIRETORIO_SAIDAS / "contradicoes_falas.json"),
    )
    parser.add_argument(
        "--output-contradicoes-partido",
        default=str(DIRETORIO_SAIDAS / "contradicoes_partido.json"),
    )
    parser.add_argument(
        "--output-tensoes-transversais",
        default=str(DIRETORIO_SAIDAS / "tensoes_transversais.json"),
    )
    parser.add_argument(
        "--output-falas-defensivas",
        default=str(DIRETORIO_SAIDAS / "falas_defensivas.json"),
    )
    parser.add_argument(
        "--topic-model",
        default=None,
        help="Modelo SentenceTransformer usado para similaridade temática.",
    )
    parser.add_argument(
        "--max-deputados",
        type=int,
        default=None,
        help="Limita a quantidade de deputados processados (útil para "
             "testar rápido antes de rodar o conjunto completo).",
    )
    parser.add_argument(
        "--max-falas",
        type=int,
        default=None,
        help="Limita o total de falas em uma amostra distribuída entre deputados.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Dispositivo para os três modelos; auto prioriza CUDA.",
    )
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=16,
        help="Tamanho do batch do NLI (padrão: 16). Reduza se houver OOM.",
    )
    parser.add_argument(
        "--progress-log",
        default=None,
        help="Arquivo opcional com marcos de progresso por deputado.",
    )
    args = parser.parse_args()

    run(
        args.input,
        args.output_classificado,
        args.output_contradicoes,
        args.output_contradicoes_partido,
        args.output_tensoes_transversais,
        args.output_falas_defensivas,
        args.topic_model,
        max_deputados=args.max_deputados,
        max_falas=args.max_falas,
        device=args.device,
        nli_batch_size=args.nli_batch_size,
        progress_log_path=args.progress_log,
    )
