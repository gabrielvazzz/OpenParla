"""
pipeline.py

Script principal: lê o JSON de deputados, classifica a orientação de cada
fala e gera relatórios separados para (1) oposição entre fala e partido
(com severidade "contradicao"/"tensao" avaliada pauta por pauta),
(2) contradição entre falas do mesmo deputado, (3) tensão transversal
entre pautas no mesmo deputado e (4) falas defensivas (negação de rótulo).

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
        --input deputados.json \
        --output-classificado deputados_classificados.json \
        --output-contradicoes contradicoes_falas.json \
        --output-contradicoes-partido contradicoes_partido.json \
        --output-tensoes-transversais tensoes_transversais.json \
        --output-falas-defensivas falas_defensivas.json
"""

from __future__ import annotations

import argparse
import json

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
    cross_issue_tensions_from_indicios,
)


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
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
    output_contradicoes_partido_path: str = "contradicoes_partido.json",
    output_tensoes_transversais_path: str = "tensoes_transversais.json",
    output_falas_defensivas_path: str = "falas_defensivas.json",
    max_deputados: int | None = None,
) -> None:
    deputados = load_json(input_path)
    if max_deputados is not None:
        deputados = deputados[:max_deputados]
        print(f"[info] Rodando apenas com {len(deputados)} deputado(s) "
              f"(--max-deputados).")

    classifier = IdeologyClassifier()
    detector = ContradictionDetector()

    contradicoes_falas_output = []
    contradicoes_partido_output = []
    tensoes_transversais_output = []
    falas_defensivas_output = []

    preparados = []
    todos_textos = []
    for dep in deputados:
        opinioes = dep.get("opinioes", [])
        textos = [texto_da_opiniao(o) for o in opinioes]
        preparados.append((dep, opinioes, textos))
        todos_textos.extend(textos)

    # Uma única chamada em lote evita centenas de invocações pequenas do E5.
    todos_resultados = classifier.classify_batch(todos_textos)
    cursor = 0

    for dep, opinioes, textos in preparados:
        resultados = todos_resultados[cursor:cursor + len(textos)]
        cursor += len(textos)

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
        indicios = [indicio_da_fala(t) for t in textos]
        divergencias += party_divergences_from_indicios(
            dep.get("partido"), opinioes, textos, resultados, indicios
        )
        if divergencias:
            contradicoes_partido_output.append(
                {
                    "nome": dep.get("nome"),
                    "partido": dep.get("partido"),
                    "estado": dep.get("estado"),
                    "posicionamento_politico_partido": dep.get(
                        "posicionamento_politico_partido"
                    ),
                    "contradicoes": [
                        {
                            **p.to_dict(),
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
                    "tensoes": [t.to_dict() for t in tensoes],
                }
            )

        # Fala defensiva: negação de rótulo ("não somos antivacinas").
        defensivas = detect_defensive_speech(opinioes, textos, resultados)
        if defensivas:
            falas_defensivas_output.append(
                {
                    "nome": dep.get("nome"),
                    "partido": dep.get("partido"),
                    "estado": dep.get("estado"),
                    "falas": [
                        {
                            **d.to_dict(),
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
            contradicoes_falas_output.append(
                {
                    "nome": dep.get("nome"),
                    "partido": dep.get("partido"),
                    "estado": dep.get("estado"),
                    "contradicoes": [
                        {
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
                        }
                        for p in pares
                    ],
                }
            )

    save_json(deputados, output_classificado_path)
    save_json(contradicoes_falas_output, output_contradicoes_path)
    save_json(contradicoes_partido_output, output_contradicoes_partido_path)
    save_json(tensoes_transversais_output, output_tensoes_transversais_path)
    save_json(falas_defensivas_output, output_falas_defensivas_path)

    n_divergencias = sum(
        len(c["contradicoes"]) for c in contradicoes_partido_output
    )
    n_contradicao = sum(
        1
        for c in contradicoes_partido_output
        for d in c["contradicoes"]
        if d.get("severidade") == "contradicao"
    )

    print(f"[OK] {len(deputados)} deputados processados.")
    print(f"[OK] Classificações salvas em: {output_classificado_path}")
    print(
        f"[OK] {len(contradicoes_falas_output)} deputado(s) com contradições "
        f"entre falas. Salvo em: {output_contradicoes_path}"
    )
    print(
        f"[OK] {len(contradicoes_partido_output)} deputado(s) com oposição "
        f"fala-partido ({n_divergencias} divergência(s), das quais "
        f"{n_contradicao} forte(s)). Salvo em: {output_contradicoes_partido_path}"
    )
    print(
        f"[OK] {len(tensoes_transversais_output)} deputado(s) com tensão "
        f"transversal entre pautas. Salvo em: {output_tensoes_transversais_path}"
    )
    print(
        f"[OK] {len(falas_defensivas_output)} deputado(s) com falas "
        f"defensivas. Salvo em: {output_falas_defensivas_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-classificado", default="deputados_classificados.json"
    )
    parser.add_argument("--output-contradicoes", default="contradicoes.json")
    parser.add_argument(
        "--output-contradicoes-partido",
        default="contradicoes_partido.json",
    )
    parser.add_argument(
        "--output-tensoes-transversais",
        default="tensoes_transversais.json",
    )
    parser.add_argument(
        "--output-falas-defensivas",
        default="falas_defensivas.json",
    )
    parser.add_argument(
        "--max-deputados",
        type=int,
        default=None,
        help="Limita a quantidade de deputados processados (útil para "
             "testar rápido antes de rodar o conjunto completo).",
    )
    args = parser.parse_args()

    run(
        args.input,
        args.output_classificado,
        args.output_contradicoes,
        args.output_contradicoes_partido,
        args.output_tensoes_transversais,
        args.output_falas_defensivas,
        max_deputados=args.max_deputados,
    )
