"""
pipeline.py

Script principal: lê o JSON de deputados (cada um com um campo
"opinioes", lista de falas), classifica a orientação política de cada
fala individualmente e detecta contradições entre falas do mesmo
deputado.

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
        --output-contradicoes contradicoes.json
"""

from __future__ import annotations

import argparse
import json

from ideology_classifier import IdeologyClassifier
from contradiction_detector import ContradictionDetector


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


def run(
    input_path: str,
    output_classificado_path: str,
    output_contradicoes_path: str,
    max_deputados: int | None = None,
) -> None:
    deputados = load_json(input_path)
    if max_deputados is not None:
        deputados = deputados[:max_deputados]
        print(f"[info] Rodando apenas com {len(deputados)} deputado(s) "
              f"(--max-deputados).")

    classifier = IdeologyClassifier()
    detector = ContradictionDetector()

    contradicoes_output = []

    for dep in deputados:
        opinioes = dep.get("opinioes", [])
        textos = [texto_da_opiniao(o) for o in opinioes]

        # Classifica cada fala individualmente.
        resultados = classifier.classify_batch(textos)
        dep["posicionamento_politico_fala"] = [r.label for r in resultados]
        # Scores brutos por fala, guardados para auditoria/calibração
        # posterior dos thresholds — remova este campo se não precisar.
        dep["posicionamento_politico_fala_detalhe"] = [
            r.to_dict() for r in resultados
        ]

        # Detecta contradições entre as falas do mesmo deputado.
        pares = detector.find_contradictions(textos)
        if pares:
            contradicoes_output.append(
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
                            "sessao_id_a": sessao_da_opiniao(opinioes[p.indice_a]),
                            "sessao_id_b": sessao_da_opiniao(opinioes[p.indice_b]),
                        }
                        for p in pares
                    ],
                }
            )

    save_json(deputados, output_classificado_path)
    save_json(contradicoes_output, output_contradicoes_path)

    print(f"[OK] {len(deputados)} deputados processados.")
    print(f"[OK] Classificações salvas em: {output_classificado_path}")
    print(
        f"[OK] {len(contradicoes_output)} deputado(s) com contradições "
        f"detectadas. Salvo em: {output_contradicoes_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-classificado", default="deputados_classificados.json"
    )
    parser.add_argument("--output-contradicoes", default="contradicoes.json")
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
        max_deputados=args.max_deputados,
    )
