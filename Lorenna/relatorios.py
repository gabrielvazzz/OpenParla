"""Relatórios auditáveis para os dois cenários de evidência."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def gerar_relatorios(
    deputados: list[dict],
    diretorio: Path,
    divergencias_partido: list[dict] | None = None,
    pares_incompativeis: list[dict] | None = None,
) -> dict:
    """Gera CSVs a partir do schema ativo do pipeline, para revisão humana."""
    diretorio.mkdir(parents=True, exist_ok=True)

    divergencias, contradicoes = [], []

    for deputado in deputados:
        contexto = {
            "deputado": deputado.get("nome", deputado.get("deputado")),
            "partido": deputado.get("partido"),
            "estado": deputado.get("estado"),
            "posicionamento_partido": deputado.get(
                "posicionamento_politico_partido", deputado.get("posicionamento_partido")
            ),
        }
        divergencias.extend([{**contexto, **item} for item in
                             (deputado.get("Divergencias_partidarias") or [])])
        contradicoes.extend([{**contexto, **item} for item in
                              (deputado.get("Contradicoes") or [])])

    for registro in divergencias_partido or []:
        contexto = {"deputado": registro.get("nome"), "partido": registro.get("partido"),
                    "estado": registro.get("estado"), "posicionamento_partido": registro.get("posicionamento_politico_partido")}
        divergencias.extend([{**contexto, **item} for item in
                             registro.get("candidatos_divergencia_partidaria", [])])
    for registro in pares_incompativeis or []:
        contexto = {"deputado": registro.get("nome"), "partido": registro.get("partido"),
                    "estado": registro.get("estado"), "posicionamento_partido": None}
        contradicoes.extend([{**contexto, **item} for item in
                              registro.get("pares_potencialmente_incompativeis", [])])

    with (diretorio / "divergencias_partidarias.csv").open("w", newline="", encoding="utf-8") as arquivo:

        campos = ["deputado", "partido", "estado", "posicionamento_partido",
                   "posicionamento_fala", "pauta", "distancia", "score_partido",
                   "score_fala", "evidencia_pauta", "severidade", "origem",
                   "review_required", "tipo"]

        writer = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(divergencias)

    with (diretorio / "contradicoes.csv").open("w", newline="", encoding="utf-8") as arquivo:

        campos = ["deputado", "partido", "estado", "posicionamento_partido",
                   "similaridade_tematica", "score_contradicao", "score_a_para_b",
                   "score_b_para_a", "review_required", "tipo"]

        writer = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(contradicoes)

    por_partido = Counter(item["partido"] for item in divergencias)

    resumo = ["# Resumo da execução", "",
              f"- Deputados processados: {len(deputados)}",
              f"- Candidatas a divergência partidária: {len(divergencias)}",
              f"- Pares de possível autocontradição: {len(contradicoes)}", "",
              "## Divergências por partido", ""]

    resumo.extend(f"- {partido}: {quantidade}" for partido, quantidade in sorted(por_partido.items()))

    resumo.extend(["", "Todos os itens são candidatos para revisão humana; os CSVs "
                   "preservam os scores e o JSON de saída preserva os textos e sessões."])
    (diretorio / "resumo.md").write_text("\n".join(resumo) + "\n", encoding="utf-8")


    return {"divergencias": len(divergencias), "contradicoes": len(contradicoes)}
