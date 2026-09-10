"""Testa a preparação das opiniões validadas do PublicHearingBR NLI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preparar_dados import processar_nli


def test_processar_nli_usa_somente_opinioes_validadas():
    sessoes = [{
        "id": 7,
        "metadados_extraidos": {
            "assunto": "Teste",
            "envolvidos": [{
                "nome": "Maria da Silva",
                "cargo": "Deputada Federal (PT-SP)",
                "opinioes": [
                    {
                        "opiniao": "Defendeu a política pública.",
                        "chunks_proximos": ["Trecho que sustenta a fala."],
                        "verificacao_alucinacao": {"verificacao_manual": False},
                    },
                    {
                        "opiniao": "Opinião alucinada.",
                        "chunks_proximos": ["Trecho sem suporte."],
                        "verificacao_alucinacao": {"verificacao_manual": True},
                    },
                    {
                        "opiniao": "Opinião sem revisão manual.",
                        "chunks_proximos": [],
                        "verificacao_alucinacao": {},
                    },
                ],
            }],
        },
    }]

    deputados = processar_nli(sessoes)

    assert len(deputados) == 1
    deputado = deputados[0]
    assert deputado["partido"] == "PT"
    assert deputado["estado"] == "São Paulo"
    assert deputado["opinioes"] == [{
        "opiniao": "Defendeu a política pública.",
        "sessao_id": 7,
        "assunto": "Teste",
        "trechos_transcricao": ["Trecho que sustenta a fala."],
    }]


def test_processar_nli_completa_partido_com_lds_da_mesma_sessao():
    sessoes = [{
        "id": 9,
        "metadados_extraidos": {
            "assunto": "Teste",
            "envolvidos": [{
                "nome": "João da Silva",
                "cargo": "Deputado Federal",
                "opinioes": [{
                    "opiniao": "Defendeu a política pública.",
                    "verificacao_alucinacao": {"verificacao_manual": False},
                }],
            }],
        },
    }]
    cargos_lds = {(9, "joao da silva"): "Deputado (PL-RJ)"}

    deputados = processar_nli(sessoes, cargos_lds=cargos_lds)

    assert deputados[0]["partido"] == "PL"
    assert deputados[0]["estado"] == "Rio de Janeiro"


if __name__ == "__main__":
    test_processar_nli_usa_somente_opinioes_validadas()
    test_processar_nli_completa_partido_com_lds_da_mesma_sessao()
    print("[OK] Preparação NLI: todos os casos de teste passaram.")
