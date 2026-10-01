"""Testa o casamento opinião -> trechos da transcrição (sem modelos)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enriquecedor import (
    _normalizar_tokens,
    dividir_trechos,
    trechos_proximos,
    enriquecer,
)


TRANSCRICAO = """
O SR. PRESIDENTE(Fulano. Bloco/PSD - SP) - Bom dia a todos.

Declaro aberta a reunião de audiência pública da Comissão de Minas e Energia.

O SR. ESPECIALISTA(Autoridade em Energia) - A tarifa social é um mecanismo
que funciona muito bem e protege o consumidor de baixa renda. É possível
ampliar esse mecanismo e transferir o subsídio para o orçamento da União,
aprovado por esta Casa.

O SR. OUTRO(Representante da Associação) - As concessionárias precisam de
previsibilidade regulatória para as prorrogações das concessões.
"""

OPINIAO_TARIFA = (
    "Há consenso sobre a migração da Tarifa Social de Energia Elétrica "
    "para o orçamento da União."
)


def test_normalizar_tokens():
    toks = _normalizar_tokens("A Tarifa Social de Energia Elétrica!")
    assert "tarifa" in toks and "social" in toks and "energia" in toks
    assert "de" not in toks
    assert "a" not in toks


def test_dividir_trechos():
    blocos = dividir_trechos(TRANSCRICAO)
    assert len(blocos) == 4
    assert "Bom dia a todos" in blocos[0]


def test_trechos_proximos():
    blocos = dividir_trechos(TRANSCRICAO)
    blocos_tok = [(b, _normalizar_tokens(b)) for b in blocos]

    trechos = trechos_proximos(OPINIAO_TARIFA, blocos_tok, k=4)
    assert trechos  # pelo menos um blob casa
    # O melhor trecho é a fala real do especialista sobre tarifa social.
    assert "tarifa social" in trechos[0].lower()
    assert "orçamento da União" in trechos[0]


def test_enriquecer():
    deputados = [
        {
            "nome": "Fulano",
            "partido": "PSD",
            "opinioes": [
                {"opiniao": OPINIAO_TARIFA, "sessao_id": 1, "assunto": "x"},
                {"opiniao": "Assunto totalmente diferente de economia.", "sessao_id": 1},
            ],
            "posicionamento_politico_partido": "centro",
        }
    ]
    sessoes_por_id = {1: TRANSCRICAO}
    out = enriquecer(deputados, sessoes_por_id)
    trechos = out[0]["opinioes"][0]["trechos_transcricao"]
    assert trechos and trechos[0].startswith("O SR. ESPECIALISTA")
    assert "tarifa social" in trechos[0].lower()
    # Opinião sem correspondência lexical -> lista vazia, sem quebrar.
    assert out[0]["opinioes"][1]["trechos_transcricao"] == []


if __name__ == "__main__":
    test_normalizar_tokens()
    test_dividir_trechos()
    test_trechos_proximos()
    test_enriquecer()
    print("[OK] Enriquecedor: todos os casos de teste passaram.")