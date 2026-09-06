"""
Teste de integração: roda pipeline.run() de ponta a ponta usando os
mesmos stubs/mocks do test_logic_smoke.py, com um JSON de entrada no
formato exato que o usuário forneceu (deputado com uma fala) + um
segundo deputado com duas falas contraditórias sobre aborto, pra
validar o formato de saída dos dois arquivos JSON.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ideology_classifier as ic
import contradiction_detector as cd
from test_logic_smoke import _FakeModel, _FakeTopicModel, _fake_nli_pipeline

import pipeline


ic.SentenceTransformer = _FakeModel
cd.SentenceTransformer = _FakeTopicModel
cd.pipeline = lambda *a, **kw: _fake_nli_pipeline()

INPUT = [
    {
        "nome": "Filipe Barros",
        "cargo": "Deputado",
        "estado": "Paraná",
        "partido": "PL",
        "opinioes": [
            "Ressaltou que o Marco Civil da Internet é muito claro ao "
            "estabelecer que, se houver afronta à legislação, deve-se "
            "censurar uma publicação específica mediante decisão "
            "judicial motivada. Apontou que as decisões são genéricas."
        ],
        "posicionamento_politico_partido": "direita",
        "posicionamento_politico_fala": None,
    },
    {
        "nome": "Maria Exemplo",
        "cargo": "Deputada",
        "estado": "Paraíba",
        "partido": "PT",
        "opinioes": [
            "Sou a favor do aborto em qualquer circunstância.",
            "Sou totalmente contra o aborto, defendo a vida.",
        ],
        "posicionamento_politico_partido": "esquerda",
        "posicionamento_politico_fala": None,
    },
]

with open("exemplo_entrada.json", "w", encoding="utf-8") as f:
    json.dump(INPUT, f, ensure_ascii=False, indent=2)

pipeline.run(
    "exemplo_entrada.json",
    "exemplo_saida_classificado.json",
    "exemplo_saida_contradicoes.json",
)

with open("exemplo_saida_classificado.json", encoding="utf-8") as f:
    classificado = json.load(f)
with open("exemplo_saida_contradicoes.json", encoding="utf-8") as f:
    contradicoes = json.load(f)

assert len(classificado) == 2
assert isinstance(classificado[0]["posicionamento_politico_fala"], list)
assert len(classificado[0]["posicionamento_politico_fala"]) == 1
assert len(contradicoes) == 1
assert contradicoes[0]["nome"] == "Maria Exemplo"
assert len(contradicoes[0]["contradicoes"]) == 1

print("[OK] Teste de integração do pipeline passou.")
print("\n--- exemplo_saida_classificado.json (deputado 1) ---")
print(json.dumps(classificado[0], ensure_ascii=False, indent=2))
print("\n--- exemplo_saida_contradicoes.json ---")
print(json.dumps(contradicoes, ensure_ascii=False, indent=2))
