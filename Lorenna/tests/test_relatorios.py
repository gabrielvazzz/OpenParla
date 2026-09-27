import tempfile
import unittest
from pathlib import Path

from Lorenna.relatorios import gerar_relatorios


class RelatoriosTest(unittest.TestCase):
    def test_gera_arquivos_para_o_schema_ativo(self):
        deputados = [{
            "nome": "Exemplo",
            "partido": "PL",
            "estado": "minas gerais",
            "posicionamento_politico_partido": "direita",
        }]
        divergencias = [{
            "nome": "Exemplo", "partido": "PL", "estado": "minas gerais",
            "posicionamento_politico_partido": "direita",
            "candidatos_divergencia_partidaria": [{
                "posicionamento_fala": "esquerda", "review_required": True,
                "tipo": "candidate_party_divergence",
            }],
        }]
        pares = [{
            "nome": "Exemplo", "partido": "PL", "estado": "minas gerais",
            "pares_potencialmente_incompativeis": [{
                "similaridade_tematica": 0.8, "score_contradicao": 0.9,
                "score_a_para_b": 0.9, "score_b_para_a": 0.8,
                "review_required": True, "tipo": "candidate_incompatible_pair",
            }],
        }]
        with tempfile.TemporaryDirectory() as temporario:
            contagens = gerar_relatorios(
                deputados, Path(temporario), divergencias, pares
            )
            self.assertEqual(contagens, {"divergencias": 1, "contradicoes": 1})
            self.assertTrue((Path(temporario) / "divergencias_partidarias.csv").exists())
            self.assertTrue((Path(temporario) / "contradicoes.csv").exists())
            self.assertTrue((Path(temporario) / "resumo.md").exists())


if __name__ == "__main__":
    unittest.main()
