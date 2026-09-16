import tempfile
import unittest
from pathlib import Path

from Lorenna.publichearingbr_ideology.relatorios import gerar_relatorios


class RelatoriosTest(unittest.TestCase):
    def test_gera_arquivos_para_os_dois_cenarios(self):
        deputados = [{
            "deputado": "Exemplo",
            "partido": "PL",
            "estado": "minas gerais",
            "posicionamento_partido": "direita",
            "Divergencias_partidarias": [{
                "posicionamento_fala": "esquerda", "distancia_ideologica": 4,
                "confianca_embedding": 0.9, "margem_embedding": 0.1,
                "status": "revisao_necessaria",
            }],
            "Contradicoes": [{
                "similaridade_topica": 0.8, "score_contradicao": 0.9,
                "score_a_para_b": 0.9, "score_b_para_a": 0.8,
                "status": "revisao_necessaria",
            }],
        }]
        with tempfile.TemporaryDirectory() as temporario:
            contagens = gerar_relatorios(deputados, Path(temporario))
            self.assertEqual(contagens, {"divergencias": 1, "contradicoes": 1})
            self.assertTrue((Path(temporario) / "divergencias_partidarias.csv").exists())
            self.assertTrue((Path(temporario) / "contradicoes.csv").exists())
            self.assertTrue((Path(temporario) / "resumo.md").exists())


if __name__ == "__main__":
    unittest.main()
