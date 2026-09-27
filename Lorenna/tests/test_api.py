import json

from fastapi.testclient import TestClient

from Lorenna.app import OutputRepository, create_app


class FakeSimilarityIndex:
    def proximas(self, opinion_id: str, limite: int) -> list[dict]:
        return [{"opiniao": {"id": "0:1"}, "similaridade": 0.9}][:limite]


def test_api_expoe_dados_e_opinioes(tmp_path):
    deputados = [
        {
            "nome": "Ana",
            "partido": "PT",
            "estado": "SP",
            "opinioes": [
                {
                    "opiniao": "Defende política pública.",
                    "sessao_id": 1,
                    "assunto": "Teste",
                    "trechos_transcricao": ["Trecho da audiência."],
                },
                {"opiniao": "Defende investimento social."},
            ],
            "posicionamento_politico_fala_detalhe": [{"label": "esquerda"}],
        }
    ]
    (tmp_path / "deputados_classificados.json").write_text(
        json.dumps(deputados), encoding="utf-8"
    )
    repository = OutputRepository(tmp_path)
    client = TestClient(create_app(repository, FakeSimilarityIndex()))

    assert client.get("/dados").json() == {"arquivos": ["deputados_classificados.json"]}
    resposta = client.get("/opinioes")
    assert resposta.status_code == 200
    assert resposta.json()["total"] == 2
    assert "trechos_transcricao" not in resposta.json()["itens"][0]

    detalhe = client.get("/opinioes/0:0")
    assert detalhe.status_code == 200
    assert detalhe.json()["trechos_transcricao"] == ["Trecho da audiência."]

    proximas = client.get("/opinioes/0:0/proximas?limite=1")
    assert proximas.status_code == 200
    assert proximas.json()["vizinhas"][0]["similaridade"] == 0.9

    assert client.get("/dados/inexistente.json").status_code == 404
    assert client.get("/opinioes/inexistente").status_code == 404
