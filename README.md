# OpenParla

OpenParla organiza falas de deputados em audiências públicas brasileiras para revisão humana. O sistema recebe opiniões extraídas do [PublicHearingBR](https://huggingface.co/datasets/unicamp-dl/PublicHearingBR), classifica postura por pauta e gera candidatas de divergência fala--partido, pares de falas potencialmente incompatíveis, tensões entre pautas e falas defensivas.

As saídas são sinais para leitura e auditoria, não conclusões sobre ideologia, coerência ou conduta de parlamentares.

## Como funciona

1. `Lorenna/preparar_dados.py` lê o PublicHearingBR, mantém opiniões com validação manual da fonte e agrupa falas por deputado.
2. `Lorenna/pipeline.py` usa E5 para identificar pauta e projetar a postura em eixos contrastivos.
3. O pipeline compara falas do mesmo deputado por similaridade semântica e aplica NLI bidirecional apenas nos pares recuperados.
4. Os JSONs finais preservam texto, sessão, assunto, trechos de transcrição e escores para revisão.

## Requisitos

- Python 3.10 ou superior.
- Acesso à internet na primeira execução para baixar os modelos do Hugging Face.
- Espaço em disco para os pesos dos modelos e os dados.

Instale as dependências a partir da raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Dados

O fluxo padrão usa a variante NLI do PublicHearingBR. Ela contém resumos de falas e trechos de transcrição associados. O repositório espera os arquivos locais neste diretório:

```text
Leo/data/PublicHearingBR_NLI.jsonl
Leo/data/PublicHearingBR_LDS.jsonl
```

Caso o arquivo NLI não esteja disponível localmente, `preparar_dados.py` o baixa automaticamente. Para reproduzir a execução com os arquivos já presentes no repositório:

```bash
python3 Lorenna/preparar_dados.py \
  --fonte nli \
  --arquivo-local Leo/data/PublicHearingBR_NLI.jsonl \
  --saida Lorenna/saidas/deputados.json
```

O comando produz uma lista de deputados com `opinioes`, incluindo `opiniao`, `sessao_id`, `assunto` e `trechos_transcricao`.

## Executar a análise

Execute o pipeline completo a partir da raiz:

```bash
python3 Lorenna/pipeline.py \
  --input Lorenna/saidas/deputados.json \
  --output-classificado Lorenna/saidas/deputados_classificados.json \
  --output-contradicoes Lorenna/saidas/pares_potencialmente_incompativeis.json \
  --output-contradicoes-partido Lorenna/saidas/contradicoes_partido.json \
  --output-tensoes-transversais Lorenna/saidas/tensoes_transversais.json \
  --output-falas-defensivas Lorenna/saidas/falas_defensivas.json
```

Para uma verificação rápida antes da execução integral, limite a análise a um deputado:

```bash
python3 Lorenna/pipeline.py \
  --input Lorenna/saidas/deputados.json \
  --max-deputados 1
```

A execução completa pode levar bastante tempo em CPU porque o NLI é aplicado aos pares de falas recuperados. Os modelos usados são baixados e armazenados no cache local do Hugging Face na primeira execução.

## Saídas

Os arquivos gerados em `Lorenna/saidas/` são:

| Arquivo | Conteúdo |
| --- | --- |
| `deputados.json` | Entrada preparada a partir do PublicHearingBR. |
| `deputados_classificados.json` | Entrada com pauta, rótulo, escore e evidências por fala. |
| `pares_potencialmente_incompativeis.json` | Pares do mesmo deputado recuperados para revisão. |
| `contradicoes_partido.json` | Candidatas de divergência entre fala e referência partidária configurada. |
| `tensoes_transversais.json` | Posições fortes em sentidos opostos em pautas diferentes. |
| `falas_defensivas.json` | Falas com negação de rótulos que exigem leitura contextual. |

Todos os itens candidatos incluem `review_required: true`. Eles não são prova de contradição, mudança de posição ou posicionamento oficial.

## Explorar os dados de scraping

Os dados em `scrapping/` contêm discursos, votos e cadastro de deputados. O notebook `exploracao_scrapping.ipynb` mostra inventário, esquema e amostras dos arquivos.

Para converter os discursos ao formato de entrada do pipeline, execute:

```bash
python3 processar_scrapping.py
```

As entradas anuais e agregadas são salvas em `saidas_scrapping/`. O pipeline atual aceita falas textuais; votos nominais não são usados como entrada direta.

## API local

Após gerar as saídas, inicie a API para navegar pelos artefatos e consultar falas semelhantes:

```bash
uvicorn Lorenna.app:app --reload
```

Com o servidor ativo, acesse `http://127.0.0.1:8000/docs`. A rota `/dados` lista os JSONs disponíveis; `/opinioes` lista falas; e `/opinioes/{id}/proximas` recupera falas semanticamente próximas.

## Testes

Execute a suíte a partir da raiz:

```bash
PYTHONPATH=Lorenna pytest Lorenna/tests -q
```

O teste lógico independente de downloads pode ser executado com:

```bash
python3 Lorenna/tests/test_logic_smoke.py
```

## Estrutura

```text
Lorenna/
  preparar_dados.py       # prepara PublicHearingBR
  pipeline.py             # fluxo principal de análise
  ideology_classifier.py  # pauta e postura por embedding
  contradiction_detector.py # recuperação temática e NLI
  party_alignment.py      # referência partidária configurada
  app.py                  # API FastAPI
  tests/                  # testes automatizados
Leo/data/                 # arquivos locais do PublicHearingBR
scrapping/                # discursos, votos e cadastro da Câmara
ARTIGO.md                 # descrição metodológica e limitações
```

Consulte `AGENTS.md` para orientações de manutenção e revisão automatizada.
