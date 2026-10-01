# AGENTS.md

## Objetivo do repositório

OpenParla gera candidatas auditáveis para revisão humana de falas parlamentares. Não trate classificações, pares NLI ou divergências partidárias como fatos sobre pessoas. A linguagem do código, documentação e interfaces deve preservar termos como "candidata", "sinal" e "revisão humana".

## Arquitetura

| Área | Arquivos principais | Responsabilidade |
| --- | --- | --- |
| Preparação | `Lorenna/preparar_dados.py`, `Lorenna/enriquecedor.py` | Converte PublicHearingBR em entrada por deputado. |
| Classificação | `Lorenna/ideology_classifier.py`, `Lorenna/anchors.py` | Detecta pauta e projeta postura em eixos contrastivos. |
| Pares | `Lorenna/contradiction_detector.py` | Recupera pares por embedding e os prioriza com NLI bidirecional. |
| Sinais adicionais | `Lorenna/party_alignment.py`, `Lorenna/padroes_politicos.py`, `Lorenna/indicios.py` | Divergência fala--partido, tensões e regras léxicas. |
| Orquestração | `Lorenna/pipeline.py` | Produz os cinco JSONs de saída. |
| API | `Lorenna/app.py` | Expõe saídas e busca semântica local. |
| Dados | `Leo/data/`, `scrapping/` | PublicHearingBR local e dados brutos da Câmara. |

## Fluxo de dados

1. `preparar_dados.py` produz `Lorenna/saidas/deputados.json`.
2. `pipeline.py` lê essa entrada e grava os JSONs classificados e candidatos em `Lorenna/saidas/`.
3. `app.py` lê esses JSONs; não recalcula a análise política.
4. `processar_scrapping.py` converte discursos de `scrapping/` em entradas compatíveis em `saidas_scrapping/`.

Não trate `mandato_votos_*.jsonl.gz` como fala: o pipeline não possui adaptador de votos nominais.

## Regras para alterações

- Preserve `review_required: true` em toda saída candidata.
- Mantenha `opinioes` e `posicionamento_politico_fala` como listas paralelas; o índice deve identificar a mesma fala nos dois campos.
- Preserve `sessao_id`, `assunto` e `trechos_transcricao` ao transformar dados. Eles são a trilha de auditoria.
- Não altere limiares, âncoras ou referências partidárias sem atualizar testes e `ARTIGO.md`.
- Separe recuperação temática de decisão NLI. Similaridade de embedding não é contradição.
- Trate mapas de partido e regras lexicais como hipóteses editoriais revisáveis, não como dados observados.
- Não inclua tokens, credenciais, modelos baixados ou arquivos de cache no repositório.
- Não sobrescreva dados brutos em `Leo/data/` ou `scrapping/`.

## Convenções de saída

O pipeline gera:

- `deputados_classificados.json`
- `pares_potencialmente_incompativeis.json`
- `contradicoes_partido.json`
- `tensoes_transversais.json`
- `falas_defensivas.json`

Os nomes históricos devem ser preservados, inclusive `pares_potencialmente_incompativeis.json` sem o segundo "t" em "incompativeis", para evitar quebrar consumidores existentes.

## Validação obrigatória

Para mudanças de lógica semântica, execute:

```bash
python3 Lorenna/tests/test_logic_smoke.py
PYTHONPATH=Lorenna pytest Lorenna/tests -q
```

Para mudanças em preparação, execute também:

```bash
python3 Lorenna/tests/test_preparar_dados.py
```

Para mudanças no pipeline, faça ao menos uma execução reduzida antes da execução integral:

```bash
python3 Lorenna/pipeline.py \
  --input Lorenna/saidas/deputados.json \
  --max-deputados 1
```

Modelos do Hugging Face podem ser baixados na primeira execução. Não considere uma falha de rede como falha da lógica; valide primeiro os testes que usam modelos sintéticos.

## Revisão de mudanças

Priorize estes riscos durante revisão:

- Rótulos políticos tratados como verdade ou atribuídos à pessoa em vez da fala.
- Perda de contexto transcrito, sessão ou assunto em saídas.
- Regressões de índice entre opinião e classificação.
- Regras que transformem similaridade em contradição sem NLI e revisão humana.
- Alterações silenciosas nos thresholds, âncoras ou mapas partidários.
- JSONs incompatíveis com `Lorenna/app.py` ou `Lorenna/relatorios.py`.

Ao relatar uma mudança, informe os arquivos alterados, a validação executada e se a execução dos modelos reais foi realizada.
