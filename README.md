# GCI 2026 — Home Credit Default Risk

Histórico de modelos, submissões e resultados da competição Home Credit do GCI.
O objetivo é prever a probabilidade de inadimplência, com avaliação por ROC AUC.

Este repositório no GitHub reúne **somente o material da competição**.
O trabalho final de Telecom / Company A é uma atividade separada do curso e fica
fora deste repositório, nas pastas locais `final/` e `entregaveis/final/`.

## Organização

| Caminho | Conteúdo |
| --- | --- |
| [`competition/`](competition/) | Código de pesquisa, geração de notebooks, auditoria e empacotamento |
| [`competition/sources/`](competition/sources/) | Fontes históricas e enunciado oficial preservado |
| [`entregaveis/`](entregaveis/README.md) | Notebooks, ZIPs, CSVs gerados, relatórios e evidências, separados por versão |
| [`entregaveis/ranking.csv`](entregaveis/ranking.csv) | Scores e posições informados, com data e fonte |

Os dados brutos oficiais, ambientes Python, caches e registros locais de upload
ficam fora do Git. As cópias de execução em `competition/` também ficam locais;
os pacotes preservados estão em `entregaveis/vNN/`.

## Resultados registrados

O maior score público registrado é **0.77877**, da [v20](entregaveis/v20/README.md),
com posição **38 de 2.741**, informada em **26/10/2026**.
A [v21](entregaveis/v21/README.md) tem score público **0.77876** e posição **38**,
informados em **28/10/2026**; o total de participantes desse relato não foi informado.

A pesquisa mais recente, [v24](entregaveis/v24/RESULTADOS_v24.md), não passou pelo
critério de ganho local e não gerou CSV de submissão.
O [índice completo](entregaveis/README.md) mantém também as tentativas sem ganho.
Esses resultados são históricos: não representam uma consulta ao placar atual.
Métricas de validação local e scores públicos são registrados separadamente.

## Reproduzir uma submissão

Os registros de execução usam Python 3.11.15 em macOS ARM64. Cada versão inclui
seu `requirements.txt`, relatório de execução e limitações conhecidas.

Para reproduzir a v20, crie uma pasta de trabalho nova, copie para ela
`gci_home_credit_v20.ipynb`, `execute_notebook.py` e `requirements.txt` de
[`entregaveis/v20/`](entregaveis/v20/). Obtenha os dois CSVs oficiais pelo material
do curso e coloque-os em `input/train.csv` e `input/test.csv` nessa mesma pasta.
Use a estrutura e os hashes descritos em
[`competition/input/README.md`](competition/input/README.md).

Na pasta de trabalho, prepare um ambiente Python 3.11 e execute:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
GCI_RUN_CV=0 GCI_CV_ONLY=0 PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v20.ipynb --name reproducao_nova
```

O notebook treina os componentes e grava `output/submission.csv`. O runner guarda
o notebook executado e o relatório em `runs/reproducao_nova/`; use um nome novo
em cada execução. O procedimento completo está no
[relatório da v20](entregaveis/v20/RESULTADOS_v20.md).

## Preservação do histórico

Cada versão conserva seus próprios artefatos, `rank.json`, `RANK.md` e manifesto
SHA-256. Os notebooks, CSVs e ZIPs validados são mantidos com os bytes originais;
organizar o repositório não exige treinar os modelos novamente.
Os manifestos preservam o inventário local original, que pode incluir arquivos
de sistema omitidos do Git.

Posições não são deduzidas a partir do score. Relatos de datas diferentes e CSVs
diferentes com o mesmo número de versão permanecem separados no histórico.
As dependências ausentes da v16 estão documentadas em
[`DEPENDENCIAS_HISTORICAS.md`](entregaveis/v16/DEPENDENCIAS_HISTORICAS.md).
