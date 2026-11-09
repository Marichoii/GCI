# Resultado da pesquisa v19

**Não houve ganho aprovado; nenhum CSV novo foi gerado.**

As duas configurações do LightGBM com folhas lineares foram comparadas ao ensemble v18 nas partes 0–1. Nenhuma recebeu peso positivo na seleção. As partes 2–4 não foram executadas, portanto não há uma nova AUC total.

| Configuração | AUC v18, partes 0–1 | Melhor AUC do ensemble | Peso selecionado | Melhor AUC com peso > 0 |
|---|---:|---:|---:|---:|
| linear_lambda_10 | 0.77272055 | 0.77272055 | 0.000 | 0.77269810 (peso 0.025) |
| linear_lambda_100 | 0.77272055 | 0.77272055 | 0.000 | 0.77268919 (peso 0.025) |

Os percentis da triagem usam somente as linhas das partes 0–1. Não comparar esta AUC diretamente com a AUC das mesmas partes calculada com os percentis globais da validação completa da v18.

| Parte | v18 recalculada na triagem | v19 com o peso selecionado (zero) |
|---|---:|---:|
| 0 | 0.77031688 | 0.77031688 |
| 1 | 0.77534805 | 0.77534805 |
| 0–1 | 0.77272055 | 0.77272055 |
| 2, 3, 4 e total | Não executado nesta pesquisa | Não executado |

O empate significa que a seleção excluiu o novo componente; não é um ganho.


## O que foi executado

- Notebook de pesquisa executado em processo Python novo, todas as células de código em ordem, pelo runner incluído. A versão com saídas e o log estão em `evidence/`.
- Apenas `input/train.csv` e `input/test.csv`. Todos os modelos usados nesta comparação treinados do zero; OOFs apenas na memória. Não foi carregado o CSV da v18 para validar.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, ordem original. `Supervised.add` usa o treino da parte e os mesmos cálculos internos da v17. Trocar os TARGET da validação não alterou os frames do novo componente.
- Funções originais da v17 preservadas; wrapper estrito da MLP mantido. Sementes e duas threads fixas.
- Tempo efetivo: 38.03 minutos, macOS ARM64 CPU. Não medido no Colab.
- Não houve treino integral de uma nova candidata nem duas reproduções de CSV; não existe um novo SHA-256 de submissão.
- Score público e rank da v19 não existem: nenhuma submissão nova foi feita.

## Receita testada

LightGBM binário: 9 folhas, profundidade máxima 4, mínimo 500 linhas/folha, lr 0,03, até 4.000 árvores, early stopping 150, fração de variáveis 0,44, bagging 0,74, L2 2, `linear_lambda` em {10, 100}, seed 42. Quantis normais 256 ajustados no treino, imputação pela mediana e indicadores de ausência. Categorias nativas. Sem feature de ID.

Pesos da v18 mantidos: 0,19875; 0,03825; 0,2235; 0,2895; 0,25. Peso do componente linear em 0–30%, passo 2,5%; proporções anteriores preservadas.

Fonte técnica: [documentação primária do LightGBM 4.6](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree).

## Como repetir

Coloque o notebook ao lado de `input/train.csv` e `input/test.csv` e execute todas as células. As versões necessárias são instaladas/verificadas na primeira célula. Para usar o runner, no ambiente com `requirements.txt` instalado:

```sh
python execute_notebook.py --notebook gci_home_credit_v19_pesquisa.ipynb --name pesquisa_linear_01
```

O diretório `runs/pesquisa_linear_01` precisa estar ausente antes de uma nova execução; use outro nome se já existir. O notebook escreve relatórios em `v19/evidence/`; não escreve `submission.csv`.

Notebook SHA-256: `f061b9f6fb6d453cbe0d19463ee886e51276451efd4603d682a80c8a4693c0d3`.

Versões e hashes dos dados estão no relatório JSON. Bibliotecas fixadas em `requirements.txt`; a lista do ambiente local está em `installed_versions_local.txt`.
