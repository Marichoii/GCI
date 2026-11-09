# Home Credit — resultado v21

**Ganho confirmado; CSV reproduzido duas vezes.**

Referência: v20 (público informado 0,77877, rank histórico 38/2.741 em 26/10/2026). Somente input/train.csv e input/test.csv; todos os seis componentes da referência treinados novamente. Nenhuma previsão, OOF, checkpoint ou modelo salvo carregado. Nenhuma feature de ID, nenhum dado externo e nenhuma regra manual de previsão. 24 definições originais da v17 preservadas; Supervised fora da parte e verificação por inversão do TARGET avaliado.

Candidatos fixados antes da execução: CatBoost Ordered depth 5 e 6, lr 0,04, L2=5, seed 42; LightGBM B com extra_trees=True e extra_seed=42. São esquemas de treinamento diferentes dos CatBoost Plain descartados anteriormente.

StratifiedKFold(5, shuffle=True, random_state=42), ordem original. Escolher uma família e peso novo na grade 0–0,30 com passo 0,025 apenas nas partes 0–1. Os seis pesos antigos mantêm suas proporções; percentis ao quadrado. Empates preferem peso menor. Sem ganho na triagem, não avaliar 2–4. Se passar, completar OOFs e refixar o peso com percentis globais usando somente rótulos 0–1; gravar frozen_selection.json antes de medir 2–4 e total. Não ajustar após confirmação.

## Seleção nas partes 0–1

AUC de referência com percentis das linhas disponíveis: 0.77297592.

| Candidato | Peso novo | Melhor AUC do ensemble | Diferença |
|---|---:|---:|---:|
| cat_ordered_depth5 | 0.100 | 0.77300263 | +0.00002671 |
| cat_ordered_depth6 | 0.025 | 0.77297900 | +0.00000309 |
| lgb_extra_trees | 0.000 | 0.77297592 | +0.00000000 |

## Confirmação com percentis globais

| Parte | v20 | v21 | Diferença |
|---|---:|---:|---:|
| parte_0 | 0.77030816 | 0.77039940 | +0.00009124 |
| parte_1 | 0.77566397 | 0.77561421 | -0.00004976 |
| parte_2 | 0.77865069 | 0.77868255 | +0.00003186 |
| parte_3 | 0.77115858 | 0.77127463 | +0.00011606 |
| parte_4 | 0.77454245 | 0.77454752 | +0.00000507 |
| partes_0_1 | 0.77298495 | 0.77300713 | +0.00002218 |
| partes_2_4 | 0.77471634 | 0.77478091 | +0.00006457 |
| total | 0.77400566 | 0.77405676 | +0.00005111 |

Componente selecionado: `{'label': 'cat_ordered_depth5', 'kind': 'cat_ordered', 'depth': 5}`. Peso novo: 0.100.

| Componente | Peso final | Árvores / épocas calculadas |
|---|---:|---:|
| LightGBM B + EXT auxiliares | 15.6571875% | 1646 |
| LightGBM B — variáveis v7 | 0.5900625% | 1793 |
| MLP — 8 sementes | 21.0948750% | 11 |
| CatBoost original | 17.4453750% | 1103 |
| TabM periódico original | 19.4625000% | 14 |
| TabM periódico por pares | 15.7500000% | 11 |
| cat_ordered_depth5 | 10.0000000% | 1089 |

Tamanhos calculados para treino integral: referência `[1646, 1793, 11, 1103, 14, 11]`; novo `1089`. Média do early stopping × 1,15, arredondada; a execução integral depende da aprovação e está documentada abaixo.

## Execução real

Pesquisa: 112.87 minutos de tempo decorrido, macOS-27.2-arm64-arm-64bit, Python 3.11.15. CV com dois processos independentes e duas threads por modelo. Tempo e bytes no Colab/Linux não foram verificados.

Duas reproduções integrais: 37.64 e 37.63 minutos. Todas as células de código executadas pelo runner Python incluído, sem kernel Jupyter, em dois processos novos e diretórios separados. As duas reproduções foram iniciadas em paralelo; cada uma treina seus componentes sequencialmente.

CSV SHA-256, execução 1: `48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756`.
CSV SHA-256, execução 2: `48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756`.

CSV com 61.500 linhas, colunas SK_ID_CURR,TARGET e IDs na ordem de input/test.csv. Score público e rank da v21 aguardam submissão; melhora local não garante ganho no placar.

Para reproduzir: coloque gci_home_credit_v21.ipynb ao lado da pasta input e execute todas as células em runtime novo. O padrão treina tudo e grava output/submission.csv, sem depender do CSV incluído. AUCs da validação são registros fixos; GCI_RUN_CV=1 refaz a pesquisa. GCI_CV_ONLY=1 evita treino integral nessa opção.

Alternativa sem Jupyter, igual ao método realmente executado nesta verificação, em processo Python novo:

```sh
GCI_RUN_CV=0 GCI_CV_ONLY=0 PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v21.ipynb --name reproducao_nova
```

Use outro nome para cada execução; o runner preserva os arquivos da execução anterior.

## Versões e fontes

Python 3.11.15; numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; lightgbm 4.6.0; catboost 1.2.8; torch 2.6.0; tabm 0.0.3; rtdl-num-embeddings 0.0.12; threadpoolctl 3.6.0; joblib 1.6.0.

[CatBoost Ordered](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type) · [LightGBM extra_trees](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#extra_trees).

Os folds já foram usados em pesquisas anteriores. Esta confirmação é um critério operacional, não uma estimativa independente de todas as tentativas. O top 10 fornecido pelo usuário não revela os métodos usados nem estabelece um teto teórico para modelos permitidos.
