# Resultado v24 — Factorization Machine PyTorch

**Sem ganho aprovado; nenhum CSV de submissão v24 foi gerado.**

Pesquisa de 09/11/2026, encerramento até 14/11/2026. Uma única configuração FM foi executada, sem bibliotecas novas. Os sete componentes v21 foram retreinados a partir dos dois CSVs input/. Sem carregar previsões, OOFs, checkpoints ou modelos. Os arquivos NPZ gerados são evidências de saída, nunca entradas do notebook. Nenhuma variável SK_ID_CURR, transformação de ID, dado externo ou regra manual de previsão foi usado.

## Protocolo

Configuração e estratos constam em PLANO.md e evidence/frozen_protocol.json, congelados antes de qualquer treino. FM de segunda ordem, posto 16, termo linear, BCE sem pesos, AdamW lr=0,002/weight_decay=0,001, batch 2048, seed 42 e duas threads. Features plusx; categorias one-hot e numéricas contínuas por quantis normais. Todos os ajustes novos são feitos apenas no treino correspondente.

Épocas escolhidas em validação interna de 15% do treino externo; depois novo modelo treinado do zero no treino externo completo. Inverter TARGET da parte externa, e também da validação interna quando aplicável, não alterou os frames supervisionados. O early stopping e o pré-processamento não supervisionado train+test da referência v21 foram preservados. Esse isolamento da FM não torna todo o pipeline herdado estritamente indutivo.

Grade de peso [0%,5%,10%,15%,20%], preservando proporções v21. Empates pelo menor peso. Percentis ao quadrado; seleção somente com rótulos 0–1. Se a triagem passa, a mesma grade é reaplicada aos ranks globais com somente rótulos 0–1 antes de avaliar 2–4. Promoção exige delta 2–4 ≥0,00010 e delta total >0. Estratos são diagnóstico, sem regra adicional de promoção e sem uso em features ou pesos de treino.

## Corte rápido 0–1

Na triagem, os ranks usam somente as OOF disponíveis 0–1. Isso difere levemente dos ranks globais dos registros históricos da v21; a comparação nova usa os mesmos ranks disponíveis para referência e candidata.

| Peso FM | AUC ensemble 0–1 | Delta v21 |
|---|---|---|
| 0.00000000 | 0.77300263 | 0.00000000 |
| 0.05000000 | 0.77297630 | -0.00002633 |
| 0.10000000 | 0.77289892 | -0.00010371 |
| 0.15000000 | 0.77278480 | -0.00021783 |
| 0.20000000 | 0.77262211 | -0.00038052 |

Peso escolhido na triagem: 0%.

**Corte rápido acionado: 2–4 não foram treinadas nem avaliadas. Não houve treino integral, CSV de submissão, SHA-256 de CSV ou duas reproduções.**

## AUCs por parte

| Parte | v21 histórica | v21 recalculada | Candidata | FM isolada | Delta recalculado |
|---|---|---|---|---|---|
| parte_0 | 0.77039940 | 0.77039629 | 0.77039629 | 0.76233991 | 0.00000000 |
| parte_1 | 0.77561421 | 0.77560976 | 0.77560976 | 0.76601042 | 0.00000000 |
| parte_2 | 0.77868255 | Não calculada | Não calculada | Não calculada | Não calculada |
| parte_3 | 0.77127463 | Não calculada | Não calculada | Não calculada | Não calculada |
| parte_4 | 0.77454752 | Não calculada | Não calculada | Não calculada | Não calculada |
| partes_0_1 | 0.77300713 | 0.77300263 | 0.77300263 | 0.76390442 | 0.00000000 |
| partes_2_4 | 0.77478091 | Não calculada | Não calculada | Não calculada | Não calculada |
| total | 0.77405676 | Não calculada | Não calculada | Não calculada | Não calculada |

Valores históricos são registros, não execuções novas. Células não calculadas respeitam o corte rápido. AUCs conjuntas são calculadas nas linhas reunidas, não por média das AUCs das partes.

## Estratos de tarifa

Chave round(AMT_CREDIT/AMT_ANNUITY,2); frequência apenas no treino externo. Raras=0–9 (inclui chaves inéditas), 10–49, 50–199 e ≥200, com ausentes/inválidas em separado. Cada percentual usa como denominador as linhas do respectivo conjunto avaliado. Definições e atribuições foram congeladas antes do primeiro fit em frozen_strata.csv. AUC não calculada indica ausência de linhas ou de ambas as classes.

| Estrato | Linhas | % linhas | Positivos | v21 | Candidata | Delta | FM isolada |
|---|---|---|---|---|---|---|---|
| raras_0_9 | 1670 | 2.43859700 | 115 | 0.73498672 | 0.73498672 | 0.00000000 | 0.70879911 |
| pouco_frequentes_10_49 | 7772 | 11.34896761 | 535 | 0.76670304 | 0.76670304 | 0.00000000 | 0.74645765 |
| comuns_50_199 | 19031 | 27.78978418 | 1502 | 0.77854514 | 0.77854514 | 0.00000000 | 0.77319115 |
| muito_frequentes_200_mais | 40009 | 58.42265121 | 3377 | 0.77227069 | 0.77227069 | 0.00000000 | 0.76353443 |
| ausentes_invalidas | 0 | 0.00000000 | 0 | Não calculada | Não calculada | Não calculada | Não calculada |

Escopo da tabela: partes_0_1. Tabelas por parte e demais agregados disponíveis em strata_table.csv.

## Correlação de ranking com a FM

Spearman das previsões OOF brutas de cada componente, antes dos percentis ao quadrado. Maior diversidade de ranking, sozinha, não implica ganho de AUC.

| Componente v21 | Spearman com FM |
|---|---|
| LightGBM B + EXT | 0.95069149 |
| LightGBM B v7 | 0.94223035 |
| MLP 8 sementes | 0.95241428 |
| CatBoost Plain + EXT | 0.94856504 |
| TabM periódico | 0.93261852 |
| TabM por pares | 0.93272310 |
| CatBoost Ordered depth 5 | 0.94430435 |

## Pesos avaliados

| Componente | Peso |
|---|---|
| LightGBM B + EXT | 0.15657187 |
| LightGBM B v7 | 0.00590062 |
| MLP 8 sementes | 0.21094875 |
| CatBoost Plain + EXT | 0.17445375 |
| TabM periódico | 0.19462500 |
| TabM por pares | 0.15750000 |
| CatBoost Ordered depth 5 | 0.10000000 |
| FM PyTorch | 0.00000000 |

Sem aprovação, esses pesos não constituem uma nova receita para submissão; a referência permanece v21. Nenhum ajuste foi feito após a triagem/confirmação.

## O que foi executado de verdade

Pesquisa concluída em 30.83 minutos, macOS-27.2-arm64-arm-64bit, Python 3.11.15. Dois workers de CV, duas threads por modelo; não é uma medição em apenas dois núcleos totais. Google Colab/Linux não foi executado. O runner incluído executa todas as células como Python, sem kernel Jupyter.

A primeira tentativa foi interrompida pelo sandbox macOS (OpenMP/SHM/sysctl) antes dos treinos externos. O prepare havia ajustado os auxiliares EXT sem TARGET. Log e notebook com erro foram preservados.

A segunda tentativa revelou um erro de implementação: colunas de contagem com dois níveis (NAME_CONTRACT_TYPE_COUNT e CODE_GENDER_COUNT, até 210.782) foram tratadas como indicadores e não normalizadas. AUCs FM dessa tentativa: 0,71212829 e 0,62620075; não são um teste válido da hipótese. A tentativa foi interrompida antes de selecionar peso ou executar 2–4. Código, protocolo, diagnóstico de escala e logs inválidos foram preservados.

Correção única, registrada e congelada na revisão 2 antes da terceira tentativa: somente indicadores com valores em {0,1} pulam os quantis; contagens de dois níveis são normalizadas. Sem mudança de hiperparâmetros, seeds, arquitetura, estratos, grade ou promoção. A terceira tentativa começou em processo novo e retreinou a referência; todos os resultados apresentados nas tabelas pertencem a essa tentativa válida.

Não executados: treino final em 100%, dupla reprodução, verificação do limite final de 60 minutos e teste em Colab. Nenhum SHA-256 de CSV de submissão está disponível.

## Reprodução da pesquisa

Coloque input/train.csv e input/test.csv ao lado do notebook de pesquisa e execute todas as células em runtime novo. Use pasta limpa para preservar evidências anteriores. Não é necessário nem permitido carregar os NPZ de evidência.

```sh
PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v24_pesquisa.ipynb --name pesquisa_nova
```

Versões: numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; lightgbm 4.6.0; catboost 1.2.8; torch 2.6.0; tabm 0.0.3; rtdl-num-embeddings 0.0.12; threadpoolctl 3.6.0; joblib 1.6.0.

Score público, posição e total de participantes v24 não informados; nenhuma submissão automática. Os folds já foram usados em pesquisas anteriores, portanto a confirmação não é um holdout intocado.
