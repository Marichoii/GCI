# Home Credit — resultado v20

**Ganho confirmado em 2–4 e no total; duas reproduções com CSV idêntico.**

Somente input/train.csv e input/test.csv. Nenhuma feature de ID; nenhuma previsão, OOF, checkpoint ou modelo salvo carregado. As 24 definições originais da v17 e o wrapper estrito da MLP foram preservados.

Quatro candidatos fixados antes dos resultados: LightGBM B com max_bin 255 ou 1023, uma semente 42 cada; TabM periódico com perda por pares em proporção 50% ou 100%, mesma arquitetura e sementes por parte. Também se avalia reequilíbrio dos cinco pesos existentes.

StratifiedKFold(5, shuffle=True, random_state=42), ordem original. A família é escolhida nas partes 0–1; depois de completar as OOFs, os pesos são fixados usando apenas os rótulos 0–1 e os percentis globais. O arquivo frozen_selection.json é escrito antes de calcular a AUC do ensemble de confirmação. Não há ajustes após 2–4.

Busca dos pesos: valores não negativos somando 1; cinco pesos antigos dentro de ±0,10 da v18; novo componente até 0,30. Inicialização em grade e transferências de massa 0,025/0,01, no máximo quatro rodadas por passo. Mantém percentis ao quadrado. Nenhum modelo de stacking.

## Resultado da seleção

AUC v18 na triagem, percentis das linhas disponíveis: 0.77272055.

| Estratégia | AUC nas partes 0–1 da triagem |
|---|---:|
| Somente reequilíbrio | 0.77275606 |
| lgb_bins255 | 0.77275606 |
| lgb_bins1023 | 0.77275606 |
| tabm_pair_half | 0.77296965 |
| tabm_pair_full | 0.77297867 |

## Confirmação com percentis globais

| Parte | v18 | v20 | Diferença |
|---|---:|---:|---:|
| parte_0 | 0.77031278 | 0.77030816 | -0.00000462 |
| parte_1 | 0.77535567 | 0.77566397 | +0.00030830 |
| parte_2 | 0.77844291 | 0.77865069 | +0.00020777 |
| parte_3 | 0.77118114 | 0.77115858 | -0.00002256 |
| parte_4 | 0.77438509 | 0.77454245 | +0.00015736 |
| partes_0_1 | 0.77273013 | 0.77298495 | +0.00025482 |
| partes_2_4 | 0.77458625 | 0.77471634 | +0.00013009 |
| total | 0.77382186 | 0.77400566 | +0.00018380 |

Componente novo selecionado: `{'label': 'tabm_pair_full', 'kind': 'tabm_pairwise', 'pair_weight': 1.0}`.

| Componente | Peso | Árvores / épocas no treino integral |
|---|---:|---:|
| LightGBM B + EXT auxiliares | 17.396875% | 1646 |
| LightGBM B — variáveis v7 | 0.655625% | 1793 |
| MLP — 8 sementes | 23.438750% | 11 |
| CatBoost + EXT auxiliares | 19.383750% | 1103 |
| TabM periódico original | 21.625000% | 14 |
| TabM periódico — perda por pares | 17.500000% | 11 |

Tamanhos = média do early stopping × 1,15, arredondada. Apenas os pesos foram escolhidos com rótulos 0–1; não houve ajuste após a confirmação. As partes 0 e 3 tiveram pequenas quedas; os dois agregados exigidos melhoraram.

## Execução real

Pesquisa: 95.61 minutos de tempo decorrido, macOS ARM64 CPU. A CV usa dois processos independentes com duas threads por modelo; a reprodução final é sequencial com duas threads. Não foi medido o tempo no Colab.

Primeira tentativa de inicialização paralela bloqueada pelo sandbox no OpenMP antes de produzir resultados de validação. A execução registrada em evidence/validation é a pesquisa_02, retomada com acesso à memória compartilhada.

Execução 1: 36.14 minutos; execução 2: 36.21 minutos.

As duas reproduções foram iniciadas em paralelo neste Mac, cada uma em processo novo, diretório separado e com duas threads por modelo. Cada reprodução treina seus componentes sequencialmente. Todas as células de código foram executadas pelo runner Python incluído, sem kernel Jupyter. O tempo e a igualdade de bytes no Colab/Linux não foram verificados.

CSV SHA-256, execução 1: `59a6a3ca75f9a8eb2deaddc9a3613669182fea7708ebcd271bb5f9308a9d75ce`.
CSV SHA-256, execução 2: `59a6a3ca75f9a8eb2deaddc9a3613669182fea7708ebcd271bb5f9308a9d75ce`.

O CSV tem 61.500 linhas, IDs na ordem de input/test.csv e colunas SK_ID_CURR,TARGET. Score público e rank da v20 permanecem desconhecidos até uma submissão. A melhora local não comprova aumento no placar público.

## Como reproduzir o CSV

Coloque gci_home_credit_v20.ipynb ao lado de input/train.csv e input/test.csv e execute todas as células em runtime novo. A primeira célula instala/verifica versões. O padrão faz o treino integral e grava output/submission.csv. Não exige artefatos anteriores nem usa o CSV incluído como entrada.

Para refazer a validação, defina GCI_RUN_CV=1 e GCI_CV_ONLY=1 antes da execução; para repetir só o CSV, mantenha os dois em 0.

## Versões e fontes

Versões exatas em requirements.txt e installed_versions_local.txt; Python e plataforma nos relatórios JSON.

Python 3.11.15. numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; lightgbm 4.6.0; catboost 1.2.8; torch 2.6.0; tabm 0.0.3; rtdl-num-embeddings 0.0.12; threadpoolctl 3.6.0; joblib 1.6.0.

[LightGBM 4.6 — max_bin](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#max_bin) · [LibAUC — objetivo por pares](https://docs.libauc.org/api/libauc.losses.html#libauc.losses.PairwiseAUCLoss). A perda é implementada em PyTorch. Nenhum dado externo ou modelo pré-treinado foi usado.

A validação usa early stopping e folds já usados em pesquisas anteriores. O ganho é um critério operacional, sem garantia no teste privado.
