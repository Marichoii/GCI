# Resultado v23 — ensemble em log-odds

**Sem ganho aprovado; nenhum CSV v23 foi gerado.**

Referência: v21, público informado0,77876 e posição38 em28/10/2026, total de participantes não informado. Melhor público registrado: v20,0,77877,38/2.741 em26/10/2026. Variação pública de0,00001 não demonstra superioridade privada.

Todos os sete componentes da v21 treinados novamente a partir dos dois arquivos input/train.csv e input/test.csv. Sem carregar previsões, OOFs, modelos ou checkpoints. Sem feature de ID, dados externos ou regras manuais de previsão. As 24 definições originais da v17 foram preservadas. Supervised.add exclui os rótulos da parte avaliada. O pré-processamento não supervisionado herdado da v17 usa covariáveis train+test; o README examinado não esclarece expressamente essa opção.

Única hipótese: média ponderada de log-odds com pesos dos sete componentes fixos. Clip1e-6 apenas para domínio numérico do log. Transformar o score agregado em percentil ao quadrado e combinar com o blend atual. Alpha na grade0,0.1,...,1, escolhido somente em0–1; empates preferem menor alpha. Alpha1 equivale à ordenação da média em log-odds. Não houve treino de meta-modelo ou recalibração. Probabilidades dos componentes não são necessariamente calibradas.

StratifiedKFold(5,shuffle=True,random_state=42), ordem original. Na triagem, percentis somente nas OOFs disponíveis0–1. Se a triagem passar, completar OOFs e refixar alpha apenas com rótulos0–1 e percentis globais. Salvar frozen_selection.json antes de medir2–4 e total. Nenhum ajuste após confirmação.

## Seleção nas partes0–1

AUC de referência na triagem: 0.77300263.

| Alpha log-odds | AUC0–1 | Diferença |
|---|---:|---:|
| 0.0 | 0.77300263 | +0.00000000 |
| 0.1 | 0.77301525 | +0.00001262 |
| 0.2 | 0.77302559 | +0.00002297 |
| 0.3 | 0.77303619 | +0.00003357 |
| 0.4 | 0.77304631 | +0.00004369 |
| 0.5 | 0.77305580 | +0.00005317 |
| 0.6 | 0.77306597 | +0.00006334 |
| 0.7 | 0.77307449 | +0.00007187 |
| 0.8 | 0.77308182 | +0.00007919 |
| 0.9 | 0.77308948 | +0.00008685 |
| 1.0 | 0.77309639 | +0.00009377 |

## Confirmação

| Parte | v21 | v23 | Diferença |
|---|---:|---:|---:|
| parte_0 | 0.77039940 | 0.77045830 | +0.00005891 |
| parte_1 | 0.77561421 | 0.77571411 | +0.00009990 |
| parte_2 | 0.77868255 | 0.77854463 | -0.00013792 |
| parte_3 | 0.77127463 | 0.77125765 | -0.00001699 |
| parte_4 | 0.77454752 | 0.77443036 | -0.00011716 |
| partes_0_1 | 0.77300713 | 0.77309639 | +0.00008926 |
| partes_2_4 | 0.77478091 | 0.77469646 | -0.00008445 |
| total | 0.77405676 | 0.77403751 | -0.00001926 |

Alpha congelado: 1.0.

| Componente | Peso dentro de cada blend | Árvores/épocas |
|---|---:|---:|
| LightGBM B + EXT auxiliares | 0.156571875 | 1646 |
| LightGBM B — variáveis v7 | 0.005900625 | 1793 |
| MLP — oito sementes | 0.210948750 | 11 |
| CatBoost Plain | 0.174453750 | 1103 |
| TabM periódico | 0.194625000 | 14 |
| TabM periódico por pares | 0.157500000 | 11 |
| CatBoost Ordered depth5 | 0.100000000 | 1089 |

Esses pesos são usados em ambos os blends. Score final=(1−alpha)×blend atual+alpha×percentil² da média ponderada dos log-odds. Tamanhos calculados para treino integral=média do early stopping×1,15, arredondada. Quando a candidata é reprovada, esses tamanhos não são usados e os pesos avaliados não constituem uma receita aprovada.

## Execução real e reprodução

Validação: 107.66 minutos; macOS-27.2-arm64-arm-64bit, Python 3.11.15. Dois processos de CV independentes, duas threads por modelo. Colab/Linux não executado.

Nenhum treino integral novo, CSV, SHA-256 de CSV ou dupla reprodução foi executado após a reprovação. Para repetir a pesquisa, coloque input/ ao lado do notebook de pesquisa e execute todas as células.

```sh
PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v23_pesquisa.ipynb --name pesquisa_nova
```

Use um nome novo por execução. O ZIP inclui código e evidências; não inclui os dados.

Versões: numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; lightgbm 4.6.0; catboost 1.2.8; torch 2.6.0; tabm 0.0.3; rtdl-num-embeddings 0.0.12; threadpoolctl 3.6.0; joblib 1.6.0.

Hipótese trazida pelo resumo do Slack do usuário: https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965992208839 . Nenhuma thread foi consultada ao vivo. Ganhos relatados por outros participantes não garantem ganho aqui. Os folds foram reutilizados em pesquisas anteriores; a confirmação não é uma estimativa independente de todas as tentativas.
