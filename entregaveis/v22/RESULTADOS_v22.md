# Resultado v22 — densidade do empréstimo

**Sem ganho aprovado na triagem; nenhum CSV v22 foi gerado.**

Execução concluída em 28/10/2026; fechamento dos entregáveis em 30/10/2026. A comparação é uma ablação de LightGBM B com uma semente (42), não uma comparação entre ensembles completos. O filtro previamente declarado exigia AUC conjunta 0–1 e média das diferenças por parte positivas. Ambas as variantes falharam.

| Parte | LightGBM de referência | Densidade geral | Densidade por contrato |
|---|---:|---:|---:|
| 0 | 0.76703766 | 0.76742443 | 0.76794206 |
| 1 | 0.77387364 | 0.77342040 | 0.77245519 |
| 0–1 juntas | 0.77049997 | 0.77044930 | 0.77017167 |
| 2–4 | Não executado | Não executado | Não executado |
| Total | Não executado | Não executado | Não executado |

Não há pesos novos, treino integral, SHA-256 de CSV ou dupla reprodução de CSV nesta versão. A reprovação desta implementação não prova que toda forma de KDE ou seu uso em um ensemble falharia.

StratifiedKFold(5, shuffle=True, random_state=42), ordem original do arquivo. Ajuste da densidade somente nas linhas de treino da parte, usando AMT_CREDIT, AMT_ANNUITY e NAME_CONTRACT_TYPE. Sem TARGET ou SK_ID_CURR no estimador. Supervised.add exclui o TARGET da parte avaliada; a inversão dos rótulos de validação não alterou os frames supervisionados. As 24 definições originais da v17 foram preservadas.

O pré-processamento original continua usando covariáveis de train+test em etapas não supervisionadas. Portanto, somente a nova densidade foi ajustada estritamente no treino de cada parte; esta execução não comprova um pipeline totalmente indutivo.

Tempo real: 4.19 minutos incluindo prepare; macOS-27.2-arm64-arm-64bit, Python 3.11.15. Dois processos de validação, duas threads por modelo. Colab/Linux não executado.

O notebook de pesquisa inclui todo o código e refaz a triagem a partir de input/train.csv e input/test.csv. Não carrega previsões, OOFs, modelos ou checkpoints. Coloque input/ ao lado do notebook e execute todas as células. O ZIP não inclui os dados.

```sh
PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v22_pesquisa.ipynb --name pesquisa_nova
```

Use um nome novo por execução. O runner executa as células em um processo Python; não usa kernel Jupyter.

Versões: numpy 2.3.5; pandas 2.2.3; scipy 1.17.0; scikit-learn 1.8.0; lightgbm 4.6.0; catboost 1.2.8; torch 2.6.0; tabm 0.0.3; rtdl-num-embeddings 0.0.12; threadpoolctl 3.6.0; joblib 1.6.0.

Ideia fornecida no resumo do Slack pelo usuário, atribuída a Theophile Avenel e sameenakb. Implementação própria com histograma suavizado, largura de banda Silverman, detecção de picos e mínimos. Os ganhos de outros participantes não foram reproduzidos aqui.
