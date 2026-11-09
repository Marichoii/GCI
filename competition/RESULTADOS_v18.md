# GCI Home Credit v18 — resultados executados

A v18 passou o critério solicitado: AUC maior nas partes 2–4 e no total.
O peso novo foi escolhido somente nas partes 0–1, numa grade de 0 a 50%, passo
de 2,5 pontos percentuais. As proporções dos quatro componentes da v17 foram
preservadas. Não houve ajuste após consultar as partes 2–4.

| Parte | v17 recalculada | v18 | Delta |
|---|---:|---:|---:|
| parte_0 | 0.77006649 | 0.77031278 | +0.00024629 |
| parte_1 | 0.77479888 | 0.77535567 | +0.00055679 |
| parte_2 | 0.77819588 | 0.77844291 | +0.00024704 |
| parte_3 | 0.77117097 | 0.77118114 | +0.00001017 |
| parte_4 | 0.77378264 | 0.77438509 | +0.00060245 |
| partes_0_1 | 0.77241009 | 0.77273013 | +0.00032004 |
| partes_2_4 | 0.77434624 | 0.77458625 | +0.00024001 |
| total | 0.77356845 | 0.77382186 | +0.00025341 |

A referência acima é a v17 recalculada com isolamento do TARGET na chamada de
CV da MLP. A referência histórica informada era 0,77371 no total, 0,77268 nas
partes 0–1 e 0,77440 nas partes 2–4. A chamada original montava o teste com
todos os TARGETs antes de ajustar os quantis conjuntos de treino/validação/teste;
isso permitia influência indireta dos rótulos de validação. A nova chamada
monta também o teste usando somente os rótulos do treino da parte. Ela é usada
igualmente na referência e na candidata. As 24 funções/classes da v17 foram
preservadas textualmente, inclusive Supervised.add, _nn_split e train_nn.

O arquivo target_isolation_audit.json registra o teste nos cinco folds reais:
inverter todos os TARGETs da validação não modifica os frames supervisionados
do treino, da validação ou do teste calculados a partir do treino da parte.

| Componente | Peso | Árvores/épocas finais |
|---|---:|---:|
| LightGBM B + EXT auxiliares | 19.875000% | 1646 |
| LightGBM B (v7) | 3.825000% | 1793 |
| MLP (8 sementes) | 22.350000% | 11 |
| CatBoost + EXT auxiliares | 28.950000% | 1103 |
| TabM periódico | 25.000000% | 14 |

Os tamanhos finais foram recalculados: round(média de todas as melhores
iterações/épocas da CV × 1,15). O treino final usa 100% das linhas.
O TabM recebe base da v17 + 13 auxiliares EXT e as 10 colunas de Supervised.add.
Configuração: arch_type=tabm, k=8, n_blocks=2, d_block=128, dropout=0,15;
PeriodicEmbeddings(d_embedding=8, frequency_init_scale=0,1, lite=True);
AdamW(lr=0,002, weight_decay=0,01), batch 1024, até 40 épocas, patience 7.
Na CV, ReduceLROnPlateau(factor=0,5, patience=2); treino final sem validação,
LR constante e número fixo de épocas. Sementes da CV: 42+100×fold; final: 42.
Os outros componentes conservam as sementes e hiperparâmetros da v17.

SHA-256 do CSV, execução 1: `1dfc63131cc4f064621b82f8e1042b620647676f36e2b7d8da3539f1dbc0aada`

SHA-256 do CSV, execução 2: `1dfc63131cc4f064621b82f8e1042b620647676f36e2b7d8da3539f1dbc0aada`

Os CSVs foram comparados byte a byte. Todas as células de código do mesmo
notebook foram executadas em ordem em dois processos Python novos pelo runner
execute_notebook.py. Não foi usado servidor/kernel Jupyter nessas execuções.
As duas execuções reconstruíram os auxiliares e todos os modelos do zero;
nenhuma previsão, OOF, checkpoint ou modelo salvo foi carregado.

Tempo da validação completa: 57.00 minutos.
Tempo das reproduções completas (incluindo auxiliares):
12.42 e 14.78 minutos.
Ambiente medido: macOS-27.0.1-arm64-arm-64bit; Python 3.11.15, CPU,
2 threads por biblioteca/modelo e pool inter-op do PyTorch fixo em 2.
Colab gratuito não foi executado ou cronometrado. A reprodução padrão pula a
CV e treina todos os modelos a partir de input/ com a receita já validada.

Versões efetivamente instaladas:

- numpy: 2.3.5
- pandas: 2.2.3
- scipy: 1.17.0
- scikit-learn: 1.8.0
- lightgbm: 4.6.0
- catboost: 1.2.8
- torch: 2.6.0
- tabm: 0.0.3
- rtdl-num-embeddings: 0.0.12
- threadpoolctl: 3.6.0

As 69 versões instaladas no ambiente local estão em installed_versions_local.txt.
requirements.txt fixa as bibliotecas do pipeline; no Linux o notebook instala
o PyTorch pelo índice oficial de CPU quando necessário. Reiniciar o runtime
antes de reproduzir evita módulos antigos já importados.

## Reproduzir

Coloque gci_home_credit_v18.ipynb ao lado da pasta input/ contendo apenas os
dois arquivos train.csv e test.csv originais e execute todas as células em um
runtime novo. Se necessário, defina GCI_WORKDIR para essa pasta antes da
primeira célula. O notebook não depende de sample_submission.csv.
O resultado é output/submission.csv, com IDs na ordem de test.csv.

Para repetir também toda a validação, defina GCI_RUN_CV=1 antes de iniciar
(ou mude RUN_CV para True). GCI_CV_ONLY=1 executa somente a validação.
Para o runner usado nesta revisão, com Python 3.11 e as dependências instaladas:

```sh
PYTHONHASHSEED=42 python execute_notebook.py --name reproduction_1
PYTHONHASHSEED=42 python execute_notebook.py --name reproduction_2
```

Cada nome de execução deve ser novo. O runner arquiva o notebook executado,
o relatório e uma cópia do CSV recém-treinado; nunca os lê como entrada de modelo.
Os hashes comprovam repetição neste ambiente. Não foi medida identidade dos
bytes entre macOS arm64 e Linux/Colab.

## Dados e evidência

- train.csv SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- test.csv SHA-256: `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`
- Divisão: StratifiedKFold(n_splits=5, shuffle=True, random_state=42), ordem do arquivo.
- Ensemble: soma ponderada de (percentil)^2 usando to_rank original da v17.
- SK_ID_CURR é descartado antes das features; usado somente para identificar a saída.
- Sem dados externos, regras manuais sobre previsões ou modelos pré-treinados.
- O ZIP v13 fornecido estava truncado. experiment_v13.py foi recuperado de
  uma entrada local completa, com tamanho e CRC conferidos. Somente a
  arquitetura periódica foi adaptada; os artefatos treinados antigos não foram usados.
- As evidências incluem os relatórios e os notebooks executados na validação
  e nas duas reproduções, além da auditoria de TARGET e do teste de hashes.
- As partes 2–4 foram usadas para confirmação no protocolo solicitado; o
  early stopping de cada modelo usa a validação de seu fold, como na v17.
  A melhora medida não constitui uma medição do leaderboard privado.

Fontes de implementação: v17 e experiment_v13.py fornecidos pelo usuário;
[TabM](https://github.com/yandex-research/tabm) e
[embeddings numéricos](https://github.com/yandex-research/rtdl-num-embeddings).
Implementação e auditoria assistidas por ChatGPT. Acrescente o link da conversa
à entrega conforme as instruções do curso.
