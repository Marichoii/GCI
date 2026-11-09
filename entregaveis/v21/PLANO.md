# Pesquisa v21 — 28/10/2026

Referência: v20, público 0,77877, posição histórica 38/2.741 no relato de 26/10/2026; OOF 0,7740056552208389.

Três candidatos fixados antes de executar: CatBoost Ordered com profundidade 5 ou 6, learning_rate 0,04, L2=5, semente 42; LightGBM B com extra_trees=True e extra_seed=42. Mantêm variáveis e pré-processamento da v20. Não repetem CatBoost Plain depth 7/8 ou learning_rate 0,02, descartados anteriormente.

Reajustar do zero os seis componentes da referência com as mesmas sementes, nas partes 0–1 originais do protocolo StratifiedKFold(5, shuffle=True, random_state=42), na ordem do CSV. Todos usam duas threads; a CV usa dois processos independentes. Não carregar modelos, OOFs, checkpoints ou previsões salvos. Dados somente input/train.csv e input/test.csv.

Escolher uma única família e um peso adicional na grade 0–0,30, passo 0,025, usando apenas TARGET de 0–1. Proporções antigas mantidas; percentis ao quadrado. Não reotimizar os seis pesos individualmente. Em caso de empate, peso menor; sem ganho, parar sem consultar 2–4.

Se passar: completar OOFs da família escolhida, recalcular percentis globais, refixar somente o peso com rótulos 0–1 e gravar frozen_selection.json antes de medir 2–4. Exigir ganho em 2–4 e total e reprodução exata da AUC de referência v20. Não ajustar após confirmação.

Somente se aprovada: notebook final autocontido, treino integral com média do early stopping × 1,15, CSV e duas execuções independentes do zero com SHA-256 idêntico. Tempo em Colab ainda não medido. Nenhum CSV de v21 até aprovação.

As 24 definições originais da v17 são preservadas por hash. Cada candidato é verificado com TARGET da validação invertido: as variáveis supervisionadas devem permanecer idênticas.

Fontes: [CatBoost Ordered](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type), [LightGBM extra_trees](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#extra_trees).
