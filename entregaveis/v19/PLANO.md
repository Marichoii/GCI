# Pesquisa v19 — plano fixado antes dos resultados

Comparar o ensemble v18 com a adição de um LightGBM com folhas lineares. Não alterar os cinco componentes, sementes ou proporções de pesos anteriores.

- Configurações candidatas: linear_lambda 10 e 100; demais parâmetros idênticos entre si.
- Peso novo: 0 a 0,30, passo 0,025; empate escolhe o menor peso.
- Triagem: partes 0–1 dos mesmos cinco folds. Percentis calculados nas linhas disponíveis. Se as duas configurações escolherem peso zero, encerrar sem avaliar 2–4.
- Se passar: completar OOFs; escolher configuração e peso com os rótulos de 0–1 usando percentis globais. Salvar a receita antes de calcular a AUC do ensemble nas partes 2–4.
- Aprovação: AUC maior nas partes 2–4 e no total. Não retunar depois da confirmação.
- Candidata aprovada: treino integral; tamanho médio do early stopping × 1,15; executar duas vezes em processos novos e conferir SHA-256 idêntico antes de apresentar o CSV.
- Dados: só input/train.csv e input/test.csv. Estatísticas supervisionadas sem rótulos de validação; sem ID como feature. Sementes fixas, duas threads, sem carregar previsões ou modelos salvos.

Fonte primária: [LightGBM 4.6](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree).
