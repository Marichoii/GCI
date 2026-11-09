# Plano da v20

Referência: v18 reproduzível. Treinar do zero os mesmos cinco componentes, sem carregar OOFs ou previsões anteriores. Seleção de família nas partes 0–1. Candidatos: LightGBM B com max_bin 255/1023; TabM periódico com perda por pares 50%/100%; reequilíbrio dos cinco pesos existentes.

Sementes mantidas; cada modelo usa duas threads. Duas partes podem treinar em processos independentes durante a CV. O notebook final mantém treino integral sequencial e duas threads.

Busca dos pesos: valores não negativos somando 1, percentis ao quadrado. Pesos antigos dentro de ±0,10 dos pesos v18, novo componente até 0,30. Inicialização em grade, transferências determinísticas de massa 0,025 e 0,01, no máximo quatro rodadas por passo. Sem stacking.

Após triagem positiva, completar OOFs da família escolhida. Recalcular percentis globais e fixar os pesos usando somente rótulos 0–1. Gravar frozen_selection.json antes de medir o ensemble em 2–4. Aceitar somente se AUC aumentar em 2–4 e no total. Não retunar depois.

Se aprovada: treino integral com média do early stopping × 1,15; duas execuções novas e mesmo SHA-256. Não apresentar subida pública como comprovada sem submissão.
