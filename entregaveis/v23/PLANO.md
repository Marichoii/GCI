# v23 — validação de ensemble em log-odds

Todos os sete componentes da v21 treinados novamente usando apenas input/train.csv e input/test.csv. Sem carregar previsões, OOFs, modelos ou checkpoints. Pesos e sementes da v21 fixos; duas threads por modelo. Comparar percentis ao quadrado ao rank da média ponderada dos log-odds (clip 1e-6). Selecionar apenas alpha=0,0.1,...,1 nas partes0–1, preservando pesos dos componentes. Se não houver ganho0–1, parar. Caso contrário, completar OOF, refixar alpha apenas0–1 com ranks globais e congelar antes de medir2–4 e total. Somente ganhos em ambos permitem treino integral. Este notebook de pesquisa não escreve CSV. As funções originais da v17, inclusive pré-processamento não supervisionado train+test, são preservadas. Os log-odds dos modelos não são necessariamente calibrados.

Fonte da hipótese: resumo do Slack fornecido pelo usuário, https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965992208839 . Não há garantia de reproduzir o ganho informado por outro participante.

Plano fixado em 30/10/2026, antes dos resultados. Os folds já foram consultados em pesquisas anteriores; esta confirmação não é um holdout intocado.
