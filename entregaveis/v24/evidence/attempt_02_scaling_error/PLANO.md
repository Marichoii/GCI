# v24 — Factorization Machine PyTorch

Protocolo fixado em 09/11/2026 antes dos treinos; pesquisa termina em 14/11/2026.
Usar somente input/train.csv e input/test.csv. Nenhuma biblioteca nova.
Sem carregar previsões, OOFs, checkpoints ou modelos. Sem SK_ID_CURR como
variável, sem transformações de ID e sem regras manuais nas previsões.

FM de segunda ordem: termo linear e fatores de posto 16, perda BCE sem pesos,
AdamW lr=0,002, weight_decay=0,001, batch=2048, seed=42 e duas threads.
Variáveis plusx da v21, categorias one-hot; numéricas contínuas com quantis
normais (256 quantis), clip [-4,4], ausentes=0 com indicadores. Ajustar tudo
somente no treino correspondente. Inicialização dos fatores N(0,0,01), termo
linear zero e intercepto logit da prevalência do treino.

Para cada parte externa, reservar 15% de seu treino (estratificado, seed 42)
para escolher épocas: até 40, patience 6, min_delta 1e-6 por AUC. Recalcular
features supervisionadas dentro desse treino interno. Recomeçar do zero no
treino externo completo com as épocas escolhidas. Épocas finais = média
arredondada das épocas internas, sem multiplicador. Nenhum rótulo da parte
externa escolhe épocas da FM. Os sete componentes v21 e seu early stopping
original são preservados para reproduzir a referência; seu pré-processamento
não supervisionado herdado usa covariáveis train+test.

Cinco partes StratifiedKFold(5,shuffle=True,random_state=42), ordem original.
Estratos exclusivamente diagnósticos: chave round(crédito/prestação,2),
frequência contada apenas no treino externo; 0–9 (raras, inclui inéditas),
10–49, 50–199, >=200; ausentes/inválidas separados (crédito/prestação não
positivos ou não finitos). Congelar atribuição a estratos antes do primeiro
fit. Não usar estratos para features, pesos de treino ou regras de previsão.
Reportar linhas, %, positivos, AUC v21, FM e ensemble candidato por estrato.

Grade fixa de peso FM [0%,5%,10%,15%,20%], preservando proporções v21;
empates preferem menor peso. Percentis ao quadrado, como v21/v18. Selecionar
somente em 0–1. Corte rápido com percentis das OOF disponíveis: se peso zero
ou sem ganho conjunto 0–1, parar sem treinar 2–4 ou gerar CSV. Se passar,
completar OOF e refixar peso na mesma grade usando somente rótulos 0–1 e
percentis globais; congelar antes de calcular AUCs 2–4/total. Sem retuning.
Reportar AUC FM isolada e Spearman com cada um dos sete componentes.

Promoção exige delta 2–4 >=0,00010 e delta total >0 sobre v21 reproduzida
(2–4=0,7747809092432562; total=0,7740567649584406). Estratos não acrescentam
um critério de promoção; servem para testar a justificativa. Se falhar,
nenhum CSV. Se passar, notebook final treina tudo em 100% do treino;
duas execuções novas, mesmo SHA-256 e menos de 60 min por execução completa
em CPU. Tempos de Colab precisam de verificação; macOS não prova identidade
entre plataformas. Esta pesquisa usa dois workers de CV, duas threads cada;
o treino final será sequencial com duas threads.

Este notebook de pesquisa não escreve CSV. Somente após aprovação será
preparado o notebook final com receita congelada. Os folds já foram usados
em pesquisas anteriores e não são um holdout intocado.

Referência do mecanismo: https://libfm.org/ . Não são usados dados externos.
