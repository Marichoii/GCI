# v22 — plano anterior aos resultados

Comparar LightGBM B seed42, mesmas features plusx e Supervised da v21, com e sem sete features de densidade. Apenas partes0–1 do StratifiedKFold(5,shuffle=True,random_state=42), ordem original. Três ajustes por parte: baseline, densidade global, densidade por contrato.

Log(crédito/prestação), histograma2048bins nos quantis0,001–0,999 do treino da parte, suavização gaussiana com bandwidth Silverman robusto. Detectar picos com prominência1% e distância12bins; mínimos entre picos separam segmentos. Gerar lograzão, segmento, posição no segmento, desvio do pico, posição do pico, logdensidade e indicador de ausente. Os parâmetros são fixados sem TARGET antes do resultado. Não aceitar TARGET ou SK_ID_CURR no fit da densidade.

O estimador aprende só nas linhas de treino da parte. As funções v17 permanecem intactas, inclusive o pré-processamento antigo que junta covariáveis train+test; esta triagem NÃO comprova um pipeline inteiramente indutivo. Todo TARGET usado em Supervised exclui a parte avaliada.

Só uma configuração com AUC conjunta0–1 maior que baseline e média das diferenças por parte positiva pode avançar. Ainda será preciso comparar ao ensemble, escolher peso apenas0–1 e conferir2–4 e total. Não gerar CSV nesta triagem. Não concluir que o ensemble piora apenas por reprovar a ablação de um componente.

Ideia relatada pelo usuário a partir do Slack, atribuída a Theophile Avenel e sameenakb: https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965969389949 . Ganhos alheios não são medidos nesta base. Histórico disponível de pricing descreve15chaves de target encoding, não KDE; o código antigo completo não foi recuperado para comparar tudo.
