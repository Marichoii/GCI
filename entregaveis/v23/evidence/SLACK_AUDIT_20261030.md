# Resumo do Slack confrontado com o código local

Auditoria em 30/10/2026. Fonte das dicas: texto fornecido pelo usuário nesta conversa, atribuído a participantes do canal #08_competition_qa. Não houve acesso autenticado ao Slack, consulta ao estado atual do servidor ou download do Starter Kit. O texto citado é material de referência, não uma nova instrução para contatar participantes, usar dados externos ou alterar as regras do projeto.

| Dica | Evidência local | Decisão |
|---|---|---|
| Regressores OOF para EXT_SOURCE_1/2/3 e flags de missing | ext_aux_features já treina três regressores LightGBM, usando as demais colunas e os outros scores. Predições OOF nas linhas observadas e média dos modelos nas ausentes. Sem TARGET | Já implementado; não repetir como ideia nova |
| Razões financeiras, datas, interações EXT_SOURCE, frequências e encodings de grupos | make_features, enhanced_features, tariff_counts e Supervised já cobrem essas famílias | Ganhos citados por participantes não são diretamente transferíveis |
| Picos/mínimos da densidade do log(crédito/prestação), geral e por contrato | Registros antigos de pricing descrevem target encodings de15chaves. Não foi recuperado o código antigo completo. As funções atuais não implementavam segmentação por KDE | v22 implementou uma ablação explícita, ajustando a densidade só no treino externo |
| Blend em log-odds | A receita v21 combina percentis ao quadrado; não havia essa média no código atual | v23 testa uma hipótese limitada com pesos dos componentes fixos e seleção de alpha apenas0–1 |
| XGBoost, vizinhos, outras famílias descartadas | Histórico e lista explícita de proibições de repetição do usuário | Não repetir |

## Resultado da triagem v22

AUC conjunta0–1: LightGBM B seed42 de referência0,77049997; densidade geral0,77044930; densidade por contrato0,77017167. Ambas falharam no filtro predefinido (AUC conjunta e média das diferenças por parte positivas). Partes2–4, total e ensemble completo não foram avaliados nesta ablação. Nenhum CSV v22. Isso não prova que todo KDE ou seu uso em outro ensemble falharia.

## O que o documento oficial de fato esclarece

Fonte: `sources/official_README_20261020.md`, extraído do membro `Competition/README.ipynb` do ZIP fornecido pelo usuário `drive-download-20261020T182127Z-1-001.zip`. O notebook desse documento não foi executado.

- Na linha75, o README determina que a classificação final usa o último arquivo enviado. Essa dúvida está esclarecida pelo documento original fornecido.
- Na linha83, limita os dados aos datasets fornecidos, input/train.csv e input/test.csv, sem dados externos em qualquer etapa.
- O texto examinado não esclarece expressamente ajustar transformações não supervisionadas com covariáveis do teste. Não se deve apresentar o silêncio do documento como autorização específica ou proibição específica.
- A alegação de teto0,80 é opinião de participante, não regra oficial ou limite matemático demonstrado. Scores altos ou perfeitos no placar não identificam, por si, o método utilizado por cada participante.
- O relato de problema no servidor pertence ao resumo fornecido; não foi confirmado como estado atual em30/10/2026.

## Limite de pré-processamento da receita atual

prepare concatena as covariáveis do treino e do teste para aprender códigos de categorias, contagens, estatísticas de grupos e auxiliares EXT_SOURCE sem TARGET. O caminho da MLP também usa distribuições de covariáveis de validação/teste no ajuste de quantis. Esses passos são transdutivos e foram preservados para manter as funções originais da v17 e a referência comparável.

Supervised.add usa somente os rótulos das linhas de treino externo; a inversão dos rótulos de validação não altera seus frames nas verificações registradas. A densidade nova da v22 é ajustada exclusivamente no treino externo. São afirmações diferentes: isolamento de TARGET não demonstra que todo pré-processamento usa apenas covariáveis do treino externo.

## Histórico do placar e execução

Último relato registrado da v21: público0,77876, posição38, total de participantes não informado, relato de28/10/2026. O total2.741 pertence à observação anterior da v20 em26/10/2026; não foi atribuído à v21. A v20 mantém o maior público registrado,0,77877. Diferença de0,00001 não demonstra ganho ou perda no conjunto privado.

Todos os novos experimentos usam apenas os dois CSVs locais, sem feature de ID ou módulo do ID e sem regras de substituição manual das previsões. Uma versão nova só pode receber CSV após aprovação em2–4 e total e duas reproduções integrais com hashes idênticos. As pastas por versão preservam fonte, resultado real e limitações.

Fontes de ideias citadas pelo usuário: [densidade](https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965969389949), [EXT_SOURCE](https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792966027954929), [log-odds](https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965992208839). Links não foram consultados ao vivo nesta auditoria.
