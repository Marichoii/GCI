# Como rodar a v21 no Google Colab CPU

Preparado em **10/11/2026**. A pesquisa está encerrada. Este é um ajuste de
ambiente e operação da v21, sem nova versão de modelagem. Use
`gci_home_credit_v21_colab.ipynb`.

## 1. Abrir um ambiente CPU

1. Acesse [Google Colab](https://colab.research.google.com/) e abra o notebook
   por **Arquivo > Fazer upload de notebook**. Salve uma cópia no seu Drive.
2. Em **Ambiente de execução > Alterar tipo de ambiente de execução**, escolha
   **Python 3** e **Nenhum** como acelerador. Não é necessário GPU ou plano pago.
3. **Execute somente a primeira célula de código** neste primeiro momento.
   Ela mostra a versão real do Python e a pasta de trabalho.

O Colab passou para **Python 3.13**. A página oficial lista o runtime 2026.10
com Python 3.13.18 e o 2026.07 com Python 3.12.13. A versão exata da sua sessão
é a impressa pela primeira célula; não foi aberta uma sessão Colab nesta revisão.
Fontes: [anúncio oficial](https://github.com/googlecolab/colabtools/issues/6081)
e [versões de runtime](https://research.google.com/colaboratory/runtime-version-faq.html).

## 2. Instalar e reiniciar quando solicitado

A primeira célula mantém estes pins:

| Pacote | Versão | Wheel Linux x86-64 / Python 3.13 e 3.12 |
|---|---|---|
| numpy | 2.3.5 | Sim |
| pandas | 2.2.3 | Sim |
| scipy | 1.17.0 | Sim |
| scikit-learn | 1.8.0 | Sim |
| lightgbm | 4.6.0 | Sim |
| catboost | 1.2.8 | Sim |
| torch | 2.6.0 | Sim, build CPU `2.6.0+cpu` |
| tabm | 0.0.3 | Sim, wheel Python puro |
| rtdl-num-embeddings | 0.0.12 | Sim, wheel Python puro |
| threadpoolctl | 3.6.0 | Sim, wheel Python puro |
| joblib | 1.6.0 | Sim, wheel Python puro |

Foram consultados os metadados oficiais do PyPI e o
[índice CPU do PyTorch](https://download.pytorch.org/whl/cpu/torch/).
Também foi resolvida a árvore de dependências para as duas versões de Python,
usando somente wheels, sem instalar. Os arquivos `resolved_linux_cp*.txt` são
evidências dessa resolução, **não novos requirements para o treinamento**.

Se a célula instalar qualquer pacote, ela para propositalmente com:

> reinicie o ambiente e rode tudo de novo

Nesse caso:

1. Use **Ambiente de execução > Reiniciar sessão** (*Runtime > Restart session*;
   o nome pode aparecer como reiniciar ambiente).
2. Confirme o reinício. Não basta executar a primeira célula outra vez.
3. Depois do reinício, execute a primeira célula novamente. Ela deve imprimir
   **Ambiente conferido** e continuar sem instalar.
4. Se a instalação falhar, não continue com o ambiente parcialmente instalado.
   Reinicie e repita. Não troque pins nem use `pip install --upgrade` livremente.

O reinício é obrigatório nesta cópia após qualquer instalação, inclusive de
NumPy, pandas, SciPy ou PyTorch: o Python pode conservar módulos e bibliotecas
nativas antigos em memória. O pip é executado em subprocesso pelo mesmo
interpretador, conforme a [documentação do pip](https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program).

Python 3.14 não é aceito: faltam wheels dos pins pandas, CatBoost e PyTorch.
Se aparecer uma versão não coberta ou erro de wheel, selecione **Runtime
Version 2026.07**, com Python 3.12, em Alterar tipo de ambiente. Reconecte e
repita a primeira célula; não tente mudar o Python do kernel com pip.

## 3. Fornecer os dois arquivos oficiais

Na célula **Entrada opcional**, escolha `INPUT_MODE`:

- **upload**: ao executar, selecione juntos `train.csv` e `test.csv`, com esses
  nomes exatos. A transferência é isolada, validada e depois copiada para input/.
- **drive**: coloque os arquivos, por exemplo, em `Meu Drive/GCI/input/` e ajuste
  `DRIVE_INPUT_DIR` para `/content/drive/MyDrive/GCI/input`. Autorize a montagem
  quando o próprio Colab solicitar. Os dados são conferidos no Drive e copiados
  para o disco local da VM antes de treinar.
- **existente**: use se já colocou os dois arquivos na pasta input/ da pasta de
  trabalho mostrada na primeira célula.

Se ambos já estão em input/, a célula só confere os hashes. Se estiverem
incorretos, ela para: corrija/remova os arquivos errados e forneça os oficiais.
O notebook nunca substitui os dados por outro recorte ou por dados do Kaggle.

Hashes copiados de `competition/input/README.md`:

```text
train.csv  b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1
test.csv   dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65
```

## 4. Executar o treino integral

Com o ambiente conferido, ajuste a opção de entrada e use **Executar tudo**.
Se havia instalação, esse comando deve ser usado **depois** do reinício.

Não ative `GCI_RUN_CV` ou `GCI_CV_ONLY`: esta cópia bloqueia a pesquisa e usa
somente `FROZEN_V21`. Todos os componentes são treinados do zero a partir dos
dois CSVs. Não há retomada por modelos, previsões, OOFs ou checkpoints salvos.

A célula de treino é idêntica à validada. Os sete pesos continuam
`[0.156571875, 0.005900625, 0.21094875, 0.17445374999999996,
0.194625, 0.15750000000000003, 0.1]`. Os tamanhos continuam
`[1646, 1793, 11, 1103, 14, 11, 1089]`, com as mesmas seeds e duas threads
por modelo. O pré-processamento não supervisionado train+test herdado também
permanece igual.

O relatório final mostra ambiente, entrada, preparo/auxiliares, cada um dos sete
componentes, agregação/gravação e verificação. Os tempos por componente vêm do
registro já existente da v21; eventos de célula do IPython medem a célula de
treino sem reescrevê-la. Fora de IPython há fallback entre células, identificado
no relatório. O total inclui entrada e possíveis esperas entre células, a
partir da primeira célula na sessão após o reinício.

Se o total passar de **90 minutos**, aparece um aviso ao concluir a célula de
treino e na verificação final. O aviso não reduz árvores, épocas ou seeds e não
cancela modelos. O tempo da instalação anterior ao reinício é registrado
separadamente quando o arquivo persiste. Não há garantia de tempo no Colab CPU
gratuito: os recursos e limites de uso variam, conforme a
[FAQ oficial](https://research.google.com/colaboratory/faq.html).

## 5. Conferir e guardar os resultados

A célula final reabre `output/submission.csv` e exige:

- 61.500 linhas, duas colunas exatamente `SK_ID_CURR,TARGET`;
- IDs na mesma ordem de `input/test.csv`;
- ausência de NaN e valores numéricos finitos no intervalo [0,1];
- SHA-256 igual ao calculado imediatamente depois da gravação.

O SHA-256 aparece na tela e em `output/SHA256.txt`. A célula também informa
se os bytes coincidem com o histórico macOS:
`48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756`.
Coincidência entre Colab/Linux e macOS não foi testada. Se diferir, preserve
ambos e registre o novo hash; não ajuste previsões para forçar identidade.

A última célula oferece um download de `resultados_v21_colab.zip`, contendo
o CSV e os relatórios. Se o navegador não iniciar o download, use o painel
**Arquivos** do Colab e abra a pasta de trabalho mostrada na primeira célula.
Também é possível habilitar `COPIAR_PARA_DRIVE` e escolher `DRIVE_OUTPUT_DIR`;
será criada uma subpasta nova com data/hora UTC, preservando execuções anteriores.
Baixe os arquivos assim que a validação terminar. Não há envio automático à
competição: o ranking final considera o último CSV que você enviar.

## Se o Colab desconectar

1. **Só a conexão do navegador caiu, mas a VM continua:** reconecte à mesma
   sessão. Verifique se o treino continua ou se terminou. Não execute de novo
   a célula de treino enquanto ela está ativa. Se terminou e as variáveis ainda
   existem, execute apenas as células finais de verificação e download.
2. **A sessão foi reiniciada ou a VM foi perdida:** memória e arquivos locais
   podem ter sido apagados. Repita ambiente, reinício quando solicitado, entrada
   dos dados e treino completo. O notebook não retoma etapas por resultados
   parciais. O arquivo salvo do notebook não conserva a memória do treino.
3. **Já havia um pacote de resultados guardado no Drive ou baixado:** ele
   preserva o CSV e o relatório daquela execução concluída. Para uma nova
   reprodução, faça outro treino integral em uma pasta limpa. Não carregue esse
   CSV ou previsões como entrada para o modelo.

A guarda antes do treino para se `output/submission.csv` já existe. Para
reproduzir outra vez, use uma VM nova ou defina `GCI_WORKDIR` para outra pasta
antes da primeira célula. Não apague um resultado validado sem antes guardá-lo.
Não são usados mecanismos para contornar desconexões ou limites do Colab.

## O que foi testado nesta entrega

Veja `evidence/test_report.json` e `evidence/model_preservation.json`.
Foram conferidos wheels e resolução de dependências, código e invariância das
células originais, guardas de instalação/reinício, hashes de entrada e validação
do CSV histórico, com casos negativos. Os testes de upload/Drive usam dublês
locais; não são transferências autenticadas no Colab.

**Não foram executados:** treino novo, instalação real dos pins no Colab,
montagem real de Drive/upload no Colab, reprodução completa em Linux/Python
3.13, benchmark de 90 minutos ou teste real de desconexão. O diff e a auditoria
comprovam preservação de código/receita; não comprovam identidade numérica entre
plataformas sem executar o treinamento.
