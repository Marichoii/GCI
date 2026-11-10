# Dados oficiais da competição

Obtenha `train.csv` e `test.csv` pelo material oficial do GCI e coloque os arquivos
nesta pasta para trabalhar em `competition/`. Os dados brutos não são versionados
no GitHub. Use o recorte fornecido pelo curso; ele não corresponde necessariamente
ao conjunto completo da competição original do Kaggle.

| Arquivo | SHA-256 registrado nas execuções |
| --- | --- |
| `train.csv` | `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1` |
| `test.csv` | `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65` |

O treino contém a coluna `TARGET`. A submissão usa as colunas `SK_ID_CURR,TARGET`,
com as probabilidades na ordem dos IDs de `test.csv`.

Ao reproduzir um pacote de `entregaveis/vNN/`, crie uma pasta de trabalho nova
e coloque `input/` ao lado do notebook, como indicado no relatório da versão.
