"""Package only the self-contained submission code and measured evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
validation = json.loads((ROOT / "runs/validation/execution_report.json").read_text())
repro = json.loads((ROOT / "reproducibility.json").read_text())
assert validation["comparison"]["improved"] and repro["identical_csv_bytes"]
assert hashlib.sha256((ROOT / "output/submission.csv").read_bytes()).hexdigest() == repro["csv_sha256_run_1"]
comparison = validation["comparison"]
table = ["| Parte | v17 recalculada | v18 | Delta |", "|---|---:|---:|---:|"]
for r in comparison["table"]:
    table.append(f'| {r["parte"]} | {r["v17"]:.8f} | {r["v18"]:.8f} | {r["delta"]:+.8f} |')
labels = ["LightGBM B + EXT auxiliares", "LightGBM B (v7)", "MLP (8 sementes)",
          "CatBoost + EXT auxiliares", "TabM periódico"]
weight_table = ["| Componente | Peso | Árvores/épocas finais |", "|---|---:|---:|"]
for name, weight, size in zip(labels, comparison["weights"], validation["final_sizes"]):
    weight_table.append(f"| {name} | {weight:.6%} | {size} |")
versions = "\n".join(f"- {k}: {v}" for k, v in repro["versions"].items())
table_text = "\n".join(table)
weight_table_text = "\n".join(weight_table)
report = f'''# GCI Home Credit v18 — resultados executados

A v18 passou o critério solicitado: AUC maior nas partes 2–4 e no total.
O peso novo foi escolhido somente nas partes 0–1, numa grade de 0 a 50%, passo
de 2,5 pontos percentuais. As proporções dos quatro componentes da v17 foram
preservadas. Não houve ajuste após consultar as partes 2–4.

{table_text}

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

{weight_table_text}

Os tamanhos finais foram recalculados: round(média de todas as melhores
iterações/épocas da CV × 1,15). O treino final usa 100% das linhas.
O TabM recebe base da v17 + 13 auxiliares EXT e as 10 colunas de Supervised.add.
Configuração: arch_type=tabm, k=8, n_blocks=2, d_block=128, dropout=0,15;
PeriodicEmbeddings(d_embedding=8, frequency_init_scale=0,1, lite=True);
AdamW(lr=0,002, weight_decay=0,01), batch 1024, até 40 épocas, patience 7.
Na CV, ReduceLROnPlateau(factor=0,5, patience=2); treino final sem validação,
LR constante e número fixo de épocas. Sementes da CV: 42+100×fold; final: 42.
Os outros componentes conservam as sementes e hiperparâmetros da v17.

SHA-256 do CSV, execução 1: `{repro["csv_sha256_run_1"]}`

SHA-256 do CSV, execução 2: `{repro["csv_sha256_run_2"]}`

Os CSVs foram comparados byte a byte. Todas as células de código do mesmo
notebook foram executadas em ordem em dois processos Python novos pelo runner
execute_notebook.py. Não foi usado servidor/kernel Jupyter nessas execuções.
As duas execuções reconstruíram os auxiliares e todos os modelos do zero;
nenhuma previsão, OOF, checkpoint ou modelo salvo foi carregado.

Tempo da validação completa: {validation["total_seconds"]/60:.2f} minutos.
Tempo das reproduções completas (incluindo auxiliares):
{repro["seconds"][0]/60:.2f} e {repro["seconds"][1]/60:.2f} minutos.
Ambiente medido: {validation["platform"]}; Python {validation["python"].split()[0]}, CPU,
2 threads por biblioteca/modelo e pool inter-op do PyTorch fixo em 2.
Colab gratuito não foi executado ou cronometrado. A reprodução padrão pula a
CV e treina todos os modelos a partir de input/ com a receita já validada.

Versões efetivamente instaladas:

{versions}

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

- train.csv SHA-256: `{repro["input_sha256"]["train.csv"]}`
- test.csv SHA-256: `{repro["input_sha256"]["test.csv"]}`
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
'''
(ROOT / "RESULTADOS_v18.md").write_text(report)
archive_path = ROOT / "gci_home_credit_v18.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ["gci_home_credit_v18.ipynb", "execute_notebook.py", "requirements.txt",
                 "installed_versions_local.txt", "RESULTADOS_v18.md", "output/submission.csv"]:
        z.write(ROOT / name, name)
    for name in ["reproducibility.json", "target_isolation_audit.json", "v17_definitions_sha256.json"]:
        z.write(ROOT / name, "evidence/" + name)
    for run in ["validation", "reproduction_1", "reproduction_2"]:
        for name in ["execution_report.json", "executed.ipynb"]:
            z.write(ROOT / "runs" / run / name, f"evidence/{run}/{name}")
print("Pacote criado:", archive_path, archive_path.stat().st_size)
from organizar_entregaveis import organize
print("Entregáveis organizados em:", organize("v18"))
