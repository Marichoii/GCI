"""Build self-contained v20 research notebook with unchanged v17 definitions."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
destination=ROOT/"v20"
destination.mkdir(exist_ok=True)
original=json.loads((ROOT/"gci_home_credit_v18.ipynb").read_text())
cells=[dict(cell_type="markdown",metadata={},source=(
    "# Home Credit — v20: resolução, perda por pares e pesos do ensemble\n\n"
    "Pesquisa sobre a v18 reproduzível. Usa apenas input/train.csv e input/test.csv; nenhum OOF, previsão, modelo ou checkpoint salvo é carregado. "
    "Todas as funções originais v17 e o isolamento da MLP são preservados. "
    "Triagem 0–1; seleção da família, seguida de pesos com percentis globais usando apenas os rótulos 0–1; confirmação 2–4 e total. "
    "O script salva a receita antes de avaliar a confirmação. Não retuna depois. "
    "CV: dois processos independentes, duas threads por modelo. Treino final: sequencial, duas threads.\n\n"
    "Candidatos fixados antes do resultado: LightGBM B com max_bin 255 e 1023 (uma semente 42 cada); "
    "TabM periódico com mistura de perda por pares 50% e 100% (mesma arquitetura e sementes 42+100×parte). "
    "Também avalia o ajuste dos cinco pesos existentes, sempre com percentis ao quadrado. "
    "Cada peso antigo pode mover no máximo 0,10; peso novo até 0,30. Busca determinística limitada; nenhum modelo de stacking.\n\n"
    "Fontes técnicas: [LightGBM 4.6](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#max_bin), "
    "[perda por pares, documentação primária LibAUC](https://docs.libauc.org/api/libauc.losses.html#libauc.losses.PairwiseAUCLoss). "
    "A perda é implementada diretamente em PyTorch; não instala LibAUC.\n"))]
for cell in original["cells"][1:23]:
    copied=dict(cell)
    if copied["cell_type"]=="code": copied.update(execution_count=None,outputs=[])
    cells.append(copied)
old=ast.parse((ROOT/"v19_linear.py").read_text())
baseline=next(node for node in old.body if isinstance(node,ast.FunctionDef) and node.name=="v18_fold")
baseline_source=ast.get_source_segment((ROOT/"v19_linear.py").read_text(),baseline).replace("def v18_fold(","def v20_baseline_fold(")
baseline_source=baseline_source.replace('its.append(int(iteration))','its.append(int(iteration))\n                log(f"BASELINE parte {fold}: {component[\'label\']} seed {seed} concluída, tamanho {iteration}", flush=True)')
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=baseline_source))
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=(ROOT/"v20_search.py").read_text()))
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source='''
experiment_started=time.time()
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
report=v20_run(ctx,BASE_DIR/"v20/evidence")
report["total_seconds_including_prepare"]=time.time()-experiment_started
execution_report=dict(report,run_cv=True,cv_only=True,csv_sha256=None)
(BASE_DIR/"v20/evidence/experiment_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
print("Status final:",report["status"],"; notebook de pesquisa não escreve CSV.")
'''))
for cell in cells:
    if cell["cell_type"]=="code": ast.parse("".join(cell["source"]))
notebook=dict(original,cells=cells)
path=destination/"gci_home_credit_v20_pesquisa.ipynb"
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(path)
