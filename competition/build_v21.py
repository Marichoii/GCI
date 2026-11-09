"""Assemble a self-contained research notebook without editing frozen versions."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
dest=ROOT/"v21"
dest.mkdir(exist_ok=True)
original=json.loads((ROOT/"gci_home_credit_v20.ipynb").read_text())
cells=[dict(cell_type="markdown",metadata={},source=(
    "# Home Credit — v21: Ordered CatBoost e árvores aleatórias\n\n"
    "Pesquisa predefinida em 28/10/2026 sobre a v20 (público 0,77877; rank histórico 38/2741 em 26/10). "
    "Candidatos: CatBoost Ordered depth 5 e 6 (lr 0,04, L2 5, seed 42), LightGBM B extra_trees=True (seed 42). "
    "Todos usam as variáveis originais, Supervised fora da parte e duas threads por modelo. "
    "Os seis componentes da referência são treinados novamente nesta execução, sem carregar modelos ou previsões. "
    "Seleção de uma família e peso novo na grade 0–0,30, passo 0,025, somente rótulos 0–1; "
    "proporções da v20 preservadas nos demais pesos. Se passar, completar OOFs, refixar o peso usando apenas 0–1 "
    "com percentis globais e salvar a receita antes de consultar a confirmação 2–4. "
    "Ganho exige melhora em 2–4 e no total. Não ajustar após a confirmação. Nenhum CSV neste notebook de pesquisa.\n\n"
    "Fontes: [CatBoost Ordered](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type), "
    "[LightGBM extra_trees](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#extra_trees).\n"))]
for c in original["cells"][1:25]:
    copied=dict(c)
    if copied["cell_type"]=="code":copied.update(execution_count=None,outputs=[])
    cells.append(copied)
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=(ROOT/"v21_search.py").read_text()))
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source='''
experiment_started=time.time()
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
assert all("SK_ID_CURR" not in frame and "TARGET" not in frame for sets in [ctx["X"],ctx["Xt"]] for frame in sets.values())
report=v21_run(ctx,BASE_DIR/"v21/evidence")
report["total_seconds_including_prepare"]=time.time()-experiment_started
execution_report=dict(report,run_cv=True,cv_only=True,csv_sha256=None)
(BASE_DIR/"v21/evidence/experiment_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
print("Status final:",report["status"],"; notebook de pesquisa não escreve CSV.")
'''))
cells.append(original["cells"][-1])
found={}
for c in cells:
    if c["cell_type"]!="code":continue
    source="".join(c["source"])
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            assert node.name not in found
            found[node.name]=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
manifest=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[name]==digest for name,digest in manifest.items())
nb=dict(original,cells=cells)
path=dest/"gci_home_credit_v21_pesquisa.ipynb"
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
print(path,"; 24 definições v17 preservadas")
