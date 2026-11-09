"""Build a self-contained research notebook; never alter frozen v18 files."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
destination = ROOT / "v19"
destination.mkdir(exist_ok=True)
original = json.loads((ROOT / "gci_home_credit_v18.ipynb").read_text())
cells = [dict(cell_type="markdown", metadata={}, source=(
    "# Home Credit — v19: pesquisa com folhas lineares\n\n"
    "Referência: v18 reproduzível (score público informado 0,77865, rank 44/2.695 em 25/10/2026). "
    "Esta pesquisa treina novamente os cinco componentes da v18 e duas configurações do LightGBM linear. "
    "A triagem usa as partes 0–1; se nenhuma acrescentar AUC ao ensemble, interrompe sem avaliar 2–4. "
    "Se passar, completa as mesmas cinco partes, fixa pesos usando exclusivamente rótulos de 0–1 e confere 2–4 e total. "
    "Só os dois CSVs de input são usados como dados. Não carrega previsões, OOFs ou modelos salvos. "
    "Este notebook de pesquisa não escreve submissões; uma candidata aprovada precisa do treino integral e de duas reproduções.\n\n"
    "Fontes: [LightGBM 4.6: linear_tree e linear_lambda](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree). "
    "As folhas ajustam relações lineares, em vez de um valor constante. O eventual ganho é uma hipótese a testar.\n"))]
for cell in original["cells"][1:23]:
    copied = dict(cell)
    if copied["cell_type"] == "code":
        copied.update(execution_count=None, outputs=[])
    cells.append(copied)
cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=(ROOT / "v19_linear.py").read_text()))
cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source='''
experiment_started = time.time()
train = pd.read_csv(BASE_DIR / "input/train.csv")
test = pd.read_csv(BASE_DIR / "input/test.csv")
assert train.shape == (171202, 34) and test.shape == (61500, 33)
input_hashes = {name: hashlib.sha256((BASE_DIR / "input" / name).read_bytes()).hexdigest()
                for name in ["train.csv", "test.csv"]}
ctx = prepare(train, test)
add_tabm_feature_set(ctx)
report_directory = BASE_DIR / "v19" / "evidence"
report = run_v19_experiment(ctx, report_directory)
report["total_seconds_including_prepare"] = time.time() - experiment_started
execution_report = dict(report, csv_sha256=None, run_cv=True, cv_only=True)
(report_directory / "experiment_report.json").write_text(json.dumps(execution_report, indent=2, ensure_ascii=False) + "\\n")
print("Status final:", report["status"], "; nenhum CSV novo escrito.")
'''))
for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
notebook = dict(original, cells=cells)
(destination / "gci_home_credit_v19_pesquisa.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination / "gci_home_credit_v19_pesquisa.ipynb")
