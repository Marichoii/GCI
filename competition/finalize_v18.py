"""Freeze a validated recipe or verify two genuinely independent executions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_report(name):
    return json.loads((ROOT / "runs" / name / "execution_report.json").read_text())


def freeze():
    report = read_report("validation")
    comparison = report["comparison"]
    assert report["run_cv"] and report["cv_only"]
    assert report["csv_sha256"] is None
    assert comparison["improved"], "Sem ganho confirmado: não congelar receita nem gerar CSV."
    recipe = dict(weights=comparison["weights"], sizes=report["final_sizes"],
                  improved=True, metrics=comparison["table"], alpha=comparison["alpha"],
                  input_sha256=report["input_sha256"])
    p = ROOT / "gci_home_credit_v18.ipynb"
    assert p.read_bytes() == (ROOT / "runs/validation/source.ipynb").read_bytes()
    nb = json.loads(p.read_text())
    s = "".join(nb["cells"][24]["source"])
    assert 'FROZEN_RECIPE = None' in s
    s = s.replace('os.environ.get("GCI_RUN_CV", "1")', 'os.environ.get("GCI_RUN_CV", "0")')
    s = s.replace("FROZEN_RECIPE = None", "FROZEN_RECIPE = " + repr(recipe))
    nb["cells"][24]["source"] = s
    nb["cells"][0]["source"] = "".join(nb["cells"][0]["source"]) + (
        "\nA receita abaixo foi aprovada na validação executada nesta revisão. "
        "Por padrão, refaz apenas o treino completo; para recalcular a validação, "
        "defina `GCI_RUN_CV=1` antes de iniciar ou mude `RUN_CV` para `True`. "
        "OOFs e modelos de execuções anteriores nunca são carregados.\n")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (ROOT / "requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in report["versions"].items()))
    print("Receita congelada:", recipe["weights"], recipe["sizes"])


def verify():
    reports = [read_report(name) for name in ["reproduction_1", "reproduction_2"]]
    assert all(not r["run_cv"] and not r["cv_only"] for r in reports)
    assert all(r["csv_sha256"] and r["comparison"]["improved"] for r in reports)
    assert all(r["threads_per_model"] == 2 and r["torch_interop_pool"] == 2 for r in reports)
    assert all(p["num_threads"] <= 2 for r in reports for p in r["threadpools"])
    for key in ["input_sha256", "versions", "exact_installed_versions", "final_sizes"]:
        assert reports[0][key] == reports[1][key], key
    assert reports[0]["comparison"]["weights"] == reports[1]["comparison"]["weights"]
    csv_bytes = [(ROOT / "runs" / name / "submission.csv").read_bytes()
                 for name in ["reproduction_1", "reproduction_2"]]
    assert csv_bytes[0] == csv_bytes[1], "CSV bytes differ; do not claim reproducibility."
    hashes = [hashlib.sha256(b).hexdigest() for b in csv_bytes]
    assert hashes == [r["csv_sha256"] for r in reports]
    source = (ROOT / "gci_home_credit_v18.ipynb").read_bytes()
    manifest = json.loads((ROOT / "v17_definitions_sha256.json").read_text())
    definitions = {}
    for cell in json.loads(source)["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        for node in ast.parse(s).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in definitions, "Função/classe redefinida."
                definitions[node.name] = hashlib.sha256(ast.get_source_segment(s, node).encode()).hexdigest()
    assert all(definitions.get(k) == v for k, v in manifest.items())
    for name in ["reproduction_1", "reproduction_2"]:
        assert (ROOT / "runs" / name / "source.ipynb").read_bytes() == source
    result = dict(identical_csv_bytes=True, csv_sha256_run_1=hashes[0], csv_sha256_run_2=hashes[1],
                  notebook_sha256=hashlib.sha256(source).hexdigest(),
                  seconds=[r["total_seconds"] for r in reports],
                  final_train_seconds=[r["full_train_seconds"] for r in reports],
                  execution_method="All code cells of the same notebook executed sequentially "
                                   "via execute_notebook.py in two fresh Python processes.",
                  environment="Local macOS arm64 CPU, not Colab; no Colab timing was measured.",
                  input_sha256=reports[0]["input_sha256"], versions=reports[0]["exact_installed_versions"])
    (ROOT / "reproducibility.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "verify"])
    args = parser.parse_args()
    {"freeze": freeze, "verify": verify}[args.action]()
