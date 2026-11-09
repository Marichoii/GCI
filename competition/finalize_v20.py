"""Verify two fresh complete-training executions, never fabricate evidence."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def verify():
    validation=json.loads((ROOT/"v20/evidence/experiment_report.json").read_text())
    assert validation["confirmed"] and validation["status"]=="gain_confirmed"
    by_name={r["parte"]:r for r in validation["table"]}
    assert by_name["partes_2_4"]["delta"]>0 and by_name["total"]["delta"]>0
    sources=[ROOT/f"v20_final_{i}/runs/reproduction_{i}" for i in [1,2]]
    reports=[json.loads((p/"execution_report.json").read_text()) for p in sources]
    assert all(not r["run_cv"] and not r["cv_only"] and r["csv_sha256"] for r in reports)
    assert all(r["comparison"]["improved"] and r["threads_per_model"]==2 and r["torch_interop_pool"]==2 for r in reports)
    assert all(pool["num_threads"]<=2 for r in reports for pool in r["threadpools"])
    for key in ["input_sha256","versions","exact_installed_versions","final_sizes","selected_candidate"]:
        assert reports[0][key]==reports[1][key],key
    assert reports[0]["comparison"]["weights"]==reports[1]["comparison"]["weights"]==validation["weights"]
    assert reports[0]["input_sha256"]==validation["input_sha256"]
    notebook=(ROOT/"gci_home_credit_v20.ipynb").read_bytes()
    assert all((p/"source.ipynb").read_bytes()==notebook for p in sources)
    manifest=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
    found={}
    for cell in json.loads(notebook)["cells"]:
        if cell["cell_type"]!="code":continue
        source="".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                assert node.name not in found
                found[node.name]=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
    assert all(found[k]==v for k,v in manifest.items())
    csvs=[(p/"submission.csv").read_bytes() for p in sources]
    assert csvs[0]==csvs[1],"Different CSV bytes: do not claim reproduction."
    hashes=[hashlib.sha256(p).hexdigest() for p in csvs]
    assert hashes==[r["csv_sha256"] for r in reports]
    evidence=dict(identical_csv_bytes=True,csv_sha256_run_1=hashes[0],csv_sha256_run_2=hashes[1],
                  notebook_sha256=hashlib.sha256(notebook).hexdigest(),original_definitions_preserved=24,
                  seconds=[r["total_seconds"] for r in reports],
                  full_train_seconds=[r["full_train_seconds"] for r in reports],
                  versions=reports[0]["exact_installed_versions"],input_sha256=validation["input_sha256"],
                  platform=reports[0]["platform"],python=reports[0]["python"],
                  method="Same notebook, all code cells executed by execute_notebook.py in two new Python processes; no saved predictions or models loaded.",
                  limitation="Local macOS ARM64 CPU; Colab was not executed.")
    (ROOT/"v20/evidence/reproducibility.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=="__main__":verify()
