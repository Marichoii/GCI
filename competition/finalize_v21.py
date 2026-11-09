"""Check the two real clean executions before publishing any v21 CSV."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/"v21/v21/evidence/experiment_report.json").read_text())
assert r["confirmed"] and r["status"]=="gain_confirmed"
by_name={row["parte"]:row for row in r["table"]}
assert all(by_name[k]["delta"]>0 for k in ["partes_2_4","total"])
dirs=[ROOT/f"v21_final_{i}/runs/reproduction_{i}" for i in [1,2]]
reports=[json.loads((d/"execution_report.json").read_text()) for d in dirs]
assert all(not x["run_cv"] and not x["cv_only"] and x["comparison"]["improved"] for x in reports)
assert all(x["threads_per_model"]==2 and x["torch_interop_pool"]==2 for x in reports)
assert all(p["num_threads"]<=2 for x in reports for p in x["threadpools"])
for key in ["input_sha256","versions","exact_installed_versions","final_sizes","selected_candidate"]:
    assert reports[0][key]==reports[1][key],key
assert reports[0]["input_sha256"]==r["input_sha256"]
assert all(x["comparison"]["weights"]==r["weights"] for x in reports)
assert all(x["comparison"]["table"]==r["table"] for x in reports)
source=(ROOT/"gci_home_credit_v21.ipynb").read_bytes()
assert all((d/"source.ipynb").read_bytes()==source for d in dirs)
found={}
for c in json.loads(source)["cells"]:
    if c["cell_type"]!="code":continue
    s="".join(c["source"])
    for node in ast.parse(s).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            assert node.name not in found
            found[node.name]=hashlib.sha256(ast.get_source_segment(s,node).encode()).hexdigest()
manifest=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k]==v for k,v in manifest.items())
csv_bytes=[(d/"submission.csv").read_bytes() for d in dirs]
assert csv_bytes[0]==csv_bytes[1],"Different CSV bytes; do not claim reproducibility."
digest=hashlib.sha256(csv_bytes[0]).hexdigest()
assert all(x["csv_sha256"]==digest for x in reports)
with (ROOT/"input/test.csv").open() as f:ids=[row["SK_ID_CURR"] for row in csv.DictReader(f)]
with (dirs[0]/"submission.csv").open() as f:
    reader=csv.DictReader(f)
    assert reader.fieldnames==["SK_ID_CURR","TARGET"]
    rows=list(reader)
assert len(rows)==len(ids)==61500 and [v["SK_ID_CURR"] for v in rows]==ids
assert all(math.isfinite(float(v["TARGET"])) and 0<=float(v["TARGET"])<=1 for v in rows)
proof=dict(identical_csv_bytes=True,csv_sha256_run_1=digest,csv_sha256_run_2=digest,
           notebook_sha256=hashlib.sha256(source).hexdigest(),original_definitions_preserved=24,
           seconds=[v["total_seconds"] for v in reports],full_train_seconds=[v["full_train_seconds"] for v in reports],
           input_sha256=r["input_sha256"],versions=reports[0]["exact_installed_versions"],
           platform=reports[0]["platform"],python=reports[0]["python"],
           method="Same notebook; all code cells executed using execute_notebook.py in two new Python processes in separate directories; no saved predictions or models loaded.",
           limitation="Local macOS ARM64 CPU; Colab/Linux and equality across platforms not tested.")
(ROOT/"v21/v21/evidence/reproducibility.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(proof,ensure_ascii=False,indent=2))
