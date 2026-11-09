"""Verify clean v24 reproductions before publishing the submission artifact."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
reports=[]
csvs=[]
sources=[]
for run in [1,2]:
    directory=ROOT/f"v24_final_{run}/runs/reproduction_{run}"
    report=json.loads((directory/"execution_report.json").read_text())
    reports.append(report)
    csvs.append((directory/"submission.csv").read_bytes())
    sources.append((directory/"source.ipynb").read_bytes())
    assert report["threads_per_model"]==report["torch_interop_pool"]==2
    assert all(pool["num_threads"]<=2 for pool in report["threadpools"])
    assert report["total_seconds"]<3600,"Runtime exceeds frozen limit; do not publish."
assert csvs[0]==csvs[1] and sources[0]==sources[1]
assert reports[0]["comparison"]==reports[1]["comparison"]
digest=hashlib.sha256(csvs[0]).hexdigest()
assert all(r["csv_sha256"]==digest for r in reports)
actual=pd.read_csv(ROOT/"v24_final_1/runs/reproduction_1/submission.csv")
test=pd.read_csv(ROOT/"input/test.csv")
assert actual.shape==(61500,2) and actual.columns.tolist()==["SK_ID_CURR","TARGET"]
assert actual.SK_ID_CURR.tolist()==test.SK_ID_CURR.tolist()
proof=dict(identical_csv_bytes=True,csv_sha256_run_1=digest,csv_sha256_run_2=digest,
           notebook_sha256=hashlib.sha256(sources[0]).hexdigest(),
           seconds=[r["total_seconds"] for r in reports],
           platform=[r["platform"] for r in reports],fresh_python_processes=True,
           colab_executed=False,threads_per_model=2)
path=ROOT/"v24/evidence/reproducibility.json"
assert not path.exists()
path.write_text(json.dumps(proof,indent=2)+"\n")
print(json.dumps(proof,indent=2))
