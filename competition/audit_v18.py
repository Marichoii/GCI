"""Audit original definitions and TARGET isolation on the actual five splits."""
import ast
import hashlib
import json
import os
from pathlib import Path

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent


def definitions(nb):
    result = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        for node in ast.parse(s).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in result, f"Redefinição: {node.name}"
                result[node.name] = ast.get_source_segment(s, node)
    return result


original = definitions(json.loads((ROOT / "sources/gci_home_credit_v17.ipynb").read_text()))
current = definitions(json.loads((ROOT / "gci_home_credit_v18.ipynb").read_text()))
assert all(current.get(k) == v for k, v in original.items())
ns = dict(np=np, pd=pd, StratifiedKFold=StratifiedKFold, PRIOR=0.0807)
for name in ["person_groups", "tariff_keys", "_rate_stats", "_rates", "nested_rates", "Supervised"]:
    exec(current[name], ns)

train = pd.read_csv(ROOT / "input/train.csv")
test = pd.read_csv(ROOT / "input/test.csv")
raw = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
y = train.TARGET.to_numpy()
y_all = np.r_[y, np.zeros(len(test))]
groups = ns["person_groups"](raw)
keys = ns["tariff_keys"](raw)
Supervised = ns["Supervised"]
sup = Supervised(groups, keys, y_all)
idx_te = np.arange(len(train), len(raw))
rows = []
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
for fold, (a, b) in enumerate(splits):
    changed = y_all.copy()
    changed[b] = 1 - changed[b]
    changed_sup = Supervised(groups, keys, changed)
    for label, dst in [("train", a), ("valid", b), ("test", idx_te)]:
        frame = pd.DataFrame(index=np.arange(len(dst)))
        before = sup.add(frame, a, dst)
        after = changed_sup.add(frame, a, dst)
        pd.testing.assert_frame_equal(before, after)
    rows.append(dict(fold=fold, train=len(a), valid=len(b), target_isolation=True))
    print("PASS TARGET isolation", fold, flush=True)

report = dict(original_definitions_preserved=len(original), folds=rows,
              inputs={f: hashlib.sha256((ROOT / "input" / f).read_bytes()).hexdigest()
                      for f in ["train.csv", "test.csv"]})
(ROOT / "target_isolation_audit.json").write_text(json.dumps(report, indent=2))
