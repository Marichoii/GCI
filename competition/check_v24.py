"""Focused checks of FM mathematics and the frozen experiment boundaries."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb_path = ROOT/"v24/gci_home_credit_v24_pesquisa.ipynb"
nb = json.loads(nb_path.read_text())
os.chdir(nb_path.parent)
namespace = {"__name__":"v24_checks"}
# Execute definitions and imports, excluding the research execution cell.
for cell in nb["cells"][:-2]:
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), "definitions", "exec"), namespace)
np, torch = namespace["np"], namespace["torch"]
torch.manual_seed(123)
model = namespace["V24FactorizationMachine"](5, 3, 0.1).double()
x = torch.randn(7, 5, dtype=torch.float64)
direct = model.linear(x).squeeze(-1)
for i in range(5):
    for j in range(i+1, 5):
        direct = direct + (model.factors[i]*model.factors[j]).sum()*x[:,i]*x[:,j]
torch.testing.assert_close(model(x), direct, rtol=1e-12, atol=1e-12)
assert torch.autograd.gradcheck(model, (x.requires_grad_(),), eps=1e-6, atol=1e-5)
checks = ["FM equals explicit pairwise interactions", "FM analytical gradient"]
pd = namespace["pd"]
training = pd.DataFrame(dict(value=[0.,1.,2.,3.,4.,5.], binary=[0,1,0,1,0,1],
                             large_count=[21920.,210782.,21920.,210782.,21920.,210782.],
                             missing=[1.,2.,np.nan,4.,5.,6.], category=[0,0,1,1,0,1]))
held = training.iloc[:2].copy()
held["category"] = 999
data1, (_,), meta1 = namespace["v24_preprocess"](training, [held], ["category"])
held["value"] = 1e12
data2, (_,), meta2 = namespace["v24_preprocess"](training, [held], ["category"])
torch.testing.assert_close(data1, data2, rtol=0, atol=0)
assert meta1 == meta2
assert "large_count" in meta1["continuous"] and "large_count" not in meta1["binary"]
assert float(data1.abs().max()) <= 4.0
checks.append("held-out covariates do not alter FM training transforms; unknown category accepted")
checks.append("two-level large counts are normalized; only genuine zero-one flags bypass quantiles")
raw = pd.DataFrame(dict(AMT_CREDIT=[100.,100.,120.,120.,100.,110.,0.,100.],
                        AMT_ANNUITY=[10.,10.,10.,10.,10.,10.,10.,np.nan]))
splits = [(np.arange(4), np.arange(4,8)), (np.arange(4,8), np.arange(4))]
labels, frequency = namespace["v24_strata"](raw, splits)
assert frequency.tolist() == [1,1,0,0,2,0,-1,-1], frequency
assert labels.tolist() == ["raras_0_9"]*6+["ausentes_invalidas"]*2
checks.append("stratum counts exclude each evaluated row and its outer fold")
chosen, grid = namespace["v24_choose"](np.array([0,0,1,1]), np.array([.1,.2,.8,.9]), np.array([.9,.8,.2,.1]))
assert chosen["alpha"] == 0.0
assert [row["alpha"] for row in grid] == [0.,.05,.10,.15,.20]
for alpha in [0.,.05,.10,.15,.20]:
    weights = np.r_[(1-alpha)*namespace["V24_WEIGHTS"],alpha]
    assert abs(weights.sum()-1) < 1e-12
checks.append("fixed alpha grid; tie chooses zero; v21 proportions preserved")
source = (ROOT/"v24_fm.py").read_text()
tree = ast.parse(source)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and n.func.attr in {"load","loadtxt","read_pickle","read_parquet"} for n in ast.walk(tree))
protocol = json.loads((ROOT/"v24/evidence/frozen_protocol.json").read_text())
assert protocol == namespace["V24_PROTOCOL"]
checks.append("no prediction/model loader in FM module; frozen protocol matches notebook")
report = dict(checks=checks, passed=True, real_competition_model_training_executed=False,
              source_sha256=hashlib.sha256(nb_path.read_bytes()).hexdigest())
out = ROOT/"v24/evidence/implementation_checks.json"
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
