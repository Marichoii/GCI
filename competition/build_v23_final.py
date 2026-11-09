"""Build the complete v23 only after its frozen recipe passes confirmation."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT/"v23/evidence/experiment_report.json").read_text())
assert r["confirmed"] and r["status"] == "gain_confirmed"
rows = {v["parte"]: v for v in r["table"]}
assert all(rows[k]["delta"] > 0 for k in ["partes_2_4", "total"])
frozen = json.loads((ROOT/"v23/evidence/frozen_selection.json").read_text())
assert frozen["status"] == "recipe_frozen_before_confirmation" and "table" not in frozen
assert frozen["alpha"] == r["alpha"] and frozen["weights"] == r["weights"]
recipe = {k: r[k] for k in ["weights", "alpha", "final_sizes", "table", "input_sha256"]}
nb = json.loads((ROOT/"v23/gci_home_credit_v23_pesquisa.ipynb").read_text())
nb["cells"] = nb["cells"][:-1]
nb["cells"][0]["source"] = (
    "# Home Credit — v23: treino completo com ensemble em log-odds\n\n"
    "Todos os sete componentes treinam do zero usando apenas input/train.csv e input/test.csv. "
    "Sem carregar previsões, OOFs, checkpoints ou modelos. Sementes fixas e duas threads por modelo. "
    "Pesos dos componentes da v21 preservados; alpha escolhido somente em0–1, congelado antes de medir2–4 e total, "
    "com ganho em ambos. Tamanhos=média do early stopping×1,15, arredondada. "
    "AUCs registradas da validação; execução padrão treina em100% e grava output/submission.csv. "
    "GCI_RUN_CV=1 refaz a validação; GCI_CV_ONLY=1 evita treino integral nessa opção. "
    "24 funções/classes originais da v17 preservadas, inclusive uso não supervisionado de covariáveis train+test. "
    "O rank da média ponderada dos log-odds é combinado com o score atual, sem meta-modelo treinado. "
    "Os scores de saída são de ordenação e não probabilidades calibradas.\n")
v21 = json.loads((ROOT/"gci_home_credit_v21.ipynb").read_text())
s = "".join(v21["cells"][26]["source"])
helpers = []
for node in ast.parse(s).body:
    if isinstance(node, ast.FunctionDef) and node.name in ["v20_fit_full_candidate", "v21_fit_full_candidate"]:
        helpers.append(ast.get_source_segment(s, node))
assert len(helpers) == 2
code = '''
RUN_CV=os.environ.get("GCI_RUN_CV","0")=="1"
CV_ONLY=os.environ.get("GCI_CV_ONLY","0")=="1"
FROZEN_V23=__RECIPE__
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes==FROZEN_V23["input_sha256"]
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
assert all("SK_ID_CURR" not in frame and "TARGET" not in frame for sets in [ctx["X"],ctx["Xt"]] for frame in sets.values())
recipe=FROZEN_V23
approved=True
if RUN_CV:
    fresh=v23_run(ctx,BASE_DIR/"evidence")
    approved=fresh["confirmed"]
    if approved:recipe={key:fresh[key] for key in FROZEN_V23}
else:
    print("Receita fixa: AUCs registradas, não recalculadas nesta reprodução.")
print(pd.DataFrame(recipe["table"]).to_string(index=False))
print("Pesos",recipe["weights"],"alpha",recipe["alpha"],"tamanhos",recipe["final_sizes"])
assert np.allclose(recipe["weights"],V23_WEIGHTS,rtol=0,atol=0)
output_dir=BASE_DIR/"output"
output_dir.mkdir(exist_ok=True)
output_file=output_dir/"submission.csv"
csv_sha256=None
full_train_seconds=None
full_timings=[]
if approved and not CV_ONLY:
    started=time.time()
    predictions=[]
    for component,size in zip(COMPONENTS,recipe["final_sizes"][:4]):
        t=time.time()
        if component["kind"]=="nn":
            p=fit_full_nn(ctx,component["fset"],size,seeds=tuple(component["seeds"]),hp=component["hp"])
        else:
            p=fit_full(ctx,component["fset"],component["kind"],size,seeds=tuple(component["seeds"]))
        predictions.append(p)
        full_timings.append(dict(component=component["label"],seconds=time.time()-t))
        print(component["label"],"concluído",round(time.time()-started),"s",flush=True)
    t=time.time()
    predictions.append(fit_full_periodic(ctx,recipe["final_sizes"][4]))
    full_timings.append(dict(component="TabM periódico original",seconds=time.time()-t))
    t=time.time()
    predictions.append(v20_fit_full_candidate(ctx,dict(label="tabm_pair_full",kind="tabm_pairwise",pair_weight=1.0),recipe["final_sizes"][5]))
    full_timings.append(dict(component="TabM periódico por pares",seconds=time.time()-t))
    t=time.time()
    predictions.append(v21_fit_full_candidate(ctx,V21_CONFIGS[0],recipe["final_sizes"][6]))
    full_timings.append(dict(component="cat_ordered_depth5",seconds=time.time()-t))
    assert len(predictions)==7
    baseline,alternative=v23_scores(np.column_stack(predictions))
    alpha=recipe["alpha"]
    p=(1-alpha)*baseline+alpha*alternative
    assert len(p)==61500 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    submission=pd.DataFrame({"SK_ID_CURR":test.SK_ID_CURR.to_numpy(),"TARGET":p})
    submission.to_csv(output_file,index=False,float_format="%.17g",lineterminator="\\n")
    csv_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest()
    full_train_seconds=time.time()-started
    print("CSV",output_file,"SHA-256",csv_sha256,flush=True)
else:
    print("Nenhum CSV novo foi escrito.")
execution_report=dict(comparison=dict(improved=approved,weights=recipe["weights"],alpha=recipe["alpha"],table=recipe["table"]),
    final_sizes=recipe["final_sizes"],input_sha256=input_hashes,versions=PINNED,
    exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
    csv_sha256=csv_sha256,full_train_seconds=full_train_seconds,full_timings=full_timings,
    total_seconds=time.time()-notebook_wall_started,python=sys.version,platform=platform.platform(),
    threads_per_model=2,torch_interop_pool=TORCH_INTEROP_THREADS,threadpools=threadpool_info(),run_cv=RUN_CV,cv_only=CV_ONLY)
(output_dir/"execution_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
'''.replace("__RECIPE__", repr(recipe))
nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source="\n\n".join(helpers)+"\n\n"+code))
nb["cells"].append(v21["cells"][-1])
found = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    c.update(execution_count=None, outputs=[])
    source = "".join(c["source"])
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name not in found
            found[node.name] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
original = json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k] == v for k, v in original.items())
data = (json.dumps(nb, ensure_ascii=False, indent=1)+"\n").encode()
(ROOT/"gci_home_credit_v23.ipynb").write_bytes(data)
for i in [1, 2]:
    directory = ROOT/f"v23_final_{i}"
    directory.mkdir(exist_ok=True)
    path = directory/"gci_home_credit_v23.ipynb"
    assert not path.exists(), "Do not overwrite a previous reproduction."
    path.write_bytes(data)
    (directory/"input").symlink_to(ROOT/"input", target_is_directory=True)
print("Frozen v23 source", hashlib.sha256(data).hexdigest())
