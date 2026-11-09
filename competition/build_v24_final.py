"""Create a standalone final notebook only after v24 meets frozen promotion."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/"v24/evidence/experiment_report.json").read_text())
assert report["status"]=="gain_confirmed" and report["confirmed"]
rows={row["parte"]:row for row in report["table"]}
assert rows["partes_2_4"]["delta"]>=0.00010 and rows["total"]["delta"]>0
recipe={key:report[key] for key in ["weights","alpha","baseline_final_sizes","fm_final_epochs","table","input_sha256","protocol_sha256"]}
nb=json.loads((ROOT/"v24/gci_home_credit_v24_pesquisa.ipynb").read_text())
nb["cells"]=nb["cells"][:27]
nb["cells"][0]["source"]=("# Home Credit v24 — treino integral reproduzível\n\n"
    "Receita FM aprovada pelos critérios congelados: ganho 2–4 ≥0,00010 e ganho total positivo. "
    "Treina todos os componentes do zero usando somente input/train.csv e input/test.csv. "
    "Sem carregar previsões, OOFs, checkpoints ou modelos salvos; sem feature de ID. "
    "Seeds fixas e duas threads; CPU. AUCs registradas na receita são evidências históricas, "
    "não recalculadas nesta reprodução. Executar todas as células em runtime novo. "
    "O resultado é output/submission.csv. Pré-processamento não supervisionado train+test herdado da v21.\n")
old=json.loads((ROOT/"gci_home_credit_v21.ipynb").read_text())
source="".join(old["cells"][26]["source"])
definitions=[]
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and node.name in {"v20_fit_full_candidate","v21_fit_full_candidate"}:
        definitions.append(ast.get_source_segment(source,node))
assert len(definitions)==2
code='''
FROZEN_V24=__RECIPE__
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes==FROZEN_V24["input_sha256"],"Inputs differ from validation."
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
assert all("TARGET" not in frame and "SK_ID_CURR" not in frame for key in ["X","Xt"] for frame in ctx[key].values())
recipe=FROZEN_V24
print("AUCs registradas, não recalculadas nesta execução:")
print(pd.DataFrame(recipe["table"]).to_string(index=False))
output_dir=BASE_DIR/"output"
output_dir.mkdir(exist_ok=True)
output_file=output_dir/"submission.csv"
assert not output_file.exists(),"Preserve earlier output; use a fresh directory."
started=time.time()
predictions=[]
full_timings=[]
for component,size in zip(COMPONENTS,recipe["baseline_final_sizes"][:4]):
    t=time.time()
    if component["kind"]=="nn":
        p=fit_full_nn(ctx,component["fset"],size,seeds=tuple(component["seeds"]),hp=component["hp"])
    else:
        p=fit_full(ctx,component["fset"],component["kind"],size,seeds=tuple(component["seeds"]))
    predictions.append(p)
    full_timings.append(dict(component=component["label"],seconds=time.time()-t))
    print("Integral",component["label"],"concluído",flush=True)
t=time.time()
predictions.append(fit_full_periodic(ctx,recipe["baseline_final_sizes"][4]))
full_timings.append(dict(component="TabM periódico",seconds=time.time()-t))
t=time.time()
predictions.append(v20_fit_full_candidate(ctx,dict(label="tabm_pair_full",kind="tabm_pairwise",pair_weight=1.0),recipe["baseline_final_sizes"][5]))
full_timings.append(dict(component="TabM por pares",seconds=time.time()-t))
t=time.time()
predictions.append(v21_fit_full_candidate(ctx,dict(label="cat_ordered_depth5",kind="cat_ordered",depth=5),recipe["baseline_final_sizes"][6]))
full_timings.append(dict(component="CatBoost Ordered depth 5",seconds=time.time()-t))
t=time.time()
predictions.append(v24_fit_full_fm(ctx,recipe["fm_final_epochs"]))
full_timings.append(dict(component="FM PyTorch",seconds=time.time()-t))
assert len(predictions)==len(recipe["weights"])==8
prediction=sum(weight*to_rank(p)**2 for weight,p in zip(recipe["weights"],predictions))
assert len(prediction)==61500 and np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
submission=pd.DataFrame({"SK_ID_CURR":test.SK_ID_CURR.to_numpy(),"TARGET":prediction})
submission.to_csv(output_file,index=False,float_format="%.17g",lineterminator="\\n")
csv_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest()
execution_report=dict(csv_sha256=csv_sha256,total_seconds=time.time()-notebook_wall_started,
    full_train_seconds=time.time()-started,full_timings=full_timings,comparison=recipe,
    input_sha256=input_hashes,exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
    python=sys.version,platform=platform.platform(),threads_per_model=2,
    torch_interop_pool=torch.get_num_interop_threads(),threadpools=threadpool_info(),
    run_cv=False,cv_only=False)
v24_json(output_dir/"execution_report.json",execution_report)
print("SHA-256 CSV",csv_sha256,"tempo total",execution_report["total_seconds"],flush=True)
'''.replace("__RECIPE__",repr(recipe))
nb["cells"].append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source="\n\n".join(definitions)+"\n\n"+code))
for cell in nb["cells"]:
    if cell["cell_type"]=="code":cell.update(execution_count=None,outputs=[])
data=(json.dumps(nb,ensure_ascii=False,indent=1)+"\n").encode()
path=ROOT/"gci_home_credit_v24.ipynb"
assert not path.exists(),"Preserve an existing frozen final notebook."
path.write_bytes(data)
for run in [1,2]:
    directory=ROOT/f"v24_final_{run}"
    directory.mkdir(exist_ok=False)
    (directory/path.name).write_bytes(data)
    (directory/"input").symlink_to(ROOT/"input",target_is_directory=True)
print("Frozen final notebook",hashlib.sha256(data).hexdigest())
