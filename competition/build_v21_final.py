"""Freeze the accepted v21 recipe; never run when confirmation fails."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/"v21/v21/evidence/experiment_report.json").read_text())
assert r["confirmed"] and r["status"]=="gain_confirmed"
by_name={v["parte"]:v for v in r["table"]}
assert by_name["partes_2_4"]["delta"]>0 and by_name["total"]["delta"]>0
recipe={k:r[k] for k in ["weights","alpha","baseline_sizes","candidate_size","selected_candidate","table","input_sha256"]}
nb=json.loads((ROOT/"v21/gci_home_credit_v21_pesquisa.ipynb").read_text())
nb["cells"]=nb["cells"][:-2] # Remove research execution and put credits back below.
nb["cells"][0]["source"]=(
    "# Home Credit — v21: treino integral reproduzível\n\n"
    "Usa somente input/train.csv e input/test.csv; treina todos os componentes do zero. "
    "Não carrega previsões, OOFs, checkpoints ou modelos salvos. Duas threads por modelo e sementes fixas. "
    "24 definições da v17 preservadas; pré-processamento supervisionado fora da parte. "
    "Família e peso escolhidos somente em 0–1; confirmação positiva em 2–4 e total. Nenhum ajuste posterior. "
    "Tamanhos = média do early stopping × 1,15, arredondada. As AUCs são registros da validação. "
    "Por padrão treina em 100% dos dados e grava output/submission.csv. GCI_RUN_CV=1 refaz a pesquisa.\n")
v20=json.loads((ROOT/"gci_home_credit_v20.ipynb").read_text())
s="".join(v20["cells"][25]["source"])
node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=="v20_fit_full_candidate")
old_pair_full=ast.get_source_segment(s,node)
code='''
def v21_fit_full_candidate(ctx, configuration, size):
    a=np.arange(ctx["n"])
    fset="catx" if configuration["kind"]=="cat_ordered" else "plusx"
    b=np.arange(ctx["n"],ctx["n"]+len(ctx["Xt"][fset]))
    xa=ctx["sup"].add(ctx["X"][fset],a,a)
    xt=ctx["sup"].add(ctx["Xt"][fset],a,b)
    model=v21_candidate_model(configuration,size=size,cat_cols=ctx["cat_cols"])
    model.fit(xa,ctx["y"])
    return model.predict_proba(xt)[:,1]


RUN_CV=os.environ.get("GCI_RUN_CV","0")=="1"
CV_ONLY=os.environ.get("GCI_CV_ONLY","0")=="1"
FROZEN_V21=__RECIPE__
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes==FROZEN_V21["input_sha256"],"Input files differ from validation."
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
assert all("SK_ID_CURR" not in frame and "TARGET" not in frame for sets in [ctx["X"],ctx["Xt"]] for frame in sets.values())
recipe=FROZEN_V21
approved=True
if RUN_CV:
    fresh=v21_run(ctx,BASE_DIR/"v21/evidence")
    approved=fresh["confirmed"]
    if approved:recipe={key:fresh[key] for key in FROZEN_V21}
else:
    print("Receita fixa: AUCs registradas, não recalculadas nesta reprodução.")
print(pd.DataFrame(recipe["table"]).to_string(index=False))
print("Pesos",recipe["weights"],"tamanhos",recipe["baseline_sizes"],recipe["candidate_size"])
output_dir=BASE_DIR/"output"
output_dir.mkdir(exist_ok=True)
output_file=output_dir/"submission.csv"
csv_sha256=None
full_train_seconds=None
full_timings=[]
if approved and not CV_ONLY:
    started=time.time()
    predictions=[]
    for component,size in zip(COMPONENTS,recipe["baseline_sizes"][:4]):
        t=time.time()
        if component["kind"]=="nn":
            p=fit_full_nn(ctx,component["fset"],size,seeds=tuple(component["seeds"]),hp=component["hp"])
        else:
            p=fit_full(ctx,component["fset"],component["kind"],size,seeds=tuple(component["seeds"]))
        predictions.append(p)
        full_timings.append(dict(component=component["label"],seconds=time.time()-t))
        print(component["label"],"concluído",round(time.time()-started),"s",flush=True)
    t=time.time()
    predictions.append(fit_full_periodic(ctx,recipe["baseline_sizes"][4]))
    full_timings.append(dict(component="TabM periódico original",seconds=time.time()-t))
    t=time.time()
    predictions.append(v20_fit_full_candidate(ctx,dict(label="tabm_pair_full",kind="tabm_pairwise",pair_weight=1.0),recipe["baseline_sizes"][5]))
    full_timings.append(dict(component="TabM periódico por pares",seconds=time.time()-t))
    t=time.time()
    predictions.append(v21_fit_full_candidate(ctx,recipe["selected_candidate"],recipe["candidate_size"]))
    full_timings.append(dict(component=recipe["selected_candidate"]["label"],seconds=time.time()-t))
    assert len(predictions)==len(recipe["weights"])
    p=sum(w*to_rank(pred)**2 for w,pred in zip(recipe["weights"],predictions))
    assert len(p)==61500 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    submission=pd.DataFrame({"SK_ID_CURR":test.SK_ID_CURR.to_numpy(),"TARGET":p})
    submission.to_csv(output_file,index=False,float_format="%.17g",lineterminator="\\n")
    csv_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest()
    full_train_seconds=time.time()-started
    print("CSV",output_file,"SHA-256",csv_sha256)
else:
    print("Nenhum CSV novo foi escrito.")
execution_report=dict(comparison=dict(improved=approved,weights=recipe["weights"],table=recipe["table"]),
    final_sizes=recipe["baseline_sizes"]+[recipe["candidate_size"]],selected_candidate=recipe["selected_candidate"],
    input_sha256=input_hashes,versions=PINNED,exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
    csv_sha256=csv_sha256,full_train_seconds=full_train_seconds,full_timings=full_timings,
    total_seconds=time.time()-notebook_wall_started,python=sys.version,platform=platform.platform(),
    threads_per_model=2,torch_interop_pool=TORCH_INTEROP_THREADS,threadpools=threadpool_info(),run_cv=RUN_CV,cv_only=CV_ONLY)
(output_dir/"execution_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
'''.replace("__RECIPE__",repr(recipe))
nb["cells"].append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=old_pair_full+"\n\n"+code))
nb["cells"].append(v20["cells"][-1])
found={}
for c in nb["cells"]:
    if c["cell_type"]!="code":continue
    s="".join(c["source"])
    for node in ast.parse(s).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            assert node.name not in found
            found[node.name]=hashlib.sha256(ast.get_source_segment(s,node).encode()).hexdigest()
    c.update(execution_count=None,outputs=[])
manifest=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k]==v for k,v in manifest.items())
data=(json.dumps(nb,ensure_ascii=False,indent=1)+"\n").encode()
(ROOT/"gci_home_credit_v21.ipynb").write_bytes(data)
for i in [1,2]:
    directory=ROOT/f"v21_final_{i}"
    directory.mkdir(exist_ok=True)
    (directory/"gci_home_credit_v21.ipynb").write_bytes(data)
    if not (directory/"input").exists():(directory/"input").symlink_to(ROOT/"input",target_is_directory=True)
print("Frozen final source",hashlib.sha256(data).hexdigest())
