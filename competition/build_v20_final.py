"""Freeze a complete-training v20 only after both confirmation gains."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/"v20/evidence/experiment_report.json").read_text())
assert report["confirmed"] and report["status"]=="gain_confirmed", "No confirmed improvement: do not create a new submission."
recipe={key:report[key] for key in ["weights","baseline_sizes","selected_candidate","candidate_size","table","input_sha256"]}
nb=json.loads((ROOT/"v20/gci_home_credit_v20_pesquisa.ipynb").read_text())
nb["cells"][0]["source"]=(
    "# Home Credit — v20: treino completo e reproduzível\n\n"
    "Usa somente input/train.csv e input/test.csv. Todos os componentes treinados do zero; "
    "não carrega previsões, OOFs, modelos ou checkpoints salvos. Sementes e duas threads por modelo fixas. "
    "Funções da v17 preservadas. O isolamento da MLP continua corrigido. "
    "Receita escolhida nas partes 0–1, aceita por melhorar 2–4 e o total; nenhum ajuste depois da confirmação. "
    "Tamanhos = média do early stopping × 1,15, arredondada. "
    "Por padrão executa o treino integral sequencial; GCI_RUN_CV=1 refaz a pesquisa, com dois processos de duas threads. "
    "As AUCs fixadas são registros da validação, não medições de uma execução sem CV.\n"
)
# Joblib is used only for optional CV. Pin it as well for the complete package.
setup="".join(nb["cells"][1]["source"])
setup=setup.replace('"threadpoolctl": "3.6.0"}', '"threadpoolctl": "3.6.0", "joblib": "1.6.0"}')
nb["cells"][1]["source"]=setup
source='''
def v20_fit_full_candidate(ctx, configuration, size):
    a=np.arange(ctx["n"])
    fset="plusx" if configuration["kind"]=="lgb_resolution" else "tabm_periodic"
    b=np.arange(ctx["n"],ctx["n"]+len(ctx["Xt"][fset]))
    xa=ctx["sup"].add(ctx["X"][fset],a,a)
    xt=ctx["sup"].add(ctx["Xt"][fset],a,b)
    if configuration["kind"]=="lgb_resolution":
        parameters=dict(LGB_PARAMS["B"],max_bin=configuration["max_bin"])
        model=lgb.LGBMClassifier(**parameters,n_estimators=int(size),random_state=42).fit(xa,ctx["y"])
        return model.predict_proba(xt)[:,1]
    pair,(test_pair,),cards=periodic_preprocess(ctx,xa,[xt])
    model,_,_=v20_train_pairwise(pair,ctx["y"],None,None,cards,42,configuration["pair_weight"],n_epochs_fixed=int(size))
    return periodic_predict(model,test_pair)


RUN_CV=os.environ.get("GCI_RUN_CV","0")=="1"
CV_ONLY=os.environ.get("GCI_CV_ONLY","0")=="1"
FROZEN_V20=__RECIPE__
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes==FROZEN_V20["input_sha256"],"Input files differ from validation."
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
assert all("SK_ID_CURR" not in frame and "TARGET" not in frame for sets in [ctx["X"],ctx["Xt"]] for frame in sets.values())
recipe=FROZEN_V20
approved=True
if RUN_CV:
    fresh=v20_run(ctx,BASE_DIR/"v20/evidence")
    approved=fresh["confirmed"]
    if approved: recipe={key:fresh[key] for key in FROZEN_V20}
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
        component_started=time.time()
        if component["kind"]=="nn":
            p=fit_full_nn(ctx,component["fset"],size,seeds=tuple(component["seeds"]),hp=component["hp"])
        else:
            p=fit_full(ctx,component["fset"],component["kind"],size,seeds=tuple(component["seeds"]))
        predictions.append(p)
        full_timings.append(dict(component=component["label"],seconds=time.time()-component_started))
        print(component["label"],"concluído",round(time.time()-started),"s",flush=True)
    component_started=time.time()
    predictions.append(fit_full_periodic(ctx,recipe["baseline_sizes"][4]))
    full_timings.append(dict(component="TabM periódico",seconds=time.time()-component_started))
    if recipe["selected_candidate"] is not None:
        component_started=time.time()
        predictions.append(v20_fit_full_candidate(ctx,recipe["selected_candidate"],recipe["candidate_size"]))
        full_timings.append(dict(component=recipe["selected_candidate"]["label"],seconds=time.time()-component_started))
    p=sum(w*to_rank(pred)**2 for w,pred in zip(recipe["weights"],predictions))
    assert len(predictions)==len(recipe["weights"])
    assert len(p)==61500 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    submission=pd.DataFrame({"SK_ID_CURR":test.SK_ID_CURR.to_numpy(),"TARGET":p})
    submission.to_csv(output_file,index=False,float_format="%.17g",lineterminator="\\n")
    csv_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest()
    full_train_seconds=time.time()-started
    print("CSV",output_file,"SHA-256",csv_sha256)
else:
    print("Nenhum CSV novo foi escrito.")
execution_report=dict(
    comparison=dict(improved=approved,weights=recipe["weights"],table=recipe["table"]),
    final_sizes=recipe["baseline_sizes"]+([] if recipe["selected_candidate"] is None else [recipe["candidate_size"]]),
    selected_candidate=recipe["selected_candidate"],input_sha256=input_hashes,
    versions=PINNED,exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
    imported_versions=LOADED_VERSIONS,csv_sha256=csv_sha256,full_train_seconds=full_train_seconds,
    full_timings=full_timings,total_seconds=time.time()-notebook_wall_started,
    python=sys.version,platform=platform.platform(),threads_per_model=2,torch_interop_pool=TORCH_INTEROP_THREADS,
    threadpools=threadpool_info(),run_cv=RUN_CV,cv_only=CV_ONLY)
(output_dir/"execution_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
'''.replace("__RECIPE__",repr(recipe))
nb["cells"][-1]["source"]=source
base_notebook=json.loads((ROOT/"gci_home_credit_v18.ipynb").read_text())
nb["cells"].append(dict(base_notebook["cells"][30]))
for cell in nb["cells"]:
    if cell["cell_type"]=="code":
        ast.parse("".join(cell["source"]))
        cell.update(execution_count=None,outputs=[])
data=(json.dumps(nb,ensure_ascii=False,indent=1)+"\n").encode()
(ROOT/"gci_home_credit_v20.ipynb").write_bytes(data)
for run in [1,2]:
    directory=ROOT/f"v20_final_{run}"
    directory.mkdir(exist_ok=True)
    (directory/"gci_home_credit_v20.ipynb").write_bytes(data)
    if not (directory/"input").exists(): (directory/"input").symlink_to(ROOT/"input",target_is_directory=True)
print("v20 final pronta; execute duas reproduções em diretórios separados.")
