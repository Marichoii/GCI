"""Build a full-training v19 only after a real confirmed validation gain."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
report_path = ROOT / "v19/evidence/experiment_report.json"
report = json.loads(report_path.read_text())
assert report["status"] == "gain_confirmed" and report["confirmed"], "No confirmed gain: no final candidate."
assert report["alpha"] > 0
recipe = {key: report[key] for key in ["weights", "baseline_sizes", "linear_size", "configuration", "alpha", "table", "input_sha256"]}
nb = json.loads((ROOT / "v19/gci_home_credit_v19_pesquisa.ipynb").read_text())
nb["cells"][0]["source"] = (
    "# Home Credit — v19: treino integral, folhas lineares\n\n"
    "Somente input/train.csv e input/test.csv; seis componentes treinados do zero. "
    "Sem carregar previsões, OOFs, modelos ou checkpoints. "
    "Sementes e duas threads fixas. Os pesos abaixo foram escolhidos com os rótulos das partes 0–1; "
    "a receita só foi aceita por melhorar a AUC nas partes 2–4 e no total. "
    "Tamanhos de treino = média do early stopping × 1,15, arredondada. "
    "Por padrão faz apenas o treino integral; GCI_RUN_CV=1 repete a pesquisa completa.\n"
)
full_source = '''
def fit_full_linear(ctx, configuration, n_trees):
    a = np.arange(ctx["n"])
    b = np.arange(ctx["n"], ctx["n"] + len(ctx["Xt"]["plusx"]))
    xa = ctx["sup"].add(ctx["X"]["plusx"], a, a)
    xt = ctx["sup"].add(ctx["Xt"]["plusx"], a, b)
    za, (zt,) = linear_inputs(ctx, xa, [xt])
    model = linear_model(configuration, n_trees=n_trees).fit(za, ctx["y"])
    return model.predict_proba(zt)[:, 1]


RUN_CV = os.environ.get("GCI_RUN_CV", "0") == "1"
CV_ONLY = os.environ.get("GCI_CV_ONLY", "0") == "1"
FROZEN_V19 = __RECIPE__
train = pd.read_csv(BASE_DIR / "input/train.csv")
test = pd.read_csv(BASE_DIR / "input/test.csv")
assert train.shape == (171202, 34) and test.shape == (61500, 33)
input_hashes = {name: hashlib.sha256((BASE_DIR / "input" / name).read_bytes()).hexdigest()
                for name in ["train.csv", "test.csv"]}
assert input_hashes == FROZEN_V19["input_sha256"], "Different inputs: revalidate the recipe."
ctx = prepare(train, test)
add_tabm_feature_set(ctx)
for sets in [ctx["X"], ctx["Xt"]]:
    assert all("SK_ID_CURR" not in x and "TARGET" not in x for x in sets.values())
recipe = FROZEN_V19
approved = True
if RUN_CV:
    fresh = run_v19_experiment(ctx, BASE_DIR / "v19/evidence")
    approved = fresh["confirmed"]
    if approved:
        recipe = {key: fresh[key] for key in FROZEN_V19}
else:
    print("Receita já validada: AUCs registradas, não recalculadas nesta reprodução.")
print(pd.DataFrame(recipe["table"]).to_string(index=False))
print("Pesos", recipe["weights"], "tamanhos", recipe["baseline_sizes"] + [recipe["linear_size"]])
output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "submission.csv"
csv_sha256 = None
full_train_seconds = None
if approved and not CV_ONLY:
    full_started = time.time()
    predictions = []
    for component, size in zip(COMPONENTS, recipe["baseline_sizes"][:4]):
        if component["kind"] == "nn":
            p = fit_full_nn(ctx, component["fset"], size, seeds=tuple(component["seeds"]), hp=component["hp"])
        else:
            p = fit_full(ctx, component["fset"], component["kind"], size, seeds=tuple(component["seeds"]))
        predictions.append(p)
        print(component["label"], "concluído", round(time.time()-full_started), "s", flush=True)
    predictions.append(fit_full_periodic(ctx, recipe["baseline_sizes"][4]))
    predictions.append(fit_full_linear(ctx, recipe["configuration"], recipe["linear_size"]))
    p = sum(weight * to_rank(pred)**2 for weight, pred in zip(recipe["weights"], predictions))
    assert len(p) == 61500 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR.to_numpy(), "TARGET": p}).to_csv(
        output_file, index=False, float_format="%.17g", lineterminator="\\n")
    csv_sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    full_train_seconds = time.time() - full_started
    print("CSV SHA-256", csv_sha256)
else:
    print("Nenhum CSV novo escrito nesta execução.")
execution_report = dict(
    comparison=dict(improved=approved, weights=recipe["weights"], table=recipe["table"], alpha=recipe["alpha"]),
    final_sizes=recipe["baseline_sizes"] + [recipe["linear_size"]],
    linear_configuration=recipe["configuration"], input_sha256=input_hashes,
    versions=PINNED, exact_installed_versions={p: importlib.metadata.version(p) for p in PINNED},
    imported_versions=LOADED_VERSIONS, csv_sha256=csv_sha256,
    full_train_seconds=full_train_seconds, total_seconds=time.time()-notebook_wall_started,
    python=sys.version, platform=platform.platform(), threads_per_model=N_JOBS,
    torch_interop_pool=TORCH_INTEROP_THREADS, threadpools=threadpool_info(), run_cv=RUN_CV, cv_only=CV_ONLY)
(output_dir / "execution_report.json").write_text(json.dumps(execution_report, ensure_ascii=False, indent=2) + "\\n")
'''.replace("__RECIPE__", repr(recipe))
nb["cells"][-1]["source"] = full_source
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
        cell.update(execution_count=None, outputs=[])
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode()
(ROOT / "gci_home_credit_v19.ipynb").write_bytes(data)
runtime = ROOT / "v19_final"
runtime.mkdir(exist_ok=True)
(runtime / "gci_home_credit_v19.ipynb").write_bytes(data)
if not (runtime / "input").exists():
    (runtime / "input").symlink_to(ROOT / "input", target_is_directory=True)
print(runtime / "gci_home_credit_v19.ipynb")
