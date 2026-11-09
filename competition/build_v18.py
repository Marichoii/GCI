"""Build the auditable notebook, preserving every v17 definition verbatim."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "sources/gci_home_credit_v18.preexisting.ipynb").read_text())


def source(i, s):
    nb["cells"][i]["source"] = s.strip() + "\n"


source(1, '''
import os, sys, subprocess, importlib.metadata, random, platform
from pathlib import Path
notebook_wall_started = __import__("time").time()
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
random.seed(42)
PINNED = {"numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.17.0",
          "scikit-learn": "1.8.0", "lightgbm": "4.6.0", "catboost": "1.2.8",
          "torch": "2.6.0", "tabm": "0.0.3", "rtdl-num-embeddings": "0.0.12",
          "threadpoolctl": "3.6.0"}
needed = []
for name, version in PINNED.items():
    try:
        ok = importlib.metadata.version(name).split("+")[0] == version
    except importlib.metadata.PackageNotFoundError:
        ok = False
    if not ok:
        needed.append(f"{name}=={version}")
if needed:
    # O índice CPU é usado para PyTorch em Linux/Colab; macOS usa PyPI.
    torch_needed = [p for p in needed if p.startswith("torch==")]
    other_needed = [p for p in needed if not p.startswith("torch==")]
    if torch_needed:
        command = [sys.executable, "-m", "pip", "install", "--quiet", *torch_needed]
        if sys.platform == "linux":
            command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.run(command, check=True)
    if other_needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *other_needed], check=True)
assert {p: importlib.metadata.version(p).split("+")[0] for p in PINNED} == PINNED
# PyTorch fornece libomp no macOS; importar primeiro também resolve LightGBM
# em instalações sem o pacote libomp do Homebrew. Sem alterar as funções v17.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
import torch
BASE_DIR = Path(os.environ.get("GCI_WORKDIR", Path.cwd())).resolve()
assert (BASE_DIR / "input/train.csv").is_file(), f"Falta input/train.csv em {BASE_DIR}"
assert (BASE_DIR / "input/test.csv").is_file(), f"Falta input/test.csv em {BASE_DIR}"
''')

s = "".join(nb["cells"][17]["source"])
start = s.index("# Pool inter-op")
end = s.index("TABM_CONFIG")
s = s[:start] + '''# Pool fixo em cada processo novo; os modelos ficam na CPU.
TORCH_INTEROP_THREADS = 2
if torch.get_num_interop_threads() != TORCH_INTEROP_THREADS:
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
from threadpoolctl import threadpool_limits, threadpool_info
thread_limiter = threadpool_limits(limits=N_JOBS)
torch.set_num_threads(N_JOBS)
torch.use_deterministic_algorithms(True)

''' + s[end:]
source(17, s)

source(21, '''
# A única busca nova é o peso do TabM: 21 valores, sem retunar modelos.
ALPHA_GRID = np.round(np.arange(0.0, 0.50001, 0.025), 3)
V17_WEIGHTS = np.array([0.265, 0.051, 0.298, 0.386], dtype=float)


def select_periodic_weight(y_select, squared_ranks_select):
    # Não recebe TARGET das partes 2–4. Os percentis são os mesmos da tabela.
    baseline = squared_ranks_select[:, :4] @ V17_WEIGHTS
    scores = []
    for alpha in ALPHA_GRID:
        p = (1 - alpha) * baseline + alpha * squared_ranks_select[:, 4]
        scores.append(dict(alpha=float(alpha), auc=float(roc_auc_score(y_select, p))))
    best = max(scores, key=lambda r: r["auc"])  # empate: primeiro/menor peso
    return best["alpha"], scores


def compare_v17_v18(y, oofs):
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), dtype=np.int8)
    for fold, (_, b) in enumerate(splits):
        fold_id[b] = fold
    selection = fold_id < 2
    ranks = np.column_stack([to_rank(p) ** 2 for p in oofs])
    alpha, search = select_periodic_weight(y[selection], ranks[selection])
    weights = np.r_[(1 - alpha) * V17_WEIGHTS, alpha]
    v17 = ranks[:, :4] @ V17_WEIGHTS
    v18 = ranks @ weights if alpha > 0 else v17.copy()
    rows = []
    masks = [(f"parte_{i}", fold_id == i) for i in range(5)]
    masks += [("partes_0_1", selection), ("partes_2_4", ~selection),
              ("total", np.ones(len(y), bool))]
    for label, mask in masks:
        a = float(roc_auc_score(y[mask], v17[mask]))
        b = float(roc_auc_score(y[mask], v18[mask]))
        rows.append(dict(parte=label, linhas=int(mask.sum()), v17=a, v18=b, delta=b-a))
    by_name = {r["parte"]: r for r in rows}
    improved = (by_name["partes_2_4"]["delta"] > 0 and by_name["total"]["delta"] > 0)
    return dict(alpha=alpha, weights=weights.tolist(), search=search,
                table=rows, improved=bool(improved),
                baseline_note="v17 recalculada com isolamento do TARGET na chamada da MLP; "
                              "as 24 funções/classes originais estão intactas.",
                selection_note="Mesmos percentis globais ao quadrado da v17; "
                               "só TARGET das partes 0–1 escolhe o peso. "
                               "Nenhum ajuste após consultar partes 2–4.")
''')

source(22, '''
import scipy, sklearn, catboost, tabm, rtdl_num_embeddings
LOADED_VERSIONS = {"numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                   "lightgbm": lgb.__version__, "catboost": catboost.__version__,
                   "torch": str(torch.__version__)}
assert all(PINNED[k] == v.split("+")[0] for k, v in LOADED_VERSIONS.items()), (
    "Reinicie o runtime: versões carregadas diferem das instaladas.", LOADED_VERSIONS)
''')

source(24, '''
RUN_CV = os.environ.get("GCI_RUN_CV", "1") == "1"
CV_ONLY = os.environ.get("GCI_CV_ONLY", "0") == "1"
# Preenchida somente depois de uma validação completa executada e auditada.
FROZEN_RECIPE = None
notebook_started = time.time()
train = pd.read_csv(BASE_DIR / "input/train.csv")
test = pd.read_csv(BASE_DIR / "input/test.csv")
assert train.shape == (171202, 34) and test.shape == (61500, 33)
assert train.TARGET.isin([0, 1]).all()
assert train.SK_ID_CURR.is_unique and test.SK_ID_CURR.is_unique
input_hashes = {name: hashlib.sha256((BASE_DIR / "input" / name).read_bytes()).hexdigest()
                for name in ["train.csv", "test.csv"]}
if not RUN_CV:
    assert FROZEN_RECIPE and FROZEN_RECIPE["improved"], "Validação completa necessária."
    assert FROZEN_RECIPE["input_sha256"] == input_hashes, "Dados diferentes: use RUN_CV=True."
ctx = prepare(train, test)
add_tabm_feature_set(ctx)
for frames in [ctx["X"], ctx["Xt"]]:
    assert all("SK_ID_CURR" not in x.columns and "TARGET" not in x.columns for x in frames.values())
print("Versões:", PINNED)
print("Hashes dos dados:", input_hashes)
''')

source(26, '''
cv_results = []
cv_timings = []
if RUN_CV:
    for c in COMPONENTS:
        started = time.time()
        print("CV iniciada:", c["label"], flush=True)
        if c["kind"] == "nn":
            r = run_cv_nn_strict(ctx, c["fset"], seeds=tuple(c["seeds"]), hp=c["hp"])
        else:
            r = run_cv(ctx, c["fset"], c["kind"], seeds=tuple(c["seeds"]))
        r.pop("test", None)
        cv_results.append(r)
        cv_timings.append(dict(component=c["label"], seconds=time.time()-started))
    started = time.time()
    cv_results.append(run_cv_periodic(ctx))
    cv_timings.append(dict(component="TabM periódico", seconds=time.time()-started))
    comparison = compare_v17_v18(ctx["y"], [r["oof"] for r in cv_results])
    weights = comparison["weights"]
    sizes = [max(1, round(float(np.mean(r["its"])) * 1.15)) for r in cv_results]
    approved = comparison["improved"]
    print(pd.DataFrame(comparison["table"]).to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("Busca nas partes 0–1:", comparison["search"])
    print("Pesos:", weights, "tamanhos finais:", sizes)
else:
    assert FROZEN_RECIPE and FROZEN_RECIPE["improved"], "Validação completa necessária."
    approved = True
    weights, sizes = FROZEN_RECIPE["weights"], FROZEN_RECIPE["sizes"]
    comparison = {"improved": True, "alpha": FROZEN_RECIPE["alpha"],
                  "weights": weights, "table": FROZEN_RECIPE["metrics"],
                  "validation_recomputed_in_this_run": False}
    print("Receita fixa: AUCs registradas na validação, não recalculadas nesta reprodução.")
    print(pd.DataFrame(comparison["table"]).to_string(index=False))
    print("Pesos efetivos v18:", weights, "tamanhos finais:", sizes)
if not approved:
    print("SEM GANHO nas partes 2–4 e total. Nenhum CSV novo será gerado.")
''')

source(28, '''
output_dir = BASE_DIR / "output"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "submission.csv"
csv_sha256 = None
full_train_seconds = None
full_timings = []
if approved and not CV_ONLY:
    full_started = time.time()
    preds = []
    for c, size in zip(COMPONENTS, sizes[:4]):
        component_started = time.time()
        if c["kind"] == "nn":
            p = fit_full_nn(ctx, c["fset"], size, seeds=tuple(c["seeds"]), hp=c["hp"])
        else:
            p = fit_full(ctx, c["fset"], c["kind"], size, seeds=tuple(c["seeds"]))
        preds.append(p)
        full_timings.append(dict(component=c["label"], seconds=time.time()-component_started))
        print(c["label"], "concluído", round(time.time()-full_started), "s", flush=True)
    component_started = time.time()
    preds.append(fit_full_periodic(ctx, sizes[4]))
    full_timings.append(dict(component="TabM periódico", seconds=time.time()-component_started))
    pred = sum(w * to_rank(p) ** 2 for w, p in zip(weights, preds))
    submission = pd.DataFrame({"SK_ID_CURR": test.SK_ID_CURR.to_numpy(), "TARGET": pred})
    assert list(submission.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(submission) == 61500 and np.isfinite(pred).all()
    assert submission.TARGET.between(0, 1).all()
    submission.to_csv(output_file, index=False, float_format="%.17g", lineterminator="\\n")
    csv_sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()
    full_train_seconds = time.time()-full_started
    print("CSV:", output_file, "SHA-256:", csv_sha256)
else:
    print("Nenhum CSV foi escrito nesta execução.")
execution_report = dict(comparison=comparison, final_sizes=sizes, versions=PINNED,
                        exact_installed_versions={p: importlib.metadata.version(p) for p in PINNED},
                        imported_versions=LOADED_VERSIONS,
                        input_sha256=input_hashes, csv_sha256=csv_sha256,
                        full_train_seconds=full_train_seconds,
                        total_seconds=time.time()-notebook_wall_started,
                        cv_timings=cv_timings, full_timings=full_timings,
                        best_iterations=[r["its"] for r in cv_results],
                        tabm_history=cv_results[-1]["histories"] if cv_results else None,
                        component_oof_auc=[float(r["auc"]) for r in cv_results],
                        python=sys.version, platform=platform.platform(),
                        threads_per_model=N_JOBS, torch_interop_pool=TORCH_INTEROP_THREADS,
                        threadpools=threadpool_info(), run_cv=RUN_CV, cv_only=CV_ONLY)
(output_dir / "execution_report.json").write_text(json.dumps(execution_report, indent=2, ensure_ascii=False))
print("Tempo total desta execução:", round(execution_report["total_seconds"]), "s")
''')

s = "".join(nb["cells"][29]["source"])
s = s.replace("fornecido no pacote v16", "recuperado do ZIP v13 fornecido")
source(29, s)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        ast.parse("".join(cell["source"]))

v17 = json.loads((ROOT / "sources/gci_home_credit_v17.ipynb").read_text())


def definitions(notebook):
    out = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        for node in ast.parse(s).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert node.name not in out
                out[node.name] = ast.get_source_segment(s, node)
    return out


original = definitions(v17)
current = definitions(nb)
assert all(current.get(k) == v for k, v in original.items())
(ROOT / "gci_home_credit_v18.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Notebook criado; {len(original)} funções/classes da v17 textualmente idênticas.")
