"""One predeclared blend hypothesis; all seven v21 learners refit from input."""

V23_WEIGHTS = np.array([0.156571875, 0.005900625, 0.21094875,
                        0.17445374999999996, 0.194625, 0.15750000000000003, 0.1])
V23_ALPHAS = np.round(np.arange(0, 1.00001, 0.1), 1)


def v23_scores(predictions):
    assert predictions.ndim == 2 and predictions.shape[1] == 7
    assert np.isfinite(predictions).all()
    ranks = np.column_stack([to_rank(p)**2 for p in predictions.T])
    baseline = ranks @ V23_WEIGHTS
    clipped = np.clip(predictions, 1e-6, 1-1e-6)
    logmean = (np.log(clipped)-np.log1p(-clipped)) @ V23_WEIGHTS
    # Monotone rank transform puts both ensemble scores on the same scale.
    # At alpha=1, the AUC equals that of the weighted log-odds mean.
    alternative = to_rank(logmean)**2
    return baseline, alternative


def v23_choose(y, baseline, alternative):
    grid = [dict(alpha=float(a), auc=float(roc_auc_score(y, (1-a)*baseline+a*alternative)))
            for a in V23_ALPHAS]
    return max(grid, key=lambda r: r["auc"]), grid


def v23_run(ctx, directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    y, n = ctx["y"], ctx["n"]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(n), y))
    ids = np.empty(n, dtype=np.int8)
    for f, (_, b) in enumerate(splits):
        ids[b] = f
    selection = ids < 2
    predictions = np.full((n, 7), np.nan)
    iterations = [[] for _ in range(7)]
    report = dict(status="running_selection", confirmed=False, csv_sha256=None,
                  hypothesis="Fixed v21 weights; convex blend of current rank-squared score and rank-squared weighted log-odds score.",
                  weights=V23_WEIGHTS.tolist(), alpha_grid=V23_ALPHAS.tolist(),
                  selection_folds=[0, 1], confirmation_folds_executed=[],
                  fold_results=[], input_sha256=input_hashes,
                  exact_installed_versions={p: importlib.metadata.version(p) for p in PINNED},
                  python=sys.version, platform=platform.platform(), cv_workers=2, threads_per_model=2,
                  source_note="Refit all seven v21 components; no saved predictions, OOFs, checkpoints or models loaded. Original v17 unlabeled train+test preprocessing preserved.")

    def save():
        report["seconds"] = time.time()-started
        (directory/"experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")

    def receive(results):
        for r in results:
            f = r["fold"]
            b = splits[f][1]
            predictions[b] = np.column_stack([r["baseline"], r["predictions"][0]])
            for dst, values in zip(iterations[:6], r["baseline_iterations"]):
                dst.extend(values)
            iterations[6].append(r["candidate_sizes"][0])
            report["fold_results"].append({k: r[k] for k in ["fold", "seconds", "threads", "threadpools", "torch_interop_pool"]})
            if f >= 2:
                report["confirmation_folds_executed"].append(f)
            save()

    save()
    compact = {k: v for k, v in ctx.items() if k != "raw"}
    with joblib.parallel_config(backend="loky", n_jobs=2, inner_max_num_threads=2, initializer=v20_initialize_worker):
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v21_job)(compact, f, splits[f], [0]) for f in [0, 1]))
        base, alt = v23_scores(predictions[selection])
        chosen, grid = v23_choose(y[selection], base, alt)
        reference = float(roc_auc_score(y[selection], base))
        report.update(screening_baseline_auc=reference, screening_grid=grid, screening_choice=chosen)
        print("LOGODDS TRIAGEM", json.dumps(dict(reference=reference, chosen=chosen, grid=grid)), flush=True)
        if chosen["alpha"] == 0 or chosen["auc"] <= reference+1e-12:
            report.update(status="rejected_in_screening", reason="No gain on folds 0–1; no confirmation or CSV.")
            save()
            return report
        report["status"] = "completing_oof"
        save()
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v21_job)(compact, f, splits[f], [0]) for f in [2, 3, 4]))
    base, alt = v23_scores(predictions)
    chosen, grid = v23_choose(y[selection], base[selection], alt[selection])
    alpha = chosen["alpha"]
    report.update(status="recipe_frozen_before_confirmation", alpha=alpha, final_selection_grid=grid,
                  iterations=iterations, final_sizes=[max(1, round(float(np.mean(v))*1.15)) for v in iterations])
    (directory/"frozen_selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    save()
    new = (1-alpha)*base+alpha*alt
    masks = [(f"parte_{f}", ids == f) for f in range(5)]
    masks += [("partes_0_1", selection), ("partes_2_4", ~selection), ("total", np.ones(n, bool))]
    table = []
    for label, mask in masks:
        old_auc = float(roc_auc_score(y[mask], base[mask]))
        new_auc = float(roc_auc_score(y[mask], new[mask]))
        table.append(dict(parte=label, linhas=int(mask.sum()), v21=old_auc, v23=new_auc, delta=new_auc-old_auc))
    rows = {r["parte"]: r for r in table}
    assert abs(rows["total"]["v21"]-0.7740567649584406) < 1e-12, "v21 reference not reproduced."
    assert abs(rows["partes_2_4"]["v21"]-0.7747809092432562) < 1e-12
    confirmed = alpha > 0 and rows["partes_2_4"]["delta"] > 0 and rows["total"]["delta"] > 0
    report.update(table=table, confirmed=bool(confirmed), status="gain_confirmed" if confirmed else "rejected_in_confirmation")
    save()
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.8f}"), flush=True)
    return report
