"""Bounded v19 experiment: linear-leaf LightGBM versus the frozen v18.

The notebook builder embeds this source after the unmodified v18 definitions.
Only raw competition CSVs are read for training. No predictions are loaded.
"""

LINEAR_CONFIGS = [
    dict(label="linear_lambda_10", linear_lambda=10.0),
    dict(label="linear_lambda_100", linear_lambda=100.0),
]
V18_FIXED_WEIGHTS = np.array([0.19875, 0.03825, 0.2235, 0.2895, 0.25])
LINEAR_ALPHA_GRID = np.round(np.arange(0.0, 0.30001, 0.025), 3)


def linear_inputs(ctx, train_frame, other_frames):
    """Train-only normal quantiles, median imputation and missing indicators.

    Imputed zero is a valid numeric value, never a LightGBM missing sentinel.
    Categorical codes stay native categories and enter splits, not leaf fits.
    """
    cats = ctx["cat_cols"]
    nums = [c for c in train_frame if c not in cats and train_frame[c].nunique(dropna=True) > 1]
    a = train_frame[nums].replace([np.inf, -np.inf], np.nan)
    missing = [c for c in nums if a[c].isna().any()]
    qt = QuantileTransformer(n_quantiles=256, output_distribution="normal",
                             subsample=100000, random_state=42)
    qt.fit(a.to_numpy(dtype=np.float64))
    medians = np.nanmedian(qt.transform(a.to_numpy(dtype=np.float64)), axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    reference = ctx["X"]["plusx"]
    test_reference = ctx["Xt"]["plusx"]
    levels = {c: list(range(int(max(reference[c].max(), test_reference[c].max())) + 1)) for c in cats}

    def transform(frame):
        num = frame[nums].replace([np.inf, -np.inf], np.nan)
        z = qt.transform(num.to_numpy(dtype=np.float64))
        z = np.where(np.isnan(z), medians[None, :], z).clip(-4, 4)
        out = pd.DataFrame(z, columns=nums, index=frame.index)
        for c in missing:
            out[c + "__LINEAR_MISSING"] = num[c].isna().astype(np.float64)
        for c in cats:
            out[c] = pd.Categorical(frame[c], categories=levels[c])
        assert "SK_ID_CURR" not in out and "TARGET" not in out
        return out

    return transform(train_frame), [transform(frame) for frame in other_frames]


def linear_model(configuration, n_trees=4000):
    # Two regularization settings, fixed before inspecting any results.
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=0.03, n_estimators=int(n_trees),
        num_leaves=9, max_depth=4, min_child_samples=500,
        feature_fraction=0.44, bagging_fraction=0.74, bagging_freq=1,
        reg_lambda=2.0, reg_alpha=0.0, min_split_gain=0.05, max_bin=63,
        linear_tree=True, linear_lambda=configuration["linear_lambda"],
        deterministic=True, force_row_wise=True, n_jobs=2, random_state=42,
        verbosity=-1)


def run_linear_fold(ctx, configuration, fold, split, log=print):
    a, b = split
    X, sup, y = ctx["X"]["plusx"], ctx["sup"], ctx["y"]
    xa = sup.add(X.iloc[a], a, a)
    xb = sup.add(X.iloc[b], a, b)
    altered = sup.y_all.copy()
    altered[b] = 1 - altered[b]
    isolated = Supervised(sup.groups, sup.tkeys, altered)
    pd.testing.assert_frame_equal(xa, isolated.add(X.iloc[a], a, a))
    pd.testing.assert_frame_equal(xb, isolated.add(X.iloc[b], a, b))
    za, (zb,) = linear_inputs(ctx, xa, [xb])
    started = time.time()
    model = linear_model(configuration)
    model.fit(za, y[a], eval_set=[(zb, y[b])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = model.predict_proba(zb)[:, 1]
    log(f"LINEAR {configuration['label']} parte {fold}: "
        f"AUC {roc_auc_score(y[b], pred):.8f}, árvores {model.best_iteration_}, "
        f"{time.time()-started:.1f}s", flush=True)
    return pred, int(model.best_iteration_), time.time() - started


def v18_fold(ctx, fold, split, log=print):
    """Same learners and seeds as v18, no unused test prediction overhead."""
    a, b = split
    y, sup = ctx["y"], ctx["sup"]
    predictions, iterations = [], []
    for component in COMPONENTS:
        started = time.time()
        if component["kind"] == "nn":
            result = run_cv_nn_strict(ctx, component["fset"], seeds=tuple(component["seeds"]),
                                      hp=component["hp"], folds_to_run=[fold], log=log)
            pred, its = result["oof"][b], result["its"]
        else:
            X = ctx["X"][component["fset"]]
            xa, xb = sup.add(X.iloc[a], a, a), sup.add(X.iloc[b], a, b)
            pred, its = np.zeros(len(b)), []
            for seed in component["seeds"]:
                model = make_model(component["kind"], seed, cat_cols=ctx["cat_cols"])
                model, iteration = fit_with_es(model, component["kind"], xa, y[a], xb, y[b])
                pred += model.predict_proba(xb)[:, 1] / len(component["seeds"])
                its.append(int(iteration))
        predictions.append(pred)
        iterations.append(its)
        log(f"BASELINE parte {fold}: {component['label']}, "
            f"AUC {roc_auc_score(y[b], pred):.8f}, {time.time()-started:.1f}s", flush=True)
    X = ctx["X"]["tabm_periodic"]
    xa, xb = sup.add(X.iloc[a], a, a), sup.add(X.iloc[b], a, b)
    pair, (valid,), cards = periodic_preprocess(ctx, xa, [xb])
    model, epoch, _ = train_periodic(pair, y[a], valid, y[b], cards, 42 + 100 * fold,
                                    log=lambda message: None)
    predictions.append(periodic_predict(model, valid))
    iterations.append([int(epoch)])
    log(f"BASELINE TabM parte {fold}: AUC {roc_auc_score(y[b], predictions[-1]):.8f}, época {epoch}", flush=True)
    del model, pair, valid
    gc.collect()
    return np.column_stack(predictions), iterations


def linear_weight_search(target, squared_ranks, baseline):
    """Receives selection labels only. Exact ties choose zero/lower weight."""
    scores = []
    for alpha in LINEAR_ALPHA_GRID:
        pred = (1 - alpha) * baseline + alpha * squared_ranks
        scores.append(dict(alpha=float(alpha), auc=float(roc_auc_score(target, pred))))
    return max(scores, key=lambda row: row["auc"]), scores


def linear_comparison(y, fold_ids, predictions, linear_prediction, alpha):
    ranks = np.column_stack([to_rank(p) ** 2 for p in predictions.T])
    baseline = ranks @ V18_FIXED_WEIGHTS
    candidate = (1 - alpha) * baseline + alpha * to_rank(linear_prediction) ** 2
    masks = [(f"parte_{f}", fold_ids == f) for f in range(5)]
    masks += [("partes_0_1", fold_ids < 2), ("partes_2_4", fold_ids >= 2),
              ("total", np.ones(len(y), bool))]
    table = []
    for label, mask in masks:
        old = float(roc_auc_score(y[mask], baseline[mask]))
        new = float(roc_auc_score(y[mask], candidate[mask]))
        table.append(dict(parte=label, linhas=int(mask.sum()), v18=old, v19=new, delta=new-old))
    by_name = {row["parte"]: row for row in table}
    improved = by_name["partes_2_4"]["delta"] > 0 and by_name["total"]["delta"] > 0
    return table, bool(improved)


def run_v19_experiment(ctx, report_directory):
    report_directory.mkdir(parents=True, exist_ok=True)
    path = report_directory / "experiment_report.json"
    started = time.time()
    y = ctx["y"]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_ids = np.empty(len(y), dtype=np.int8)
    for fold, (_, b) in enumerate(splits):
        fold_ids[b] = fold
    predictions = np.full((len(y), 5), np.nan)
    linear_predictions = [np.full(len(y), np.nan) for _ in LINEAR_CONFIGS]
    base_iterations = [[] for _ in range(5)]
    linear_iterations = [[] for _ in LINEAR_CONFIGS]
    report = dict(status="running", input_sha256=input_hashes, versions=PINNED,
                  python=sys.version, platform=platform.platform(), threads=2,
                  configurations=LINEAR_CONFIGS, alpha_grid=LINEAR_ALPHA_GRID.tolist(),
                  baseline_weights=V18_FIXED_WEIGHTS.tolist(), screening=[], confirmed=False,
                  csv_sha256=None, source_note="All models refit from input CSVs in this process.")

    def save():
        report["seconds"] = time.time() - started
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    # Inexpensive model fits first. Baseline is refit once, reused in RAM only.
    for fold in [0, 1]:
        a, b = splits[fold]
        for index, configuration in enumerate(LINEAR_CONFIGS):
            p, iteration, _ = run_linear_fold(ctx, configuration, fold, splits[fold])
            linear_predictions[index][b] = p
            linear_iterations[index].append(iteration)
        predictions[b], iterations = v18_fold(ctx, fold, splits[fold])
        for dest, source in zip(base_iterations, iterations):
            dest.extend(source)
        report["last_completed_fold"] = fold
        save()
    selection = fold_ids < 2
    # Screening ranks use only available rows. Final selection, if reached,
    # uses the exact global percentiles of the v18 comparison protocol.
    ranks = np.column_stack([to_rank(p) ** 2 for p in predictions[selection].T])
    baseline = ranks @ V18_FIXED_WEIGHTS
    for index, configuration in enumerate(LINEAR_CONFIGS):
        best, grid = linear_weight_search(y[selection], to_rank(linear_predictions[index][selection]) ** 2, baseline)
        row = dict(configuration=configuration, best=best, grid=grid,
                   baseline_auc=float(roc_auc_score(y[selection], baseline)),
                   delta=float(best["auc"] - roc_auc_score(y[selection], baseline)))
        report["screening"].append(row)
        print("SCREEN", json.dumps(row), flush=True)
    qualified = [i for i, r in enumerate(report["screening"]) if r["delta"] > 0 and r["best"]["alpha"] > 0]
    report["screen_table"] = []
    for fold in [0, 1]:
        mask = fold_ids[selection] == fold
        old = float(roc_auc_score(y[selection][mask], baseline[mask]))
        for index, row in enumerate(report["screening"]):
            p = (1-row["best"]["alpha"]) * baseline + row["best"]["alpha"] * to_rank(linear_predictions[index][selection])**2
            score = float(roc_auc_score(y[selection][mask], p[mask]))
            report["screen_table"].append(dict(parte=fold, configuration=LINEAR_CONFIGS[index]["label"], v18=old, v19=score, delta=score-old))
    if not qualified:
        report.update(status="rejected_in_screening", reason="Neither configuration improves the v18 ensemble on folds 0–1; folds 2–4 not run.")
        save()
        return report
    report["qualified_configurations"] = qualified
    report["status"] = "completing_oof_without_inspecting_confirmation_auc"
    save()
    for fold in [2, 3, 4]:
        a, b = splits[fold]
        for index in qualified:
            p, iteration, _ = run_linear_fold(ctx, LINEAR_CONFIGS[index], fold, splits[fold])
            linear_predictions[index][b] = p
            linear_iterations[index].append(iteration)
        predictions[b], iterations = v18_fold(ctx, fold, splits[fold], log=lambda *a, **k: None)
        for dest, source in zip(base_iterations, iterations):
            dest.extend(source)
        report["last_completed_fold"] = fold
        save()
        print(f"OOF parte {fold} concluída; AUC do ensemble de confirmação ainda não consultada.", flush=True)
    ranks = np.column_stack([to_rank(p) ** 2 for p in predictions.T])
    baseline = ranks @ V18_FIXED_WEIGHTS
    searches = []
    for index in qualified:
        best, grid = linear_weight_search(y[selection], (to_rank(linear_predictions[index]) ** 2)[selection], baseline[selection])
        searches.append(dict(index=index, best=best, grid=grid))
    chosen = max(searches, key=lambda row: row["best"]["auc"])
    alpha = chosen["best"]["alpha"]
    index = chosen["index"]
    # Freeze before first evaluation of the final ensemble on folds 2–4.
    report.update(status="recipe_frozen_before_confirmation", selection_searches=searches,
                  configuration=LINEAR_CONFIGS[index], alpha=alpha,
                  weights=np.r_[(1-alpha)*V18_FIXED_WEIGHTS, alpha].tolist(),
                  baseline_sizes=[max(1, round(float(np.mean(v))*1.15)) for v in base_iterations],
                  linear_size=max(1, round(float(np.mean(linear_iterations[index]))*1.15)),
                  baseline_iterations=base_iterations, linear_iterations=linear_iterations[index])
    save()
    table, improved = linear_comparison(y, fold_ids, predictions, linear_predictions[index], alpha)
    report.update(table=table, confirmed=improved, status="gain_confirmed" if improved else "rejected_in_confirmation",
                  threadpools=threadpool_info())
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda x: f"{x:.8f}"), flush=True)
    save()
    return report
