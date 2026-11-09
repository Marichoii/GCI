"""One frozen PyTorch factorization machine; selection labels 0–1 only.

All new supervised preprocessing and epoch selection exclude the outer
evaluation labels. Historical v21 learners are refitted without modifications.
Saved predictions are evidence only: this module never loads any prediction.
"""
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr

V24_WEIGHTS = np.array([0.156571875, 0.005900625, 0.21094875,
                        0.17445374999999996, 0.194625, 0.15750000000000003, 0.1])
V24_LABELS = ["LightGBM B + EXT", "LightGBM B v7", "MLP 8 sementes",
              "CatBoost Plain + EXT", "TabM periódico", "TabM por pares",
              "CatBoost Ordered depth 5"]
V24_PROTOCOL = dict(
    version="v24", hypothesis="PyTorch second-order factorization machine",
    deadline="2026-11-14", seed=42, threads_per_model=2, cv_workers=2,
    folds=dict(n_splits=5, shuffle=True, random_state=42),
    selection_folds=[0, 1], confirmation_folds=[2, 3, 4],
    alpha_grid=[0.0, 0.05, 0.10, 0.15, 0.20], tie_break="smaller_alpha",
    baseline_weights=V24_WEIGHTS.tolist(),
    baseline_final_sizes=[1646, 1793, 11, 1103, 14, 11, 1089],
    fm=dict(feature_set="plusx", rank=16, linear=True, loss="unweighted_BCE",
            optimizer="AdamW", lr=0.002, weight_decay=0.001, batch_size=2048,
            max_epochs=40, patience=6, min_delta=0.000001,
            inner_fraction=0.15, inner_random_state=42,
            factor_init_std=0.01, linear_init="zero",
            bias_init="logit_of_training_prevalence", model_seed=42,
            quantiles=256, quantile_distribution="normal", clip=4.0,
            categorical="dense_one_hot_unknown_ignore",
            missing="zero_numeric_plus_training_missing_indicators",
            constant_columns="drop_by_training_only",
            epoch_selection="inner_AUC_then_refit_all_outer_training_from_scratch",
            final_epochs="rounded_mean_of_outer_inner_selected_epochs_no_multiplier"),
    strata=dict(key="round(AMT_CREDIT / AMT_ANNUITY, 2)", decimals=2,
                count_source="outer_training_only_excluding_evaluated_fold",
                bins=[[0, 9], [10, 49], [50, 199], [200, None]],
                names=["raras_0_9", "pouco_frequentes_10_49", "comuns_50_199", "muito_frequentes_200_mais"],
                invalid="missing_nonfinite_or_nonpositive_credit_or_annuity",
                use="diagnostics_only_never_feature_weight_or_prediction_rule"),
    rank_procedure="screen_on_0_1_ranks_then_if_pass_reselect_same_grid_with_global_ranks_using_only_labels_0_1",
    screening="stop_if_best_alpha_zero_or_no_positive_gain_on_0_1",
    promotion=dict(min_delta_2_4=0.00010, total_delta_strictly_positive=True,
                   reference_2_4=0.7747809092432562, reference_total=0.7740567649584406),
    final_limit_seconds=3600, no_prediction_or_model_loading=True,
    note="Inherited v21 unlabeled train+test preprocessing and v21 early stopping are preserved for exact comparison. New FM epoch selection uses an inner split.")


def v24_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def v24_frozen_protocol(directory):
    """Write before any model fit; a rerun must use identical definitions."""
    path = Path(directory)/"frozen_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == V24_PROTOCOL, "Protocol changed after freezing."
    else:
        v24_json(path, V24_PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v24_strata(train, splits):
    credit = train.AMT_CREDIT.to_numpy(dtype=float)
    annuity = train.AMT_ANNUITY.to_numpy(dtype=float)
    valid = np.isfinite(credit) & np.isfinite(annuity) & (credit > 0) & (annuity > 0)
    ratio = np.full(len(train), np.nan)
    ratio[valid] = np.round(credit[valid]/annuity[valid], 2)
    valid &= np.isfinite(ratio)
    frequency = np.full(len(train), -1, dtype=np.int64)
    labels = np.full(len(train), "ausentes_invalidas", dtype=object)
    for a, b in splits:
        counts = pd.Series(ratio[a][valid[a]]).value_counts()
        dst = b[valid[b]]
        frequency[dst] = pd.Series(ratio[dst]).map(counts).fillna(0).to_numpy(dtype=np.int64)
        for (lo, hi), name in zip(V24_PROTOCOL["strata"]["bins"], V24_PROTOCOL["strata"]["names"]):
            chosen = dst[frequency[dst] >= lo]
            if hi is not None:
                chosen = chosen[frequency[chosen] <= hi]
            labels[chosen] = name
    return labels, frequency


def v24_preprocess(training, others, cats):
    """Every learned transformation fits training only, including inner ES."""
    forbidden = [c for c in training if c == "TARGET" or "SK_ID_CURR" in c]
    assert not forbidden, forbidden
    nums = [c for c in training if c not in cats and training[c].nunique(dropna=True) > 1]
    continuous = [c for c in nums if training[c].nunique(dropna=True) > 2]
    binary = [c for c in nums if c not in continuous]
    missing = [c for c in nums if training[c].isna().any()]
    qt = QuantileTransformer(n_quantiles=min(256, len(training)), output_distribution="normal",
                             subsample=10**9, random_state=42)
    qt.fit(training[continuous].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    enc.fit(training[cats].to_numpy())

    def transform(frame):
        z = qt.transform(frame[continuous].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float))
        parts = [np.nan_to_num(z, nan=0.0, posinf=4.0, neginf=-4.0).clip(-4, 4).astype(np.float32)]
        if binary:
            parts.append(frame[binary].fillna(0).to_numpy(dtype=np.float32))
        if missing:
            parts.append(frame[missing].isna().to_numpy(dtype=np.float32))
        parts.append(enc.transform(frame[cats].to_numpy()))
        result = np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)
        assert np.isfinite(result).all()
        return torch.from_numpy(result)

    metadata = dict(numeric=nums, continuous=continuous, binary=binary, missing=missing,
                    category_sizes=[len(c) for c in enc.categories_], fit_rows=len(training))
    return transform(training), [transform(x) for x in others], metadata


class V24FactorizationMachine(nn.Module):
    def __init__(self, dimensions, rank, prevalence):
        super().__init__()
        self.linear = nn.Linear(dimensions, 1)
        self.factors = nn.Parameter(torch.empty(dimensions, rank))
        nn.init.normal_(self.factors, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(np.log(prevalence/(1-prevalence))))

    def forward(self, x):
        interaction = 0.5*((x @ self.factors).square() - x.square() @ self.factors.square()).sum(dim=1)
        return self.linear(x).squeeze(-1) + interaction


def v24_predict(model, data):
    model.eval()
    with torch.no_grad():
        # Logits have the same ranking as sigmoid and avoid numerical ties.
        return torch.cat([model(x) for x in data.split(4096)]).numpy()


def v24_train(data, target, valid=None, y_valid=None, fixed_epochs=None):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = V24_PROTOCOL["fm"]
    model = V24FactorizationMachine(data.shape[1], cfg["rank"], float(np.mean(target)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    target = torch.as_tensor(target, dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    best, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(1, (fixed_epochs or cfg["max_epochs"])+1):
        model.train()
        for ids in torch.randperm(len(target), generator=generator).split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(data[ids]), target[ids])
            loss.backward()
            optimizer.step()
        if fixed_epochs is None:
            auc = float(roc_auc_score(y_valid, v24_predict(model, valid)))
            history.append(dict(epoch=epoch, auc=auc))
            print(f"FM inner época {epoch}: AUC {auc:.8f}", flush=True)
            if auc > best + cfg["min_delta"]:
                best, best_epoch, stale = auc, epoch, 0
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
    # No best checkpoint is needed: every outer/full fit is a new model.
    return model, int(fixed_epochs or best_epoch), history


def v24_frames(ctx, src, dst, forbidden_labels):
    X, sup = ctx["X"]["plusx"], ctx["sup"]
    assert not np.intersect1d(src, forbidden_labels).size
    a = sup.add(X.iloc[src], src, src)
    b = sup.add(X.iloc[dst], src, dst)
    altered = sup.y_all.copy()
    altered[forbidden_labels] = 1-altered[forbidden_labels]
    isolated = Supervised(sup.groups, sup.tkeys, altered)
    pd.testing.assert_frame_equal(a, isolated.add(X.iloc[src], src, src))
    pd.testing.assert_frame_equal(b, isolated.add(X.iloc[dst], src, dst))
    return a, b


def v24_fm_fold(ctx, fold, split):
    a, b = split
    started = time.time()
    ia, ib = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                  .split(np.zeros(len(a)), ctx["y"][a]))
    inner_train, inner_valid = a[ia], a[ib]
    xa, xv = v24_frames(ctx, inner_train, inner_valid, np.r_[inner_valid, b])
    data, (valid,), inner_meta = v24_preprocess(xa, [xv], ctx["cat_cols"])
    model, epochs, history = v24_train(data, ctx["y"][inner_train], valid, ctx["y"][inner_valid])
    del model, data, valid, xa, xv
    gc.collect()
    xa, xb = v24_frames(ctx, a, b, b)
    data, (valid,), outer_meta = v24_preprocess(xa, [xb], ctx["cat_cols"])
    model, _, _ = v24_train(data, ctx["y"][a], fixed_epochs=epochs)
    prediction = v24_predict(model, valid)
    details = dict(fold=fold, epochs=epochs, inner_history=history, inner_preprocessing=inner_meta,
                   outer_preprocessing=outer_meta, dimensions=int(data.shape[1]),
                   target_flip_audit=True, seconds=time.time()-started,
                   epoch_selection_uses_outer_evaluation_labels=False)
    print(f"FM parte {fold}: AUC {roc_auc_score(ctx['y'][b], prediction):.8f}, épocas {epochs}", flush=True)
    del model, data, valid, xa, xb
    gc.collect()
    return prediction, details


def v24_job(ctx, fold, split):
    started = time.time()
    print(f"V24 parte {fold} iniciada", flush=True)
    # Fit FM first; all baseline learners reset their own streams thereafter.
    fm, fm_details = v24_fm_fold(ctx, fold, split)
    reference = v21_job(ctx, fold, split, [0])
    reference.update(fm=fm, fm_details=fm_details, seconds=time.time()-started)
    return reference


def v24_scores(predictions, fm):
    ranks = np.column_stack([to_rank(p)**2 for p in predictions.T])
    return ranks @ V24_WEIGHTS, to_rank(fm)**2


def v24_choose(target, baseline, fm):
    # This function only receives selection labels; exact ties prefer less FM.
    grid = [dict(alpha=alpha, auc=float(roc_auc_score(target, (1-alpha)*baseline+alpha*fm)))
            for alpha in V24_PROTOCOL["alpha_grid"]]
    return max(grid, key=lambda row: row["auc"]), grid


def v24_auc(target, prediction):
    return float(roc_auc_score(target, prediction)) if len(np.unique(target)) == 2 else None


def v24_diagnostics(y, components, fm, baseline, candidate, strata, fold_ids):
    masks = [(f"parte_{f}", fold_ids == f) for f in sorted(np.unique(fold_ids))]
    masks.append(("partes_0_1", fold_ids < 2))
    if (fold_ids >= 2).any():
        masks += [("partes_2_4", fold_ids >= 2), ("total", np.ones(len(y), dtype=bool))]
    table, correlations, strata_table = [], [], []
    names = V24_PROTOCOL["strata"]["names"]+["ausentes_invalidas"]
    for label, mask in masks:
        if not mask.any():
            continue
        old, new = v24_auc(y[mask], baseline[mask]), v24_auc(y[mask], candidate[mask])
        table.append(dict(parte=label, linhas=int(mask.sum()), v21=old, v24=new,
                          delta=None if old is None or new is None else new-old,
                          fm=v24_auc(y[mask], fm[mask])))
        for index, component in enumerate(V24_LABELS):
            rho = float(spearmanr(fm[mask], components[mask, index]).statistic)
            correlations.append(dict(parte=label, componente=component,
                                     spearman=rho if np.isfinite(rho) else None))
        for name in names:
            chosen = mask & (strata == name)
            old, new = v24_auc(y[chosen], baseline[chosen]), v24_auc(y[chosen], candidate[chosen])
            strata_table.append(dict(parte=label, estrato=name, linhas=int(chosen.sum()),
                                     percentual_linhas=100*float(chosen.sum())/int(mask.sum()),
                                     positivos=int(y[chosen].sum()), v21=old, v24=new,
                                     delta=None if old is None or new is None else new-old,
                                     fm=v24_auc(y[chosen], fm[chosen])))
    return dict(table=table, rank_correlations=correlations, strata_table=strata_table)


def v24_run(ctx, train, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/"experiment_report.json").exists(), "Use a new research directory; preserve prior evidence."
    protocol_sha = v24_frozen_protocol(directory)
    started = time.time()
    y, n = ctx["y"], ctx["n"]
    splits = list(StratifiedKFold(**V24_PROTOCOL["folds"]).split(np.zeros(n), y))
    ids = np.empty(n, dtype=np.int8)
    for f, (_, b) in enumerate(splits):
        ids[b] = f
    strata, frequency = v24_strata(train, splits)
    # Assignment is frozen before the first fit and contains no outcomes.
    assignment = pd.DataFrame(dict(row=np.arange(n), fold=ids, outer_train_frequency=frequency,
                                   stratum=strata))
    assignment_path = directory/"frozen_strata.csv"
    if assignment_path.exists():
        pd.testing.assert_frame_equal(assignment, pd.read_csv(assignment_path), check_dtype=False)
    else:
        assignment.to_csv(assignment_path, index=False)
    selection = ids < 2
    predictions, fm = np.full((n, 7), np.nan), np.full(n, np.nan)
    iterations, fm_epochs = [[] for _ in range(7)], []
    report = dict(status="running_selection", confirmed=False, csv_sha256=None,
                  protocol=V24_PROTOCOL, protocol_sha256=protocol_sha,
                  frozen_strata_sha256=hashlib.sha256((directory/"frozen_strata.csv").read_bytes()).hexdigest(),
                  input_sha256=input_hashes, exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
                  python=sys.version, platform=platform.platform(), fold_results=[],
                  confirmation_folds_executed=[], selection_folds=[0, 1],
                  source_note="All seven v21 components and FM refit from input; no saved predictions or models loaded.")

    def save():
        report["seconds"] = time.time()-started
        v24_json(directory/"experiment_report.json", report)

    def receive(results):
        for result in results:
            f = result["fold"]
            b = splits[f][1]
            predictions[b] = np.column_stack([result["baseline"], result["predictions"][0]])
            fm[b] = result["fm"]
            for dest, values in zip(iterations[:6], result["baseline_iterations"]):
                dest.extend(values)
            iterations[6].append(result["candidate_sizes"][0])
            fm_epochs.append(result["fm_details"]["epochs"])
            report["fold_results"].append({k:result[k] for k in ["fold", "seconds", "threads", "threadpools", "torch_interop_pool", "fm_details"]})
            if f >= 2:
                report["confirmation_folds_executed"].append(f)
            save()

    save()
    compact = {k:v for k,v in ctx.items() if k != "raw"}
    with joblib.parallel_config(backend="loky", n_jobs=2, inner_max_num_threads=2, initializer=v20_initialize_worker):
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v24_job)(compact, f, splits[f]) for f in [0, 1]))
        base, alternative = v24_scores(predictions[selection], fm[selection])
        chosen, grid = v24_choose(y[selection], base, alternative)
        alpha = chosen["alpha"]
        report.update(screening_choice=chosen, screening_grid=grid,
                      screening_baseline_auc=float(roc_auc_score(y[selection], base)),
                      screening_diagnostics=v24_diagnostics(y[selection], predictions[selection], fm[selection],
                          base, (1-alpha)*base+alpha*alternative, strata[selection], ids[selection]))
        v24_json(directory/"screening_selection.json", report)
        print("V24 TRIAGEM", json.dumps(dict(choice=chosen, grid=grid)), flush=True)
        if alpha == 0 or chosen["auc"] <= report["screening_baseline_auc"]:
            report.update(status="rejected_in_screening", reason="Best FM weight is zero or no gain on 0–1; no 2–4 fits, full training, or CSV.",
                          evaluated_weights=np.r_[(1-alpha)*V24_WEIGHTS, alpha].tolist())
            report.update(report["screening_diagnostics"])
            np.savez_compressed(directory/"screening_oof_evidence.npz", rows=np.flatnonzero(selection),
                                fold=ids[selection], v21_components=predictions[selection], fm=fm[selection])
            save()
            return report
        report["status"] = "completing_oof"
        save()
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v24_job)(compact, f, splits[f]) for f in [2, 3, 4]))
    base, alternative = v24_scores(predictions, fm)
    chosen, grid = v24_choose(y[selection], base[selection], alternative[selection])
    alpha = chosen["alpha"]
    report.update(status="recipe_frozen_before_confirmation", alpha=alpha, final_selection_grid=grid,
                  weights=np.r_[(1-alpha)*V24_WEIGHTS, alpha].tolist(),
                  baseline_final_sizes=V24_PROTOCOL["baseline_final_sizes"],
                  fm_final_epochs=max(1, round(float(np.mean(fm_epochs)))),
                  baseline_iterations=iterations, fm_inner_selected_epochs=fm_epochs)
    v24_json(directory/"frozen_selection.json", report)
    save()
    report.update(v24_diagnostics(y, predictions, fm, base, (1-alpha)*base+alpha*alternative, strata, ids))
    rows = {row["parte"]:row for row in report["table"]}
    assert abs(rows["partes_2_4"]["v21"]-V24_PROTOCOL["promotion"]["reference_2_4"]) < 1e-12, "v21 not reproduced."
    assert abs(rows["total"]["v21"]-V24_PROTOCOL["promotion"]["reference_total"]) < 1e-12, "v21 not reproduced."
    approved = alpha > 0 and rows["partes_2_4"]["delta"] >= 0.00010 and rows["total"]["delta"] > 0
    report.update(confirmed=bool(approved), status="gain_confirmed" if approved else "rejected_in_confirmation")
    np.savez_compressed(directory/"oof_evidence.npz", fold=ids, v21_components=predictions, fm=fm)
    save()
    print(pd.DataFrame(report["table"]).to_string(index=False), flush=True)
    return report


def v24_fit_full_fm(ctx, epochs):
    a = np.arange(ctx["n"])
    b = np.arange(ctx["n"], ctx["n"]+len(ctx["Xt"]["plusx"]))
    xa = ctx["sup"].add(ctx["X"]["plusx"], a, a)
    xt = ctx["sup"].add(ctx["Xt"]["plusx"], a, b)
    data, (test_data,), _ = v24_preprocess(xa, [xt], ctx["cat_cols"])
    model, _, _ = v24_train(data, ctx["y"], fixed_epochs=int(epochs))
    return v24_predict(model, test_data)
