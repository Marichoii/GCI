"""v20: bounded new-model search and rebalancing, selection labels 0–1 only."""
import joblib

V20_BASE_WEIGHTS = np.array([0.19875, 0.03825, 0.22349999999999998, 0.2895, 0.25])
V20_CANDIDATES = [
    dict(label="lgb_bins255", kind="lgb_resolution", max_bin=255),
    dict(label="lgb_bins1023", kind="lgb_resolution", max_bin=1023),
    dict(label="tabm_pair_half", kind="tabm_pairwise", pair_weight=0.5),
    dict(label="tabm_pair_full", kind="tabm_pairwise", pair_weight=1.0),
]
V20_CV_WORKERS = 2


def v20_initialize_worker():
    import os
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = "2"
    # Import before deserializing LightGBM functions: supplies libomp on macOS.
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 2:
        torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_limits
    global _v20_worker_limit
    _v20_worker_limit = threadpool_limits(limits=2)


def v20_train_pairwise(pair, target, valid, y_valid, cards, seed, pair_weight, n_epochs_fixed=None):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = periodic_model(pair[0].shape[1], cards)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    labels = torch.tensor(target, dtype=torch.float32)
    # Pair sampling has its own fixed stream; it does not change dropout/shuffle.
    generator = torch.Generator().manual_seed(seed + 8000)
    best, best_epoch, stale, state = -np.inf, 0, 0, None
    history = []
    for epoch in range(1, (n_epochs_fixed or 40) + 1):
        model.train()
        for ids in torch.randperm(len(labels)).split(1024):
            optimizer.zero_grad(set_to_none=True)
            logits = model(pair[0][ids], pair[1][ids]).squeeze(-1)
            batch_y = labels[ids]
            positive = torch.where(batch_y == 1)[0]
            negative = torch.where(batch_y == 0)[0]
            bce = nn.functional.binary_cross_entropy_with_logits(logits, batch_y[:, None].expand_as(logits))
            if len(positive) and len(negative):
                pi = positive[torch.randint(len(positive), (4096,), generator=generator)]
                ni = negative[torch.randint(len(negative), (4096,), generator=generator)]
                pair_loss = nn.functional.softplus(logits[ni] - logits[pi]).mean()
                loss = (1 - pair_weight) * bce + pair_weight * pair_loss
            else:
                # No valid pairs in a rare last batch: no update of the model.
                continue
            loss.backward()
            optimizer.step()
        if n_epochs_fixed is None:
            auc = float(roc_auc_score(y_valid, periodic_predict(model, valid)))
            scheduler.step(auc)
            history.append(dict(epoch=epoch, auc=auc, lr=float(optimizer.param_groups[0]["lr"])))
            if auc > best + 1e-6:
                best, best_epoch, stale = auc, epoch, 0
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 7:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, int(best_epoch), history


def v20_candidate_fold(ctx, fold, split, configuration):
    a, b = split
    fset = "plusx" if configuration["kind"] == "lgb_resolution" else "tabm_periodic"
    X, sup, y = ctx["X"][fset], ctx["sup"], ctx["y"]
    xa, xb = sup.add(X.iloc[a], a, a), sup.add(X.iloc[b], a, b)
    altered = sup.y_all.copy()
    altered[b] = 1 - altered[b]
    isolated = Supervised(sup.groups, sup.tkeys, altered)
    pd.testing.assert_frame_equal(xa, isolated.add(X.iloc[a], a, a))
    pd.testing.assert_frame_equal(xb, isolated.add(X.iloc[b], a, b))
    started = time.time()
    if configuration["kind"] == "lgb_resolution":
        parameters = dict(LGB_PARAMS["B"], max_bin=configuration["max_bin"])
        model = lgb.LGBMClassifier(**parameters, n_estimators=10000, random_state=42)
        model, iteration = fit_with_es(model, "B", xa, y[a], xb, y[b])
        p = model.predict_proba(xb)[:, 1]
    else:
        pair, (valid,), cards = periodic_preprocess(ctx, xa, [xb])
        model, iteration, _ = v20_train_pairwise(pair, y[a], valid, y[b], cards,
                                                 42 + 100 * fold, configuration["pair_weight"])
        p = periodic_predict(model, valid)
    print(f"NOVO parte {fold}: {configuration['label']}, AUC {roc_auc_score(y[b], p):.8f}, "
          f"tamanho {iteration}, {time.time()-started:.1f}s", flush=True)
    return p, int(iteration)


def v20_fold_job(ctx, fold, split, candidate_indices):
    # All returned predictions were fitted in this job; nothing read from disk.
    started = time.time()
    print(f"PARTE {fold} iniciada, candidatos {candidate_indices}", flush=True)
    baseline, iterations = v20_baseline_fold(ctx, fold, split)
    predictions, candidate_iterations = {}, {}
    for index in candidate_indices:
        predictions[index], candidate_iterations[index] = v20_candidate_fold(ctx, fold, split, V20_CANDIDATES[index])
    print(f"PARTE {fold} concluída, {time.time()-started:.1f}s", flush=True)
    return dict(fold=fold, baseline=baseline, iterations=iterations, candidates=predictions,
                candidate_iterations=candidate_iterations, seconds=time.time()-started,
                threads=2, torch_interop_pool=torch.get_num_interop_threads(), threadpools=threadpool_info())


def v20_weights(y_selection, squared_ranks, has_candidate):
    # Original proportions are a starting point; mass transfers stay within
    # +/-0.10 of each original weight. No regression/stacking model is fitted.
    reference = np.r_[V20_BASE_WEIGHTS, 0.] if has_candidate else V20_BASE_WEIGHTS.copy()
    lower = np.maximum(reference - 0.10, 0)
    upper = reference + 0.10
    if has_candidate:
        upper[-1] = 0.30
    starts = [reference]
    if has_candidate:
        starts += [np.r_[(1-alpha)*V20_BASE_WEIGHTS, alpha] for alpha in np.arange(0.025, 0.30001, 0.025)]
    evaluations = 0

    def evaluate(weights):
        nonlocal evaluations
        evaluations += 1
        return float(roc_auc_score(y_selection, squared_ranks @ weights))

    scores = [evaluate(w) for w in starts]
    best_index = int(np.argmax(scores))
    weights, score = starts[best_index].copy(), scores[best_index]
    for step in [0.025, 0.01]:
        for _ in range(4):
            move, next_score = None, score
            for source in range(len(weights)):
                for destination in range(len(weights)):
                    if source == destination or weights[source]-step < lower[source]-1e-12 or weights[destination]+step > upper[destination]+1e-12:
                        continue
                    proposed = weights.copy()
                    proposed[source] -= step
                    proposed[destination] += step
                    value = evaluate(proposed)
                    if value > next_score + 1e-12:
                        next_score, move = value, proposed
            if move is None:
                break
            weights, score = move, next_score
    weights[np.abs(weights) < 1e-12] = 0
    assert np.all(weights >= 0) and abs(weights.sum()-1) < 1e-10
    return dict(weights=weights.tolist(), auc=score, evaluations=evaluations)


def v20_run(ctx, directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    y, n = ctx["y"], ctx["n"]
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    fold_ids = np.empty(n, dtype=np.int8)
    for fold, (_, b) in enumerate(splits):
        fold_ids[b] = fold
    selection = fold_ids < 2
    baseline_oof = np.full((n, 5), np.nan)
    candidate_oof = np.full((n, len(V20_CANDIDATES)), np.nan)
    iterations = [[] for _ in range(5)]
    candidate_iterations = [[] for _ in V20_CANDIDATES]
    report = dict(status="running", confirmed=False, csv_sha256=None, input_sha256=input_hashes,
                  configurations=V20_CANDIDATES, versions=PINNED, python=sys.version, platform=platform.platform(),
                  cv_workers=V20_CV_WORKERS, threads_per_model=2, fold_results=[],
                  baseline_weights=V20_BASE_WEIGHTS.tolist(), selection_folds=[0,1], confirmation_folds=[2,3,4])

    def save():
        report["seconds"] = time.time()-started
        (directory / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def receive(results):
        for result in results:
            fold = result["fold"]
            b = splits[fold][1]
            baseline_oof[b] = result["baseline"]
            for dst, values in zip(iterations, result["iterations"]):
                dst.extend(values)
            for index, values in result["candidates"].items():
                candidate_oof[b, index] = values
                candidate_iterations[index].append(result["candidate_iterations"][index])
            report["fold_results"].append({k: result[k] for k in ["fold", "seconds", "threads", "torch_interop_pool", "threadpools"]})
            save()

    save()
    # Two independent clean worker processes, two threads per fitted model.
    compact_ctx = {k:v for k,v in ctx.items() if k != "raw"}
    with joblib.parallel_config(backend="loky", n_jobs=V20_CV_WORKERS, inner_max_num_threads=2,
                                initializer=v20_initialize_worker):
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v20_fold_job)(compact_ctx, fold, splits[fold], list(range(4)))
                                                 for fold in [0,1]))
        ranks = np.column_stack([to_rank(p)**2 for p in baseline_oof[selection].T])
        base_score = float(roc_auc_score(y[selection], ranks @ V20_BASE_WEIGHTS))
        searches = [dict(candidate=None, **v20_weights(y[selection], ranks, False))]
        for index in range(len(V20_CANDIDATES)):
            r = np.column_stack([ranks, to_rank(candidate_oof[selection,index])**2])
            searches.append(dict(candidate=index, **v20_weights(y[selection], r, True)))
        chosen = max(searches, key=lambda row:row["auc"])
        report.update(screening=searches, screening_baseline_auc=base_score, chosen_family=chosen["candidate"])
        print("TRIAGEM", json.dumps(dict(baseline=base_score, candidates=searches)), flush=True)
        if chosen["auc"] <= base_score + 1e-12:
            report.update(status="rejected_in_screening", reason="No strategy improves folds 0–1; no new CSV.")
            save()
            return report
        selected_candidate = chosen["candidate"]
        qualified = [] if selected_candidate is None else [selected_candidate]
        report["status"] = "completing_oof"
        save()
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v20_fold_job)(compact_ctx, fold, splits[fold], qualified)
                                                 for fold in [2,3,4]))
    # Ranks now use all OOF rows, exactly as v18. Only labels 0–1 tune weights.
    ranks = np.column_stack([to_rank(p)**2 for p in baseline_oof.T])
    candidate_ranks = ranks if selected_candidate is None else np.column_stack([ranks, to_rank(candidate_oof[:,selected_candidate])**2])
    frozen = v20_weights(y[selection], candidate_ranks[selection], selected_candidate is not None)
    report.update(status="recipe_frozen_before_confirmation", weights=frozen["weights"], selection_auc=frozen["auc"],
                  selected_candidate=None if selected_candidate is None else V20_CANDIDATES[selected_candidate],
                  baseline_iterations=iterations, baseline_sizes=[max(1,round(float(np.mean(it))*1.15)) for it in iterations],
                  candidate_iterations=None if selected_candidate is None else candidate_iterations[selected_candidate],
                  candidate_size=None if selected_candidate is None else max(1,round(float(np.mean(candidate_iterations[selected_candidate]))*1.15)))
    (directory / "frozen_selection.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    save()
    base, new = ranks @ V20_BASE_WEIGHTS, candidate_ranks @ np.array(frozen["weights"])
    masks = [(f"parte_{fold}", fold_ids==fold) for fold in range(5)]
    masks += [("partes_0_1",selection),("partes_2_4",~selection),("total",np.ones(n,bool))]
    table = []
    for label, mask in masks:
        a, b = float(roc_auc_score(y[mask],base[mask])),float(roc_auc_score(y[mask],new[mask]))
        table.append(dict(parte=label,linhas=int(mask.sum()),v18=a,v20=b,delta=b-a))
    by_name = {r["parte"]:r for r in table}
    confirmed = by_name["partes_2_4"]["delta"]>0 and by_name["total"]["delta"]>0
    report.update(table=table,confirmed=bool(confirmed),status="gain_confirmed" if confirmed else "rejected_in_confirmation")
    save()
    print(pd.DataFrame(table).to_string(index=False,float_format=lambda v:f"{v:.8f}"),flush=True)
    return report
