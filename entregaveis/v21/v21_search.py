"""Predeclared v21 candidates; train from input, select on folds 0–1 only."""

V21_WEIGHTS = np.array([0.17396875, 0.00655625, 0.23438749999999997,
                        0.19383749999999997, 0.21625, 0.17500000000000002])
V21_CONFIGS = [
    dict(label="cat_ordered_depth5", kind="cat_ordered", depth=5),
    dict(label="cat_ordered_depth6", kind="cat_ordered", depth=6),
    dict(label="lgb_extra_trees", kind="lgb_extra"),
]
V21_ALPHAS = np.round(np.arange(0, 0.30001, 0.025), 3)


def v21_candidate_model(configuration, size=None, cat_cols=None):
    if configuration["kind"] == "cat_ordered":
        parameters = dict(CAT_PARAMS, depth=configuration["depth"], boosting_type="Ordered",
                          random_seed=42, cat_features=cat_cols)
        if size is None:
            parameters.update(iterations=4000, eval_metric="AUC", od_type="Iter", od_wait=200)
        else:
            parameters["iterations"] = int(size)
        return CatBoostClassifier(**parameters)
    parameters = dict(LGB_PARAMS["B"], extra_trees=True, extra_seed=42,
                      n_estimators=int(size) if size is not None else 10000, random_state=42)
    return lgb.LGBMClassifier(**parameters)


def v21_fit_candidate(ctx, fold, split, configuration):
    a, b = split
    fset = "catx" if configuration["kind"] == "cat_ordered" else "plusx"
    X, sup, y = ctx["X"][fset], ctx["sup"], ctx["y"]
    xa, xb = sup.add(X.iloc[a], a, a), sup.add(X.iloc[b], a, b)
    altered = sup.y_all.copy()
    altered[b] = 1 - altered[b]
    isolated = Supervised(sup.groups, sup.tkeys, altered)
    pd.testing.assert_frame_equal(xa, isolated.add(X.iloc[a], a, a))
    pd.testing.assert_frame_equal(xb, isolated.add(X.iloc[b], a, b))
    assert "SK_ID_CURR" not in xa and "TARGET" not in xa
    started = time.time()
    print(f"NOVO parte {fold}: {configuration['label']} iniciado", flush=True)
    model = v21_candidate_model(configuration, cat_cols=ctx["cat_cols"])
    if configuration["kind"] == "cat_ordered":
        model.fit(xa, y[a], eval_set=(xb, y[b]), use_best_model=True, verbose=500)
        iteration = model.get_best_iteration()+1
    else:
        model, iteration = fit_with_es(model, "B", xa, y[a], xb, y[b])
    p = model.predict_proba(xb)[:, 1]
    print(f"NOVO parte {fold}: {configuration['label']}, AUC {roc_auc_score(y[b],p):.8f}, "
          f"tamanho {iteration}, {time.time()-started:.1f}s", flush=True)
    return p, int(iteration)


def v21_job(ctx, fold, split, candidates):
    started = time.time()
    # Reuse original v20 learners in RAM, refitting all six in this new run.
    print(f"PARTE {fold} iniciada; candidatos {candidates}", flush=True)
    baseline, iterations = v20_baseline_fold(ctx, fold, split)
    pair_pred, pair_size = v20_candidate_fold(ctx, fold, split,
                                            dict(label="tabm_pair_full",kind="tabm_pairwise",pair_weight=1.0))
    baseline = np.column_stack([baseline, pair_pred])
    iterations.append([pair_size])
    predictions, sizes = {}, {}
    for index in candidates:
        predictions[index], sizes[index] = v21_fit_candidate(ctx, fold, split, V21_CONFIGS[index])
    print(f"PARTE {fold} concluída; {time.time()-started:.1f}s", flush=True)
    return dict(fold=fold, baseline=baseline, baseline_iterations=iterations,
                predictions=predictions, candidate_sizes=sizes, seconds=time.time()-started,
                threads=2, threadpools=threadpool_info(), torch_interop_pool=torch.get_num_interop_threads())


def v21_choose(target, baseline, candidate):
    results = []
    for alpha in V21_ALPHAS:
        p = (1-alpha)*baseline + alpha*candidate
        results.append(dict(alpha=float(alpha), auc=float(roc_auc_score(target,p))))
    # Stable max resolves exact ties in favor of zero/smaller new weight.
    return max(results, key=lambda r:r["auc"]), results


def v21_run(ctx, directory):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    y, n = ctx["y"], ctx["n"]
    splits = list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(n),y))
    fold_ids = np.empty(n, dtype=np.int8)
    for fold, (_, b) in enumerate(splits):
        fold_ids[b] = fold
    selection = fold_ids < 2
    baseline_oof = np.full((n,6),np.nan)
    new_oof = np.full((n,3),np.nan)
    baseline_iterations = [[] for _ in range(6)]
    new_iterations = [[] for _ in range(3)]
    report = dict(status="running",confirmed=False,csv_sha256=None,
                  configurations=V21_CONFIGS,alpha_grid=V21_ALPHAS.tolist(),
                  baseline_weights=V21_WEIGHTS.tolist(),input_sha256=input_hashes,
                  versions=PINNED,exact_installed_versions={p:importlib.metadata.version(p) for p in PINNED},
                  python=sys.version,platform=platform.platform(),fold_results=[],selection_folds=[0,1],
                  confirmation_folds=[2,3,4],cv_workers=2,threads_per_model=2,
                  source_note="All six baseline learners and all candidates refit from the two input CSVs; no predictions or models loaded.")

    def save():
        report["seconds"] = time.time()-started
        (directory/"experiment_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")

    def receive(results):
        for r in results:
            b = splits[r["fold"]][1]
            baseline_oof[b] = r["baseline"]
            for dst, values in zip(baseline_iterations,r["baseline_iterations"]):
                dst.extend(values)
            for index,p in r["predictions"].items():
                new_oof[b,index] = p
                new_iterations[index].append(r["candidate_sizes"][index])
            report["fold_results"].append({k:r[k] for k in ["fold","seconds","threads","threadpools","torch_interop_pool"]})
            save()

    save()
    compact = {k:v for k,v in ctx.items() if k!="raw"}
    with joblib.parallel_config(backend="loky",n_jobs=2,inner_max_num_threads=2,initializer=v20_initialize_worker):
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v21_job)(compact,f,splits[f],[0,1,2]) for f in [0,1]))
        ranks = np.column_stack([to_rank(p)**2 for p in baseline_oof[selection].T])
        base = ranks@V21_WEIGHTS
        baseline_auc = float(roc_auc_score(y[selection],base))
        searches=[]
        for index in range(3):
            result,grid = v21_choose(y[selection],base,to_rank(new_oof[selection,index])**2)
            searches.append(dict(candidate=index,**result,grid=grid))
        chosen=max(searches,key=lambda r:r["auc"])
        report.update(screening=searches,screening_baseline_auc=baseline_auc)
        print("TRIAGEM",json.dumps(dict(baseline=baseline_auc,searches=searches)),flush=True)
        if chosen["alpha"]==0 or chosen["auc"]<=baseline_auc+1e-12:
            report.update(status="rejected_in_screening",reason="None of the three candidates improves folds 0–1; no new CSV.")
            save()
            return report
        index=chosen["candidate"]
        report.update(status="completing_oof",chosen_family=index)
        save()
        receive(joblib.Parallel(max_nbytes=None)(joblib.delayed(v21_job)(compact,f,splits[f],[index]) for f in [2,3,4]))
    assert np.isfinite(baseline_oof).all() and np.isfinite(new_oof[:,index]).all()
    ranks = np.column_stack([to_rank(p)**2 for p in baseline_oof.T])
    base = ranks@V21_WEIGHTS
    candidate = to_rank(new_oof[:,index])**2
    chosen,grid = v21_choose(y[selection],base[selection],candidate[selection])
    alpha=chosen["alpha"]
    weights=np.r_[(1-alpha)*V21_WEIGHTS,alpha]
    report.update(status="recipe_frozen_before_confirmation",alpha=alpha,weights=weights.tolist(),
                  selected_candidate=V21_CONFIGS[index],selection_auc=chosen["auc"],final_selection_grid=grid,
                  baseline_iterations=baseline_iterations,candidate_iterations=new_iterations[index],
                  baseline_sizes=[max(1,round(float(np.mean(it))*1.15)) for it in baseline_iterations],
                  candidate_size=max(1,round(float(np.mean(new_iterations[index]))*1.15)))
    (directory/"frozen_selection.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    save()
    new=(1-alpha)*base+alpha*candidate
    masks=[(f"parte_{f}",fold_ids==f) for f in range(5)]
    masks += [("partes_0_1",selection),("partes_2_4",~selection),("total",np.ones(n,bool))]
    table=[]
    for label,mask in masks:
        old_auc=float(roc_auc_score(y[mask],base[mask]))
        new_auc=float(roc_auc_score(y[mask],new[mask]))
        table.append(dict(parte=label,linhas=int(mask.sum()),v20=old_auc,v21=new_auc,delta=new_auc-old_auc))
    by_name={r["parte"]:r for r in table}
    assert abs(by_name["total"]["v20"]-0.7740056552208389)<1e-12,"Reference v20 not reproduced; investigate before generating a CSV."
    assert abs(by_name["partes_2_4"]["v20"]-0.7747163432212283)<1e-12
    confirmed=alpha>0 and by_name["partes_2_4"]["delta"]>0 and by_name["total"]["delta"]>0
    report.update(table=table,confirmed=bool(confirmed),status="gain_confirmed" if confirmed else "rejected_in_confirmation")
    save()
    print(pd.DataFrame(table).to_string(index=False,float_format=lambda v:f"{v:.8f}"),flush=True)
    return report
