"""Feature probe: train-fold-only density segments of log(credit/annuity).

No TARGET or ID is accepted by the density estimator. This does not replace
the inherited v17 preprocessing, which still uses unlabeled train+test data.
"""
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

V22_DENSITY_CONFIGS=[dict(label="density_pooled",by_contract=False),
                     dict(label="density_by_contract",by_contract=True)]


class LoanDensitySegments:
    columns=["AMT_CREDIT","AMT_ANNUITY","NAME_CONTRACT_TYPE"]

    def __init__(self,by_contract=False):
        self.by_contract=by_contract

    @staticmethod
    def coordinates(raw):
        ratio=raw.AMT_CREDIT.to_numpy(dtype=float)/raw.AMT_ANNUITY.to_numpy(dtype=float)
        with np.errstate(divide="ignore",invalid="ignore"):
            x=np.log(ratio)
        x[~np.isfinite(x)]=np.nan
        contract=raw.NAME_CONTRACT_TYPE.fillna("__NA__").astype(str).to_numpy()
        return x,contract

    @staticmethod
    def estimate(x):
        x=x[np.isfinite(x)]
        assert len(x)>0
        low,high=np.quantile(x,[0.001,0.999])
        if high-low<1e-6:
            low,high=float(np.median(x)-0.05),float(np.median(x)+0.05)
        edges=np.linspace(low,high,2049)
        grid=(edges[:-1]+edges[1:])/2
        step=float(grid[1]-grid[0])
        sigma=min(float(np.std(x)),float(np.subtract(*np.quantile(x,[0.75,0.25]))/1.349))
        bandwidth=max(0.9*max(sigma,1e-8)*len(x)**(-0.2),step)
        count,_=np.histogram(np.clip(x,low,high),bins=edges)
        density=gaussian_filter1d(count.astype(float),sigma=bandwidth/step,mode="reflect")
        density=density/(max(float(density.sum()),1)*step)
        peaks,_=find_peaks(density,prominence=0.01*float(density.max()),distance=12)
        if not len(peaks):peaks=np.array([int(np.argmax(density))])
        minima=[int(a+np.argmin(density[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]
        bounds=np.r_[low,grid[minima],high]
        assert np.all(np.diff(bounds)>0)
        return dict(grid=grid,density=density,peaks=grid[peaks],bounds=bounds,
                    bandwidth=bandwidth,observations=len(x))

    def fit(self,raw):
        assert list(raw.columns)==self.columns
        x,contract=self.coordinates(raw)
        self.pooled=self.estimate(x)
        self.groups={}
        if self.by_contract:
            for key in sorted(set(contract)):
                values=x[contract==key]
                if np.isfinite(values).sum()>=100:self.groups[key]=self.estimate(values)
        return self

    def transform(self,raw):
        assert list(raw.columns)==self.columns
        x,contract=self.coordinates(raw)
        z=np.full((len(x),6),np.nan,dtype=float)
        for key in sorted(set(contract)):
            rows=np.flatnonzero((contract==key)&np.isfinite(x))
            d=self.groups.get(key,self.pooled)
            value=x[rows]
            segment=np.searchsorted(d["bounds"][1:-1],value,side="right")
            left,right=d["bounds"][segment],d["bounds"][segment+1]
            z[rows,0]=value
            z[rows,1]=segment
            z[rows,2]=np.clip((value-left)/(right-left),0,1)
            z[rows,3]=value-d["peaks"][segment]
            z[rows,4]=d["peaks"][segment]
            z[rows,5]=np.log(np.maximum(np.interp(value,d["grid"],d["density"]),1e-12))
        out=pd.DataFrame(z,index=raw.index,columns=["DENS_LOG_RATIO","DENS_SEGMENT","DENS_POSITION",
                            "DENS_PEAK_DEVIATION","DENS_PEAK_LOG","DENS_LOG_DENSITY"])
        out["DENS_MISSING"]=(~np.isfinite(x)).astype(np.int8)
        return out

    def describe(self):
        def describe(d):
            return dict(observations=d["observations"],bandwidth=d["bandwidth"],
                        peaks_log=d["peaks"].tolist(),bounds_log=d["bounds"].tolist())
        return dict(by_contract=self.by_contract,pooled=describe(self.pooled),
                    groups={k:describe(v) for k,v in self.groups.items()})


def v22_probe_fold(ctx,fold,split):
    started=time.time()
    a,b=split
    X,sup,y=ctx["X"]["plusx"],ctx["sup"],ctx["y"]
    xa,xb=sup.add(X.iloc[a],a,a),sup.add(X.iloc[b],a,b)
    altered=sup.y_all.copy();altered[b]=1-altered[b]
    other=Supervised(sup.groups,sup.tkeys,altered)
    pd.testing.assert_frame_equal(xa,other.add(X.iloc[a],a,a))
    pd.testing.assert_frame_equal(xb,other.add(X.iloc[b],a,b))
    predictions=[];details=[]
    for configuration in [None]+V22_DENSITY_CONFIGS:
        label="reference_lgb_B_seed42" if configuration is None else configuration["label"]
        A,B=xa,xb
        description=None
        if configuration is not None:
            raw=ctx["raw"][LoanDensitySegments.columns]
            transform=LoanDensitySegments(configuration["by_contract"]).fit(raw.iloc[a])
            description=transform.describe()
            A=pd.concat([xa.reset_index(drop=True),transform.transform(raw.iloc[a]).reset_index(drop=True)],axis=1)
            B=pd.concat([xb.reset_index(drop=True),transform.transform(raw.iloc[b]).reset_index(drop=True)],axis=1)
        model=make_model("B",42)
        model,iteration=fit_with_es(model,"B",A,y[a],B,y[b])
        p=model.predict_proba(B)[:,1]
        predictions.append(p)
        auc=float(roc_auc_score(y[b],p))
        details.append(dict(label=label,auc=auc,iteration=int(iteration),density=description))
        print(f"PROBE parte {fold}: {label}, AUC {auc:.8f}, {iteration} árvores",flush=True)
    return dict(fold=fold,predictions=np.column_stack(predictions),details=details,
                seconds=time.time()-started,threads=2,threadpools=threadpool_info())


def v22_probe(ctx,directory):
    directory.mkdir(parents=True,exist_ok=True)
    started=time.time()
    y=ctx["y"]
    splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    compact={k:v for k,v in ctx.items() if k!="Xt"}
    with joblib.parallel_config(backend="loky",n_jobs=2,inner_max_num_threads=2,initializer=v20_initialize_worker):
        results=joblib.Parallel(max_nbytes=None)(joblib.delayed(v22_probe_fold)(compact,f,splits[f]) for f in [0,1])
    predictions=np.full((len(y),3),np.nan)
    for r in results:predictions[splits[r["fold"]][1]]=r["predictions"]
    selection=np.isfinite(predictions[:,0])
    auc=[float(roc_auc_score(y[selection],predictions[selection,j])) for j in range(3)]
    rows=[]
    for i,configuration in enumerate(V22_DENSITY_CONFIGS,1):
        changes=[r["details"][i]["auc"]-r["details"][0]["auc"] for r in results]
        rows.append(dict(configuration=configuration,auc_0_1=auc[i],delta_0_1=auc[i]-auc[0],
                         deltas_by_fold=changes,mean_fold_delta=float(np.mean(changes))))
    eligible=[row for row in rows if row["delta_0_1"]>0 and row["mean_fold_delta"]>0]
    chosen=max(eligible,key=lambda r:r["auc_0_1"]) if eligible else None
    report=dict(status="feature_probe_positive" if chosen else "rejected_in_feature_probe",
                confirmed=False,csv_sha256=None,probe_baseline_auc=auc[0],candidates=rows,chosen=chosen,
                folds=[{k:v for k,v in r.items() if k!="predictions"} for r in results],
                seconds=time.time()-started,selection_folds=[0,1],confirmation_folds_executed=[],
                input_sha256=input_hashes,versions=PINNED,python=sys.version,platform=platform.platform(),
                limitation="Only a matched single-seed LightGBM feature ablation, not validation of the v20/v21 ensemble. Inherited v17 preprocessing still uses unlabeled train+test; density itself is fitted on outer training rows only.")
    (directory/"probe_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=report["status"],baseline=auc[0],candidates=rows),ensure_ascii=False),flush=True)
    return report
