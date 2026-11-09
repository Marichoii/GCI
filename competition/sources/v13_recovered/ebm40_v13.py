"""EBM screening: smooth additive effects plus learned pair interactions.

Algorithm reference: https://interpret.ml/docs/ebm.html
Only the supplied data and fold-local target features are used.
"""
import argparse,gc,hashlib,importlib.metadata,json,pickle,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
from experiment_v13 import Context,atomic,save_npz,screen_report,sha

PARAMS=dict(interactions=40,outer_bags=4,inner_bags=0,max_bins=128,
            max_interaction_bins=32,learning_rate=.04,max_rounds=4000,
            early_stopping_rounds=100,min_samples_leaf=50,max_leaves=3,
            smoothing_rounds=100,interaction_smoothing_rounds=50,
            reg_lambda=10,n_jobs=4,random_state=42)

def run(inputs,reference,output,folds):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);ctx=Context(inputs,reference)
    ctx.signature=hashlib.sha256((ctx.signature+sha(__file__)+importlib.metadata.version('interpret-core')).encode()).hexdigest()
    name='ebm40'
    for fold in folds:
        path=out/f'{name}_fold{fold}.npz'
        if path.is_file():
            with np.load(path) as z:
                if str(z['signature'])==ctx.signature:continue
        tr,va=ctx.splits[fold];A,B,T=ctx.frames(fold,core=True)
        for frame,idx in [(A,tr),(B,va),(T,ctx.jt)]:
            for c in ctx.cats:frame[c]=ctx.raw.iloc[idx][c].fillna('__NA__').astype(str).to_numpy()
        types=['nominal' if c in ctx.cats else 'continuous' for c in A.columns]
        model=ExplainableBoostingClassifier(**PARAMS,feature_types=types)
        print('Training EBM',fold,A.shape,flush=True);start=time.time()
        with joblib.parallel_config(backend='threading'):
            model.fit(A,ctx.y[tr])
        pv=model.predict_proba(B)[:,1];pt=model.predict_proba(T)[:,1]
        meta=dict(name=name,fold=fold,auc=float(roc_auc_score(ctx.y[va],pv)),
            seconds=time.time()-start,params=PARAMS,package=importlib.metadata.version('interpret-core'),
            best_iteration=np.asarray(model.best_iteration_).tolist())
        save_npz(path,signature=np.array(ctx.signature),valid_indices=va,valid=pv,test=pt,meta=np.array(json.dumps(meta)))
        atomic(out/f'{name}_fold{fold}.pkl',pickle.dumps(model))
        print('FINISHED',json.dumps(meta),flush=True);screen_report(ctx,name,out)
        del A,B,T,model;gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/Competition/input')
    p.add_argument('--reference',default='delivery_v12');p.add_argument('--output',default='output_v13')
    p.add_argument('--folds',type=int,nargs='+',default=[0,1]);a=p.parse_args()
    run(a.input,a.reference,a.output,a.folds)
