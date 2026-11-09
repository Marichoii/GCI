"""Train selected v13 components or verify checkpoints, then reproduce the CSV."""
import argparse,gc,hashlib,importlib,importlib.metadata,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from experiment_v13 import Context,atomic,save_npz,rank,sha

PINS={'numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0','scikit-learn':'1.8.0',
      'lightgbm':'4.7.0','catboost':'1.2.10','xgboost':'3.4.1',
      'torch':'2.14.1+cpu','tabm':'0.0.3','rtdl-num-embeddings':'0.0.12',
      'interpret-core':'0.7.8'}
MODULES={'ebm12':'ebm_v13','ebm40':'ebm40_v13','pairblocks4':'pair_blocks_v13'}


def run(inputs,reference,output,recipe_path,retrain=False):
    inputs,reference,output,recipe_path=map(Path,[inputs,reference,output,recipe_path])
    output.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(recipe_path.read_text());weights=recipe['weights']
    assert all(w>=0 for w in weights.values()) and abs(sum(weights.values())-1)<1e-12
    selected=[n for n in weights if n!='reference_v12' and weights[n]>0]
    needed={'numpy','pandas','scipy','scikit-learn','lightgbm'}
    if any(n.startswith('ebm') for n in selected):needed.add('interpret-core')
    if any(n.startswith('tabm') for n in selected):needed.update(['torch','tabm','rtdl-num-embeddings'])
    if any(n.startswith('pair') for n in selected):needed.add('xgboost')
    if any(n.startswith('cat') for n in selected):needed.add('catboost')
    versions={p:importlib.metadata.version(p) for p in sorted(needed)}
    assert versions=={p:PINS[p] for p in needed},f'Use the pinned versions. Found: {versions}'
    training_output=output
    if retrain:
        training_output=output/('retrain_'+str(time.time_ns()))
        training_output.mkdir()
    # Each runner verifies its model/data/source signature before cache reuse.
    for name in selected:
        if name in MODULES:
            module=importlib.import_module(MODULES[name])
            module.run(inputs,reference,training_output,list(range(5)))
        else:
            from experiment_v13 import run as fit_components
            ctx=fit_components(inputs,reference,training_output,name,list(range(5)))
            del ctx
        gc.collect()
    ctx=Context(inputs,reference)
    candidate_oof=weights['reference_v12']*rank(ctx.base_oof)
    candidate_test=weights['reference_v12']*rank(ctx.base_test)
    details={};component_oof={};component_test={};source_hashes={}
    for name in selected:
        expected=ctx.signature
        if name in MODULES:
            module=importlib.import_module(MODULES[name])
            package='xgboost' if name=='pairblocks4' else 'interpret-core'
            expected=hashlib.sha256((expected+sha(module.__file__)+versions[package]).encode()).hexdigest()
            source_hashes[MODULES[name]+'.py']=sha(module.__file__)
        oof=np.empty(ctx.n);pt=np.empty((5,len(ctx.test)));metadata=[]
        for fold,(_,va) in enumerate(ctx.splits):
            with np.load(training_output/f'{name}_fold{fold}.npz') as z:
                assert str(z['signature'])==expected,(name,fold,'signature mismatch')
                assert np.array_equal(va,z['valid_indices'])
                oof[va]=z['valid'];pt[fold]=z['test']
                metadata.append(json.loads(str(z['meta'])))
        assert np.isfinite(oof).all() and np.isfinite(pt).all()
        component_oof[name]=oof;component_test[name]=pt
        candidate_oof+=weights[name]*rank(oof)
        candidate_test+=weights[name]*rank(pt.mean(axis=0))
        details[name]=metadata
    metrics={}
    for name,mask in [('all',np.ones(ctx.n,dtype=bool)),('selection_01',ctx.fold_ids<2),('additional_234',ctx.fold_ids>=2)]:
        ref=float(roc_auc_score(ctx.y[mask],ctx.base_oof[mask]));new=float(roc_auc_score(ctx.y[mask],candidate_oof[mask]))
        metrics[name]=dict(reference_auc=ref,candidate_auc=new,delta=new-ref,rows=int(mask.sum()))
    sample=pd.read_csv(inputs/'sample_submission.csv')
    assert sample.SK_ID_CURR.is_unique and set(sample.SK_ID_CURR)==set(ctx.test.SK_ID_CURR)
    sample['TARGET']=sample.SK_ID_CURR.map(pd.Series(candidate_test,index=ctx.test.SK_ID_CURR))
    assert list(sample.columns)==['SK_ID_CURR','TARGET'] and len(sample)==61500
    assert sample.TARGET.between(0,1).all()
    csv_path=output/'submission_v13.csv'
    atomic(csv_path,sample.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode())
    readback=pd.read_csv(csv_path,float_precision='round_trip')
    assert np.array_equal(readback.SK_ID_CURR,sample.SK_ID_CURR)
    assert np.array_equal(readback.TARGET,sample.TARGET)
    archive=dict(train_id=ctx.train.SK_ID_CURR.to_numpy(),fold_ids=ctx.fold_ids,
                 reference_oof=ctx.base_oof,candidate_oof=candidate_oof,
                 reference_test=ctx.base_test,candidate_test=candidate_test)
    archive.update({'oof_'+n:p for n,p in component_oof.items()})
    archive.update({'test_'+n:p for n,p in component_test.items()})
    save_npz(output/'validation_v13.npz',**archive)
    source_hashes.update({n:sha(Path(__file__).parent/n) for n in ['experiment_v13.py','deliver_v13.py']})
    report=dict(recipe=recipe,metrics=metrics,details=details,versions=versions,
        reference_public_score_user_reported=.7782,reference_rank_user_reported=23,
        reference_leaderboard_entries_user_reported=1949,candidate_public_score=None,
        reference_csv_sha256=sha(reference/'submission_v12_pricing.csv'),
        candidate_csv_sha256=sha(csv_path),runtime_source_sha256=source_hashes,
        data_sha256={f:sha(inputs/f) for f in ['train.csv','test.csv']},
        training_output=str(training_output),
        interpretation='OOF local scores are not public scores. Models and blend weights were screened on folds 0/1; additional folds check consistency, but earlier model selection and neural early stopping use OOF. No independent holdout of the entire process; no guarantee of improving public AUC or reaching 0.80.')
    atomic(output/'resultados_v13.json',json.dumps(report,ensure_ascii=False,indent=2).encode())
    print(json.dumps(dict(metrics=metrics,csv=str(csv_path),sha256=sha(csv_path)),indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='Competition/input')
    p.add_argument('--reference',default='reference_v12');p.add_argument('--output',default='output_v13')
    p.add_argument('--recipe',default='v13_recipe.json');p.add_argument('--retrain',action='store_true')
    a=p.parse_args();run(a.input,a.reference,a.output,a.recipe,a.retrain)
