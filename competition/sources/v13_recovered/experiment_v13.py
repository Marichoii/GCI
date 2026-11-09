"""GCI v13: pairwise trees, pricing CatBoost and numerical-embedding TabM.

All supervised transformations use outer-training labels only. No external
data or weights. Reference v12 package is required alongside original inputs.
"""
import argparse, gc, hashlib, importlib.metadata, io, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, payload):
    path = Path(path)
    temp = path.with_name(path.name+'.writing')
    with temp.open('wb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def save_npz(path, **items):
    stream=io.BytesIO();np.savez_compressed(stream,**items)
    atomic(path,stream.getvalue())


def rank(p):
    return rankdata(p)/len(p)


class Context:
    def __init__(self, inputs, reference):
        self.inputs,self.reference=Path(inputs),Path(reference)
        sys.path.insert(0,str(self.reference.resolve()))
        import pricing_v12 as previous
        self.previous=previous
        hashes={f:sha(self.inputs/f) for f in previous.EXPECTED_DATA}
        assert hashes==previous.EXPECTED_DATA
        self.train=pd.read_csv(self.inputs/'train.csv')
        self.test=pd.read_csv(self.inputs/'test.csv')
        self.y=self.train.TARGET.to_numpy();self.n=len(self.y)
        self.jt=np.arange(self.n,self.n+len(self.test));self.it=np.arange(self.n)
        self.ns,self.extra=previous.reference_environment(self.reference,self.train,self.test)
        self.raw=self.ns['raw_all'];self.cats=self.ns['cat_cols'];self.add_mates=self.ns['add_mates']
        self.core_cols=self.extra.columns[:74].tolist()
        with np.load(self.reference/'output_v12/auxiliary_features_v12.npz') as z:
            aux_cols=[s+suffix for s in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
                      for suffix in ['_AUX_PRED','_AUX_RESIDUAL','_IMPUTED']]
            aux_cols+=['EXT_AUX_MEAN','EXT_AUX_MIN','EXT_AUX_MAX','EXT_AUX_PRODUCT']
            self.aux=pd.DataFrame(z['values'],columns=aux_cols)
        with np.load(self.reference/'output_v12/validation_v12.npz') as z:
            assert np.array_equal(z['train_id'],self.train.SK_ID_CURR)
            self.base_oof=z['candidate_oof'].copy();self.base_test=z['candidate_test'].copy()
        assert sha(self.reference/'submission_v12_pricing.csv')=='3e992c0cd75b188968775833e6c69ce6ebc080152037f5e4cf151449646973e7'
        self.price=previous.PricingFeatures(self.raw,self.y)
        self.splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(self.y,self.y))
        self.fold_ids=np.empty(self.n,dtype='int8')
        for fold,(_,va) in enumerate(self.splits):self.fold_ids[va]=fold
        self.X=self.extra.iloc[:self.n].reset_index(drop=True)
        self.Xtest=self.add_mates(self.extra.iloc[self.n:].reset_index(drop=True),self.it,self.jt)
        for key in ['full','person','tariff','train','test']:
            self.ns.pop(key,None)
        self.signature=hashlib.sha256(json.dumps(dict(
            data=hashes,reference_csv=sha(self.reference/'submission_v12_pricing.csv'),
            auxiliary=sha(self.reference/'output_v12/auxiliary_features_v12.npz'),
            previous_source=sha(self.reference/'pricing_v12.py'),
            runtime=sha(__file__),versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm']},
            outer_seed=42,pricing_seed=17),sort_keys=True).encode()).hexdigest()

    def frames(self,fold,pricing=True,core=False):
        tr,va=self.splits[fold]
        def extend(frame,idx,price):
            if core:
                frame=frame[self.core_cols+[c for c in frame if c not in self.extra.columns]]
            parts=[frame.reset_index(drop=True),self.aux.iloc[idx].reset_index(drop=True)]
            if pricing:parts.append(price)
            return pd.concat(parts,axis=1)
        A=extend(self.add_mates(self.X.iloc[tr],tr,tr),tr,self.price.crossfit(tr) if pricing else None)
        B=extend(self.add_mates(self.X.iloc[va],tr,va),va,self.price.transform(tr,va) if pricing else None)
        T=extend(self.Xtest,self.jt,self.price.transform(tr,self.jt) if pricing else None)
        return A,B,T.reindex(columns=A.columns)


def prepare_cat(ctx,frame,idx,extra_categories=False):
    frame=frame.copy();cats=list(ctx.cats)
    for c in cats:frame[c]=ctx.raw.iloc[idx][c].fillna('__NA__').astype(str).to_numpy()
    term=ctx.raw.AMT_CREDIT/ctx.raw.AMT_ANNUITY
    series={f'TARIFF_CAT_{d}':term.round(d) for d in [1,2,3]}
    if extra_categories:
        series.update(EXT3_CATEGORY=ctx.raw.EXT_SOURCE_3,
                      CREDIT_CATEGORY=ctx.raw.AMT_CREDIT,
                      ANNUITY_CATEGORY=ctx.raw.AMT_ANNUITY,
                      GOODS_CREDIT_CAT=(ctx.raw.AMT_CREDIT/ctx.raw.AMT_GOODS_PRICE).round(3))
    for name,s in series.items():
        frequency=s.map(s.value_counts())
        values=s.astype(str).where(frequency>=30,'__RARE__')
        frame[name]=values.iloc[idx].to_numpy();cats.append(name)
    return frame,cats


def fit_tree(ctx,name,fold,out):
    tr,va=ctx.splits[fold];A,B,T=ctx.frames(fold)
    start=time.time()
    if name.startswith('pair'):
        import xgboost as xgb
        params=dict(objective='rank:pairwise',eval_metric='auc',tree_method='hist',
                    n_estimators=3000,early_stopping_rounds=150,learning_rate=.03,
                    max_depth=4 if name=='pair4' else 6,min_child_weight=30,
                    subsample=.85,colsample_bytree=.7,reg_lambda=20,reg_alpha=.5,
                    gamma=.05,max_bin=256,n_jobs=4,random_state=42,
                    lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=4)
        model=xgb.XGBRanker(**params)
        model.fit(A,ctx.y[tr],group=[len(tr)],eval_set=[(B,ctx.y[va])],
                  eval_group=[[len(va)]],verbose=False)
        pv,pt=model.predict(B),model.predict(T)
        iterations=model.best_iteration+1
        model.save_model(out/f'{name}_fold{fold}.ubj')
    else:
        from catboost import CatBoostClassifier
        params=dict(iterations=5000,learning_rate=.035,depth=4 if name=='cat4price' else 6,
                    l2_leaf_reg=20,rsm=.8,random_strength=1,bagging_temperature=.7,
                    eval_metric='AUC',od_type='Iter',od_wait=200,random_seed=42,
                    verbose=0,thread_count=4,allow_writing_files=False)
        A,cats=prepare_cat(ctx,A,tr,name=='cat6categories')
        B,_=prepare_cat(ctx,B,va,name=='cat6categories')
        T,_=prepare_cat(ctx,T,ctx.jt,name=='cat6categories')
        model=CatBoostClassifier(**params,cat_features=cats)
        model.fit(A,ctx.y[tr],eval_set=(B,ctx.y[va]),use_best_model=True)
        pv,pt=model.predict_proba(B)[:,1],model.predict_proba(T)[:,1]
        iterations=model.get_best_iteration()+1
        model.save_model(str(out/f'{name}_fold{fold}.cbm'))
    meta=dict(name=name,fold=fold,auc=float(roc_auc_score(ctx.y[va],pv)),
              iterations=int(iterations),seconds=time.time()-start,params=params,
              package=importlib.metadata.version('xgboost' if name.startswith('pair') else 'catboost'))
    return pv,pt,meta


def tabm_preprocess(ctx,A,others):
    import torch
    nums=[c for c in A if c not in ctx.cats and A[c].nunique(dropna=True)>1]
    qt=QuantileTransformer(n_quantiles=256,output_distribution='normal',subsample=100000,random_state=42)
    qt.fit(A[nums].to_numpy(dtype='float32'))
    z=qt.transform(A[nums].to_numpy(dtype='float32'))
    keep=np.array([np.unique(v[np.isfinite(v)]).size>1 for v in z.T])
    missing_cols=[c for c in nums if A[c].isna().any()]
    cards=[int(max(ctx.extra[c].max(),0))+2 for c in ctx.cats]+[2]*len(missing_cols)
    def transform(frame):
        z=np.nan_to_num(qt.transform(frame[nums].to_numpy(dtype='float32')),
                        nan=0,posinf=4,neginf=-4).clip(-4,4)[:,keep].astype('float32')
        cats=(frame[ctx.cats].to_numpy(dtype='int64')+1).clip(min=0)
        if missing_cols:cats=np.column_stack([cats,frame[missing_cols].isna().to_numpy(dtype='int64')])
        return torch.from_numpy(z),torch.from_numpy(cats)
    return transform(A),[transform(frame) for frame in others],cards,dict(qt=qt,nums=nums,keep=keep,missing_cols=missing_cols)


def fit_tabm(ctx,name,fold,out):
    import torch
    from torch import nn
    from tabm import TabM
    import rtdl_num_embeddings as embeddings
    torch.set_num_threads(4);torch.manual_seed(42+100*fold);np.random.seed(42+100*fold)
    tr,va=ctx.splits[fold]
    A,B,T=ctx.frames(fold,pricing=False,core=True)
    pair,others,cards,preprocessing=tabm_preprocess(ctx,A,[B,T])
    del A,B,T;gc.collect()
    if name=='tabmple':
        bins=embeddings.compute_bins(pair[0],n_bins=16,
            tree_kwargs=dict(min_samples_leaf=512,min_impurity_decrease=1e-6,random_state=42),
            y=torch.tensor(ctx.y[tr]),regression=False)
        embedding=embeddings.PiecewiseLinearEmbeddings(bins,8,activation=False,version='B')
    else:
        embedding=embeddings.PeriodicEmbeddings(pair[0].shape[1],d_embedding=8,frequency_init_scale=.1,lite=True)
        bins=None
    config=dict(k=8,n_blocks=2,d_block=128,dropout=.15,arch_type='tabm')
    model=TabM.make(n_num_features=pair[0].shape[1],num_embeddings=embedding,
                    cat_cardinalities=cards,d_out=1,**config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-5)
    target=torch.tensor(ctx.y[tr],dtype=torch.float32)
    def predict(data):
        model.eval();pred=[]
        with torch.no_grad():
            for begin in range(0,len(data[0]),2048):
                logits=model(data[0][begin:begin+2048],data[1][begin:begin+2048]).squeeze(-1)
                pred.append(logits.sigmoid().mean(dim=1).numpy())
        return np.concatenate(pred)
    best=-np.inf;best_epoch=0;state=None;stale=0;start=time.time();history=[]
    for epoch in range(1,41):
        es=time.time();model.train()
        for ids in torch.randperm(len(tr)).split(1024):
            optimizer.zero_grad(set_to_none=True)
            logits=model(pair[0][ids],pair[1][ids]).squeeze(-1)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,target[ids,None].expand_as(logits))
            loss.backward();optimizer.step()
        auc=float(roc_auc_score(ctx.y[va],predict(others[0])));scheduler.step(auc)
        history.append(dict(epoch=epoch,auc=auc,seconds=time.time()-es))
        if auc>best+1e-6:
            best=auc;best_epoch=epoch;stale=0
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        print(name,fold,'epoch',epoch,'auc',round(auc,7),'best',best_epoch,'seconds',round(time.time()-es,1),flush=True)
        if stale>=7:break
    model.load_state_dict(state)
    pv,pt=predict(others[0]),predict(others[1])
    torch.save(dict(state=state,bins=bins,config=config,cards=cards,n_num=pair[0].shape[1]),out/f'{name}_fold{fold}.pt')
    atomic(out/f'{name}_fold{fold}_preprocessing.pkl',pickle.dumps(preprocessing))
    meta=dict(name=name,fold=fold,auc=float(roc_auc_score(ctx.y[va],pv)),iterations=best_epoch,
              seconds=time.time()-start,history=history,config=config,
              packages={p:importlib.metadata.version(p) for p in ['torch','tabm','rtdl-num-embeddings']})
    return pv,pt,meta


def screen_report(ctx,name,out):
    p=np.zeros(ctx.n);mask=np.zeros(ctx.n,dtype=bool);meta=[]
    for fold,(_,va) in enumerate(ctx.splits):
        path=out/f'{name}_fold{fold}.npz'
        if not path.is_file():continue
        with np.load(path) as z:
            if str(z['signature'])!=ctx.signature:continue
            assert np.array_equal(va,z['valid_indices'])
            p[va]=z['valid'];mask[va]=True;meta.append(json.loads(str(z['meta'])))
    if not mask.any():return
    baseline=float(roc_auc_score(ctx.y[mask],ctx.base_oof[mask]))
    result=dict(model=name,folds=[m['fold'] for m in meta],reference_auc=baseline,
                component_auc=float(roc_auc_score(ctx.y[mask],p[mask])),weights={})
    for w in [.05,.1,.2,.3,.4]:
        blend=(1-w)*rank(ctx.base_oof[mask])+w*rank(p[mask])
        auc=float(roc_auc_score(ctx.y[mask],blend))
        result['weights'][str(w)]=dict(auc=auc,delta=auc-baseline)
    atomic(out/f'{name}_summary.json',json.dumps(result,indent=2).encode())
    print('SUMMARY',json.dumps(result),flush=True)


def run(inputs,reference,out,name,folds):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ctx=Context(inputs,reference)
    for fold in folds:
        path=out/f'{name}_fold{fold}.npz'
        if path.is_file():
            with np.load(path) as z:
                if str(z['signature'])==ctx.signature:
                    print('Reusing',path,flush=True);continue
        print('Training',name,fold,flush=True)
        fn=fit_tabm if name.startswith('tabm') else fit_tree
        pv,pt,meta=fn(ctx,name,fold,out)
        _,va=ctx.splits[fold]
        assert np.isfinite(pv).all() and np.isfinite(pt).all()
        save_npz(path,signature=np.array(ctx.signature),valid_indices=va,valid=pv,test=pt,meta=np.array(json.dumps(meta)))
        print('FINISHED',json.dumps(meta),flush=True)
        screen_report(ctx,name,out);gc.collect()
    return ctx


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='Competition/input')
    parser.add_argument('--reference',default='reference_v12')
    parser.add_argument('--output',default='output_v13')
    parser.add_argument('--model',required=True,choices=['pair4','pair6','cat4price','cat6categories','tabmple','tabmperiodic'])
    parser.add_argument('--folds',type=int,nargs='+',default=[0,1])
    args=parser.parse_args()
    run(args.input,args.reference,args.output,args.model,args.folds)
