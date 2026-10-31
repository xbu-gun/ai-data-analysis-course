"""网站树的历史训练期五折回测；无调参、不使用2020—2021测试样本。"""
import hashlib,inspect,json,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,brier_score_loss,confusion_matrix
from sklearn.calibration import calibration_curve
from study_common import *
sys.path.insert(0,str(CODE/'scripts'))
from model_tools import plot_setup

def fit_fold(train,valid,features):
    X=train[features].replace([np.inf,-np.inf],np.nan)
    disabled=list(X.columns[X.isna().all()])
    active=[c for c in features if c not in disabled]
    med=X[active].median()
    assert med.notna().all()
    model=DecisionTreeClassifier(**PROTOCOL['backtest']['parameters'])
    model.fit(X[active].fillna(med),train.is_fraud)
    score=model.predict_proba(valid[active].replace([np.inf,-np.inf],np.nan).fillna(med))[:,1]
    return score,{'disabled_all_missing_features':disabled,'active_features':active,'depth':model.get_depth(),
                  'leaves':int(model.get_n_leaves()),'medians':{c:float(v) for c,v in med.items()}}

def metrics(y,p,flag,weights=None):
    w=np.ones(len(y)) if weights is None else np.asarray(weights)
    y=np.asarray(y);flag=np.asarray(flag)
    tp=float(w[(y==1)&flag].sum());fp=float(w[(y==0)&flag].sum());fn=float(w[(y==1)&~flag].sum());tn=float(w[(y==0)&~flag].sum())
    return {'roc_auc':float(roc_auc_score(y,p,sample_weight=w)),'pr_auc_ap':float(average_precision_score(y,p,sample_weight=w)),
            'positive_baseline':float(np.average(y,weights=w)),'brier':float(np.average((np.asarray(p)-y)**2,weights=w)),
            'recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'flagged_share':(tp+fp)/w.sum(),
            'tn':int(tn) if weights is None else tn,'fp':int(fp) if weights is None else fp,'fn':int(fn) if weights is None else fn,'tp':int(tp) if weights is None else tp}

def quota(y,p,budget):
    y=np.asarray(y);p=np.asarray(p);k=int(np.ceil(len(y)*budget))
    threshold=np.sort(p)[::-1][k-1];strict=p>threshold;tied=p==threshold
    nstrict=int(strict.sum());ntie=int(tied.sum());fill=k-nstrict
    strict_pos=int(y[strict].sum());tie_pos=int(y[tied].sum());tie_neg=ntie-tie_pos
    lower=strict_pos+max(0,fill-tie_neg);upper=strict_pos+min(fill,tie_pos);expected=strict_pos+fill*tie_pos/ntie
    return {'budget':budget,'rows':len(y),'quota':k,'boundary_score':float(threshold),'strict_higher_rows':nstrict,'boundary_tied_rows':ntie,
            'tied_rows_to_select':fill,'precision_min':lower/k,'precision_max':upper/k,'precision_expected_uniform_tie_sampling':expected/k,
            'recall_min':lower/y.sum(),'recall_max':upper/y.sum(),'recall_expected_uniform_tie_sampling':expected/y.sum(),
            'include_all_boundary_ties_rows':nstrict+ntie,'include_all_boundary_ties_share':(nstrict+ntie)/len(y),
            'include_all_boundary_ties_precision':(strict_pos+tie_pos)/(nstrict+ntie),'positive_baseline':float(y.mean())}

def main():
    setup();freeze_v1()
    source=DATA/'舞弊多年份/合并后数据/panel_ml.csv'
    config={'protocol':PROTOCOL['backtest'],'source_sha256':sha(source),'fit_implementation_sha256':hashlib.sha256(inspect.getsource(fit_fold).encode()).hexdigest(),
            'scikit_learn':importlib.metadata.version('scikit-learn')}
    run=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    out=OUT/'02_网站树历史回测'/run;out.mkdir(parents=True,exist_ok=True)
    cached=out/'fold_predictions.csv'
    if cached.exists():
        old=json.loads((out/'experiment_manifest.json').read_text(encoding='utf-8'));assert old['configuration']==config
        pred=pd.read_csv(cached,dtype={'Stkcd':str});folds=old['folds'];print('Using exact matching training-backtest cache',run,flush=True)
    else:
        splits=json.loads((CODE/'中间数据/split_indices_v1.json').read_text(encoding='utf-8'))
        raw=pd.read_csv(source,dtype={'Stkcd':str});raw.Stkcd=raw.Stkcd.str.zfill(6)
        train=raw.loc[splits['fraud']['train']].copy();assert train.Year.between(2007,2019).all()
        # 后续对象只含原训练部分，释放原表；不读取或统计2020—2021的特征、标签。
        del raw
        features=[c for c in train if c not in ['Stkcd','Year','is_fraud','is_serious']]
        years=np.sort(train.Year.unique());predictions=[];folds=[]
        for fold,(a,b) in enumerate(TimeSeriesSplit(5).split(years),1):
            tr=train[train.Year.isin(years[a])];val=train[train.Year.isin(years[b])]
            assert tr.Year.max()<val.Year.min()
            score,details=fit_fold(tr,val,features)
            block=val[['Stkcd','Year','is_fraud']].copy();block['source_row']=block.index;block['fold']=fold;block['score']=score
            block['trust_score']=np.round(100*(1-score),1)
            predictions.append(block)
            folds.append({'fold':fold,'train_years':years[a].tolist(),'validation_years':years[b].tolist(),'train_rows':len(tr),'validation_rows':len(val),
                          'train_positive_baseline':float(tr.is_fraud.mean()),'validation_positive_baseline':float(val.is_fraud.mean()),**details})
            print('FOLD',fold,tr.Year.min(),tr.Year.max(),'->',val.Year.min(),val.Year.max(),'disabled',details['disabled_all_missing_features'],flush=True)
        pred=pd.concat(predictions,ignore_index=True);save(pred,cached)
        dump(out/'experiment_manifest.json',{'run_id':run,'created_beijing':now(),'configuration':config,'folds':folds,'holdout_used':False})
    rows=[];quotas=[]
    groups=[(int(n),g) for n,g in pred.groupby('fold')]+[('pooled',pred)]
    for fold,g in groups:
        y=g.is_fraud.to_numpy();p=g.score.to_numpy()
        for name,flag in [('score>=0.5',p>=.5),('website_trust<60',g.trust_score.to_numpy()<60)]:rows.append({'fold':fold,'decision_rule':name,'rows':len(g),**metrics(y,p,flag)})
        for b in [.05,.1]:quotas.append({'fold':fold,**quota(y,p,b)})
    result=pd.DataFrame(rows);save(result,out/'逐折与汇总回测指标.csv');q=pd.DataFrame(quotas);save(q,out/'核查预算与同分边界.csv')
    # 固定预测的公司分块自助法；只描述历史回测的不确定性，不重新拟合或挑模型。
    ids,inv=np.unique(pred.Stkcd,return_inverse=True);rng=np.random.default_rng(42);boot=[]
    y=pred.is_fraud.to_numpy();p=pred.score.to_numpy();flag=pred.trust_score.to_numpy()<60
    for rep in range(500):
        count=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids));weights=count[inv]
        m=metrics(y,p,flag,weights)
        boot.append({key:m[key] for key in ['roc_auc','pr_auc_ap','recall','precision','f1','brier']})
    boot=pd.DataFrame(boot);intervals={c:{'lower_95':float(boot[c].quantile(.025)),'upper_95':float(boot[c].quantile(.975))} for c in boot}
    dump(out/'公司分块Bootstrap区间.json',{'replications':500,'unit':'company; all its year records sampled together','intervals':intervals,
         'limits':'conditional on these fixed predictions and selected historical years; does not remove time shocks or model-selection uncertainty'})
    plt=plot_setup();per=result[result.decision_rule=='website_trust<60'].copy();per=per[per.fold!='pooled']
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot(per.fold.astype(str),per.roc_auc,'o-',label='ROC-AUC');axes[0].plot(per.fold.astype(str),per.pr_auc_ap,'o-',label='PR-AUC(AP)');axes[0].plot(per.fold.astype(str),per.positive_baseline,'--',label='正类基线');axes[0].legend();axes[0].set_title('网站树：逐折排序能力')
    axes[1].plot(per.fold.astype(str),per.recall,'o-',label='召回率');axes[1].plot(per.fold.astype(str),per.precision,'o-',label='精确率');axes[1].plot(per.fold.astype(str),per.flagged_share,'--',label='提示比例');axes[1].legend();axes[1].set_title('沿用网站可信度<60的规则');fig.tight_layout();fig.savefig(out/'网站树历史回测.png');plt.close(fig)
    fraction,mean=calibration_curve(y,p,n_bins=8,strategy='quantile');fig,ax=plt.subplots(figsize=(5,4));ax.plot(mean,fraction,'o-');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='加权分数',ylabel='实际正类比例',title='历史折外预测：分数校准诊断');fig.tight_layout();fig.savefig(out/'应用分数校准诊断.png');plt.close(fig)
    pooled=result[(result.fold=='pooled')&(result.decision_rule=='website_trust<60')].iloc[0].to_dict()
    summary={'time_beijing':now(),'run_id':run,'prediction_rows':len(pred),'companies':len(ids),'validation_years':sorted(int(v) for v in pred.Year.unique()),
             'pooled_website_rule':pooled,'year_2020_2021_used':False,'tuning_performed':False,'v1_files_verified_unchanged':assert_v1_unchanged()}
    dump(out/'回测摘要.json',summary)
    text=f"""# 网站实际应用树：历史训练期回测

时间（北京时间）：{now()}；实验编号：{run}。

## 检验对象

使用与网站相同的加权决策树配方：max_depth=5，min_samples_leaf=100，balanced，random_state=42。没有调参或根据本次结果换阈值。只用v1训练部分2007—2019年，按课程既定5折时间前推。每折用当时更早年份训练，预测更晚年份；2020—2021年留出样本完全未使用。

原本36列输入，前四折的rd_ratio全空；用户确认仅在这些折停用，绝不借未来中位数。第五折保留。其余缺失用各折训练中位数；不缩尾、不标准化，沿用网站的处理方式。

## 汇总结果

共{len(pred):,}条历史折外预测、{len(ids):,}家公司。沿用网站可信度<60的提示规则：ROC-AUC={pooled['roc_auc']:.4f}，PR-AUC(AP)={pooled['pr_auc_ap']:.4f}，正类基线={pooled['positive_baseline']:.4f}；召回率={pooled['recall']:.2%}，精确率={pooled['precision']:.2%}，提示比例={pooled['flagged_share']:.2%}。Brier={pooled['brier']:.4f}。误报{pooled['fp']}条，漏报{pooled['fn']}条。

这些结果来自重新训练的同配方历史模型，不能当成当前全量树或2025应用集的已知准确率。balanced分数的校准图另附，尤其不能把加权分数等同真实造假概率。

## 有限核查预算与同分

给出前5%和10%的核查预算。叶节点分数会大量并列；不使用股票代码或真实标签来挑同分公司。报告给出：若精确满足预算，同分边界中任意抽取的最差/最好范围，以及均匀随机抽取的期望；另列包括全部边界同分公司时会扩大到多少样本。期望是统计计算，不是一次真实随机核查的成绩。

## 不确定性

对固定折外预测按公司分块自助抽样500次，保留同一公司跨年的关联。区间只描述这些历史样本与固定预测的抽样不确定性，不能覆盖未来年份冲击、源数据错位或模型选择不确定性。原始数据口径核查结论应与本回测一起阅读。

## 使用边界

这是原训练期内的发展性回测，不是新的未使用最终测试。v1原测试成绩封存，不以它挑改进方法。本轮所有输入、参数和实现指纹写入experiment_manifest.json，缓存仅在指纹完全一致时使用。
"""
    (out/'网站树历史回测报告.md').write_text(text,encoding='utf-8')
    entry=out.parent/'回测入口.md';entry.write_text(f"# 网站树历史回测\n\n当前实验：{run}\n\n[回测报告]({run}/网站树历史回测报告.md)\n\n[指标表]({run}/逐折与汇总回测指标.csv)\n\n[核查预算与同分边界]({run}/核查预算与同分边界.csv)\n",encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
