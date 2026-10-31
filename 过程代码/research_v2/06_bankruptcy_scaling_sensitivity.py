"""用户确认的分位数标准化聚类对照，只用原训练样本。"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer,RobustScaler
from sklearn.cluster import MiniBatchKMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score
from scipy.stats import chi2_contingency
from study_common import *
sys.path.insert(0,str(CODE/'scripts'))
from model_tools import plot_setup

def main():
    setup();out=OUT/'03_破产聚类标准化对照';out.mkdir(parents=True,exist_ok=True)
    split=json.loads((CODE/'中间数据/split_indices_v1.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(DATA/'破产预测/bankruptcy_data.csv');raw.columns=raw.columns.str.strip();ids=split['bankruptcy']['train']
    raw=raw.iloc[ids].reset_index(drop=True);x=raw.drop(columns='Bankrupt?').replace([np.inf,-np.inf],np.nan)
    x=x.loc[:,(x.isna().mean()<=.5)&(x.nunique()>1)]
    clean=x.clip(x.quantile(.01),x.quantile(.99),axis=1).fillna(x.median())
    transformer=QuantileTransformer(n_quantiles=1000,output_distribution='uniform',subsample=100000,random_state=42)
    z=transformer.fit_transform(clean)
    assert np.isfinite(z).all() and z.min()>=0 and z.max()<=1
    rng=np.random.default_rng(42);sample=np.sort(rng.choice(len(z),size=min(4000,len(z)),replace=False))
    scan=[];diagnostics=[];assignments={}
    for k in range(2,11):
        model=MiniBatchKMeans(k,random_state=42,n_init=10,batch_size=1024).fit(z);labels=model.labels_
        score=silhouette_score(z[sample],labels[sample]);scan.append({'k':k,'inertia':float(model.inertia_),'silhouette':float(score)})
        if k in [2,4]:
            old=pd.read_csv(OUTPUT_BASE/f'02_全流程分析/破产预测/无监督探索/k{k}_公司样本簇归属_v1.csv').set_index('source_row').loc[ids].cluster.to_numpy()
            ward=AgglomerativeClustering(k,linkage='ward').fit_predict(z[sample[:2000]])
            ct=pd.crosstab(labels,raw['Bankrupt?']);chi,p,dof,expected=chi2_contingency(ct)
            v=float(np.sqrt(chi/(len(raw)*min(ct.shape[0]-1,ct.shape[1]-1))))
            diagnostics.append({'k':k,'silhouette':float(score),'ari_vs_v1_robust':float(adjusted_rand_score(old,labels)),
                                'ward_ari_sample':float(adjusted_rand_score(model.predict(z[sample[:2000]]),ward)),
                                'cramers_v_posthoc':v,'chi2_p_posthoc':float(p)})
            rates=pd.DataFrame({'cluster':labels,'label':raw['Bankrupt?']}).groupby('cluster').label.agg(['size','sum','mean']).reset_index()
            save(rates,out/f'分位数标准化_k{k}_标签事后对照.csv');save(clean.assign(cluster=labels).groupby('cluster').median().reset_index(),out/f'分位数标准化_k{k}_原指标中位数.csv')
            save(pd.DataFrame({'source_row':ids,'cluster':labels}),out/f'分位数标准化_k{k}_簇归属.csv')
        print('quantile k',k,'silhouette',round(score,4),flush=True)
    scan=pd.DataFrame(scan);save(scan,out/'分位数标准化_簇数比较.csv');save(pd.DataFrame(diagnostics),out/'标准化敏感性对照.csv')
    baseline=pd.read_csv(OUTPUT_BASE/'02_全流程分析/破产预测/无监督探索/簇数比较_v1.csv')
    plt=plot_setup();fig,ax=plt.subplots(figsize=(8,4));ax.plot(baseline.k,baseline.silhouette,'o-',label='v1 RobustScaler');ax.plot(scan.k,scan.silhouette,'o-',label='分位数映射0—1');ax.set(xlabel='簇数k',ylabel='轮廓系数',title='标准化改变后的聚类结构；不同空间的轮廓系数不能直接当模型优劣');ax.legend();fig.tight_layout();fig.savefig(out/'标准化对照_轮廓系数.png');plt.close(fig)
    variance=np.var(z,axis=0);share=variance/variance.sum();max_feature=str(clean.columns[int(share.argmax())])
    result={'time_beijing':now(),'training_rows':len(raw),'features':len(clean.columns),'user_approved':True,'held_out_used':False,
            'max_distance_contribution_share':float(share.max()),'max_contributor':max_feature,
            'silhouette_best_k':int(scan.loc[scan.silhouette.idxmax(),'k']),'k2_k4_diagnostics':diagnostics,'v1_replaced':False}
    dump(out/'标准化对照摘要.json',result)
    (out/'标准化敏感性分析报告.md').write_text(f"""# 破产数据聚类：标准化敏感性分析

时间（北京时间）：{now()}。用户确认增加本对照，v1保留。

相同的4,773条训练样本、同样94个特征、同样1%/99%缩尾和中位数填补，仅将RobustScaler替换为训练样本分位数到0—1的映射。标签未进入预处理或聚类，留出数据未使用。MiniBatchKMeans参数和k=2—10扫描范围保持一致。

本空间最大单列的平方距离贡献为{share.max():.2%}（{max_feature}）。k=2和k=4与v1、Ward对照的一致性见标准化敏感性对照.csv。轮廓系数在两个不同距离空间中不能直接当作同一量尺比较，低于v1不等于本对照更差；重点看原结论是否依赖少量变量的数值尺度。

分位数映射保留排序、压缩极端间距，也有损失：金额或比率间真实差距不再保留，不能声称它恢复了变量的真实经济单位。本轮用它检验原分组是否稳定，没有自动选它替换主结果。所有标签关联均为训练内事后统计，不是预测成绩。
""",encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

OUTPUT_BASE=ROOT/'成果交付'
if __name__=='__main__':main()
