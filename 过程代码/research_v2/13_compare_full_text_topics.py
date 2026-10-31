"""旧聚类复现、全文对照和预先固定的随机种子敏感性。"""
import argparse,gzip,inspect
from importlib.metadata import version
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import joblib
from study_common import *

TEXT_OUT=OUT/'06_全文主题对照'
TP=json.loads((STUDY/'text_protocol.json').read_text(encoding='utf-8'))

def fit_topics(docs,emb,seed,label):
    configuration={'protocol':TP,'seed':seed,'label':label,'embedding_bytes_sha256':hashlib.sha256(emb.tobytes()).hexdigest(),
                   'corpus_sha256':sha(CODE/'中间数据/tokenized_corpus_v1.json.gz'),
                   'fit_code_sha256':hashlib.sha256(inspect.getsource(fit_topics).encode()).hexdigest(),
                   'versions':{p:version(p) for p in ['bertopic','umap-learn','hdbscan','scikit-learn','numpy','scipy']}}
    key=hashlib.sha256(json.dumps(configuration,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:12]
    directory=WORK/'full_text_topics'/key;directory.mkdir(parents=True,exist_ok=True)
    cache=directory/'fitted.joblib'
    if cache.exists():
        saved=joblib.load(cache);assert saved['configuration']==configuration
        return saved['model'],saved['initial_topics'],directory
    model=BERTopic(embedding_model=None,language='multilingual',
                   umap_model=UMAP(**TP['umap'],random_state=seed),hdbscan_model=HDBSCAN(**TP['hdbscan']),
                   vectorizer_model=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False),
                   calculate_probabilities=False,verbose=False,top_n_words=10)
    joined=[' '.join(d['tokens']) for d in docs]
    initial,_=model.fit_transform(joined,embeddings=emb)
    initial=np.asarray(initial).copy();initial_count=len(set(initial)-{-1})
    if initial_count>17:model.reduce_topics(joined,nr_topics=13)
    saved={'configuration':configuration,'model':model,'initial_topics':initial}
    joblib.dump(saved,cache,compress=3)
    dump(directory/'manifest.json',{'configuration':configuration,'finished_beijing':now(),
                                  'initial_topics':initial_count,'final_topics':len(set(model.topics_)-{-1})})
    print(json.dumps({'fitted':label,'seed':seed,'initial_topics':initial_count,'final_topics':len(set(model.topics_)-{-1}),
                      'outliers':int(np.sum(np.asarray(model.topics_)==-1))}),flush=True)
    return model,initial,directory

def compare_labels(a,b):
    common=(a!=-1)&(b!=-1)
    return {'ari_including_outliers':float(adjusted_rand_score(a,b)),
            'nmi_including_outliers':float(normalized_mutual_info_score(a,b)),
            'common_non_outlier_documents':int(common.sum()),
            'ari_on_common_non_outliers':float(adjusted_rand_score(a[common],b[common])) if common.sum()>1 else None,
            'old_outlier_to_topic':int(np.sum((a==-1)&(b!=-1))),
            'old_topic_to_outlier':int(np.sum((a!=-1)&(b==-1))),
            'both_outliers':int(np.sum((a==-1)&(b==-1)))}

def align_topics(a,b):
    old_ids=sorted(set(a)-{-1});new_ids=sorted(set(b)-{-1})
    matrix=np.array([[np.sum((a==old)&(b==new)) for new in new_ids] for old in old_ids])
    mapping={-1:-1}
    if matrix.size:
        rows,cols=linear_sum_assignment(-matrix)
        mapping.update({int(new_ids[j]):int(old_ids[i]) for i,j in zip(rows,cols)})
    return mapping

def topic_excerpt(text,words):
    # 年报TXT常把整章压为一行。按句末标点找连续片段，避免总是截取开头套话。
    import re
    generic=set('产品 情况 研发 行业 发展 项目 有限公司 资金 营业 技术 业务 服务 投资 市场 生产 销售 经营 收入 报告'.split())
    best=None
    for m in re.finditer(r'[^。！？；\n]+',text):
        for lo in range(m.start(),m.end(),300):
            hi=min(lo+440,m.end());snippet=text[lo:hi]
            if len(snippet)<35:continue
            hits=[w for w in words if w in snippet]
            if not hits:continue
            distinct=[w for w in hits if w not in generic]
            score=(len(distinct),len(hits),-abs(len(snippet)-260),-lo)
            if best is None or score>best[0]:best=(score,lo,hi,snippet,hits)
    if best is None:return 0,min(440,len(text)),text[:440],[]
    _,lo,hi,snippet,hits=best
    return lo,hi,snippet,hits

def export_topic_files(model,initial,docs,emb,seed):
    directory=TEXT_OUT/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    topics=np.asarray(model.topics_)
    table=pd.DataFrame([{k:d[k] for k in ['code','company','year','file']} for d in docs])
    table['initial_topic']=initial;table['topic']=topics;save(table,directory/'逐文档主题.csv')
    info=model.get_topic_info()
    info['top_words']=info.Topic.map(lambda topic:'、'.join(w for w,s in model.get_topic(int(topic))[:10]))
    save(info[['Topic','Count','Name','top_words']],directory/'主题词与篇数.csv')
    if seed!=TP['primary_seed']:return table,info
    representatives=[];nearest=[]
    for topic in sorted(set(topics)):
        idx=np.flatnonzero(topics==topic);center=emb[idx].mean(axis=0);center/=np.linalg.norm(center)
        ranked_all=idx[np.argsort(-(emb[idx]@center),kind='stable')]
        for rank,i in enumerate(ranked_all[:3],1):
            nearest.append({'topic':int(topic),'center_rank':rank,**{k:docs[int(i)][k] for k in ['code','company','year','file']}})
        ranked=[];seen=set()
        for i in ranked_all:
            code=docs[int(i)]['code']
            if code not in seen:ranked.append(i);seen.add(code)
            if len(ranked)==3:break
        if len(ranked)<min(3,len(idx)):
            for i in ranked_all:
                if i not in ranked:ranked.append(i)
                if len(ranked)==min(3,len(idx)):break
        words=[w for w,s in model.get_topic(int(topic))[:10]]
        for i in ranked:
            d=docs[int(i)];path=CODE/'中间数据/corpus_sample'/d['file'];text=path.read_text(encoding='utf-8-sig')
            start,end,snippet,hits=topic_excerpt(text,words)
            assert text[start:end]==snippet
            hits=[w for w in words if w in snippet]
            representatives.append({'topic':int(topic),'representative_rank':len([x for x in representatives if x['topic']==topic])+1,
                                    'center_rank':int(np.flatnonzero(ranked_all==i)[0])+1,'selection_rule':'优先不同公司，按文档中心距离顺序选择',
                                    'code':d['code'],'company':d['company'],'year':d['year'],'file':d['file'],
                                    'source_sha256':sha(path),'character_start':start,'character_end':end,
                                    'top_words':'、'.join(words),'matched_words':'、'.join(hits),'evidence':snippet})
    save(pd.DataFrame(representatives),directory/'主题代表原文_待审阅.csv')
    save(pd.DataFrame(nearest),directory/'主题中心最近原文_不限制公司.csv')
    return table,info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');args=parser.parse_args()
    setup();TEXT_OUT.mkdir(exist_ok=True);assert_v1_unchanged()
    with gzip.open(CODE/'中间数据/tokenized_corpus_v1.json.gz','rt',encoding='utf-8') as f:docs=json.load(f)
    old_emb=np.load(CODE/'中间数据/mda_three_segment_embeddings_v1.npy')
    original=pd.read_csv(ROOT/'成果交付/04_文本分析/doc_topics_v1.csv',dtype={'code':str})
    assert list(zip(original.code.str.zfill(6),original.year))==[(str(d['code']).zfill(6),int(d['year'])) for d in docs]
    baseline,initial,cache_dir=fit_topics(docs,old_emb,TP['primary_seed'],'old_head_middle_tail')
    old_labels=original.topic.to_numpy();reproduced=np.asarray(baseline.topics_)
    replication=compare_labels(old_labels,reproduced)
    replication['exact_same_numbered_assignments']=bool(np.array_equal(old_labels,reproduced))
    replication['baseline_cache_directory']=str(cache_dir);replication['time_beijing']=now()
    dump(TEXT_OUT/'旧版聚类复现核查.json',replication)
    print(json.dumps({'baseline_replication':replication}),flush=True)
    if args.baseline_only:return
    entry=json.loads((TEXT_OUT/'嵌入运行入口.json').read_text(encoding='utf-8'))
    emb_path=Path(entry['work_directory'])/'full_text_embeddings.npy'
    emb=np.load(emb_path);summary_emb=json.loads((TEXT_OUT/'全文嵌入摘要.json').read_text(encoding='utf-8'))
    assert summary_emb['run_id']==entry['run_id'] and sha(emb_path)==summary_emb['embedding_sha256']
    assert emb.shape==old_emb.shape==(1123,512)
    models={};labels={};seed_rows=[]
    for seed in [TP['primary_seed']]+TP['sensitivity_seeds']:
        model,initial,directory=fit_topics(docs,emb,seed,'full_text_balanced_blocks')
        table,info=export_topic_files(model,initial,docs,emb,seed);models[seed]=model;labels[seed]=table.topic.to_numpy()
        seed_rows.append({'seed':seed,'initial_topics':len(set(initial)-{-1}),'final_topics':len(set(table.topic)-{-1}),
                          'outlier_documents':int(np.sum(table.topic==-1)),'outlier_rate':float(np.mean(table.topic==-1)),
                          'fit_directory':str(directory)})
    seed_table=pd.DataFrame(seed_rows);save(seed_table,TEXT_OUT/'预设随机种子敏感性.csv')
    stability=[]
    for a,b in combinations(labels,2):stability.append({'seed_a':a,'seed_b':b,**compare_labels(labels[a],labels[b])})
    save(pd.DataFrame(stability),TEXT_OUT/'种子间主题一致性.csv')
    primary=labels[TP['primary_seed']];comparison=compare_labels(old_labels,primary);mapping=align_topics(old_labels,primary)
    aligned=np.array([mapping.get(int(n),-100-int(n)) for n in primary])
    comparison['one_to_one_aligned_agreement_all']=float(np.mean(old_labels==aligned))
    comparison['baseline_replication_ari']=replication['ari_including_outliers']
    changes=pd.DataFrame([{k:d[k] for k in ['code','company','year','file']} for d in docs])
    changes['old_topic']=old_labels;changes['full_text_topic']=primary
    changes['aligned_old_topic']=pd.array([mapping.get(int(n),pd.NA) for n in primary],dtype='Int64')
    changes['alignment_note']=['离群对应离群' if n==-1 else ('按重叠篇数一对一匹配' if int(n) in mapping else '无一对一对应旧主题') for n in primary]
    changes['outlier_status_changed']=(old_labels==-1)!=(primary==-1)
    changes['same_after_one_to_one_alignment']=old_labels==aligned
    cov=pd.read_csv(TEXT_OUT/'全文分块与旧版覆盖率对照.csv')
    changes['old_coverage']=cov.head_middle_tail_coverage;changes['embedding_cosine']=cov.cosine_old_vs_full
    save(changes,TEXT_OUT/'逐文档归属变化对照.csv')
    cross=pd.crosstab(pd.Series(old_labels,name='old_topic'),pd.Series(primary,name='full_text_topic'))
    save(cross.reset_index(),TEXT_OUT/'旧主题与全文主题_交叉篇数.csv')
    dump(TEXT_OUT/'主题编号对齐_仅用于对照.json',{str(k):v for k,v in mapping.items()})
    # 所有原文留在本地。比较图不把离群点编号解释为有序数值。
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    axs[0].hist(cov.head_middle_tail_coverage,bins=22,color='#2e7183');axs[0].set(title='原首、中、尾覆盖率',xlabel='已覆盖词元占比',ylabel='文档数')
    axs[1].hist(cov.cosine_old_vs_full,bins=22,color='#b37742');axs[1].set(title='两种文档向量的余弦相似度',xlabel='相似度',ylabel='文档数')
    fig.tight_layout();fig.savefig(TEXT_OUT/'覆盖率与向量变化.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7));image=ax.imshow(cross.to_numpy(),cmap='Blues')
    ax.set_xticks(range(len(cross.columns)),cross.columns);ax.set_yticks(range(len(cross.index)),cross.index)
    ax.set(xlabel='全文主题号（独立编号）',ylabel='原主题号',title='旧主题与全文主题的文档交叉篇数')
    for i in range(len(cross.index)):
        for j in range(len(cross.columns)):
            n=int(cross.iloc[i,j])
            if n:ax.text(j,i,str(n),ha='center',va='center',fontsize=8,color='white' if n>cross.to_numpy().max()*.55 else '#182a34')
    fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(TEXT_OUT/'主题归属交叉图.png',dpi=150);plt.close(fig)
    import plotly.express as px
    xy=models[TP['primary_seed']].umap_model.embedding_[:,:2]
    plot=changes.copy();plot['x']=xy[:,0];plot['y']=xy[:,1];plot['theme']=plot.full_text_topic.astype(str)
    px.scatter(plot,x='x',y='y',color='theme',hover_data=['company','year','old_topic','old_coverage'],
               title='全文主题对照：UMAP五维结果的前两维（仅示意邻域）').write_html(TEXT_OUT/'全文主题互动图.html',include_plotlyjs='directory')
    summary={'time_beijing':now(),'embedding_run_id':entry['run_id'],'documents':len(docs),'primary_seed':TP['primary_seed'],
             'baseline_replication':replication,'old_vs_full':comparison,'seed_runs':seed_rows,'seed_pair_stability':stability,
             'v1_preserved_files':assert_v1_unchanged(),'selected_seed_by_results':False}
    dump(TEXT_OUT/'全文主题对照摘要.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
