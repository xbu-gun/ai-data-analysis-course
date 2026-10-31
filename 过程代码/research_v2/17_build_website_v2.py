"""独立构建本地网站v2：全文为主、旧方案对照，保留v1。"""
import shutil
import pandas as pd
from study_common import *

def main():
    setup();assert_v1_unchanged()
    source=ROOT/'成果交付/05_网站';site=OUT/'08_网站_v2';assets=site/'assets';assets.mkdir(parents=True,exist_ok=True)
    raw=(source/'assets/data.js').read_text(encoding='utf-8')
    payload=json.loads(raw.removeprefix('window.PROJECT_DATA=').removesuffix(';'))
    topics_dir=OUT/'06_全文主题对照'
    topics=pd.read_csv(topics_dir/'seed42/逐文档主题.csv',dtype={'code':str})
    names=json.loads((STUDY/'text_topic_names_v2.json').read_text(encoding='utf-8'))
    embeddings=json.loads((topics_dir/'全文嵌入摘要.json').read_text(encoding='utf-8'))
    comparison=json.loads((topics_dir/'全文主题对照摘要.json').read_text(encoding='utf-8'))
    changes=pd.read_csv(topics_dir/'逐文档归属变化对照.csv',dtype={'code':str})
    previous_names=payload['topicNames'].copy()
    for code,t in payload['text'].items():
        group=topics[topics.code.str.zfill(6)==code];paired=changes[changes.code.str.zfill(6)==code]
        assert len(group)==t['documentCount']>0
        t['oldTopics']=t['topics'];t['topics']=[[int(k),int(v)] for k,v in group.topic.value_counts().items()]
        t['topicDocuments']=[{'year':int(row.year),'fullTopic':int(row.full_text_topic),'oldTopic':int(row.old_topic)} for row in paired.sort_values('year').itertuples()]
    financial_codes={c['code'] for c in payload['companies']}
    payload['textOnlyCompanies']=[]
    for code in sorted(set(payload['text'])-financial_codes):
        latest=topics[topics.code.str.zfill(6)==code].sort_values('year').iloc[-1]
        payload['textOnlyCompanies'].append({'code':code,'company':str(latest.company),'textOnly':True})
    primary=next(x for x in comparison['seed_runs'] if x['seed']==42)
    payload['oldTopicNames']=previous_names;payload['topicNames']={k:v['name'] for k,v in names.items()}
    payload['topicNotes']={k:v['note'] for k,v in names.items()}
    payload['oldOutlierRate']=payload['outlierRate'];payload['outlierRate']=primary['outlier_rate'];payload['topicCount']=primary['final_topics']
    payload['topicComparison']={'oldCoverageMedian':embeddings['old_coverage_median'],'fullCoverage':1.0,
        'oldOutliers':306,'fullOutliers':primary['outlier_documents'],'ari':comparison['old_vs_full']['ari_including_outliers'],
        'seedOutlierMin':min(x['outlier_rate'] for x in comparison['seed_runs']),'seedOutlierMax':max(x['outlier_rate'] for x in comparison['seed_runs']),
        'genericTopicDocuments':int((topics.topic==0).sum()),'documents':len(topics)}
    financial=json.loads((OUT/'首轮优化核查摘要.json').read_text(encoding='utf-8'))
    payload['applicationBacktest']={'meanFoldAuc':financial['application_tree_mean_fold_roc_auc'],
        'flaggedShare':financial['application_tree_pooled_threshold_metrics']['flagged_share'],
        'precision':financial['application_tree_pooled_threshold_metrics']['precision'],
        'developmentOnly':True,'financialDefinitionsUnresolved':True}
    payload['version']='2.0 / 全文主题主展示'
    (assets/'data.js').write_text('window.PROJECT_DATA='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';',encoding='utf-8')
    for name in ['index.html','app.js']:shutil.copyfile(STUDY/'web_v2'/name,site/name)
    css=(CODE/'web/style.css').read_text(encoding='utf-8')+'\n'+(STUDY/'web_v2/style_extra.css').read_text(encoding='utf-8')
    (site/'style.css').write_text(css,encoding='utf-8')
    clouds=assets/'wordcloud';clouds.mkdir(exist_ok=True)
    for p in (source/'assets/wordcloud').glob('*.png'):shutil.copyfile(p,clouds/p.name)
    docs=site/'docs';docs.mkdir(exist_ok=True)
    for name in ['第二阶段_全文主题对照报告.md','逐文档归属变化对照.csv','覆盖率与向量变化.png','主题归属交叉图.png']:
        shutil.copyfile(topics_dir/name,docs/name)
    for name in ['文化指标贡献解释.md','各维主要贡献词_语境审阅.csv']:
        shutil.copyfile(OUT/'07_文化指标解释'/name,docs/name)
    shutil.copyfile(OUT/'向老师核实的口径清单.md',docs/'向老师核实的口径清单.md')
    summary={'time_beijing':now(),'website':str(site/'index.html'),'companies':len(payload['companies']),'text_companies':len(payload['text']),
             'text_only_companies_without_financial_predictions':len(payload['textOnlyCompanies']),
             'main_topic_strategy':'full_text','old_strategy_retained':True,'financial_values_or_scores_changed':False,
             'source_payload_sha256':sha(source/'assets/data.js'),'new_payload_sha256':sha(assets/'data.js'),
             'published':False,'v1_preserved_files':assert_v1_unchanged()}
    dump(site/'网站构建核查_v2.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
