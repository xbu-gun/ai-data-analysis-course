"""将本次结果整合为可离线打开的静态网站；不执行发布。"""
import json
import shutil
import numpy as np
import pandas as pd
from common import *


def main():
    setup();out=OUTPUT/"05_网站";assets=out/"assets";assets.mkdir(parents=True,exist_ok=True)
    predictions=json.loads((OUTPUT/"03_2025风险预测/company_predictions_v1.json").read_text(encoding="utf-8"))
    rules=json.loads((OUTPUT/"03_2025风险预测/tree_rules_v1.json").read_text(encoding="utf-8"))
    culture=pd.read_csv(OUTPUT/"04_文本分析/公司年度文化指标_v2.csv",dtype={"code":str})
    topics=pd.read_csv(OUTPUT/"04_文本分析/doc_topics_v1.csv",dtype={"code":str})
    topic_summary=json.loads((OUTPUT/"04_文本分析/主题建模摘要_v1.json").read_text(encoding="utf-8"))
    topic_names=json.loads((CODE/"topic_names_v1.json").read_text(encoding="utf-8"))
    text={};dims=["创新","诚信","质量","尊重","团队"]
    for code,g in culture.groupby("code"):
        row=g.sort_values("year").iloc[-1];tt=topics[topics.code==code]
        text[code]={"latest":{"year":int(row.year),**{d:float(row[d]) for d in dims}},"firstYear":int(g.year.min()),
                    "topics":[[int(t),int(n)] for t,n in tt.topic.value_counts().items()],"documentCount":len(tt)}
    train=pd.read_csv(DATA/"舞弊多年份/合并后数据/panel_ml.csv",dtype={"Stkcd":str})
    app=pd.read_csv(DATA/"舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv",dtype={"Stkcd":str});app.Stkcd=app.Stkcd.str.zfill(6);app=app.set_index("Stkcd")
    used=set(x["feature"] for leaf in rules.values() for x in leaf["conditions"])
    distributions={}
    for f in used:
        s=train[f].replace([np.inf,-np.inf],np.nan)
        distributions[f]={"q01":float(s.quantile(.01)),"q99":float(s.quantile(.99)),
                          "negative_median":float(s[train.is_fraud==0].median()),"positive_median":float(s[train.is_fraud==1].median())}
    for c in predictions["companies"]:
        c["path_values"]={f:None if pd.isna(app.loc[c["code"],f]) else float(app.loc[c["code"],f]) for f in used}
    payload={"companies":predictions["companies"],"rules":rules,"text":text,"topicNames":topic_names,
             "cultureMean":{d:float(culture[culture.year==2022][d].mean()) for d in dims},
             "cultureMeanByYear":{str(y):{d:float(g[d].mean()) for d in dims} for y,g in culture.groupby("year")},
             "topicCount":topic_summary["final_topics_excluding_outlier"],"outlierRate":topic_summary["outlier_rate"],
             "textExamples":[code for code in ["600518","600519","000027","000513"] if code in text]+list(text)[:2],
             "featureDistributions":distributions,
             "evaluations":{t:json.loads((OUTPUT/f"02_全流程分析/{t}/最终测试评估_v1.json").read_text(encoding="utf-8")) for t in ["舞弊预测","破产预测"]}}
    for f in ["index.html","style.css","app.js"]:shutil.copyfile(CODE/"web"/f,out/f)
    (assets/"data.js").write_text("window.PROJECT_DATA="+json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False)+";",encoding="utf-8")
    cloud_dir=assets/"wordcloud";cloud_dir.mkdir(exist_ok=True)
    for p in (OUTPUT/"04_文本分析/wordcloud").glob("*.png"):shutil.copyfile(p,cloud_dir/p.name)
    assert len(payload["companies"])==5574 and len(text)==225
    write_json(OUTPUT/"05_网站/网站构建核查_v1.json",{"time_beijing":now(),"companies":5574,"text_companies":225,"wordclouds":len(list(cloud_dir.glob('*.png'))),"offline_assets":True,"published":False})
    print(out/"index.html")


if __name__=="__main__":main()
