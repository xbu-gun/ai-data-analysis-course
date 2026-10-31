"""最终数字核对与来源完整性；只读既有预测，不重新用测试集选模型。"""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score,recall_score,f1_score,confusion_matrix
from common import *


def main():
    setup();checks={}
    fingerprints=pd.read_csv(OUTPUT/"01_数据核查/来源文件指纹_v1.csv")
    checks["source_fingerprints_unchanged"]=all(digest(SOURCE/r.source_relative_path)==r.sha256 for r in fingerprints.itertuples(index=False))
    for task in ["舞弊预测","破产预测"]:
        out=OUTPUT/"02_全流程分析"/task
        s=json.loads((out/"最终测试评估_v1.json").read_text(encoding="utf-8"))
        p=pd.read_csv(out/"测试集逐条预测与错误类型_v1.csv")
        recomputed={"roc_auc":roc_auc_score(p.actual,p.score),"pr_auc_ap":average_precision_score(p.actual,p.score),"recall":recall_score(p.actual,p.prediction),"f1":f1_score(p.actual,p.prediction)}
        assert all(abs(s[k]-v)<1e-10 for k,v in recomputed.items())
        assert confusion_matrix(p.actual,p.prediction).ravel().tolist()==s["confusion_matrix_tn_fp_fn_tp"]
        checks[task+"_saved_predictions_match_report"]=True
    payload=json.loads((OUTPUT/"05_网站/assets/data.js").read_text(encoding="utf-8").removeprefix("window.PROJECT_DATA=").removesuffix(";"))
    assert len(payload["companies"])==5574 and len({c['code'] for c in payload['companies']})==5574
    assert len(payload["text"])==225
    assert all((OUTPUT/f"05_网站/assets/wordcloud/{code}.png").exists() for code in payload['text'])
    assert len(pd.read_csv(OUTPUT/"04_文本分析/公司年度文化指标_v2.csv"))==1123
    assert len(pd.read_csv(OUTPUT/"04_文本分析/五维文化词典_审阅后_v2.csv"))==784
    assert len(pd.read_csv(OUTPUT/"04_文本分析/doc_topics_v1.csv"))==1123
    # 验证应用模型实际预测、叶节点、表格和网页完全一致，不触碰任务a的测试选择。
    fitted=joblib.load(MODELS/"全量应用决策树_v1.joblib")
    app=pd.read_csv(DATA/"舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv",dtype={"Stkcd":str});app.Stkcd=app.Stkcd.str.zfill(6)
    A=app[fitted['features']].fillna(fitted['median'])
    predictions=pd.read_csv(OUTPUT/"03_2025风险预测/2025年报_风险预测表_v1.csv",dtype={"code":str}).set_index('code').loc[app.Stkcd]
    assert np.allclose(fitted['model'].predict_proba(A)[:,1],predictions.relative_risk_score)
    assert np.array_equal(fitted['model'].apply(A),predictions.leaf)
    checks['application_predictions_match_saved_model']=True
    checks['website_counts_and_all_wordclouds_match']=True
    checks['browser_visual_and_interaction_test']='未完成：宿主工具因无法可靠识别Windows浏览器网址而中止'
    checks['test_models_re_evaluated_or_retuned']=False
    checks['website_published']=False
    checks['video_recorded']=False
    # 复现源码包只包含过程代码和说明，排除输入、模型和公司级输出。
    with zipfile.ZipFile(OUTPUT/"06_提交材料/复现源码包_v1.zip") as z:
        assert not any(n.endswith(('.csv','.joblib','.safetensors','.npy')) or '.venv' in n or '中间数据' in n for n in z.namelist())
        assert 'web/index.html' in z.namelist()
    checks['source_archive_excludes_course_data']=True
    write_json(OUTPUT/"06_提交材料/最终文件与数字核查_v1.json",{"time_beijing":now(),"checks":checks})
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
