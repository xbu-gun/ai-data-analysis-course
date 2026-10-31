"""任务 b：全量历史树与外推预测，独立于任务 a 的留出评估模型。"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from common import *
from model_tools import plot_setup


def main():
    setup()
    out=OUTPUT/"03_2025风险预测";out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(DATA/"舞弊多年份/合并后数据/panel_ml.csv",dtype={"Stkcd":str})
    app=pd.read_csv(DATA/"舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv",dtype={"Stkcd":str})
    train["Stkcd"]=train["Stkcd"].str.zfill(6)
    app["Stkcd"]=app["Stkcd"].str.zfill(6)
    features=[c for c in train if c not in ["Stkcd","Year","is_fraud","is_serious"]]
    X=train[features].replace([np.inf,-np.inf],np.nan);med=X.median()
    assert not med.isna().any()
    A=app[features].replace([np.inf,-np.inf],np.nan).fillna(med)
    y=train.is_fraud
    model=DecisionTreeClassifier(**CONFIG["application_tree"],class_weight=CONFIG["application_class_weight_fraud"])
    model.fit(X.fillna(med),y)
    p=model.predict_proba(A)[:,1];leaves=model.apply(A)
    train_leaves=model.apply(X.fillna(med))
    teacher_meta=json.loads((DATA/"舞弊多年份/财务可信度应用/data.json").read_text(encoding="utf-8"))
    # 只读取公司名称作为元数据，绝不沿用演示评分、规则或财务指标。
    names={r["code"]:r["name"] for r in teacher_meta["companies"]}
    rules={}
    def visit(node,conditions):
        t=model.tree_
        if t.children_left[node]==t.children_right[node]:
            mask=train_leaves==node
            rules[str(node)]={"conditions":conditions,"training_rows":int(mask.sum()),"training_positive_rate":float(y[mask].mean())}
            return
        feature=features[t.feature[node]];threshold=float(t.threshold[node])
        visit(t.children_left[node],conditions+[{"feature":feature,"operator":"<=","threshold":threshold}])
        visit(t.children_right[node],conditions+[{"feature":feature,"operator":">","threshold":threshold}])
    visit(0,[])
    trust=np.round(100*(1-p),1)
    results=pd.DataFrame({"code":app.Stkcd,"company":app.Stkcd.map(names),"financial_report_year":2025,
                          "prediction_target_year":app.Year,"relative_risk_score":p,"trust_score":trust,
                          "level":np.select([trust>=80,trust>=60],["较高","中等"],default="偏低"),"leaf":leaves,
                          "missing_features_before_imputation":app[features].isna().sum(axis=1)})
    results["risk_rank"]=results.relative_risk_score.rank(method="min",ascending=False).astype(int)
    assert results.company.notna().all(), "公司名称映射仍有缺失，不能凭空补名称。"
    results["tree_path"]=results.leaf.map(lambda n:" 且 ".join(f"{x['feature']} {x['operator']} {x['threshold']:.6g}" for x in rules[str(n)]["conditions"]))
    save_csv(results.sort_values(["relative_risk_score","code"],ascending=[False,True]),out/"2025年报_风险预测表_v1.csv")
    records=[]
    metric_cols=["size_assets","size_revenue","lev","current_ratio","roa","roe","net_margin","ar_ratio","asset_turnover","cash_share","rev_growth"]
    for i,r in results.iterrows():
        rec={k:(int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else v) for k,v in r.items() if k!="tree_path"}
        # 网页财务值保留原始缺失，不把模型填补值冒充公司实际披露值。
        rec["metrics"]={k:None if pd.isna(app.loc[i,k]) else float(app.loc[i,k]) for k in metric_cols}
        records.append(rec)
    write_json(out/"company_predictions_v1.json",{"meta":{"financial_report_year":2025,"prediction_target_year":2026,"score_interpretation":"相对风险排序，非校准的造假概率","name_source":"课程演示应用，仅使用代码与名称映射"},"companies":records})
    write_json(out/"tree_rules_v1.json",rules)
    save_csv(med.rename("median").rename_axis("feature").reset_index(),out/"应用模型训练集中位数_v1.csv")
    joblib.dump({"model":model,"features":features,"median":med},MODELS/"全量应用决策树_v1.joblib")
    (out/"全量应用树规则_v1.txt").write_text(export_text(model,feature_names=features),encoding="utf-8")
    plt=plot_setup();fig,ax=plt.subplots(figsize=(24,12));plot_tree(model,feature_names=features,filled=True,max_depth=3,fontsize=7,ax=ax);fig.tight_layout();fig.savefig(out/"应用树前四层_v1.png");plt.close(fig)
    summary={"time_beijing":now(),"training_rows":len(train),"application_rows":len(app),"features":len(features),
             "depth":model.get_depth(),"leaves":model.get_n_leaves(),"nodes":model.tree_.node_count,
             "class_weight":"balanced","levels":{str(k):int(v) for k,v in results.level.value_counts().items()},
             "median_trust":float(np.median(trust)),"application_accuracy_known":False}
    write_json(out/"应用模型摘要_v1.json",summary)
    (out/"模型说明_v1.md").write_text(f"""# 2025 年报外推预测：模型说明

生成时间（北京时间）：{now()}。

## 数据与方法

使用全部历史面板 {len(train):,} 个公司年度、36个财务特征，训练深度5、叶节点至少100条、balanced权重、随机种子42的决策树。应用于 {len(app):,} 家公司。缺失值用历史训练集中位数，未用应用数据重新估计填补值。

文件名中的2025表示年报年度；Year=2026依课程对齐规则解释为预测目标年。未回查原始报表生成过程，因此保留这一来源说明。公司名来自课程演示应用的代码名称映射；没有使用其评分、规则或财务数字。

## 分数的含义

树输出的加权正类分数用于相对排序；可信度=100×(1−分数)。balanced改变了类别权重，分数不等于真实造假概率。档位≥80、60—80、<60沿用教程，属于演示展示规则，并非经业务成本验证的风险阈值。当前可信度中位数 {np.median(trust):.1f}。

本模型与任务a的留出评估模型分开：这里使用全量历史训练、按应用教程加权；任务a保留测试集、不加舞弊类别权重。不能把任务a冠军的测试成绩直接当成本应用树的准确率。

## 信心来自哪里、不能承诺什么

信心的依据是流程可复现、训练和应用列对齐、训练中位数复用、每家公司都有实际树路径、叶节点保留历史样本数和未加权正类比例。外推到较新年份可能发生分布漂移，缺失较多的公司还会更多依赖填补值。2025年报应用集没有真实标签，因此尚不知道其真实准确率。

评分不能作为违法违规认定、审计意见或投资建议。财务指标、路径、历史叶节点占比共同展示，用于提供进一步核查的线索。
""",encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":main()
