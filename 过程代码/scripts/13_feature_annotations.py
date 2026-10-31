"""为重要性表补充原始字段含义及冠军模型分裂统计，不重新拟合。"""
import re
import json
import joblib
import pandas as pd
from common import *


def main():
    setup()
    fraud_dictionary={}
    for line in (DATA/"舞弊多年份/DATA.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells=[x.strip().strip('`') for x in line.split("|")[1:-1]]
            if len(cells)>=3 and re.fullmatch(r"[a-z][a-z0-9_]+",cells[0]):
                fraud_dictionary[cells[0]]={"definition":cells[1],"meaning":cells[-1]}
    bank_names={}
    for line in (DATA/"破产预测/说明书.txt").read_text(encoding="utf-8-sig").splitlines():
        match=re.match(r"X(\d+)[-－](.*)",line)
        if match:bank_names[int(match.group(1))]=match.group(2)
    _,bank=load_finance();bank_columns=[c for c in bank if c!="Bankrupt?"]
    for task in ["舞弊预测","破产预测"]:
        out=OUTPUT/"02_全流程分析"/task
        model=joblib.load(MODELS/f"{task}_冠军_v1.joblib")
        columns=model.named_steps['prepare'].columns_
        split_counts=model.named_steps['model'].get_booster().get_score(importance_type='weight')
        split_rows=[{"feature":columns[int(f[1:])],"split_count":int(n),"scope":"冠军模型全部树，训练拟合后统计；非因果效应"} for f,n in split_counts.items()]
        save_csv(pd.DataFrame(split_rows).sort_values('split_count',ascending=False),out/"冠军树模型分裂统计_v1.csv")
        imp=pd.read_csv(out/"特征重要性_v1.csv")
        definitions=[];meanings=[]
        for feature in imp.feature:
            if task=="舞弊预测":
                entry=fraud_dictionary.get(feature,{"definition":"见课程DATA.md附录","meaning":"含义待逐项核对"})
                definitions.append(entry['definition']);meanings.append(entry['meaning'])
            else:
                idx=bank_columns.index(feature)+1
                definition=bank_names.get(idx,f"原始说明书X{idx}，中文解释未解析")
                definitions.append(f"X{idx}: {definition}");meanings.append("沿用课程字段定义；预测贡献不代表因果效应")
        imp['field_definition']=definitions;imp['economic_meaning_or_caveat']=meanings
        imp['importance_measure']='XGBoost归一化gain，训练模型内统计'
        save_csv(imp,out/"特征重要性_含字段说明_v1.csv")
        print(task,len(imp),'feature definitions;',len(split_rows),'features used in splits')
    card=["# MODEL_CARD：课程预测模型",f"\n整理时间（北京时间）：{now()}。模型版本v1，课程原始数据只读。", "\n## 用途\n\n比较财务指标对舞弊与破产标签的预测能力，为进一步核查提供线索。模型输出不替代审计、违法认定或投资判断。"]
    log=["# 实验记录", "\n所有时间为北京时间。完整逐折分数与参数网格见成果交付/02_全流程分析，各次程序输出见本目录log文件。"]
    for task in ['舞弊预测','破产预测']:
        out=OUTPUT/'02_全流程分析'/task
        result=json.loads((out/'最终测试评估_v1.json').read_text(encoding='utf-8'))
        lock=json.loads((out/'模型与阈值锁定_v1.json').read_text(encoding='utf-8'))
        comparison=pd.read_csv(out/'模型对比_v1.csv');row=comparison[comparison.model==result['model']].iloc[0]
        card.append(f"\n## {task}\n\n冠军：{result['model']}。阈值：{result['threshold']:.6f}。保留特征{len(lock['retained_features'])}个。参数和列名见模型与阈值锁定_v1.json。\n\n选择方法：{result['selection']}\n\nCV ROC-AUC={row.roc_auc_mean:.4f}，PR-AUC(AP)={row.pr_auc_ap_mean:.4f}，F1={row.f1_mean:.4f}。最终测试ROC-AUC={result['roc_auc']:.4f}，PR-AUC(AP)={result['pr_auc_ap']:.4f}，正类基线={result['positive_baseline']:.4f}，召回={result['recall']:.4f}，Brier={result['brier']:.4f}。\n\n最终终评时间：{result['time_beijing']}。测试集未用于选择模型或阈值。重要性为训练模型的gain，分裂次数另表保存，均不代表因果效应。")
        log.append(f"\n## {task}\n\n输入版本：课程panel_ml.csv或bankruptcy_data.csv；指纹见来源文件指纹_v1.csv。预处理与候选配置见03脚本。CV参数搜索结果见参数搜索记录_v1.csv；测试锁定与终评时间分别为{lock['time_beijing']}、{result['time_beijing']}。冠军{result['model']}，固定阈值{result['threshold']:.6f}，测试ROC-AUC{result['roc_auc']:.6f}。没有将测试结果返回调参。")
    card.append("\n## 应用树及限制\n\n任务b另用全部42,511条历史记录拟合深度5、31叶的balanced决策树，预测5,574家公司。它与上面的任务a冠军不同，不能借用冠军测试成绩作为自身准确率。应用集无真实标签，未知未来准确率。缺失用历史中位数；网页保留原始缺失提示。\n\n历史标签、年度对齐、右截断、样本选择和未来分布变化限制结论。具体校准曲线及误报/漏报逐条记录均已保留。当前只作本地课程审阅，公开发布等待用户确认数据许可。")
    (OUTPUT/'02_全流程分析/MODEL_CARD.md').write_text('\n'.join(card),encoding='utf-8')
    (LOGS/'experiment_log.md').write_text('\n'.join(log),encoding='utf-8')


if __name__=='__main__':main()
