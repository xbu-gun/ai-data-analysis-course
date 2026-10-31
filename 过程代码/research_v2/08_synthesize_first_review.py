"""汇总首轮核查，按年度核算预算，并列明尚未解决的口径问题。"""
import json,sys
import numpy as np
import pandas as pd
from study_common import *
from importlib import import_module

def main():
    setup();source_out=OUT/'01_来源与口径核查'
    # 训练期与无标签应用集的结构诊断；不涉及v1的2020—2021留出集。
    tr=pd.read_csv(DATA/'舞弊多年份/合并后数据/panel_ml.csv');tr=tr[tr.Year<=2019]
    app=pd.read_csv(DATA/'舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv')
    structure=[]
    for c in app.columns:
        if c in ['Stkcd','Year']:continue
        structure.append({'feature':c,'historical_train_unique':int(tr[c].nunique()),'application_unique':int(app[c].nunique()),
                          'historical_missing':float(tr[c].isna().mean()),'application_missing':float(app[c].isna().mean()),
                          'application_median':float(app[c].median())})
    save(pd.DataFrame(structure),source_out/'训练期与应用集_缺失和常量诊断.csv')
    # 预算在每个年度内独立计算，不跨年度、跨模型混排分数。
    backtest=import_module('04_application_tree_backtest')
    directory=next((OUT/'02_网站树历史回测').glob('*/fold_predictions.csv')).parent
    pred=pd.read_csv(directory/'fold_predictions.csv',dtype={'Stkcd':str})
    budgets=[]
    for year,g in pred.groupby('Year'):
        for budget in [.05,.10]:budgets.append({'year':int(year),**backtest.quota(g.is_fraud.to_numpy(),g.score.to_numpy(),budget)})
    yearly=pd.DataFrame(budgets);save(yearly,directory/'按年度核查预算_同分范围与期望.csv')
    total_positive=int(pred.is_fraud.sum());aggregate=[]
    for budget,g in yearly.groupby('budget'):
        quota=int(g.quota.sum());low=float((g.precision_min*g.quota).sum());high=float((g.precision_max*g.quota).sum());expected=float((g.precision_expected_uniform_tie_sampling*g.quota).sum())
        aggregate.append({'budget_per_year':budget,'total_review_quota':quota,'expected_positives_uniform_boundary_sampling':expected,
                          'precision_expected':expected/quota,'precision_min':low/quota,'precision_max':high/quota,
                          'recall_expected':expected/total_positive,'recall_min':low/total_positive,'recall_max':high/total_positive,
                          'positive_baseline':float(pred.is_fraud.mean()),'years':len(g)})
    budget_total=pd.DataFrame(aggregate);save(budget_total,directory/'按年度预算汇总.csv')
    fold=pd.read_csv(directory/'逐折与汇总回测指标.csv')
    fold=fold[(fold.decision_rule=='website_trust<60')&(fold.fold!='pooled')]
    bank=json.loads((source_out/'破产数据来源核查摘要.json').read_text(encoding='utf-8'))
    finance=json.loads((source_out/'财务原始口径抽查摘要.json').read_text(encoding='utf-8'))
    scaling=json.loads((OUT/'03_破产聚类标准化对照/标准化对照摘要.json').read_text(encoding='utf-8'))
    tree=json.loads((directory/'回测摘要.json').read_text(encoding='utf-8'))
    first=budget_total.iloc[0];second=budget_total.iloc[1]
    summary={'time_beijing':now(),'v1_preserved_files':assert_v1_unchanged(),
             'bankruptcy_official_numeric_equal':bank['all_numeric_cells_equal'],'largest_v1_distance_contribution':bank['largest_contribution_share'],
             'largest_quantile_distance_contribution':scaling['max_distance_contribution_share'],'quantile_k2_ari_vs_v1':scaling['k2_k4_diagnostics'][0]['ari_vs_v1_robust'],
             'annual_alignment_sample_matches':finance['expected_year_size_assets_matches'],'sample_target_records':finance['sample_target_records'],
             'late_correction_cases':finance['late_correction_cases'],
             'application_tree_mean_fold_roc_auc':float(fold.roc_auc.mean()),'application_tree_sd_fold_roc_auc':float(fold.roc_auc.std(ddof=1)),
             'application_tree_pooled_threshold_metrics':tree['pooled_website_rule'],
             'year_separate_budget_results':aggregate,'original_2020_2021_holdout_used':False,
             'source_data_rebuilt':False,'financial_formula_or_asof_issues_resolved':False}
    added_path=OUT/'04_新增资料核查/新增资料核查摘要.json'
    added=json.loads(added_path.read_text(encoding='utf-8')) if added_path.exists() else None
    if added is not None:
        summary['added_material_review']={k:added[k] for k in ['all_files','compared_files','comparison_counts','new_or_changed_files']}
    added_note=(f"用户补充的‘新增’目录已复核：全部 {added['compared_files']:,} 个文件与原课程包对应文件逐字节一致，是原资料的部分副本。本次未补足面板生成脚本、指标口径或另一套历史披露版本；详细路径和 SHA-256 见 04_新增资料核查。" if added and not added['new_or_changed_files'] else '')
    old_path=OUT/'05_10月23日版本核查/10月23日版本核查摘要.json'
    if old_path.exists():
        older=json.loads(old_path.read_text(encoding='utf-8'))
        summary['older_course_review']={k:older[k] for k in ['files','comparison_counts','data_and_rule_documents','python_scripts','panel_files_present','review_conclusion']}
        added_note+='\n\n其后补入的 10 月 23 日版另含 89 个文件，其中 83 个与新版逐字节一致。6 个差异文件已复核：主要是课件增补及示例路径、排版变化；13 份数据与规则说明、5 份脚本均一致，未补足财务口径证据。第 53 页的四项作业是新版增补，继续以新版交付要求为准。'
    dump(OUT/'首轮优化核查摘要.json',summary)
    questions="""# 向老师核实的口径清单

本清单仅请求说明，不预设课程数据错误。

1. 能否提供panel_ml.csv和panel_ml_2025_unlabeled.csv的生成脚本，或逐字段的最终公式？目前包内DATA.md与若干列的抽查数值不一致。
2. ar_ratio和inv_ratio的分母是总资产、营业收入还是营业总收入？ap_ratio是否以营业成本为分母？历史非缺失抽查中，收入/成本分母吻合，资产分母不吻合。
3. 历史accruals在12条抽查中都吻合(净利润−营业利润)/总资产，应用集的可比较记录又不吻合这一式。两份表的定义分别是什么？应怎样解释其经济含义？
4. 历史size_revenue等列更吻合营业总收入，应用集的部分列更吻合营业收入。贵州茅台和银行等两者不同的公司应如何统一训练与应用口径？
5. 原始表中的IfCorrect和DeclareDate怎样处理？例如康美2018年报有2020-06-18更正披露，平安银行2011年报有2014-10-24更正披露。面板财务数使用首次披露值还是更正后数？是否有历史时点版本可核对？
6. 破产数据已与UCI官方逐单元核对一致；某些比率数值极大，是否有额外的单位/编码说明？RobustScaler后单列贡献约99.4%的平方距离，分位数标准化对照与原分组几乎不一致。课程建议如何解释这一现象？

证据在本目录01_来源与口径核查：原始Excel候选行、原始行号、候选公式逐项匹配、晚于预测时点的更正记录和UCI比对表。未修改课程输入。
"""
    (OUT/'向老师核实的口径清单.md').write_text(questions,encoding='utf-8')
    report=f"""# 第二轮优化：第一阶段研究记录

时间（北京时间）：{now()}。

## 这轮完成了什么

本轮先核实原始数据与指标口径，再检验网站实际使用的加权决策树，并按用户确认增加破产数据的分位数标准化对照。v1的{summary['v1_preserved_files']}份交付与代码文件指纹未变化，原始数据未改，2020—2021留出数据未用于新增评估。

{added_note}

## 1. 破产数据：原始数值一致，原分组缺少标准化稳健性

课程文件与UCI官方数据6,819行、96列的数值全部一致，14列名称有改写。巨大数值并非本项目读取造成。原单位或编码的详细机制仍未查明。

v1训练空间中，current asset turnover rate单列贡献{bank['largest_contribution_share']:.2%}的平方距离；分位数映射后最大单列贡献{scaling['max_distance_contribution_share']:.2%}。k=2与原分组的ARI为{scaling['k2_k4_diagnostics'][0]['ari_vs_v1_robust']:.4f}，说明两种表示下分组几乎不一致。当前证据不足以把原分组解释成稳定的综合财务形态。

分位数映射改变了距离含义，并不是自动修复真实单位。本对照用于揭示敏感性，不因某种处理的标签关联更高就宣称能预测破产。

## 2. 财务面板：年度对齐通过抽查，公式和时点版本仍待核实

4家公司16个目标记录的总资产对数都匹配目标年前一年的原始年报。该结果支持年度对齐，但不等同于所有字段都已验证。

候选公式核对发现：历史应收/存货指标更吻合以收入为分母，应付指标更吻合以营业成本为分母；这些与字典所写的总资产分母不同。历史accruals的12个抽查值吻合(净利润−营业利润)/总资产；应用集3个可比较值均不吻合该候选式。部分收入相关列在历史与应用集中也有营业总收入/营业收入口径差异。

两条历史抽查记录带有晚于模拟预测时点的更正披露。仅凭标记不能断言面板数值一定是事后更正数，但目前缺少能排除这一风险的首次披露版本证据。因此应把当前实验限定为“课程提供财务版本上的历史时间前推检验”，暂不作真实事前预警能力的强表述。

## 3. 网站树：存在历史排序信号，固定提示规则覆盖面很宽

沿用深度5、叶节点至少100条、balanced和种子42，不调参，只在原训练期做五折时间前推。用户确认前四折停用全空研发列，其余保持原方法。

各折ROC-AUC均值{fold.roc_auc.mean():.4f}、折间标准差{fold.roc_auc.std(ddof=1):.4f}。混合历史折外预测的ROC-AUC为{tree['pooled_website_rule']['roc_auc']:.4f}、PR-AUC(AP)为{tree['pooled_website_rule']['pr_auc_ap']:.4f}，正类基线{tree['pooled_website_rule']['positive_baseline']:.4f}。混合指标仅为历史概括，各折属于不同模型和不同年份。

沿用网站可信度<60的提示规则，历史总体召回约{tree['pooled_website_rule']['recall']:.1%}、精确率约{tree['pooled_website_rule']['precision']:.1%}，会提示{tree['pooled_website_rule']['flagged_share']:.1%}的样本。它能集中部分线索，但不能描述成少量、精准的风险警报；加权分数也不是校准概率。

## 4. 每年只核查一部分公司时

年度内分别排序，避免不同年份模型的分数混排。若每年核查前5%，同分边界均匀抽取的汇总期望精确率{first.precision_expected:.1%}、召回{first.recall_expected:.1%}；前10%对应期望精确率{second.precision_expected:.1%}、召回{second.recall_expected:.1%}。完整同分上下界及包括全部同分样本所需规模另表保存。

这里是固定历史预测上的数学期望，不是实际抽样已经取得的成绩；同一公司跨年可重复出现，单位是公司年度。由于口径和时点版本仍待核实，也不能把它当作2025应用集的已知表现。

## 下一步及停止点

可以继续做不依赖财务口径的文本质量与复现优化。财务特征重建、替换主模型或把新版结果写成最终事前预测结论，应先取得构建脚本/口径说明，或明确接受保留课堂数值并限制解释范围。当前没有按猜测重算或删除公司。
"""
    (OUT/'第一阶段核查与回测报告.md').write_text(report,encoding='utf-8')
    text_report=OUT/'06_全文主题对照/第二阶段_全文主题对照报告.md'
    text_link='- [第二阶段：全文主题对照报告](06_全文主题对照/第二阶段_全文主题对照报告.md)' if text_report.exists() else ''
    culture_link='- [文化指标复算、贡献分解与原文语境](07_文化指标解释/文化指标贡献解释.md)' if (OUT/'07_文化指标解释/文化指标贡献解释.md').exists() else ''
    index=f"""# 第二轮优化交付入口

更新时间（北京时间）：{now()}。本轮结果与v1并列，未覆盖v1。

- [先读：第一阶段核查与回测报告](第一阶段核查与回测报告.md)
- [向老师核实的口径清单](向老师核实的口径清单.md)
- [财务原始口径抽查](01_来源与口径核查/财务原始口径抽查报告.md)
- [资料字典与面板口径差异](01_来源与口径核查/资料字典与面板口径差异_待核实.md)
- [破产数据来源与量纲核查](01_来源与口径核查/破产数据来源与量纲核查.md)
- [网站树历史回测入口](02_网站树历史回测/回测入口.md)
- [破产聚类标准化敏感性对照](03_破产聚类标准化对照/标准化敏感性分析报告.md)
- [用户补充资料核查](04_新增资料核查/新增资料核查报告.md)
- [10 月 23 日旧版差异核查](05_10月23日版本核查/10月23日版本核查报告.md)
{text_link}
{culture_link}

本阶段发现了实质数据口径问题，不能仅靠继续调参解决。新增模型结果只来自原训练期，不使用已终评留出数据；单位问题和历史时点版本仍待外部说明。
"""
    (OUT/'00_第二轮交付入口.md').write_text(index,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
