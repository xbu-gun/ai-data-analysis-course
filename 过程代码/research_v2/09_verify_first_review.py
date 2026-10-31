"""核查回测时间边界、缓存指纹、预算同分公式及v1保留状态。"""
import ast,hashlib,importlib,inspect,json
import numpy as np
import pandas as pd
from study_common import *

def main():
    setup();module=importlib.import_module('04_application_tree_backtest')
    folder=next((OUT/'02_网站树历史回测').glob('*/fold_predictions.csv')).parent
    manifest=json.loads((folder/'experiment_manifest.json').read_text(encoding='utf-8'))
    pred=pd.read_csv(folder/'fold_predictions.csv',dtype={'Stkcd':str})
    splits=json.loads((CODE/'中间数据/split_indices_v1.json').read_text(encoding='utf-8'))
    assert set(pred.source_row)<=set(splits['fraud']['train'])
    assert not set(pred.source_row)&set(splits['fraud']['test'])
    assert pred.source_row.is_unique and pred.Year.between(2010,2019).all()
    for fold in manifest['folds']:
        assert max(fold['train_years'])<min(fold['validation_years'])
        assert len(set(fold['train_years'])&set(fold['validation_years']))==0
        assert fold['disabled_all_missing_features']==(['rd_ratio'] if fold['fold']<=4 else [])
        assert all(v is not None and np.isfinite(v) for v in fold['medians'].values())
    source=DATA/'舞弊多年份/合并后数据/panel_ml.csv'
    assert manifest['configuration']['source_sha256']==sha(source)
    assert manifest['configuration']['fit_implementation_sha256']==hashlib.sha256(inspect.getsource(module.fit_fold).encode()).hexdigest()
    # 全部同分的已知小例子：预算2条，1/4为正类；预期精确率1/4，范围0—1/2。
    tied=module.quota(np.array([1,0,0,0]),np.array([.5,.5,.5,.5]),.5)
    assert tied['quota']==2 and tied['boundary_tied_rows']==4
    assert tied['precision_expected_uniform_tie_sampling']==.25 and tied['precision_min']==0 and tied['precision_max']==.5
    # 无同分边界时，上下界和期望一致。
    distinct=module.quota(np.array([1,0,1,0]),np.array([.9,.8,.2,.1]),.5)
    assert distinct['precision_min']==distinct['precision_max']==distinct['precision_expected_uniform_tie_sampling']==.5
    q=pd.read_csv(folder/'按年度核查预算_同分范围与期望.csv')
    assert np.all(q.precision_min<=q.precision_expected_uniform_tie_sampling)
    assert np.all(q.precision_expected_uniform_tie_sampling<=q.precision_max)
    assert np.all(q.include_all_boundary_ties_rows>=q.quota)
    finance=pd.read_csv(OUT/'01_来源与口径核查/财务抽查_年度与更正信息.csv')
    assert finance[finance.expected_year].size_assets_matches.sum()==16
    candidate=pd.read_csv(OUT/'01_来源与口径核查/财务抽查_候选公式匹配汇总.csv')
    historic=candidate[(candidate.dataset=='historical_panel')&(candidate.feature=='accruals')]
    assert int(historic.numeric_matches.iloc[0])==12
    for p in STUDY.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    summary={'verified_beijing':now(),'v1_files_unchanged':assert_v1_unchanged(),'historical_holdout_intersection':0,
             'forward_year_boundaries_verified':True,'all_missing_column_policy_verified':True,'source_and_fit_cache_keys_verified':True,
             'known_tie_budget_examples_passed':True,'annual_budget_bounds_verified':True,'raw_financial_year_anchors_verified':16,
             'raw_financial_formulas_or_asof_issue_resolved':False,'scripts_syntax_checked':True}
    dump(OUT/'首轮优化结果核查.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
