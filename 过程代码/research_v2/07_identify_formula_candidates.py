"""对已抽查原始行识别能解释数值的候选口径，不替换课程特征。"""
import json,math
import numpy as np
import pandas as pd
from study_common import *

def val(r,k):
    try:return float(r.get(k,''))
    except (ValueError,TypeError):return np.nan

def ratio(a,b):return a/b if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan

def main():
    setup();out=OUT/'01_来源与口径核查'
    balances=json.loads((WORK/'raw_finance/FS_Combas.xlsx.selected.json').read_text(encoding='utf-8'))['selected_rows']
    incomes=json.loads((WORK/'raw_finance/FS_Comins.xlsx.selected.json').read_text(encoding='utf-8'))['selected_rows']
    records=[]
    for kind,filename in [('historical_panel','panel_ml.csv'),('application_2025','panel_ml_2025_unlabeled.csv')]:
        data=pd.read_csv(DATA/'舞弊多年份/合并后数据'/filename,dtype={'Stkcd':str});data.Stkcd=data.Stkcd.str.zfill(6)
        data=data[data.Stkcd.isin(PROTOCOL['raw_finance_spotcheck']['company_codes'])]
        if kind=='historical_panel':data=data[data.Year.isin([2007,2012,2019])]
        for _,row in data.iterrows():
            fy=int(row.Year)-1
            bs=[b for b in balances if b['Stkcd']==row.Stkcd and b['Accper']==f'{fy}-12-31']
            ins=[i for i in incomes if i['Stkcd']==row.Stkcd and i['Accper']==f'{fy}-12-31']
            if len(bs)!=1 or len(ins)!=1:continue
            b,i=bs[0],ins[0];assets=val(b,'A001000000');total=val(i,'B001100000');oper=val(i,'B001101000');cost=val(i,'B001201000')
            candidates={}
            for feature,key in [('ar_ratio','A001111000'),('inv_ratio','A001123000'),('ap_ratio','A002108000')]:
                candidates[feature]={f'{key}/A001000000':ratio(val(b,key),assets),f'{key}/B001100000':ratio(val(b,key),total),f'{key}/B001101000':ratio(val(b,key),oper)}
                if feature=='ap_ratio':candidates[feature][f'{key}/B001201000']=ratio(val(b,key),cost)
            candidates['accruals']={'(B002000000-B001300000)/A001000000':ratio(val(i,'B002000000')-val(i,'B001300000'),assets)}
            candidates['size_revenue']={'ln(B001100000)':math.log(total) if total>0 else np.nan,'ln(B001101000)':math.log(oper) if oper>0 else np.nan}
            for feature,key in [('net_margin','B002000000'),('op_margin','B001300000'),('profit_to_rev','B001000000'),
                                 ('cost_ratio','B001201000'),('selling_ratio','B001209000'),('admin_ratio','B001210000'),
                                 ('fin_exp_ratio','B001211000'),('rd_ratio','B001216000')]:
                candidates[feature]={f'{key}/B001100000':ratio(val(i,key),total),f'{key}/B001101000':ratio(val(i,key),oper)}
            candidates['asset_turnover']={'B001100000/A001000000':ratio(total,assets),'B001101000/A001000000':ratio(oper,assets)}
            for feature,options in candidates.items():
                supplied=float(row[feature])
                for formula,candidate in options.items():
                    finite=np.isfinite(supplied) and np.isfinite(candidate)
                    records.append({'dataset':kind,'Stkcd':row.Stkcd,'target_year':int(row.Year),'financial_year':fy,'feature':feature,'candidate_formula':formula,
                                    'supplied':supplied,'candidate_value':candidate,'numeric_comparable':finite,
                                    'numeric_match':bool(np.isclose(supplied,candidate,rtol=1e-6,atol=1e-8)) if finite else False,
                                    'both_missing':bool(not np.isfinite(supplied) and not np.isfinite(candidate))})
    detail=pd.DataFrame(records);save(detail,out/'财务抽查_候选公式逐项辨识.csv')
    summary=detail.groupby(['dataset','feature','candidate_formula'],sort=False).agg(sample_records=('Stkcd','size'),numeric_comparable=('numeric_comparable','sum'),numeric_matches=('numeric_match','sum'),both_missing=('both_missing','sum')).reset_index()
    save(summary,out/'财务抽查_候选公式匹配汇总.csv')
    focus=summary[summary.feature.isin(['ar_ratio','inv_ratio','ap_ratio','accruals','size_revenue'])]
    lines=[]
    for r in focus.itertuples(index=False):lines.append(f"| {r.dataset} | {r.feature} | {r.candidate_formula} | {r.numeric_matches}/{r.numeric_comparable} | {r.both_missing} |")
    text=f"""# 资料字典与面板数值：候选公式辨识

时间（北京时间）：{now()}。原始值和原始字段名均保留，未重建面板。

## 抽查范围与解释限制

使用4家公司、历史3个目标年和2025应用记录，共16条记录。表中“吻合”仅指该候选公式在可比较的抽查数值上相同；若营业收入与营业总收入相等，多个候选会同时吻合。因此不能仅凭样本数值就断言全表生成脚本的实现。

## 关键候选结果

| 数据 | 特征 | 候选公式 | 数值吻合/可比较 | 双方缺失 |
|---|---|---|---:|---:|
{chr(10).join(lines)}

## 需要向老师确认的具体问题

1. 资料字典将ar_ratio和inv_ratio写为除以总资产，但抽查更支持除以营业总收入；ap_ratio更支持除以营业成本，而非总资产。应确认原始构建代码的口径，再确定列名与经济解释。
2. 历史面板accruals的12个抽查值均吻合(净利润−营业利润)/总资产；应用集中3个可比较记录均不吻合这一公式，应用口径仍未识别。历史候选公式与资料写的“净利润减经营现金流或营运资本近似”含义不同。在得到说明前，不能把该列解释成利润中缺少现金流支持的应计项。
3. 历史size_revenue等收入相关列更支持营业总收入；2025应用记录的部分列更支持营业收入。对平安银行、贵州茅台等两类收入不同或营业收入缺失的公司，训练与应用口径可能不一致。需核对，不能只检查列名相同。
4. 已发现晚于模拟预测时点的更正披露记录。需确认课程原始表中的财务值采用首次披露版本、还是更正后的版本。

以上问题不会通过自动改分母、填补或删除公司来解决。当前回测只检验课程现有数值版本，应保留这一前提。若没有构建脚本，建议将差异作为数据审计发现，保留课程基线，暂不作真实事前预警能力的强表述。
"""
    (out/'资料字典与面板口径差异_待核实.md').write_text(text,encoding='utf-8')
    print(focus.to_string(index=False))

if __name__=='__main__':main()
