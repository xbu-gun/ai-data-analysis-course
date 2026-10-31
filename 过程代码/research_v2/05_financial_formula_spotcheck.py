"""按原始财务行重算抽查指标，区分年度对齐与事前可得性。"""
import json,math,re
from itertools import product
import numpy as np
import pandas as pd
from study_common import *

def number(record,key):
    try:return float(record.get(key,''))
    except (ValueError,TypeError):return np.nan

def safe_ratio(a,b):return a/b if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan

def calculate(b,i,prev_b=None,prev_i=None):
    n=lambda key:number(b,key);m=lambda key:number(i,key)
    A=n('A001000000');R=m('B001101000');NI=m('B002000000')
    values={'size_assets':math.log(A) if A>0 else np.nan,'size_revenue':math.log(R) if R>0 else np.nan,
            'lev':safe_ratio(n('A002000000'),A),'current_ratio':safe_ratio(n('A001100000'),n('A002100000')),
            'quick_ratio':safe_ratio(n('A001100000')-n('A001123000'),n('A002100000')),
            'st_debt_ratio':safe_ratio(n('A002101000'),A),'roa':safe_ratio(NI,A),'roe':safe_ratio(NI,n('A003000000')),
            'net_margin':safe_ratio(NI,R),'op_margin':safe_ratio(m('B001300000'),R),'profit_to_rev':safe_ratio(m('B001000000'),R),
            'cost_ratio':safe_ratio(m('B001201000'),R),'tax_ratio':safe_ratio(m('B002100000'),m('B001000000')),
            'ar_ratio':safe_ratio(n('A001111000'),A),'inv_ratio':safe_ratio(n('A001123000'),A),
            'ap_ratio':safe_ratio(n('A002108000'),A),'other_ar_ratio':safe_ratio(n('A001121000'),A),
            'asset_turnover':safe_ratio(R,A),'cur_asset_share':safe_ratio(n('A001100000'),A),
            'fixed_asset_share':safe_ratio(n('A001212000'),A),'intangible_share':safe_ratio(n('A001218000'),A),
            'inventory_share':safe_ratio(n('A001123000'),A),'cash_share':safe_ratio(n('A001101000'),A),
            'equity_ratio':safe_ratio(n('A003000000'),A),
            'selling_ratio':safe_ratio(m('B001209000'),R),'admin_ratio':safe_ratio(m('B001210000'),R),
            'fin_exp_ratio':safe_ratio(m('B001211000'),R),'rd_ratio':safe_ratio(m('B001216000'),R)}
    if prev_b is not None and prev_i is not None:
        for name,current,previous in [('rev_growth',R,number(prev_i,'B001101000')),('asset_growth',A,number(prev_b,'A001000000')),
                ('profit_growth',NI,number(prev_i,'B002000000')),('ar_growth',n('A001111000'),number(prev_b,'A001111000')),
                ('inv_growth',n('A001123000'),number(prev_b,'A001123000'))]:
            values[name]=safe_ratio(current,previous)-1
        values['ar_rev_gap']=values['ar_growth']-values['rev_growth'];values['inv_rev_gap']=values['inv_growth']-values['rev_growth']
        delta_wc=(n('A001100000')-n('A002100000'))-(number(prev_b,'A001100000')-number(prev_b,'A002100000'))
        values['accruals']=safe_ratio(NI-delta_wc,A)
    return values

def main():
    setup();out=OUT/'01_来源与口径核查';out.mkdir(parents=True,exist_ok=True)
    meta=[]
    for fn in ['FS_Combas.xlsx','FS_Comins.xlsx']:
        obj=json.loads((WORK/f'raw_finance/{fn}.selected.json').read_text(encoding='utf-8'))
        meta.append(obj);save(pd.DataFrame(obj['selected_rows']),out/f'{fn}_抽查原始候选行.csv')
    balance,income=[x['selected_rows'] for x in meta]
    index=lambda rows:{(c,y):[r for r in rows if r['Stkcd']==c and r['Accper']==f'{y}-12-31']
                       for c in PROTOCOL['raw_finance_spotcheck']['company_codes'] for y in range(2005,2026)}
    bi,ii=index(balance),index(income)
    panel=pd.read_csv(DATA/'舞弊多年份/合并后数据/panel_ml.csv',dtype={'Stkcd':str});panel.Stkcd=panel.Stkcd.str.zfill(6)
    panel=panel[panel.Year.isin([2007,2012,2019])&panel.Stkcd.isin(PROTOCOL['raw_finance_spotcheck']['company_codes'])]
    app=pd.read_csv(DATA/'舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv',dtype={'Stkcd':str});app.Stkcd=app.Stkcd.str.zfill(6)
    app=app[app.Stkcd.isin(PROTOCOL['raw_finance_spotcheck']['company_codes'])]
    cases=[('historical_panel',r) for _,r in panel.iterrows()]+[('application_2025',r) for _,r in app.iterrows()]
    rows=[];summaries=[]
    for kind,s in cases:
        target=int(s.Year);expected=target-1;c=s.Stkcd
        candidates=[target-1,target] if kind=='historical_panel' else [2024,2025]
        for fy in candidates:
            for b,i in product(bi.get((c,fy),[]),ii.get((c,fy),[])):
                prev_b=bi.get((c,fy-1),[]);prev_i=ii.get((c,fy-1),[])
                values=calculate(b,i,prev_b[0] if len(prev_b)==1 else None,prev_i[0] if len(prev_i)==1 else None)
                good=bad=missing=0
                for feature,value in values.items():
                    if feature not in s.index:continue
                    supplied=float(s[feature]);finite=np.isfinite(value) and np.isfinite(supplied)
                    match=bool(np.isclose(value,supplied,rtol=1e-6,atol=1e-8)) if finite else None
                    if finite:
                        if match:good+=1
                        else:bad+=1
                    elif not np.isfinite(value) and not np.isfinite(supplied):missing+=1
                    status='数值一致' if match else '数值不一致' if finite else '双方缺失' if not np.isfinite(value) and not np.isfinite(supplied) else '缺失状态不同'
                    rows.append({'dataset':kind,'Stkcd':c,'target_year':target,'candidate_financial_year':fy,'expected_financial_year':expected,
                        'feature':feature,'supplied':supplied,'recomputed_from_raw':value,'status':status,
                        'balance_excel_row':b['source_excel_row'],'income_excel_row':i['source_excel_row']})
                dates=sorted(set(re.findall(r'\d{4}-\d{2}-\d{2}',b.get('DeclareDate','')+' '+i.get('DeclareDate',''))))
                late=[d for d in dates if d>f'{target}-04-30']
                summaries.append({'dataset':kind,'Stkcd':c,'target_year':target,'candidate_financial_year':fy,'expected_year':fy==expected,
                    'numeric_matches':good,'numeric_mismatches':bad,'both_missing':missing,'comparable_numeric':good+bad,
                    'size_assets_matches':bool(np.isclose(values['size_assets'],s.size_assets,rtol=1e-10,atol=1e-10)),
                    'balance_corrected_flag':b.get('IfCorrect',''),'income_corrected_flag':i.get('IfCorrect',''),
                    'correction_disclosure_dates':' ; '.join(dates),'correction_dates_after_target_april30':' ; '.join(late),
                    'historical_asof_numbers_verified':False,'balance_excel_row':b['source_excel_row'],'income_excel_row':i['source_excel_row']})
    detail=pd.DataFrame(rows);summary=pd.DataFrame(summaries);save(detail,out/'财务抽查_逐指标与候选年份.csv');save(summary,out/'财务抽查_年度与更正信息.csv')
    expected_summary=summary[summary.expected_year];expected_rows=detail[detail.candidate_financial_year==detail.expected_financial_year]
    mismatch=expected_rows[expected_rows.status.isin(['数值不一致','缺失状态不同'])]
    save(mismatch,out/'财务抽查_需要解释的指标差异.csv')
    late=expected_summary[expected_summary.correction_dates_after_target_april30!='']
    result={'time_beijing':now(),'sample_company_codes':PROTOCOL['raw_finance_spotcheck']['company_codes'],'sample_target_records':len(cases),
            'expected_year_size_assets_matches':int(expected_summary.size_assets_matches.sum()),'expected_year_candidates':len(expected_summary),
            'expected_year_numeric_matches':int((expected_rows.status=='数值一致').sum()),'expected_year_numeric_mismatches':int((expected_rows.status=='数值不一致').sum()),
            'expected_year_missing_status_differences':int((expected_rows.status=='缺失状态不同').sum()),
            'late_correction_case_count':len(late),'late_correction_cases':late[['dataset','Stkcd','target_year','candidate_financial_year','correction_dates_after_target_april30']].to_dict('records'),
            'all_original_financial_values_point_in_time_verified':False,'raw_inputs_modified':False,'historical_test_years_2020_2021_audited':False}
    dump(out/'财务原始口径抽查摘要.json',result)
    late_lines='\n'.join(f"- {r.Stkcd}：目标年{r.target_year}，财务年{r.candidate_financial_year}，更正披露{r.correction_dates_after_target_april30}。" for r in late.itertuples(index=False)) or '本次抽查未检出晚于简化预测时点的更正标记；这不构成完整时点版本验证。'
    (out/'财务原始口径抽查报告.md').write_text(f"""# 财务原始表口径抽查

时间（北京时间）：{now()}。抽查4家公司：平安银行、万科、康美药业、贵州茅台；历史目标年2007、2012、2019，另核对2025年报应用记录。未读取2020—2021留出样本作此核查。

## 年度对应

共{len(cases)}个目标记录；预期财务年为目标年减1。总资产对数的预期年度匹配{result['expected_year_size_assets_matches']}/{len(expected_summary)}。例如历史Year=2019应匹配2018年报；应用Year=2026应匹配2025年报。结果只支持本次抽查，不能等同全表所有公司和指标均已验证。

## 指标重算

从原始两张表的字段重算静态财务比率和能够以连续年报核算的增长率；公式、原始Excel行号、实际输入值和差异逐项保存。预期年份数值一致{result['expected_year_numeric_matches']}项，数值不一致{result['expected_year_numeric_mismatches']}项，缺失状态不同{result['expected_year_missing_status_differences']}项。双方缺失不计为数值证据。

accruals采用课程文字说明中“净利润减营运资本变动后除以总资产”的候选公式。若不一致，表示文字公式或面板实际构建口径需进一步查明，不能自行用候选公式替换。

## 年度对齐与事前可得性分别判断

原始记录中的IfCorrect和DeclareDate是差错更正信息，不是原始年报首次公布时间。晚于目标年4月30日的更正案例：

{late_lines}

这提示存在事后信息风险，但仅凭更正标记还不能断言当前字段一定已被替换成更正数。本包没有能直接还原所有首次披露数值的历史版本，所以“特征年早于目标年”已抽查，与“这些数在当时可得”尚未完全确认，是两项不同结论。

## 下一步

需要面板构建脚本或字段口径确认，并视情况取得首次披露/历史时点版本。当前不改原始面板、不删除公司、不重写既有结果；网站树回测应暂表述为课程提供财务版本上的历史时间前推检验。
""",encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
