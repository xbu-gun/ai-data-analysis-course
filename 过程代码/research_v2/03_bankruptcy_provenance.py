"""全表仅比来源一致性，分布和距离诊断只用原训练样本。"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from study_common import *

def main():
    setup();out=OUT/'01_来源与口径核查';out.mkdir(parents=True,exist_ok=True)
    course_path=DATA/'破产预测/bankruptcy_data.csv'
    reference=WORK/'external_sources/uci572/data.csv'
    course=pd.read_csv(course_path);official=pd.read_csv(reference)
    normalize=lambda s:''.join(str(s).split()).casefold()
    names_match=[normalize(a)==normalize(b) for a,b in zip(course.columns,official.columns)]
    aliases={
        7:('tax Pre-net interest rate','Pre-tax net Interest Rate'),
        16:('per Net Share Value (B)','Net Value Per Share (B)'),
        21:('Revenue Per Share (Yuan)','Revenue Per Share (Yuan ¥)'),
        22:('Operating Profit Per Share (Yuan)','Operating Profit Per Share (Yuan ¥)'),
        23:('Per Share Net profit before tax (yuan)','Per Share Net profit before tax (Yuan ¥)'),
        55:('Quick asset/Total asset','Quick Assets/Total Assets'),
        58:('Quick asset /current liabilities','Quick Assets/Current Liability'),
        64:('current liability / liability','Current Liabilities/Liability'),
        66:('current liability/equity','Current Liabilities/Equity'),
        68:('Retained Earnings/Total assets','Retained Earnings to Total Assets'),
        76:('fix assets to assets','Fixed Assets to Assets'),
        84:('current liabilities to current assets','Current Liability to Current Assets'),
        85:('one if total liabilities exceeds total assets zero otherwise','Liability-Assets Flag'),
        94:('one if net income was negative for the last two year zero otherwise','Net Income Flag')}
    assert course.shape==official.shape
    for i,ok in enumerate(names_match):
        if not ok:
            assert i in aliases and tuple(normalize(s) for s in (course.columns[i],official.columns[i]))==tuple(normalize(s) for s in aliases[i]),'未审阅的字段对应差异'
    A=course.to_numpy(float);B=official.to_numpy(float)
    same=np.isclose(A,B,rtol=1e-12,atol=1e-14,equal_nan=True)
    compare=pd.DataFrame({'course_column':course.columns,'uci_column':official.columns,'equal_cells':same.sum(axis=0),'different_cells':(~same).sum(axis=0)})
    save(compare,out/'破产数据_与UCI逐列比对.csv')
    course.columns=course.columns.str.strip()
    splits=json.loads((CODE/'中间数据/split_indices_v1.json').read_text(encoding='utf-8'))
    x=course.iloc[splits['bankruptcy']['train']].drop(columns='Bankrupt?').replace([np.inf,-np.inf],np.nan)
    x=x.loc[:,(x.isna().mean()<=.5)&(x.nunique()>1)]
    clean=x.clip(x.quantile(.01),x.quantile(.99),axis=1).fillna(x.median())
    scaler=RobustScaler().fit(clean);z=scaler.transform(clean)
    variance=np.var(z,axis=0);share=variance/variance.sum()
    diag=pd.DataFrame({'feature':clean.columns,'q25':clean.quantile(.25).values,'q75':clean.quantile(.75).values,
                       'iqr':(clean.quantile(.75)-clean.quantile(.25)).values,'scaler_scale':scaler.scale_,
                       'scaled_variance':variance,'squared_distance_contribution_share':share,
                       'training_min':x.min().values,'training_median':x.median().values,'training_max':x.max().values})
    diag=diag.sort_values('squared_distance_contribution_share',ascending=False)
    save(diag,out/'破产训练集_距离贡献与量纲诊断.csv')
    largest=diag.iloc[0]
    summary={'time_beijing':now(),'official_landing_page':PROTOCOL['bankruptcy_source_check']['official_landing_page'],
             'course_sha256':sha(course_path),'reference_sha256':sha(reference),'rows':len(course),'columns':len(course.columns),
             'all_numeric_cells_equal':bool(same.all()),'different_cells':int((~same).sum()),
             'header_names_equal_ignoring_case_and_spaces':all(names_match),'train_only_rows':len(x),
             'explicit_header_differences':len(aliases),
             'largest_distance_contributor':str(largest.feature),'largest_contribution_share':float(largest.squared_distance_contribution_share),
             'top3_contribution_share':float(diag.head(3).squared_distance_contribution_share.sum()),
             'units_or_encoding_explained_by_official_metadata':False,'input_modified':False,'holdout_used_for_distribution_diagnosis':False}
    dump(out/'破产数据来源核查摘要.json',summary)
    text=f"""# 破产数据来源与量纲核查

时间（北京时间）：{now()}。

## 来源比对

课程文件与UCI官方下载文件逐行逐列数值比对：{summary['all_numeric_cells_equal']}；不同单元格{summary['different_cells']}个。比较容差rtol=1e-12、atol=1e-14。82列名称忽略大小写和空白后一致，14列存在已逐项列出的名称改写，按固定列序核对数值。原始课程文件没有修改。

因此，若数值一致，巨大数值至少不是本项目读取或预处理造成的；这不等于已经确认原始数值的单位和业务合理性。UCI页面的变量描述未提供足以还原异常数值单位的详细规则。课程改写后的变量名（尤其二元标记的完整文字定义）也不能仅凭数值相同就视为定义已验证。

## 仅训练部分的距离诊断

按v1相同的1%/99%缩尾、中位数填补和RobustScaler重建训练空间。训练部分{len(x)}行。各特征标准化后方差占比对应随机样本对的期望平方欧氏距离贡献。

最高贡献列为{largest.feature}，占{largest.squared_distance_contribution_share:.2%}；前三列合计{summary['top3_contribution_share']:.2%}。这是对表示的诊断，并非新增模型的效果成绩。

若贡献高度集中，清晰的聚类可能主要反映少量变量的数值量级。需要将其与其他标准化方法作敏感性对照后，才能稳妥解释财务形态。尚未据此删除、缩放修正或猜测任何原始数值。

## 待解决事项

1. 可疑比率的具体单位/编码机制仍缺少权威说明；可向老师或数据提供者核实。
2. 若要增加按秩变换、其他标准化或剔除可疑列的对照，应先确认方案、独立保存结果，不覆盖v1。

来源：[UCI Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)。官方下载及文件指纹见过程代码research_v2/work/external_sources/uci572/retrieval_manifest.json。
"""
    (out/'破产数据来源与量纲核查.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
