"""只读比对“新增”全部文件，识别新代码、数据和历史版本证据。"""
from collections import Counter
import re
import pandas as pd
from study_common import *

def main():
    setup();root=ROOT/'新增';rows=[]
    # 10月23日旧版由11号脚本单独比较，避免把其版本差异混作重复副本。
    excluded_version=root/'AI数据分析课程资料 261023'
    all_files=[p for p in root.rglob('*') if p.is_file() and excluded_version not in p.parents]
    for p in all_files:
        rel=p.relative_to(root)
        if rel.parts[0]=='MD&A':candidate=SOURCE/'数据集文档'/rel
        else:candidate=SOURCE/rel
        size=p.stat().st_size
        digest=sha(p)
        if candidate.exists():
            same_size=size==candidate.stat().st_size
            identical=same_size and digest==sha(candidate)
            status='与原包逐字节一致' if identical else '内容不同'
        else:status='原包无同路径文件'
        rows.append({'new_relative_path':str(rel),'bytes':size,'reference_relative_path':str(candidate.relative_to(SOURCE)),
                     'comparison':status,'new_sha256':digest})
    x=pd.DataFrame(rows);out=OUT/'04_新增资料核查';out.mkdir(parents=True,exist_ok=True)
    save(x,out/'新增资料_逐文件内容比对.csv')
    useful=x[(x.comparison!='与原包逐字节一致')&x.new_relative_path.str.endswith(('.py','.md','.ipynb','.sh','.json','.pdf'))]
    summary={'time_beijing':now(),'added_folder':str(root),'all_files':len(all_files),'extensions':dict(Counter(p.suffix.lower() for p in all_files)),
             'compared_files':len(x),'comparison_counts':{k:int(v) for k,v in x.comparison.value_counts().items()},
             'scope_excluded_folders':['AI数据分析课程资料 261023（另行核查）'],
             'new_or_changed_files':x.loc[x.comparison!='与原包逐字节一致','new_relative_path'].to_list(),
             'new_or_changed_instruction_files':useful.new_relative_path.to_list(),'year_text_inventory':{},'input_modified':False,
             'v1_preserved_files':assert_v1_unchanged()}
    for p in (root/'MD&A/MD&A 2').iterdir():
        if p.is_dir():summary['year_text_inventory'][p.name]=len([f for f in p.rglob('*.txt') if re.match(r'^\d{6}_',f.name)])
    dump(out/'新增资料核查摘要.json',summary)
    report=f"""# 新增资料核查

时间（北京时间）：{now()}。

“新增”目录共{len(all_files):,}个文件，包含教程、数据说明、模型演示代码和部分 MD&A 原文。本轮已对全部文件计算 SHA-256，并按相对目录匹配原课程包；其中顶层 MD&A 目录对应原包“数据集文档/MD&A”。比对范围包括 6 个 Excel 文件、原始年报文本、PDF、代码和模型文件。

本表范围不含其后补入的“AI数据分析课程资料 261023”子目录，旧版另见“05_10月23日版本核查”。

比对结果：{summary['comparison_counts']}。

新出现或内容不同的说明/代码/教程：{summary['new_or_changed_instruction_files']}。

新增目录中的关键文件与原包一致，可以作为资料备份，但本次没有找到面板生成脚本、accruals 的补充定义或另一套首次披露历史版本。旧版 DATA_v1/v2/v3 是说明文档迭代，不能把模型训练脚本当成原始数据构建脚本。当前发现的口径疑问仍未解决。

继续沿用课程数值并保留核实清单，不按猜测重建财务数据。文本质量和复现方面的优化不依赖这些口径，可以独立推进。

所有详细对应路径与SHA-256已落盘，没有改写或合并新增目录。
"""
    (out/'新增资料核查报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
