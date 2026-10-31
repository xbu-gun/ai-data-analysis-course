"""只读核对用户补充的261023旧版，保存文件指纹和文本差异。"""
from collections import Counter,defaultdict
from difflib import unified_diff
import re
from study_common import *

def pdf_key(name):
    return re.sub(r'\s*v\d+(?:\.\d+)*$', '',Path(name).stem,flags=re.I)

def pdf_text(path):
    from pypdf import PdfReader
    return '\n'.join(f'\n--- 第{i+1}页 ---\n{p.extract_text() or ""}' for i,p in enumerate(PdfReader(path).pages))

def main():
    setup();root=ROOT/'新增/AI数据分析课程资料 261023'
    out=OUT/'05_10月23日版本核查';out.mkdir(parents=True,exist_ok=True)
    evidence=WORK/'older_course_diffs';evidence.mkdir(parents=True,exist_ok=True)
    original_files=[p for p in SOURCE.rglob('*') if p.is_file()]
    by_name=defaultdict(list);by_pdf_title=defaultdict(list)
    for p in original_files:
        by_name[p.name].append(p)
        if p.suffix.lower()=='.pdf':by_pdf_title[pdf_key(p.name)].append(p)
    rows=[];text_changes=[]
    for i,p in enumerate(sorted(root.rglob('*'))):
        if not p.is_file():continue
        rel=p.relative_to(root);candidate=SOURCE/rel;method='相对路径'
        if not candidate.is_file():
            matches=by_name[p.name]
            if len(matches)==1:candidate=matches[0];method='唯一同名文件'
            elif p.suffix.lower()=='.pdf' and len(by_pdf_title[pdf_key(p.name)])==1:
                candidate=by_pdf_title[pdf_key(p.name)][0];method='课件标题去版本号'
            else:candidate=None;method='未找到对应文件'
        old_sha=sha(p);new_sha=sha(candidate) if candidate else None
        status='与10月31日版逐字节一致' if old_sha==new_sha else ('内容不同' if candidate else '未找到对应文件')
        row={'old_relative_path':str(rel),'old_bytes':p.stat().st_size,'old_sha256':old_sha,
             'reference_relative_path':str(candidate.relative_to(SOURCE)) if candidate else '',
             'reference_sha256':new_sha,'match_method':method,'comparison':status}
        if status!='与10月31日版逐字节一致' and p.suffix.lower() in {'.py','.md','.txt','.pdf'}:
            old_text=pdf_text(p) if p.suffix.lower()=='.pdf' else p.read_text(encoding='utf-8-sig',errors='replace')
            new_text=(pdf_text(candidate) if p.suffix.lower()=='.pdf' else candidate.read_text(encoding='utf-8-sig',errors='replace')) if candidate else ''
            stub=f'{len(text_changes)+1:02d}_{p.stem}'
            (evidence/f'{stub}_261023.txt').write_text(old_text,encoding='utf-8')
            (evidence/f'{stub}_261031.txt').write_text(new_text,encoding='utf-8')
            delta='\n'.join(unified_diff(old_text.splitlines(),new_text.splitlines(),fromfile=str(p),tofile=str(candidate),lineterm=''))
            (evidence/f'{stub}_diff.txt').write_text(delta,encoding='utf-8')
            row['extracted_text_equal']=old_text==new_text
            text_changes.append({'path':str(rel),'comparison':status,'old_characters':len(old_text),
                                 'new_characters':len(new_text),'extracted_text_equal':old_text==new_text,'diff_path':str(evidence/f'{stub}_diff.txt')})
        rows.append(row)
    import csv
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'10月23日版本_逐文件比对.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={'time_beijing':now(),'old_folder':str(root),'files':len(rows),'bytes':sum(r['old_bytes'] for r in rows),
             'extensions':dict(Counter(Path(r['old_relative_path']).suffix.lower() for r in rows)),
             'comparison_counts':dict(Counter(r['comparison'] for r in rows)),
             'changed_or_unmatched':[r for r in rows if r['comparison']!='与10月31日版逐字节一致'],
             'text_changes':text_changes,'v1_preserved_files':assert_v1_unchanged(),'input_modified':False}
    rule_rows=[r for r in rows if Path(r['old_relative_path']).name.startswith(('DATA','PROJECT_RULES'))]
    code_rows=[r for r in rows if Path(r['old_relative_path']).suffix.lower()=='.py']
    summary['data_and_rule_documents']={'files':len(rule_rows),'all_byte_identical':all(r['comparison']=='与10月31日版逐字节一致' for r in rule_rows)}
    summary['python_scripts']={'files':len(code_rows),'all_byte_identical':all(r['comparison']=='与10月31日版逐字节一致' for r in code_rows)}
    summary['panel_files_present']={name:(root/'数据集文档/舞弊多年份/合并后数据'/name).is_file() for name in ['panel_ml.csv','panel_ml_2025_unlabeled.csv']}
    summary['review_conclusion']='可用于课程版本记录；未补足财务面板构建脚本、最终公式或历史披露版本。'
    dump(out/'10月23日版本核查摘要.json',summary)
    report=f"""# 10 月 23 日课程版本核查

时间（北京时间）：{now()}。

## 结论

这份旧版有助于确认课程内容和作业要求的变化，但没有补足目前待核实的财务数据口径。原文件仅作只读比对，未替换新版资料，也未据此重算指标。

共核对 {len(rows)} 个文件，其中 83 个与 10 月 31 日版对应文件逐字节一致，6 个有差异。13 份 DATA / PROJECT_RULES 说明、5 份 Python 脚本全部一致。旧版没有 panel_ml.csv 和 panel_ml_2025_unlabeled.csv 本体；其中 10 份 CSV 是演示分析输出。

## 六个差异文件逐项看

| 文件 | 核对发现 | 对当前作业的影响 |
|---|---|---|
| 数据获取 browser-skill 教程 | 新版把示例用户名替换为“你的用户名”，并调整换行、分页及表头 | 没有新增财务指标构建方法 |
| data_visual_report.md | 仅示例电脑路径中的用户名改变 | 分析结果和数据公式没有补充 |
| prompt_mosaic.md | 仅示例电脑路径中的用户名改变 | 不涉及指标定义 |
| prompt_task.md | 仅示例电脑路径中的用户名改变 | 不涉及指标定义 |
| Day3 上午，v6 → v9 | 新版补充 TF-IDF 公式、文档向量、情感分析实战等，并调整页序 | 继续用新版学习和解释文本方法；旧版可作简版参考 |
| Day3 下午，v6 → v9 | 新版补充抽样核验说明、BERTopic 介绍及第 53 页四项课后作业 | 交付范围继续以 10 月 31 日版为准 |

## 能用上的部分与仍缺的证据

旧版说明了课程从 v6 到 v9 的内容增补，可以保留为版本记录；它没有提供独立的数据构建证据。5 份脚本分别用于数据健康检查、特征区分分析、绘图和模型训练，不能替代从原始财务表生成面板的脚本。

收入分母、accruals 历史与应用集定义，以及更正后的历史数值在当时是否可得，仍按现有证据列为待核实。保留课堂数值和旧版成果，不按候选公式直接改数，也不把这些候选式写成已确认的最终定义。

补充资料核查阶段保留首、中、尾三段方案；用户其后要求继续改进，新增全文对照另见“06_全文主题对照”。本次旧版资料本身没有导致样本或参数变更。

## 可复核记录

- `10月23日版本_逐文件比对.csv`：89 个文件的路径对应、匹配方式和双方 SHA-256。
- `10月23日版本核查摘要.json`：文件统计、差异清单和原文件保护状态。
- 过程目录 `research_v2/work/older_course_diffs/`：6 个差异文件的文本提取和逐行差异；PDF 保留页码标记。

v1 的 {summary['v1_preserved_files']} 份交付与主要代码文件指纹未变化。
"""
    (out/'10月23日版本核查报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
