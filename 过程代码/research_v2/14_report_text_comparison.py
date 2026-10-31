"""汇总文本全文对照；不把覆盖率、离群率或稳定性当成准确率。"""
import csv,zipfile
import numpy as np
import pandas as pd
from study_common import *

T=OUT/'06_全文主题对照'

def main():
    emb=json.loads((T/'全文嵌入摘要.json').read_text(encoding='utf-8'))
    result=json.loads((T/'全文主题对照摘要.json').read_text(encoding='utf-8'))
    protocol=json.loads((STUDY/'text_protocol.json').read_text(encoding='utf-8'))
    names_path=STUDY/'text_topic_names_v2.json'
    names=json.loads(names_path.read_text(encoding='utf-8')) if names_path.exists() else {}
    info=pd.read_csv(T/'seed42/主题词与篇数.csv')
    info['descriptive_name']=info.Topic.map(lambda n:names.get(str(n),{}).get('name','关键词暂定标签'))
    info['interpretation_note']=info.Topic.map(lambda n:names.get(str(n),{}).get('note','尚未结合代表原文完成审阅'))
    save(info,T/'seed42/主题描述性命名.csv')
    paired=result['old_vs_full'];baseline=result['baseline_replication'];seeds=result['seed_runs']
    primary=next(x for x in seeds if x['seed']==42)
    assignments=pd.read_csv(T/'seed42/逐文档主题.csv',dtype={'code':str})
    additional=[]
    for r in assignments.itertuples():
        if r.topic==5 and '金陵' in r.company:
            p=CODE/'中间数据/corpus_sample'/r.file;text=p.read_text(encoding='utf-8-sig');pos=text.find('酒店');assert pos>=0
            start=max(0,pos-45);end=min(len(text),pos+180)
            additional.append({'topic':5,'company':r.company,'year':r.year,'file':r.file,'source_sha256':sha(p),
                               'character_start':start,'character_end':end,'evidence':text[start:end],
                               'reason':'为酒店混合主题命名补充核查，不参与聚类或样本选择'})
            break
    save(pd.DataFrame(additional),T/'seed42/补充命名原文_酒店业务.csv')
    company_topics=assignments.groupby(['topic','code','company']).size().rename('documents').reset_index()
    save(company_topics,T/'seed42/各主题公司与篇数.csv')
    original=pd.read_csv(ROOT/'成果交付/04_文本分析/doc_topics_v1.csv',dtype={'code':str})
    company_consistency=[]
    for label,frame in [('原首中尾',original),('全文主种子42',assignments)]:
        counts=frame.groupby('code').topic.nunique()
        company_consistency.append({'representation':label,'companies':len(counts),'all_years_in_one_topic':int((counts==1).sum()),
                                    'multiple_topics_across_years':int((counts>1).sum()),'same_topic_share':float((counts==1).mean())})
    save(pd.DataFrame(company_consistency),T/'同公司跨年主题一致性.csv')
    table_lines=['| 主题 | 描述性名称 | 篇数 | 代表词 |','|---|---|---:|---|']
    for row in info.itertuples():table_lines.append(f'| {row.Topic} | {row.descriptive_name} | {row.Count} | {row.top_words} |')
    seed_lines=['| 预设种子 | 初始主题数 | 合并后主题数 | 离群篇数 | 离群比例 |','|---|---:|---:|---:|---:|']
    for x in seeds:seed_lines.append(f"| {x['seed']} | {x['initial_topics']} | {x['final_topics']} | {x['outlier_documents']} | {x['outlier_rate']:.1%} |")
    stability_lines=['| 种子对 | ARI（含离群） | 共同非离群篇数 | 共同非离群 ARI |','|---|---:|---:|---:|']
    for x in result['seed_pair_stability']:stability_lines.append(f"| {x['seed_a']} / {x['seed_b']} | {x['ari_including_outliers']:.4f} | {x['common_non_outlier_documents']} | {x['ari_on_common_non_outliers']:.4f} |")
    report=f"""# 第二阶段：全文主题分析对照

时间（北京时间）：{now()}。

## 这次改进解决什么

原首、中、尾方案的文档词元覆盖率中位数为 {emb['old_coverage_median']:.2%}；本次对同样 1,123 篇、225 家公司、2018—2022 年 MD&A 逐段处理，模型分词后的词元覆盖率每篇均为 100%。全部文本共 {emb['original_tokens']:,} 个词元，切成 {emb['total_blocks']:,} 段。

样本、原文、本地模型、分词结果、聚类参数及主题合并规则均沿用；改进的是文档向量的文本覆盖范围。文化词频指标和词云已使用全篇分词，本轮不改这两项。

## 先确认旧结果能够复现

12 篇预先等距选取的文档按旧方案重算向量，最大绝对差 {emb['old_embedding_max_absolute_difference']:.3g}。用旧向量重新运行聚类，逐篇主题编号与旧版完全一致：{baseline['exact_same_numbered_assignments']}，ARI 为 {baseline['ari_including_outliers']:.4f}。这支持把本轮主对照中的变化归因于文本分块表示，而非随意改变参数或沿用错位缓存。

本次缓存指纹绑定原文顺序与 SHA-256、分词文件、模型文件、方案、代码及相关软件版本。计算按文档留存检查点，可从中断处继续。

## 全文怎么变成一个文档向量

每篇按词元顺序切成尽量等长的相邻块，块长相差最多 1，每块正文最多 510 个词元，加上起止标记后不超过模型的 512 上限。没有跨块重叠或遗漏。每块取 CLS 向量并归一化，全部块等权平均后再归一化；汇总方式沿用原方案。

这是本项目为长文本设计的分块汇总方案，不是声称模型官方推荐的完整文档理解算法。它覆盖了全文，但每块仍独立编码，不能完整建模跨段叙事、长距离否定或事件顺序。

BGE 的 CLS 及归一化实现依据[官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)；BERTopic 支持传入自算嵌入，见[官方嵌入文档](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html)。

## 与原主题有多大差异

- 两种文档向量的余弦相似度中位数为 {emb['cosine_median']:.4f}，最低 {emb['cosine_min']:.4f}。余弦描述表示差异，不是准确率。
- 原版 12 个非离群主题、306 篇离群；全文主种子 42 得到 {primary['initial_topics']} 个初始非离群主题，按既定规则合并后为 {primary['final_topics']} 个，离群 {primary['outlier_documents']} 篇（{primary['outlier_rate']:.2%}）。
- 新旧主题 ARI 为 {paired['ari_including_outliers']:.4f}、NMI 为 {paired['nmi_including_outliers']:.4f}，均含离群类别。共同非离群的 {paired['common_non_outlier_documents']} 篇上，ARI 为 {paired['ari_on_common_non_outliers']:.4f}；这部分会排除离群状态变化的文档，不能单独当成整体一致性。
- 原离群转入主题 {paired['old_outlier_to_topic']} 篇；原主题转为离群 {paired['old_topic_to_outlier']} 篇；两次均离群 {paired['both_outliers']} 篇。
- 对非离群主题做一对一最大重叠匹配、离群固定对应离群后，编号对齐一致率为 {paired['one_to_one_aligned_agreement_all']:.1%}。主题编号本身没有共同语义，不能直接按编号变化统计换组。

![覆盖率与向量变化](覆盖率与向量变化.png)

![新旧主题交叉图](主题归属交叉图.png)

## 预设随机种子敏感性

主结果预先固定为 42；43 和 44 仅作敏感性对照，没有按结果挑种子。各次均保留离群，不为降低离群比例强行回填。

{chr(10).join(seed_lines)}

三次最终都是 12 个主题，来自预先保留的合并规则，不能据此宣称 12 是数据独立支持的最优数目；初始主题数仍有变化。

{chr(10).join(stability_lines)}

这些数值检验同一流程对随机降维的敏感程度，并不提供外部真实主题标签。即使一致性高，也不能单凭这一点证明主题含义正确。

## 仍然没有解决的文本边界

主结果的通用主题 0 有 448 篇，约占 39.9%，说明全文覆盖仍未把所有文本分成清晰、细致的业务主题。个别小组也混入不同业务的公司，例如肥料农业组包含安记食品的 5 篇文本。命名应直接保留这些边界，不把聚类包装成正式行业分类。

同公司跨年都归于同一状态（含均为离群）的公司，旧方案为 {company_consistency[0]['all_years_in_one_topic']}/{company_consistency[0]['companies']}，全文方案为 {company_consistency[1]['all_years_in_one_topic']}/{company_consistency[1]['companies']}。跨年经营业务及写作模板相似都可能形成这一现象；当前没有将两者的影响识别开。另表保留每主题公司构成，不能仅用中心附近的少数公司代替全组。

原始 TXT 含表格连写和数字格式损失；本轮按原输入覆盖全文，没有凭猜测修复文本或财务数值。覆盖完整不等于消除了表格、套话和模板噪声。

## 主结果主题和原文核查

{chr(10).join(table_lines)}

描述性名称根据代表词与每主题最多 3 篇靠近向量中心的原文提出，优先覆盖不同公司，状态和解释见 `seed42/主题描述性命名.csv`。另外保存不限公司的中心最近三篇清单，避免隐藏同公司跨年文本高度相似这一特点。它们是 AI 辅助解释，不是正式行业分类，也没有虚称人工独立标注准确率。离群组不作单一主题解释。

代表原文表保存实际连续片段、原文文件名、SHA-256 和字符起止位置。文本常整章连成一行，因此按句末标点及主题词定位片段，避免只展示开头套话。它用于核查名称是否过强，不代表主题下所有文档，也不等于做过随机抽样的外部效度检验。同公司跨年措辞接近，分组还会反映企业模板，不能把 1,123 篇都理解为相互独立的公司样本。

## 怎样使用这份结果

可以确认的改进是：全文词元进入了编码流程，旧计算可复现，主题变化和随机性都有逐文档记录。主题数减少、离群率下降或组间一致性提高，都不自动等于更准确。

本次作为新增对照交付，不直接覆盖旧网站主结果。是否把全文方案作为最终展示版本，应结合主题原文审阅及本报告的稳定性结果决定；财务数据口径疑问仍待外部说明，与本轮文本改进分开记录。

## 文件入口

- `全文分块与旧版覆盖率对照.csv`：每篇词元、块数、覆盖率和向量变化。
- `逐文档归属变化对照.csv`、`旧主题与全文主题_交叉篇数.csv`：完整配对记录。
- `seed42/`：主结果主题、代表词、原文证据与描述性命名。
- `seed43/`、`seed44/`：预设敏感性结果。
- `全文主题互动图.html`：本地互动图；使用同目录 Plotly 资源，不上传原文。
- `文本对照结果核查.json`：样本、覆盖、原文证据、缓存和旧版保护的检查结果。

v1 的 {result['v1_preserved_files']} 份交付与主要代码文件指纹保持不变。
"""
    (T/'第二阶段_全文主题对照报告.md').write_text(report,encoding='utf-8')
    package=OUT/'第二轮研究_复现源码包.zip'
    sources=sorted([p for p in STUDY.iterdir() if p.is_file() and p.suffix.lower() in {'.py','.json','.md'}])
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,'research_v2/'+p.name)
    dump(T/'交付摘要.json',{'time_beijing':now(),'report':str(T/'第二阶段_全文主题对照报告.md'),
                          'source_package':str(package),'source_files':len(sources),'v1_preserved_files':assert_v1_unchanged(),
                          'main_website_replaced':False,'public_release':False})
    print('全文主题报告及第二轮复现源码包已生成。')

if __name__=='__main__':main()
