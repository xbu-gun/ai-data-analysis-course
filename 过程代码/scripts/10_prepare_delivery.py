"""整理可交付材料、主题命名证据、源码包。所有成绩读取已锁定结果。"""
import importlib.metadata
import json
import zipfile
import pandas as pd
from common import *


def main():
    setup();out=OUTPUT/"06_提交材料";out.mkdir(parents=True,exist_ok=True)
    results={t:json.loads((OUTPUT/f"02_全流程分析/{t}/最终测试评估_v1.json").read_text(encoding="utf-8")) for t in ["舞弊预测","破产预测"]}
    fraud,bank=results["舞弊预测"],results["破产预测"]
    topics=json.loads((OUTPUT/"04_文本分析/主题建模摘要_v1.json").read_text(encoding="utf-8"))
    names=json.loads((CODE/"topic_names_v1.json").read_text(encoding="utf-8"))
    info=pd.read_csv(OUTPUT/"04_文本分析/topics_info_待命名_v1.csv")
    info["topic_name"]=info.Topic.map(lambda n:names[str(n)])
    info["naming_basis"]="AI结合Top10代表词与每主题3份代表原文作描述性命名，不是官方行业分类"
    info["topic_name_status"]="已作描述性命名；保留关键词与原文证据供用户复核"
    save_csv(info,OUTPUT/"04_文本分析/topics_info_v1.csv")
    table="\n".join(f"| {r.Topic} | {names[str(r.Topic)]} | {r.Count} | {r.top_words} |" for r in info.itertuples(index=False))
    structured=f"""# MD&A文化与主题分析报告

生成时间（北京时间）：{now()}。

## 研究问题与数据

本项目把公司公开披露的管理层讨论与分析转为可比较的文化表述指标与语义主题。样本按老师演示表中的公司年度键恢复，共225家公司、1,123篇、2018—2022年。所有原文都能在课程包中找到；恢复过程逐篇校验SHA-256，原始文件只读。演示评分表仅用于定位样本，不用其中得分作为本次结果。

## 五维文化怎么计算

五维为创新、诚信、质量、尊重、团队。种子词采用课堂设定，Word2Vec扩展词典，每维取相似度Top200，跨维冲突归给更相似的维度。为避免低频种子“正直”被过滤，统一在同一个min_count=1的向量空间训练；扩词候选要求出现至少5次，使用单线程固定随机种子。

词典初稿812词，AI提出28个明确噪声，用户确认删除，得到784词。保留原始词典、剔除清单和审阅后新旧排名；其余边界词没有为获得理想结果而大规模删改。

公司年度得分=Σ[词频×log(N/df)×1/log(1+rank)]/文档过滤后词数。这里按教程公式，不使用示例函数默认额外加1的IDF，也不混合两套独立训练的词向量。公式修正已事先获得用户确认。详见文化指标方法说明_v2.md和词典审阅对指标的影响_v2.csv。

## BERTopic怎么做

嵌入模型使用课程包内bge-small-zh-v1.5，全部本地推理。原文较长，用户选择取开头、中间、结尾三段，每段最多512词元。每段按BGE官方CLS方法归一化，平均后再归一化为文档向量。各篇实际覆盖率另表列出，不能说模型完整读懂了每篇年报。

UMAP将512维压到5维，HDBSCAN的min_cluster_size=10；先得到{topics['initial_topics_excluding_outlier']}个非离群簇，按课堂建议将过细主题合并到12个。中文代表词采用多语言设置和显式中文分词，修复英文默认预处理删除中文的问题。主题数量和离群比例来自本次实际运行，不强求等于老师演示。

## 本次主题结果与命名

| 主题 | 描述性名称 | 文档数 | 代表词 |
|---|---|---:|---|
{table}

离群文档{topics['outlier_documents']}篇，占{topics['outlier_rate']:.2%}，没有静默删除。主题名称结合代表词和每主题3份靠近中心的原文证据提出，证据见主题代表原文证据_v1.csv。通用主题较大，部分主题含多个行业，故名称用于概括文本，不当作正式行业分类。

## 结论强度

五维指标测量MD&A中的文化表述强度，并非公司真正的文化好坏，更不能由“诚信”词多推导实际更诚信。主题结构在一定程度上反映行业及业务叙述，但同一家公司可能跨年改变归属。每公司条形图展示的是历年文档归属篇数，不是未经说明的主题概率分布。

本项目没有外部员工评价、人工文化标签或业务审计结果，不能声称外部效度或因果效应。语料只覆盖225家公司，财务应用覆盖5,574家公司；不在语料中的公司在网站上明确标注缺失，不用虚构文本补齐。

## 可核查交付

公司年度五维表、784词词典、28词剔除表、主题词表、逐文档主题、代表原文、三张交互HTML、静态图和225张公司词云均保留。网站同时标明2025财务年报与2018—2022文本的时间差。

方法来源：课程Tutorial及课程原文；BGE实现核对 [BGE官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5)。未额外获取新的公司数据。
"""
    (OUTPUT/"04_文本分析/文本分析结构化报告_v1.md").write_text(structured,encoding="utf-8")
    onepage=f"""# 财务信号与文本画像：一页结果说明

AI数据分析课程作业｜2026-10-31

**问题与数据。** 用财务指标探索舞弊、破产风险，再结合MD&A观察公司的文化表述和业务主题。舞弊历史面板42,511条；破产样本6,819条；无标签应用集5,574家公司；文本为225家公司1,123篇，覆盖2018—2022年。原始资料只读，样本和结果可追溯。

**流程与判断。** 先做训练集无监督探索，再按统一协议比较模型。聚类正文2组、附录4组；舞弊按年份前推验证，破产分层留出。预处理只在训练部分拟合，冠军和阈值锁定后，各测试集终评一次。

| 测试结果 | 舞弊预测 | 破产预测 |
|---|---:|---:|
| 冠军 | {fraud['model']} | {bank['model']} |
| ROC-AUC | {fraud['roc_auc']:.3f} | {bank['roc_auc']:.3f} |
| PR-AUC(AP) / 正类基线 | {fraud['pr_auc_ap']:.3f} / {fraud['positive_baseline']:.3f} | {bank['pr_auc_ap']:.3f} / {bank['positive_baseline']:.3f} |
| 召回率 | {fraud['recall']:.1%} | {bank['recall']:.1%} |
| 误报 / 漏报 | {fraud['confusion_matrix_tn_fp_fn_tp'][1]} / {fraud['confusion_matrix_tn_fp_fn_tp'][2]} | {bank['confusion_matrix_tn_fp_fn_tp'][1]} / {bank['confusion_matrix_tn_fp_fn_tp'][2]} |

**外推与文本。** 另用全量历史训练深度5、31叶、balanced权重的应用树，为5,574家公司输出风险排序和实际判定路径。2025为报表年，Year=2026按课程规则解释为预测目标年。文化词典经用户确认剔除28个噪声后784词，按教程公式重新计算。首中尾三段嵌入的BERTopic保留12个主题，另有306篇离群，占{topics['outlier_rate']:.1%}。

**局限与人的作用。** 聚类只能描述形态；应用分数不是校准的造假概率；2025应用集无标签，不能报告其真实准确率。MD&A文化分数衡量自我表述，无外部效度或因果验证。用户决定样本、权重、簇数、公式修正、文本取段和词典剔除；AI执行并保留代码、指纹、对比表与错误记录。

**交付状态。** 分析报告、数据表、网站本地版与复现源码已准备。在线网址和代码仓库网址待确定发布账户及可公开范围；两分钟介绍稿已备，视频待本人录制。
"""
    (out/"一页结果说明_v1.md").write_text(onepage,encoding="utf-8")
    video=f"""# 两分钟项目介绍稿（待本人录制）

本项目想回答一个问题：只看财务指标能发现多少风险，再加上年报文字，能不能让结果更容易理解和核查？

我使用老师提供的舞弊、破产和MD&A数据。先做无监督探索，再比较逻辑回归、决策树、随机森林和XGBoost。舞弊数据按年份前推，破产数据分层划分。测试集没有用来选模型或调阈值。

最终，舞弊模型测试ROC-AUC约{fraud['roc_auc']:.2f}，召回约{fraud['recall']:.0%}，但误报较多；破产模型ROC-AUC约{bank['roc_auc']:.2f}，召回约{bank['recall']:.0%}。这提醒我，不能只展示一个看起来很好的准确率，还要报告漏掉了谁、误判了多少。

我另外用全量历史数据训练了一棵可解释的树，为5,574家公司生成风险排序。网站可以查询财务指标、实际判定路径，以及有文本公司的文化雷达图、主题和词云。应用分数只作相对排序，不是真实造假概率。

文本部分恢复了1,123篇原文，五维文化词典经确认删除噪声后保留784词，BERTopic得到12个主题。文化分数测的是管理层怎么说自己，不代表真实文化，也不支持因果判断。

AI负责写代码、运行和整理结果；我负责确定样本、权重、簇数、公式修正和词典规则。每一项决定及结果都留下记录。这份作业最重要的收获，是知道为什么这样做，以及哪些话不能根据当前证据说出来。
"""
    (out/"两分钟项目介绍稿_v1.md").write_text(video,encoding="utf-8")
    evidence=[
        {"claim_id":"C01","claim":f"舞弊测试ROC-AUC={fraud['roc_auc']:.6f}","evidence":"02_全流程分析/舞弊预测/最终测试评估_v1.json","field":"roc_auc"},
        {"claim_id":"C02","claim":f"舞弊测试召回={fraud['recall']:.6f}","evidence":"02_全流程分析/舞弊预测/测试集逐条预测与错误类型_v1.csv","field":"actual,prediction"},
        {"claim_id":"C03","claim":f"破产测试ROC-AUC={bank['roc_auc']:.6f}","evidence":"02_全流程分析/破产预测/最终测试评估_v1.json","field":"roc_auc"},
        {"claim_id":"C04","claim":"应用集5574家公司无标签，未报告真实准确率","evidence":"03_2025风险预测/应用模型摘要_v1.json","field":"application_rows,application_accuracy_known"},
        {"claim_id":"C05","claim":"恢复1123篇、225家公司，逐篇校验指纹","evidence":"01_数据核查/恢复文本样本指纹_v1.csv","field":"code,year,sha256"},
        {"claim_id":"C06","claim":"审阅后文化词典784词","evidence":"04_文本分析/五维文化词典_审阅后_v2.csv","field":"word,dimension,rank"},
        {"claim_id":"C07","claim":"12个主题、306篇离群","evidence":"04_文本分析/doc_topics_v1.csv","field":"topic"},
        {"claim_id":"C08","claim":"主题命名依据代表词和原文，非正式行业分类","evidence":"04_文本分析/主题代表原文证据_v1.csv","field":"topic,source_file,evidence"},
    ]
    save_csv(pd.DataFrame(evidence),out/"CLAIM_EVIDENCE_MAP.csv")
    packages=["numpy","pandas","scipy","scikit-learn","joblib","matplotlib","xgboost","jieba","gensim","bertopic","sentence-transformers","transformers","torch","umap-learn","hdbscan","wordcloud","plotly"]
    (CODE/"requirements.txt").write_text("\n".join(f"{p}=={importlib.metadata.version(p)}" for p in packages)+"\n",encoding="utf-8")
    progress=f"""# 成果交付入口

更新时间（北京时间）：{now()}。

## 从哪里看起

1. `06_提交材料/一页结果说明_v1.pdf`：一页概览。
2. `05_网站/index.html`：双击可离线查询，无需安装Python。
3. `02_全流程分析/`：两套数据的聚类、模型比较、留出测试与逐条错误记录。
4. `03_2025风险预测/`：5,574家公司的预测表与应用树说明。
5. `04_文本分析/文本分析结构化报告_v1.md`：文化指标、主题命名和局限。
6. `06_提交材料/两分钟项目介绍稿_v1.md`：待本人录制的视频讲稿。

## 已完成与待完成

已完成：a两套数据完整分析；b历史全量应用树和风险表；c重新计算的文化指标、主题与报告；d本地网站及可上传源码包。

待完成：确认数据许可与公开范围、确定网站和代码仓库发布账户、上线得到实际可访问链接、本人录制介绍视频。

网页检查：JavaScript语法和数据一致性可核查；实际浏览器检查被宿主工具中止，原因是无法可靠识别Windows浏览器当前网址，未将其写成通过。

## 数据公开边界

当前网站包含公司级课程派生数据，仅准备为本地审阅版本。课程明确提醒商业数据库数据不可直接公开；发布前需确认许可。源码包不含原始CSV、MD&A原文、模型权重、独立环境或公司级数据。

## 复现

源码在同级“过程代码”。`requirements.txt`记录本次模型与文本工具版本。原始输入仍引用课程资料目录；固定划分和中间数据在本地过程目录，不随公开源码包分发。
"""
    (OUTPUT/"00_交付入口与待办.md").write_text(progress,encoding="utf-8")
    readme=(CODE/"README.md").read_text(encoding="utf-8")
    readme=readme.replace("第一阶段：核查来源、固定训练/验证/测试划分、恢复文本样本。后续结果均需实际重跑，不把老师演示输出当成本次结果。", "本地分析与网站已生成。公开发布范围、账户及仓库网址待确认。每项成绩来自本次实际运行，未复制演示分数。")
    readme=readme.replace("当前可用环境：`C:\\Users\\Admin\\miniconda3\\envs\\stata_ml\\python.exe`。本阶段只使用该环境现有的 numpy、pandas、scipy、scikit-learn、joblib，没有安装或升级包。", "第一步核查使用现有stata_ml环境；后续在本项目 `.venv` 中安装绘图、XGBoost及文本工具，继承的基础环境未被升级。正式运行使用 `.venv/Scripts/python.exe`；版本见requirements.txt。环境不可直接搬迁，换机器应依据requirements.txt重新创建。")
    if "## 完整执行顺序" not in readme:
        readme+="\n## 完整执行顺序\n\n01核查 → 02恢复文本 → 04无监督候选 → 03模型比较与一次终评 → 05全量应用预测 → 06文化初稿 → 07主题与词云 → 08确认后文化/聚类报告 → 09网站 → 10交付材料 → 11一页PDF → 12核查。研究决定需先体现在配置文件中。03检测到终评记录会跳过，避免重复使用测试集。\n\n`web/`是网站源码；09脚本将本次结果组装成最终网站。本地数据与课程输入不进入公开仓库。\n"
    (CODE/"README.md").write_text(readme,encoding="utf-8")
    source_files=[CODE/"README.md",CODE/"project_config.json",CODE/"topic_names_v1.json",CODE/"requirements.txt",CODE/".gitignore"]
    source_files+=list((CODE/"scripts").glob("*.py"))+list((CODE/"scripts").glob("*.mjs"))+list((CODE/"web").glob("*"))+[CODE/"运行记录/研究决定_v1.md",CODE/"运行记录/experiment_log.md"]
    with zipfile.ZipFile(out/"复现源码包_v1.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in source_files:z.write(p,str(p.relative_to(CODE)))
    print("Prepared",out,flush=True)


if __name__=="__main__":main()
