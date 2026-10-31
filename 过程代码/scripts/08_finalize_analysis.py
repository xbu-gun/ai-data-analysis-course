"""确认决定后的正式报告与文化得分；不重做测试评估。"""
import gzip
import json
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from common import *
from model_tools import plot_setup


def finalize_culture():
    out=OUTPUT/"04_文本分析"
    original=pd.read_csv(out/"五维文化词典_待审阅_v1.csv")
    kept=original[~original.word.isin(CONFIG["culture_review_drop_words"])].copy()
    kept["original_rank"]=kept["rank"]
    kept["rank"]=kept.groupby("dimension",sort=False).cumcount()+1
    kept["review_status"]="AI初审；用户确认保守剔除28词，其余保留"
    assert len(original)-len(kept)==28
    save_csv(kept,out/"五维文化词典_审阅后_v2.csv")
    with gzip.open(WORK/"tokenized_corpus_v1.json.gz","rt",encoding="utf-8") as f:docs=json.load(f)
    vec=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False)
    C=vec.fit_transform([" ".join(d["tokens"]) for d in docs]);terms=vec.get_feature_names_out();lookup={w:i for i,w in enumerate(terms)}
    idf=np.log(len(docs)/np.asarray((C>0).sum(axis=0)).ravel())
    lengths=np.array([len(d["tokens"]) for d in docs])
    scores=pd.DataFrame([{k:d[k] for k in ["code","company","year"]} for d in docs])
    dims=["创新","诚信","质量","尊重","团队"]
    for dim in dims:
        w=np.zeros(len(terms))
        for r in kept[kept.dimension==dim].itertuples(index=False):w[lookup[r.word]]=idf[lookup[r.word]]/math.log(1+r.rank)
        scores[dim]=np.asarray(C@w).ravel()/np.maximum(lengths,1)
    scores["total"]=scores[dims].sum(axis=1);scores["tokens"]=lengths
    save_csv(scores,out/"公司年度文化指标_v2.csv")
    before=pd.read_csv(out/"公司年度文化指标_初稿_v1.csv")
    effect=pd.DataFrame({"dimension":dims,"before_mean":before[dims].mean().values,"after_mean":scores[dims].mean().values})
    effect["relative_change"]=effect.after_mean/effect.before_mean-1;save_csv(effect,out/"词典审阅对指标的影响_v2.csv")
    summary={"time_beijing":now(),"dictionary_words":len(kept),"dropped_words":CONFIG["culture_review_drop_words"],
             "documents":len(scores),"companies":int(scores.code.nunique()),"means":{d:float(scores[d].mean()) for d in dims},
             "words_per_dimension":{k:int(v) for k,v in kept.dimension.value_counts().items()},"user_confirmed_review_policy":True,
             "external_validity_verified":False,"teacher_scores_used":False}
    write_json(out/"文化指标审阅后摘要_v2.json",summary)
    plt=plot_setup();latest=scores[scores.year==2022]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));means=scores[dims].mean();axes[0].bar(means.index,means);axes[0].set_title("五维文化表述强度：公司年度均值")
    top=latest.nlargest(15,"创新").iloc[::-1];axes[1].barh(top.company,top["创新"]);axes[1].set_title("2022年创新表述强度Top15（描述性）");fig.tight_layout();fig.savefig(out/"文化指标概览_v2.png");plt.close(fig)
    (out/"文化指标方法说明_v2.md").write_text(f"""# 公司文化五维指标：方法与边界

时间（北京时间）：{now()}。本次重新处理1,123篇、225家公司的2018—2022年MD&A，不沿用示例得分。

## 方法

五维为创新、诚信、质量、尊重、团队。按老师演示的种子词，训练100维、window=5、epochs=30、seed=42的Word2Vec；单线程确保同环境下可重复。因“正直”只出现3次，统一用min_count=1训练同一空间，候选扩词仍要求频数>=5。每维取相似度Top200，跨维冲突保留相似度较高的归属。

初稿812词。AI逐词初审后，用户确认保守删除28个明确噪声，保留{len(kept)}词；原词典、剔除清单和新旧排名均保留。审阅后的词按原相似度重新排名，权重为1/log(1+rank)。边界词暂未大规模删改，不能称为已做外部效度验证的文化词典。

指标=Σ[词频×log(N/df)×1/log(1+rank)]/文档过滤后词数。这里不使用sklearn默认的IDF额外+1，按教程公式计算。没有把独立训练的词向量混入同一质心。实现修正和样本选择已获得用户确认。

## 怎样解释

得分是长度归一的文化词表述强度，不是百分制评分。比如诚信得分较高，只能说明该维度词汇在文本中的相对强度较高，不能认定公司实际更诚信。MD&A属于管理层自我披露；与电话会即兴问答不同，可能有套话和粉饰。五维绝对量级也受词典大小、种子词、分词和IDF定义影响，不能直接拿本次数值与教程旧代码的演示数字比较。

## 本次保留的可核查证据

恢复文本的原文路径和SHA-256、种子词频数、完整词典和排名、28词剔除记录、公司年度五维得分、新旧均值变化均落盘。没有外部员工评价或人工文化标签，故不声称因果效应、真实文化排名或预测效度。
""",encoding="utf-8")
    return summary


def finalize_clusters():
    for task in ["舞弊预测","破产预测"]:
        out=OUTPUT/"02_全流程分析"/task/"无监督探索"
        s=json.loads((out/"无监督探索摘要_v1.json").read_text(encoding="utf-8"))
        k=CONFIG["unsupervised_main_k"];s["final_k"]=k;s["needs_user_choice"]=False;s["decision_time_beijing"]=now()
        write_json(out/"无监督探索摘要_确认后_v1.json",s)
        diag=s["candidate_diagnostics"][str(k)];rates=pd.DataFrame(diag["rates"])
        risk=rates.sort_values("positive_rate",ascending=False).iloc[0]
        total_pos=int(rates.positives.sum());base=total_pos/int(rates.rows.sum())
        precision=float(risk.positive_rate);recall=int(risk.positives)/total_pos;coverage=int(risk.rows)/int(rates.rows.sum())
        text=f"""# {task}：无监督探索正式记录

时间（北京时间）：{now()}。用户确认正文采用2组，4组保留为附录对照。

## 1. 问题与样本

这一步研究的是财务特征是否自然形成不同形态，不是训练一个带标签的分类器。只使用任务a的训练部分{s['rows']:,}个样本；留出测试部分没有参与探索。公司代码、年份、是否舞弊/严重舞弊或破产标签均不进入特征矩阵。保留{s['features']}个特征，删除{s['dropped']}。样本范围和删列依据均能追溯至第一步核查报告。

## 2. 为什么这样预处理

先把无穷值作为缺失，再按训练特征的缺失率和常量情况筛列。对其余列采用1%/99%分位缩尾、中位数填补和RobustScaler。这是为了避免原始财务比率的量纲和厚尾主导距离。规模变量已经取自然对数，没有再次取对数。处理参数只来自训练部分，完全不按标签分组填补。

这些处理不是没有代价。缩尾压缩极端差异；中位数填补会让缺失较多的样本靠近群体中心；RobustScaler仍可能受极小IQR或近离散特征影响。因此分组首先是对当前表示的总结，不能当作自然界中唯一真实的公司类型。

## 3. 降维与聚类分开

PCA和t-SNE用于看图，图上叠加标签只发生在无监督表示完成之后。正式聚类在完整标准化特征空间中运行MiniBatchKMeans，k从2到10比较。t-SNE中的簇间距离、面积和形状不具有直接定量意义，不能因为红点在图上相邻就宣称可识别正类。

## 4. 为什么正文选2组

本次轮廓系数最高为k={s['silhouette_best_k']}。用户在看过比较依据后选择正文用2组，附录保留4组。k=2在Ward子样本对照中的ARI为{diag['ward_ari_sample']:.3f}；k=4的对应值为{s['candidate_diagnostics']['4']['ward_ari_sample']:.3f}。Ward最多使用2,000条同一特征空间样本，这一一致性数字不能外推成全样本分类准确率。

选择2组强调结构简洁与算法对照；4组保留更细的画像，但没有为了接近老师演示的4组而指定它为唯一正确结果。各组的原始财务特征中位数、相对IQR画像和公司样本归属均保存在同目录。簇号只是算法编号，不表示风险高低的天然次序。

## 5. 事后看标签：有差异不等于能识别

训练样本正类基准为{base:.2%}。若事后把正类比例最高的簇{int(risk.cluster)}视作探索性风险池：覆盖{int(risk.rows):,}条、占全部样本{coverage:.2%}；池内正类比例{precision:.2%}，约为总体的{precision/base:.2f}倍；但只覆盖全部正类的{recall:.2%}。

注意：这个风险池是观察标签后选出的，以上精确率和召回率都是训练样本内的探索统计，没有经过独立测试。它们用于说明聚类能否集中线索，不是可部署的预测成绩。

卡方检验p={diag['chi2_p']:.4g}，Cramér's V={diag['cramers_v']:.4f}。即便p很小，效应量仍可能很小；反之，形态上轮廓清晰也不保证破产或舞弊标签能分开。必须把形态结构、标签关联、未来预测三件事分别说明。

## 6. 这一步对后续的帮助

探索帮助发现厚尾、制度性缺失与样本异质性，也帮助建立“分组不等于识别”的判断。后续有监督模型会在独立协议下利用标签学习，并在留出数据上评估。没有把本次聚类簇号偷偷加入预测特征，也没有拿聚类图来替代ROC-AUC、PR-AUC或错误分析。

## 7. 可复现检查与局限

随机种子固定42。输入指纹、固定划分、逐k指标、标准化空间的聚类、Ward子样本对照、簇画像和事后标签比例都保存到文件。没有使用测试集，没有使用标签参与清洗或聚类，没有将相关性解释成因果效应。

局限包括：历史样本不代表未来公司总体；缺失与披露制度有关；不同算法会改变边界；只扫描2—10组；参数和特征表示影响结果；标签反映历史标注而非所有真实行为；训练样本内风险池存在事后选择。保留这些限制，比只展示分组图或显著性更有助于理解结果。
"""
        (out/"无监督探索报告_v1.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    setup();print(finalize_culture());finalize_clusters()
