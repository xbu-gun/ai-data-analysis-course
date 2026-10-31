"""按教程公式计算五维文化指标；修正 TF-IDF 常数与向量空间混用。"""
import gzip
import json
import math
import re
from collections import Counter
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from common import *
from model_tools import plot_setup

SEEDS={"创新":["创新","创造力","卓越"],"诚信":["诚信","正直","合规"],"质量":["质量","品质"],"尊重":["尊重","关怀"],"团队":["团队","协作","合作"]}
STOP=set("的 了 是 在 和 及 与 等 公司 我们 主要 报告 年度 本 为 对 并 或 也 都 这 那 其 之 而 以 以及 进行 通过 方面 由于 因此 同时 目前 根据 截至 实现 完成 提高 推动 加强 保持 一个 一些 已 亦 即 该 各 此 这些 那些 没有 可以 可能 能够 继续 逐步 进一步 相关 包括 涉及 较大 一定 不断 积极 有效 整体 总体 基础上 随着 本公司 报告期 适用 不适用 万元 亿元".split())


def main():
    setup()
    out=OUTPUT/"04_文本分析";out.mkdir(parents=True,exist_ok=True)
    jieba.dt.tmp_dir=str(WORK)
    manifest=pd.read_csv(WORK/"corpus_sample/_manifest.csv",dtype={"code":str})
    cached=WORK/"tokenized_corpus_v1.json.gz"
    if cached.exists():
        with gzip.open(cached,"rt",encoding="utf-8") as f:docs=json.load(f)
    else:
        docs=[]
        for i,r in enumerate(manifest.itertuples(index=False),1):
            text=(WORK/"corpus_sample"/r.file).read_text(encoding="utf-8-sig")
            sentences=[]
            for line in re.split(r"[。！？；\n]",text):
                tokens=[w for w in jieba.lcut(line) if 2<=len(w)<=12 and re.search(r"[\u4e00-\u9fff]",w) and w not in STOP]
                if tokens:sentences.append(tokens)
            docs.append({"code":r.code,"company":r.company,"year":int(r.year),"file":r.file,
                         "sentences":sentences,"tokens":[w for s in sentences for w in s]})
            if i%100==0:print("tokenized",i,flush=True)
        with gzip.open(cached,"wt",encoding="utf-8") as f:json.dump(docs,f,ensure_ascii=False)
    counts=Counter(w for d in docs for w in d["tokens"])
    seed_counts={dim:{w:counts[w] for w in ss} for dim,ss in SEEDS.items()}
    # 所有种子和候选词都在同一个模型空间中，不混用独立训练的备用词向量。
    # 低频种子存在时，统一训练 min_count=1；扩词候选仍限定语料频数>=5。
    min_count=1 if any(counts[w]<5 for ss in SEEDS.values() for w in ss) else 5
    absent=[w for ss in SEEDS.values() for w in ss if counts[w]==0]
    if absent:raise ValueError(f"种子词完全未出现，需要用户决定：{absent}")
    model_path=MODELS/"culture_word2vec_v1.model"
    if model_path.exists():model=Word2Vec.load(str(model_path))
    else:
        print("Word2Vec training",len(counts),"tokens types; min_count",min_count,flush=True)
        model=Word2Vec(sentences=[s for d in docs for s in d["sentences"]],vector_size=100,window=5,
                       min_count=min_count,workers=1,epochs=30,seed=SEED)
        model.save(str(model_path))
    candidates=[w for w in model.wv.index_to_key if counts[w]>=5]
    vectors=model.wv[candidates];vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    best={}
    for dim,seeds in SEEDS.items():
        center=np.mean([model.wv[w] for w in seeds],axis=0);center=center/np.linalg.norm(center)
        similarities=vectors@center
        for idx in np.argsort(-similarities)[:200]:
            word=candidates[idx];similarity=float(similarities[idx])
            if word not in best or similarity>best[word]["similarity"]:
                best[word]={"word":word,"dimension":dim,"similarity":similarity,"corpus_frequency":counts[word]}
    dictionary=[]
    for dim in SEEDS:
        words=sorted([v for v in best.values() if v["dimension"]==dim],key=lambda r:-r["similarity"])
        for rank,r in enumerate(words,1):dictionary.append({**r,"rank":rank,"review_status":"待人工审阅"})
    dictionary=pd.DataFrame(dictionary)
    save_csv(dictionary,out/"五维文化词典_待审阅_v1.csv")
    joined=[" ".join(d["tokens"]) for d in docs]
    vectorizer=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False)
    C=vectorizer.fit_transform(joined)
    terms=vectorizer.get_feature_names_out();lookup={w:i for i,w in enumerate(terms)}
    df=np.asarray((C>0).sum(axis=0)).ravel()
    idf=np.log(len(docs)/df)  # 按教程公式，无 sklearn 默认的额外 +1。
    lengths=np.array([len(d["tokens"]) for d in docs],dtype=float)
    scores=pd.DataFrame([{k:d[k] for k in ["code","company","year"]} for d in docs])
    for dim in SEEDS:
        weights=np.zeros(len(terms))
        for r in dictionary[dictionary.dimension==dim].itertuples(index=False):
            weights[lookup[r.word]]=idf[lookup[r.word]]/math.log(1+r.rank)
        scores[dim]=np.asarray(C@weights).ravel()/np.maximum(lengths,1)
    scores["total"]=scores[list(SEEDS)].sum(axis=1);scores["tokens"]=lengths.astype(int)
    scores["dictionary_status"]="待人工审阅；当前为自动扩词初稿"
    save_csv(scores,out/"公司年度文化指标_初稿_v1.csv")
    summary={"time_beijing":now(),"documents":len(docs),"companies":int(scores.code.nunique()),"total_tokens":int(lengths.sum()),
             "seed_counts":seed_counts,"word2vec_min_count":min_count,"candidate_min_frequency":5,
             "dictionary_words":len(dictionary),"words_per_dimension":{k:int(v) for k,v in dictionary.dimension.value_counts().items()},
             "formula":"sum(tf * log(N/df) / log(1+rank)) / document_filtered_token_count",
             "means":{dim:float(scores[dim].mean()) for dim in SEEDS},"dictionary_human_review_complete":False}
    write_json(out/"文化指标计算摘要_v1.json",summary)
    plt=plot_setup();fig,ax=plt.subplots(figsize=(8,4));means=scores[list(SEEDS)].mean();ax.bar(means.index,means.values)
    ax.set_title("五维文化表述强度：公司年度均值（词典待审阅）");ax.set_ylabel("按文档词数归一的加权词频");fig.tight_layout();fig.savefig(out/"文化五维均值_初稿_v1.png");plt.close(fig)
    lines=["# 五维文化词典：人工审阅用初稿",f"\n生成时间（北京时间）：{now()}。下列为每维相似度最高的20个词；完整词典见同目录 CSV。", "\n请重点识别：公司名、人名、行业专用词、泛词、与维度含义相反的词。剔除或调整须记录理由，再重新计算指标。"]
    for dim in SEEDS:
        top=dictionary[dictionary.dimension==dim].head(20)
        lines.append(f"\n## {dim}\n\n"+"、".join(top.word))
    lines.append("\n## 方法差异\n\n用户已确认按教程公式修正实现：IDF=log(N/df)，没有额外+1；所有词向量来自同一个空间。为容纳低频种子统一降低训练min_count，扩词候选仍要求至少出现5次。Word2Vec单线程、固定种子，原文不变。分母采用过滤后的词数，这一操作定义明确记录，不能与按原始字符数归一的指标混用。\n\n这是MD&A自我表述的描述性指标，不是员工行为或公司真实文化的直接测量。")
    (out/"文化词典审阅说明_v1.md").write_text("\n".join(lines),encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":main()
