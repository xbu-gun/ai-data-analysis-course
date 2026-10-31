"""BERTopic：用户确认的首中尾三段嵌入，本地运行，不上传文本。"""
import gzip
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import plotly.express as px
from common import *
from model_tools import plot_setup


def main():
    setup()
    out=OUTPUT/"04_文本分析";out.mkdir(parents=True,exist_ok=True)
    with gzip.open(WORK/"tokenized_corpus_v1.json.gz","rt",encoding="utf-8") as f:docs=json.load(f)
    emb_file=WORK/"mda_three_segment_embeddings_v1.npy"
    torch.set_num_threads(4)
    if emb_file.exists():emb=np.load(emb_file)
    else:
        model_dir=DATA/"情感分析演示/语言模型/bge-small-zh-v1.5"
        tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
        model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).eval()
        windows=[];owners=[];coverage=[]
        for i,d in enumerate(docs):
            text=(WORK/"corpus_sample"/d["file"]).read_text(encoding="utf-8-sig")
            ids=tokenizer(text,add_special_tokens=False,truncation=False)["input_ids"]
            starts=sorted(set([0,max(0,len(ids)//2-255),max(0,len(ids)-510)]))
            positions=set()
            for start in starts:
                positions.update(range(start,min(start+510,len(ids))))
                windows.append({"input_ids":[tokenizer.cls_token_id]+ids[start:start+510]+[tokenizer.sep_token_id]});owners.append(i)
            coverage.append({"code":d["code"],"year":d["year"],"original_tokens":len(ids),"segments":len(starts),"covered_tokens":len(positions),"coverage_ratio":len(positions)/max(len(ids),1)})
        acc=np.zeros((len(docs),512),dtype=np.float32);counts=np.zeros(len(docs))
        with torch.inference_mode():
            for offset in range(0,len(windows),16):
                chunk=windows[offset:offset+16]
                size=max(len(w["input_ids"]) for w in chunk)
                input_ids=torch.full((len(chunk),size),tokenizer.pad_token_id,dtype=torch.long)
                attention_mask=torch.zeros((len(chunk),size),dtype=torch.long)
                for j,w in enumerate(chunk):
                    n=len(w["input_ids"]);input_ids[j,:n]=torch.tensor(w["input_ids"]);attention_mask[j,:n]=1
                batch={"input_ids":input_ids,"attention_mask":attention_mask}
                # BGE 官方方法：CLS pooling，再作 L2 归一化。
                output=F.normalize(model(**batch).last_hidden_state[:,0],p=2,dim=1).cpu().numpy()
                for j,vector in enumerate(output):acc[owners[offset+j]]+=vector;counts[owners[offset+j]]+=1
                if offset%160==0:print("embedded",offset,"/",len(windows),flush=True)
        emb=acc/counts[:,None];emb=emb/np.linalg.norm(emb,axis=1,keepdims=True)
        np.save(emb_file,emb.astype(np.float32))
        save_csv(pd.DataFrame(coverage),out/"长文本取样覆盖率_v1.csv")
    joined=[" ".join(d["tokens"]) for d in docs]
    topic_model=BERTopic(embedding_model=None,language="multilingual",
        umap_model=UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric="cosine",random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=10,metric="euclidean",cluster_selection_method="eom",prediction_data=True),
        vectorizer_model=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False),
        calculate_probabilities=False,verbose=True,top_n_words=10)
    topics,_=topic_model.fit_transform(joined,embeddings=emb)
    initial_topics=len(set(topics)-{-1})
    if initial_topics>17:
        topic_model.reduce_topics(joined,nr_topics=13)
        topics=topic_model.topics_
    table=pd.DataFrame([{k:d[k] for k in ["code","company","year"]} for d in docs]);table["topic"]=topics
    save_csv(table,out/"doc_topics_v1.csv")
    info=topic_model.get_topic_info();info["top_words"]=info.Topic.map(lambda n:"、".join(w for w,s in topic_model.get_topic(int(n))[:10]))
    info["topic_name_status"]="待结合代表原文命名；当前为关键词标签"
    save_csv(info,out/"topics_info_待命名_v1.csv")
    representatives=[]
    for n in sorted(set(topics)):
        indices=np.flatnonzero(np.asarray(topics)==n)
        center=emb[indices].mean(axis=0);center/=np.linalg.norm(center)
        ranked=indices[np.argsort(-(emb[indices]@center))[:3]]
        keywords=[w for w,s in topic_model.get_topic(int(n))[:10]]
        for i in ranked:
            d=docs[int(i)];text=(WORK/"corpus_sample"/d["file"]).read_text(encoding="utf-8-sig")
            snippets=[]
            for word in keywords:
                pos=text.find(word)
                if pos>=0:snippets.append(text[max(0,pos-40):pos+140].replace("\n"," "))
                if len(snippets)>=2:break
            representatives.append({"topic":int(n),"code":d["code"],"company":d["company"],"year":d["year"],
                                    "source_file":d["file"],"top_words":"、".join(keywords),"evidence":" / ".join(snippets)})
    save_csv(pd.DataFrame(representatives),out/"主题代表原文证据_v1.csv")
    xy=UMAP(n_neighbors=15,n_components=2,min_dist=.1,metric="cosine",random_state=SEED).fit_transform(emb)
    plot_data=table.copy();plot_data["x"]=xy[:,0];plot_data["y"]=xy[:,1];plot_data["topic_label"]=plot_data.topic.astype(str)
    fig=px.scatter(plot_data,x="x",y="y",color="topic_label",hover_data=["company","year"],title="MD&A主题地图：首中尾三段语义表示")
    fig.write_html(out/"topics_scatter_v1.html",include_plotlyjs="directory")
    topic_model.visualize_barchart(top_n_topics=min(12,len(set(topics)-{-1}))).write_html(out/"topics_bar_v1.html",include_plotlyjs="directory")
    non_outliers=info[info.Topic!=-1].sort_values("Topic")
    ts=sorted(set(topics)-{-1});centers=np.array([emb[np.asarray(topics)==n].mean(axis=0) for n in ts]);sim=cosine_similarity(centers)
    px.imshow(sim,x=[str(x) for x in ts],y=[str(x) for x in ts],title="主题语义相似度").write_html(out/"topics_heatmap_v1.html",include_plotlyjs="directory")
    plt=plot_setup();fig,ax=plt.subplots(figsize=(9,5));ax.bar(info.Topic.astype(str),info.Count);ax.set(title="MD&A主题文档数（-1为离群）",xlabel="主题号",ylabel="文档数");fig.tight_layout();fig.savefig(out/"主题文档数_v1.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6));points=ax.scatter(xy[:,0],xy[:,1],c=topics,cmap="tab20",s=12,alpha=.8);ax.set_title("MD&A主题地图（UMAP，仅用于观察邻域）");fig.colorbar(points,ax=ax,label="主题号");fig.tight_layout();fig.savefig(out/"主题地图_v1.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(sim,cmap="YlOrRd",vmin=0,vmax=1);ax.set_xticks(range(len(ts)),ts);ax.set_yticks(range(len(ts)),ts);ax.set_title("主题语义相似度");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/"主题相似度_v1.png");plt.close(fig)
    yearly=pd.crosstab(table.year,table.topic);save_csv(yearly.reset_index(),out/"主题年度篇数_v1.csv")
    fig,ax=plt.subplots(figsize=(10,5));yearly.plot.bar(stacked=True,ax=ax,cmap="tab20");ax.set(title="各年度文本主题归属",ylabel="篇数");fig.tight_layout();fig.savefig(out/"主题年度分布_v1.png");plt.close(fig)
    word_dir=out/"wordcloud";word_dir.mkdir(exist_ok=True)
    company_counts=defaultdict(Counter)
    cloud_stop={"经营","业务","产品","市场","主要","相关","情况","发展","年度","企业","报告","管理","技术","项目","其中"}
    for d in docs:company_counts[d["code"]].update(w for w in d["tokens"] if len(w)<=8 and w not in cloud_stop)
    for code,c in company_counts.items():
        if (word_dir/f"{code}.png").exists():
            continue
        freq={w:n for w,n in c.items() if n>=2}
        WordCloud(font_path="C:/Windows/Fonts/msyh.ttc",width=620,height=350,background_color="white",colormap="PuRd",max_words=100,random_state=SEED).generate_from_frequencies(freq).to_file(str(word_dir/f"{code}.png"))
    summary={"time_beijing":now(),"documents":len(docs),"embedding_shape":list(emb.shape),"embedding_strategy":"首、中、尾，最多3段×512词元；CLS归一化后平均再归一化",
             "initial_topics_excluding_outlier":initial_topics,"final_topics_excluding_outlier":len(ts),"outlier_documents":int(np.sum(np.asarray(topics)==-1)),
             "outlier_rate":float(np.mean(np.asarray(topics)==-1)),"wordcloud_companies":len(company_counts),"topics_human_named":False,
             "model_source":"课程附带bge-small-zh-v1.5，仅本地推理","method_source":"https://huggingface.co/BAAI/bge-small-zh-v1.5"}
    write_json(out/"主题建模摘要_v1.json",summary)
    print(summary,flush=True)


if __name__=="__main__":main()
