"""训练集内无监督探索；输出候选簇数，最终取舍交给用户确认。"""
import inspect
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import RobustScaler
from common import *
from model_tools import plot_setup


def main():
    setup()
    fraud, bank = load_finance()
    splits = json.loads((WORK / "split_indices_v1.json").read_text(encoding="utf-8"))
    summaries = {}
    plt = plot_setup()
    for task, frame, ids, target, excluded in [
        ("舞弊预测", fraud, splits["fraud"]["train"], "is_fraud", ["Stkcd", "Year", "is_fraud", "is_serious"]),
        ("破产预测", bank, splits["bankruptcy"]["train"], "Bankrupt?", ["Bankrupt?"])]:
        out = OUTPUT / "02_全流程分析" / task / "无监督探索"
        out.mkdir(parents=True, exist_ok=True)
        raw = frame.loc[ids].reset_index(drop=True)
        X = raw.drop(columns=excluded).replace([np.inf, -np.inf], np.nan)
        # 只用训练集的特征本身定规则，标签不进入任何预处理或聚类。
        dropped = list(X.columns[(X.isna().mean() > 0.5) | (X.nunique() <= 1)])
        X = X.drop(columns=dropped)
        med = X.median();low=X.quantile(.01);high=X.quantile(.99)
        clean = X.clip(low, high, axis=1).fillna(med)
        Z = RobustScaler().fit_transform(clean)
        assert np.isfinite(Z).all() and target not in X.columns
        rng = np.random.default_rng(SEED)
        sample = np.sort(rng.choice(len(Z), size=min(4000,len(Z)), replace=False))
        scan=[];models={};profiles={};assignments=[]
        for k in range(2,11):
            model = MiniBatchKMeans(k, random_state=SEED, n_init=10, batch_size=1024).fit(Z)
            labels=model.labels_
            score=silhouette_score(Z[sample],labels[sample]) if len(np.unique(labels[sample]))>1 else -1
            scan.append({"k":k,"inertia":float(model.inertia_),"silhouette":float(score)})
            models[k]=model
            print(task,"k",k,"silhouette",round(score,4),flush=True)
        scan_df=pd.DataFrame(scan);save_csv(scan_df,out/"簇数比较_v1.csv")
        top_k=int(scan_df.sort_values("silhouette",ascending=False).iloc[0].k)
        for k in sorted({top_k,4}):
            labels=models[k].labels_
            table=pd.crosstab(labels,raw[target]);chi,p,dof,expected=chi2_contingency(table)
            cramer=float(np.sqrt(chi/(table.to_numpy().sum()*min(table.shape[0]-1,table.shape[1]-1))))
            profile=clean.assign(cluster=labels).groupby("cluster").median()
            rates=pd.DataFrame({"cluster":labels,"target":raw[target]}).groupby("cluster").target.agg(["size","sum","mean"]).reset_index()
            rates.columns=["cluster","rows","positives","positive_rate"]
            save_csv(profile.reset_index(),out/f"k{k}_簇特征中位数_v1.csv");save_csv(rates,out/f"k{k}_簇标签事后对照_v1.csv")
            save_csv(pd.DataFrame({"source_row":ids,"cluster":labels}),out/f"k{k}_公司样本簇归属_v1.csv")
            ward_sample=sample[:min(2000,len(sample))]
            ward=AgglomerativeClustering(k,linkage="ward").fit_predict(Z[ward_sample])
            ari=float(adjusted_rand_score(models[k].predict(Z[ward_sample]),ward))
            profiles[k]={"cramers_v":cramer,"chi2_p":float(p),"ward_ari_sample":ari,
                         "sample_rows_for_ward":len(ward_sample),"rates":rates.to_dict("records")}
            view=profile.copy()
            view=(view-clean.median())/(clean.quantile(.75)-clean.quantile(.25)).replace(0,1)
            show=view.var().sort_values(ascending=False).head(12).index
            fig,ax=plt.subplots(figsize=(12,4));mat=ax.imshow(view[show].to_numpy(),cmap="RdBu_r",vmin=-3,vmax=3,aspect="auto")
            ax.set_xticks(range(len(show)),show,rotation=60,ha="right",fontsize=8);ax.set_yticks(range(k),[f"簇{i}" for i in range(k)])
            ax.set_title(f"{task} k={k}：相对训练集中位数的财务画像（IQR单位）");fig.colorbar(mat,ax=ax);fig.tight_layout();fig.savefig(out/f"k{k}_簇画像_v1.png");plt.close(fig)
        pc=PCA(n_components=min(30,Z.shape[1]),random_state=SEED).fit_transform(Z[sample])
        tsne_args={"n_components":2,"random_state":SEED,"perplexity":30,"init":"pca","learning_rate":"auto"}
        tsne_args["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"]=1000
        xy=TSNE(**tsne_args).fit_transform(pc)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        axes[0].scatter(*xy.T,c=models[top_k].labels_[sample],s=5,cmap="tab10",alpha=.6)
        axes[0].set_title(f"无监督形态（k={top_k}候选）")
        axes[1].scatter(*xy.T,c=raw.iloc[sample][target],s=5,cmap="coolwarm",alpha=.6)
        axes[1].set_title("事后叠加标签；标签未用于聚类")
        fig.suptitle(task+"：t-SNE仅供观察，聚类在标准化特征空间完成")
        fig.tight_layout();fig.savefig(out/"tSNE_结构与标签_v1.png");plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot(scan_df.k,scan_df.inertia,"o-");axes[1].plot(scan_df.k,scan_df.silhouette,"o-")
        axes[0].set(title="肘部图",xlabel="k");axes[1].set(title="抽样轮廓系数",xlabel="k");fig.tight_layout();fig.savefig(out/"簇数选择_v1.png");plt.close(fig)
        summary={"time_beijing":now(),"rows":len(raw),"features":len(X.columns),"dropped":dropped,"silhouette_best_k":top_k,
                 "final_k":None,"needs_user_choice":True,"candidate_diagnostics":profiles,
                 "held_out_test_used":False,"labels_used_in_clustering":False}
        write_json(out/"无监督探索摘要_v1.json",summary);summaries[task]=summary
        (out/"无监督探索说明_v1.md").write_text(f"""# {task}：无监督探索（候选结果）

时间（北京时间）：{now()}。只分析训练部分 {len(raw):,} 条，未使用留出测试集。

标签完全隔离于预处理和聚类。删除训练缺失率>50%及常量特征：{dropped}。其余采用训练特征的1%/99%缩尾、中位数填补、RobustScaler。最终 {len(X.columns)} 维。聚类使用标准化后的完整特征空间；t-SNE仅用于观察，不能把二维图上的距离当成真实距离。

k 从2到10比较。轮廓系数最高为 k={top_k}，另保留 k=4 作为更细的财务画像对照。最终簇数尚未决定，不把某个候选说成唯一正确答案。完整指标和画像见同目录 CSV、PNG。

事后标签对照只说明风险浓度差异。卡方检验须结合 Cramér's V，不能仅用很小的 p 值宣称能识别舞弊或破产。Ward 对照使用至多2,000条同一特征空间样本；其 ARI 只能代表该子样本的一致性。

当前阶段是探索候选，待确认簇数后补充各簇的解释、风险池召回率和正式报告。
""",encoding="utf-8")
    write_json(LOGS/"unsupervised_candidates_v1.json",summaries)


if __name__=="__main__":main()
