"""统一比较模型，锁定冠军和阈值后，各数据集只执行一次测试终评。"""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                            precision_score, brier_score_loss, confusion_matrix, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, ParameterGrid
from xgboost import XGBClassifier
from common import *
from model_tools import pipeline, plot_setup


def metrics(y, p, threshold):
    z = p >= threshold
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc_ap": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, z, zero_division=0)), "recall": float(recall_score(y, z, zero_division=0)),
            "precision": float(precision_score(y, z, zero_division=0)), "brier": float(brier_score_loss(y, p))}


def best_threshold(y, p):
    pr, re, th = precision_recall_curve(y, p)
    f1 = 2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1], 1e-15)
    return float(th[int(np.argmax(f1))])


def candidates(weighted):
    w = "balanced" if weighted else None
    return {
        "Logistic": (LogisticRegression(penalty="l1", solver="liblinear", class_weight=w, max_iter=3000, random_state=SEED), {"C": [0.1, 1.0, 10.0]}),
        "DecisionTree": (DecisionTreeClassifier(class_weight=w, random_state=SEED), {"max_depth": [3, 5, 6], "min_samples_leaf": [50, 100]}),
        "RandomForest": (RandomForestClassifier(n_estimators=100, class_weight=w, n_jobs=4, random_state=SEED), {"max_depth": [5, 8], "min_samples_leaf": [30, 100]}),
        "XGBoost": (XGBClassifier(n_estimators=150, learning_rate=0.05, objective="binary:logistic", eval_metric="logloss", n_jobs=4, random_state=SEED), {"max_depth": [3, 5], "n_estimators": [100, 200]}),
    }


def train(task):
    setup()
    out = OUTPUT / "02_全流程分析" / task
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "最终测试评估_v1.json"
    if marker.exists():
        print(task, "已终评；跳过，避免反复使用测试集。", flush=True)
        return
    fraud, bank = load_finance()
    splits = json.loads((WORK / "split_indices_v1.json").read_text(encoding="utf-8"))
    is_fraud = task == "舞弊预测"
    df = fraud if is_fraud else bank
    target = "is_fraud" if is_fraud else "Bankrupt?"
    excluded = ["Stkcd", "Year", "is_fraud", "is_serious"] if is_fraud else [target]
    features = [c for c in df if c not in excluded]
    sp = splits["fraud" if is_fraud else "bankruptcy"]
    # 在冠军锁定之前不读取留出测试集的特征或标签。
    X = df.loc[sp["train"], features].reset_index(drop=True)
    y = df.loc[sp["train"], target].reset_index(drop=True)
    if is_fraud:
        yy = df.loc[sp["train"], "Year"].reset_index(drop=True)
        years = np.sort(yy.unique())
        folds = [(np.flatnonzero(yy.isin(years[a])), np.flatnonzero(yy.isin(years[b])))
                 for a, b in TimeSeriesSplit(n_splits=5).split(years)]
    else:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    trials, fold_rows, best_models, best_oofs = [], [], {}, {}
    cached = pd.read_csv(out / "模型对比_v1.csv") if (out / "模型对比_v1.csv").exists() else None
    for name, (base, grid) in candidates(not is_fraud).items():
        if cached is not None and name in set(cached.model):
            chosen = cached[cached.model == name].iloc[0].to_dict()
            estimator = clone(base).set_params(**json.loads(chosen["parameters"]))
            if name == "XGBoost" and not is_fraud:
                estimator.set_params(scale_pos_weight=float((y == 0).sum()/y.sum()))
            best_models[name] = pipeline(estimator).fit(X, y)
            best_oofs[name] = (chosen, None)
            print(task, name, "恢复已完成的CV选择，仅重建训练模型；未读取测试集。", flush=True)
            continue
        model_trials = []
        for params in ParameterGrid(grid):
            t0 = time.time()
            pred = np.full(len(y), np.nan)
            per_fold = []
            for k, (a, b) in enumerate(folds, 1):
                estimator = clone(base).set_params(**params)
                if name == "XGBoost" and not is_fraud:
                    estimator.set_params(scale_pos_weight=float((y.iloc[a] == 0).sum()/y.iloc[a].sum()))
                pipe = pipeline(estimator)
                pipe.fit(X.iloc[a], y.iloc[a])
                pred[b] = pipe.predict_proba(X.iloc[b])[:, 1]
                per_fold.append((k, b))
            seen = np.isfinite(pred)
            threshold = best_threshold(y[seen], pred[seen]) if is_fraud else 0.5
            scores = [metrics(y.iloc[b], pred[b], threshold) for k, b in per_fold]
            record = {"model": name, "parameters": json.dumps(params, sort_keys=True), "threshold_cv": threshold,
                      "seconds": time.time()-t0, "cv_positive_baseline": float(y[seen].mean())}
            for metric in scores[0]:
                record[metric+"_mean"] = float(np.mean([m[metric] for m in scores]))
                record[metric+"_std"] = float(np.std([m[metric] for m in scores], ddof=1))
            model_trials.append((record, params, pred))
            trials.append(record)
            print(task, name, params, "CV F1", round(record["f1_mean"], 4), "AUC", round(record["roc_auc_mean"], 4), flush=True)
        key = (lambda r: (r[0]["f1_mean"], r[0]["pr_auc_ap_mean"])) if is_fraud else (lambda r: (r[0]["roc_auc_mean"], r[0]["pr_auc_ap_mean"]))
        chosen, params, pred = max(model_trials, key=key)
        estimator = clone(base).set_params(**params)
        if name == "XGBoost" and not is_fraud:
            estimator.set_params(scale_pos_weight=float((y == 0).sum()/y.sum()))
        fitted = pipeline(estimator).fit(X, y)
        best_models[name] = fitted
        best_oofs[name] = (chosen, pred)
        for k, (a, b) in enumerate(folds, 1):
            fold_rows.append({"model": name, "fold": k, "rows": len(b), "positive_baseline": float(y.iloc[b].mean()),
                              **metrics(y.iloc[b], pred[b], chosen["threshold_cv"])})
    if trials:
        save_csv(pd.DataFrame(trials), out / "参数搜索记录_v1.csv")
    if fold_rows:
        save_csv(pd.DataFrame(fold_rows), out / "逐折验证指标_v1.csv")
    comparison = pd.DataFrame([v[0] for v in best_oofs.values()])
    if is_fraud:
        comparison = comparison.sort_values(["f1_mean", "pr_auc_ap_mean"], ascending=False)
        champion = comparison.iloc[0]["model"]
        threshold = float(best_oofs[champion][0]["threshold_cv"])
        selection = "训练内部时间前推 CV，按平均 F1、PR-AUC 排序；阈值使用同一 CV 的折外预测选择，CV 分数含选择乐观偏差，最终测试独立。"
    else:
        vx = df.loc[sp["valid"], features]
        vy = df.loc[sp["valid"], target]
        vrows = []
        for name, pipe in best_models.items():
            prob = pipe.predict_proba(vx)[:, 1]
            vrows.append({"model": name, "threshold": best_threshold(vy, prob),
                          **metrics(vy, prob, best_threshold(vy, prob))})
        vframe = pd.DataFrame(vrows).sort_values(["roc_auc", "pr_auc_ap"], ascending=False)
        save_csv(vframe, out / "独立验证集指标_v1.csv")
        champion = vframe.iloc[0]["model"]
        threshold = float(vframe.iloc[0]["threshold"])
        selection = "训练集内分层5折 CV 按 ROC-AUC 选每个模型的参数；独立验证集按 ROC-AUC 选冠军、按 F1 选阈值；测试不参与。"
    save_csv(comparison, out / "模型对比_v1.csv")
    # 冠军、特征处理和阈值在这里锁定，随后只做一次测试终评。
    lock = {"time_beijing": now(), "champion": champion, "threshold": threshold, "selection": selection,
            "parameters": {k: ("NaN (XGBoost default missing marker)" if isinstance(v, float) and np.isnan(v) else v)
                           for k, v in best_models[champion].named_steps["model"].get_params().items()},
            "retained_features": best_models[champion].named_steps["prepare"].columns_}
    write_json(out / "模型与阈值锁定_v1.json", lock)
    joblib.dump(best_models[champion], MODELS / f"{task}_冠军_v1.joblib")
    tx, ty = df.loc[sp["test"], features], df.loc[sp["test"], target]
    prob = best_models[champion].predict_proba(tx)[:, 1]
    final = {"time_beijing": now(), "model": champion, "threshold": threshold, "rows": len(ty),
             "positive_baseline": float(ty.mean()), **metrics(ty, prob, threshold),
             "confusion_matrix_tn_fp_fn_tp": confusion_matrix(ty, prob >= threshold).ravel().tolist(),
             "dummy_most_frequent": metrics(ty, np.zeros(len(ty)), 0.5),
             "selection": selection, "test_used_for_selection": False}
    # 在画图等后处理之前写终评标记，后续失败重跑也不重新训练/改阈值。
    write_json(marker, final)
    test_preds = pd.DataFrame({"source_row": sp["test"], "actual": ty.to_numpy(), "score": prob, "prediction": (prob >= threshold).astype(int)})
    if is_fraud:
        test_preds["Stkcd"] = df.loc[sp["test"], "Stkcd"].to_numpy()
        test_preds["Year"] = df.loc[sp["test"], "Year"].to_numpy()
    test_preds["error_type"] = np.select([(test_preds.actual == 1) & (test_preds.prediction == 0), (test_preds.actual == 0) & (test_preds.prediction == 1)], ["漏报", "误报"], default="正确")
    save_csv(test_preds, out / "测试集逐条预测与错误类型_v1.csv")
    plt = plot_setup()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, title in zip(axes, ["f1_mean", "pr_auc_ap_mean", "roc_auc_mean"], ["CV F1", "CV PR-AUC (AP)", "CV ROC-AUC"]):
        ax.bar(comparison.model, comparison[metric]);ax.set_title(title);ax.tick_params(axis="x", rotation=35)
    fig.suptitle(task+"：统一协议下的模型比较");fig.tight_layout();fig.savefig(out/"模型比较_v1.png");plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    frac, mean = calibration_curve(ty, prob, n_bins=8, strategy="quantile")
    ax.plot(mean, frac, "o-", label="冠军模型");ax.plot([0,1],[0,1],"--",color="#777777")
    ax.set(xlabel="预测分数均值", ylabel="实际正类比例", title=task+"：测试集校准曲线");ax.legend();fig.tight_layout();fig.savefig(out/"校准曲线_v1.png");plt.close(fig)
    tree_pipe = best_models["DecisionTree"]
    tree = tree_pipe.named_steps["model"]
    feature_names = tree_pipe.named_steps["prepare"].columns_
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(tree, feature_names=feature_names, class_names=["正常", "正类"], filled=True, rounded=True, max_depth=3, fontsize=7, ax=ax)
    ax.set_title("解释用决策树：仅显示前4层；阈值为预处理后的标准化值，完整规则见文本")
    fig.tight_layout();fig.savefig(out/"决策树前四层_v1.png");plt.close(fig)
    (out/"决策树完整规则_v1.txt").write_text(export_text(tree, feature_names=feature_names), encoding="utf-8")
    pipe = best_models[champion]
    fitted = pipe.named_steps["model"]
    importance = np.abs(fitted.coef_[0]) if hasattr(fitted, "coef_") else fitted.feature_importances_
    imp = pd.DataFrame({"feature": pipe.named_steps["prepare"].columns_, "importance": importance}).sort_values("importance", ascending=False)
    save_csv(imp, out/"特征重要性_v1.csv")
    fig, ax = plt.subplots(figsize=(9, 6));top=imp.head(15).iloc[::-1];ax.barh(top.feature, top.importance)
    ax.set_title(task+"：Top15 预测贡献（不代表因果效应）");fig.tight_layout();fig.savefig(out/"特征重要性_v1.png");plt.close(fig)
    tn, fp, fn, tp = final["confusion_matrix_tn_fp_fn_tp"]
    report = f"""# {task}：有监督建模报告

生成时间（北京时间）：{now()}。所有数字均为本次实际运行，未复制演示成绩。

## 数据和评估

{selection}

训练 {len(X):,} 行，最终测试 {len(ty):,} 行。任务 a 的舞弊模型不加类别权重；破产数据按其项目规则加权。预处理在每个训练折内拟合：删除缺失率>50%的列和常量列、按列顺序去除相关系数绝对值>0.95的冗余列、1%/99%缩尾、训练中位数填补、RobustScaler。逻辑回归另用 L1 正则。各候选模型共享同一训练折下的特征处理。

模型对比和逐折指标见同目录 CSV。PR-AUC 使用 average precision (AP) 定义，报告正类基线。CV 数字用于模型/参数选择，不能当作完全无偏的外部成绩。

## 冠军及测试结果

冠军：{champion}；固定阈值：{threshold:.6f}。

| 指标 | 本次测试结果 |
|---|---:|
| ROC-AUC | {final['roc_auc']:.4f} |
| PR-AUC (AP) | {final['pr_auc_ap']:.4f} |
| 正类基线 | {final['positive_baseline']:.4f} |
| F1 | {final['f1']:.4f} |
| 召回率 | {final['recall']:.4f} |
| 精确率 | {final['precision']:.4f} |
| Brier | {final['brier']:.4f} |

## 判错了什么

- 漏报 {fn} 条：真实为正类，却没有被模型提示为高风险。
- 误报 {fp} 条：真实为正常，却被提示为高风险。
- 正确识别正类 {tp} 条，正确识别正常 {tn} 条。
- 逐条错误记录保留在测试集逐条预测与错误类型_v1.csv。不要把低风险分数理解为保证安全。

## 解释与局限

特征重要性只反映预测贡献，不代表降低某个指标就能避免舞弊或破产。展示树为解释用决策树，未用其测试成绩决定选型，可能与冠军不是同一个模型。历史标签存在被查处与实际发生的差别；舞弊数据的年度对齐简化披露时间，尾部年份存在右截断。当前结果不含外部有效性检验。

测试成绩没有返回调参流程。对分数是否能作概率使用，应结合 Brier 和校准曲线判断，不能仅因 ROC-AUC 较高就宣称概率准确。
"""
    (out/"有监督建模报告_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--task", choices=["舞弊预测", "破产预测", "all"], default="all")
    args = parser.parse_args()
    for task in (["舞弊预测", "破产预测"] if args.task == "all" else [args.task]): train(task)
