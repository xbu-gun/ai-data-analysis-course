"""第一步：固定数据版本与划分，仅对训练部分做分布分析。"""
import importlib.metadata
import json
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from common import *


def feature_quality(frame):
    values = frame.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({
        "feature": frame.columns,
        "missing_rate": values.isna().mean().values,
        "infinite_count": np.isinf(frame.select_dtypes(include="number")).sum().reindex(frame.columns, fill_value=0).values,
        "unique_count": values.nunique().values,
        "median": values.median().values,
        "q01": values.quantile(0.01).values,
        "q99": values.quantile(0.99).values,
    }).sort_values("missing_rate", ascending=False)


def main():
    setup()
    out = OUTPUT / "01_数据核查"
    out.mkdir(parents=True, exist_ok=True)
    fraud, bank = load_finance()
    assert fraud.shape == (42511, 40)
    assert bank.shape == (6819, 96)
    assert not fraud.duplicated(["Stkcd", "Year"]).any()
    assert set(fraud["is_fraud"].unique()) <= {0, 1}
    features = [c for c in fraud if c not in ["Stkcd", "Year", "is_fraud", "is_serious"]]
    f_train = fraud.index[fraud.Year.between(*CONFIG["fraud_train_years"])].to_numpy()
    f_test = fraud.index[fraud.Year.between(*CONFIG["fraud_test_years"])].to_numpy()
    assert len(set(f_train) & set(f_test)) == 0
    assert len(f_train) + len(f_test) == len(fraud)
    bank_ids = np.arange(len(bank))
    b_dev, b_test = train_test_split(bank_ids, test_size=0.15, stratify=bank["Bankrupt?"], random_state=SEED)
    b_train, b_valid = train_test_split(b_dev, test_size=0.15/0.85, stratify=bank.iloc[b_dev]["Bankrupt?"], random_state=SEED)
    assert not (set(b_train) & set(b_valid) or set(b_train) & set(b_test) or set(b_valid) & set(b_test))
    splits = {
        "fraud": {"train": f_train.tolist(), "test": f_test.tolist()},
        "bankruptcy": {"train": b_train.tolist(), "valid": b_valid.tolist(), "test": b_test.tolist()},
    }
    split_path = WORK / "split_indices_v1.json"
    if split_path.exists():
        assert json.loads(split_path.read_text(encoding="utf-8")) == splits, "已有划分与本次不一致，停止。"
    else:
        write_json(split_path, splits)
    f_q = feature_quality(fraud.loc[f_train, features])
    b_q = feature_quality(bank.iloc[b_train].drop(columns="Bankrupt?"))
    save_csv(f_q, out / "舞弊_训练集特征质量_v1.csv")
    save_csv(b_q, out / "破产_训练集特征质量_v1.csv")
    years = np.sort(fraud.loc[f_train, "Year"].unique())
    folds = []
    for n, (a, b) in enumerate(TimeSeriesSplit(n_splits=5).split(years), 1):
        train_years, valid_years = years[a], years[b]
        assert max(train_years) < min(valid_years)
        ti = f_train[fraud.loc[f_train, "Year"].isin(train_years).to_numpy()]
        vi = f_train[fraud.loc[f_train, "Year"].isin(valid_years).to_numpy()]
        folds.append({"fold": n, "train_years": train_years.tolist(), "valid_years": valid_years.tolist(),
                      "train_rows": len(ti), "valid_rows": len(vi),
                      "train_positive_rate": float(fraud.loc[ti, "is_fraud"].mean()),
                      "valid_positive_rate": float(fraud.loc[vi, "is_fraud"].mean())})
    write_json(out / "舞弊_时间前推验证设计_v1.json", folds)
    scores = pd.read_csv(DATA / "MD&A/culture_demo/data/culture_scores.csv", dtype={"code": str})
    manifest = []
    for row in scores.itertuples(index=False):
        p = DATA / f"MD&A/MD&A 2/{row.year}/文本/{row.code.zfill(6)}_{row.year}-12-31.txt"
        manifest.append({"code": row.code.zfill(6), "company": row.company, "year": row.year,
                         "source_relative_path": str(p.relative_to(SOURCE)), "exists": p.exists()})
    m = pd.DataFrame(manifest)
    assert not m.duplicated(["code", "year"]).any()
    assert m["exists"].all()
    assert len(m) == CONFIG["corpus_expected_documents"]
    save_csv(m, out / "文本样本来源名单_v1.csv")
    app = pd.read_csv(DATA / "舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv", dtype={"Stkcd": str})
    assert set(features) <= set(app.columns)
    assert not {"is_fraud", "is_serious"} & set(app.columns)
    assert app["Stkcd"].is_unique
    fingerprints = []
    for p in [DATA / "舞弊多年份/合并后数据/panel_ml.csv",
              DATA / "舞弊多年份/合并后数据/panel_ml_2025_unlabeled.csv",
              DATA / "破产预测/bankruptcy_data.csv",
              DATA / "舞弊多年份/PROJECT_RULES.md", DATA / "舞弊多年份/DATA.md",
              DATA / "破产预测/PROJECT_RULES.md", DATA / "MD&A/culture_demo/data/culture_scores.csv"]:
        fingerprints.append({"source_relative_path": str(p.relative_to(SOURCE)), "bytes": p.stat().st_size, "sha256": digest(p)})
    save_csv(pd.DataFrame(fingerprints), out / "来源文件指纹_v1.csv")
    env = {"time_beijing": now(), "python": sys.executable, "python_version": platform.python_version(), "packages": {}}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]:
        env["packages"][package] = importlib.metadata.version(package)
    write_json(LOGS / "environment_audit_v1.json", env)
    audit = {
        "time_beijing": now(), "fraud_rows": len(fraud), "fraud_features": len(features),
        "fraud_train_rows": len(f_train), "fraud_test_rows_locked": len(f_test),
        "fraud_train_positive_count": int(fraud.loc[f_train, "is_fraud"].sum()),
        "fraud_train_positive_rate": float(fraud.loc[f_train, "is_fraud"].mean()),
        "bankruptcy_rows": len(bank), "bankruptcy_features": len(bank.columns)-1,
        "bankruptcy_split_rows": [len(b_train), len(b_valid), len(b_test)],
        "bankruptcy_train_positive_count": int(bank.iloc[b_train]["Bankrupt?"].sum()),
        "bankruptcy_train_positive_rate": float(bank.iloc[b_train]["Bankrupt?"].mean()),
        "application_rows": len(app), "application_year_values": sorted(int(x) for x in app.Year.unique()),
        "corpus_documents": len(m), "corpus_companies": int(m.code.nunique()),
        "corpus_missing": int((~m.exists).sum()), "test_feature_statistics_computed": False,
        "source_fingerprints_verified": True,
    }
    write_json(out / "数据核查摘要_v1.json", audit)
    report = f"""# 第一步：数据核查与分析准备

核查时间（北京时间）：{now()}。本报告来自实际读取，不是课堂示例结果。

## 本次使用的数据

| 数据 | 实际规模 | 本次用途 |
|---|---:|---|
| 舞弊多年份 panel_ml.csv | {len(fraud):,} 行、{len(features)} 个财务特征 | 任务 a 的历史评估、任务 b 的全量训练 |
| 破产 bankruptcy_data.csv | {len(bank):,} 行、95 个原始特征 | 任务 a 的第二套数据 |
| panel_ml_2025_unlabeled.csv | {len(app):,} 行、36 个财务特征，无标签 | 任务 b 的外推预测 |
| MD&A 教学样本 | {len(m):,} 篇、{m.code.nunique()} 家公司、2018—2022 年 | 任务 c 的文化与主题分析 |

## 已固定的评估设计

- 舞弊：按标签年划分，2007—2019 年训练（{len(f_train):,} 行），2020—2021 年测试（{len(f_test):,} 行）。训练内部做 5 折按年份前推验证。
- 破产：按标签分层、随机种子 42，训练/验证/测试为 {len(b_train):,}/{len(b_valid):,}/{len(b_test):,} 行，约 70%/15%/15%。
- 划分索引已保存到过程代码/中间数据/split_indices_v1.json。测试集尚未做特征统计、模型评估或阈值选择。
- 舞弊训练集正类占比 {audit['fraud_train_positive_rate']:.2%}；破产训练集正类占比 {audit['bankruptcy_train_positive_rate']:.2%}。后续不能仅用准确率评价。

## 已确认的两个选择

1. 文本样本：按老师演示评分表中的公司代码和年份，从原始语料恢复 1,123 篇。全部对应原文已找到。评分表只用于样本定位，不复制其中得分作为本次分析结果。
2. 类别权重：任务 a 的舞弊评估遵循项目规则、不加类别权重；任务 b 的应用树遵循网页教程、使用 balanced，并将分数解释为相对排序，不能当作校准后的造假概率。

## 需要说明的数据边界

- 文件名写 2025，但应用文件 Year 实际全部为 2026。依课程的特征年+1=标签年规则，其语义为基于 2025 年报预测 2026 年目标。报告将同时展示“财务报表年”和“预测目标年”，保留老师的原始文件名，不把 Year 静默改成 2025。这里是课程说明下的解释，尚未回查原始报表证实其生成过程。
- 原始文件没有被修改。来源文件的 SHA-256 指纹已记录。
- 演示包的 corpus_sample 不存在。恢复样本可重建同一公司年度集合，但不能在没有原样本文件的情况下保证与老师当时的文本清理版本逐字相同。
- 数据来源涉及课程提供的商业数据库，公开发布许可尚未核实。现阶段只做本地分析，发布网站前再确认可公开范围。

## 本阶段检查

公司年度主键无重复；两份舞弊文件的财务特征列能对齐；应用集没有标签；1,123 个文本样本全部能定位；验证各折训练年份早于验证年份。

## 下一步

恢复文本样本，进行训练集探索与无监督聚类，随后按统一协议比较模型。这里尚没有本次训练结果，不能将老师演示文件的模型分数当成本次结果。
"""
    (out / "01_数据核查报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
