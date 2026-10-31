"""按用户确认的教学样本名单恢复原文，逐篇校验指纹，不使用演示得分。"""
import pandas as pd
from common import *


def main():
    setup()
    manifest = pd.read_csv(OUTPUT / "01_数据核查/文本样本来源名单_v1.csv", dtype={"code": str})
    out = WORK / "corpus_sample"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in manifest.itertuples(index=False):
        source = SOURCE / r.source_relative_path
        # 保留原始字节；以原始代码和报告期命名，名称只进入名单，避免混淆原文与元数据。
        target = out / f"{r.code}_{r.year}-12-31.txt"
        content = source.read_bytes()
        if target.exists():
            assert target.read_bytes() == content, f"现有恢复文本不一致：{target}"
        else:
            target.write_bytes(content)
        sha = digest(source)
        assert digest(target) == sha
        text = content.decode("utf-8-sig")
        rows.append({"code": r.code, "company": r.company, "year": r.year,
                     "file": target.name, "source_relative_path": r.source_relative_path,
                     "sha256": sha, "characters": len(text), "source_bytes": len(content)})
    m = pd.DataFrame(rows)
    assert len(m) == 1123 and m.code.nunique() == 225
    save_csv(m, out / "_manifest.csv")
    save_csv(m, OUTPUT / "01_数据核查/恢复文本样本指纹_v1.csv")
    summary = {"time_beijing": now(), "documents": len(m), "companies": int(m.code.nunique()),
               "characters": int(m.characters.sum()), "bytes": int(m.source_bytes.sum()),
               "hash_verified_documents": len(m), "teacher_scores_used_as_results": False}
    write_json(LOGS / "corpus_recovery_v1.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
