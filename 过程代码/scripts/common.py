"""路径、时间与可复现输出；课程原始资料始终只读。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os

CODE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((CODE / "project_config.json").read_text(encoding="utf-8"))
SOURCE = (CODE / CONFIG["source_root"]).resolve()
DATA = SOURCE / "数据集文档"
OUTPUT = (CODE / CONFIG["output_root"]).resolve()
WORK = CODE / "中间数据"
LOGS = CODE / "运行记录"
MODELS = CODE / "模型"
SEED = CONFIG["seed"]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def scalar(v):
        if type(v).__module__.startswith("numpy") and hasattr(v, "item"):
            return v.item()
        raise TypeError(f"Unsupported JSON type: {type(v).__name__}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar), encoding="utf-8")


def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def setup():
    for p in [WORK, LOGS, MODELS, OUTPUT]:
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def load_finance():
    import pandas as pd
    fraud = pd.read_csv(DATA / "舞弊多年份/合并后数据/panel_ml.csv", dtype={"Stkcd": str})
    fraud["Stkcd"] = fraud["Stkcd"].str.zfill(6)
    bank = pd.read_csv(DATA / "破产预测/bankruptcy_data.csv")
    bank.columns = bank.columns.str.strip()
    return fraud, bank
