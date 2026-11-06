"""部署回执写入本地最终包，建立逐文件指纹并生成完整 ZIP。"""
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
FINAL=ROOT/'成果交付/10_最终提交_zs'


def main():
    receipt=json.loads((HERE/'work/publication_receipt.json').read_text(encoding='utf-8'))
    (FINAL/'07_核查与说明/发布与提交回执.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[p for p in FINAL.rglob('*') if p.is_file() and p.name!='文件指纹_SHA256.csv']
    import csv
    with (FINAL/'07_核查与说明/文件指纹_SHA256.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['文件','字节','SHA256'])
        for p in sorted(files):writer.writerow([p.relative_to(FINAL).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    archive=ROOT/'成果交付/课程作业_最终提交_zs_20261106.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in FINAL.rglob('*'):
            if p.is_file():z.write(p,str(Path(FINAL.name)/p.relative_to(FINAL)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    root_entry=f'''# 当前最终交付入口

整理日期：2026-11-06（北京时间）。公开署名 zs。

- 完整包：`课程作业_最终提交_zs_20261106.zip`。
- 解压 / 文件夹入口：`10_最终提交_zs/00_先看这里.html`。
- 完整清单：`10_最终提交_zs/00_提交清单.md`。
- 网站：{receipt['site_url']}
- 代码：{receipt['repository_url']}

包内有一页说明、综合报告、两分钟 AI 视频、全部 a—c 分析成果、离线网站、源码和核查记录。旧目录保留为历史版本，不必逐个挑选提交。
'''
    (ROOT/'成果交付/00_最终提交入口.md').write_text(root_entry,encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
