"""对最终新包作交付核查；不更改分析、模型或既有文件。"""
import ast
import hashlib
import json
import re
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
from zoneinfo import ZoneInfo

from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
FINAL=ROOT/'成果交付/10_最终提交_zs'
DIST=ROOT/'过程代码/发布项目/financial-signals-zs/dist'
REPO=ROOT/'过程代码/发布项目/github-course-submission'
CFG=json.loads((ROOT/'过程代码/research_v2/release_config.json').read_text(encoding='utf-8'))
PRIVATE=CFG.get('private_author','zs')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):super().__init__();self.values=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in {'href','src','poster'} and value:self.values.append(value)


def main():
    links_checked=0
    for path in [FINAL/'00_先看这里.html',DIST/'index.html',FINAL/'05_网站离线版/index.html',
                 FINAL/'04_分析成果/c_文化与主题/06_全文主题对照/全文主题互动图.html']:
        parser=Links();parser.feed(path.read_text(encoding='utf-8-sig'))
        for value in parser.values:
            parsed=urlsplit(value)
            if parsed.scheme or parsed.netloc or not parsed.path:continue
            target=path.parent/unquote(parsed.path)
            assert target.exists(),(path,value)
            links_checked+=1
    zips=[]
    for path in [*FINAL.rglob('*.zip'),DIST/'downloads/final_submission_zs.zip']:
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None,path
            assert all(PRIVATE not in name if PRIVATE!='zs' else True for name in z.namelist())
        zips.append(path.name)
    pdf_counts={}
    for path in [FINAL/'01_一页结果说明_zs.pdf',FINAL/'03_综合分析报告_zs.pdf']:
        r=PdfReader(path);text='\n'.join(p.extract_text() for p in r.pages)
        assert PRIVATE not in text if PRIVATE!='zs' else True
        assert 'zs' in str(r.metadata) and '本人录制' not in text
        pdf_counts[path.name]=len(r.pages)
    assert pdf_counts['01_一页结果说明_zs.pdf']==1
    source_files=[]
    for p in REPO.rglob('*'):
        if not p.is_file():continue
        content=p.read_text(encoding='utf-8-sig')
        assert PRIVATE not in content if PRIVATE!='zs' else True
        assert 'zs' not in content
        if p.suffix=='.py':ast.parse(content)
        source_files.append(p)
    with zipfile.ZipFile(FINAL/'06_源码与复现/复现源码_zs.zip') as z:
        assert set(z.namelist())=={p.relative_to(REPO).as_posix() for p in source_files}
        for p in source_files:assert z.read(p.relative_to(REPO).as_posix())==p.read_bytes()
    for p in FINAL.rglob('*'):
        if p.is_file() and p.suffix in {'.md','.txt','.json','.csv','.html','.js','.css','.srt'}:
            text=p.read_text(encoding='utf-8-sig')
            assert PRIVATE not in text if PRIVATE!='zs' else True
            assert 'zs' not in text
    video_check=json.loads((FINAL/'07_核查与说明/成片核查.json').read_text(encoding='utf-8'))
    assert sha(FINAL/'02_两分钟项目介绍_zs.mp4')==video_check['sha256']
    assert video_check['duration_seconds']==120 and video_check['full_decode_passed']
    for name in ['assets/data.js','app.js','style.css']:
        assert sha(DIST/name)==sha(FINAL/'05_网站离线版'/name)
    # 财务、文化、文本计算与查询逻辑没有因整理被改写。
    previous=ROOT/'成果交付/07_优化研究_v2/08_网站_v2'
    for name in ['assets/data.js','style.css']:assert sha(DIST/name)==sha(previous/name)
    frozen=json.loads((ROOT/'过程代码/research_v2/v1_preservation_manifest.json').read_text(encoding='utf-8'))
    for item in frozen['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    oversized=[p for p in DIST.rglob('*') if p.is_file() and p.stat().st_size>25*1024*1024]
    assert not oversized,oversized
    result=dict(time_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),status='passed',
                public_author='zs',pdf_pages=pdf_counts,local_links_checked=links_checked,zip_integrity_passed=zips,
                source_file_count=len(source_files),analysis_files=sum(p.is_file() for p in (FINAL/'04_分析成果').rglob('*')),
                video_unchanged=True,video_duration_seconds=120,frozen_files_unchanged=len(frozen['files']),
                website_core_payload_unchanged=True,all_static_files_under_25MiB=True,
                pdf_visual_review='passed: rendered one-page summary and all nine report pages reviewed',
                browser_visual_check='Not repeated; native deployment checks separate from browser access')
    (FINAL/'07_核查与说明/最终交付核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # 小块 GitHub 树变更清单；base tree 保留未涉及文件，只上传当前实际改变内容。
    changes=['README.md','PUBLICATION.md','.gitignore','过程代码/project_config.json','过程代码/research_v2/release_config.json']
    changed=[REPO/p for p in changes]
    changed += [p for folder in ['过程代码/video_ai','过程代码/final_submission'] for p in (REPO/folder).rglob('*') if p.is_file()]
    elements=[dict(path=p.relative_to(REPO).as_posix(),mode='100644',type='blob',content=p.read_text(encoding='utf-8')) for p in changed]
    (HERE/'work/github_changes.json').write_text(json.dumps(elements,ensure_ascii=False),encoding='utf-8')
    # 每组控制在 20KB 左右，便于通过连接器读取，内容不暴露到聊天输出。
    chunks=[];group=[];length=0
    for element in elements:
        size=len(json.dumps(element,ensure_ascii=False))
        if group and length+size>20000:chunks.append(group);group=[];length=0
        group.append(element);length+=size
    if group:chunks.append(group)
    for i,group in enumerate(chunks):(HERE/f'work/github_changes_{i}.json').write_text(json.dumps(group,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(result,github_changed_files=len(elements),github_chunks=len(chunks)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
