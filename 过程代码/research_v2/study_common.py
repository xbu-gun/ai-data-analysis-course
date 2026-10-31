"""第二轮独立目录、配置指纹与只读保护。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys

STUDY=Path(__file__).resolve().parent
CODE=STUDY.parent
ROOT=CODE.parent
BASE_CONFIG=json.loads((CODE/'project_config.json').read_text(encoding='utf-8'))
SOURCE=(CODE/BASE_CONFIG['source_root']).resolve()
DATA=SOURCE/'数据集文档'
OUT=ROOT/'成果交付/07_优化研究_v2'
WORK=STUDY/'work'
LOG=STUDY/'logs'
PROTOCOL=json.loads((STUDY/'protocol.json').read_text(encoding='utf-8'))

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def default(x):
        if type(x).__module__.startswith('numpy') and hasattr(x,'item'):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=default),encoding='utf-8')

def save(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,encoding='utf-8-sig')

def setup():
    for p in [OUT,WORK,LOG]:p.mkdir(parents=True,exist_ok=True)

def freeze_v1():
    setup();target=STUDY/'v1_preservation_manifest.json'
    if target.exists():
        rows=json.loads(target.read_text(encoding='utf-8'))['files']
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in rows),'v1文件已变化，先核查，不继续实验。'
        return
    files=[p for p in (ROOT/'成果交付').rglob('*') if p.is_file() and OUT not in p.parents]
    files += [CODE/'project_config.json',CODE/'README.md',CODE/'requirements.txt']
    files += list((CODE/'scripts').glob('*.py'))+list((CODE/'scripts').glob('*.mjs'))+list((CODE/'web').glob('*'))
    dump(target,{'created_beijing':now(),'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(files))]})

def assert_v1_unchanged():
    rows=json.loads((STUDY/'v1_preservation_manifest.json').read_text(encoding='utf-8'))['files']
    changed=[r['path'] for r in rows if not (ROOT/r['path']).exists() or sha(ROOT/r['path'])!=r['sha256']]
    assert not changed,changed
    return len(rows)

if __name__=='__main__':
    freeze_v1();print('v1 preservation verified:',assert_v1_unchanged(),'files')
