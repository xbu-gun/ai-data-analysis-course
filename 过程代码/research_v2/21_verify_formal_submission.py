"""核查正式提交版的链接、公开别名、归档完整性及源码上传清单。"""
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import ast,zipfile
from study_common import *

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if 'id' in values:
            assert values['id'] not in self.ids;self.ids.add(values['id'])
        for name in ['href','src']:
            if name in values:self.refs.append(values[name])

def main():
    config=json.loads((STUDY/'release_config.json').read_text(encoding='utf-8'))
    private=config.get('private_author',config['public_author']);site=Path(config['site_checkout'])/'dist'
    final=ROOT/f'成果交付/08_正式提交_{private}';repo=CODE/'发布项目/github-course-submission'
    parser=Links();parser.feed((site/'index.html').read_text(encoding='utf-8'))
    resources=[]
    for ref in parser.refs:
        if ref.startswith('#'):assert ref[1:] in parser.ids,ref
        elif not urlparse(ref).scheme:
            target=site/unquote(ref.split('#')[0]);assert target.is_file(),ref;resources.append(ref)
    assert len(parser.ids)>=10
    from pypdf import PdfReader
    public_pdf=site/'downloads/onepage_zs.pdf';reader=PdfReader(public_pdf)
    assert len(reader.pages)==1 and 'zs' in reader.pages[0].extract_text()
    assert private not in reader.pages[0].extract_text() if private!='zs' else True
    assert private not in str(reader.metadata) if private!='zs' else True
    assert len(PdfReader(final/f'一页结果说明_{private}.pdf').pages)==1
    archives=[]
    for name in ['source_code.zip','teacher_results.zip']:
        p=site/'downloads'/name
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for item in z.infolist():
                assert not item.filename.startswith(('/', '\\')) and '..' not in Path(item.filename).parts
                if Path(item.filename).suffix.lower() in {'.py','.mjs','.js','.css','.html','.md','.json','.csv','.txt'}:
                    value=z.read(item).decode('utf-8-sig')
                    assert private not in value if private!='zs' else True
                    assert 'zs' not in value
            archives.append({'file':name,'files':len(z.infolist()),'bytes':p.stat().st_size,'sha256':sha(p)})
    entries=[]
    for p in sorted(repo.rglob('*')):
        if not p.is_file():continue
        value=p.read_text(encoding='utf-8')
        assert private not in value if private!='zs' else True
        assert 'zs' not in value
        assert not any(part in {'.git','.sites','.venv','node_modules'} for part in p.relative_to(repo).parts)
        if p.suffix=='.py':ast.parse(value)
        entries.append({'path':p.relative_to(repo).as_posix(),'mode':'100644','type':'blob','content':value})
    dump(STUDY/'work/github_tree_entries.json',entries)
    large=[str(p.relative_to(site)) for p in site.rglob('*') if p.is_file() and p.stat().st_size>25*1024*1024]
    assert not large,large
    result={'time_beijing':now(),'public_author':'zs','public_real_name_check_passed':True,
            'all_page_resource_links_exist':len(resources),'onepage_pdf_pages':1,'archives':archives,'github_source_files':len(entries),
            'site_files':len([p for p in site.rglob('*') if p.is_file()]),'no_file_over_25_mib':True,
            'browser_visual_qa':'未完成：Windows浏览器当前URL识别限制；不把程序逻辑核查冒充浏览器实测',
            'v1_files_unchanged':assert_v1_unchanged()}
    dump(final/'正式交付核查.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
