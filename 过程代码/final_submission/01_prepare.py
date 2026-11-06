"""汇总最终交付；仅生成新版本，原数据及 590 份冻结结果保持只读。"""
import ast
import html
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether

HERE=Path(__file__).resolve().parent
CODE=HERE.parent
ROOT=CODE.parent
RESULTS=ROOT/'成果交付'
OLD=next(RESULTS.glob('08_正式提交_*'))
VIDEO=RESULTS/'09_AI视频_zs'
FINAL=RESULTS/'10_最终提交_zs'
SITE=CODE/'发布项目/financial-signals-zs'
DIST=SITE/'dist'
REPO=CODE/'发布项目/github-course-submission'
WORK=HERE/'work'
CFG=json.loads((CODE/'research_v2/release_config.json').read_text(encoding='utf-8'))
PUBLIC='zs'
PRIVATE=CFG.get('private_author',PUBLIC)
DATE=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
WEB=CFG['expected_site_url']
GITHUB=CFG['repository_url']
TEXT_EXT={'.md','.txt','.json','.csv','.html','.js','.css','.py','.ps1','.mjs'}


def public(s):
    return s.replace(PRIVATE,PUBLIC).replace('zs','zs')


def copy(src,dst,sanitize=False):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if sanitize and src.suffix.lower() in TEXT_EXT:
        dst.write_text(public(src.read_text(encoding='utf-8-sig')),encoding='utf-8')
    else:shutil.copyfile(src,dst)


def zip_pairs(path,pairs):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for src,name in pairs:z.write(src,str(name).replace('\\','/'))


def onepage():
    # 复用已核查的一页排版；只更新日期和视频状态，锁定的模型数值不变。
    original=(CODE/'research_v2/20_create_final_onepage.py').read_text(encoding='utf-8')
    tree=ast.parse(original)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create')
    source=ast.get_source_segment(original,fn)
    source=source.replace('2026-10-31',DATE).replace(
        '本人决定样本、权重、噪声词和展示方案；AI 协助实现、计算、语境初审与证据整理。没有将 AI 初审写成独立人工标注，真人介绍视频由本人录制。',
        '本人确认样本、权重、删词和展示方案；AI 协助实现、计算、语境初审与证据整理。两分钟视频采用 AI 合成男声和动态图表，已标注；未模拟本人声线或形象。')
    env=dict(globals());env['CFG']=dict(expected_site_url=WEB,repository_url=GITHUB)
    exec(compile(source,'onepage_layout','exec'),env)
    p=FINAL/'01_一页结果说明_zs.pdf'
    env['create'](p,PUBLIC)
    reader=PdfReader(p)
    assert len(reader.pages)==1
    value=reader.pages[0].extract_text()
    assert 'AI 合成男声' in value and '本人录制' not in value
    (FINAL/'07_核查与说明/一页结果说明_可检索文本.md').write_text(value,encoding='utf-8')
    copy(p,DIST/'downloads/onepage_zs.pdf')


def report():
    pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
    styles={
        'body':ParagraphStyle('Body',fontName='CJK',fontSize=9.4,leading=16,spaceAfter=7,wordWrap='CJK'),
        'h1':ParagraphStyle('H1',fontName='CJKBold',fontSize=18,leading=25,textColor=HexColor('#166B62'),spaceAfter=14,keepWithNext=True),
        'h2':ParagraphStyle('H2',fontName='CJKBold',fontSize=12,leading=18,spaceBefore=10,spaceAfter=7,keepWithNext=True),
        'cell':ParagraphStyle('Cell',fontName='CJK',fontSize=8.1,leading=12,wordWrap='CJK'),
        'caption':ParagraphStyle('Caption',fontName='CJK',fontSize=8,leading=12,textColor=HexColor('#647575'),spaceAfter=12,alignment=1),
    }
    def rich(s):
        s=html.escape(s.strip()).replace('`','')
        s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#166B62">{m[1]}</link>',s)
        return s
    story=[]
    def para(s,style='body'):story.append(Paragraph(rich(s),styles[style]))
    def add_md(path):
        lines=path.read_text(encoding='utf-8-sig').splitlines();i=0
        while i<len(lines):
            s=lines[i].strip();i+=1
            if not s:continue
            if s.startswith('|'):
                rows=[s]
                while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
                rows=[r for r in rows if not re.fullmatch(r'[| :\-]+',r)]
                values=[[c.strip() for c in r.strip('|').split('|')] for r in rows]
                n=len(values[0]);assert all(len(r)==n for r in values)
                if '代表词' in values[0]:widths=[28,154,34,291]
                elif n==3:widths=[217,145,145]
                else:widths=[507/n]*n
                tab=Table([[Paragraph(rich(v),styles['cell']) for v in row] for row in values],colWidths=widths,repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#E3EBE3')),('TEXTCOLOR',(0,0),(-1,0),HexColor('#172F37')),
                                         ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
                                         ('LINEBELOW',(0,0),(-1,0),.5,HexColor('#B8CAC0')),('LINEBELOW',(0,1),(-1,-1),.3,HexColor('#DDE4DD'))]))
                story.extend([tab,Spacer(1,10)]);continue
            img=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',s)
            if img:
                src=path.parent/img[2]
                if src.exists():
                    im=Image(str(src));scale=min(507/im.imageWidth,260/im.imageHeight)
                    im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale
                    story.extend([im,Paragraph(rich(img[1]),styles['caption'])])
                continue
            if s.startswith('# '):para(s[2:],'h1')
            elif s.startswith('## '):para(s[3:],'h2')
            elif s.startswith('### '):para(s[4:],'h2')
            else:para(s)
    para('财务信号与文本画像','h1')
    para(f'AI 数据分析课程作业 | zs | 最终整理：{DATE}')
    para('本报告汇总课程任务 a—d 已实际运行的成果。模型、样本、词典和主题方案均沿用已确认版本；本次仅整理交付，未重新使用测试集调参或改变数值。')
    para('阅读顺序','h2')
    para('a：两套数据的探索、模型选择与测试错误；b：2025 年报应用风险排序与信心边界；c：五维文化指标及全文主题方法、变化与解释；d：网站、代码、两分钟视频与可核查证据。')
    para('网站：'+WEB);para('代码：'+GITHUB)
    para('完整明细见同包“04_分析成果”中的 CSV、图和逐条错误记录；本报告不以节选替代完整结果。视频为普通话 AI 合成男声与动态图表，已披露合成方式，未模拟本人声线或形象。')
    for heading,src in [
        ('a / 两套数据完整流程',FINAL/'04_分析成果/a_完整流程/流程记录.md'),
        ('b / 应用模型及信心说明',FINAL/'04_分析成果/b_2025风险预测/模型说明.md'),
        ('c / 五维文化指标',FINAL/'04_分析成果/c_文化与主题/文化指标方法说明_v2.md'),
        ('c / 全文主题及原方案对照',FINAL/'04_分析成果/c_文化与主题/06_全文主题对照/第二阶段_全文主题对照报告.md')]:
        story.append(PageBreak());para(heading,'h1');add_md(src)
    story.append(PageBreak());para('d / 网站与人的判断','h1')
    para('网站可查询公司财务指标、风险排序和应用树实际判定路径，展示五维文化得分、词云及全文与首中尾主题对照。5,574 家公司有财务应用结果；225 家公司有文本结果，其中 15 家无对应年财务值，页面明确显示缺失，不补造分数。')
    para('人的判断与 AI 分工','h2')
    para('作者确认样本恢复方式、任务 a 与 b 权重分开、文化公式修正、28 词剔除、聚类正文 2 组并保留 4 组附录，以及全文主题作为网站主展示。AI 协助代码实现、计算、语境初审、证据整理与视频制作；没有把 AI 初审称为独立人工标注。')
    para('复现与未解决问题','h2')
    para('源码包包含分析、优化、网站、视频和本次整理代码。分析仍需教师课程原包和课程附带 BGE 模型；环境和模型权重不随源码上传。原始财务口径、应计项定义和历史更正版本仍待核实，保留课堂值；现有结果不证明真实事前预警、公司实际文化或官方行业归属。')
    para('最终提交项目','h2')
    para('网站链接、GitHub 仓库链接、一页结果说明和两分钟介绍视频为优先提交项。附件提供 a—c 完整表格与图、综合报告、离线网站、源码、视频字幕以及核查说明。')
    def page(c,doc):
        c.setTitle('财务信号与文本画像：综合分析报告');c.setAuthor('zs')
        c.setFont('CJK',8);c.setFillColor(HexColor('#647575'))
        c.drawString(44,23,'AI 数据分析课程作业 | zs | '+DATE)
        c.drawRightString(A4[0]-44,23,str(doc.page))
    p=FINAL/'03_综合分析报告_zs.pdf'
    SimpleDocTemplate(str(p),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=40,bottomMargin=43,
                      title='财务信号与文本画像：综合分析报告',author='zs').build(story,onFirstPage=page,onLaterPages=page)
    copy(p,DIST/'downloads/analysis_report_zs.pdf')


def docs():
    video_name='02_两分钟项目介绍_zs.mp4'
    value=f'''# 最终提交清单

署名：zs。最终整理日期：{DATE}（北京时间）。本目录为当前提交版本；其他旧目录保留为研究与制作历史。

## 交给老师的内容

1. 网站：{WEB}
2. 代码仓库：{GITHUB}
3. 一页结果说明：`01_一页结果说明_zs.pdf`。
4. 两分钟介绍视频：`{video_name}`。120 秒，1080p，AI 合成男声与画面内中文字幕，已注明合成方式。

老师课件明确的三项正式提交为网站链接、代码仓库和一页结果说明；同一课件另要求录制两分钟介绍视频，本包一并备齐。提交渠道和截止时间未在已核查课件页中注明。

## 随附完整材料

- `03_综合分析报告_zs.pdf`：流程、成绩、错误、文化公式、主题解释、敏感性和局限的可阅读汇总。
- `04_分析成果/a_完整流程/`：舞弊与破产的探索、模型比较、测试成绩、错误明细及来源 / 尺度核查。
- `04_分析成果/b_2025风险预测/`：5,574 家公司风险表、模型说明、实际树规则及历史回测。
- `04_分析成果/c_文化与主题/`：784 词词典、五维指标、全文主题、图、原文证据、原方案和种子对照。互动图附本地 Plotly 资源。
- `05_网站离线版/index.html`：解压完整包后可打开的离线网站；依赖文件均在本目录中。
- `06_源码与复现/复现源码_zs.zip`：与当前公开代码整理范围一致；含视频及本次整理代码。复跑分析需教师原包与模型，未把大环境、权重或原始大表放入源码。
- `07_核查与说明/`：发布回执、文件指纹、视频与结果核查。
- `08_视频辅助材料/`：字幕、封面、旁白和分镜。MP4 已含声音与字幕，可直接单独播放。

## 怎么提交

若平台按项填写，粘贴两个链接，上传一页 PDF 和 MP4；综合报告与分析目录可作为附件。若平台接收压缩包，上传本目录同名的完整 ZIP；线上也提供不含离线网站的精简提交包。请勿把原始课程包、旧版目录或整个开发环境一并上传。

数据和结论沿用已确认方案。公开材料统一 zs；原实名材料和旧“本人录制”安排保留原位置，本包以已完成的 AI 视频为准。
'''
    (FINAL/'00_提交清单.md').write_text(value,encoding='utf-8')
    (FINAL/'提交链接.txt').write_text(f'网站：{WEB}\nGitHub：{GITHUB}\n公开署名：zs\n视频：{video_name}\n',encoding='utf-8')
    (FINAL/'网站链接.url').write_text('[InternetShortcut]\nURL='+WEB+'\n',encoding='utf-8')
    (FINAL/'代码仓库.url').write_text('[InternetShortcut]\nURL='+GITHUB+'\n',encoding='utf-8')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最终提交 · zs</title><style>body{{margin:0;background:#f4f2eb;color:#172f37;font:17px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1000px;margin:50px auto;padding:0 25px}}small{{color:#647575}}h1{{font-size:40px;line-height:1.3}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}article{{background:#fffef9;padding:24px;border-radius:14px}}a{{color:#087e78;text-decoration:none}}h2{{margin:0 0 10px;font-size:23px}}video{{width:100%;border-radius:12px;margin-top:15px}}.note{{padding:20px 0;color:#647575}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}</style><main><small>AI 数据分析课程作业 / zs / {DATE}</small><h1>财务信号与文本画像<br>最终提交入口</h1><p>老师要求的网站、代码仓库、一页说明，以及两分钟介绍视频，均从这里打开。</p><div class="grid"><article><h2>01 · 网站</h2><a href="{WEB}" target="_blank" rel="noopener">打开公开网站 ↗</a><br><a href="05_网站离线版/index.html">打开离线网站 ↗</a></article><article><h2>02 · 代码</h2><a href="{GITHUB}" target="_blank" rel="noopener">打开 GitHub 仓库 ↗</a><br><a href="06_源码与复现/复现源码_zs.zip" download>下载源码 ↗</a></article><article><h2>03 · PDF</h2><a href="01_一页结果说明_zs.pdf">一页结果说明 ↗</a><br><a href="03_综合分析报告_zs.pdf">综合分析报告 ↗</a></article><article><h2>04 · 分析明细</h2><a href="04_分析成果/a_完整流程/流程记录.md">a：完整流程 ↗</a><br><a href="04_分析成果/b_2025风险预测/2025年报_风险预测表_v1.csv">b：风险预测表 ↗</a><br><a href="04_分析成果/c_文化与主题/06_全文主题对照/全文主题互动图.html">c：全文主题互动图 ↗</a></article></div><h2 style="margin-top:30px">两分钟项目介绍</h2><p>普通话 AI 合成男声、中文字幕，未模拟作者本人声音或形象。</p><video controls preload="metadata" poster="08_视频辅助材料/视频封面_zs.png"><source src="{video_name}" type="video/mp4"></video><p><a href="{video_name}" download>直接下载视频 ↗</a> · <a href="00_提交清单.md">查看完整清单 ↗</a></p><p class="note">解压完整包后使用本页。评分提供相对线索，文化指标衡量表述，主题不是官方行业分类。</p></main></html>'''
    (FINAL/'00_先看这里.html').write_text(page,encoding='utf-8')


def sources():
    # 既有 60 个公开源码为基础，新增视频与最终整理代码；不纳入依赖、缓存或凭据。
    for p in (CODE/'video_ai').iterdir():
        if p.is_file() and (p.suffix in {'.py','.ps1','.json','.md'} or p.name=='.gitignore'):
            copy(p,REPO/'过程代码/video_ai'/p.name,sanitize=True)
    for p in HERE.glob('*.py'):copy(p,REPO/'过程代码/final_submission'/p.name,sanitize=True)
    for p in HERE.glob('*.md'):copy(p,REPO/'过程代码/final_submission'/p.name,sanitize=True)
    readme=(REPO/'README.md').read_text(encoding='utf-8')
    readme=re.sub(r'https://financial-signals-zs\.[a-z0-9.-]+\.chatgpt\.site',WEB,readme)
    readme=readme.replace('真人视频另由本人录制。','两分钟介绍视频已完成，采用 AI 合成男声与动态图表，片头披露；未模拟本人声音或形象。')
    additions=f'\n- 两分钟视频：[播放 / 下载 MP4]({WEB}/downloads/project_video_zs.mp4)\n- 综合报告：[下载 PDF]({WEB}/downloads/analysis_report_zs.pdf)\n- 最终提交材料：[精简提交包]({WEB}/downloads/final_submission_zs.zip)\n'
    readme=readme.replace('## 作业对应',additions+'\n## 作业对应') if '精简提交包' not in readme else readme
    (REPO/'README.md').write_text(readme,encoding='utf-8')
    config_path=REPO/'过程代码/research_v2/release_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'));config['video_mode']='AI 合成男声与动态图表，成片 120 秒，已披露合成方式';config['status']='final_submission_assembled';config['date_beijing']=DATE;config['expected_site_url']=WEB
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    config_path=REPO/'过程代码/project_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'));config['public_release_status']='public_data_permission_confirmed_2026-10-31'
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (REPO/'PUBLICATION.md').write_text(f'# 当前交付状态\n\n公开署名 zs，整理日期 {DATE}。网站：{WEB}；仓库：{GITHUB}。两分钟 AI 旁白视频、一页说明、综合报告及 a—c 明细已整理。源码中的旧准备脚本保留其历史行为；当前入口以根 README 和网站“提交成果”为准。原始数据及模型权重需课程原包。\n',encoding='utf-8')
    gitignore=REPO/'.gitignore'
    value=gitignore.read_text(encoding='utf-8')
    for entry in ['过程代码/video_ai/dependencies/','过程代码/video_ai/work/','过程代码/final_submission/work/']:
        if entry not in value:value+='\n'+entry
    gitignore.write_text(value+'\n',encoding='utf-8')
    files=[p for p in REPO.rglob('*') if p.is_file()]
    for p in files:
        assert PRIVATE not in p.read_text(encoding='utf-8') if PRIVATE!=PUBLIC else True
    dst=FINAL/'06_源码与复现/复现源码_zs.zip'
    zip_pairs(dst,[(p,p.relative_to(REPO)) for p in files]);copy(dst,DIST/'downloads/source_code.zip')
    (FINAL/'06_源码与复现/复现说明.md').write_text(f'# 复现范围\n\n源码包含 {len(files)} 个文本源码与配置文件。网站、表格和报告已经计算完成；重新分析需要课程原包、BGE 模型及原有 Python 环境。分析脚本顺序见过程代码 README，第二轮见 research_v2/README，视频见 video_ai/README。视频编码组件不打包进 GitHub。\n',encoding='utf-8')
    return files


def website():
    downloads=DIST/'downloads'
    copy(VIDEO/'财务信号与文本画像_AI视频_zs.mp4',downloads/'project_video_zs.mp4')
    copy(VIDEO/'视频封面_zs.png',downloads/'video_cover_zs.png')
    copy(VIDEO/'中文字幕_zs.srt',downloads/'project_video_zs.srt')
    # 保留原下载地址兼容性，将旧真人稿替换为当前成片对应的公开稿。
    copy(VIDEO/'分镜与旁白_zs.md',downloads/'两分钟介绍稿_zs.md')
    analysis=[p for p in (FINAL/'04_分析成果').rglob('*') if p.is_file()]
    zip_pairs(downloads/'teacher_results.zip',[(p,p.relative_to(FINAL/'04_分析成果')) for p in analysis])
    index=DIST/'index.html';page=index.read_text(encoding='utf-8')
    snippet='''<div class="final-video" style="margin-top:24px"><h3>两分钟项目介绍 · zs</h3><p>AI 合成普通话男声与动态图表，中文字幕已嵌入；未模拟作者本人声音或形象。</p><video controls preload="metadata" poster="downloads/video_cover_zs.png" style="display:block;width:100%;max-width:1040px;border-radius:12px"><source src="downloads/project_video_zs.mp4" type="video/mp4">可通过下方链接下载视频。</video><p><a href="downloads/project_video_zs.mp4" download>下载视频 ↗</a> · <a href="downloads/analysis_report_zs.pdf" download>综合报告 ↗</a> · <a href="downloads/final_submission_zs.zip" download>最终提交材料 ↗</a></p></div>'''
    if 'class="final-video"' not in page:
        marker='</section><section id="method"'
        assert marker in page
        page=page.replace(marker,snippet+marker,1)
    index.write_text(page,encoding='utf-8')
    # 线上精简包包含所有需交作业成果；完整离线网站留在本地完整包，避免循环打包。
    compact=[]
    for p in FINAL.rglob('*'):
        if p.is_file() and not p.relative_to(FINAL).parts[0].startswith('05_') and p.name!='00_先看这里.html' and p.suffix.lower()!='.url':
            compact.append((p,p.relative_to(FINAL)))
    zip_pairs(downloads/'final_submission_zs.zip',compact)
    assert (downloads/'final_submission_zs.zip').stat().st_size<25*1024*1024
    # 线下副本具有和网站相同的下载资源；来源状态由后续发布回执另行记录。
    for p in DIST.rglob('*'):
        if p.is_file():copy(p,FINAL/'05_网站离线版'/p.relative_to(DIST))


def main():
    FINAL.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if '--refresh-sources' in sys.argv:
        files=sources();website()
        print(json.dumps({'source_files':len(files),'refreshed_sources_and_downloads':True}))
        return
    for d in ['04_分析成果','06_源码与复现','07_核查与说明','08_视频辅助材料']:(FINAL/d).mkdir(exist_ok=True)
    for folder in ['a_完整流程','b_2025风险预测','c_文化与主题']:
        for src in (OLD/folder).rglob('*'):
            if src.is_file():copy(src,FINAL/'04_分析成果'/folder/src.relative_to(OLD/folder),sanitize=True)
    for name in ['全文主题互动图.html','plotly.min.js']:
        copy(RESULTS/'07_优化研究_v2/06_全文主题对照'/name,FINAL/'04_分析成果/c_文化与主题/06_全文主题对照'/name,sanitize=True)
    for name in ['中文字幕_zs.srt','视频封面_zs.png','分镜与旁白_zs.md','成片关键帧预览_zs.jpg','旁白_zs.wav']:
        copy(VIDEO/name,FINAL/'08_视频辅助材料'/name)
    copy(VIDEO/'财务信号与文本画像_AI视频_zs.mp4',FINAL/'02_两分钟项目介绍_zs.mp4')
    for name in ['成片核查.json','视频制作回执.json']:copy(VIDEO/name,FINAL/'07_核查与说明'/name)
    onepage();report();docs();files=sources();website()
    summary=dict(date_beijing=DATE,public_author='zs',onepage_pages=len(PdfReader(FINAL/'01_一页结果说明_zs.pdf').pages),
                 report_pages=len(PdfReader(FINAL/'03_综合分析报告_zs.pdf').pages),source_files=len(files),
                 analysis_files=sum(p.is_file() for p in (FINAL/'04_分析成果').rglob('*')),
                 compact_zip_bytes=(DIST/'downloads/final_submission_zs.zip').stat().st_size)
    (WORK/'prepared.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
