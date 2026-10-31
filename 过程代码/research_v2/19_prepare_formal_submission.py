"""准备老师要求的正式成果；公开副本使用别名，原有分析快照保持不变。"""
import csv,html,re,shutil,zipfile
from urllib.parse import quote
import pandas as pd
from study_common import *

CFG=json.loads((STUDY/'release_config.json').read_text(encoding='utf-8'))
PUBLIC=CFG['public_author'];PRIVATE=CFG.get('private_author',PUBLIC)
FINAL=ROOT/f'成果交付/08_正式提交_{PRIVATE}'
SITE=Path(CFG['site_checkout']);DIST=SITE/'dist';DOWNLOADS=DIST/'downloads'
REPO=CODE/'发布项目/github-course-submission'

def copy(src,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def archive_files(destination,pairs):
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for source,name in pairs:z.write(source,str(name).replace('\\','/'))

def public_text(text):
    return text.replace(PRIVATE,PUBLIC).replace('zs','zs')

def main():
    setup();assert_v1_unchanged()
    for p in [FINAL,DOWNLOADS,REPO]:p.mkdir(parents=True,exist_ok=True)
    f=json.loads((ROOT/'成果交付/02_全流程分析/舞弊预测/最终测试评估_v1.json').read_text(encoding='utf-8'))
    b=json.loads((ROOT/'成果交付/02_全流程分析/破产预测/最终测试评估_v1.json').read_text(encoding='utf-8'))
    metrics='| 测试指标 | 舞弊 | 破产 |\n|---|---:|---:|\n'+f"| ROC-AUC | {f['roc_auc']:.3f} | {b['roc_auc']:.3f} |\n| AP / 正类基线 | {f['pr_auc_ap']:.3f} / {f['positive_baseline']:.3f} | {b['pr_auc_ap']:.3f} / {b['positive_baseline']:.3f} |\n| 召回率 | {f['recall']:.1%} | {b['recall']:.1%} |\n| 误报 / 漏报 | {f['confusion_matrix_tn_fp_fn_tp'][1]} / {f['confusion_matrix_tn_fp_fn_tp'][2]} | {b['confusion_matrix_tn_fp_fn_tp'][1]} / {b['confusion_matrix_tn_fp_fn_tp'][2]} |"
    a=FINAL/'a_完整流程';a.mkdir(exist_ok=True)
    report=f"""# 两套财务数据的完整流程记录

署名：{PRIVATE}。时间（北京时间）：{now()}。

舞弊面板42,511条，破产数据6,819条。先只在训练样本做分布与聚类探索，再比较逻辑回归、决策树、随机森林和XGBoost。正文聚类采用2组，4组保留作附录。聚类标签不进入有监督特征。

舞弊按标签年2007—2019训练、2020—2021测试；训练内部五折时间前推。破产按种子42分层，训练/验证/测试为4,773/1,023/1,023。填补、缩尾、相关性筛选及尺度处理在训练部分拟合；模型及阈值锁定后各终评一次。

## 测试结果与错误

两项任务均选定XGBoost；分数读取已锁定的测试结果，未因网站改版重新调参。

{metrics}

## 无监督结论需要怎样解释

舞弊探索用于描述财务形态，而不是用簇编号认定舞弊。破产原RobustScaler空间的平方距离约99.4%由单列贡献；分位数标准化后，2组与原分组ARI约-0.007，说明原分组缺少尺度稳健性。两种结果均保留，不能只凭高轮廓系数宣称发现稳定的综合财务类型。

## 来源与时点限制

课程破产表已与UCI官方逐单元核对一致，异常量纲并非读取错误，但单位机制仍待说明。财务原表16条资产年度锚点抽查支持前一年特征对齐；收入分母、accruals定义以及历史更正版本仍有疑问，未按猜测改数。因此历史时间前推检验条件是课程提供的财务版本，不能声称已经证明真实事前预警能力。

本目录附原有监督与无监督报告、逐条错误记录，以及第二轮来源和标准化核查。早期记录反映其生成时点，当前解释以本页和新增核查为准。
"""
    (a/'流程记录.md').write_text(report,encoding='utf-8')
    for task in ['舞弊预测','破产预测']:
        src=ROOT/f'成果交付/02_全流程分析/{task}'
        for p in src.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.md','.json','.csv','.png'}:copy(p,a/task/p.relative_to(src))
    for relative in ['01_来源与口径核查','03_破产聚类标准化对照']:
        for p in (OUT/relative).glob('*'):
            if p.is_file() and p.suffix.lower() in {'.md','.json','.csv','.png'}:copy(p,a/relative/p.name)
    dest=FINAL/'b_2025风险预测';dest.mkdir(exist_ok=True)
    for name in ['2025年报_风险预测表_v1.csv','应用模型摘要_v1.json','应用模型训练集中位数_v1.csv','全量应用树规则_v1.txt','应用树前四层_v1.png']:
        copy(ROOT/'成果交付/03_2025风险预测'/name,dest/name)
    model_note=f"""# 2025年报风险预测：模型与信心说明

使用全量42,511条历史面板，36个特征；决策树深度5、叶节点至少100条、balanced权重、种子42。缺失值用历史训练中位数填补，得到31个叶节点，为5,574家公司生成相对排序和实际路径。文件内financial_report_year=2025，prediction_target_year=2026，保留课程年度定义。

任务a的XGBoost测试成绩不能当成这棵应用树的成绩。这棵树在原训练期做五折时间前推，平均ROC-AUC=0.621；固定课堂提示规则会覆盖约65.8%的历史样本，精确率约19.8%。它提供线索，不能解释为少量精准警报。应用集无标签，未知其真实准确率。

课堂可信度分=100×(1−加权正类分数)，不是真实造假概率；叶节点未加权正类比例仅描述历史同路径样本。财务字段的收入分母、应计项和历史更正时点仍待核实，数值按课堂保留。第二轮回测不使用已经终评的2020—2021留出样本。
"""
    (dest/'模型说明.md').write_text(model_note,encoding='utf-8')
    for p in (OUT/'02_网站树历史回测').rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.md','.json','.csv','.png'}:copy(p,dest/'历史回测'/p.relative_to(OUT/'02_网站树历史回测'))
    c=FINAL/'c_文化与主题';c.mkdir(exist_ok=True)
    for name in ['公司年度文化指标_v2.csv','五维文化词典_审阅后_v2.csv','文化指标方法说明_v2.md','文化词典_建议剔除清单_v1.csv']:
        copy(ROOT/'成果交付/04_文本分析'/name,c/name)
    for relative in ['06_全文主题对照','07_文化指标解释']:
        for p in (OUT/relative).rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.md','.json','.csv','.png'}:copy(p,c/relative/p.relative_to(OUT/relative))
    # 当前正式说明反映用户已经确认的展示选择，历史快照本身不改。
    final_text=(c/'06_全文主题对照/第二阶段_全文主题对照报告.md')
    text=final_text.read_text(encoding='utf-8').replace('本次作为新增对照交付，不直接覆盖旧网站主结果。是否把全文方案作为最终展示版本，应结合主题原文审阅及本报告的稳定性结果决定；','用户已确认全文方案作为网站主展示，原首、中、尾方案保留逐年对照；')
    final_text.write_text(text,encoding='utf-8')
    main_copy=DOWNLOADS/'全文主题报告.md';main_copy.write_text(public_text(text),encoding='utf-8')
    copy(ROOT/'成果交付/03_2025风险预测/2025年报_风险预测表_v1.csv',DOWNLOADS/'2025风险预测表.csv')
    copy(ROOT/'成果交付/04_文本分析/公司年度文化指标_v2.csv',DOWNLOADS/'五维文化指标.csv')
    copy(OUT/'06_全文主题对照/seed42/主题描述性命名.csv',DOWNLOADS/'全文主题表.csv')
    (DOWNLOADS/'模型说明.md').write_text(model_note,encoding='utf-8')
    script=f"""# 两分钟项目介绍稿

署名：{PRIVATE}。建议先试读，再按本人语速调整停顿；时间段为录制安排，不是已测得时长。

大家好，我是{PRIVATE}。我的项目叫“财务信号与文本画像”，想回答：财务指标能提供哪些风险线索，年报文字又能帮助我们理解什么？

我使用老师提供的舞弊、破产和管理层讨论与分析数据。先做聚类探索，再比较四类模型。舞弊按年份前推，破产分层留出；预处理只在训练部分拟合，测试集不参与调参。

两个任务最终都选出XGBoost，测试ROC-AUC分别约为零点七四和零点九四。不过舞弊任务误报较多，破产任务也有漏报，所以不能只展示一个好看的指标。

我另用全部历史数据训练一棵可解释的树，为五千五百七十四家公司生成风险排序。网站能查财务指标和判定路径。这个加权分数不是真实造假概率，而且应用集没有标签，不能声称知道其准确率。

文本部分有一千一百二十三篇。文化词典经确认删词后保留七百八十四词；主题采用全文分段，保留十二个描述性主题和离群文本，同时展示原方案对照。文化分测的是管理层怎么说自己，不代表实际行为。

AI协助写代码、运行和整理证据；我负责样本、权重、删词和展示方案的选择。核查还发现财务口径和历史更正版本的疑问，因此保留课堂数值、如实限制结论。这份作业的收获，是既能展示结果，也能说明为什么这样做。
"""
    (FINAL/f'两分钟介绍稿_{PRIVATE}.md').write_text(script,encoding='utf-8')
    material=FINAL/'录制素材';material.mkdir(exist_ok=True)
    for src,name in [(OUT/'06_全文主题对照/覆盖率与向量变化.png','01_文本覆盖与表示变化.png'),(OUT/'06_全文主题对照/主题归属交叉图.png','02_新旧主题对照.png'),(ROOT/'成果交付/03_2025风险预测/应用树前四层_v1.png','03_应用树.png')]:copy(src,material/name)
    guide=f"""# 真人录制步骤

署名：{PRIVATE}；公开项目使用别名{PUBLIC}。视频需由本人录制，本目录提供讲稿、图表和网站，不冒充已完成成片。

| 建议时间 | 讲什么 | 展示什么 |
|---|---|---|
| 0—15秒 | 姓名、项目问题 | 网站首页 |
| 15—40秒 | 数据、划分、两项模型成绩 | 网站测试结果区 |
| 40—70秒 | 应用树与分数边界 | 查询公司，展示财务指标和判定路径 |
| 70—100秒 | 文化指标、全文主题与旧版对照 | 查询有文本公司，点击两种主题方案，展开逐年对照 |
| 100—120秒 | 可信依据、局限、本人和AI分工 | 网站方法与边界区 |

网站：{CFG['expected_site_url']}。代码仓库：{CFG['repository_url']}。

先试读一次，以约两分钟为目标；可以出镜，也可以本人旁白配录屏。录制前关闭无关窗口和通知。若发到公开平台，使用别名{PUBLIC}，不要展示本地实名文件路径；仅向老师私下提交时可使用本地实名稿。录制完将视频放入本正式提交目录，与网站链接、仓库链接和一页说明一起提交。
"""
    (material/'录制步骤.md').write_text(guide,encoding='utf-8')
    (DOWNLOADS/'两分钟介绍稿_zs.md').write_text(public_text(script),encoding='utf-8')
    # 公开源代码使用可审阅文本，不包含环境、令牌、模型权重或庞大运行缓存。
    files=[CODE/n for n in ['README.md','project_config.json','topic_names_v1.json','requirements.txt','.gitignore']]
    files+=list((CODE/'scripts').glob('*.py'))+list((CODE/'scripts').glob('*.mjs'))+list((CODE/'web').glob('*'))
    files+=list(STUDY.glob('*.py'))+list(STUDY.glob('*.json'))+list(STUDY.glob('*.md'))+list(STUDY.glob('*.mjs'))+list((STUDY/'web_v2').glob('*'))
    for p in files:
        if p.name=='release_config.json':continue
        target=REPO/'过程代码'/p.relative_to(CODE);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(public_text(p.read_text(encoding='utf-8-sig')),encoding='utf-8')
    public_cfg={k:v for k,v in CFG.items() if k not in {'private_author','site_checkout'}}
    dump(REPO/'PUBLICATION.json',public_cfg)
    portable_config={**public_cfg,'site_checkout':'过程代码/发布项目/financial-signals-zs'}
    dump(REPO/'过程代码/research_v2/release_config.json',portable_config)
    (REPO/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n.env\n.env.*\n.openai/\n.sites/\n.input-cache/\n过程代码/模型/\n过程代码/research_v2/work/\n',encoding='utf-8')
    readme=f"""# 财务信号与文本画像

署名：**{PUBLIC}**。AI数据分析课程作业。

- 网站：[{CFG['expected_site_url']}]({CFG['expected_site_url']})
- 一页结果说明：[下载PDF]({CFG['expected_site_url']}/downloads/onepage_zs.pdf)
- a—c完整成果：[教师核查包]({CFG['expected_site_url']}/downloads/teacher_results.zip)
- 数据与方法：全部数值来自本项目实际运行；来源为课程资料。使用AI符合课程安排，人的选择与AI执行分开记录。

## 作业对应

| 任务 | 交付 |
|---|---|
| a 两套数据完整流程 | 舞弊/破产的探索、模型比较、留出测试和逐条错误记录；破产聚类增加尺度敏感性核查 |
| b 无标签预测 | 5,574家公司风险表、应用树和模型信心说明 |
| c 文本指标与主题 | 1,123篇、225家公司；784词文化指标；全文BERTopic主结果与首中尾对照 |
| d 个人网站 | 公司查询、实际判定路径、文化图、词云、两种主题方案和下载成果 |

## 复现

仓库的`过程代码/scripts/`保留v1分析，`过程代码/research_v2/`保存来源核查、应用树回测、全文对照和新版网站。`requirements.txt`在过程代码目录中，记录运行版本。

分析依赖课程原包。将老师的“AI数据分析课程资料 261031 完整版”目录放在仓库根目录，或在`过程代码/project_config.json`中调整source_root；所有输入只读。BGE模型使用课程附带bge-small-zh-v1.5，本仓库不重复上传模型权重。完整分析执行次序见过程代码README，第二轮见research_v2/README。首次训练与文本编码需要较长时间。

本仓库提供源代码，网站提供已经计算的可核查表、报告与图。不把没有打包的环境、缓存或原始财务大表说成已经随代码提供。已有终评锁定记录用于避免反复使用测试集；重做研究应另设输出目录并记录改变。

## 主要结果与边界

{metrics}

上述是任务a的XGBoost留出成绩。网站应用树另按课堂规则用全量历史和balanced权重训练，发展性五折平均AUC约0.621；不能借用XGBoost成绩证明网页分数可靠。加权分数只作相对排序，应用集无标签。

全文主题主种子42：12个合并后的非离群主题、110篇离群；不按种子结果择优。最终12来自保留的合并规则，并不是独立验证的最优主题数。文本分组含通用和混合业务，文化指标测表述而不是实际行为。

财务来源抽查仍有收入分母、accruals及历史更正版本问题，课程数值保留并限制解释。破产原分组对标准化敏感，不能将高轮廓系数等同于综合财务类型。报告中如实保留误报、漏报和未解决事项。

## 本人与AI

{PUBLIC}决定样本、权重、噪声词处理及展示方案；AI协助代码实现、计算、语境初审和证据整理。没有声称AI审阅等于独立人工标注。真人视频另由本人录制。

## 公开范围

用户于2026-10-31明确确认课程数据可以公开。公开材料统一使用别名{PUBLIC}。第三方模型及库沿用其原有许可证；课程原包和模型权重不作为本项目原创内容重新授权。
"""
    (REPO/'README.md').write_text(readme,encoding='utf-8')
    # 源码下载与GitHub内容一致，教师无需下载整个环境。
    source_paths=[p for p in REPO.rglob('*') if p.is_file()]
    for p in source_paths:
        assert PRIVATE not in p.read_text(encoding='utf-8') if PRIVATE!=PUBLIC else True
    archive_files(DOWNLOADS/'source_code.zip',[(p,p.relative_to(REPO)) for p in source_paths])
    archive_files(FINAL/'复现源码_zs.zip',[(p,p.relative_to(REPO)) for p in source_paths])
    teacher_files=[p for folder in [a,dest,c] for p in folder.rglob('*') if p.is_file()]
    # 核查包也使用公开别名。当前分析报告本来不含实名；复制前再核对文本。
    for p in teacher_files:
        if p.suffix.lower() in {'.md','.json','.csv','.txt'}:
            value=p.read_text(encoding='utf-8-sig')
            if PRIVATE in value:p.write_text(public_text(value),encoding='utf-8')
    archive_files(DOWNLOADS/'teacher_results.zip',[(p,p.relative_to(FINAL)) for p in teacher_files])
    copy(DOWNLOADS/'teacher_results.zip',FINAL/'教师核查包_zs.zip')
    base=(STUDY/'web_v2/index.html').read_text(encoding='utf-8')
    section=f"""<section id="submission" class="section-block"><div class="section-title"><div><p class="eyebrow">COURSE SUBMISSION / {PUBLIC}</p><h2>按作业要求查看成果</h2></div><span class="badge">署名 {PUBLIC}</span></div><div class="method-grid"><article><span>a</span><h3>两套数据完整流程</h3><p>探索、模型比较、测试和错误记录，附来源与尺度敏感性核查。</p><a href="downloads/teacher_results.zip" download>下载完整核查包 ↗</a></article><article><span>b</span><h3>2025 年风险预测</h3><p>5,574 家公司、实际判定路径；应用分数只作相对排序。</p><a href="downloads/2025风险预测表.csv" download>预测表 ↗</a> · <a href="downloads/模型说明.md" download>模型说明 ↗</a></article><article><span>c</span><h3>文化指标与全文主题</h3><p>784 词文化指标、全文主结果和原方案对照，保留解释边界。</p><a href="downloads/五维文化指标.csv" download>文化表 ↗</a> · <a href="downloads/全文主题报告.md" download>报告 ↗</a></article><article><span>d</span><h3>网站、代码与一页说明</h3><p>查询、证据、研究选择和复现代码放在一起。</p><a href="{CFG['repository_url']}" target="_blank" rel="noopener">GitHub 仓库 ↗</a><br><a href="downloads/onepage_zs.pdf" download>一页说明 ↗</a> · <a href="downloads/source_code.zip" download>源码包 ↗</a></article></div></section>"""
    base=base.replace('AI 数据分析课程项目 · 全文主题版',f'AI 数据分析课程项目 · {PUBLIC}').replace('href="#method">方法与边界</a>','href="#submission">提交成果</a>')
    base=base.replace('<section id="method"',section+'<section id="method"')
    base=base.replace('本地网站 v2：全文主题为主，原方案保留对照。',f'署名 {PUBLIC} · 全文主题为主，原方案保留对照。')
    base=base.replace('公开发布范围尚待确认。','课程数据公开已获确认。')
    base=base.replace('财务信号与文本画像｜全文主题版',f'财务信号与文本画像｜{PUBLIC}')
    (DIST/'index.html').write_text(base,encoding='utf-8')
    app=(STUDY/'web_v2/app.js').read_text(encoding='utf-8').replace('新版只在本地查询与审阅，全部资源来自本地，本页面不调用外部 API。','公开版使用别名 zs，数据与模型结果在页面内读取，不把查询内容发送到外部模型 API。')
    (DIST/'app.js').write_text(app,encoding='utf-8')
    # 在线版的流程源代码均可定位，公开材料不包含私人署名。
    for p in DIST.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.html','.js','.css','.md','.json','.csv','.txt'}:
            value=p.read_text(encoding='utf-8-sig')
            assert PRIVATE not in value if PRIVATE!=PUBLIC else True
    manifest={'time_beijing':now(),'public_author':PUBLIC,'repository_url':CFG['repository_url'],'teacher_files':len(teacher_files),'source_files':len(source_paths),'v1_preserved_files':assert_v1_unchanged(),'public_data_authorized':True}
    dump(FINAL/'正式成果准备核查.json',manifest);dump(STUDY/'work/formal_submission_manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
