"""生成本地实名与公开别名两种一页说明，数值取自锁定结果。"""
import re,shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from study_common import *

CFG=json.loads((STUDY/'release_config.json').read_text(encoding='utf-8'))
FINAL=ROOT/f"成果交付/08_正式提交_{CFG.get('private_author',CFG['public_author'])}"

def create(path,author):
    pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CJKBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
    c=canvas.Canvas(str(path),pagesize=A4);c.setTitle('财务信号与文本画像：一页结果说明');c.setAuthor(author)
    w,h=A4;x=39;y=h-45;usable=w-78
    c.setFillColor(HexColor('#9e285c'));c.setFont('CJKBold',19);c.drawString(x,y,'财务信号与文本画像')
    y-=25;c.setFont('CJK',9);c.setFillColor(HexColor('#62645f'));c.drawString(x,y,f'AI 数据分析课程作业 | {author} | 2026-10-31')
    y-=13;c.setStrokeColor(HexColor('#9e285c'));c.line(x,y,w-x,y);y-=22
    def para(title,text):
        nonlocal y
        c.setFont('CJKBold',11);c.setFillColor(HexColor('#9e285c'));c.drawString(x,y,title);y-=18
        c.setFont('CJK',10);c.setFillColor(HexColor('#25282a'))
        text=re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9])',r'\1 \2',text)
        text=re.sub(r'([A-Za-z0-9%])([\u4e00-\u9fff])',r'\1 \2',text)
        line=''
        for token in re.findall(r'[A-Za-z0-9,.%/=+()-]+|[\u4e00-\u9fff][，。；：！？、）]*|.',text):
            if pdfmetrics.stringWidth(line+token,'CJK',10)>usable:
                c.drawString(x,y,line);y-=16.5;line=token
            else:line+=token
        if line:c.drawString(x,y,line);y-=16.5
        y-=10
    f=json.loads((ROOT/'成果交付/02_全流程分析/舞弊预测/最终测试评估_v1.json').read_text(encoding='utf-8'))
    b=json.loads((ROOT/'成果交付/02_全流程分析/破产预测/最终测试评估_v1.json').read_text(encoding='utf-8'))
    para('问题与数据','用财务指标探索风险线索，用 MD&A 描述文化表述和语义主题。舞弊面板 42,511 条，破产 6,819 条；2025 年报应用集 5,574 家公司；文本为 225 家公司、1,123 篇，覆盖 2018—2022 年。')
    para('流程与留出评估','先做训练集无监督探索，再比较四类模型。舞弊按年份前推，破产分层划分；预处理在训练部分拟合，模型和阈值锁定后终评。聚类正文 2 组、附录 4 组；破产分组对标准化敏感，另保留分位数对照。')
    rows=[['测试结果','舞弊预测','破产预测'],['选定模型','XGBoost','XGBoost'],['ROC-AUC',f"{f['roc_auc']:.3f}",f"{b['roc_auc']:.3f}"],['AP / 正类基线',f"{f['pr_auc_ap']:.3f} / {f['positive_baseline']:.3f}",f"{b['pr_auc_ap']:.3f} / {b['positive_baseline']:.3f}"],['召回率',f"{f['recall']:.1%}",f"{b['recall']:.1%}"],['误报 / 漏报',f"{f['confusion_matrix_tn_fp_fn_tp'][1]} / {f['confusion_matrix_tn_fp_fn_tp'][2]}",f"{b['confusion_matrix_tn_fp_fn_tp'][1]} / {b['confusion_matrix_tn_fp_fn_tp'][2]}"]]
    for i,row in enumerate(rows):
        if i==0:c.setFillColor(HexColor('#eee8e1'));c.rect(x,y-6,usable,20,fill=1,stroke=0)
        c.setFillColor(HexColor('#25282a'));c.setFont('CJKBold' if i==0 else 'CJK',9.6)
        for px,value in zip([x+8,x+205,x+365],row):c.drawString(px,y,value)
        y-=23
    y-=10
    para('全量应用与网页解释','另用全部历史数据训练深度 5、31 叶的 balanced 决策树，生成 5,574 家公司的风险排序和路径。其训练期五折平均 AUC 为 0.621，不能借用上表 XGBoost 成绩；加权分数非真实造假概率。2025 为报表年，预测目标年为 2026。')
    para('文化指标与全文主题','五维词典经确认剔除 28 词后保留 784 词，按教程公式计算并独立复算。全文分为 41,099 段，覆盖全部词元；主种子 42 合并后为 12 个主题，110 篇离群（9.8%）。网站保留首中尾方案对照及混合业务边界。')
    para('可信依据与结论限制','保留数据指纹、固定划分、错误记录和可定位原文。财务收入分母、accruals 及历史更正版本仍待核实，保留课堂数值，不作真实事前预警的强表述。文化指标衡量自我披露，主题不等于官方行业分类；覆盖完整和离群减少不等于准确率提升。')
    para('人的判断与 AI 分工','本人决定样本、权重、噪声词和展示方案；AI 协助实现、计算、语境初审与证据整理。没有将 AI 初审写成独立人工标注，真人介绍视频由本人录制。')
    c.setFont('CJKBold',9);c.setFillColor(HexColor('#26695d'));c.drawString(x,y,'提交入口');y-=14
    for label,url in [('网站',CFG['expected_site_url']),('代码',CFG['repository_url'])]:
        c.setFont('CJK',7.3);c.drawString(x,y,label+'：'+url);c.linkURL(url,(x,y-2,w-x,y+9),relative=0);y-=13
    assert y>32,f'一页排版超出：{y}'
    c.setFont('CJK',7.4);c.setFillColor(HexColor('#78786f'));c.drawString(x,21,'课程研究演示；不构成违法认定、审计意见或投资建议。')
    c.showPage();c.save()
    return y

def main():
    FINAL.mkdir(parents=True,exist_ok=True)
    private=CFG.get('private_author',CFG['public_author']);paths=[(FINAL/f'一页结果说明_{private}.pdf',private),(FINAL/'一页结果说明_zs.pdf',CFG['public_author'])]
    positions=[]
    for p,name in paths:positions.append(create(p,name))
    shutil.copyfile(FINAL/'一页结果说明_zs.pdf',Path(CFG['site_checkout'])/'dist/downloads/onepage_zs.pdf')
    from pypdf import PdfReader
    for p,name in paths:
        r=PdfReader(p);assert len(r.pages)==1 and name in r.pages[0].extract_text()
        if name==CFG['public_author'] and private!=name:assert private not in r.pages[0].extract_text() and private not in str(r.metadata)
    dump(FINAL/'一页说明核查.json',{'time_beijing':now(),'pages_each':1,'public_author':CFG['public_author'],'private_and_public_copies_separate':True,'remaining_y_points':positions})
    print('两种署名的一页PDF已生成，均为1页。')

if __name__=='__main__':main()
