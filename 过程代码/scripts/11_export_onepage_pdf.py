"""用宿主提供的reportlab导出一页说明；不重新估计任何成绩。"""
import json
import re
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit
from common import OUTPUT


def main():
    out=OUTPUT/"06_提交材料/一页结果说明_v1.pdf"
    pdfmetrics.registerFont(TTFont("Chinese","C:/Windows/Fonts/msyh.ttc",subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ChineseBold","C:/Windows/Fonts/msyhbd.ttc",subfontIndex=0))
    c=canvas.Canvas(str(out),pagesize=A4);c.setTitle("财务信号与文本画像：一页结果说明")
    width,height=A4;x=42;y=height-48;usable=width-84
    c.setFillColor(HexColor("#9e285c"));c.setFont("ChineseBold",19);c.drawString(x,y,"财务信号与文本画像")
    y-=26;c.setFillColor(HexColor("#666660"));c.setFont("Chinese",9);c.drawString(x,y,"AI 数据分析课程作业 · 一页结果说明 · 2026-10-31")
    y-=15;c.setStrokeColor(HexColor("#9e285c"));c.line(x,y,width-x,y);y-=24
    def para(title,text):
        nonlocal y
        c.setFillColor(HexColor("#9e285c"));c.setFont("ChineseBold",10);c.drawString(x,y,title);y-=16
        c.setFillColor(HexColor("#25282a"));c.setFont("Chinese",9)
        # 按真实字体宽度断行，保留中文标点和完整数值。
        text=re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])",r"\1 \2",text)
        text=re.sub(r"([A-Za-z0-9%])([\u4e00-\u9fff])",r"\1 \2",text)
        line=""
        for ch in re.findall(r"[A-Za-z0-9,.%/=—-]+|.",text):
            if pdfmetrics.stringWidth(line+ch,"Chinese",9)>usable:
                c.drawString(x,y,line);y-=14;line=ch
            else:line+=ch
        if line:c.drawString(x,y,line);y-=14
        y-=10
    f=json.loads((OUTPUT/"02_全流程分析/舞弊预测/最终测试评估_v1.json").read_text(encoding="utf-8"));b=json.loads((OUTPUT/"02_全流程分析/破产预测/最终测试评估_v1.json").read_text(encoding="utf-8"))
    para("问题与数据","用财务指标探索舞弊、破产风险，结合MD&A衡量文化表述与语义主题。舞弊历史面板42,511条，破产样本6,819条；应用集5,574家公司；文本为225家公司、1,123篇，覆盖2018—2022年。原始资料只读，恢复文本逐篇校验指纹。")
    para("流程与评估","先做训练集无监督探索，再统一比较四类模型。聚类正文2组、附录4组；舞弊按年份前推，破产分层划分。预处理在训练折内拟合，冠军和阈值固定后各终评一次。")
    headers=["测试指标","舞弊预测","破产预测"]
    rows=[headers,["冠军","XGBoost","XGBoost"],["ROC-AUC",f"{f['roc_auc']:.3f}",f"{b['roc_auc']:.3f}"],["PR-AUC(AP) / 基线",f"{f['pr_auc_ap']:.3f} / {f['positive_baseline']:.3f}",f"{b['pr_auc_ap']:.3f} / {b['positive_baseline']:.3f}"],["召回率",f"{f['recall']:.1%}",f"{b['recall']:.1%}"],["误报 / 漏报",f"{f['confusion_matrix_tn_fp_fn_tp'][1]} / {f['confusion_matrix_tn_fp_fn_tp'][2]}",f"{b['confusion_matrix_tn_fp_fn_tp'][1]} / {b['confusion_matrix_tn_fp_fn_tp'][2]}"]]
    col=[x+10,x+205,x+350]
    for i,row in enumerate(rows):
        if i==0:c.setFillColor(HexColor("#eee8e1"));c.rect(x,y-6,usable,21,fill=1,stroke=0)
        c.setFillColor(HexColor("#25282a"));c.setFont("ChineseBold" if i==0 else "Chinese",9)
        for xx,txt in zip(col,row):c.drawString(xx,y,txt)
        y-=22
    y-=10
    para("外推与文本","全量历史应用树深度5、31叶，按教程使用balanced权重，为5,574家公司输出排序和实际路径。2025为报表年，Year=2026按课程规则解释为目标年。五维词典经确认剔除28词后保留784词，按教程公式重新计算。首中尾三段嵌入的BERTopic保留12个主题，306篇离群（27.2%）。")
    para("结论边界与人的作用","聚类描述形态；应用分数不是校准的造假概率；应用集无标签，未知其真实准确率。MD&A衡量自我表述，无外部效度或因果验证。用户决定样本、权重、簇数、公式修正、文本取段和词典剔除；AI执行并保留代码、指纹、比较与错误记录。")
    para("交付状态","分析报告、数据表、本地网站和源码包已准备。在线网址与仓库网址待确认发布账户及数据许可；两分钟介绍稿已备，视频待本人录制。网页浏览器检查被宿主工具中止，文件与数据核查另有记录。")
    assert y>35,f"一页排版溢出：y={y}"
    c.setFillColor(HexColor("#777770"));c.setFont("Chinese",8);c.drawString(x,24,"课程研究演示，不构成违法认定、审计意见或投资建议。")
    c.showPage();c.save();print(out)


if __name__=="__main__":main()
