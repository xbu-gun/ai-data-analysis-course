"""从锁定成果建立公开别名视频分镜；不重算、不改变分析结果。"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / '成果交付/09_AI视频_zs'

SCENES = [
    dict(id='opening', title='财务信号与文本画像', eyebrow='AI 数据分析课程作业', cues=[
        ('这份课程作业叫财务信号与文本画像。', '这份课程作业叫“财务信号与文本画像”。'),
        ('它想回答：财务指标提供什么风险线索，年报文字又能帮助我们理解什么？', '财务指标提供什么风险线索？年报文字又能帮助我们理解什么？'),
    ]),
    dict(id='data', title='从数据开始，给测试留出位置', eyebrow='01 / 数据与流程', cues=[
        ('数据来自老师课程包，包括舞弊、破产数据，以及一千一百二十三篇管理层讨论与分析。', '数据来自课程包：舞弊、破产，以及 1,123 篇管理层讨论与分析。'),
        ('先做聚类探索，再比较四类模型。舞弊按年份前推，破产分层留出，测试集不参与调参。', '先聚类，再比较四类模型。舞弊按年份前推，破产分层留出；测试集不参与调参。'),
    ]),
    dict(id='models', title='成绩与错误，一起展示', eyebrow='02 / 独立留出测试', cues=[
        ('两项任务都选出梯度提升树。测试曲线下面积，舞弊约零点七四，破产约零点九四。', '两项任务均选出 XGBoost。测试 ROC-AUC：舞弊约 0.74，破产约 0.94。'),
        ('但舞弊任务误报三千四百一十四条，破产任务漏报十四条，因此成绩必须和错误一起看。', '舞弊误报 3,414 条，破产漏报 14 条。成绩必须和错误一起看。'),
    ]),
    dict(id='website', title='把结果放进可查询的网站', eyebrow='03 / 全量应用与解释', cues=[
        ('网站使用另一棵可解释决策树，为五千五百七十四家公司生成风险排序，能查指标和判定路径。', '网站另用可解释决策树，为 5,574 家公司生成风险排序，提供指标和判定路径。'),
        ('加权分数不是真实造假概率；应用集没有标签，也不能声称知道准确率。', '加权分数不是真实造假概率；应用集无标签，无法报告实际准确率。'),
    ]),
    dict(id='text', title='从年报文字到文化与主题', eyebrow='04 / 文本分析', cues=[
        ('文本部分计算五维文化指标，词典保留七百八十四个词。', '文本部分计算五维文化指标，词典保留 784 个词。'),
        ('主题分析覆盖全文，得到十二个描述性主题，另有一百一十篇离群文档，并保留原方案对照。', '全文分段后汇总：12 个描述性主题，110 篇离群文档，并保留原方案对照。'),
        ('文化分衡量自我披露，主题也不等于行业分类。', '文化分衡量自我披露；主题不等于行业分类。'),
    ]),
    dict(id='evidence', title='每个结论，都要找得到依据', eyebrow='05 / 可信依据与局限', cues=[
        ('可信依据包括固定划分、错误明细和可定位的原文。', '保留固定划分、错误明细和可定位的原文。'),
        ('财务口径与历史更正版本仍有疑问，所以保留课堂数值，明确限制结论。', '财务口径与历史更正版本仍待核实：保留课堂数值，明确限制结论。'),
    ]),
    dict(id='closing', title='可以展示，也经得起追问', eyebrow='06 / 人的判断与完整交付', cues=[
        ('样本、权重和删词等选择由项目作者确认，人工智能协助编程、计算与核查。', '作者确认样本、权重和删词等选择；AI 协助编程、计算与核查。'),
        ('完整成果包括网站、代码仓库和一页说明。欢迎查看证据，再判断结论。', '完整成果：网站、代码仓库、一页说明。欢迎查看证据，再判断结论。'),
    ]),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / 'work').mkdir(exist_ok=True)
    fraud_path = ROOT / '成果交付/02_全流程分析/舞弊预测/最终测试评估_v1.json'
    bank_path = ROOT / '成果交付/02_全流程分析/破产预测/最终测试评估_v1.json'
    fraud = json.loads(fraud_path.read_text(encoding='utf-8'))
    bank = json.loads(bank_path.read_text(encoding='utf-8'))
    text_path = ROOT / '成果交付/07_优化研究_v2/06_全文主题对照/全文主题对照摘要.json'
    topics = json.loads(text_path.read_text(encoding='utf-8'))
    primary = next(x for x in topics['seed_runs'] if x['seed'] == topics['primary_seed'])
    assert primary['final_topics'] == 12 and primary['outlier_documents'] == 110
    assert fraud['confusion_matrix_tn_fp_fn_tp'][1] == 3414
    assert bank['confusion_matrix_tn_fp_fn_tp'][2] == 14
    assert not fraud['test_used_for_selection'] and not bank['test_used_for_selection']
    app_path = ROOT / '成果交付/03_无标签预测/应用模型摘要_v1.json'
    # 应用文件位置按实际交付目录发现，避免猜测读取不存在的文件。
    if not app_path.exists():
        paths = list((ROOT / '成果交付').glob('03*/*摘要*.json'))
        assert len(paths) == 1, paths
        app_path = paths[0]
    app = json.loads(app_path.read_text(encoding='utf-8'))
    index = 0
    for scene in SCENES:
        scene['cues'] = [dict(id=f'{index+j:02d}', speech=s, subtitle=t)
                         for j, (s, t) in enumerate(scene['cues'])]
        index += len(scene['cues'])
    data = dict(author='zs', voice='Microsoft Kangkang', voice_rate=0,
                disclosure='AI 合成旁白与动态图表；未模拟作者本人声线或形象。',
                date_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                target_seconds=120, resolution=[1920,1080], fps=25,
                scenes=SCENES,
                facts=dict(fraud=fraud, bankruptcy=bank, topics=primary, application=app),
                references=[str(p.relative_to(ROOT)).replace('\\','/') for p in [fraud_path,bank_path,text_path,app_path]],
                website='https://financial-signals-zs.satiny-grebe-2478.chatgpt.site',
                github='https://github.com/xbu-gun/ai-data-analysis-course')
    (HERE / 'storyboard.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    text = '# AI 视频分镜与旁白（公开署名 zs）\n\n'
    text += data['disclosure'] + '\n\n'
    for scene in SCENES:
        text += f"## {scene['eyebrow']}｜{scene['title']}\n\n"
        text += '\n\n'.join(c['speech'] for c in scene['cues']) + '\n\n'
    (OUT / '分镜与旁白_zs.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(scenes=len(SCENES), cues=index,
                         speech_characters=sum(len(c['speech']) for s in SCENES for c in s['cues'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
