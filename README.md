# 财务信号与文本画像

署名：**zs**。AI数据分析课程作业。

- 网站：[https://financial-signals-zs.xbu.chatgpt.site](https://financial-signals-zs.xbu.chatgpt.site)
- 一页结果说明：[下载PDF](https://financial-signals-zs.xbu.chatgpt.site/downloads/onepage_zs.pdf)
- a—c完整成果：[教师核查包](https://financial-signals-zs.xbu.chatgpt.site/downloads/teacher_results.zip)
- 数据与方法：全部数值来自本项目实际运行；来源为课程资料。使用AI符合课程安排，人的选择与AI执行分开记录。


- 两分钟视频：[播放 / 下载 MP4](https://financial-signals-zs.xbu.chatgpt.site/downloads/project_video_zs.mp4)
- 综合报告：[下载 PDF](https://financial-signals-zs.xbu.chatgpt.site/downloads/analysis_report_zs.pdf)
- 最终提交材料：[精简提交包](https://financial-signals-zs.xbu.chatgpt.site/downloads/final_submission_zs.zip)

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

| 测试指标 | 舞弊 | 破产 |
|---|---:|---:|
| ROC-AUC | 0.744 | 0.940 |
| AP / 正类基线 | 0.412 / 0.184 | 0.496 / 0.032 |
| 召回率 | 80.4% | 57.6% |
| 误报 / 漏报 | 3414 / 313 | 39 / 14 |

上述是任务a的XGBoost留出成绩。网站应用树另按课堂规则用全量历史和balanced权重训练，发展性五折平均AUC约0.621；不能借用XGBoost成绩证明网页分数可靠。加权分数只作相对排序，应用集无标签。

全文主题主种子42：12个合并后的非离群主题、110篇离群；不按种子结果择优。最终12来自保留的合并规则，并不是独立验证的最优主题数。文本分组含通用和混合业务，文化指标测表述而不是实际行为。

财务来源抽查仍有收入分母、accruals及历史更正版本问题，课程数值保留并限制解释。破产原分组对标准化敏感，不能将高轮廓系数等同于综合财务类型。报告中如实保留误报、漏报和未解决事项。

## 本人与AI

zs决定样本、权重、噪声词处理及展示方案；AI协助代码实现、计算、语境初审和证据整理。没有声称AI审阅等于独立人工标注。两分钟介绍视频已完成，采用 AI 合成男声与动态图表，片头披露；未模拟本人声音或形象。

## 公开范围

用户于2026-10-31明确确认课程数据可以公开。公开材料统一使用别名zs。第三方模型及库沿用其原有许可证；课程原包和模型权重不作为本项目原创内容重新授权。
