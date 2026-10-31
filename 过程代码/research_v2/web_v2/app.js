(() => {
  'use strict';
  const D = window.PROJECT_DATA;
  const $ = selector => document.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct = (v, n = 1) => v == null ? '未提供' : (v * 100).toFixed(n) + '%';
  const num = (v, n = 2) => v == null ? '未提供' : Number(v).toLocaleString('zh-CN', {maximumFractionDigits:n});
  const dims = ['创新','诚信','质量','尊重','团队'];
  const names = {
    size_assets:'总资产', size_revenue:'营业收入', lev:'资产负债率', current_ratio:'流动比率',
    roa:'总资产收益率 ROA', roe:'净资产收益率 ROE', net_margin:'净利率', ar_ratio:'应收账款比率',
    asset_turnover:'总资产周转率', cash_share:'现金占比', rev_growth:'营收增长率', admin_ratio:'管理费用率',
    finexp_ratio:'财务费用率', other_ar_ratio:'其他应收比率', fixed_ratio:'固定资产占比', asset_growth:'资产增长率',
    cost_ratio:'营业成本率', fin_exp_ratio:'财务费用率', fixed_asset_share:'固定资产占比', st_debt_ratio:'短期借款占资产比'
  };
  function value(k, v) {
    if (v == null) return '未提供';
    if (k.startsWith('size_')) return num(Math.exp(v) / 1e8, 1) + ' 亿元';
    if (['current_ratio','quick_ratio','asset_turnover'].includes(k)) return num(v, 3) + ' 倍';
    return pct(v, 2);
  }
  const companies = [...D.companies,...D.textOnlyCompanies];
  let activeCode = null;
  let topicView = 'full';
  const topicName = (topic, view = 'full') => (view === 'full' ? D.topicNames : D.oldTopicNames)[String(topic)] || '主题 ' + topic;

  $('#overview-stats').innerHTML = [['42,511','历史公司年度 / 应用树训练'],['5,574','2025 年报公司 / 新样本预测'],['1,123','MD&A 原文 / 225 家公司'],[D.topicCount,'全文语义主题 / 另列离群']]
    .map(([v,t]) => `<div class="stat"><strong>${v}</strong><span>${t}</span></div>`).join('');
  $('#evaluation').innerHTML = Object.entries(D.evaluations).map(([task,m]) => `<article class="eval-card"><h3>${esc(task)} <small>· ${esc(m.model)}</small></h3><div class="eval-nums"><div><b>${m.roc_auc.toFixed(3)}</b><span>ROC-AUC</span></div><div><b>${m.pr_auc_ap.toFixed(3)}</b><span>PR-AUC · 基线 ${pct(m.positive_baseline)}</span></div><div><b>${pct(m.recall)}</b><span>召回率</span></div></div><p class="error-note">测试 ${num(m.rows,0)} 条 · 漏报 ${m.confusion_matrix_tn_fp_fn_tp[2]} 条 · 误报 ${num(m.confusion_matrix_tn_fp_fn_tp[1],0)} 条</p></article>`).join('');
  const backtest = D.applicationBacktest;
  $('#application-note').innerHTML = `<strong>公司查询树单独核验</strong><br>在原训练期做五折时间前推，平均 ROC-AUC 为 ${backtest.meanFoldAuc.toFixed(3)}。沿用固定提示规则时，会标记约 ${pct(backtest.flaggedShare)} 的历史样本，覆盖面较宽。这是发展性历史回测，不能当成 2025 年预测的已知表现。`;
  const comparison = D.topicComparison;
  $('#text-comparison').innerHTML = `<div class="coverage-cards"><article class="coverage-card primary"><h3>全文分段 · 主结果</h3><div class="coverage-big">100% <small>词元覆盖</small></div><p>1,123 篇逐段编码后汇总 · ${comparison.fullOutliers} 篇离群</p><p>预设三组随机种子的离群比例为 ${pct(comparison.seedOutlierMin)}—${pct(comparison.seedOutlierMax)}，主结果固定采用种子 42。</p></article><article class="coverage-card secondary"><h3>首、中、尾 · 原方案对照</h3><div class="coverage-big">${pct(comparison.oldCoverageMedian,2)} <small>覆盖率中位数</small></div><p>每篇最多三段 · ${comparison.oldOutliers} 篇离群</p><p>新旧分组有较大变化。查询公司后可切换主题篇数，并逐年比较归属。</p></article></div><p class="note">主结果的通用主题仍有 ${comparison.genericTopicDocuments} 篇，约占 ${pct(comparison.genericTopicDocuments/comparison.documents)}；部分小组混合不同业务。三次最终均为 12 个主题来自既定合并规则，不表示 12 是独立验证的最佳数目。</p>`;

  $('#company-options').innerHTML = companies.map(c => `<option value="${esc(c.code+' '+c.company)}"></option>`).join('');
  $('#examples').innerHTML = ['600519','600518',...D.textExamples.slice(0,2)].filter((v,i,a) => a.indexOf(v) === i)
    .map(code => companies.find(c => c.code === code)).filter(Boolean)
    .map(c => `<button type="button" data-code="${c.code}">${esc(c.company)} · ${c.code}</button>`).join('');
  $('#examples').addEventListener('click', e => {if (e.target.dataset.code) show(e.target.dataset.code);});

  function radar(row) {
    const mean = D.cultureMeanByYear[String(row.year)] || D.cultureMean;
    const cx=180, cy=145, r=100;
    const max=Math.max(...dims.map(k => Math.max(row[k],mean[k]))) * 1.15 || 1;
    const pos=(i,v) => [cx+Math.sin(i*2*Math.PI/5)*r*v,cy-Math.cos(i*2*Math.PI/5)*r*v];
    const polygon=data => dims.map((k,i) => pos(i,data[k]/max).join(',')).join(' ');
    return `<svg class="radar" viewBox="0 0 360 300" role="img" aria-label="文化五维表述强度雷达图">${[.25,.5,.75,1].map(s => `<polygon points="${dims.map((k,i)=>pos(i,s).join(',')).join(' ')}" fill="none" stroke="#dedbd3"/>`).join('')}${dims.map((k,i) => {const p=pos(i,1.23),q=pos(i,1);return `<line x1="${cx}" y1="${cy}" x2="${q[0]}" y2="${q[1]}" stroke="#dedbd3"/><text x="${p[0]}" y="${p[1]}" text-anchor="middle">${k}</text>`;}).join('')}<polygon points="${polygon(mean)}" fill="#26695d15" stroke="#26695d" stroke-dasharray="4 3"/><polygon points="${polygon(row)}" fill="#9e285c26" stroke="#9e285c" stroke-width="2"/></svg>`;
  }

  function topicPanel(text) {
    const data = topicView === 'full' ? text.topics : text.oldTopics;
    const rows = data.map(([topic,n]) => `<div class="theme-row"><span>${esc(topicName(topic,topicView))}</span><div class="bar-track"><div class="bar" style="width:${100*n/text.documentCount}%"></div></div><span>${n} 篇</span></div>`).join('');
    const years = text.topicDocuments.map(d => `<tr><td>${d.year}</td><td>${esc(topicName(d.fullTopic))}</td><td>${esc(topicName(d.oldTopic,'old'))}</td></tr>`).join('');
    return `<div class="text-card ${topicView==='old'?'topic-view-old':''}"><h4>历年文本主题归属</h4><div class="topic-tabs" role="group" aria-label="主题表示方法"><button type="button" data-topic-view="full" aria-pressed="${topicView==='full'}">全文分段 · 主结果</button><button type="button" data-topic-view="old" aria-pressed="${topicView==='old'}">首中尾 · 对照</button></div><span class="topic-mode">当前：${topicView==='full'?'全文分段主结果':'首、中、尾三段对照'}</span><p>样本期共 ${text.documentCount} 篇；每篇归入一个主题。离群单列，条形长度表示篇数占比，不是主题概率。</p>${rows}<p class="topic-note">名称概括主要文本表述；通用组、混合业务和同公司模板都会影响分组，不能据此认定公司的全部业务。</p><details class="topic-compare-details"><summary>逐年比较全文与原方案</summary><table class="topic-yearly"><thead><tr><th scope="col">年份</th><th scope="col">全文主结果</th><th scope="col">首中尾对照</th></tr></thead><tbody>${years}</tbody></table></details></div>`;
  }

  function show(code, preserveView = false) {
    const c = companies.find(x => x.code === code);
    if (!c) return;
    if (!preserveView || activeCode !== code) topicView = 'full';
    activeCode = code;
    $('#company-search').value = c.code+' '+c.company;
    $('#search-status').textContent = '';
    const leaf = c.textOnly ? null : D.rules[String(c.leaf)];
    const text = D.text[c.code];
    let html = c.textOnly ? `<div class="company-head"><div><h3>${esc(c.company)}</h3><small>${c.code} · 本次仅有样本期文本</small></div></div><div class="missing-text">该公司不在本次 2025 年财务应用表中，暂无该年财务数据、评分和判定路径。下方保留已有 MD&A 文本画像。</div>` : `<div class="company-head"><div><h3>${esc(c.company)}</h3><small>${c.code} · 2025 年报 · 缺失后填补 ${c.missing_features_before_imputation} / 36 个模型特征</small></div><div class="trust"><strong>${c.trust_score.toFixed(1)}</strong><span>课堂可信度分 · ${esc(c.level)}<br>加权分数的反向转换，非真实概率</span></div></div><p class="source-note">财务数值沿用课程数据；部分收入分母、应计项和历史更正版本仍待核实。评分只提供课堂研究线索。</p><div class="metrics">${Object.entries(c.metrics).map(([k,v]) => `<div class="metric"><small>${esc(names[k]||k)}</small><b>${value(k,v)}</b></div>`).join('')}</div><div class="path-panel"><h4>模型为什么这样判断</h4><div class="steps">${leaf.conditions.map(x => `<span class="step">${esc(names[x.feature]||x.feature)} ${esc(x.operator)} ${value(x.feature,x.threshold)}</span>`).join('<span aria-hidden="true">且</span>')}</div><p>该叶节点包含 ${num(leaf.training_rows,0)} 条历史训练样本，未加权正类占比 ${pct(leaf.training_positive_rate)}。这描述历史同路径样本，不等于本公司未来的真实概率。</p></div>`;
    if (leaf) html += `<div class="text-card"><h4>路径特征在历史样本中的位置</h4><p>横轴为历史训练特征的 1%—99% 分位区间；色条为正常/正类中位数的位置范围；实线是本公司值，虚线是当前路径阈值。展示未填补的原始值。</p>${leaf.conditions.map(x => {
      const s=D.featureDistributions[x.feature], v=c.metrics[x.feature]??c.path_values?.[x.feature];
      if (!s) return '';
      const position=n => Math.max(0,Math.min(100,100*(n-s.q01)/(s.q99-s.q01||1)));
      const a=position(s.negative_median),b=position(s.positive_median);
      return `<div class="dist-row">${esc(names[x.feature]||x.feature)}：正常中位数 ${value(x.feature,s.negative_median)}，正类 ${value(x.feature,s.positive_median)}${v==null?'；公司值缺失，模型用历史中位数填补':''}<div class="dist-track"><div class="fill" style="margin-left:${Math.min(a,b)}%;width:${Math.abs(b-a)}%"></div>${v==null?'':`<span class="dist-marker" style="left:${position(v)}%"></span>`}<span class="dist-threshold" style="left:${position(x.threshold)}%"></span></div></div>`;
    }).join('')}</div>`;
    if (text) {
      html += `<div class="section-title" style="margin-top:35px"><h2>MD&A 文本画像</h2><span class="badge">${text.firstYear}—${text.latest.year} 年文本 · 与财务年份不同</span></div><div class="text-grid"><div class="text-card"><h4>${text.latest.year} 年文化表述强度</h4>${radar(text.latest)}<div class="legend"><i></i>该公司 <i class="average"></i>${text.latest.year} 年语料均值</div><p>${dims.map(k=>`${k} ${num(text.latest[k],6)}`).join(' · ')}</p><p>数值为加权词频除以过滤后词数，不是百分制，也不代表实际文化好坏。</p></div>${topicPanel(text)}<div class="text-card" style="grid-column:1/-1"><h4>管理层在反复谈论什么</h4><p>合并样本期 MD&A 的高频词。词云反映频次，不验证陈述真假。</p><img class="cloud" loading="lazy" src="assets/wordcloud/${c.code}.png" alt="${esc(c.company)}的MD&A高频词云"></div></div>`;
    } else {
      html += '<div class="missing-text" style="margin-top:25px">该公司不在本次 225 家 MD&A 语料范围内，暂无文本画像。上方示例包含有文本的公司。</div>';
    }
    $('#company-card').innerHTML = html;
  }
  $('#company-card').addEventListener('click', e => {
    const button = e.target.closest('[data-topic-view]');
    if (!button || !['full','old'].includes(button.dataset.topicView)) return;
    topicView = button.dataset.topicView;
    show(activeCode, true);
  });
  function search() {
    const q = $('#company-search').value.trim();
    if (!q) {$('#search-status').textContent='请输入股票代码或公司简称。';return;}
    const c = companies.find(c=>c.code===q||c.code+' '+c.company===q||c.company===q) || companies.find(c=>c.code.includes(q)||c.company.includes(q));
    if (!c) {$('#search-status').textContent='没有匹配到公司，请检查股票代码或简称。';return;}
    show(c.code);
  }
  $('#search-button').addEventListener('click', search);
  $('#company-search').addEventListener('keydown', e => {if (e.key==='Enter') search();});
  $('#method-details').innerHTML = `<p>文化指标：Σ[词频 × log(N/df) ÷ log(1+rank)] ÷ 文档过滤后词数。原文 1,123 篇，词典初稿 812 词，确认剔除 28 词后保留 784 词。已有得分完成独立复算，另提供 25 条目的性选取的语境例证；它们不是总体准确率的随机样本。</p><p>全文 BERTopic：本地 BGE，按顺序切成不超过 510 个正文词元的等长块，逐块归一化后平均再归一化；UMAP、HDBSCAN 和中文 c-TF-IDF 沿用既有设置。主结果离群率 ${pct(D.outlierRate)}；种子 43/44 只检验敏感性，不据此挑选更好看的结果。</p><p>应用树与任务 a 的模型和权重口径分开。无监督正文保留两组、附录保留四组；破产聚类新增标准化敏感性对照，旧分组缺少尺度稳健性。</p><p>原始资料、v1 代码和旧网站保留。新版只在本地查询与审阅，全部资源来自本地，本页面不调用外部 API。</p>`;
  show(D.textExamples[0] || '600519');
})();
