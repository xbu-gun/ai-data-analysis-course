import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const site=process.argv[2];
const oldSite=process.argv[3];
if(!site||!oldSite) throw new Error('需要新版和旧版网站目录');
const nodes=new Map();
function node(selector){
  if(!nodes.has(selector)) nodes.set(selector,{innerHTML:'',textContent:'',value:'',listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}});
  return nodes.get(selector);
}
const sandbox={window:{},document:{querySelector:node},console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(site,'assets/data.js'),'utf8'),sandbox);
vm.runInContext(fs.readFileSync(path.join(site,'app.js'),'utf8'),sandbox);
const data=JSON.parse(JSON.stringify(sandbox.window.PROJECT_DATA));
const original=JSON.parse(fs.readFileSync(path.join(oldSite,'assets/data.js'),'utf8').replace(/^window\.PROJECT_DATA=/,'').replace(/;$/,''));
assert.equal(data.companies.length,5574);
assert.equal(Object.keys(data.text).length,225);
for(const field of ['companies','rules','cultureMean','cultureMeanByYear','featureDistributions','evaluations']) assert.deepEqual(data[field],original[field]);
for(const [code,t] of Object.entries(data.text)){
  assert.deepEqual(t.latest,original.text[code].latest);
  assert.deepEqual(t.oldTopics,original.text[code].topics);
  assert.equal(t.topics.reduce((n,row)=>n+row[1],0),t.documentCount);
  assert.equal(t.oldTopics.reduce((n,row)=>n+row[1],0),t.documentCount);
  assert.equal(t.topicDocuments.length,t.documentCount);
  for(const [topic] of t.topics) assert(data.topicNames[topic]);
  for(const [topic] of t.oldTopics) assert(data.oldTopicNames[topic]);
}
function search(q){node('#company-search').value=q;node('#search-button').listeners.click();return node('#company-card').innerHTML;}
function toggle(mode){const button={dataset:{topicView:mode},closest(){return this;}};node('#company-card').listeners.click({target:button});return node('#company-card').innerHTML;}
let rendered=0,paired=0;
assert(node('#company-card').innerHTML.includes('MD&A 文本画像'));
for(const company of data.companies){
  let html=search(company.code);
  assert(html.includes('模型为什么这样判断'),company.code);
  assert(!html.includes('NaN')&&!html.includes('undefined'),company.code);
  if(data.text[company.code]){
    assert(html.includes('当前：全文分段主结果'));
    assert(html.includes('逐年比较全文与原方案'));
    html=toggle('old');assert(html.includes('当前：首、中、尾三段对照'));
    assert(html.includes('data-topic-view="old" aria-pressed="true"'));
    html=toggle('full');assert(html.includes('当前：全文分段主结果'));
    paired++;
  }else assert(html.includes('暂无文本画像'));
  rendered++;
}
assert.equal(data.textOnlyCompanies.length,15);
for(const company of data.textOnlyCompanies){
  let html=search(company.code);assert(html.includes('暂无该年财务数据'));
  assert(html.includes('MD&A 文本画像')&&!html.includes('课堂可信度分 ·'));
  assert(!html.includes('NaN')&&!html.includes('undefined'));
  html=toggle('old');assert(html.includes('当前：首、中、尾三段对照'));
  html=toggle('full');assert(html.includes('当前：全文分段主结果'));paired++;
}
assert(search('茅台').includes('贵州茅台'));
assert(search('600519').includes('亿元'));
search('不存在的公司_test_919');assert(node('#search-status').textContent.includes('没有匹配到公司'));
search('');assert(node('#search-status').textContent.includes('请输入'));
node('#company-search').value='600519';node('#company-search').listeners.keydown({key:'Enter'});
assert(node('#company-card').innerHTML.includes('贵州茅台'));
const first=Object.keys(data.text)[0];search(first);toggle('old');search(first);assert(node('#company-card').innerHTML.includes('当前：全文分段主结果'));
assert(node('#text-comparison').innerHTML.includes('110 篇离群'));
const summary={time_beijing:new Date(Date.now()+8*3600000).toISOString().replace('Z','+08:00'),status:'passed',
  companies_rendered:rendered,paired_company_topic_toggles_verified:paired,financial_payload_and_scores_unchanged:true,
  text_only_companies_verified:15,
  culture_scores_unchanged:true,all_topic_counts_and_names_valid:true,invalid_empty_and_enter_search:true,
  default_resets_to_full_text:true,scope:'纯JavaScript逻辑核查，不替代真实浏览器视觉与交互检查'};
fs.writeFileSync(path.join(site,'网站交互逻辑核查_v2.json'),JSON.stringify(summary,null,2),'utf8');
console.log(JSON.stringify(summary,null,2));
