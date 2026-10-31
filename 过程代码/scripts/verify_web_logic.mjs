import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
const site=process.argv[2];
const nodes=new Map();
function node(selector){if(!nodes.has(selector))nodes.set(selector,{innerHTML:'',textContent:'',value:'',listeners:{},addEventListener(type,fn){this.listeners[type]=fn}});return nodes.get(selector)}
const sandbox={window:{},document:{querySelector:node},console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(site,'assets/data.js'),'utf8'),sandbox);
vm.runInContext(fs.readFileSync(path.join(site,'app.js'),'utf8'),sandbox);
const data=sandbox.window.PROJECT_DATA;
assert.equal(data.companies.length,5574);
function search(value){node('#company-search').value=value;node('#search-button').listeners.click();return node('#company-card').innerHTML}
let html=search('600519');assert(html.includes('贵州茅台'));assert(html.includes('模型为什么这样判断'));assert(html.includes('亿元'));
const withText=data.companies.find(c=>data.text[c.code]);html=search(withText.code);assert(html.includes('MD&A 文本画像'));assert(html.includes('文化表述强度'));assert(html.includes('wordcloud/'));
const without=data.companies.find(c=>!data.text[c.code]);html=search(without.code);assert(html.includes('不在本次225家MD&A语料范围内'));
html=search('茅台');assert(html.includes('贵州茅台'));assert(!html.includes('NaN'));assert(!html.includes('undefined'));
search('不存在的测试公司xyz');assert(node('#search-status').textContent.includes('没有匹配到公司'));
console.log(JSON.stringify({status:'passed',company_lookup:true,text_present_and_missing:true,invalid_search:true,numeric_formatting:true,scope:'纯JavaScript逻辑核查；未替代浏览器视觉和真实交互检查'},null,2));
