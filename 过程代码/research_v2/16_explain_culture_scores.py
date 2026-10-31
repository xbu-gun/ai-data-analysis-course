"""对已有784词得分作独立复算和贡献分解，不改变词典或得分。"""
from collections import Counter,defaultdict
import gzip,math
import numpy as np
import pandas as pd
from study_common import *

def main():
    setup();out=OUT/'07_文化指标解释';out.mkdir(parents=True,exist_ok=True)
    old=ROOT/'成果交付/04_文本分析'
    dictionary=pd.read_csv(old/'五维文化词典_审阅后_v2.csv')
    scores=pd.read_csv(old/'公司年度文化指标_v2.csv',dtype={'code':str})
    with gzip.open(CODE/'中间数据/tokenized_corpus_v1.json.gz','rt',encoding='utf-8') as f:docs=json.load(f)
    assert len(dictionary)==784 and dictionary.word.is_unique
    assert list(zip(scores.code.str.zfill(6),scores.year))==[(str(d['code']).zfill(6),d['year']) for d in docs]
    by_word=dictionary.set_index('word').to_dict('index');words=set(by_word)
    counts=[Counter(d['tokens']) for d in docs];df=Counter()
    for c in counts:df.update(set(c)&words)
    totals=Counter();best={};recomputed=defaultdict(lambda:np.zeros(len(docs)))
    for i,(d,c) in enumerate(zip(docs,counts)):
        for word in sorted(c.keys()&words):
            row=by_word[word];value=c[word]*math.log(len(docs)/df[word])/math.log(1+row['rank'])/len(d['tokens'])
            totals[word]+=value;recomputed[row['dimension']][i]+=value
            if word not in best or value>best[word][0]:best[word]=(value,i,c[word])
    errors={dim:float(np.max(np.abs(values-scores[dim].to_numpy()))) for dim,values in recomputed.items()}
    assert max(errors.values())<1e-12,errors
    rows=[];examples=[]
    for word,row in by_word.items():
        dimension=row['dimension'];denominator=float(np.sum(recomputed[dimension]))
        rows.append({'word':word,'dimension':dimension,'rank':row['rank'],'document_frequency':df[word],
                     'idf':math.log(len(docs)/df[word]),'sum_length_normalized_contribution':totals[word],
                     'share_of_dimension_total':totals[word]/denominator if denominator else 0})
    contributions=pd.DataFrame(rows).sort_values(['dimension','sum_length_normalized_contribution'],ascending=[True,False])
    save(contributions,out/'784词_得分贡献分解.csv')
    top5=contributions.groupby('dimension',sort=False).head(5)
    for r in top5.itertuples():
        value,i,tf=best[r.word];d=docs[i];path=CODE/'中间数据/corpus_sample'/d['file']
        text=path.read_text(encoding='utf-8-sig');pos=text.find(r.word);assert pos>=0
        start=max(0,pos-90);end=min(len(text),pos+len(r.word)+160)
        examples.append({'dimension':r.dimension,'word':r.word,'dictionary_rank':r.rank,
                         'share_of_dimension_total':r.share_of_dimension_total,'example_selection':'该词归一贡献最大的一篇，非随机样本',
                         'code':d['code'],'company':d['company'],'year':d['year'],'file':d['file'],
                         'source_sha256':sha(path),'word_count_in_document':tf,'character_start':start,'character_end':end,'evidence':text[start:end]})
    save(pd.DataFrame(examples),out/'各维主要贡献词_原文例证.csv')
    review_path=STUDY/'culture_context_review_v2.json'
    review=json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
    if review:
        assert set(review)==set(top5.word)
        reviewed=pd.DataFrame(examples)
        reviewed['context_review_status']=reviewed.word.map(lambda w:review[w]['status'])
        reviewed['context_review_note']=reviewed.word.map(lambda w:review[w]['note'])
        reviewed['reviewer']='AI对所列连续片段的语义初审；非人工独立标注'
        save(reviewed,out/'各维主要贡献词_语境审阅.csv')
    aggregate=[]
    for dim,g in contributions.groupby('dimension'):
        aggregate.append({'dimension':dim,'dictionary_words':len(g),'top5_contribution_share':float(g.head(5).share_of_dimension_total.sum()),
                          'documents_with_nonzero_score':int(np.sum(recomputed[dim]>0)),
                          'max_recalculation_absolute_error':errors[dim]})
    save(pd.DataFrame(aggregate),out/'五维贡献集中度与复算核查.csv')
    summary={'time_beijing':now(),'documents':len(docs),'dictionary_words':len(dictionary),'source_dictionary_sha256':sha(old/'五维文化词典_审阅后_v2.csv'),
             'source_scores_sha256':sha(old/'公司年度文化指标_v2.csv'),'max_recalculation_absolute_error':max(errors.values()),
             'dimensions':aggregate,'purposive_quote_examples':len(examples),'context_review_status_counts':dict(Counter(x['status'] for x in review.values())),
             'dictionary_or_scores_changed':False,'v1_preserved_files':assert_v1_unchanged()}
    dump(out/'文化指标解释摘要.json',summary)
    lines=['# 文化指标的主要得分成分',f'\n时间（北京时间）：{now()}。',
           '\n本轮对已经确认剔除28词后的784词版本独立复算，未重新训练词向量，也未删除、增加或调整词典。',
           f"\n1,123篇的五维得分与既有表逐条核对，最大绝对误差为{max(errors.values()):.3g}。这检验了公式与文件的一致性，不等于验证文化含义。",
           '\n按每篇长度归一后的得分贡献加总，查看每维主要由哪些词支撑。贡献比例是一个维度内部的成分，不能当作跨维度的重要性权重。',
           '\n| 维度 | 前5词 | 前5词贡献占比 | 非零得分文档数 |','|---|---|---:|---:|']
    for row in aggregate:
        g=top5[top5.dimension==row['dimension']]
        lines.append(f"| {row['dimension']} | {'、'.join(g.word)} | {row['top5_contribution_share']:.1%} | {row['documents_with_nonzero_score']} |")
    lines+=['\n每个主要贡献词选取其归一贡献最大的一篇，提取一段真实连续原文，保存文件名、指纹及字符起止位置。这25个用例用于解释和发现边界含义，属于目的性选取，不能由此估计整部词典的准确率。',
            '\n相似词扩展可能包含管理绩效、商业服务、财务合规等邻近概念。即使某词在一句话里适用，也不能推出它在所有语境里适用。五维分数仍表示管理层披露中的词汇强度；“诚信”得分高不能证明实际更诚信。',
            '\n本轮原文语义审阅若发现边界含义，将单独记录并保留词典；后续剔除、改维度或改权重需按具体证据另作决定。']
    if review:
        lines+=['\n## 原文语境初审',
                '\n已对25个用例做AI语义初审，具体解释逐条保存在“各维主要贡献词_语境审阅.csv”。分类仅描述这些目的性选取的片段，不是词典总体准确率。',
                '\n例如团队维度中的“品牌、平台”，在所列原文中指业务品牌或互联网平台；尊重维度的“全体、广大”在用例中是范围限定或修饰词；质量维度的“安全”对应网络安全行业。它们说明主题、行业与泛词可能进入文化得分。',
                '\n诚信维度的“风控”例证在评论行业内中小贸易商的风险，并非直接描述本公司诚信。词频计数无法自动分清说话对象、否定和外部环境。',
                '\n当前继续沿用用户此前确定的保守处理：仅剔除已确认的28词，其余保留并揭示限制。本轮未按这些用例直接删除词或重算新排名。']
    (out/'文化指标贡献解释.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
