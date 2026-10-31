"""核查全文对照实际产物、样本顺序、缓存、原文位置和交付完整性。"""
import ast,gzip,re,zipfile
import numpy as np
import pandas as pd
import joblib
from study_common import *

def main():
    directory=OUT/'06_全文主题对照'
    entry=json.loads((directory/'嵌入运行入口.json').read_text(encoding='utf-8'))
    work=Path(entry['work_directory']);manifest=json.loads((work/'input_manifest.json').read_text(encoding='utf-8'))
    assert entry['run_id']==hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:12]
    assert manifest['implementation_sha256']==sha(STUDY/'12_full_text_embeddings.py')
    assert manifest['protocol']==json.loads((STUDY/'text_protocol.json').read_text(encoding='utf-8'))
    corpus_file=CODE/'中间数据/tokenized_corpus_v1.json.gz'
    assert manifest['corpus_sha256']==sha(corpus_file)
    with gzip.open(corpus_file,'rt',encoding='utf-8') as f:docs=json.load(f)
    assert len(docs)==1123
    original_keys=[(str(d['code']).zfill(6),int(d['year'])) for d in docs]
    assert len(set(original_keys))==len(original_keys)
    for row in manifest['documents']:assert sha(CODE/'中间数据/corpus_sample'/row['file'])==row['sha256']
    model_dir=DATA/'情感分析演示/语言模型/bge-small-zh-v1.5'
    for row in manifest['model_files']:assert sha(model_dir/row['path'])==row['sha256']
    packed=np.load(work/'all_token_ids.npz');lengths=np.diff(packed['offsets'])
    cov=pd.read_csv(directory/'全文分块与旧版覆盖率对照.csv',dtype={'code':str})
    assert list(zip(cov.code.str.zfill(6),cov.year))==original_keys
    assert np.array_equal(cov.original_tokens.to_numpy(),lengths)
    assert np.all(cov.full_text_coverage==1) and np.all(cov.original_tokens>0)
    assert np.all(cov.full_text_blocks*510>=cov.original_tokens)
    assert np.all((cov.full_text_blocks-1)*510<cov.original_tokens)
    vectors=np.load(work/'full_text_embeddings.npy');checkpoint=np.load(work/'embedding_checkpoint.npz')
    assert vectors.shape==(1123,512) and np.isfinite(vectors).all()
    assert checkpoint['complete'].all() and np.array_equal(checkpoint['vectors'],vectors)
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    old_check=pd.read_csv(directory/'旧版嵌入抽样重算核查.csv')
    assert len(old_check)==12 and old_check.max_absolute_difference.max()<1e-4
    replication=json.loads((directory/'旧版聚类复现核查.json').read_text(encoding='utf-8'))
    assert replication['exact_same_numbered_assignments'] and replication['ari_including_outliers']==1
    summary=json.loads((directory/'全文主题对照摘要.json').read_text(encoding='utf-8'))
    assert summary['primary_seed']==42 and not summary['selected_seed_by_results']
    for seed_row in summary['seed_runs']:
        frame=pd.read_csv(directory/f"seed{seed_row['seed']}/逐文档主题.csv",dtype={'code':str})
        assert list(zip(frame.code.str.zfill(6),frame.year))==original_keys
        assert int((frame.topic==-1).sum())==seed_row['outlier_documents']
        assert frame.topic.nunique()-(int((frame.topic==-1).any()))==seed_row['final_topics']
        fitted=joblib.load(Path(seed_row['fit_directory'])/'fitted.joblib')
        assert np.array_equal(fitted['model'].topics_,frame.topic.to_numpy())
        assert fitted['configuration']['seed']==seed_row['seed']
        assert fitted['configuration']['embedding_bytes_sha256']==hashlib.sha256(vectors.tobytes()).hexdigest()
    evidence=pd.read_csv(directory/'seed42/主题代表原文_待审阅.csv',dtype={'code':str})
    for row in evidence.itertuples():
        p=CODE/'中间数据/corpus_sample'/row.file;text=p.read_text(encoding='utf-8-sig')
        assert sha(p)==row.source_sha256
        assert text[int(row.character_start):int(row.character_end)]==row.evidence
        if isinstance(row.matched_words,str):assert all(w in row.evidence for w in row.matched_words.split('、'))
    extra=pd.read_csv(directory/'seed42/补充命名原文_酒店业务.csv')
    for row in extra.itertuples():
        p=CODE/'中间数据/corpus_sample'/row.file;text=p.read_text(encoding='utf-8-sig')
        assert sha(p)==row.source_sha256 and text[int(row.character_start):int(row.character_end)]==row.evidence
    names=json.loads((STUDY/'text_topic_names_v2.json').read_text(encoding='utf-8'))
    primary=pd.read_csv(directory/'seed42/逐文档主题.csv')
    assert set(names)==set(primary.topic.astype(str))
    for value in names.values():assert value['name'] and value['note']
    culture_dir=OUT/'07_文化指标解释'
    culture=json.loads((culture_dir/'文化指标解释摘要.json').read_text(encoding='utf-8'))
    assert culture['max_recalculation_absolute_error']<1e-12 and not culture['dictionary_or_scores_changed']
    contexts=pd.read_csv(culture_dir/'各维主要贡献词_语境审阅.csv')
    assert len(contexts)==25 and contexts.word.is_unique and contexts.context_review_status.notna().all()
    for row in contexts.itertuples():
        p=CODE/'中间数据/corpus_sample'/row.file;text=p.read_text(encoding='utf-8-sig')
        assert sha(p)==row.source_sha256 and text[int(row.character_start):int(row.character_end)]==row.evidence
        assert row.word in row.evidence
    html=(directory/'全文主题互动图.html').read_text(encoding='utf-8')
    references=re.findall(r'<script[^>]+src="([^"]+)"',html)
    assert references and all(not ref.startswith(('http:','https:','//')) and (directory/ref).exists() for ref in references)
    with zipfile.ZipFile(OUT/'第二轮研究_复现源码包.zip') as z:
        assert z.testzip() is None
        assert all(Path(name).suffix.lower() in {'.py','.json','.md'} for name in z.namelist())
        assert all('/work/' not in name and '/logs/' not in name for name in z.namelist())
    for p in STUDY.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    result={'time_beijing':now(),'documents_and_order_verified':1123,'full_text_coverage_all_documents':1.0,
            'source_model_protocol_and_code_fingerprints_verified':True,'old_embedding_sample_verified':12,
            'old_clustering_exactly_reproduced':True,'predefined_seeds_verified':[42,43,44],
            'representative_quotes_verified':len(evidence),'additional_naming_quotes_verified':len(extra),'topic_names_complete':True,'interactive_plot_resources_local':True,
            'culture_recalculation_and_25_context_quotes_verified':True,
            'source_package_integrity_passed':True,'v1_files_unchanged':assert_v1_unchanged(),'published':False}
    dump(directory/'文本对照结果核查.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
