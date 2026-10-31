"""本地全文分块向量；输入与代码指纹绑定缓存，逐文档可恢复。"""
import gzip,math,os,time
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from study_common import *

TEXT_PROTOCOL=json.loads((STUDY/'text_protocol.json').read_text(encoding='utf-8'))
TEXT_OUT=OUT/'06_全文主题对照'

def encode(ids_list,tokenizer,model):
    if not ids_list:raise ValueError('空批次')
    width=max(len(x)+2 for x in ids_list)
    assert width<=512
    x=torch.full((len(ids_list),width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(x)
    for j,ids in enumerate(ids_list):
        row=[tokenizer.cls_token_id]+[int(v) for v in ids]+[tokenizer.sep_token_id]
        x[j,:len(row)]=torch.tensor(row);mask[j,:len(row)]=1
    with torch.inference_mode():
        return F.normalize(model(input_ids=x,attention_mask=mask).last_hidden_state[:,0],p=2,dim=1).cpu().numpy()

def document_vector(parts,tokenizer,model):
    acc=np.zeros(model.config.hidden_size,dtype=np.float32)
    batch=TEXT_PROTOCOL['batch_size']
    for start in range(0,len(parts),batch):
        for vector in encode(parts[start:start+batch],tokenizer,model):acc+=vector
    mean=acc/len(parts)
    return (mean/np.linalg.norm(mean)).astype(np.float32)

def balanced_bounds(n,width=510):
    if n<=0:raise ValueError('原文分词后为空，需人工核查')
    return np.linspace(0,n,math.ceil(n/width)+1,dtype=np.int64)

def main():
    setup();TEXT_OUT.mkdir(parents=True,exist_ok=True);assert_v1_unchanged()
    source_work=CODE/'中间数据';corpus_file=source_work/'tokenized_corpus_v1.json.gz'
    with gzip.open(corpus_file,'rt',encoding='utf-8') as f:docs=json.load(f)
    assert len(docs)==1123
    model_dir=DATA/'情感分析演示/语言模型/bge-small-zh-v1.5'
    sources=[{'file':d['file'],'code':d['code'],'year':d['year'],'sha256':sha(source_work/'corpus_sample'/d['file'])} for d in docs]
    fingerprint={'protocol':TEXT_PROTOCOL,'implementation_sha256':sha(__file__),'corpus_sha256':sha(corpus_file),
                 'documents':sources,'model_files':[{'path':str(p.relative_to(model_dir)),'sha256':sha(p)} for p in sorted(model_dir.rglob('*')) if p.is_file()],
                 'old_embeddings_sha256':sha(source_work/'mda_three_segment_embeddings_v1.npy'),
                 'versions':{p:version(p) for p in ['torch','transformers','tokenizers','numpy']}}
    run_id=hashlib.sha256(json.dumps(fingerprint,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:12]
    directory=WORK/'full_text_embeddings'/run_id;directory.mkdir(parents=True,exist_ok=True)
    dump(directory/'input_manifest.json',fingerprint)
    dump(TEXT_OUT/'嵌入运行入口.json',{'run_id':run_id,'work_directory':str(directory),'protocol':TEXT_PROTOCOL,'started_beijing':now()})
    torch.set_num_threads(TEXT_PROTOCOL['torch_threads'])
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    tokenizer.model_max_length=10**12  # 全文先分词，随后显式切成不超过模型上限的块。
    model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).eval()
    assert model.config.hidden_size==512 and model.config.max_position_embeddings>=512
    token_cache=directory/'all_token_ids.npz'
    if token_cache.exists():
        packed=np.load(token_cache);tokens=packed['tokens'];offsets=packed['offsets']
    else:
        arrays=[];offsets=[0]
        for i,d in enumerate(docs):
            text=(source_work/'corpus_sample'/d['file']).read_text(encoding='utf-8-sig')
            ids=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
            assert len(ids)>0,d['file']
            arrays.append(np.asarray(ids,dtype=np.int32));offsets.append(offsets[-1]+len(ids))
        tokens=np.concatenate(arrays);offsets=np.asarray(offsets,dtype=np.int64)
        np.savez_compressed(token_cache,tokens=tokens,offsets=offsets)
    assert len(offsets)==len(docs)+1 and offsets[-1]==len(tokens)
    lengths=np.diff(offsets);blocks=np.ceil(lengths/TEXT_PROTOCOL['chunk_content_tokens_max']).astype(int)
    old=np.load(source_work/'mda_three_segment_embeddings_v1.npy')
    coverage=pd.read_csv(ROOT/'成果交付/04_文本分析/长文本取样覆盖率_v1.csv',dtype={'code':str})
    assert [(str(d['code']).zfill(6),int(d['year'])) for d in docs]==list(zip(coverage.code.str.zfill(6),coverage.year))
    assert np.array_equal(lengths,coverage.original_tokens.to_numpy())
    table=pd.DataFrame([{k:d[k] for k in ['code','company','year','file']} for d in docs])
    table['original_tokens']=lengths;table['full_text_blocks']=blocks
    table['head_middle_tail_coverage']=coverage.coverage_ratio.to_numpy();table['full_text_coverage']=1.0
    save(table,TEXT_OUT/'全文分块与旧版覆盖率对照.csv')
    print(json.dumps({'run_id':run_id,'documents':len(docs),'tokens':int(lengths.sum()),'blocks':int(blocks.sum()),'resumable':True}),flush=True)
    checks=[]
    for i in np.linspace(0,len(docs)-1,12,dtype=int):
        ids=tokens[offsets[i]:offsets[i+1]];n=len(ids)
        starts=sorted(set([0,max(0,n//2-255),max(0,n-510)]))
        vector=document_vector([ids[start:start+510] for start in starts],tokenizer,model)
        checks.append({'code':docs[i]['code'],'year':docs[i]['year'],'cosine_old_vs_recomputed':float(vector@old[i]),'max_absolute_difference':float(np.max(np.abs(vector-old[i])))})
    assert min(x['cosine_old_vs_recomputed'] for x in checks)>.99999
    assert max(x['max_absolute_difference'] for x in checks)<1e-4
    save(pd.DataFrame(checks),TEXT_OUT/'旧版嵌入抽样重算核查.csv')
    checkpoint=directory/'embedding_checkpoint.npz'
    if checkpoint.exists():
        packed=np.load(checkpoint);vectors=packed['vectors'];complete=packed['complete']
    else:vectors=np.zeros_like(old);complete=np.zeros(len(docs),dtype=bool)
    start_time=time.perf_counter();completed_before=int(complete.sum());processed_blocks=0
    for i,d in enumerate(docs):
        if complete[i]:continue
        ids=tokens[offsets[i]:offsets[i+1]];bounds=balanced_bounds(len(ids),TEXT_PROTOCOL['chunk_content_tokens_max'])
        assert bounds[0]==0 and bounds[-1]==len(ids) and np.all(np.diff(bounds)>0) and np.max(np.diff(bounds))<=510
        parts=[ids[lo:hi] for lo,hi in zip(bounds[:-1],bounds[1:])]
        vectors[i]=document_vector(parts,tokenizer,model);complete[i]=True;processed_blocks+=len(parts)
        if (i+1)%20==0 or i+1==len(docs):
            temporary=directory/'embedding_checkpoint_pending.npz'
            np.savez(temporary,vectors=vectors,complete=complete);os.replace(temporary,checkpoint)
            elapsed=time.perf_counter()-start_time
            status={'time_beijing':now(),'run_id':run_id,'documents_done':int(complete.sum()),'total_documents':len(docs),
                    'blocks_done_this_run':processed_blocks,'elapsed_seconds_this_run':round(elapsed,1),
                    'estimated_remaining_seconds':round((blocks[~complete].sum()/max(processed_blocks,1))*elapsed,1)}
            dump(directory/'progress.json',status);print(json.dumps(status),flush=True)
    assert complete.all() and np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    np.save(directory/'full_text_embeddings.npy',vectors)
    cosine=np.sum(old*vectors,axis=1)
    table['cosine_old_vs_full']=cosine;save(table,TEXT_OUT/'全文分块与旧版覆盖率对照.csv')
    summary={'finished_beijing':now(),'run_id':run_id,'documents':len(docs),'original_tokens':int(lengths.sum()),'total_blocks':int(blocks.sum()),
             'old_coverage_median':float(coverage.coverage_ratio.median()),'old_coverage_mean':float(coverage.coverage_ratio.mean()),
             'full_coverage_each_document':1.0,'cosine_median':float(np.median(cosine)),'cosine_min':float(np.min(cosine)),
             'old_embedding_check_documents':len(checks),'old_embedding_max_absolute_difference':max(x['max_absolute_difference'] for x in checks),
             'embedding_sha256':sha(directory/'full_text_embeddings.npy'),'v1_preserved_files':assert_v1_unchanged()}
    dump(TEXT_OUT/'全文嵌入摘要.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
