"""Train-only calibration and locked evaluation; 45 attacks + clean."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from proposed import *
from baselines import embed_paper,extract_paper,verify_paper,prepare,oam_embed,oam_read
from attacks import ATTACKS,TRAIN,distort
from skimage.metrics import structural_similarity
import pandas as pd
PAPERS=('liu_tan_2002','ganic_eskicioglu_2005','luo_2021')
MAIN=('proposed',*PAPERS)
ABLATIONS=('docx_unsigned','docx_luo_corrected','no_cost_filter','fixed_margin','no_sync','no_ecc','direct_update')
TEST={'athens','butfly','manhatan','milkdrop','safari','sailboat','tiffany'}
BENCH_KEY=hashlib.sha256(b'public reproducibility key; not a deployment secret').digest()
CAL_ATTACKS=('jpeg70','jpeg30','gn_0.005','median5','scale0.5','rotation5','hist_eq')
def params_default(n):
 return {'liu_tan_2002':{'ll':ALPHA[n]/2},'ganic_eskicioglu_2005':{'ll':.05,'detail':.005},'luo_2021':{'ll':.05,'detail':.005,'T':.03},'proposed':{'alpha':ALPHA[n],'cap':.4,'tau':40.}}
def quality(im,out):return float(psnr(im,out)),float(structural_similarity(im,out,channel_axis=2,data_range=255))
def metrics(w,ws):
 ns=[nc(w,z) for z in ws];cl=[nc(w,np.clip(z,0,255)) for z in ws]
 return dict(nc=float(np.mean(ns)),nc_clipped=float(np.mean(cl)),rmse=float(np.mean([np.sqrt(np.mean((w-z)**2)) for z in ws])),**{f'nc_{name}':v for name,v in zip(('LL','LH','HL','HH') if len(ws)==4 else ('single',),ns)})
def candidates(m,n):
 if m=='liu_tan_2002':return [{'ll':a} for a in (.005,.01,.02,.03,.05,.08,.12,.2)]
 if m=='ganic_eskicioglu_2005':return [{'ll':.05*f,'detail':.005*f} for f in (.25,.5,1,2,4)]
 if m=='luo_2021':return [{'ll':i/100,'detail':j/1000,'T':.03} for i in range(3,7) for j in range(2,9)]
 raise ValueError(m)
def calibrate(hosts,wms,sizes,outdir):
 assert set(hosts)<=TRAIN and hosts
 table=[];frozen={str(n):params_default(n) for n in sizes}
 for m in PAPERS:
  cache={h:prepare(im,m) for h,im in hosts.items()}
  for n in sizes:
   scores=[]
   for p in candidates(m,n):
    vals=[]
    for h,im in hosts.items():
     for wn,wm in wms.items():
      w=np.asarray(wm.resize((n,n),Image.Resampling.BICUBIC),float);z,meta=embed_paper(im,w,m,p,cache[h]);pq,sq=quality(im,z)
      ncs=[metrics(w,extract_paper(distort(z,a),meta))['nc'] for a in CAL_ATTACKS]
      obj=1/max(pq,1e-9)+1/max(sq,1e-9)+1-np.mean(ncs);vals.append((pq,sq,np.mean(ncs),obj))
    pq,sq,nn,obj=np.mean(vals,axis=0);row=dict(method=m,n=n,params=json.dumps(p,sort_keys=True),psnr=pq,ssim=sq,nc=nn,objective=obj,psnr_floor_met=bool(pq>=43));table.append(row);scores.append((p,row));pd.DataFrame(table).to_csv(outdir/'calibration.csv',index=False)
   if m=='luo_2021':best=min(scores,key=lambda x:x[1]['objective'])
   else:
    eligible=[x for x in scores if x[1]['psnr_floor_met']];best=max(eligible,key=lambda x:(x[1]['nc'],x[1]['psnr'])) if eligible else max(scores,key=lambda x:x[1]['psnr'])
   frozen[str(n)][m]=best[0];print('CALIBRATED',m,n,best,flush=True)
 return frozen
def signature_metrics(im,k,rec,ids,code,cap,tau,angles):
 want=expected(k,rec,ids,code,cap,tau,BENCH_KEY);true_bits=np.unpackbits(np.frombuffer(want,dtype=np.uint8));coded=enc(true_bits,code)
 ss=score_grid(im,ids,angles=angles);decodes=[dec(z,code) for z in ss]
 ok=[i for i,b in enumerate(decodes) if b is not None and hmac.compare_digest(np.packbits(b).tobytes(),want)];zero=angles.index(0.)
 chosen=min(ok,key=lambda i:(abs(angles[i]),angles[i])) if ok else zero;decoded=decodes[chosen]
 return dict(asr=int(bool(ok)),raw_ber=float(np.mean((ss[zero]>0)!=coded)),aligned_raw_ber=float(np.mean((ss[chosen]>0)!=coded)) if ok else None,signature_ber=float(np.mean(decoded!=true_bits)) if decoded is not None else None,decode_failure=int(decoded is None),angle=angles[chosen] if ok else None,n_hypotheses=len(angles),tag_bits=128,carrier_bits=len(coded))
def case(task):
 h,im,wn,wm,n,conf,attacks,methods,outdir=task
 w=np.asarray(wm.resize((n,n),Image.Resampling.BICUBIC),float);rec=f'{h}|{wn}|{n}'
 pc=Config(**conf['proposed']);t=time.perf_counter();base,k,checks=embed(im,w,pc.alpha,prep(im),True);base_ms=1000*(time.perf_counter()-t)
 assert checks['iwt_exact'] and checks['arnold_exact'] and checks['ideal_w_rel']<1e-8
 g=geometry(base);items={};rows=[]
 if any(m in methods for m in ('proposed','no_sync')):
  t=time.perf_counter();z,ids,_,_=embed_sig(g,k,rec,'bch',pc.cap,pc.tau,BENCH_KEY);ms=base_ms+1000*(time.perf_counter()-t);items['proposed']=(z,(ids,'bch',pc.cap,pc.tau),ms);items['no_sync']=items['proposed']
 for method in methods:
  t=time.perf_counter()
  if method in PAPERS:
   z,meta=embed_paper(im,w,method,conf[method]);items[method]=(z,meta,1000*(time.perf_counter()-t))
  elif method=='docx_unsigned':items[method]=(base,None,base_ms)
  elif method=='docx_luo_corrected':
   b=tag(k,'direct_fixed',8,[],rec);z,ids=oam_embed(base,b,.12,True);items[method]=(z,ids,base_ms+1000*(time.perf_counter()-t))
  elif method in ('no_cost_filter','fixed_margin','no_ecc','direct_update'):
   cap=1. if method=='no_cost_filter' else pc.cap;tau=.03 if method=='fixed_margin' else pc.tau;code='raw' if method=='no_ecc' else 'bch'
   z,ids,_,_=embed_sig(g,k,rec,code,cap,tau,BENCH_KEY,direct_update=method=='direct_update');items[method]=(z,(ids,code,cap,tau),base_ms+1000*(time.perf_counter()-t))
 for method in methods:
  z,meta,ms=items[method];pq,sq=quality(im,z)
  for a in attacks:
   at=distort(z,a);t=time.perf_counter();extra={}
   if method in PAPERS:
    ws=extract_paper(at,meta);ok,ber=verify_paper(at,meta);extra=dict(asr=ok,signature_ber=ber,tag_bits=8 if method=='luo_2021' else 0,carrier_bits=8 if method=='luo_2021' else 0)
   else:
    ws=[extract(at,k)]
    if method=='docx_luo_corrected':
     got=oam_read(at,meta,True);b=tag(k,'direct_fixed',8,[],rec);ber=float(np.mean(got!=b));extra=dict(asr=int(ber==0),signature_ber=ber,tag_bits=8,carrier_bits=8)
    elif method!='docx_unsigned':
     ids,code,cap,tau=meta;extra=signature_metrics(at,k,rec,ids,code,cap,tau,[0.] if method=='no_sync' else ANGLES)
    else:extra=dict(asr=None,tag_bits=0,carrier_bits=0)
   row=dict(host=h,split='train' if h in TRAIN else 'test',watermark=wn,n=n,method=method,group='paper_comparison' if method in MAIN else 'ablation',attack=a,psnr=pq,ssim=sq,embed_ms=ms,extract_ms=1000*(time.perf_counter()-t),**metrics(w,ws),**extra);row['authorized_nc']=row['nc'] if row.get('asr')==1 else None;rows.append(row)
  print(h,wn,n,method,'done',flush=True)
 path=outdir/'cases'/f'{h}__{wn}__{n}.csv';tmp=path.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(path);return len(rows)
def summarize(outdir):
 fs=sorted((outdir/'cases').glob('*.csv'))
 if not fs:return
 d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True);d.to_csv(outdir/'all_results.csv',index=False)
 for names,filename in [(['split','group','method','n'],'summary_by_size.csv'),(['split','group','method','attack'],'summary_by_attack.csv'),(['split','group','method','host'],'summary_by_host.csv'),(['split','group','method'],'summary.csv')]:
  rows=[]
  for key,x in d.groupby(names,dropna=False):
   clean=x[x.attack=='clean'];att=x[x.attack!='clean'];emb=x.drop_duplicates(['host','watermark','n','method']);a=att['asr'].dropna();c=clean['asr'].dropna()
   rows.append(dict(zip(names,key),cases=len(x),embeddings=len(emb),psnr=emb.psnr.mean(),ssim=emb.ssim.mean(),nc_clean=clean.nc.mean(),nc_attack=att.nc.mean(),nc_clipped_attack=att.nc_clipped.mean(),asr_clean=c.mean(),asr_attack=a.mean(),auth_pass=int(a.sum()),auth_total=len(a),auth_clean_pass=int(c.sum()),auth_clean_total=len(c)))
  pd.DataFrame(rows).to_csv(outdir/filename,index=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='results/full');p.add_argument('--phase',choices=['calibrate','benchmark','all'],default='all');p.add_argument('--split',choices=['all','train','test'],default='all');p.add_argument('--sizes',default='64,128,256');p.add_argument('--workers',type=int,default=4);p.add_argument('--config');p.add_argument('--ablation',action='store_true');p.add_argument('--ablation-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 hosts,wms=load(a.data);sizes=[int(x) for x in a.sizes.split(',')];assert sizes and len(set(sizes))==len(sizes) and set(sizes)<={64,128,256};assert set(hosts)==TRAIN|TEST and len(wms)==2 and len(ATTACKS)==46
 out=Path(a.out);(out/'cases').mkdir(parents=True,exist_ok=True);train={h:hosts[h] for h in sorted(TRAIN)}
 if a.smoke:
  assert a.phase=='benchmark','Use --smoke with --phase benchmark';hosts={h:hosts[h] for h in ('airplane','athens')};wms=dict(list(wms.items())[:1]);attacks=['clean','jpeg70','jp2_5','rotation5','gn_0.05']
 else:attacks=ATTACKS
 if a.phase in ('calibrate','all'):
  conf=calibrate(train,wms,sizes,out)
 else:conf=json.loads(Path(a.config).read_text()) if a.config else {str(n):params_default(n) for n in sizes}
 if a.phase=='calibrate':
  (out/'locked_config.json').write_text(json.dumps(conf,indent=2));return
 methods=ABLATIONS if a.ablation_only else MAIN+ABLATIONS if a.ablation else MAIN
 protocol=dict(train=sorted(TRAIN),test=sorted(TEST),hosts=sorted(hosts),sizes=sizes,watermarks=sorted(wms),methods=methods,attacks=attacks,conf=conf,smoke=a.smoke,split=a.split)
 manifest=out/'protocol.json'
 if manifest.exists() and json.loads(manifest.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Use a new output directory for a changed protocol')
 manifest.write_text(json.dumps(protocol,indent=2));(out/'locked_config.json').write_text(json.dumps(conf,indent=2));tasks=[]
 for h,im in hosts.items():
  if a.split=='train' and h not in TRAIN or a.split=='test' and h not in TEST:continue
  for wn,wm in wms.items():
   for n in sizes:
    if a.resume and (out/'cases'/f'{h}__{wn}__{n}.csv').exists():continue
    tasks.append((h,im,wn,wm,n,conf[str(n)],attacks,methods,out))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for num in pool.map(case,tasks):print('completed rows:',num,flush=True)
 summarize(out)
if __name__=='__main__':main()
