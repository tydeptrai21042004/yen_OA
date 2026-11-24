"""Small independent correctness checks on real images and BCH errors."""
import argparse,tempfile
from proposed import *
from baselines import embed_paper,extract_paper,verify_paper
from attacks import ATTACKS,distort
from benchmark import params_default,PAPERS,BENCH_KEY

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--host',default='athens');p.add_argument('--size',type=int,default=64);a=p.parse_args()
 hosts,wms=load(a.data);im=hosts[a.host];w=np.asarray(next(iter(wms.values())).resize((a.size,a.size),Image.Resampling.BICUBIC),float);n=a.size;key=BENCH_KEY
 assert len(ATTACKS)==46 and len(set(ATTACKS))==46
 rng=np.random.default_rng(2024);assert bch.setup(18)[2]==131
 for _ in range(100):
  bits=rng.integers(0,2,128,dtype=np.uint8);c=bch.encode(bits);c[rng.choice(255,18,replace=False)]^=1;assert np.array_equal(bits,fast(c.astype(float)-.5))
 print('BCH: 100/100 eighteen-error recoveries',flush=True)
 z,m,base,ch=embed_image(im,w,key,'validation',Config(ALPHA[n]));assert ch['iwt_exact'] and ch['arnold_exact'] and ch['ideal_w_rel']<1e-8
 for name in ['schur_rel','svd_rel','second_svd_rel','rebuild_spectrum_rel']:assert ch[name]<1e-10
 # Physical extraction, never given original tag or attack identity.
 with tempfile.TemporaryDirectory() as d:
  save_meta(Path(d)/'metadata.npz',m);m=load_meta(Path(d)/'metadata.npz');ww,ok,angle=recover(z,m,key);assert ok and nc(w,ww)>.99
  assert not authenticate(z,m,b'wrong key of at least 16 bytes')[0]
  bad=dict(m,record='tampered');assert not authenticate(z,bad,key)[0]
  assert not authenticate(im,m,key)[0]
 print('DOCX algebra, metadata roundtrip, clean authentication, wrong-key and record rejection: PASS',flush=True)
 for method in PAPERS:
  out,meta=embed_paper(im,w,method,params_default(n)[method]);ws=extract_paper(out,meta);assert all(x.shape==w.shape and np.isfinite(x).all() for x in ws);print(method, 'clean',float(np.mean([nc(w,x) for x in ws])), 'ASR',verify_paper(out,meta)[0],flush=True)
 # Exact geometry and selected ids are identical; only matrix update differs.
 g=geometry(base);k=m['k'];a1,i1,_,_=embed_sig(g,k,'matched','bch',.4,40.,key);a2,i2,_,_=embed_sig(g,k,'matched','bch',.4,40.,key,direct_update=True);assert np.array_equal(i1,i2);assert a1.shape==a2.shape==im.shape
 print('Matched direct/Givens ablation: PASS',flush=True)
 for name in ATTACKS:
  out=distort(z,name);assert out.shape==im.shape and out.dtype==np.uint8
 print('All 46 conditions return valid RGB arrays',flush=True)
if __name__=='__main__':main()
