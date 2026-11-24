import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np, scipy.linalg as la, scipy.ndimage as nd, hashlib,hmac,io,zipfile,csv,json,time,sys
from PIL import Image
from pathlib import Path
P=Path(__file__).resolve().parent;OUT=P/'results';OUT.mkdir(exist_ok=True);KEY=b'test-only-key';MASK=167

def color(a):
 r,g,b=a.astype(float).transpose(2,0,1);y=.299*r+.587*g+.114*b;return np.rint(y).astype(np.int64),np.stack([(b-y)/1.772,(r-y)/1.402],-1)
def rgb(y,c):
 cb,cr=c.transpose(2,0,1);r=y+1.402*cr;b=y+1.772*cb;g=(y-.299*r-.114*b)/.587;return np.rint(np.clip(np.stack([r,g,b],-1),0,255)).astype(np.uint8)
def split(a,axis,i):
 e=np.take(a,np.arange(0,a.shape[axis],2),axis);o=np.take(a,np.arange(1,a.shape[axis],2),axis);d=o-e;return (e+d//2,d) if i else ((e+o)/np.sqrt(2),d/np.sqrt(2))
def merge(l,h,axis,i):
 e=l-h//2 if i else (l-h)/np.sqrt(2);o=e+h if i else (l+h)/np.sqrt(2);shape=list(e.shape);shape[axis]*=2;a=np.empty(shape,dtype=e.dtype);ix=[slice(None)]*2;ix[axis]=slice(0,None,2);a[tuple(ix)]=e;ix[axis]=slice(1,None,2);a[tuple(ix)]=o;return a
def wave(a,i=True):
 l,h=split(a,1,i);ll,lh=split(l,0,i);hl,hh=split(h,0,i);return ll,lh,hl,hh
def iw(b,i=True):
 ll,lh,hl,hh=b;return merge(merge(ll,lh,0,i),merge(hl,hh,0,i),1,i)
def arn(w,inv=False):
 n=len(w);y,x=np.indices(w.shape);a=w.copy();iy=(x+y)%n;ix=(2*x+y)%n
 for _ in range(5):
  if inv:a=a[iy,ix]
  else:
   out=np.empty_like(a);out[iy,ix]=a;a=out
 return a
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-15))
def nc(a,b):return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-15))
def psnr(a,b):return 10*np.log10(255**2/max(np.mean((a.astype(float)-b.astype(float))**2),1e-15))
def prep(im):
 y,c=color(im);bands=wave(y);D,Q=la.schur(bands[0].astype(float),output='real');U,s,V=la.svd(D);return y,c,bands,D,Q,U,s,V
def embed(im,w,alpha,p,check=False):
 y,c,bands,D,Q,U,s,V=p;n=len(w);S=np.diag(s);B=S.copy();wa=arn(w);B[:n,:n]+=alpha*wa;Uw,sw,Vw=la.svd(B);Dp=(U*sw)@V;Lp=Q@Dp@Q.T;k=(Uw,Vw,S[:n,:n].copy(),alpha,n);Lr=np.rint(Lp).astype(np.int64);out=rgb(iw((Lr,*bands[1:])),c);log={}
 if check:
  ideal=arn((((Uw*sw)@Vw-S)[:n,:n])/alpha,True);log=dict(iwt_exact=np.array_equal(iw(bands),y),color_roundtrip_psnr=psnr(im,rgb(y,c)),schur_rel=rel(bands[0],Q@D@Q.T),schur_orth=rel(np.eye(len(Q)),Q.T@Q),svd_rel=rel(D,(U*s)@V),partition_rel=rel(S,np.block([[S[:n,:n],S[:n,n:]],[S[n:,:n],S[n:,n:]]])),arnold_exact=np.array_equal(arn(wa,True),w),second_svd_rel=rel(B,(Uw*sw)@Vw),rebuild_spectrum_rel=rel(sw,la.svdvals(Lp)),ideal_w_rel=rel(w,ideal),ll_round_rmse=np.sqrt(np.mean((Lr-Lp)**2)),direct_schur_spectrum_rel=rel(s,la.svdvals(bands[0])))
 return out,k,log
def extract(im,k):
 U,V,S,alpha,n=k;D,_=la.schur(wave(color(im)[0])[0].astype(float),output='real');s=la.svdvals(D);return arn(((U[:n,:]*s)@V[:,:n]-S)/alpha,True)
def tag(k,mode,L,ids,rec):
 U,V,S,alpha,n=k;pack=lambda a:np.ascontiguousarray(a,dtype='<f8').tobytes()
 if mode!='givens':return np.unpackbits(np.array([hashlib.sha1(pack(U)).digest()[0]^hashlib.sha1(pack(V.T)).digest()[0]^MASK],dtype=np.uint8))
 msg=b'BT601|IWT-Haar|Arnold-1-1-5|DWT-Haar2|'+pack(U)+pack(V)+pack(S)+json.dumps([alpha,n,rec]).encode()+np.asarray(ids,dtype='<i4').tobytes();return np.unpackbits(np.frombuffer(hmac.digest(KEY,msg,'sha256'),dtype=np.uint8))[:L]
def dwt(y):
 b1=wave(y.astype(float),False);return b1,wave(b1[0],False)
def blocks(ll):return ll.reshape(32,4,32,4).transpose(0,2,1,3).reshape(-1,4,4).copy()
def unblocks(a):return a.reshape(32,32,4,4).transpose(0,2,1,3).reshape(128,128)
def sigembed(im,k,mode,L,rho,rec):
 y,c=color(im);b1,b2=dwt(y);a=blocks(b2[0]);u,s,v=np.linalg.svd(a);std=a.std(axis=(1,2));r=np.hypot(u[:,1,0],u[:,2,0]);gap=s[:,0]-s[:,1]
 if mode=='givens':
  th=np.arctan2(abs(u[:,2,0]),abs(u[:,1,0]));beta=np.arcsin(rho/np.sqrt(2));phi=abs(th-np.pi/4)+beta;cost=4*np.sin(phi/2)**2*np.sum(a[:,1:3,:]**2,axis=(1,2));score=cost/np.maximum(gap/np.maximum(s[:,0],1e-12),.05)**2;score[(r<.25)|(gap<40)]=np.inf;ids=np.argsort(score,kind='stable')[:L];assert np.isfinite(score[ids]).all()
 else:ids=np.argsort(-std,kind='stable')[:L]
 bits=tag(k,mode,L,ids,rec);out=a.copy();orth=[];spec=[];costs=[];pred=[];marg=[]
 for idx,bit in zip(ids,bits):
  b=int(bit);A=a[idx];U=u[idx];sv=s[idx];V=v[idx];x,z=U[1,0],U[2,0];Up=U.copy()
  if mode=='givens':
   rad=np.hypot(x,z);theta=np.arctan2(abs(z),abs(x));beta=np.arcsin(rho/np.sqrt(2));target=min(theta,np.pi/4-beta) if b else max(theta,np.pi/4+beta);tx=np.copysign(rad*np.cos(target),x);tz=np.copysign(rad*np.sin(target),z);phi=np.arctan2(x*tz-z*tx,x*tx+z*tz);co,si=np.cos(phi),np.sin(phi);G=np.eye(4);G[1:3,1:3]=[[co,-si],[si,co]];Up=G@U;B=G@A;pred.append(abs(np.linalg.norm(B-A)**2-4*np.sin(phi/2)**2*np.sum(A[1:3]**2)))
  else:
   av=(abs(x)+abs(z))/2;T=rho
   if mode.startswith('luo'):
    if (b==1 and abs(x-z)<T) or (b==0 and abs(x-z)>T):Up[1,0]=np.sign(x)*(av+(2*b-1)*T/2);Up[2,0]=np.sign(z)*(av-(2*b-1)*T/2)
   elif (2*b-1)*(abs(x)-abs(z))<T:
    tx=max(T,av+T/2) if b else max(0,av-T/2);tz=max(0,av-T/2) if b else max(T,av+T/2);Up[1,0]=np.copysign(tx,x);Up[2,0]=np.copysign(tz,z)
   B=(Up*sv)@V
  out[idx]=B;orth.append(np.linalg.norm(Up.T@Up-np.eye(4)));spec.append(rel(sv,la.svdvals(B)));costs.append(np.linalg.norm(B-A)**2);marg.append((2*b-1)*(abs(Up[1,0])-abs(Up[2,0])))
 res=rgb(iw((iw((unblocks(out),*b2[1:]),False),*b1[1:]),False),c);info=(ids,bits,mode,rec);log=dict(orth_max=max(orth),local_spectrum_max=max(spec),rotation_energy_residual=max(pred,default=0),cost_sum=sum(costs),min_signed_margin=min(marg),dwt_inverse_rel=rel(y,iw((iw(b2,False),*b1[1:]),False)),payload_ll_delta=np.linalg.norm(wave(color(res)[0])[0]-wave(y)[0]),signature_psnr=psnr(im,res));return res,info,log
def sigdecode(im,info):
 ids,bits,mode,rec=info;a=blocks(dwt(color(im)[0])[1][0]);idx=np.argsort(-a.std(axis=(1,2)),kind='stable')[:len(ids)] if mode=='luo_reselect' else ids;u,_,_=np.linalg.svd(a[idx]);x,z=u[:,1,0],u[:,2,0];return ((x<=z) if mode.startswith('luo') else (abs(x)>abs(z))).astype(np.uint8),np.mean(idx==ids)
def attack(im,name):
 if name=='clean':return im
 if name.startswith('jpeg'):
  f=io.BytesIO();Image.fromarray(im).save(f,format='JPEG',quality=int(name[4:]));f.seek(0);return np.array(Image.open(f).convert('RGB'))
 if name.startswith('noise'):return np.rint(np.clip(im.astype(float)+np.random.default_rng(1234).normal(0,float(name[5:]),im.shape),0,255)).astype(np.uint8)
 if name=='blur':return np.rint(nd.gaussian_filter(im.astype(float),(.8,.8,0))).astype(np.uint8)
 if name=='resize':return np.array(Image.fromarray(im).resize((256,256),Image.Resampling.BICUBIC).resize((512,512),Image.Resampling.BICUBIC))
 if name=='crop':
  a=im.copy();a[:20]=a[-20:]=0;a[:,:20]=a[:,-20:]=0;return a
 if name=='rotate':return np.array(Image.fromarray(im).rotate(5,resample=Image.Resampling.BILINEAR))
 if name=='gamma':return np.rint(255*(im.astype(float)/255)**.8).astype(np.uint8)
 raise ValueError(name)
def write(name,rows):
 if rows:
  with open(OUT/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def gray_watermark(im):
 if 'A' in im.getbands():
  im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im.convert('RGBA'))
 return im.convert('L')
def load(path):
 z=zipfile.ZipFile(path);hosts={Path(n).stem:np.array(Image.open(io.BytesIO(z.read(n))).convert('RGB')) for n in sorted(z.namelist()) if n.startswith('host/') and not n.endswith('/')};wm={Path(n).stem:gray_watermark(Image.open(io.BytesIO(z.read(n)))) for n in sorted(z.namelist()) if n.startswith('watermark/') and not n.endswith('/')};return hosts,wm
