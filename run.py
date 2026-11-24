from refine import *
import bch
ANGLES=np.arange(-5,5.01,.5).tolist()
ATT_EXT=ATT+['jpeg30','noise10','rot-3','rot2.5','rot-4.5','rot1.5']
def enc(bits,code):return bch.encode(bits) if code=='bch' else bits.copy() if code=='raw' else encode(bits,4)
def dec(s,code):return bch.decode(s) if code=='bch' else (s>0).astype(np.uint8) if code=='raw' else decode(s,4)
def expected(k,rec,ids,code,cap,rho,key=KEY):
 U,V,S,alpha,n=k;pack=lambda a:np.ascontiguousarray(a,dtype='<f8').tobytes()
 msg=b'IWT-Schur-SVD-FSDS-v3|'+pack(U)+pack(V)+pack(S)+json.dumps([alpha,n,rec,code,cap,rho]).encode()+np.asarray(ids,dtype='<i4').tobytes()
 return hmac.digest(key,msg,'sha256')[:16]
def embed_sig(g,k,rec,code,cap,rho,key=KEY,direct_update=False):
 y,c,b1,b2,a,x,z,r,th,gap,std,energy=g;L=len(enc(np.zeros(128,dtype=np.uint8),code));margin=np.maximum(.03,rho/np.maximum(np.linalg.svd(a,compute_uv=False)[:,0]*r,1e-12)) if rho>1 else np.full(len(a),rho);beta=np.arcsin(np.minimum(margin,.99)/np.sqrt(2));cost=4*np.sin((abs(th-np.pi/4)+beta)/2)**2*energy
 p=np.arange(len(a));rr,cc=p//32,p%32;valid=(r>=.25)&(gap>=40)&(margin<=.25)&(rr>=2)&(rr<30)&(cc>=2)&(cc<30)
 eligible=np.flatnonzero(valid);pool=eligible[np.argsort(cost[eligible],kind='stable')[:max(L,int(np.ceil(cap*len(eligible))))]];ids=pool[np.argsort(-std[pool],kind='stable')[:L]];ids=ids[np.random.default_rng(81023).permutation(L)]
 if len(ids)!=L:raise ValueError('capacity')
 bits=np.unpackbits(np.frombuffer(expected(k,rec,ids,code,cap,rho,key),dtype=np.uint8));coded=enc(bits,code);t=np.where(coded==1,np.minimum(th[ids],np.pi/4-beta[ids]),np.maximum(th[ids],np.pi/4+beta[ids]));tx=np.copysign(r[ids]*np.cos(t),x[ids]);tz=np.copysign(r[ids]*np.sin(t),z[ids]);phi=np.arctan2(x[ids]*tz-z[ids]*tx,x[ids]*tx+z[ids]*tz);co=np.cos(phi)[:,None];si=np.sin(phi)[:,None];out=a.copy();out[ids,1]=co*a[ids,1]-si*a[ids,2];out[ids,2]=si*a[ids,1]+co*a[ids,2];
 if direct_update:
  uu,ss,vv=np.linalg.svd(a[ids]);uu[:,1,0]=tx;uu[:,2,0]=tz;out[ids]=(uu*ss[:,None,:])@vv
 im=rgb(iw((iw((unblocks(out),*b2[1:]),False),*b1[1:]),False),c);return im,ids,bits,coded
def score_grid(im,ids,angles=ANGLES):
 y=Image.fromarray(color(im)[0].clip(0,255).astype(np.uint8));out=[]
 for angle in angles:
  z=np.asarray(y if angle==0 else y.rotate(angle,resample=Image.Resampling.BILINEAR),dtype=np.float64)
  ll=z.reshape(128,4,128,4).sum(axis=(1,3))/4
  u,_,_=np.linalg.svd(blocks(ll)[ids]);x,v=abs(u[:,1,0]),abs(u[:,2,0]);out.append((x-v)/np.maximum(np.hypot(x,v),1e-12))
 return np.stack(out)
def verify_grid(im,ids,code,want,coded=None):
 s=score_grid(im,ids);ok=[];raw=[]
 for a,z in zip(ANGLES,s):
  b=dec(z,code)
  if b is not None and hmac.compare_digest(np.packbits(b).tobytes(),want):ok.append(a)
  if coded is not None:raw.append(float(np.mean((z>0)!=coded)))
 chosen=min(ok,key=lambda a:(abs(a),a)) if ok else None
 return chosen,raw[ANGLES.index(0)] if raw else None,raw[ANGLES.index(chosen)] if raw and chosen is not None else None,len(ok)
def verify(im,k,rec,ids,key,code="bch",cap=.4,tau=40):
 """Full fixed-grid verification. Does not use original signature bits."""
 angle,_,_,_=verify_grid(im,ids,code,expected(k,rec,ids,code,cap,tau,key))
 return angle is not None,angle
def getattack(im,a):return rot(im,float(a[3:])) if a.startswith('rot') and a!='rotate' else attack(im,a)
