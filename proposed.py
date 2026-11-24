"""One proposed algorithm, unchanged IWT–Schur–SVD payload and FSDS/Givens/BCH layer."""
from run import *
from fastbch import fast
from dataclasses import dataclass
bch.decode=fast

@dataclass(frozen=True)
class Config:
    alpha: float
    cap: float=.4
    tau: float=40.
    code: str='bch'


def embed_image(host,w,key,record,conf,prepared=None):
    if not isinstance(key,bytes) or len(key)<16: raise ValueError('Use a secret key of at least 16 bytes')
    base,k,checks=embed(host,w,conf.alpha,prep(host) if prepared is None else prepared,True)
    im,ids,_,_=embed_sig(geometry(base),k,record,conf.code,conf.cap,conf.tau,key)
    return im,dict(k=k,record=record,ids=ids,conf=conf),base,checks


def authenticate(im,meta,key):
    k,rec,ids,conf=(meta[x] for x in ('k','record','ids','conf'))
    want=expected(k,rec,ids,conf.code,conf.cap,conf.tau,key)
    # Every received image, including clean, evaluates all 21 hypotheses.
    angle,_,_,count=verify_grid(im,ids,conf.code,want)
    return angle is not None,angle,count


def recover(im,meta,key):
    valid,angle,count=authenticate(im,meta,key)
    if not valid: return None,False,angle
    # The DOCX extractor is unchanged. Geometric correction is used only by
    # the authentication layer, not silently given to the payload extractor.
    return extract(im,meta['k']),True,angle


def save_meta(path,meta):
    u,v,s,a,n=meta['k'];c=meta['conf'];np.savez_compressed(path,U=u,Vt=v,S11=s,alpha=a,n=n,ids=meta['ids'],record=meta['record'],cap=c.cap,tau=c.tau,code=c.code)


def load_meta(path):
    with np.load(path,allow_pickle=False) as d:
        n=int(d['n']);a=float(d['alpha']);u,v,s=d['U'],d['Vt'],d['S11'];ids=d['ids'];cap,tau=float(d['cap']),float(d['tau']);code=str(d['code'])
        if n not in (64,128,256) or u.shape!=(256,256) or v.shape!=(256,256) or s.shape!=(n,n) or not all(np.isfinite(x).all() for x in (u,v,s)) or not np.isfinite(a) or a<=0: raise ValueError('Invalid metadata')
        if code!='bch' or ids.shape!=(255,) or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=255 or np.any((ids<0)|(ids>=1024)) or not 0<cap<=1 or not 1<tau<=1000: raise ValueError('Invalid signature configuration')
        return dict(k=(u,v,s,a,n),record=str(d['record']),ids=ids,conf=Config(a,cap,tau,code))
