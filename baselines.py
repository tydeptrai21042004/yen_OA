"""Paper-derived grayscale algorithms, adapted to shared RGB luminance interface.
References and reproduction limits: REFERENCES.md. No ground truth at detection.
"""
from core import *


def prepare(im, method):
    y,c=color(im)
    if method=='liu_tan_2002': bands=(y,)
    else: bands=wave(y, method=='luo_2021')
    return y,c,[la.svd(x.astype(float),full_matrices=False) for x in bands]


def luo_tag(keys):
    # Luo (12),(13), XOR of all four pairs, then XOR first byte of B.
    # Serialize matrices as little-endian float64 C order (paper unspecified).
    v=MASK
    for uw,vw,_,_ in keys:
        for m in (uw,vw.T):
            v ^= hashlib.sha1(np.ascontiguousarray(m,dtype='<f8').tobytes()).digest()[0]
    return np.unpackbits(np.array([v],np.uint8))


def oam_embed(im,bits,T,corrected=False):
    y,c=color(im); b1,b2=dwt(y); a=blocks(b2[0]); u,s,v=np.linalg.svd(a)
    ids=np.argsort(-a.std(axis=(1,2)),kind='stable')[:8];out=a.copy()
    for i,bit in zip(ids,bits):
        x,z=u[i,1,0],u[i,2,0];p=u[i].copy(); av=(abs(x)+abs(z))/2; b=int(bit)
        if corrected:
            if (2*b-1)*(abs(x)-abs(z))<T:
                hi=max(T,av+T/2);lo=max(0,av-T/2)
                p[1,0]=np.copysign(hi if b else lo,x);p[2,0]=np.copysign(lo if b else hi,z)
        elif (b==1 and abs(x-z)<T) or (b==0 and abs(x-z)>T):
            p[1,0]=np.sign(x)*(av+(2*b-1)*T/2);p[2,0]=np.sign(z)*(av-(2*b-1)*T/2)
        out[i]=(p*s[i])@v[i]
    return rgb(iw((iw((unblocks(out),*b2[1:]),False),*b1[1:]),False),c),ids


def oam_read(im,ids=None,corrected=False):
    a=blocks(dwt(color(im)[0])[1][0]);idx=np.argsort(-a.std(axis=(1,2)),kind='stable')[:8] if ids is None else ids
    u,_,_=np.linalg.svd(a[idx]);x,z=u[:,1,0],u[:,2,0]
    return ((abs(x)>abs(z)) if corrected else (x<=z)).astype(np.uint8)


def embed_paper(im,w,method,params,prepared=None):
    y,c,ds=prepare(im,method) if prepared is None else prepared
    n=len(w);keys=[];out=[]
    if method=='ganic_eskicioglu_2005':
        uw,sw,vw=la.svd(w,full_matrices=False)
        for j,(u,s,v) in enumerate(ds):
            alpha=params['ll'] if j==0 else params['detail'];sp=s.copy();sp[:n]+=alpha*sw
            out.append((u*sp)@v);keys.append((uw,vw,s[:n].copy(),alpha))
        return rgb(iw(out,False),c),dict(method=method,keys=keys,n=n)
    for j,(u,s,v) in enumerate(ds):
        alpha=params['ll'] if j==0 else params['detail'];S=np.diag(s);B=S.copy();B[:n,:n]+=alpha*w
        uw,sw,vw=la.svd(B,full_matrices=False);out.append((u*sw)@v);keys.append((uw,vw,S[:n,:n].copy(),alpha))
    base=rgb(out[0] if method=='liu_tan_2002' else iw(tuple(np.rint(x).astype(np.int64) for x in out)),c)
    meta=dict(method=method,keys=keys,n=n)
    if method=='luo_2021':
        bits=luo_tag(keys);base,ids=oam_embed(base,bits,params['T']);meta['ids']=ids
    return base,meta


def extract_paper(im,meta):
    method=meta['method'];n=meta['n']; y=color(im)[0]
    bands=(y,) if method=='liu_tan_2002' else wave(y,method=='luo_2021');out=[]
    for a,(u,v,s,alpha) in zip(bands,meta['keys']):
        sp=la.svdvals(a.astype(float))
        if method=='ganic_eskicioglu_2005': out.append((u*((sp[:n]-s)/alpha))@v)
        else: out.append(((u[:n,:]*sp)@v[:,:n]-s)/alpha)
    return out


def verify_paper(im,meta,corrected=False):
    if meta['method']!='luo_2021': return None,None
    # Expected tag is recomputed from auxiliary matrices, never passed to extractor.
    got=oam_read(im,meta.get('ids') if corrected else None,corrected)
    ber=float(np.mean(got!=luo_tag(meta['keys'])))
    return int(ber==0),ber
