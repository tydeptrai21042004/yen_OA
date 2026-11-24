from core import *
import itertools
from functools import lru_cache
ATT=['clean','jpeg90','jpeg70','jpeg50','noise5','blur','resize','crop','rotate','gamma']
ALPHA={64:.12,128:.06,256:.03}; DEV=['airplane','baboon','house','pepper']
@lru_cache(None)
def codebook(m):
    bits=((np.arange(2**(m+1))[:,None] >> np.arange(m+1))&1).astype(np.int8)
    x=((np.arange(2**m)[:,None] >> np.arange(m))&1).astype(np.int8)
    return bits,(bits[:,:1]+bits[:,1:]@x.T)%2

def encode(bits,m):
    if not m:return bits.copy()
    q=m+1; padded=np.pad(bits,(0,(-len(bits))%q));v=padded.reshape(-1,q)@(2**np.arange(q));return codebook(m)[1][v].reshape(-1)

def decode(scores,m):
    if not m:return (scores>0).astype(np.uint8)
    bits,cw=codebook(m);s=np.clip(scores,-.12,.12).reshape(-1,2**m);v=np.argmax(s@(2*cw-1).T,axis=1);return bits[v].reshape(-1)[:128].astype(np.uint8)

def geometry(im):
    y,c=color(im);b1,b2=dwt(y);a=blocks(b2[0]);u,s,v=np.linalg.svd(a);x,z=u[:,1,0],u[:,2,0];r=np.hypot(x,z);theta=np.arctan2(abs(z),abs(x));gap=s[:,0]-s[:,1];std=a.std(axis=(1,2));energy=np.sum(a[:,1:3]**2,axis=(1,2));return y,c,b1,b2,a,x,z,r,theta,gap,std,energy

def rot(im,a):return np.array(Image.fromarray(im).rotate(a,resample=Image.Resampling.BILINEAR))

