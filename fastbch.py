import numpy as np
from numba import njit
import bch
@njit(cache=True)
def decode_fast(scores,S,X,EXP,LOG,r,t=18):
 v=(scores>0).astype(np.uint8);s=np.zeros(2*t,np.int64)
 for j in range(2*t):
  for i in range(255):
   if v[i]:s[j]^=S[j,i]
 if np.any(s):
  C=np.zeros(2*t+1,np.int64);C[0]=1;B=C.copy();L=0;m=1;b=1
  for n in range(2*t):
   d=s[n]
   for i in range(1,L+1):
    if C[i] and s[n-i]:d^=EXP[LOG[C[i]]+LOG[s[n-i]]]
   if not d:m+=1;continue
   T=C.copy();f=EXP[(LOG[d]-LOG[b])%255]
   for j in range(2*t+1-m):
    if B[j]:C[j+m]^=EXP[LOG[f]+LOG[B[j]]]
   if 2*L<=n:L=n+1-L;B=T;b=d;m=1
   else:m+=1
  if L>t:return np.zeros(0,np.uint8)
  cnt=0
  for i in range(255):
   z=0
   for j in range(L+1):
    if C[j]:z^=EXP[LOG[C[j]]+LOG[X[j,i]]]
   if z==0:cnt+=1;v[i]^=1
  if cnt!=L:return np.zeros(0,np.uint8)
  for j in range(2*t):
   ss=0
   for i in range(255):
    if v[i]:ss^=S[j,i]
   if ss:return np.zeros(0,np.uint8)
 for i in range(r+128,255):
  if v[i]:return np.zeros(0,np.uint8)
 return v[r:r+128]
def fast(scores):
 g,r,k,S,X=bch.setup(18);o=decode_fast(scores,S,X,bch.EXP,bch.LOG,r);return o if len(o) else None
