"""Primitive narrow-sense binary BCH(255,k), GF(2^8), polynomial 0x11d."""
import numpy as np
from functools import lru_cache
EXP=np.zeros(510,dtype=np.int64); LOG=np.full(256,-1,dtype=np.int64)
x=1
for i in range(255):
 EXP[i]=x; LOG[x]=i;x<<=1
 if x&256:x^=0x11d
EXP[255:]=EXP[:255]
def mul(a,b):return int(EXP[LOG[a]+LOG[b]]) if a and b else 0
def div(a,b):return int(EXP[(LOG[a]-LOG[b])%255]) if a else 0
@lru_cache(None)
def setup(t):
 roots=set()
 for i in range(1,2*t+1):
  j=i
  while j not in roots:roots.add(j);j=j*2%255
 p=[1]
 for j in sorted(roots):
  q=[0]*(len(p)+1)
  for i,a in enumerate(p):q[i]^=mul(a,int(EXP[j]));q[i+1]^=a
  p=q
 assert set(p)<= {0,1}
 g=sum(a<<i for i,a in enumerate(p));r=len(p)-1;k=255-r
 return g,r,k,EXP[(np.arange(1,2*t+1)[:,None]*np.arange(255))%255],EXP[(-np.arange(255)[None,:]*np.arange(t+1)[:,None])%255]
def encode(bits,t=18):
 g,r,k,_,_=setup(t);assert k>=len(bits)
 a=sum(int(b)<<i for i,b in enumerate(bits));c=a<<r;v=c
 while v.bit_length()>r:v^=g<<(v.bit_length()-1-r)
 c^=v;return np.array([(c>>i)&1 for i in range(255)],dtype=np.uint8)
def decode(scores,t=18):
 g,r,k,S,X=setup(t);v=np.asarray(scores>0,dtype=np.uint8);s=np.bitwise_xor.reduce(S[:,v.astype(bool)],axis=1)
 if np.any(s):
  C=[1]+[0]*(2*t);B=C.copy();L=0;m=1;b=1
  for n in range(2*t):
   d=int(s[n])
   for i in range(1,L+1):d^=mul(C[i],int(s[n-i]))
   if not d:m+=1;continue
   T=C.copy();f=div(d,b)
   for j in range(2*t+1-m):
    if B[j]:C[j+m]^=mul(f,B[j])
   if 2*L<=n:L=n+1-L;B=T;b=d;m=1
   else:m+=1
  if L>t:return None
  z=np.zeros(255,dtype=np.int64)
  for j in range(L+1):
   if C[j]:z^=EXP[LOG[C[j]]+LOG[X[j]]]
  loc=np.flatnonzero(z==0)
  if len(loc)!=L:return None
  v[loc]^=1
  if np.any(np.bitwise_xor.reduce(S[:,v.astype(bool)],axis=1)):return None
 if np.any(v[r+128:]):return None
 return v[r:r+128]
