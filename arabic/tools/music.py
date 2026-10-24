import numpy as np, soundfile as sf, json, os, sys
EP=os.path.abspath(sys.argv[1]) if len(sys.argv)>1 else '.'
SR=44100; T=json.load(open(os.path.join(EP,'timeline.json')))['total']; n=int(T*SR); t=np.arange(n)/SR
def hz(m): return 440*2**((m-69)/12)
prog=[[57,60,64,69],[53,57,60,65],[48,55,60,64],[55,59,62,67]]  # Am F C G
bar=4.0; out=np.zeros(n)
for i in range(int(T/bar)+1):
  ch=prog[i%4]; s=int(i*bar*SR); e=min(n,int((i+1)*bar*SR+1.5*SR))
  if s>=n: break
  tt=np.arange(e-s)/SR; env=np.minimum(1,tt/1.2)*np.exp(-np.maximum(0,tt-bar)/0.6)
  for m in ch:
    f=hz(m)
    for d in (-0.15,0.15):
      out[s:e]+=0.05*env*np.sin(2*np.pi*f*(1+d/100)*tt)
  # soft bell arpeggio (8ths)
  for k in range(4):
    st=s+int(k*1.0*SR); m=ch[(k*2)%4]+12; L=int(1.6*SR)
    if st+L>n: L=n-st
    if L<=0: continue
    tt2=np.arange(L)/SR; out[st:st+L]+=0.025*np.sin(2*np.pi*hz(m)*tt2)*np.exp(-tt2/0.5)*np.minimum(1,tt2/0.01)
# simple lowpass + reverb-ish
from scipy.signal import butter,lfilter
b,a=butter(2,2500/(SR/2)); out=lfilter(b,a,out)
rev=np.zeros_like(out)
for dl,g in [(0.031,.35),(0.047,.3),(0.071,.25),(0.113,.2)]: d=int(dl*SR); rev[d:]+=g*out[:-d]
out=out+rev
fade=np.minimum(1,np.minimum(t/1.5,(T-t)/2.0)); out*=fade
out/=np.max(np.abs(out))*1.1
sf.write(os.path.join(EP,'music.wav'),np.stack([out,out],1),SR)
