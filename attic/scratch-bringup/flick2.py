import io,time,urllib.request,numpy as np
from PIL import Image
urllib.request.urlopen("http://127.0.0.1:8080/switch?cam=thermal",timeout=10).read()
time.sleep(12)
r=urllib.request.urlopen("http://127.0.0.1:8080/stream.mjpg",timeout=40)
buf=b""; fr=[]; t0=time.time()
while time.time()-t0<40 and len(fr)<300:
    c=r.read(8192)
    if not c: break
    buf+=c
    while True:
        s=buf.find(b"\xff\xd8"); e=buf.find(b"\xff\xd9",s+2)
        if s<0 or e<0: break
        fr.append((time.time(),buf[s:e+2])); buf=buf[e+2:]
r.close()
urllib.request.urlopen("http://127.0.0.1:8080/switch?cam=rgb",timeout=10).read()
t=np.array([a for a,_ in fr]); t-=t[0]
im=[np.asarray(Image.open(io.BytesIO(j)).convert("L"),float) for _,j in fr]
m=np.array([i.mean() for i in im])
lo=np.array([np.percentile(i,1) for i in im]); hi=np.array([np.percentile(i,99) for i in im])
print("%d frames over %.1fs (%.2f fps)"%(len(fr),t[-1],len(fr)/t[-1]))
print("mean ptp %.1f DN | p1 ptp %.1f | p99 ptp %.1f"%(m.ptp(),lo.ptp(),hi.ptp()))
# spectrum of the mean series
x=m-m.mean(); fs=len(fr)/t[-1]
F=np.abs(np.fft.rfft(x*np.hanning(len(x)))); f=np.fft.rfftfreq(len(x),1/fs)
k=np.argsort(F)[::-1][:5]
print("\ndominant frequencies in the brightness series:")
for i in k:
    if f[i]>0: print("   %.3f Hz  (period %.2f s)  amplitude %.1f"%(f[i],1/f[i],F[i]/len(x)*2))
# sawtooth shape: rises vs falls
d=np.diff(m)
up=d[d>0]; dn=d[d<0]
print("\nrises: n=%d median %+.2f DN/frame | falls: n=%d median %+.2f DN/frame, worst %+.2f"%(
    len(up),np.median(up),len(dn),np.median(dn),dn.min()))
big=np.where(d<-4)[0]
print("drops >4 DN at t = %s"%np.round(t[big+1],2))
if len(big)>1: print("  spacing between drops: %s s"%np.round(np.diff(t[big+1]),2))
np.save("means.npy",m); np.save("t.npy",t)
