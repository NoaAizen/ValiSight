import io,sys,time,urllib.request,numpy as np
from PIL import Image
urllib.request.urlopen("http://127.0.0.1:8080/switch?cam=thermal",timeout=10).read()
print("switched to thermal, waiting for soft reset + init ...")
time.sleep(12)
r=urllib.request.urlopen("http://127.0.0.1:8080/stream.mjpg",timeout=30)
buf=b""; frames=[]; t0=time.time()
while time.time()-t0 < 25 and len(frames)<80:
    chunk=r.read(4096)
    if not chunk: break
    buf+=chunk
    while True:
        s=buf.find(b"\xff\xd8"); e=buf.find(b"\xff\xd9",s+2)
        if s<0 or e<0: break
        frames.append((time.time(), buf[s:e+2])); buf=buf[e+2:]
r.close()
print("captured %d frames in %.1fs"%(len(frames), frames[-1][0]-frames[0][0] if len(frames)>1 else 0))
if len(frames)>1:
    dt=np.diff([f[0] for f in frames])
    print("fps: mean %.2f  median %.2f  jitter(sd) %.3fs  min dt %.3f max dt %.3f"%(
        1/dt.mean(),1/np.median(dt),dt.std(),dt.min(),dt.max()))
imgs=[np.asarray(Image.open(io.BytesIO(j)).convert("L"),dtype=float) for _,j in frames]
m=np.array([i.mean() for i in imgs])
print("\nper-frame mean: min %.1f max %.1f  peak-to-peak %.2f DN  sd %.2f"%(m.min(),m.max(),m.ptp(),m.std()))
d=np.abs(np.diff(m))
print("frame-to-frame |Δmean|: median %.2f  p90 %.2f  max %.2f DN"%(
    np.median(d),np.percentile(d,90),d.max()))
print("\nfirst 25 frame means:", np.round(m[:25],1))
# affine test on the live stream
res=[];rawd=[]
for a,b in zip(imgs,imgs[1:]):
    g,o=np.polyfit(a.ravel(),b.ravel(),1)
    res.append(np.sqrt(np.mean((b-(g*a+o))**2))); rawd.append(np.sqrt(np.mean((b-a)**2)))
print("\nglobal-affine test  raw RMS diff %.2f -> residual after removing gain/offset %.2f  (%.0f%% of the change is a global remap)"%(
    np.mean(rawd),np.mean(res),100*(1-np.mean(res)/np.mean(rawd))))
urllib.request.urlopen("http://127.0.0.1:8080/switch?cam=rgb",timeout=10).read()
print("\nswitched back to rgb")
