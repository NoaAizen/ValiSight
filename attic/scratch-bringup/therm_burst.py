import base64, os, serial, sys, time
PORT, OUT = sys.argv[1], sys.argv[2]
PROMPT=b">>> "
def rp(s,t=120):
    b,t0=b"",time.time()
    while time.time()-t0<t:
        n=s.in_waiting
        if n:
            b+=s.read(n)
            if b.endswith(PROMPT): break
        else: time.sleep(0.02)
    return b
def run(s,l,t=120):
    s.write(b"\x03"); time.sleep(0.15); s.reset_input_buffer()
    s.write(l.encode()+b"\r\n"); r=rp(s,t).decode(errors="replace")
    o=r.split("\r\n",1)[-1]
    for x in (">>> ","... "):
        while o.endswith(x): o=o[:-len(x)]
    return o.strip()
def g(e): return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")'%e
def pull(s,p,d):
    ss=run(s,"import os; print(os.stat('%s')[6])"%p)
    try: size=int(ss.strip().splitlines()[-1])
    except: print("  stat fail",p); return
    data=b""
    for off in range(0,size,512):
        r=run(s,"import binascii; f=open('%s','rb'); f.seek(%d); print(binascii.b2a_base64(f.read(512)).decode().strip()); f.close()"%(p,off))
        ln=r.strip().splitlines()[-1] if r.strip() else ""
        try: data+=base64.b64decode(ln)
        except: print("  chunk fail"); return
    open(d,"wb").write(data); print("  %s (%d bytes)"%(os.path.basename(d),len(data)))
s=serial.Serial(PORT,115200,timeout=1); time.sleep(0.5)
s.write(b"\x03"); time.sleep(0.2); s.reset_input_buffer(); s.write(b"\r\n"); rp(s,3)
os.makedirs(OUT,exist_ok=True)
print("burst:", run(s, g(
 "import csi,time; t=csi.CSI(cid=csi.LEPTON); t.reset(); t.pixformat(csi.GRAYSCALE); "
 "t.framesize(csi.QQVGA); [t.snapshot() for _ in range(60)]; time.sleep_ms(1500); "
 "t.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242); time.sleep_ms(2500); "
 "[t.snapshot() for _ in range(30)]\\n"
 " for i in range(4):\\n"
 "  im=t.snapshot(); im.save('/flash/tb%d.jpg'%i, quality=95); time.sleep_ms(700)\\n"
 " print('saved 4')"), 180))
for i in range(4):
    pull(s, "/flash/tb%d.jpg"%i, os.path.join(OUT,"tb%d.jpg"%i))
s.close()
