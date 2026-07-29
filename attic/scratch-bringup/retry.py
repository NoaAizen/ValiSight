import serial, sys, time
PORT=sys.argv[1]; PROMPT=b">>> "
def rp(s,t=45):
    b,t0=b"",time.time()
    while time.time()-t0<t:
        n=s.in_waiting
        if n:
            b+=s.read(n)
            if b.endswith(PROMPT): break
        else: time.sleep(0.05)
    return b
def run(s,l,t=45):
    s.write(b"\x03"); time.sleep(0.15); s.reset_input_buffer()
    s.write(l.encode()+b"\r\n"); r=rp(s,t).decode(errors="replace")
    o=r.split("\r\n",1)[-1]
    for x in (">>> ","... "):
        while o.endswith(x): o=o[:-len(x)]
    return o.strip().replace("\r\n"," | ")
def g(e): return 'exec("try:\\n %s\\nexcept Exception as e:\\n print(\'ERR\',e)")'%e
s=serial.Serial(PORT,115200,timeout=1); time.sleep(0.5)
s.write(b"\x03"); time.sleep(0.2); s.reset_input_buffer(); s.write(b"\r\n"); rp(s,3)
for attempt in range(1,5):
    r = run(s, g("import csi; c=csi.CSI(cid=csi.LEPTON); c.reset(); c.pixformat(csi.GRAYSCALE); "
                 "c.framesize(csi.QQVGA); img=c.snapshot(); print('OK %dx%d' % (img.width(), img.height()))"), 60)
    print("attempt %d: %s" % (attempt, r))
    time.sleep(1.5)
s.close()
