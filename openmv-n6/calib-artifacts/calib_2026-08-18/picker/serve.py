#!/usr/bin/env python3
"""Minimal pixel picker: one click per station on the reflector vertex. Saves picks.json."""
import http.server, json, os, urllib.parse, glob
ROOT=os.path.dirname(os.path.abspath(__file__)); BASE=os.path.dirname(ROOT)
FR=os.path.join(BASE,'station_frames'); PICKS=os.path.join(BASE,'picks.json')
def stations():
    st={}
    for p in sorted(glob.glob(FR+'/*.png')):
        k=os.path.basename(p).split('_')[0]; st.setdefault(k,[]).append(os.path.basename(p))
    return st
PAGE=open(os.path.join(ROOT,'page.html'),encoding='utf-8').read()
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        u=urllib.parse.urlparse(self.path)
        if u.path=='/':
            b=PAGE.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        elif u.path=='/stations':
            d={'stations':stations(),'picks':json.load(open(PICKS)) if os.path.exists(PICKS) else {}}
            b=json.dumps(d).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        elif u.path.startswith('/img/'):
            p=os.path.join(FR,os.path.basename(u.path))
            if not os.path.exists(p): self.send_error(404); return
            b=open(p,'rb').read(); self.send_response(200); self.send_header('Content-Type','image/png'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
        else: self.send_error(404)
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0')); d=json.loads(self.rfile.read(n))
        cur=json.load(open(PICKS)) if os.path.exists(PICKS) else {}
        cur.update(d); json.dump(cur,open(PICKS,'w'),indent=1)
        b=b'ok'; self.send_response(200); self.send_header('Content-Length','2'); self.end_headers(); self.wfile.write(b)
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(('0.0.0.0',8090),H).serve_forever()
