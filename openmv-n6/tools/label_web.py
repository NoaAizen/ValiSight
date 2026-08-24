#!/usr/bin/env python3
"""Browser-based manual person labeling for a recorded session.

The human replaces the yolov10n teacher for one session: draw person boxes on
the RGB video frames, and manual_to_teacher.py turns them into a
<sess>_teacher.jsonl that the normal D1 chain (build_dataset -> export)
consumes unchanged - manual labels ARE a teacher, one with conf 1.0.

Labeling every frame is wasted effort: consecutive frames are near-duplicates
and the tracker links boxes across gaps up to 8 frames (MAX_GAP), so the
default stride of 3 keeps tracks intact at a third of the work. Frames you
navigate to and leave WITHOUT boxes are saved as explicitly empty - that is a
statement ("no person here"), not a skip; frames never visited stay unknown
and are simply absent from the output.

If the machine teacher already ran on this session its boxes are shown dashed
as suggestions - 'a' accepts them onto the frame for correction, which is much
faster than drawing from scratch.

Usage:
    ./label_web.py ../captures/TEST11 [--port 8090] [--stride 3]
then open http://localhost:8090 (or the Jetson's address from another machine).

Keys: ->/Space save+next   <- save+prev   a accept teacher boxes
      c copy from last labeled frame      x clear frame (explicit empty)
      Delete remove selected box          g go to frame
      u jump to next unlabeled frame
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
from perception.dataset import LiveSession   # noqa: E402

AUTOLABEL = os.path.join(ROOT, 'perception', 'out', 'autolabel')


def extract_frames(sess, cache_dir):
    """One JPEG per video frame, once - VideoCapture seeking is not trusted
    (the repaired elementary streams have no index), so random access in the
    browser needs the frames on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    have = len([f for f in os.listdir(cache_dir) if f.endswith('.jpg')])
    if have >= min(len(sess.frames), 1):
        # a previous run got at least this far; re-extract only if empty
        if have >= len(sess.frames) - 32:   # tolerate a lost video tail
            return have
    cap = sess._open_video()
    n = 0
    try:
        for meta in sess.frames:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(cache_dir, '%05d.jpg' % meta['i']),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
            if n % 200 == 0:
                print('  extracted %d/%d' % (n, len(sess.frames)),
                      file=sys.stderr)
    finally:
        cap.release()
    return n


def load_teacher(sess_name):
    # Suggestions must come from the MACHINE teacher. After manual_to_teacher
    # has run on a session, <sess>_teacher.jsonl IS the manual labels (moved
    # machine file: <sess>.teacher-machine.jsonl) - feeding those back as
    # "suggestions" shows nothing on any frame the human has not labeled yet.
    path = os.path.join(AUTOLABEL, sess_name + '.teacher-machine.jsonl')
    if not os.path.exists(path):
        path = os.path.join(AUTOLABEL, sess_name + '_teacher.jsonl')
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            boxes = [{'x': d['x'], 'y': d['y'], 'w': d['w'], 'h': d['h']}
                     for d in r['dets'] if d['cls'] == 'person']
            if boxes:
                out[r['i']] = boxes
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>label %(name)s</title><style>
 body { background:#151515; color:#ddd; font:14px monospace; margin:12px; }
 #bar { margin-bottom:8px; }
 #bar b { color:#7f7; }
 canvas { border:1px solid #444; cursor:crosshair; }
 #help { color:#888; margin-top:6px; }
 #nav { margin-top:8px; display:flex; gap:10px; }
 #nav button { background:#2a2a2a; color:#ddd; border:1px solid #555;
   border-radius:6px; padding:10px 22px; font:bold 15px monospace;
   cursor:pointer; }
 #nav button:hover { background:#3a3a3a; }
 #nav #bnext { background:#1d4a1d; border-color:#4a8a4a; }
</style></head><body>
<div id="bar">frame <b id="fi">0</b>/%(last)d · labeled <b id="nl">0</b>
 · <span id="src"></span></div>
<canvas id="cv" width="%(w)d" height="%(h)d"></canvas>
<div id="nav">
 <button id="bprev">&#9664; prev</button>
 <button id="baccept">accept (a)</button>
 <button id="bempty">empty (x)</button>
 <button id="bunlab">unlabeled (u)</button>
 <button id="bnext">next &#9654;</button>
</div>
<div id="help">drag=box · click=select · Del=remove · a=accept teacher ·
 c=copy prev · x=clear(empty) · &larr;/&rarr;/Space=nav (stride %(stride)d) ·
 g=goto · u=next unlabeled · b=brighten (display only)</div>
<script>
const W=%(w)d, H=%(h)d, LAST=%(last)d, STRIDE=%(stride)d;
let i=%(start)d, boxes=[], teacher=[], sel=-1, dirty=false, drag=null;
const cv=document.getElementById('cv'), cx=cv.getContext('2d');
// The dev-server sends no cache validators, so the browser re-downloads every
// frame it re-visits; holding Image objects ourselves makes prefetch and
// back-navigation instant on slow links. Window of 30 bounds memory.
let img=new Image();
const icache={}, iorder=[];
function fimg(n){
  if(n<0||n>LAST) return null;
  if(!icache[n]){ const im=new Image(); im.src='/frame/'+n;
    icache[n]=im; iorder.push(n);
    if(iorder.length>30) delete icache[iorder.shift()]; }
  return icache[n];
}

// Display-only brightening for dark sessions: applied to the image draw and
// never to the pixels on disk or the box coordinates, so what is saved is
// identical whichever level the operator was looking through.
const GAINS=['none','brightness(2.6)','brightness(5) contrast(1.3)'];
let gain=0;
function draw(){
  cx.filter=GAINS[gain];
  cx.drawImage(img,0,0);
  cx.filter='none';
  cx.setLineDash([6,4]); cx.strokeStyle='#fc0'; cx.lineWidth=1;
  for(const b of teacher) cx.strokeRect(b.x,b.y,b.w,b.h);
  cx.setLineDash([]); cx.lineWidth=2;
  boxes.forEach((b,k)=>{ cx.strokeStyle = k===sel ? '#f55' : '#5f5';
    cx.strokeRect(b.x,b.y,b.w,b.h); });
  if(drag){ cx.strokeStyle='#5ff';
    cx.strokeRect(drag.x0,drag.y0,drag.x1-drag.x0,drag.y1-drag.y0); }
}
async function load(n){
  i=Math.max(0,Math.min(LAST,n)); sel=-1; drag=null;
  const r=await (await fetch('/boxes/'+i)).json();
  boxes=r.manual!==null ? r.manual : [];
  teacher=r.teacher; dirty=(r.manual===null);
  document.getElementById('fi').textContent=i;
  document.getElementById('nl').textContent=r.n_labeled;
  document.getElementById('src').textContent =
    r.manual!==null ? (r.manual.length?'manual':'manual: empty')
                    : (teacher.length?'teacher suggestion (a=accept)':'unlabeled');
  img=fimg(i);
  if(img.complete) draw(); else img.onload=draw;
  fimg(i+STRIDE); fimg(i+2*STRIDE);   // prefetch ahead in the nav direction
}
// save and load talk to independent endpoints and save snapshots its payload
// synchronously, so navigation need not wait a full round-trip for the save -
// on a high-latency link that wait was most of the felt slowness.
async function nav(n){ const s=save(); load(n); await s; }
async function save(){
  // A failed save must STOP the operator, loudly - silently losing an hour
  // of labeling to a dead server is the worst failure this tool can have.
  try{
    const r=await fetch('/boxes/'+i,{method:'POST',body:JSON.stringify({boxes:boxes})});
    if(!r.ok) throw new Error('HTTP '+r.status);
    dirty=false;
  }catch(e){
    alert('SAVE FAILED - the label server is not answering.\\n'+
          'Your boxes for this frame are NOT stored. Stop labeling,\\n'+
          'check the server, then reload the page.');
    throw e;
  }
}
function pos(e){ const r=cv.getBoundingClientRect();
  return [Math.round(e.clientX-r.left), Math.round(e.clientY-r.top)]; }

cv.onmousedown=e=>{ const [x,y]=pos(e);
  sel=boxes.findIndex(b=>x>=b.x&&x<=b.x+b.w&&y>=b.y&&y<=b.y+b.h);
  if(sel<0) drag={x0:x,y0:y,x1:x,y1:y};
  draw(); };
cv.onmousemove=e=>{ if(!drag) return; const [x,y]=pos(e);
  drag.x1=x; drag.y1=y; draw(); };
cv.onmouseup=e=>{ if(!drag) return;
  const x=Math.min(drag.x0,drag.x1), y=Math.min(drag.y0,drag.y1),
        w=Math.abs(drag.x1-drag.x0), h=Math.abs(drag.y1-drag.y0);
  drag=null;
  if(w>8&&h>8){ boxes.push({x:x,y:y,w:w,h:h}); dirty=true; }
  draw(); };

document.onkeydown=async e=>{
  // e.code names the physical key, so shortcuts survive Hebrew layout and
  // caps lock (e.key would be the mapped character, e.g. 'a' -> 'ש')
  const k=(e.code&&e.code.startsWith('Key'))?e.code.slice(3).toLowerCase():e.key;
  if(e.key==='ArrowRight'||e.key===' '||e.code==='Space'){ e.preventDefault();
    await nav(i+STRIDE); }
  else if(e.key==='ArrowLeft'){ e.preventDefault();
    await nav(i-STRIDE); }
  else if(e.key==='Delete'||e.key==='Backspace'){
    if(sel>=0){ boxes.splice(sel,1); sel=-1; dirty=true; draw(); } }
  else if(k==='a'){ boxes=boxes.concat(
      teacher.map(b=>({x:b.x,y:b.y,w:b.w,h:b.h}))); teacher=[];
    dirty=true; draw(); }
  else if(k==='x'){ boxes=[]; sel=-1; dirty=true; draw(); }
  else if(k==='b'){ gain=(gain+1)%%GAINS.length; draw(); }
  else if(k==='c'){ const r=await (await fetch('/prev/'+i)).json();
    if(r.boxes){ boxes=r.boxes; dirty=true; draw(); } }
  else if(k==='g'){ const n=prompt('frame:'); if(n!==null){
    await nav(parseInt(n)||0); } }
  else if(k==='u'){ await gounlabeled(); }
};
async function gounlabeled(){
  // save first so the current frame counts as labeled before the search
  await save();
  const r=await (await fetch('/unlabeled/'+i)).json();
  if(r.i===null) alert('all frames labeled - session done!');
  else await load(r.i);
}
// Mouse buttons for hosts that swallow the keyboard (VSCode's simple
// browser, some remote-desktop setups): same save+navigate paths as the keys.
document.getElementById('bnext').onclick=()=>nav(i+STRIDE);
document.getElementById('bprev').onclick=()=>nav(i-STRIDE);
document.getElementById('baccept').onclick=()=>{ boxes=boxes.concat(
  teacher.map(b=>({x:b.x,y:b.y,w:b.w,h:b.h}))); teacher=[]; dirty=true; draw(); };
document.getElementById('bempty').onclick=()=>{ boxes=[]; sel=-1; dirty=true; draw(); };
document.getElementById('bunlab').onclick=()=>gounlabeled();
load(i);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    # filled in main()
    ctx = None
    # ThreadingHTTPServer handles saves concurrently; without the lock two
    # rapid navigations both write store+'.tmp' and the loser's os.replace
    # finds it already gone
    save_lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if ctype == 'image/jpeg':
            # extracted frames never change; without this the browser
            # re-downloads every frame it re-visits
            self.send_header('Cache-Control', 'max-age=86400')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        c = self.ctx
        if self.path == '/':
            page = PAGE % {'name': c['name'], 'w': c['w'], 'h': c['h'],
                           'last': c['last'], 'stride': c['stride'],
                           'start': c['start']}
            return self._send(200, page.encode(), 'text/html; charset=utf-8')
        if self.path.startswith('/frame/'):
            p = os.path.join(c['cache'], '%05d.jpg' % int(self.path[7:]))
            if not os.path.exists(p):
                return self._send(404, b'{}')
            return self._send(200, open(p, 'rb').read(), 'image/jpeg')
        if self.path.startswith('/boxes/'):
            i = int(self.path[7:])
            return self._send(200, json.dumps({
                'manual': c['manual'].get(str(i)),
                'teacher': c['teacher'].get(i, []),
                'n_labeled': len(c['manual'])}).encode())
        if self.path.startswith('/unlabeled/'):
            i = int(self.path[11:])
            # next stride-slot with no manual entry ANYWHERE in [t, t+stride),
            # searching forward from i then wrapping; null when done. Slot
            # coverage (not exact-index) tolerates labels laid on a shifted
            # grid - a label at t+1 covers slot t, re-labeling t is waste.
            s = c['stride']

            def covered(t):
                return any(str(t + o) in c['manual'] for o in range(s))
            targets = range(0, c['last'] + 1, s)
            nxt = next((t for t in targets if t > i and not covered(t)),
                       next((t for t in targets if not covered(t)), None))
            return self._send(200, json.dumps({'i': nxt}).encode())
        if self.path.startswith('/prev/'):
            i = int(self.path[6:])
            prev = [int(k) for k in c['manual'] if int(k) < i]
            if not prev:
                return self._send(200, b'{"boxes": null}')
            return self._send(200, json.dumps(
                {'boxes': c['manual'][str(max(prev))]}).encode())
        self._send(404, b'{}')

    def do_POST(self):
        c = self.ctx
        if not self.path.startswith('/boxes/'):
            return self._send(404, b'{}')
        i = int(self.path[7:])
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n))
        boxes = [{'x': int(b['x']), 'y': int(b['y']),
                  'w': int(b['w']), 'h': int(b['h'])}
                 for b in body['boxes']]
        with self.save_lock:
            c['manual'][str(i)] = boxes
            tmp = c['store'] + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'stride': c['stride'], 'boxes': c['manual']}, f)
            os.replace(tmp, c['store'])
        self._send(200, b'{"ok": true}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--stride', type=int, default=3,
                    help='label every Nth frame (must stay <= 8, the tracker '
                         'MAX_GAP, or tracks break apart)')
    a = ap.parse_args()
    if a.stride > 8:
        sys.exit('stride > 8 breaks track linking (MAX_GAP) - refusing')

    sess = LiveSession(a.session)
    name = os.path.basename(os.path.normpath(a.session))
    cache = os.path.join(a.session, '.label_frames')
    print('extracting frames (first run only)...', file=sys.stderr)
    n = extract_frames(sess, cache)
    sample = cv2.imread(os.path.join(cache, '%05d.jpg' % sess.frames[0]['i']))
    if sample is None:
        sys.exit('no frames extracted - is the video readable?')

    store = os.path.join(a.session, 'manual_boxes.json')
    manual = {}
    if os.path.exists(store):
        manual = json.load(open(store)).get('boxes', {})
        print('resuming: %d frames already labeled' % len(manual),
              file=sys.stderr)
    labeled = sorted(int(k) for k in manual)
    start = labeled[-1] if labeled else 0

    Handler.ctx = {'name': name, 'cache': cache, 'store': store,
                   'manual': manual, 'teacher': load_teacher(name),
                   'w': sample.shape[1], 'h': sample.shape[0],
                   'last': n - 1, 'stride': a.stride, 'start': start}
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), Handler)
    print('labeling %s (%d frames) on http://localhost:%d' % (name, n, a.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n%d frames labeled -> %s' % (len(manual), store))


if __name__ == '__main__':
    main()
