#!/usr/bin/env python3
"""Render an Orthojets episode end-to-end.

usage: python3 tools/build_episode.py episodes/epNN_slug/script.py SLUG
output: out/SLUG_1080p.mp4 (upload copy, ~4.5 Mbps) and out/SLUG_light.mp4 (720p preview)

Body is rendered silent (channel rule); intro/outro keep their sound (add_brand.py).
Rendering runs in 2 parallel segments; each segment is a separate process so the
job survives long renders. Expect ~15-20 min for a 60 s body on 2 cores.
"""
import os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, **kw):
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

def main():
    script, slug = os.path.abspath(sys.argv[1]), sys.argv[2]
    wd = os.path.dirname(script)
    dur = float(subprocess.run([sys.executable, script, 'dur'], cwd=wd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    nf = int(round(dur * 30)); half = nf // 2
    out = os.path.join(ROOT, 'out'); os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=slug + '_')
    a, b = os.path.join(tmp, 'a.mp4'), os.path.join(tmp, 'b.mp4')
    p1 = subprocess.Popen([sys.executable, script, 'seg', '0', str(half), a], cwd=wd)
    p2 = subprocess.Popen([sys.executable, script, 'seg', str(half), str(nf), b], cwd=wd)
    if p1.wait() or p2.wait(): sys.exit('segment render failed')
    lst = os.path.join(tmp, 'l.txt'); open(lst, 'w').write(f"file '{a}'\nfile '{b}'\n")
    body = os.path.join(tmp, 'body.mp4')
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-shortest', body])
    full = os.path.join(tmp, 'full.mp4')
    run([sys.executable, os.path.join(ROOT, 'kit', 'add_brand.py'), body, full, '--light'])
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', full, '-vf', 'hqdn3d=1.5:1:2:2', '-c:v', 'libx264', '-preset', 'medium',
         '-b:v', '4.5M', '-maxrate', '6M', '-bufsize', '12M', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
         os.path.join(out, f'{slug}_1080p.mp4')])
    os.replace(os.path.join(tmp, 'full_light.mp4'), os.path.join(out, f'{slug}_light.mp4'))
    print('DONE', os.path.join(out, f'{slug}_1080p.mp4'))

if __name__ == '__main__':
    main()
