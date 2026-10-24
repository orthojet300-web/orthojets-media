#!/usr/bin/env python3
"""Arabic voice: Piper ar_JO-kareem-medium, Fusha with full tashkeel, --length-scale 0.75 (owner-approved).
usage: python3 gen_audio.py EPDIR   (reads EPDIR/script.json, writes EPDIR/aud/*.wav + aud/segments.json)
The model is downloaded once into ~/.cache/orthojets_piper from GitHub (k2-fsa sherpa-onnx release; HF is blocked).
"""
import json, os, re, subprocess, sys, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waqf import waqf
CACHE = os.path.expanduser('~/.cache/orthojets_piper')
NAME = 'vits-piper-ar_JO-kareem-medium'
M = os.path.join(CACHE, NAME, 'ar_JO-kareem-medium.onnx')

def ensure_model():
    if os.path.exists(M): return
    os.makedirs(CACHE, exist_ok=True)
    url = f'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{NAME}.tar.bz2'
    subprocess.run(['curl', '-sSL', '-o', os.path.join(CACHE, 'm.tar.bz2'), url], check=True)
    subprocess.run(['tar', 'xjf', 'm.tar.bz2'], cwd=CACHE, check=True)
    try: import piper  # noqa
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages', 'piper-tts'], check=True)

def main():
    ep = os.path.abspath(sys.argv[1]); ensure_model()
    segs = json.load(open(os.path.join(ep, 'script.json')))
    os.makedirs(os.path.join(ep, 'aud'), exist_ok=True)
    out = []
    for i, (scene, text) in enumerate(segs):
        parts = [p.strip() for p in re.split(r'[،:]', text) if p.strip()]
        clauses = []
        for j, p in enumerate(parts):
            f = os.path.join(ep, 'aud', f's{i:02d}_{j}.wav')
            if not os.path.exists(f):
                subprocess.run([sys.executable, '-m', 'piper', '-m', M, '-c', M + '.json', '--length-scale', '0.75', '-f', f],
                               input=waqf(p).encode(), check=True, capture_output=True)
            w = wave.open(f); d = w.getnframes() / w.getframerate(); w.close()
            clauses.append({'file': f, 'text': p, 'dur': d})
        out.append({'scene': scene, 'clauses': clauses})
    json.dump(out, open(os.path.join(ep, 'aud', 'segments.json'), 'w'), ensure_ascii=False, indent=1)
    print(sum(c['dur'] for s in out for c in s['clauses']))

if __name__ == '__main__':
    main()
