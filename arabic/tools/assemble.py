#!/usr/bin/env python3
"""Join the clause wavs with gaps (0.12 s clause, 0.34 s sentence, 0.4 s after the hook for the logo sting) ->
EPDIR/voice.wav + EPDIR/timeline.json (with hook_end). usage: python3 assemble.py EPDIR"""
import json, os, sys
import numpy as np, soundfile as sf
ep = os.path.abspath(sys.argv[1])
S = json.load(open(os.path.join(ep, 'aud', 'segments.json')))
SR = 22050; GAP_C = 0.12; GAP_S = 0.34; LEAD = 0.35
track = [np.zeros(int(LEAD * SR))]; t = LEAD; timeline = []; HOOK_END = None
for si, s in enumerate(S):
    st = t
    for c in s['clauses']:
        a, sr = sf.read(c['file']); idx = np.where(np.abs(a) > 0.008)[0]
        a = a[max(idx[0] - 200, 0):idx[-1] + 800]
        c['start'] = t; c['end'] = t + len(a) / SR
        track.append(a); t = c['end']
        track.append(np.zeros(int(GAP_C * SR))); t += GAP_C
    track.append(np.zeros(int((GAP_S - GAP_C) * SR))); t += GAP_S - GAP_C
    if si == 0:
        HOOK_END = s['clauses'][-1]['end'] + 0.15; track.append(np.zeros(int(0.4 * SR))); t += 0.4
    timeline.append({'scene': s['scene'], 'start': st, 'end': t, 'clauses': [{k: c[k] for k in ('text', 'start', 'end')} for c in s['clauses']]})
track.append(np.zeros(int(1.5 * SR))); t += 1.5
sf.write(os.path.join(ep, 'voice.wav'), np.concatenate(track), SR)
json.dump({'hook_end': HOOK_END, 'total': t, 'segs': timeline}, open(os.path.join(ep, 'timeline.json'), 'w'), ensure_ascii=False, indent=1)
print(t)
