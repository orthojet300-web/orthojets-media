#!/usr/bin/env python3
"""Orthojets: add original, synthesized sound effects that follow the on-screen motion.

usage: python3 tools/add_sfx.py IN.mp4 OUT.mp4 [--skip A-B ...] [--gain DB] [--debug]

Why: YouTube Content ID blocked our Pixabay background tracks (2026-10-20). These effects are generated here
from sine waves and filtered noise, so no one can claim them.

How it works (fully automatic, any episode):
- Measures the frame-to-frame change of the picture (downscaled grey frames) and finds motion ONSETS:
  moments where something starts moving/appearing after a calmer moment.
- Big change (scene change / large block moving in)  -> soft "whoosh" (band-passed noise swell)
- Medium change (a text line, label or card appearing) -> gentle "pop" (short pitched sine blip)
- Small, sharp change (a check mark, a dot, a line)     -> light "tick"
- No drums or percussion loops (owner rule). Effects are quiet: a light layer under the picture.
- --skip A-B : time ranges (s) that already have their own sound (logo sting, outro). Several allowed.
The existing audio of IN (logo chime, outro sound) is kept and the effects are mixed on top.
"""
import subprocess, sys
import numpy as np

SR = 44100
FPS = 30

def frames_diff(path):
    w, h = 108, 192
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'fps={FPS},scale={w}:{h},format=gray',
                          '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    d = np.abs(np.diff(f, axis=0))
    # fraction of pixels that changed noticeably, plus mean change
    frac = (d > 12).mean(axis=(1, 2))
    return np.concatenate([[0], frac])

def onsets(sig, skips, dur):
    # smooth a little, then find rises
    k = np.array([0.25, 0.5, 0.25]); s = np.convolve(sig, k, 'same')
    ev = []; last = -10
    base = np.maximum.accumulate(np.zeros_like(s))
    for i in range(2, len(s) - 1):
        t = i / FPS
        if any(a <= t <= b for a, b in skips) or t < 0.15 or t > dur - 0.2: continue
        prev = s[max(0, i - 6):i - 1].mean()
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.003 and s[i] > prev * 2.0 + 0.0015 and t - last > 0.3:
            # find the start of the rise (a few frames earlier) so the sound leads slightly
            j = i
            while j > 1 and s[j - 1] < s[j] and i - j < 6: j -= 1
            peak = float(s[i])
            kind = 'whoosh' if peak > 0.09 else ('pop' if peak > 0.018 else 'tick')
            ev.append((j / FPS, kind, peak)); last = t
    return ev

# ---------------------------------------------------------------- synthesized sounds
rng = np.random.default_rng(3)
def env(n, a, r):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-np.maximum(0, t - a) / r)

def bandnoise(n, lo, hi):
    x = rng.normal(0, 1, n); X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0; y = np.fft.irfft(X, n); return y / (np.abs(y).max() + 1e-9)

def whoosh(strength):
    n = int(SR * 0.55); t = np.arange(n) / SR
    y = np.zeros(n)
    # sweeping band: render in chunks with moving band centre
    chunk = 1024
    for c in range(0, n, chunk):
        u = c / n; centre = 500 + 2600 * np.sin(np.pi * u) ** 1.5
        seg = bandnoise(chunk * 2, centre * 0.6, centre * 1.5)[:min(chunk, n - c)]
        y[c:c + len(seg)] = seg
    shape = np.sin(np.pi * np.clip(t / 0.55, 0, 1)) ** 2
    return y * shape * 0.32 * min(1.0, 0.6 + strength * 3)

def pop(pitch):
    n = int(SR * 0.16); t = np.arange(n) / SR
    f = pitch * (1 + 0.6 * np.exp(-t * 60))             # tiny downward glide = soft "bloop"
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * env(n, 0.004, 0.045) + 0.25 * np.sin(2 * ph) * env(n, 0.003, 0.025)
    return y * 0.30

def tick(pitch):
    n = int(SR * 0.09); t = np.arange(n) / SR
    y = np.sin(2 * np.pi * pitch * t) * env(n, 0.001, 0.018) + 0.3 * bandnoise(n, 3000, 7000) * env(n, 0.0005, 0.006)
    return y * 0.22

PENTA = [523.25, 587.33, 659.25, 783.99, 880.0]   # C major pentatonic: pops never clash

def build(ev, dur):
    out = np.zeros(int(SR * (dur + 1)))
    for k, (t, kind, peak) in enumerate(ev):
        if kind == 'whoosh': s = whoosh(peak)
        elif kind == 'pop': s = pop(PENTA[k % len(PENTA)])
        else: s = tick(PENTA[(k * 2) % len(PENTA)] * 2)
        i = int(t * SR); out[i:i + len(s)] += s[:len(out) - i]
    # gentle stereo spread + soft limiter
    out = np.tanh(out * 1.2) / 1.2
    return out

def main():
    a = sys.argv[1:]
    skips = []; gain = 0.0; debug = '--debug' in a
    while '--skip' in a:
        i = a.index('--skip'); s0, s1 = a[i + 1].split('-'); skips.append((float(s0), float(s1))); del a[i:i + 2]
    if '--gain' in a:
        i = a.index('--gain'); gain = float(a[i + 1]); del a[i:i + 2]
    a = [x for x in a if x != '--debug']
    src, dst = a[0], a[1]
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src],
                               capture_output=True, text=True, check=True).stdout)
    sig = frames_diff(src)
    ev = onsets(sig, skips, dur)
    if debug:
        for t, k, p in ev: print(f'{t:6.2f}s  {k:6s}  {p:.3f}')
    print(f'{len(ev)} effects:', {k: sum(1 for e in ev if e[1] == k) for k in ('whoosh', 'pop', 'tick')})
    y = build(ev, dur)
    st = np.stack([y * 0.95, y], 1)
    wav = dst + '.sfx.wav'
    from scipy.io import wavfile
    wavfile.write(wav, SR, (np.clip(st * 10 ** (gain / 20), -1, 1) * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-i', wav, '-filter_complex',
                    '[0:a]aresample=44100,aformat=channel_layouts=stereo[o];[1:a]volume=1.0[s];[o][s]amix=inputs=2:normalize=0:duration=first[a]',
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst],
                   check=True)
    import os; os.remove(wav)
    print(dst)

if __name__ == '__main__':
    main()
