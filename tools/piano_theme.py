#!/usr/bin/env python3
"""Orthojets solo-piano themes, rendered with a real sampled Steinway (Splendid Grand Piano, public domain, AKAI).

usage: python3 tools/piano_theme.py SAMPLES_DIR SFZ_DATA_DIR OUTDIR

Owner brief (2026-10-21): clean sound with no hiss, PIANO, a beautiful, motivating melody that is crisp but not
harsh, flowing and fairly calm. So: one instrument, hand-written melodies (not random), full-precision rendering
(float32, no clipping), a gentle room reverb, gentle compression, no drums.
Samples: https://github.com/sfzinstruments/SplendidGrandPiano (Public Domain).
"""
import os, re, subprocess, sys
import numpy as np
import soundfile as sf

SR = 48000

# ------------------------------------------------------------------ sampler
class Piano:
    def __init__(self, sdir, ddir):
        self.layers = []                       # (lovel, hivel, {key: (file, keycenter)})
        for lo, hi, d in ((1, 52, 'PP'), (53, 84, 'MP'), (85, 127, 'MF')):
            mp = {}
            for line in open(os.path.join(ddir, f'sp_{d}.txt')):
                m = re.search(r'lokey=(\d+) hikey=(\d+) pitch_keycenter=(\d+).*sample=(.+)\.\$EXT', line)
                if not m: continue
                f = os.path.join(sdir, m[4].strip() + '.flac')
                if not os.path.exists(f): continue
                for k in range(int(m[1]), int(m[2]) + 1):
                    mp[k] = (f, int(m[3]))
            self.layers.append((lo, hi, mp))
        self.cache = {}

    def _load(self, f):
        if f not in self.cache:
            x, sr = sf.read(f, dtype='float32', always_2d=True)
            if sr != SR:
                n = int(len(x) * SR / sr); t = np.linspace(0, len(x) - 1, n)
                x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1)
            if x.shape[1] == 1: x = np.repeat(x, 2, 1)
            # trim pre-roll silence so attacks land on time
            a = np.abs(x).max(1); i0 = int(np.argmax(a > a.max() * 0.02)); x = x[max(0, i0 - 48):]
            self.cache[f] = x.astype(np.float32)
        return self.cache[f]

    def note(self, key, vel, length):
        mp = next(m for lo, hi, m in self.layers if lo <= vel <= hi)
        f, kc = mp[min(max(key, min(mp)), max(mp))]
        x = self._load(f)
        ratio = 2 ** ((key - kc) / 12)
        n = int(min(len(x) / ratio, (length + 0.6) * SR))
        t = np.arange(n) * ratio
        y = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1)
        rel = int(0.35 * SR); off = int(length * SR)          # damper: release after note-off
        if off < n:
            env = np.ones(n, np.float32); e = min(rel, n - off)
            env[off:off + e] = np.linspace(1, 0, e) ** 2; env[off + e:] = 0; y *= env[:, None]
        gain = (vel / 127) ** 1.6 * 0.9
        return y * gain

def reverb(x, wet=0.16):
    """small, clean room: a few diffuse echoes + gentle damping (no noise added)"""
    out = x.copy()
    for d, g in ((0.029, .35), (0.041, .3), (0.053, .26), (0.067, .22), (0.083, .18), (0.11, .14), (0.149, .1), (0.21, .07)):
        k = int(d * SR); tail = np.zeros_like(x); tail[k:] = x[:-k] * g; out += tail * wet * 2.2
    # soften the reverb's top end a little by a one-pole lowpass on the difference
    return out

# ------------------------------------------------------------------ hand-written themes
# Notation: melody = list of (bar, beat, midi, beats); harmony = per-bar chord (root midi, quality)
def chord_notes(root, q):
    iv = {'M': [0, 4, 7], 'm': [0, 3, 7], 'M7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10], 'sus': [0, 5, 7], 'add9': [0, 4, 7, 14]}[q]
    return [root + i for i in iv]

C3, D3, E3, F3, G3, A3, B3 = 48, 50, 52, 53, 55, 57, 59
C4, D4, E4, F4, G4, A4, B4 = 60, 62, 64, 65, 67, 69, 71
C5, D5, E5, F5, G5, A5, B5 = 72, 74, 76, 77, 79, 81, 83
C6, D6, E6 = 84, 86, 88

THEMES = {
  # Rise: C major, uplifting, 88 BPM. I - V/vii - vi - IV | I - V - IV - V
  'piano_1_rise': dict(bpm=88, chords=[(C3, 'add9'), (G3 - 12, 'M'), (A3 - 12, 'm7'), (F3 - 12, 'M7'),
                                       (C3, 'M'), (G3 - 12, 'sus'), (F3 - 12, 'M7'), (G3 - 12, 'M')],
       a=[(0, 0, E5, 1.5), (0, 1.5, D5, .5), (0, 2, C5, 1), (0, 3, D5, 1), (1, 0, B4, 2), (1, 2, G4, 2),
          (2, 0, C5, 1), (2, 1, E5, 1), (2, 2, A5, 1.5), (2, 3.5, G5, .5), (3, 0, F5, 1.5), (3, 1.5, E5, .5), (3, 2, D5, 2),
          (4, 0, E5, 1), (4, 1, G5, 1), (4, 2, C6, 2), (5, 0, B5, 1.5), (5, 1.5, A5, .5), (5, 2, G5, 2),
          (6, 0, A5, 1), (6, 1, G5, 1), (6, 2, F5, 1), (6, 3, E5, 1), (7, 0, D5, 3)]),
  # Clarity: G major, flowing, 82 BPM. vi - IV - I - V
  'piano_2_clarity': dict(bpm=82, chords=[(E3, 'm7'), (C3, 'M7'), (G3 - 12, 'add9'), (D3, 'sus'),
                                          (E3, 'm'), (C3, 'M'), (G3 - 12, 'M'), (D3, 'M')],
       a=[(0, 0, B4, 1), (0, 1, D5, 1), (0, 2, E5, 2), (1, 0, G5, 1.5), (1, 1.5, F5 + 1, .5), (1, 2, E5, 2),
          (2, 0, D5, 1), (2, 1, B4, 1), (2, 2, D5, 1), (2, 3, G5, 1), (3, 0, F5 + 1, 3),
          (4, 0, B4, 1), (4, 1, E5, 1), (4, 2, G5, 1.5), (4, 3.5, A5, .5), (5, 0, G5, 1), (5, 1, E5, 1), (5, 2, C5, 2),
          (6, 0, D5, 1), (6, 1, G5, 1), (6, 2, B5, 1.5), (6, 3.5, A5, .5), (7, 0, F5 + 1, 1), (7, 1, A5, 1), (7, 2, D5, 2)]),
  # Momentum: D major, a bit brighter and motivating, 92 BPM. I - vi - IV - V with a lift
  'piano_3_momentum': dict(bpm=92, chords=[(D3, 'M'), (B3 - 12, 'm7'), (G3 - 12, 'M7'), (A3 - 12, 'sus'),
                                           (D3, 'add9'), (F3 + 1 - 12, 'm'), (G3 - 12, 'M'), (A3 - 12, 'M')],
       a=[(0, 0, F5 + 1, .5), (0, .5, A5, .5), (0, 1, D6, 1.5), (0, 2.5, C6 + 1, .5), (0, 3, A5, 1),
          (1, 0, B5, 1.5), (1, 1.5, A5, .5), (1, 2, F5 + 1, 2),
          (2, 0, G5, 1), (2, 1, B5, 1), (2, 2, D6, 1), (2, 3, C6 + 1, 1), (3, 0, A5, 2), (3, 2, E5, 2),
          (4, 0, F5 + 1, .5), (4, .5, A5, .5), (4, 1, D6, 1), (4, 2, E6, 2), (5, 0, C6 + 1, 1.5), (5, 1.5, A5, .5), (5, 2, F5 + 1, 2),
          (6, 0, G5, 1), (6, 1, B5, 1), (6, 2, A5, 1), (6, 3, G5, 1), (7, 0, F5 + 1, 1), (7, 1, E5, 1), (7, 2, A5, 2)]),
}

def render(theme, dur, piano):
    bpm = theme['bpm']; beat = 60 / bpm; bar = 4 * beat
    nb = int(np.ceil(dur / bar)) + 2
    out = np.zeros((int((nb + 3) * bar * SR), 2), np.float32)
    def place(key, vel, start, length):
        y = piano.note(int(key), int(vel), length); i = int(start * SR); out[i:i + len(y)] += y[:len(out) - i]
    rng = np.random.default_rng(7)
    # form (bars): intro 2 (LH) | A 8 | A' 8 (melody + octave lift on 2nd half) | B 8 (fuller LH, melody 8va doubled) | outro
    for b in range(nb):
        root, q = theme['chords'][b % 8]; ch = chord_notes(root, q)
        t0 = b * bar
        sect = 'intro' if b < 2 else ('A' if b < 10 else ('A2' if b < 18 else ('B' if b < 26 else 'out')))
        dyn = {'intro': 46, 'A': 52, 'A2': 56, 'B': 62, 'out': 44}[sect]
        # left hand: flowing broken chord in 8ths, bass note held
        place(ch[0] - 12, dyn + 6, t0, bar * 0.98)
        lh = [ch[0], ch[1] + 12 if len(ch) > 3 else ch[2], ch[1], ch[2], ch[0] + 12, ch[2], ch[1], ch[2]]
        for i, n in enumerate(lh):
            place(n, dyn - 6 + rng.integers(-3, 4) + (4 if i % 4 == 0 else 0), t0 + i * beat / 2, beat * 1.2)
        # right hand melody
        if sect in ('A', 'A2', 'B'):
            for (mb, bt, key, ln) in theme['a']:
                if mb != (b - 2) % 8: continue
                v = dyn + 14 + rng.integers(-3, 3) + (4 if bt == 0 else 0)
                k = key + (12 if sect == 'A2' and (b - 2) % 8 >= 4 and key < 84 else 0)
                place(k, v, t0 + bt * beat, ln * beat * 0.98)
                if sect == 'B':                                   # octave doubling for the lift
                    place(k - 12, v - 12, t0 + bt * beat, ln * beat * 0.98)
    tend = nb * bar
    root, q = theme['chords'][0]
    for n in chord_notes(root, q) + [root - 12, chord_notes(root, q)[2] + 12]:
        place(n, 50, tend, bar * 2.5)
    out = reverb(out)
    out = out[:int((dur + 6) * SR)]
    return out

def main():
    sdir, ddir, odir = sys.argv[1:4]; os.makedirs(odir, exist_ok=True)
    piano = Piano(sdir, ddir)
    for name, th in THEMES.items():
        y = render(th, 76, piano)
        y /= max(1e-6, np.abs(y).max()) / 0.7                     # no clipping, plenty of headroom
        wav = os.path.join(odir, name + '.wav'); sf.write(wav, y, SR, subtype='FLOAT')
        mp3 = os.path.join(odir, name + '.mp3')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-af',
                        'highpass=f=35,lowpass=f=11000,acompressor=threshold=-18dB:ratio=1.8:attack=20:release=250,'
                        'afade=t=in:d=0.3,afade=t=out:st=77:d=5,loudnorm=I=-16:TP=-1.5:LRA=9',
                        '-ar', '44100', '-b:a', '256k', mp3], check=True)
        os.remove(wav); print(mp3)

if __name__ == '__main__':
    main()
