#!/usr/bin/env python3
"""Orthojets SIGNATURE theme candidates (TikTok + Instagram background), original compositions.

usage: python3 tools/compose_signature.py OUTDIR [--dur 80]

Brief from the owner's reference (analysed, not copied): warm and dark (energy mostly 80-1000 Hz, almost no highs),
no percussion, ~80 BPM with flowing eighth notes, C major / A minor colour, slow build to a peak around 50 s,
then settling. Melodies, progressions and arrangements here are our own.
"""
import os, subprocess, sys
import numpy as np
import pretty_midi as pm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose_ensemble import chord, add, accompaniment, melody_phrase, GM, SF2

SIGS = [
    dict(slug='sig_A_warm_keys', bpm=80, key=0, meter=4, prog=[5, 3, 0, 4, 5, 3, 1, 4],
         bed='warmpad', acc='piano', acc_pat='arp8', bass='contrabass', lead='cello', counter='horn', color='vibes'),
    dict(slug='sig_B_velvet_strings', bpm=78, key=0, meter=4, prog=[0, 5, 3, 4, 0, 5, 3, 3],
         bed='slowstr', acc='harp', acc_pat='arp8', bass='cello', lead='horn', counter='clarinet', color='celesta'),
    dict(slug='sig_C_evening_rhodes', bpm=82, key=0, meter=4, prog=[5, 1, 3, 4, 5, 1, 3, 0],
         bed='warmpad', acc='epiano', acc_pat='arp8', bass='fretless', lead='clarinet', counter='cello', color='vibes'),
]

def energy(t, dur):
    """slow build to a peak around 50 s, then settle (0.55 .. 1.0)"""
    peak = min(50.0, dur * 0.65)
    if t <= peak: return 0.55 + 0.45 * (t / peak) ** 1.4
    return 1.0 - 0.35 * min(1.0, (t - peak) / max(1.0, dur - peak))

def compose(pc, dur, seed):
    rng = np.random.default_rng(seed)
    beat = 60 / pc['bpm']; bar = 4 * beat; base = 45 + pc['key'] + 3   # low register: centred around C3
    m = pm.PrettyMIDI(initial_tempo=pc['bpm'])
    I = {r: pm.Instrument(program=GM[pc[r]], name=r) for r in ('bed', 'acc', 'bass', 'lead', 'counter', 'color')}
    m.instruments += list(I.values())
    nbars = int(np.ceil(dur / bar)) + 1
    phrases = {}
    for b in range(nbars):
        t0 = b * bar; e = energy(t0, dur); prog = pc['prog']; ch = chord(base, prog[b % len(prog)])
        for n in ch[:3]: add(I['bed'], 34 * e, n, t0, bar + 0.3)
        accompaniment(I['acc'], 'arp8', [n - 12 if pc['acc'] != 'harp' else n for n in ch], t0, beat, 4, int(50 * e), rng)
        if b >= 2: add(I['bass'], 52 * e, ch[0] - 24 + (12 if pc['bass'] == 'cello' else 0), t0, bar * 0.95)
        if 4 <= b and (b % 8) != 7:
            if b % 2 == 0:
                k = ((b - 4) // 4) % 2
                if k not in phrases:
                    phrases[k] = melody_phrase(rng, base, [chord(base, prog[b % 8]), chord(base, prog[(b + 1) % 8])], 4, 12)
                for off, p, ln in phrases[k]: add(I['lead'], 56 * e, p, t0 + off * beat, ln * beat)
        if e > 0.8: add(I['counter'], 40 * e, ch[2], t0, bar * 0.95)
        if e > 0.85 and b % 2: add(I['color'], 30, ch[0] + 24, t0 + 2 * beat, beat * 1.5)
    tend = nbars * bar
    for n in chord(base, pc['prog'][0])[:3]: add(I['bed'], 30, n, tend, bar * 2); add(I['acc'], 36, n - 12, tend, bar * 1.5)
    return m

def main():
    a = sys.argv[1:]; dur = 80
    if '--dur' in a: i = a.index('--dur'); dur = float(a[i + 1]); del a[i:i + 2]
    out = a[0]; os.makedirs(out, exist_ok=True)
    for k, pc in enumerate(SIGS):
        mid, wav, mp3 = (os.path.join(out, f"{pc['slug']}.{e}") for e in ('mid', 'wav', 'mp3'))
        compose(pc, dur, 300 + k).write(mid)
        subprocess.run(['fluidsynth', '-ni', '-g', '0.5', '-r', '44100', '-R', '1', '-C', '1', '-F', wav, SF2, mid], check=True, capture_output=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-af',
                        f'atrim=0:{dur + 5},lowpass=f=4500,afade=t=in:d=1,afade=t=out:st={dur}:d=5,loudnorm=I=-16:TP=-1.5',
                        '-ar', '44100', '-b:a', '192k', mp3], check=True)
        os.remove(wav); os.remove(mid); print(mp3)

if __name__ == '__main__':
    main()
