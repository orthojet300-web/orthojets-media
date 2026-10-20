#!/usr/bin/env python3
"""Orthojets original background music (ours alone -> never claimed by YouTube Content ID).

usage: python3 tools/compose_music.py OUTDIR [--n 5] [--dur 80]

Composes calm, inspirational piano pieces (soft string pad underneath, no drums or percussion: owner rule)
from a seed, renders them with FluidSynth + the FluidR3 GM soundfont (MIT licence), and writes
OUTDIR/orthojets_theme_<k>.mp3. Each seed = a different key, progression, tempo and melody.
Requires: apt fluidsynth fluid-soundfont-gm, pip pretty_midi.
"""
import os, subprocess, sys
import numpy as np
import pretty_midi as pm

SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
# progressions as scale degrees in major (0=I ... 5=vi), each chord lasts one bar
PROGS = [
    [0, 5, 3, 4],          # I vi IV V
    [5, 3, 0, 4],          # vi IV I V
    [0, 4, 5, 3],          # I V vi IV
    [3, 0, 4, 5],          # IV I V vi
    [0, 3, 5, 4],          # I IV vi V
]
MAJOR = [0, 2, 4, 5, 7, 9, 11]

def chord(root_pc, deg, seventh=True):
    notes = [MAJOR[(deg + i) % 7] + 12 * ((deg + i) // 7) for i in (0, 2, 4)] + ([MAJOR[(deg + 6) % 7] + 12 * ((deg + 6) // 7)] if seventh else [])
    return [root_pc + n for n in notes]

def compose(seed, dur):
    rng = np.random.default_rng(seed)
    key = [0, 2, 5, 7, 9][seed % 5]                     # C, D, F, G, A major
    bpm = [66, 70, 72, 76, 68][seed % 5]
    prog = PROGS[seed % len(PROGS)]
    beat = 60 / bpm; bar = 4 * beat
    m = pm.PrettyMIDI(initial_tempo=bpm)
    piano = pm.Instrument(program=0, name='piano')        # Acoustic Grand
    pad = pm.Instrument(program=49, name='strings')       # Slow strings
    nbars = int(np.ceil(dur / bar)) + 1
    base = 48 + key                                       # C3 region
    motif = None
    for b in range(nbars):
        t0 = b * bar
        deg = prog[b % len(prog)]
        ch = chord(base, deg)
        section = (b // 8) % 3                            # A (sparse) / B (fuller) / A'
        # left hand: broken-chord arpeggio in 8ths (root, 5th, 10th, 5th ...)
        lh = [ch[0] - 12, ch[2] - 12, ch[1], ch[2] - 12, ch[0] - 12, ch[2] - 12, ch[3], ch[2] - 12]
        for i, n in enumerate(lh):
            v = int(rng.integers(38, 48)) + (6 if i == 0 else 0)
            piano.notes.append(pm.Note(v, n, t0 + i * beat / 2, t0 + i * beat / 2 + beat * 1.6))
        # pad: sustained chord, very soft, from bar 2 on
        if b >= 2:
            for n in ch[:3]:
                pad.notes.append(pm.Note(30 if section != 1 else 36, n + 12, t0, t0 + bar + 0.2))
        # right hand melody: a 1-bar motif from chord tones / pentatonic, repeated with variation
        if b >= 2 and not (section == 0 and b % 4 == 3):
            if motif is None or b % 4 == 0:
                rhythm = [0, 1.5, 2, 3] if rng.random() < 0.5 else [0, 1, 2.5, 3]
                steps = rng.choice([-2, -1, 0, 1, 2], size=len(rhythm))
                motif = (rhythm, steps)
            rhythm, steps = motif
            pent = [0, 2, 4, 7, 9]
            idx = int(rng.integers(1, 4))
            for r, st in zip(rhythm, steps):
                idx = int(np.clip(idx + st, 0, 9))
                pc = pent[idx % 5] + 12 * (idx // 5)
                note = base + 24 + pc
                # nudge to nearest chord tone on strong beats
                if r in (0, 2):
                    cands = [c + 24 + 12 * o for c in ch for o in (0, 1)]
                    note = min(cands, key=lambda c: abs(c - note))
                v = int(rng.integers(44, 56)) + (6 if section == 1 else 0)
                piano.notes.append(pm.Note(v, int(note), t0 + r * beat, t0 + r * beat + beat * 1.8))
    # final chord
    tend = nbars * bar
    for n in chord(base, prog[0]):
        piano.notes.append(pm.Note(40, n, tend, tend + bar * 1.5))
    m.instruments += [piano, pad]
    return m

def render(m, mid, wav):
    m.write(mid)
    subprocess.run(['fluidsynth', '-ni', '-g', '0.6', '-r', '44100', '-R', '1', '-C', '1', '-F', wav, SF2, mid],
                   check=True, capture_output=True)

def main():
    a = sys.argv[1:]; n, dur = 5, 80
    if '--n' in a: i = a.index('--n'); n = int(a[i + 1]); del a[i:i + 2]
    if '--dur' in a: i = a.index('--dur'); dur = float(a[i + 1]); del a[i:i + 2]
    out = a[0]; os.makedirs(out, exist_ok=True)
    for k in range(1, n + 1):
        m = compose(k, dur)
        mid, wav, mp3 = (os.path.join(out, f'orthojets_theme_{k}.{e}') for e in ('mid', 'wav', 'mp3'))
        render(m, mid, wav)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-af',
                        f'atrim=0:{dur + 6},afade=t=out:st={dur + 2}:d=4,loudnorm=I=-16:TP=-1.5',
                        '-ar', '44100', '-b:a', '192k', mp3], check=True)
        os.remove(wav); os.remove(mid); print(mp3)

if __name__ == '__main__':
    main()
