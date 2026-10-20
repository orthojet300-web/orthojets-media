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

def _n(inst, v, p, t, d): inst.notes.append(pm.Note(int(np.clip(v, 1, 127)), int(p), float(t), float(t + d)))

def _melody(rng, inst, base, ch, t0, beats, beat, rhythm, vel, octave=24, length=1.6):
    pent = [0, 2, 4, 7, 9]; idx = int(rng.integers(1, 5))
    for r in rhythm:
        if r >= beats: continue
        idx = int(np.clip(idx + rng.choice([-2, -1, 1, 2]), 0, 9))
        note = base + octave + pent[idx % 5] + 12 * (idx // 5)
        if r == 0:
            note = min([c + octave + 12 * o for c in ch for o in (0, 1)], key=lambda c: abs(c - note))
        _n(inst, vel + rng.integers(-4, 5), note, t0 + r * beat, beat * length)

# Five deliberately different styles (owner: the first five sounded too alike in rhythm).
STYLES = {
    # 1  flowing piano arpeggios, 4/4, 8th notes
    1: dict(name='flowing piano', bpm=72, beats=4, key=0, prog=[0, 5, 3, 4], minor=False),
    # 2  slow waltz: piano "oom-pah-pah" 3/4 + cello line
    2: dict(name='gentle waltz', bpm=84, beats=3, key=7, prog=[0, 3, 4, 0, 5, 3, 1, 4], minor=False),
    # 3  light & bright: pizzicato strings + celesta, syncopated, faster
    3: dict(name='pizzicato & celesta', bpm=104, beats=4, key=2, prog=[0, 4, 5, 3], minor=False),
    # 4  ambient: warm pad + soft electric-piano bell notes, very slow, lots of space
    4: dict(name='ambient electric piano', bpm=58, beats=4, key=5, prog=[0, 3, 0, 4], minor=False),
    # 5  fingerpicked guitar 6/8 + strings, minor-flavoured (vi-based)
    5: dict(name='fingerpicked guitar', bpm=66, beats=6, key=9, prog=[5, 3, 0, 4], minor=True),
}

def compose(seed, dur):
    rng = np.random.default_rng(seed)
    st = STYLES[(seed - 1) % 5 + 1]
    bpm, beats, prog = st['bpm'], st['beats'], st['prog']
    beat = 60 / bpm
    if beats == 6: beat = 60 / bpm / 2           # 6/8: count eighths
    bar = beats * beat
    m = pm.PrettyMIDI(initial_tempo=bpm)
    base = 48 + st['key']
    nbars = int(np.ceil(dur / bar)) + 1
    style = (seed - 1) % 5 + 1
    I = lambda prog_no, name: pm.Instrument(program=prog_no, name=name)
    if style == 1:
        piano, pad = I(0, 'piano'), I(49, 'strings'); m.instruments += [piano, pad]
    elif style == 2:
        piano, cello = I(0, 'piano'), I(42, 'cello'); m.instruments += [piano, cello]
    elif style == 3:
        pizz, cel, bass = I(45, 'pizzicato'), I(8, 'celesta'), I(32, 'acoustic bass'); m.instruments += [pizz, cel, bass]
    elif style == 4:
        ep, pad = I(4, 'electric piano'), I(89, 'warm pad'); m.instruments += [ep, pad]
    else:
        gtr, strings = I(24, 'nylon guitar'), I(48, 'strings'); m.instruments += [gtr, strings]
    for b in range(nbars):
        t0 = b * bar; deg = prog[b % len(prog)]; ch = chord(base, deg)
        sec = (b // 8) % 3
        if style == 1:
            lh = [ch[0] - 12, ch[2] - 12, ch[1], ch[2] - 12, ch[0] - 12, ch[2] - 12, ch[3], ch[2] - 12]
            for i, n in enumerate(lh): _n(piano, 42 + (6 if i == 0 else 0) + rng.integers(-3, 4), n, t0 + i * beat / 2, beat * 1.6)
            if b >= 2:
                for n in ch[:3]: _n(pad, 32, n + 12, t0, bar + 0.2)
                _melody(rng, piano, base, ch, t0, beats, beat, [0, 1.5, 2, 3] if b % 2 else [0, 1, 2.5], 52)
        elif style == 2:
            _n(piano, 50, ch[0] - 12, t0, beat * 0.9)                         # oom
            for k in (1, 2):                                                    # pah pah
                for n in ch[1:3]: _n(piano, 36, n, t0 + k * beat, beat * 0.6)
            _n(cello, 46, ch[0] - 12, t0, bar * 0.98)
            if b >= 2 and b % 4 != 3:
                _melody(rng, piano, base, ch, t0, beats, beat, [0, 2] if b % 2 else [0, 1, 2], 54, length=1.2)
        elif style == 3:
            # syncopated pizzicato chords: 1, 2&, 4
            for r in (0, 1.5, 3):
                for n in ch[:3]: _n(pizz, 50 + rng.integers(-4, 4), n + 12, t0 + r * beat, beat * 0.4)
            _n(bass, 52, ch[0] - 12, t0, beat * 0.9); _n(bass, 44, ch[2] - 12, t0 + 2 * beat, beat * 0.9)
            if b >= 2:
                _melody(rng, cel, base, ch, t0, beats, beat, [0, 0.5, 1, 2, 2.5, 3.5] if sec == 1 else [0, 1, 2.5, 3], 48, octave=36, length=0.6)
        elif style == 4:
            for n in ch: _n(pad, 34, n, t0, bar + 0.4)
            hits = [0, 2.5] if b % 2 == 0 else [1, 3]
            for r in hits:
                if b >= 1: _n(ep, 46 + rng.integers(-4, 4), ch[int(rng.integers(0, 4))] + 24, t0 + r * beat, beat * 3)
            if b >= 3 and b % 2 == 1:
                _melody(rng, ep, base, ch, t0, beats, beat, [0.5, 2], 44, octave=36, length=2.5)
        else:
            # Travis-style 6/8 picking: bass on 1 and 4, arpeggio between
            pat = [ch[0] - 12, ch[2], ch[1] + 12, ch[2] - 12, ch[2], ch[1] + 12]
            for i, n in enumerate(pat): _n(gtr, 52 + (8 if i in (0, 3) else 0) + rng.integers(-4, 4), n, t0 + i * beat, beat * 2.2)
            if b >= 2:
                for n in ch[:3]: _n(strings, 30 if sec != 1 else 38, n + 12, t0, bar + 0.3)
            if b >= 4 and sec == 1:
                _melody(rng, strings, base, ch, t0, beats, beat, [0, 3], 46, length=2.6)
    tend = nbars * bar
    first = m.instruments[0]
    for n in chord(base, prog[0]): _n(first, 40, n, tend, bar * 1.5)
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
