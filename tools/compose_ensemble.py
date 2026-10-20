#!/usr/bin/env python3
"""Orthojets original ENSEMBLE themes: several instruments per piece, flowing and calm, no drums.

usage: python3 tools/compose_ensemble.py OUTDIR [--dur 80]

Each piece is arranged in layers that enter and leave smoothly:
  bed (sustained strings/pad) - accompaniment pattern (piano / harp / guitar / marimba) - bass line (cello / bass) -
  lead melody (flute / oboe / clarinet / violin / horn) - counter-melody - sparkle colour (celesta / glockenspiel / vibraphone)
Form: intro (bed + accompaniment) -> A (+bass +lead) -> B (+counter +colour, fuller) -> A' (lead returns, thinner) -> outro.
Our own compositions (FluidSynth + FluidR3 GM soundfont, MIT) -> safe from YouTube Content ID.
"""
import os, subprocess, sys
import numpy as np
import pretty_midi as pm

SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
MAJOR = [0, 2, 4, 5, 7, 9, 11]
GM = dict(piano=0, epiano=4, celesta=8, glock=9, vibes=11, marimba=12, harp=46, nylon=24, steel=25, abass=32, fretless=35,
          violin=40, cello=42, contrabass=43, strings=48, slowstr=49, synstr=50, choir=52, horn=60, oboe=68, clarinet=71,
          flute=73, recorder=74, warmpad=89, haloPad=94)

PIECES = [
    dict(slug='6_morning_rounds', key=7, bpm=76, meter=4, prog=[0, 4, 5, 3, 0, 4, 3, 4],
         bed='slowstr', acc='piano', acc_pat='arp8', bass='cello', lead='flute', counter='clarinet', color='celesta'),
    dict(slug='7_soft_horizon', key=2, bpm=70, meter=3, prog=[0, 5, 3, 4, 0, 5, 1, 4],
         bed='strings', acc='harp', acc_pat='waltz_arp', bass='contrabass', lead='oboe', counter='cello', color='glock'),
    dict(slug='8_quiet_focus', key=5, bpm=64, meter=4, prog=[5, 3, 0, 4, 5, 3, 0, 4],
         bed='warmpad', acc='epiano', acc_pat='broken', bass='fretless', lead='vibes', counter='slowstr', color='celesta'),
    dict(slug='9_open_road', key=0, bpm=84, meter=6, prog=[0, 3, 5, 4, 0, 3, 4, 0],
         bed='slowstr', acc='nylon', acc_pat='travis68', bass='abass', lead='recorder', counter='horn', color='marimba'),
    dict(slug='10_gentle_lift', key=9, bpm=72, meter=4, prog=[3, 4, 5, 0, 3, 4, 0, 0],
         bed='choir', acc='piano', acc_pat='pulse', bass='cello', lead='violin', counter='horn', color='harp'),
]

def chord(base, deg):
    return [base + MAJOR[(deg + i) % 7] + 12 * ((deg + i) // 7) for i in (0, 2, 4, 6)]

def add(inst, v, p, t, d):
    inst.notes.append(pm.Note(int(np.clip(v, 1, 127)), int(p), float(t), float(t + max(0.05, d))))

def accompaniment(inst, pat, ch, t0, beat, meter, vel, rng):
    lo = [n - 12 for n in ch]
    if pat == 'arp8':
        seq = [lo[0], lo[2], ch[0], ch[1], ch[2], ch[1], ch[0], lo[2]]
        for i, n in enumerate(seq): add(inst, vel + (5 if i == 0 else 0) + rng.integers(-3, 4), n, t0 + i * beat / 2, beat * 1.5)
    elif pat == 'waltz_arp':
        seq = [lo[0], lo[2], ch[0], ch[1], ch[2], ch[1]]
        for i, n in enumerate(seq): add(inst, vel + rng.integers(-3, 4), n + 12, t0 + i * beat / 2, beat * 1.8)
    elif pat == 'broken':
        for i, n in enumerate([ch[0], ch[2], ch[3], ch[1]]): add(inst, vel + rng.integers(-3, 4), n, t0 + i * beat, beat * 2.2)
    elif pat == 'travis68':
        seq = [lo[0], ch[1], ch[2], lo[2], ch[1], ch[3]]
        for i, n in enumerate(seq): add(inst, vel + (6 if i in (0, 3) else 0) + rng.integers(-3, 4), n, t0 + i * beat, beat * 2.4)
    elif pat == 'pulse':
        for k in range(meter):
            for n in ch[:3]: add(inst, vel - (6 if k % 2 else 0), n, t0 + k * beat, beat * 0.9)

def melody_phrase(rng, base, ch_list, meter, octave):
    """2-bar phrase: stepwise, long notes, ending on a chord tone. Returns [(bar_offset_beats, pitch, len_beats)]"""
    scale = [base + octave + MAJOR[i % 7] + 12 * (i // 7) for i in range(15)]
    rhythms = {4: [[0, 2, 3], [0, 1.5, 2, 3], [0, 3]], 3: [[0, 2], [0, 1, 2], [0]], 6: [[0, 3, 5], [0, 3], [0, 2, 3]]}[meter]
    out = []; pos = int(rng.integers(4, 9))
    for bi in range(2):
        r = rhythms[int(rng.integers(0, len(rhythms)))]
        for j, beat_pos in enumerate(r):
            pos = int(np.clip(pos + rng.choice([-1, -1, 1, 1, 2, -2, 0]), 0, 13))
            p = scale[pos]
            if beat_pos == 0:   # strong beat: snap to chord tone
                p = min([c + octave + 12 * o for c in ch_list[bi] for o in (0, 1)], key=lambda c: abs(c - p))
            nxt = r[j + 1] if j + 1 < len(r) else meter
            out.append((bi * meter + beat_pos, p, (nxt - beat_pos) * 0.95))
    return out

def compose(pc, dur, seed):
    rng = np.random.default_rng(seed)
    meter = pc['meter']; beat = 60 / pc['bpm'] / (2 if meter == 6 else 1)
    bar = meter * beat; base = 48 + pc['key']; prog = pc['prog']
    m = pm.PrettyMIDI(initial_tempo=pc['bpm'])
    I = {r: pm.Instrument(program=GM[pc[r]], name=r) for r in ('bed', 'acc', 'bass', 'lead', 'counter', 'color')}
    m.instruments += list(I.values())
    nbars = int(np.ceil(dur / bar)) + 1
    # sections (in bars): intro 2, A 8, B 8, A' 8, then outro
    def sec(b):
        if b < 2: return 'intro'
        if b < 10: return 'A'
        if b < 18: return 'B'
        if b < 26: return 'A2'
        return 'outro'
    phrases = {}
    for b in range(nbars):
        t0 = b * bar; deg = prog[b % len(prog)]; ch = chord(base, deg); s = sec(b)
        fade = 1.0 if s != 'outro' else max(0.4, 1 - (b - 26) * 0.12)
        # bed: always, softer in intro/outro, fuller in B
        bv = {'intro': 26, 'A': 32, 'B': 40, 'A2': 32, 'outro': 28}[s]
        for n in ch[:3]: add(I['bed'], bv * fade, n + 12, t0, bar + 0.25)
        # accompaniment: always
        av = {'intro': 40, 'A': 44, 'B': 48, 'A2': 44, 'outro': 38}[s]
        accompaniment(I['acc'], pc['acc_pat'], ch, t0, beat, meter, av * fade, rng)
        # bass: from A
        if s != 'intro':
            add(I['bass'], 46 * fade, ch[0] - 24 + (12 if pc['bass'] in ('cello',) else 0), t0, bar * (0.95 if s != 'B' else 0.48))
            if s == 'B': add(I['bass'], 42, ch[2] - 24 + (12 if pc['bass'] == 'cello' else 0), t0 + bar / 2, bar * 0.48)
        # lead melody: A, A2 (and B with variation), phrases of 2 bars, repeated as call & answer
        if s in ('A', 'B', 'A2') and b % 8 not in (7,):
            pi = (b - 2) // 2
            if b % 2 == 0:
                key_ = (s if s != 'A2' else 'A', pi % 2)
                if key_ not in phrases:
                    phrases[key_] = melody_phrase(rng, base, [chord(base, prog[b % len(prog)]), chord(base, prog[(b + 1) % len(prog)])], meter, 24)
                for off, p, ln in phrases[key_]:
                    add(I['lead'], (54 if s == 'B' else 50) * fade, p, t0 + off * beat, ln * beat)
        # counter-melody in B: long notes a third/sixth below the lead, one per bar
        if s == 'B':
            add(I['counter'], 40, ch[1] + (0 if pc['counter'] in ('cello',) else 12), t0, bar * 0.95)
        # colour: sparkles on chord tones in B and A2, sparse
        if s in ('B', 'A2') and b % 2 == 1:
            for k, n in enumerate([ch[2], ch[0] + 12]):
                add(I['color'], 38 if s == 'B' else 32, n + 24, t0 + (k * 2 + 1) * beat * (meter / 4), beat * 1.2)
    tend = nbars * bar
    for n in chord(base, prog[0])[:3]:
        add(I['bed'], 30, n + 12, tend, bar * 2); add(I['acc'], 38, n, tend, bar * 1.6)
    add(I['bass'], 40, chord(base, prog[0])[0] - 12, tend, bar * 2)
    return m

def main():
    a = sys.argv[1:]; dur = 80
    if '--dur' in a: i = a.index('--dur'); dur = float(a[i + 1]); del a[i:i + 2]
    out = a[0]; os.makedirs(out, exist_ok=True)
    for k, pc in enumerate(PIECES):
        m = compose(pc, dur, 100 + k)
        mid, wav, mp3 = (os.path.join(out, f"{pc['slug']}.{e}") for e in ('mid', 'wav', 'mp3'))
        m.write(mid)
        subprocess.run(['fluidsynth', '-ni', '-g', '0.5', '-r', '44100', '-R', '1', '-C', '1', '-F', wav, SF2, mid],
                       check=True, capture_output=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', wav, '-af',
                        f'atrim=0:{dur + 6},afade=t=in:d=0.8,afade=t=out:st={dur + 1}:d=5,loudnorm=I=-16:TP=-1.5',
                        '-ar', '44100', '-b:a', '192k', mp3], check=True)
        os.remove(wav); os.remove(mid); print(mp3)

if __name__ == '__main__':
    main()
