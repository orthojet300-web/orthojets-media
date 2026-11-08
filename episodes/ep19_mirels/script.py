"""Orthojets EP19 - ORTHO NUMBERS #5 - Mirels score 9: impending pathological fracture, fix before it breaks?
Sources (verified on PubMed abstracts / metadata / PMC full text, 2026-11-08):
  Mirels H. Metastatic disease in long bones. A proposed scoring system for diagnosing impending pathologic fractures.
    Clin Orthop Relat Res 1989;(249):256-64. PMID 2684463 (reprint PMID 14600587, doi 10.1097/01.blo.0000093045.56370.dd).
    78 long-bone lesions irradiated without prophylactic fixation: 27 fractured within 6 months, 51 did not. Mean score
    7 (no fracture) vs 10 (fracture). Lesions should be scored before irradiation.
  Benca E, et al. Bone Rep 2016;5:51-6. PMID 28326347, PMC4926839, doi 10.1016/j.bonr.2016.02.003. Mirels: 4 factors
    (pain, size / cortical involvement, lytic vs blastic, anatomic site), each scored 1-3, total 4-12; 7 or lower low
    risk, 8 borderline, 9 or higher prophylactic stabilisation. High fracture incidence with pain aggravated by function,
    lesion occupying more than two-thirds of the bone diameter, lytic lesions. NPV 86-100%, PPV 23-70%: overtreatment.
  Hoban KA, et al. JSES Int 2022;6(4):675-81. PMID 35813136, PMC9264023, doi 10.1016/j.jseint.2022.03.006. Site by region:
    non-weight-bearing (upper limb) 1, weight-bearing (lower limb) 2, pertrochanteric 3; score 8 or more = 15% fracture risk,
    9 or more = 33% (citing Mirels). Upper limb: 9 or more sensitivity 14%, specificity 73%; 7 or more sensitivity 63%,
    specificity 55%; recommend 7 or more for the upper limb.
  Damron TA, et al. Clin Orthop Relat Res 2003;(415 Suppl):S201-7. PMID 14600611, doi 10.1097/01.blo.0000093842.72468.73.
    53 raters, 12 femoral cases: sensitivity 91%, specificity 35%; more sensitive than clinical judgment.
  Crenn V, et al. Orthop Traumatol Surg Res 2020;106(6):1005-11. PMID 32782175, doi 10.1016/j.otsr.2020.03.034.
    245 patients with long-bone metastasis: 126 (51.4%) operated at fracture stage.
  Mirels table (standard): site upper limb 1 / lower limb 2 / peritrochanteric 3; pain mild 1 / moderate 2 / functional 3;
    lesion blastic 1 / mixed 2 / lytic 3; size under 1/3 / 1/3 to 2/3 / over 2/3 of the bone diameter = 1 / 2 / 3.
  Diagram: AP left femur (kit/femur.py), metastasis in the proximal femur just below the lesser trochanter
  (peritrochanteric region), drawn blastic -> mixed -> lytic; width measured against the bone diameter.
  Worked example: peritrochanteric 3 + functional pain 3 + lytic 3 + half the diameter 2 = 11.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(47.0, [(7.4, 1.2), (11.4, 1.6), (14.6, 1.4), (17.8, 1.4), (21.0, 1.6), (23.4, 1.0), (28.6, 1.6),
                     (35.0, 1.4), (41.0, 1.6)])

GOLD = (236, 196, 98)
# ------------------------------------------------------------------ femur + lesion
FIMG, _FEDGE = FEM.render()
FA = np.asarray(FIMG).astype(np.float32)
LES_C = (298, 380)                                    # lesion centre in bone space (just below the lesser trochanter)
_row = FA[LES_C[1], :, 3]
_xs = np.where(_row > 60)[0]
BX0, BX1 = int(_xs.min()), int(_xs.max())             # cortex to cortex at the lesion level
BWID = BX1 - BX0
CROP = (40, 0, 480, 700)                              # proximal femur window (bone space)
CW, CH = CROP[2] - CROP[0], CROP[3] - CROP[1]
_yy, _xx = np.mgrid[0:FEM.BH, 0:FEM.BW].astype(np.float32)

def blob_mask(frac):
    """irregular lesion; width = frac x bone diameter"""
    rx = max(4, frac * BWID / 2); ry = rx * 1.55
    th = np.arctan2((_yy - LES_C[1]) / ry, (_xx - LES_C[0]) / rx)
    rr = 1 + 0.13 * np.sin(3 * th + 1) + 0.08 * np.sin(5 * th + 2.2)
    d = np.hypot((_xx - LES_C[0]) / rx, (_yy - LES_C[1]) / ry) / rr
    return np.clip((1.12 - d) / 0.25, 0, 1)

_fc = {}
def femur(kind, frac):
    """kind: 'none' | 'blastic' | 'mixed' | 'lytic'; returns the cropped proximal femur RGBA"""
    k = (kind, round(frac, 2))
    if k in _fc: return _fc[k]
    a = FA.copy()
    if kind != 'none':
        m = blob_mask(frac) * (FA[..., 3] > 0)
        if kind == 'lytic':
            a[..., 3] *= 1 - 0.82 * m
        elif kind == 'blastic':
            a[..., :3] += m[..., None] * (255 - a[..., :3]) * 0.85; a[..., 3] = np.maximum(a[..., 3], m * 250 * (FA[..., 3] > 0))
        else:                                          # mixed: lytic core with sclerotic islands
            spots = (np.sin(_xx * 0.35) * np.sin(_yy * 0.28) > 0.35).astype(np.float32)
            a[..., 3] *= 1 - 0.75 * m * (1 - spots)
            a[..., :3] += (m * spots)[..., None] * (255 - a[..., :3]) * 0.8
            a[..., 3] = np.maximum(a[..., 3], m * spots * 240)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(CROP)
    fa = np.asarray(im).copy(); fy = np.arange(CH)[:, None]
    fa[..., 3] = (fa[..., 3] * np.clip((CH - 10 - fy) / 120, 0, 1)).astype(np.uint8)
    im = Image.fromarray(fa)
    if len(_fc) > 120: _fc.clear()
    _fc[k] = im
    return im

_sc = {}
def femur_s(kind, frac, s):
    k = (kind, round(frac, 2), round(s, 3))
    if k not in _sc:
        im = femur(kind, frac)
        _sc[k] = im.resize((int(CW * s), int(CH * s)), Image.LANCZOS) if s != 1 else im
        if len(_sc) > 120: _sc.clear()
    return _sc[k]

def bp(cx, cy, s, q): return cx - CW * s / 2 + (q[0] - CROP[0]) * s, cy - CH * s / 2 + (q[1] - CROP[1]) * s

# peritrochanteric zone highlight (bone y 150-440)
_zm = (FA[..., 3] > 40) * np.clip((_yy - 150) / 20, 0, 1) * np.clip((450 - _yy) / 20, 0, 1)
_z = np.zeros((FEM.BH, FEM.BW, 4), np.float32); _z[..., :3] = SKY; _z[..., 3] = _zm * 175
ZONE = Image.fromarray(_z.astype(np.uint8)).crop(CROP)
def zone_s(s): return ZONE.resize((int(CW * s), int(CH * s)), Image.LANCZOS)
_zcache = {}

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14 if lx > tip[0] else lx + lab.width / 2 - 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay)
    put(main, lab, lx, ly, a)

def fit(txt, font, size, col, maxw, track=0):
    im = tl(txt, font, size, col, track=track)
    while im.width > maxw and size > 30:
        size -= 4; im = tl(txt, font, size, col, track=track)
    return im

def contour(main, cx, cy, s, frac, a, col=WARN):
    if a <= 0: return
    rx = max(4, frac * BWID / 2) * 1.05; ry = rx * 1.55; pts = []
    for k in range(49):
        th = 2 * math.pi * k / 48; rr = 1 + 0.13 * math.sin(3 * th + 1) + 0.08 * math.sin(5 * th + 2.2)
        pts.append(bp(cx, cy, s, (LES_C[0] + rx * rr * math.cos(th), LES_C[1] + ry * rr * math.sin(th))))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    for k in range(0, 48, 2): d.line([pts[k], pts[k + 1]], fill=col + (int(255 * a),), width=4)
    main.alpha_composite(lay)

def rings(main, glow, c, t, a, col=WARN):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    for k in range(3):
        ph = ((t * 0.9 + k / 3) % 1.0); r = 30 + 110 * ph; al = int(220 * a * (1 - ph))
        d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), outline=col + (al,), width=5)
        g.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), outline=col + (al // 2,), width=12)
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('ORTHO NUMBERS  #5', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 400, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Breast cancer, lytic femur lesion,', 'MontXB', 42, ICE, 880), W / 2, 485, a * q)
    put(main, fit('pain on walking:', 'MontXB', 42, ICE, 880), W / 2, 540, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('MIRELS 9', 'MontBlack', 150, WHITE)
    put(glow, k1, W / 2, 680, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 680, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('FIX BEFORE IT BREAKS?', 'MontBlack', 64, SKY, 880, track=2), W / 2, 815, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, pill('CUT-OFF: 9 OF 12 POINTS', 'MontBlack', 36, NAVY, WARN + (255,), h=70), W / 2, 920 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.3)); fr = 0.3 + 0.22 * eio(prog(t, 0.3, 1.4))
    s = 0.8; cx, cy = 560, 1310
    put(main, femur_s('lytic', fr, s), cx, cy, a * hq)
    contour(main, cx, cy, s, fr, a * hq)
    rings(main, glow, bp(cx, cy, s, LES_C), t, a * eo(prog(t, 0.6, 1.1)))

def s2(t, main, glow, fx):                                     # context
    a = win(t, 3.3, 9.2, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('IMPENDING FRACTURE', 'MontBlack', 70, WHITE, 860, track=3), W / 2, 262, a * p, 1, 10 * (1 - p))
    label(main, 'metastasis in a long bone', W / 2, 345, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s = 560, 800, 1.05
    q = eo(prog(t, 3.6, 4.3)); fr = 0.52 * eio(prog(t, 4.4, 5.4))
    put(main, femur_s('lytic' if fr > 0.02 else 'none', max(fr, 0.02), s), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), bp(cx, cy, s, (150, 130)), 200, 470, 'HEAD', SKY, 34)
    pointer(main, a * eo(prog(t, 4.6, 5.1)), bp(cx, cy, s, (350, 140)), 790, 450, 'GREATER TROCH.', ICE, 32)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), bp(cx, cy, s, (240, 316)), 215, 760, 'LESSER TROCH.', ICE, 32)
    pointer(main, a * eo(prog(t, 5.4, 5.9)), bp(cx, cy, s, (LES_C[0] + 10, LES_C[1] + 10)), 800, 900, 'LYTIC LESION', WARN, 34)
    contour(main, cx, cy + 20 * (1 - q), s, max(fr, 0.02), a * eo(prog(t, 4.6, 5.0)))
    rings(main, glow, bp(cx, cy, s, LES_C), t, a * eo(prog(t, 5.4, 5.9)))
    q = eo(prog(t, 6.2, 6.7))
    if q > 0:
        y = 1430 + 20 * (1 - q)
        put(main, card(872, 230, outline=SKY + (150,)), 515, y, a * q)
        label(main, '78 lesions irradiated, no fixation:', 515, y - 60, a * q, WHITE, 36, 'MontBlack', 0)
        put(main, tl('27 FRACTURED', 'MontBlack', 64, WARN, track=2), 515, y + 15, a * q)
        label(main, 'within 6 months  ·  Mirels 1989', 515, y + 80, a * q, SUB, 30, 'MontXB', 0)

FACT = [('SITE', [('Upper limb', 1), ('Lower limb', 2), ('Peritrochanteric', 3)], 2, 9.4),
        ('PAIN', [('Mild', 1), ('Moderate', 2), ('Functional', 3)], 2, 12.6),
        ('LESION', [('Blastic', 1), ('Mixed', 2), ('Lytic', 3)], 2, 15.8),
        ('SIZE', [('Under 1/3 of width', 1), ('1/3 to 2/3', 2), ('Over 2/3', 3)], 1, 19.0)]
def s3(t, main, glow, fx):                                     # centrepiece: score built factor by factor
    a = win(t, 9.0, 24.0, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = 560, 780, 1.12
    # lesion state
    if t < 15.8: kind, fr = 'lytic', 0.52
    else:
        kind = 'lytic'; fr = 0.52
    if 15.8 <= t < 19.0:
        u = t - 15.8
        kind = 'blastic' if u < 0.45 else ('mixed' if u < 0.95 else 'lytic')
    if 19.0 <= t < 22.0:
        fr = 0.2 + 0.32 * eio(prog(t, 19.4, 20.4))
    put(main, femur_s(kind, fr, s), cx, cy, a)
    za = a * win(t, 9.6, 12.5, 0.4, 0.4)
    if za > 0:
        if s not in _zcache: _zcache[s] = zone_s(s)
        put(main, _zcache[s], cx, cy, za * (0.75 + 0.25 * math.sin(t * 5)))
        put(glow, _zcache[s], cx, cy, za * 0.8)
        pointer(main, za, bp(cx, cy, s, (262, 300)), 245, 960, 'PERITROCHANTERIC', SKY, 28)
    LC = bp(cx, cy, s, LES_C)
    contour(main, cx, cy, s, fr, a, WARN if kind != 'blastic' else ICE)
    rings(main, glow, LC, t, a * win(t, 12.8, 15.7, 0.3, 0.3))
    if 15.8 <= t < 19.0:
        nm = {'blastic': 'BLASTIC', 'mixed': 'MIXED', 'lytic': 'LYTIC'}[kind]
        pointer(main, a * win(t, 16.0, 18.9, 0.2, 0.3), (LC[0] + 20, LC[1] + 20), 820, 980, nm, WARN if kind == 'lytic' else ICE, 34)
    ma = a * win(t, 19.3, 22.1, 0.3, 0.3)
    if ma > 0:                                         # measurement: bone width with thirds, lesion width
        lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
        y0 = LC[1] - 14; x0 = bp(cx, cy, s, (BX0, 0))[0]; x1 = bp(cx, cy, s, (BX1, 0))[0]
        d.line([(x0, y0), (x1, y0)], fill=ICE + (int(255 * ma),), width=4)
        for xx in (x0, x1): d.line([(xx, y0 - 14), (xx, y0 + 6)], fill=ICE + (int(255 * ma),), width=4)
        for k in (1, 2):
            xx = x0 + (x1 - x0) * k / 3; d.line([(xx, y0 - 10), (xx, y0 + 4)], fill=SKY + (int(255 * ma),), width=3)
        lw = fr * BWID * s; ly = LC[1] + 14
        d.line([(LC[0] - lw / 2, ly), (LC[0] + lw / 2, ly)], fill=WARN + (int(255 * ma),), width=5)
        for xx in (LC[0] - lw / 2, LC[0] + lw / 2): d.line([(xx, ly - 6), (xx, ly + 12)], fill=WARN + (int(255 * ma),), width=4)
        d.line([(x1 + 8, y0), (760, LC[1] - 40)], fill=ICE + (int(160 * ma),), width=2)
        d.line([(LC[0] + lw / 2 + 8, ly), (760, LC[1] + 40)], fill=WARN + (int(160 * ma),), width=2)
        main.alpha_composite(lay)
        label(main, 'bone width', 850, LC[1] - 40, ma, ICE, 34, 'MontXB', 0)
        label(main, f'lesion {int(round(fr * 100))}%', 850, LC[1] + 40, ma, WARN, 36, 'MontBlack', 0)
    # running score + factor card
    score = sum(opts[ch][1] for name, opts, ch, t0 in FACT if t >= t0 + (1.8 if name == 'SIZE' else 1.2))
    for i, (name, opts, ch, t0) in enumerate(FACT):
        t1 = FACT[i + 1][3] if i < 3 else 22.1
        ta = a * win(t, t0 - 0.25, t1 + 0.05, 0.3, 0.25)
        if ta <= 0: continue
        p = R.eback(prog(t, t0, t0 + 0.45))
        put(main, tl(f'{i + 1}  ·  {name}', 'MontBlack', 76, WHITE, track=2), W / 2, 262, ta * clamp(p), clamp(p, 0, 1.2))
        q = eo(prog(t, t0 + 0.2, t0 + 0.6))
        y = 1395 + 15 * (1 - q)
        put(main, card(880, 300, outline=SKY + (140,)), 515, y, ta * q)
        for j, (txt, pts) in enumerate(opts):
            yy = y - 92 + 92 * j
            hd = 1.6 if name == 'SIZE' else 1.0
            on = j == ch and t >= t0 + hd
            if on:
                put(main, card(840, 80, fill=WARN + (235,), outline=WARN + (255,), r=40), 515, yy, ta * eo(prog(t, t0 + hd, t0 + hd + 0.3)))
            on = on and t >= t0 + hd + 0.12
            col = NAVY if on else ICE
            lab = tl(txt, 'MontBlack' if on else 'MontXB', 38, col)
            put(main, lab, 140 + lab.width / 2 - 20, yy, ta * q)
            put(main, tl(str(pts), 'MontBlack', 46, col), 880, yy, ta * q)
    sa = a * win(t, 9.6, 22.1, 0.3, 0.3)
    if sa > 0:
        put(main, pill(f'SCORE  {score} / 12', 'MontBlack', 38, NAVY, SKY + (255,), h=72), W / 2, 360, sa)
    ta = a * win(t, 22.0, 24.0, 0.3, 0.4)
    if ta > 0:                                         # total
        p = R.eback(prog(t, 22.0, 22.5))
        big = tl('11 / 12', 'MontBlack', 150, WARN)
        put(glow, big, W / 2, 1330, ta * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 1330, ta * clamp(p), clamp(p, 0, 1.2))
        label(main, '3 + 3 + 3 + 2', W / 2, 1200, ta, ICE, 40, 'MontBlack', 2)
        q = eo(prog(t, 22.4, 22.8))
        put(main, pill('9 OR MORE: FIX FIRST', 'MontBlack', 40, NAVY, WARN + (255,), h=76), W / 2, 1480, ta * q)
        label(main, 'MIRELS SCORE', W / 2, 262, ta, WHITE, 76, 'MontBlack', 2)
    label(main, 'Mirels H.  ·  Clin Orthop Relat Res 1989', W / 2, 1600, a * eo(prog(t, 10.0, 10.5)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # where is the line
    a = win(t, 23.8, 30.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'WHERE IS THE LINE?', W / 2, 262, a * eo(prog(t, 23.9, 24.4)), WHITE, 64, 'MontBlack', 2)
    label(main, 'score 4 to 12', W / 2, 340, a * eo(prog(t, 24.1, 24.6)), SKY, 36, 'MontXB', 0)
    x0, x1, y = 120, 910, 560
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    step = (x1 - x0) / 8
    for k in range(9):
        q = eo(prog(t, 24.3 + 0.08 * k, 24.6 + 0.08 * k))
        if q <= 0: continue
        sc = 4 + k; col = GOOD if sc <= 7 else (GOLD if sc == 8 else WARN)
        xx = x0 + step * k
        d.rounded_rectangle((xx - 40, y - 40, xx + 40, y + 40), 16, fill=col + (int(230 * a * q),))
    main.alpha_composite(lay)
    for k in range(9):
        q = eo(prog(t, 24.3 + 0.08 * k, 24.6 + 0.08 * k))
        put(main, tl(str(4 + k), 'MontBlack', 44, NAVY), x0 + step * k, y, a * q)
    q = eo(prog(t, 25.2, 25.7))
    label(main, '7 or less', x0 + step * 1.5, y + 90, a * q, GOOD, 34, 'MontBlack', 0)
    label(main, 'irradiate', x0 + step * 1.5, y + 140, a * q, ICE, 32, 'MontXB', 0)
    q = eo(prog(t, 25.6, 26.1))
    label(main, '8', x0 + step * 4, y + 90, a * q, GOLD, 34, 'MontBlack', 0)
    label(main, 'borderline', x0 + step * 4, y + 140, a * q, ICE, 32, 'MontXB', 0)
    q = eo(prog(t, 26.0, 26.5))
    label(main, '9 or more', x0 + step * 6.5, y + 90, a * q, WARN, 34, 'MontBlack', 0)
    label(main, 'fix first', x0 + step * 6.5, y + 140, a * q, ICE, 32, 'MontXB', 0)
    q = eo(prog(t, 26.6, 27.1))
    if q > 0:
        yc = 1000 + 20 * (1 - q)
        put(main, card(880, 300, outline=WARN + (170,)), 515, yc, a * q)
        label(main, 'Fracture risk', 515, yc - 100, a * q, WHITE, 40, 'MontBlack', 0)
        pk = R.eback(prog(t, 26.8, 27.3))
        put(main, tl('15%', 'MontBlack', 96, GOLD), 300, yc + 10, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'score 8 or more', 300, yc + 95, a * q, ICE, 30, 'MontXB', 0)
        pk = R.eback(prog(t, 27.1, 27.6))
        put(main, tl('33%', 'MontBlack', 96, WARN), 730, yc + 10, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'score 9 or more', 730, yc + 95, a * q, ICE, 30, 'MontXB', 0)
    q = eo(prog(t, 27.6, 28.1))
    if q > 0:
        yc = 1350 + 20 * (1 - q)
        put(main, card(880, 170, outline=SKY + (150,)), 515, yc, a * q)
        label(main, 'Mean score: 7 no fracture vs 10 fractured', 515, yc - 30, a * q, WHITE, 32, 'MontBlack', 0)
        label(main, '78 lesions  ·  Mirels 1989', 515, yc + 35, a * q, SUB, 30, 'MontXB', 0)
    label(main, 'Benca et al. Bone Rep 2016  ·  Hoban et al. JSES Int 2022', W / 2, 1560, a * eo(prog(t, 27.0, 27.5)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # sensitive, not specific
    a = win(t, 29.8, 36.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'SENSITIVE,', W / 2, 262, a * eo(prog(t, 29.9, 30.4)), WHITE, 64, 'MontBlack', 2)
    label(main, 'NOT SPECIFIC', W / 2, 345, a * eo(prog(t, 30.1, 30.6)), WARN, 64, 'MontBlack', 2)
    for cx, num, txt, col, t0 in [(285, '91%', 'sensitivity', GOOD, 30.6), (745, '35%', 'specificity', WARN, 31.2)]:
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + 20 * (1 - q)
        put(main, card(420, 300, outline=col + (160,)), cx, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        put(glow, tl(num, 'MontBlack', 120, col), cx, y - 25, a * q * 0.3 * clamp(pk), clamp(pk, 0, 1.2))
        put(main, tl(num, 'MontBlack', 120, col), cx, y - 25, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, txt, cx, y + 85, a * q, ICE, 36, 'MontXB', 0)
    label(main, 'Damron et al. Clin Orthop 2003  ·  53 raters', W / 2, 850, a * eo(prog(t, 31.4, 31.9)), SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 32.2, 32.7))
    if q > 0:
        y = 1080 + 20 * (1 - q)
        put(main, card(880, 210, outline=WARN + (170,)), 515, y, a * q)
        label(main, 'Many high scores never break', 515, y - 40, a * q, WHITE, 40, 'MontBlack', 0)
        label(main, 'PPV 23-70%: overtreatment', 515, y + 35, a * q, WARN, 36, 'MontBlack', 0)
    q = eo(prog(t, 33.2, 33.7))
    if q > 0:
        y = 1350 + 20 * (1 - q)
        put(main, card(880, 210, outline=GOOD + (170,)), 515, y, a * q)
        label(main, 'A low score is reassuring', 515, y - 40, a * q, WHITE, 40, 'MontBlack', 0)
        label(main, 'NPV 86-100%', 515, y + 35, a * q, GOOD, 40, 'MontBlack', 0)
    label(main, 'Benca et al.  ·  Bone Rep 2016', W / 2, 1560, a * eo(prog(t, 33.6, 34.1)), SUB, 28, 'MontXB', 0)

TIPS = [('Humerus? Lower the bar to 7', '9 OR MORE CAUGHT 14%, 7 OR MORE 63%', 'Hoban et al. JSES Int 2022'),
        ("Don't wait for the fracture", '51.4% OF 245 OPERATED ONLY AFTER IT BROKE', 'Crenn et al. OTSR 2020'),
        ('Score it before radiotherapy', '7 OR LESS: IRRADIATE  ·  HIGH: FIX FIRST', 'Mirels H. Clin Orthop 1989')]
def s6(t, main, glow, fx):
    a = win(t, 35.8, 41.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 35.9, 36.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 36.3 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = fit(ref, 'MontXB', 29, SKY, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 41.6, 47.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 290, a * eo(prog(t, 41.7, 42.2)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 41.9, 42.5))
    big = fit('9 OR MORE: FIX FIRST', 'MontBlack', 92, WHITE, 860)
    put(glow, big, W / 2, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '7 or less: irradiate  ·  8: borderline', W / 2, 505, a * eo(prog(t, 42.3, 42.8)), GOOD, 34, 'MontBlack', 0)
    label(main, 'Upper limb: consider fixing from 7', W / 2, 575, a * eo(prog(t, 42.6, 43.1)), WARN, 34, 'MontBlack', 0)
    q = eo(prog(t, 43.0, 43.6))
    refs = ['Mirels H. Clin Orthop Relat Res 1989;(249):256–64', 'Damron TA, et al. Clin Orthop 2003;(415S):S201–7',
            'Benca E, et al. Bone Rep 2016;5:51–6', 'Hoban KA, et al. JSES Int 2022;6(4):675–81',
            'Crenn V, et al. OTSR 2020;106(6):1005–11']
    put(main, card(880, 440, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 900, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 29, ICE if i % 2 == 0 else SUB, 820), 515, 760 + i * 70, a * q)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1260, a * eo(prog(t, 43.6, 44.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
