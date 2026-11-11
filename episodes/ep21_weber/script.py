"""Orthojets EP21 - CLASSIFIED #5 - Danis-Weber ankle fractures: fix or cast?
Sources (verified on PubMed abstracts / metadata, 2026-11-11):
  Weber BG. Die Verletzungen des oberen Sprunggelenkes. Bern: Huber, 1972 (original; not indexed). Danis-Weber = level of
    the fibular fracture relative to the syndesmosis: A below, B at the level of, C above the syndesmosis (definitional).
  Kellam PJ, Dekeyser GJ, Haller JM, et al. J Orthop Trauma 2021;35(9):473-8. PMID 33395177, doi 10.1097/BOT.0000000000002052.
    287 OTA/AO 44-B2.1 (Weber B). Zone of the distal extent of the proximal fragment: zone 1 below the plafond 17% (33/191),
    zone 2 between plafond and physeal scar 42% (24/57), zone 3 above the physeal scar 74% (29/39) syndesmotic injury
    (positive intraoperative stress exam requiring syndesmotic fixation). Kappa 0.86 / 0.94.
  Kortekangas T, et al. BMJ 2019;364:k5432. PMID 30674451, doi 10.1136/bmj.k5432. 247 patients, isolated Weber B, congruent
    mortise on static films: 3 weeks cast (OMAS 91.7) or 3 weeks orthosis (89.8) non-inferior to 6 weeks cast (87.6) at 52 wk.
  Kortekangas T, et al. (SUPER-FIN) BMJ 2026;392:e085295. PMID 41534905, doi 10.1136/bmj-2025-085295. 840 screened; 126
    unimalleolar Weber B, congruent mortise on static films but unstable on ER stress test: 6 weeks cast (n=62) vs ORIF + 6 weeks
    cast (n=64). OMAS at 2 years 89 vs 87 (difference 1.3, 95% CI -4.8 to 7.3): cast non-inferior. Surgery: 1 superficial
    infection, 1 delayed wound healing, 9 hardware removals (2 infections after them). Excluded: stable fractures, mortise
    incongruency, fracture dislocation.
  Harper MC, Keller TS. Foot Ankle 1989;10(3):156-60. PMID 2613128, doi 10.1177/107110078901000308. Normal: tibiofibular
    clear space under ~6 mm (AP and mortise), overlap over ~6 mm / 42% of fibular width on AP, over ~1 mm on mortise.
  Tornetta P 3rd, et al. J Orthop Trauma 2012;26(11):659-61. PMID 23100079, doi 10.1097/BOT.0b013e31825cf39c. 114 stress
    positive SER fibula fractures: 54 cast, 60 operated; 27 of the 60 operated (45%) had syndesmotic instability after fibular
    fixation; no patient healed with subluxation on weight-bearing films.
  Diagram: AP view of a RIGHT ankle (fibula = lateral = viewer's left; medial malleolus on viewer's right). Tibia with plafond,
    medial malleolus and incisura; fibula longer than the medial malleolus; talus dome under the plafond; syndesmosis between
    distal fibula and tibial incisura just above the plafond; physeal scar as a faint line in the distal tibia.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(50.0, [(7.6, 1.4), (11.8, 1.2), (15.2, 1.2), (18.6, 1.2), (24.6, 1.4), (30.0, 1.2), (36.6, 1.8),
                     (42.8, 1.4)])

# ------------------------------------------------------------------ diagram: AP right ankle (canvas units)
DW, DH = 620, 780
BONE = (226, 238, 250)
PLAF_Y, SCAR_Y = 486, 418

def mask_poly(poly, ss=2):
    m = Image.new('L', (DW * ss, DH * ss)); ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    return m.resize((DW, DH), Image.LANCZOS)

def bone_layer(m, a=235):
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(9))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.40 + 0.60 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: int(a * 0.9) if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

sm = lambda pts, n=8: FEM.catmull(pts, n=n)
TIB_PTS = [(318, -40), (312, 150), (296, 260), (262, 340), (238, 390), (230, 430), (234, 465), (244, 482), (300, 490),
           (360, 492), (420, 488), (450, 482), (456, 520), (466, 556), (486, 574), (506, 566), (520, 520), (526, 440),
           (508, 360), (470, 280), (440, 180), (432, -40)]
FIB_PTS = [(160, -40), (192, -40), (196, 200), (204, 330), (214, 410), (220, 470), (223, 540), (210, 604), (184, 640),
           (156, 632), (132, 590), (122, 510), (128, 430), (146, 330), (158, 200)]
TAL_PTS = [(242, 506), (300, 494), (360, 494), (420, 498), (446, 508), (450, 545), (444, 590), (424, 628), (380, 652),
           (310, 658), (252, 644), (226, 606), (230, 556)]

_yy, _xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
def fade_bottom(m, y0=640, y1=760):
    a = np.asarray(m, np.float32) * np.clip((y1 - _yy) / (y1 - y0), 0, 1) * np.clip((_yy + 40) / 160, 0, 1)
    return Image.fromarray(a.astype(np.uint8))

TIB_M = fade_bottom(mask_poly(sm(TIB_PTS)))
FIB_M = fade_bottom(mask_poly(sm(FIB_PTS)))
TAL_M = fade_bottom(mask_poly(sm(TAL_PTS)), 660, 780)
def topfade(lay, y0=60, y1=230):
    a = np.asarray(lay).copy(); a[..., 3] = (a[..., 3] * np.clip((_yy - y0) / (y1 - y0), 0, 1)).astype(np.uint8)
    return Image.fromarray(a)
TIB, FIB, TAL = topfade(bone_layer(TIB_M)), topfade(bone_layer(FIB_M)), bone_layer(TAL_M, 215)
# Weber B line (inferomedial at the plafond level -> superolateral); distal fragment = below the line
B0, B1 = (219, 500), (128, 404)
def below_line(P0, P1):
    cr = (P1[0] - P0[0]) * (_yy - P0[1]) - (P1[1] - P0[1]) * (_xx - P0[0])
    return cr < 0
_bm = below_line(B0, B1)
def cut(img, m):
    a = np.asarray(img).copy(); a[..., 3] = (a[..., 3] * m).astype(np.uint8); return Image.fromarray(a)
FIB_DIST = cut(FIB, _bm); FIB_PROX = cut(FIB, ~_bm)

def shifted(im, dx, dy=0):
    return im.transform(im.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), Image.BICUBIC)

_dc = {}
def ankle(s, shift=0.0):
    k = (round(s, 3), round(shift, 1))
    if k in _dc: return _dc[k]
    im = Image.new('RGBA', (DW, DH)); im.alpha_composite(TIB)
    if shift:
        im.alpha_composite(FIB_PROX); im.alpha_composite(shifted(FIB_DIST, shift)); im.alpha_composite(shifted(TAL, shift))
    else:
        im.alpha_composite(FIB); im.alpha_composite(TAL)
    if s != 1: im = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    if len(_dc) > 120: _dc.clear()
    _dc[k] = im
    return im

def dp(cx, cy, s, q): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def frac(main, glow, cx, cy, s, a, pl, P0, P1, col=WARN, dx=0):
    if a <= 0 or pl <= 0: return
    mid = ((P0[0] + P1[0]) / 2 + 6, (P0[1] + P1[1]) / 2 + 4)
    pts = [P0, mid, P1]; seg = []; n = 12
    for i in range(n + 1):
        u = i / n * pl
        if u <= 0.5: q = (pts[0][0] + (pts[1][0] - pts[0][0]) * u * 2, pts[0][1] + (pts[1][1] - pts[0][1]) * u * 2)
        else: q = (pts[1][0] + (pts[2][0] - pts[1][0]) * (u - .5) * 2, pts[1][1] + (pts[2][1] - pts[1][1]) * (u - .5) * 2)
        seg.append(dp(cx, cy, s, (q[0] + dx, q[1])))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    w = max(3, int(6 * s))
    for P_, Q_ in zip(seg, seg[1:]): d.line([P_, Q_], fill=(14, 22, 38, int(255 * a)), width=int(w * 1.9))
    for P_, Q_ in zip(seg, seg[1:]):
        d.line([P_, Q_], fill=col + (int(255 * a),), width=w); g.line([P_, Q_], fill=col + (int(150 * a),), width=w * 3)
    main.alpha_composite(lay)

def syndesmosis(main, glow, cx, cy, s, a, col=None):
    if a <= 0: return
    col = col or SKY
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    for y in range(394, 476, 13):
        xl = 207 + (y - 380) * 0.12; xr = 233 + (0 if y < 450 else (y - 450) * 0.15)
        P = dp(cx, cy, s, (xl - 2, y + 6)); Q = dp(cx, cy, s, (xr + 4, y - 6))
        d.line([P, Q], fill=col + (int(240 * a),), width=max(4, int(5 * s)))
        g.line([P, Q], fill=col + (int(120 * a),), width=max(6, int(10 * s)))
    main.alpha_composite(lay)

def hline(main, cx, cy, s, a, y, x0, x1, col, dash=True, w=3):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    X0, Y = dp(cx, cy, s, (x0, y)); X1, _ = dp(cx, cy, s, (x1, y))
    x = X0
    while x < X1:
        d.line([(x, Y), (min(x + 16, X1), Y)], fill=col + (int(220 * a),), width=w); x += 26 if dash else 16
    main.alpha_composite(lay)

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    lab = tl(txt, 'MontXB', size, col, track=2)
    if abs(lx - tip[0]) < 40: ex, ey = lx, ly - lab.height / 2 + 12 if ly > tip[1] else ly + lab.height / 2 - 12
    else: ex, ey = (lx - lab.width / 2 + 14 if lx > tip[0] else lx + lab.width / 2 - 14), ly
    d.line([tip, (ex, ey)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay)
    put(main, lab, lx, ly, a)

def fit(txt, font, size, col, maxw, track=0):
    im = tl(txt, font, size, col, track=track)
    while im.width > maxw and size > 30:
        size -= 4; im = tl(txt, font, size, col, track=track)
    return im

DCX, DCY, DS = 600, 840, 1.3                              # main diagram placement

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #5', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 400, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl('Ankle fracture, fibula at the joint line:', 'MontXB', 40, ICE), W / 2, 500, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FIX OR CAST?', 'MontBlack', 128, WHITE, 860)
    put(glow, k1, W / 2, 630, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 630, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, tl('WEBER  A · B · C', 'MontBlack', 66, SKY, track=2), W / 2, 770, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('A 2026 trial changed the answer', 'MontBlack', 46, WHITE), W / 2, 880 + 25 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.3)); cx, cy, s = 530, 1300, 0.85
    put(main, ankle(s), cx, cy, a * hq)
    frac(main, glow, cx, cy, s, a * hq, eio(prog(t, 0.2, 1.0)), B0, B1)
    syndesmosis(main, glow, cx, cy, s, a * eo(prog(t, 0.9, 1.4)))

def s2(t, main, glow, fx):                                     # context: normal AP ankle
    a = win(t, 3.3, 8.8, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('THE ANKLE MORTISE', 'MontBlack', 62, WHITE, track=3), W / 2, 262, a * p, 1, 10 * (1 - p))
    label(main, 'AP view, right ankle', W / 2, 340, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s = DCX, DCY, DS
    q = eo(prog(t, 3.6, 4.3))
    put(main, ankle(s), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), dp(cx, cy, s, (365, 160)), 850, 470, 'TIBIA', SKY, 34)
    pointer(main, a * eo(prog(t, 4.5, 5.0)), dp(cx, cy, s, (168, 220)), 190, 470, 'FIBULA', SKY, 34)
    pointer(main, a * eo(prog(t, 4.7, 5.2)), dp(cx, cy, s, (340, 600)), 640, 1250, 'TALUS', SKY, 34)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), dp(cx, cy, s, (500, 545)), 845, 1245, 'MEDIAL MALL.', ICE, 28)
    sa = a * eo(prog(t, 5.3, 5.8))
    syndesmosis(main, glow, cx, cy, s, sa)
    pointer(main, sa, dp(cx, cy, s, (222, 420)), 205, 790, 'SYNDESMOSIS', SKY, 30)
    q = eo(prog(t, 6.2, 6.7))
    if q > 0:
        y = 1450 + 20 * (1 - q)
        put(main, card(860, 210, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'WEBER = WHERE THE FIBULA BREAKS', 515, y - 45, a * q, SKY, 38, 'MontBlack', 1)
        label(main, 'relative to the syndesmosis', 515, y + 30, a * q, WHITE, 36, 'MontXB', 0)

TYPES = [('A', 'Below the syndesmosis', 'under the plafond level', GOOD, 9.0, (125, 566), (214, 574)),
         ('B', 'At the syndesmosis', 'starts at the plafond, runs up', SKY, 12.4, B0, B1),
         ('C', 'Above the syndesmosis', 'fibular shaft, above the tibiofibular joint', WARN, 15.8, (143, 276), (201, 222))]
def s3(t, main, glow, fx):                                     # centrepiece: A, B, C drawn one by one
    a = win(t, 8.6, 19.0, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = DCX, DCY, DS
    put(main, ankle(s), cx, cy, a)
    syndesmosis(main, glow, cx, cy, s, a * 0.85)
    hline(main, cx, cy, s, a * 0.8, PLAF_Y, 100, 540, ICE)
    pl_ = tl('plafond', 'MontXB', 31, ICE, track=1)
    put(main, pl_, dp(cx, cy, s, (100, 0))[0] - 12 - pl_.width / 2 + 20, dp(cx, cy, s, (0, PLAF_Y))[1], a)
    for i, (L, desc, sub, col, t0, P0, P1) in enumerate(TYPES):
        t1 = TYPES[i + 1][4] if i < 2 else 19.0
        ta = a * win(t, t0 - 0.3, t1 + 0.05, 0.3 if i else 0.35, 0.25 if i < 2 else 0.4)
        if ta <= 0: continue
        frac(main, glow, cx, cy, s, ta, eio(prog(t, t0 + 0.2, t0 + 0.9)), P0, P1)
        p = R.eback(prog(t, t0 + 0.05, t0 + 0.5))
        big = tl(f'WEBER {L}', 'MontBlack', 100, WHITE)
        put(glow, big, W / 2, 265, ta * 0.25 * clamp(p)); put(main, big, W / 2, 265, ta * clamp(p), clamp(p, 0, 1.2))
        dq = eo(prog(t, t0 + 0.4, t0 + 0.9))
        put(main, pill(desc.upper(), 'MontBlack', 36, NAVY, col + (255,), h=72), W / 2, 1350 + 15 * (1 - dq), ta * dq)
        eq = eo(prog(t, t0 + 0.8, t0 + 1.2))
        put(main, fit(sub, 'MontXB', 38, WHITE, 880), W / 2, 1450 + 15 * (1 - eq), ta * eq)
    label(main, 'Danis-Weber  ·  Weber 1972', W / 2, 1600, a * eo(prog(t, 9.6, 10.1)), SUB, 28, 'MontXB', 0)

ZONES = [('17%', 'below the plafond', GOOD, 19.8, (221, 556), (146, 462), 540),
         ('42%', 'plafond to physeal scar', SKY, 20.9, (219, 462), (140, 372), 452),
         ('74%', 'above the physeal scar', WARN, 22.0, (212, 400), (144, 300), 360)]
def s4(t, main, glow, fx):                                     # Weber B is not one injury
    a = win(t, 18.8, 25.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'WEBER B: 3 ZONES', W / 2, 262, a * eo(prog(t, 18.9, 19.4)), WHITE, 60, 'MontBlack', 2)
    label(main, 'syndesmosis injury by where the line starts', W / 2, 340, a * eo(prog(t, 19.1, 19.6)), SKY, 34, 'MontXB', 0)
    cx, cy, s = DCX, DCY, DS
    put(main, ankle(s), cx, cy, a)
    la = a * eo(prog(t, 19.2, 19.7))
    hline(main, cx, cy, s, la, PLAF_Y, 100, 540, ICE)
    hline(main, cx, cy, s, la, SCAR_Y, 100, 540, ICE)
    for i, (num, txt, col, t0, P0, P1, zy) in enumerate(ZONES):
        nxt = ZONES[i + 1][3] if i < 2 else 25.6
        fa = a * win(t, t0, nxt + 0.05, 0.25, 0.25 if i < 2 else 0.4)
        frac(main, glow, cx, cy, s, fa, eio(prog(t, t0 + 0.05, t0 + 0.6)), P0, P1, col)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.55))
        ylab = dp(cx, cy, s, (0, zy))[1]
        put(main, tl(num, 'MontBlack', 70, col), 190, ylab, a * clamp(pk), clamp(pk, 0, 1.2))
    q = eo(prog(t, 23.0, 23.5))
    if q > 0:
        y = 1460 + 20 * (1 - q)
        put(main, card(860, 260, outline=SKY + (150,)), 515, y, a * q)
        for k, (num, txt, col, *_r) in enumerate(reversed(ZONES)):
            label(main, f'{num}  {txt}', 515, y - 75 + 62 * k, a * q, col, 34, 'MontBlack', 0)
    label(main, 'Kellam et al.  ·  J Orthop Trauma 2021  ·  287 fractures', W / 2, 1640, a * eo(prog(t, 23.3, 23.8)),
          SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # the stress test: medial clear space opens
    a = win(t, 25.4, 31.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'STABLE OR NOT?', W / 2, 262, a * eo(prog(t, 25.5, 26.0)), WHITE, 64, 'MontBlack', 2)
    label(main, 'external rotation stress view', W / 2, 340, a * eo(prog(t, 25.7, 26.2)), SKY, 36, 'MontXB', 0)
    cx, cy, s = DCX, DCY, DS
    sh = -26 * eio(prog(t, 26.8, 28.2))
    put(main, ankle(s, round(sh, 1)), cx, cy, a)
    frac(main, glow, cx, cy, s, a, 1.0, B0, B1)
    # medial clear space bracket at the talar shoulder level
    y = 540; xl = 450 + sh; xr = 458
    P = dp(cx, cy, s, (xl, y)); Q = dp(cx, cy, s, (xr, y))
    col = WARN if sh < -12 else GOOD
    ba = a * eo(prog(t, 26.2, 26.7))
    if ba > 0:
        lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
        d.line([P, Q], fill=col + (int(255 * ba),), width=6)
        for X in (P[0], Q[0]): d.line([(X, P[1] - 18), (X, P[1] + 18)], fill=col + (int(255 * ba),), width=5)
        main.alpha_composite(lay)
        ImageDraw.Draw(glow).line([P, Q], fill=col + (int(140 * ba),), width=14)
    pointer(main, ba, ((P[0] + Q[0]) / 2, P[1] + 22), 760, 1370, 'MEDIAL CLEAR SPACE', col, 30)
    ar = a * eo(prog(t, 26.7, 27.1)) * (1 - prog(t, 28.6, 29.0))
    if ar > 0:
        lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
        X0, Y0 = dp(cx, cy, s, (330, 700)); A_ = int(255 * ar)
        d.line([(X0 + 60, Y0), (X0 - 86, Y0)], fill=WARN + (A_,), width=6)
        d.polygon([(X0 - 90, Y0), (X0 - 68, Y0 - 13), (X0 - 68, Y0 + 13)], fill=WARN + (A_,))
        main.alpha_composite(lay)
        label(main, 'talus shifts laterally', X0 - 90, Y0 + 50, ar, WARN, 30, 'MontXB', 0)
    q = eo(prog(t, 28.6, 29.1))
    if q > 0:
        yy = 1530 + 20 * (1 - q)
        put(main, card(860, 210, outline=WARN + (160,)), 515, yy, a * q)
        label(main, 'Clear space opens on stress', 515, yy - 45, a * q, WHITE, 38, 'MontBlack', 0)
        label(main, '= stress-unstable Weber B', 515, yy + 30, a * q, WARN, 38, 'MontBlack', 0)

def s6(t, main, glow, fx):                                     # the evidence: 2 Finnish RCTs
    a = win(t, 31.2, 37.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'MORTISE CONGRUENT?', W / 2, 262, a * eo(prog(t, 31.3, 31.8)), WHITE, 58, 'MontBlack', 2)
    label(main, 'what the randomised trials show', W / 2, 340, a * eo(prog(t, 31.5, 32.0)), SKY, 36, 'MontXB', 0)
    rows = [(31.8, GOOD, 'STABLE WEBER B', '3 WEEKS = 6 WEEKS', 'cast or simple orthosis  ·  247 patients',
             'OMAS 1 y: 91.7 / 89.8 vs 87.6 (6 wk cast)', 'Kortekangas et al. BMJ 2019'),
            (33.4, SKY, 'UNSTABLE ON STRESS', 'CAST = PLATE', '6 weeks cast vs ORIF  ·  126 patients',
             'OMAS 2 y: 89 vs 87  ·  fewer harms with the cast', 'SUPER-FIN  ·  Kortekangas et al. BMJ 2026')]
    for i, (t0, col, head, big, l1, l2, ref) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 480 + 25 * (1 - q)
        put(main, card(860, 430, outline=col + (170,)), 515, y, a * q)
        label(main, head, 515, y - 160, a * q, col, 36, 'MontBlack', 3)
        pk = R.eback(prog(t, t0 + 0.2, t0 + 0.7))
        bb = fit(big, 'MontBlack', 86, WHITE, 820)
        put(glow, bb, 515, y - 70, a * q * 0.25 * clamp(pk)); put(main, bb, 515, y - 70, a * q * clamp(pk), clamp(pk, 0, 1.2))
        put(main, fit(l1, 'MontXB', 32, ICE, 820), 515, y + 25, a * q)
        put(main, fit(l2, 'MontXB', 32, ICE, 820), 515, y + 85, a * q)
        put(main, fit(ref, 'MontXB', 28, SUB, 820), 515, y + 150, a * q)
    q = eo(prog(t, 35.2, 35.7))
    label(main, 'Not tested in these trials:', W / 2, 1610, a * q, WARN, 31, 'MontBlack', 0)
    label(main, 'incongruent mortise, fracture-dislocation', W / 2, 1655, a * q, WARN, 31, 'MontBlack', 0)

TIPS = [('Tibiofibular clear space under 6 mm', 'MORTISE VIEW: OVERLAP OVER 1 MM', 'Harper & Keller, Foot Ankle 1989'),
        ('Operating anyway? Stress it again', 'SYNDESMOSIS UNSTABLE IN 45% AFTER FIXATION', 'Tornetta et al. J Orthop Trauma 2012'),
        ('A plate has a price', '9 OF 64 HAD HARDWARE REMOVED, 2 INFECTED', 'SUPER-FIN, BMJ 2026')]
def s7(t, main, glow, fx):
    a = win(t, 37.6, 43.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 37.7, 38.2)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 38.2 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 42, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = fit(ref, 'MontXB', 29, SKY, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s8(t, main, glow, fx):
    a = win(t, 43.4, 50.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 290, a * eo(prog(t, 43.5, 44.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 43.7, 44.3))
    big = fit('CONGRUENT B: CAST IT', 'MontBlack', 84, WHITE, 880)
    put(glow, big, W / 2, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Stable: 3 weeks is enough', W / 2, 505, a * eo(prog(t, 44.1, 44.6)), GOOD, 36, 'MontBlack', 0)
    label(main, 'Stress-unstable: cast = plate at 2 years', W / 2, 570, a * eo(prog(t, 44.4, 44.9)), SKY, 36, 'MontBlack', 0)
    q = eo(prog(t, 44.8, 45.4))
    refs = ['Kortekangas T, et al. BMJ 2026;392:e085295', 'Kortekangas T, et al. BMJ 2019;364:k5432',
            'Kellam PJ, et al. J Orthop Trauma 2021;35(9):473–8', 'Harper MC, Keller TS. Foot Ankle 1989;10(3):156–60',
            'Tornetta P, et al. J Orthop Trauma 2012;26(11):659–61', 'Weber BG. Huber, Bern 1972 (Danis-Weber)']
    put(main, card(840, 500, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 930, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 29, ICE if i % 2 == 0 else SUB, 730), 515, 755 + i * 70, a * q)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1330, a * eo(prog(t, 45.4, 46.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7, s8]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
