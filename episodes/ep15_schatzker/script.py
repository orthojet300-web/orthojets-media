"""Orthojets EP15 - CLASSIFIED #6 - Schatzker classification of tibial plateau fractures: fix now or span first?
Sources (verified on PubMed abstracts / metadata, 2026-11-15):
  Schatzker J, McBroom R, Bruce D. Clin Orthop Relat Res 1979;(138):94-104. PMID 445923 (no DOI; abstract not available).
    The original 6 types on the AP film: I lateral split, II lateral split-depression, III lateral pure depression,
    IV medial plateau, V bicondylar, VI plateau fracture with metaphyseal-diaphyseal dissociation (definitional).
  Gardner MJ, et al. J Orthop Trauma 2005;19(2):79-84. PMID 15677922, doi 10.1097/00005131-200502000-00002.
    MRI in 103 consecutive operative plateau fractures: only 1 (1%) had no soft tissue injury; 79 (77%) complete tear or avulsion
    of at least 1 cruciate or collateral ligament; 94 (91%) lateral meniscus pathology; 45 (44%) medial meniscus tears;
    70 (68%) posterolateral corner tears. Schatzker II = most frequent pattern (60%). No pure depression (Schatzker III) seen.
  Gardner MJ, et al. J Trauma 2006;60(2):319-23. PMID 16508489, doi 10.1097/01.ta.0000203548.50829.92. 62 Schatzker II:
    depression over 6 mm AND widening over 5 mm on the AP film: lateral meniscal injury 83% vs 50%.
  Ringus VM, et al. Orthopedics 2010;33(2):80-4. PMID 20192139, doi 10.3928/01477447-20100104-05. 85 lateral plateau
    fractures, CT: depression of 10 mm or more = 8-fold increase in risk of a lateral meniscus tear.
  Deng X, et al. J Orthop Surg Res 2021;16(1):502. PMID 34399816, doi 10.1186/s13018-021-02660-7. 1119 plateau fractures:
    acute compartment syndrome 35 (3.1%); multivariate predictors: younger age, Schatzker VI (OR 5.78).
  Gamulin A, et al. BMC Musculoskelet Disord 2017;18(1):307. PMID 28720096, doi 10.1186/s12891-017-1680-4. 269 fractures:
    ACS in 28 (10.4%); higher Schatzker grade (IV-V-VI) associated (age/sex adjusted); non-contiguous tibia fracture or knee
    dislocation and AO/OTA 41-C significant on multivariable analysis.
  Egol KA, et al. J Orthop Trauma 2005;19(7):448-55. PMID 16056075, doi 10.1097/01.bot.0000171881.11205.80. Staged protocol:
    knee-spanning external fixation on the day of admission, delayed definitive fixation; 57 high-energy fractures (74% VI,
    21% V); deep wound infection 3 (5%), nonunion 2 (4%).
  Yacoubian SV, et al. J Orthop Trauma 2002;16(9):632-7. PMID 12368643, doi 10.1097/00005131-200210000-00004. Adding CT changed
    the Schatzker class in 6%, MRI in 21%; MRI changed the treatment plan in 23%.
  Diagram: AP view of a RIGHT knee (lateral = viewer's left: fibular head under the lateral plateau; medial on viewer's right).
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(50.0, [(7.6, 1.4), (10.6, 1.0), (14.2, 1.0), (17.8, 1.0), (24.8, 1.6), (30.6, 1.2), (36.8, 1.8),
                     (42.8, 1.4)])

# ------------------------------------------------------------------ diagram: AP right knee (canvas units)
DW, DH = 640, 860
BONE = (226, 238, 250)
SURF = 361                                                        # plateau articular surface level

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
FEM_PTS = [(252, -40), (250, 120), (232, 190), (184, 232), (150, 266), (142, 302), (156, 328), (200, 340), (252, 338),
           (290, 324), (306, 302), (320, 294), (334, 302), (350, 324), (392, 338), (444, 340), (486, 328), (500, 302),
           (494, 268), (462, 232), (412, 190), (392, 120), (390, -40)]
TIB_PTS = [(132, 372), (142, 362), (200, SURF), (272, 362), (300, 350), (312, 338), (320, 350), (330, 338), (342, 350),
           (370, 362), (440, SURF), (500, 362), (512, 374), (508, 402), (490, 442), (460, 490), (432, 540), (412, 600),
           (406, 700), (404, 900), (282, 900), (280, 700), (272, 600), (250, 540), (204, 490), (164, 442), (138, 406)]
FIB_PTS = [(96, 432), (110, 414), (140, 410), (162, 424), (166, 452), (152, 482), (146, 522), (142, 700), (140, 900),
           (116, 900), (118, 700), (114, 522), (104, 482), (94, 456)]

_yy, _xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
def fade_bottom(m, y0=700, y1=830):
    return Image.fromarray((np.asarray(m, np.float32) * np.clip((y1 - _yy) / (y1 - y0), 0, 1)).astype(np.uint8))
def fade_top(m, y0=30, y1=200):
    return Image.fromarray((np.asarray(m, np.float32) * np.clip((_yy - y0) / (y1 - y0), 0, 1)).astype(np.uint8))

FEMB = bone_layer(fade_top(mask_poly(sm(FEM_PTS))))
TIBB = bone_layer(fade_bottom(mask_poly(sm(TIB_PTS))))
FIBB = bone_layer(fade_bottom(mask_poly(sm(FIB_PTS))), 205)

_dc = {}
def knee(s):
    k = round(s, 3)
    if k in _dc: return _dc[k]
    im = Image.new('RGBA', (DW, DH)); im.alpha_composite(FIBB); im.alpha_composite(TIBB); im.alpha_composite(FEMB)
    if s != 1: im = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    _dc[k] = im
    return im

def dp(cx, cy, s, q): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def frac(main, glow, cx, cy, s, a, pl, P0, P1, col=WARN):
    """fracture line drawn progressively (pl 0..1) with a slight kink"""
    if a <= 0 or pl <= 0: return
    mid = ((P0[0] + P1[0]) / 2 + 7, (P0[1] + P1[1]) / 2 - 5)
    pts = [P0, mid, P1]; seg = []; n = 12
    for i in range(n + 1):
        u = i / n * pl
        if u <= 0.5: q = (pts[0][0] + (pts[1][0] - pts[0][0]) * u * 2, pts[0][1] + (pts[1][1] - pts[0][1]) * u * 2)
        else: q = (pts[1][0] + (pts[2][0] - pts[1][0]) * (u - .5) * 2, pts[1][1] + (pts[2][1] - pts[1][1]) * (u - .5) * 2)
        seg.append(dp(cx, cy, s, q))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    w = max(6, int(6 * s))
    for P_, Q_ in zip(seg, seg[1:]): d.line([P_, Q_], fill=(14, 22, 38, int(255 * a)), width=int(w * 1.9))
    for P_, Q_ in zip(seg, seg[1:]):
        d.line([P_, Q_], fill=col + (int(255 * a),), width=w); g.line([P_, Q_], fill=col + (int(150 * a),), width=w * 3)
    main.alpha_composite(lay)

def depress(main, glow, cx, cy, s, a, x0, x1, depth, col=WARN):
    """articular depression: the original surface becomes a dashed line, the sunken fragment glows below it"""
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    hole = [dp(cx, cy, s, p) for p in [(x0, SURF - 6), (x1, SURF - 6), (x1 - 4, SURF + depth), (x0 + 4, SURF + depth)]]
    d.polygon(hole, fill=(16, 28, 48, int(235 * a)))
    X0, Y = dp(cx, cy, s, (x0, SURF)); X1, _ = dp(cx, cy, s, (x1, SURF)); x = X0
    while x < X1:
        d.line([(x, Y), (min(x + 10, X1), Y)], fill=ICE + (int(230 * a),), width=3); x += 18
    P = dp(cx, cy, s, (x0 + 4, SURF + depth)); Q = dp(cx, cy, s, (x1 - 4, SURF + depth))
    w = max(4, int(7 * s))
    d.line([P, Q], fill=col + (int(255 * a),), width=w)
    ImageDraw.Draw(glow).line([P, Q], fill=col + (int(150 * a),), width=w * 3)
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

# fracture geometry per type: (lines, depressions)
LAT = ((236, SURF), (170, 452)); MED = ((322, 350), (484, 456))
TYPE_GEO = {
    1: ([((226, SURF), (166, 446))], []),
    2: ([LAT], [(180, 236, 34)]),
    3: ([], [(192, 266, 34)]),
    4: ([MED], []),
    5: ([((316, 350), (178, 462)), ((324, 350), (474, 462))], []),
    6: ([((206, 498), (448, 522)), ((320, 350), (330, 506))], []),
}
def draw_type(main, glow, cx, cy, s, a, n, pl, col):
    lines, deps = TYPE_GEO[n]
    for P0, P1 in lines: frac(main, glow, cx, cy, s, a, pl, P0, P1, col)
    for x0, x1, dep in deps: depress(main, glow, cx, cy, s, a * clamp(pl * 1.5), x0, x1, dep * eo(pl), col)

DCX, DCY, DS = 560, 860, 1.22                                     # main diagram placement

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #6', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 400, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Tibial plateau fracture, swollen knee:', 'MontXB', 40, ICE, 860), W / 2, 500, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FIX OR SPAN?', 'MontBlack', 128, WHITE, 860)
    put(glow, k1, W / 2, 630, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 630, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, tl('SCHATZKER  I – VI', 'MontBlack', 66, SKY, track=2), W / 2, 770, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('The type tells you', 'MontBlack', 46, WHITE), W / 2, 880 + 25 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.3)); cx, cy, s = 540, 1330, 0.8
    put(main, knee(s), cx, cy + 20 * (1 - hq), a * hq)
    draw_type(main, glow, cx, cy + 20 * (1 - hq), s, a * hq, 6, eio(prog(t, 0.2, 1.1)), WARN)

def s2(t, main, glow, fx):                                     # context: normal AP knee
    a = win(t, 3.3, 8.8, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('THE TIBIAL PLATEAU', 'MontBlack', 62, WHITE, track=3), W / 2, 262, a * p, 1, 10 * (1 - p))
    label(main, 'AP view, right knee', W / 2, 340, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s = DCX, DCY, DS
    q = eo(prog(t, 3.6, 4.3))
    put(main, knee(s), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), dp(cx, cy, s, (320, 140)), 790, 470, 'FEMUR', SKY, 34)
    pointer(main, a * eo(prog(t, 4.5, 5.0)), dp(cx, cy, s, (200, 380)), 160, 720, 'LATERAL', SKY, 34)
    pointer(main, a * eo(prog(t, 4.7, 5.2)), dp(cx, cy, s, (450, 380)), 860, 900, 'MEDIAL', SKY, 34)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), dp(cx, cy, s, (128, 450)), 170, 1060, 'FIBULA', ICE, 30)
    pointer(main, a * eo(prog(t, 5.1, 5.6)), dp(cx, cy, s, (330, 520)), 820, 1180, 'METAPHYSIS', ICE, 30)
    q = eo(prog(t, 6.0, 6.5))
    if q > 0:
        y = 1470 + 20 * (1 - q)
        put(main, card(860, 210, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'SCHATZKER: 6 TYPES', 515, y - 45, a * q, SKY, 40, 'MontBlack', 1)
        label(main, 'lateral to medial, low to high energy', 515, y + 30, a * q, WHITE, 36, 'MontXB', 0)

TYPES = [(1, 'I', 'LATERAL SPLIT', 'a wedge of the lateral plateau', SKY, 9.0),
         (2, 'II', 'SPLIT-DEPRESSION', 'the commonest: 60% of operative cases', SKY, 10.8),
         (3, 'III', 'PURE DEPRESSION', 'none seen in 103 operative patients', SKY, 12.6),
         (4, 'IV', 'MEDIAL PLATEAU', 'the medial condyle breaks', WARN, 14.4),
         (5, 'V', 'BICONDYLAR', 'both plateaus, metaphysis intact', WARN, 16.2),
         (6, 'VI', 'METAPHYSEAL DISSOCIATION', 'the plateau is separated from the shaft', WARN, 18.0)]
def s3(t, main, glow, fx):                                     # centrepiece: I to VI drawn one by one
    a = win(t, 8.6, 20.0, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = DCX, DCY - 30, 1.12
    put(main, knee(s), cx, cy, a)
    for i, (n, rn, desc, sub, col, t0) in enumerate(TYPES):
        t1 = TYPES[i + 1][5] if i < 5 else 20.0
        ta = a * win(t, t0 - 0.3, t1 + 0.05, 0.3 if i else 0.35, 0.25 if i < 5 else 0.4)
        if ta <= 0: continue
        draw_type(main, glow, cx, cy, s, ta, n, eio(prog(t, t0 + 0.15, t0 + 0.8)), col)
        p = R.eback(prog(t, t0 + 0.05, t0 + 0.5))
        big = tl(f'TYPE {rn}', 'MontBlack', 100, WHITE)
        put(glow, big, W / 2, 265, ta * 0.25 * clamp(p)); put(main, big, W / 2, 265, ta * clamp(p), clamp(p, 0, 1.2))
        dq = eo(prog(t, t0 + 0.35, t0 + 0.8))
        put(main, pill(desc, 'MontBlack', 36 if len(desc) < 20 else 32, NAVY, col + (255,), h=72), W / 2,
            1380 + 15 * (1 - dq), ta * dq)
        eq = eo(prog(t, t0 + 0.6, t0 + 1.0))
        put(main, fit(sub, 'MontXB', 38, WHITE, 880), W / 2, 1480 + 15 * (1 - eq), ta * eq)
    ea = a * eo(prog(t, 9.4, 9.9))
    lo = a * (0.6 + 0.4 * (t < 14.2)); hi = a * (0.6 + 0.4 * (t >= 14.2))
    label(main, 'I – III  low energy, lateral', 300, 1590, ea * lo, SKY, 30, 'MontBlack', 0)
    label(main, 'IV – VI  high energy', 770, 1590, ea * hi, WARN, 30, 'MontBlack', 0)
    label(main, 'Schatzker et al.  ·  Clin Orthop 1979', W / 2, 1660, a * eo(prog(t, 9.6, 10.1)), SUB, 28, 'MontXB', 0)

def leg_glow(main, glow, cx, cy, s, a):
    """pulsing tint over the leg below the plateau (compartments under pressure)"""
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(glow)
    P = dp(cx, cy, s, (80, 560)); Q = dp(cx, cy, s, (470, 800))
    d.rounded_rectangle((P[0], P[1], Q[0], Q[1]), 60, fill=WARN + (int(70 * a),))

def s4(t, main, glow, fx):                                     # IV to VI: compartment syndrome
    a = win(t, 19.8, 26.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'IV – VI: WATCH THE LEG', W / 2, 262, a * eo(prog(t, 19.9, 20.4)), WHITE, 58, 'MontBlack', 2)
    label(main, 'acute compartment syndrome', W / 2, 340, a * eo(prog(t, 20.1, 20.6)), WARN, 36, 'MontXB', 0)
    cx, cy, s = 540, 720, 0.78
    put(main, knee(s), cx, cy, a)
    draw_type(main, glow, cx, cy, s, a, 6, 1.0, WARN)
    leg_glow(main, glow, cx, cy, s, a * eo(prog(t, 20.6, 21.2)) * (0.6 + 0.4 * math.sin(t * 5)))
    rows = [(21.4, '3.1%', 'of 1119 plateau fractures', 'Schatzker VI: OR 5.78', 'Deng et al. J Orthop Surg Res 2021'),
            (22.9, '10.4%', 'of 269 plateau fractures', 'more with Schatzker IV – VI', 'Gamulin et al. BMC MSK Disord 2017  ·  age/sex-adjusted')]
    for i, (t0, num, l1, l2, ref) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 1170 + i * 260 + 20 * (1 - q)
        put(main, card(880, 236, outline=WARN + (150,)), 515, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        nb = tl(num, 'MontBlack', 84, WARN)
        put(main, nb, 120 + nb.width / 2, y - 20, a * q * clamp(pk), clamp(pk, 0, 1.2))
        x0 = 120 + nb.width + 10
        for k, (txt, col, sz) in enumerate([(l1, WHITE, 34), (l2, ICE, 32)]):
            im = fit(txt, 'MontXB', sz, col, 920 - x0); put(main, im, x0 + im.width / 2, y - 52 + 52 * k, a * q)
        im = fit(ref, 'MontXB', 28, SUB, 800); put(main, im, 515, y + 75, a * q)

def s5(t, main, glow, fx):                                     # span first: knee-spanning external fixator
    a = win(t, 25.8, 31.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'SWOLLEN? SPAN FIRST', W / 2, 262, a * eo(prog(t, 25.9, 26.4)), WHITE, 58, 'MontBlack', 2)
    label(main, 'knee-spanning external fixator', W / 2, 340, a * eo(prog(t, 26.1, 26.6)), SKY, 36, 'MontXB', 0)
    cx, cy, s = 520, 860, 1.05
    put(main, knee(s), cx, cy, a)
    draw_type(main, glow, cx, cy, s, a, 6, 1.0, WARN)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    pins = [(26.6, 70, 262), (26.8, 150, 262), (27.0, 650, 290), (27.2, 730, 290)]
    BX = 600
    for t0, y, xe in pins:
        pq = eio(prog(t, t0, t0 + 0.4))
        if pq <= 0: continue
        P = dp(cx, cy, s, (BX + 20, y)); Q = dp(cx, cy, s, (BX + 20 - (BX + 20 - xe) * pq, y))
        d.line([P, Q], fill=ICE + (int(250 * a),), width=9)
        d.line([P, Q], fill=(150, 165, 185, int(255 * a)), width=4)
    bq = eio(prog(t, 27.5, 28.1))
    if bq > 0:
        P = dp(cx, cy, s, (BX, 50)); Q = dp(cx, cy, s, (BX, 50 + 700 * bq))
        d.rounded_rectangle((P[0] - 13, P[1], P[0] + 13, Q[1]), 12, fill=SKY + (int(255 * a),))
        g.line([(P[0], P[1]), (Q[0], Q[1])], fill=SKY + (int(110 * a),), width=30)
        for _, y, _x in pins:
            if 50 + 700 * bq >= y:
                C = dp(cx, cy, s, (BX, y)); d.rectangle((C[0] - 22, C[1] - 16, C[0] + 22, C[1] + 16), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay)
    q = eo(prog(t, 28.6, 29.1))
    if q > 0:
        y = 1530 + 20 * (1 - q)
        put(main, card(880, 250, outline=SKY + (160,)), 515, y, a * q)
        label(main, 'Fix when the soft tissue allows', 515, y - 70, a * q, WHITE, 40, 'MontBlack', 0)
        label(main, 'deep infection 5%  ·  57 high-energy fractures', 515, y + 2, a * q, SKY, 33, 'MontBlack', 0)
        label(main, 'Egol et al.  ·  J Orthop Trauma 2005', 515, y + 70, a * q, SUB, 28, 'MontXB', 0)

BARS = [('Lateral meniscus', 91, WARN), ('Cruciate or collateral tear', 77, WARN), ('Posterolateral corner', 68, WARN),
        ('Medial meniscus', 44, WARN), ('No soft tissue injury', 1, GOOD)]
def s6(t, main, glow, fx):                                     # the X-ray hides the soft tissue
    a = win(t, 31.4, 37.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'THE X-RAY HIDES THIS', W / 2, 262, a * eo(prog(t, 31.5, 32.0)), WHITE, 62, 'MontBlack', 2)
    label(main, 'MRI in 103 operative plateau fractures', W / 2, 340, a * eo(prog(t, 31.7, 32.2)), SKY, 36, 'MontXB', 0)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    for i, (txt, pct, col) in enumerate(BARS):
        t0 = 32.2 + 0.45 * i; q = eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 520 + i * 205
        im = tl(txt, 'MontXB', 36, ICE); put(main, im, 110 + im.width / 2 - 20, y - 50, a * q)
        g = eio(prog(t, t0 + 0.2, t0 + 1.1))
        d.rounded_rectangle((110, y + 12, 930, y + 62), 25, fill=(30, 54, 84, int(200 * a * q)))
        xe = 110 + max(50, 820 * pct / 100 * g)
        d.rounded_rectangle((110, y + 12, xe, y + 62), 25, fill=col + (int(255 * a * q),))
        ImageDraw.Draw(glow).rounded_rectangle((110, y + 12, xe, y + 62), 25, fill=col + (int(70 * a * q),))
        nb = tl(f'{round(pct * g)}%', 'MontBlack', 50, col)
        put(main, nb, 930 - nb.width / 2 + 20, y - 50, a * q)
    main.alpha_composite(lay)
    label(main, 'Gardner et al.  ·  J Orthop Trauma 2005', W / 2, 1560, a * eo(prog(t, 33.5, 34.0)), SUB, 30, 'MontXB', 0)

TIPS = [('Over 6 mm down, 5 mm wide?', 'LATERAL MENISCUS INJURY 83% VS 50%', 'Gardner et al. J Trauma 2006'),
        ('10 mm or more? Plan to repair', '8-FOLD RISK OF A LATERAL MENISCUS TEAR', 'Ringus et al. Orthopedics 2010'),
        ('Add the MRI before you plan', 'IT CHANGED THE TREATMENT PLAN IN 23%', 'Yacoubian et al. J Orthop Trauma 2002')]
def s7(t, main, glow, fx):
    a = win(t, 37.6, 43.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 37.7, 38.2)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 38.2 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
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
    big = fit('IV – VI: SPAN FIRST', 'MontBlack', 84, WHITE, 860)
    put(glow, big, W / 2, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'and watch the compartments', W / 2, 500, a * eo(prog(t, 44.1, 44.6)), WARN, 36, 'MontBlack', 0)
    label(main, 'I – III: measure the depression', W / 2, 565, a * eo(prog(t, 44.4, 44.9)), SKY, 36, 'MontBlack', 0)
    q = eo(prog(t, 44.8, 45.4))
    refs = ['Schatzker J, et al. Clin Orthop 1979;(138):94–104', 'Gardner MJ, et al. J Orthop Trauma 2005;19(2):79–84',
            'Gardner MJ, et al. J Trauma 2006;60(2):319–23', 'Deng X, et al. J Orthop Surg Res 2021;16:502',
            'Gamulin A, et al. BMC Musculoskelet Disord 2017;18:307', 'Egol KA, et al. J Orthop Trauma 2005;19(7):448–55',
            'Ringus VM, et al. Orthopedics 2010;33(2):80–4', 'Yacoubian SV, et al. J Orthop Trauma 2002;16(9):632–7']
    put(main, card(860, 560, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 945, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 780), 515, 728 + i * 62, a * q)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1330, a * eo(prog(t, 45.4, 46.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7, s8]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
