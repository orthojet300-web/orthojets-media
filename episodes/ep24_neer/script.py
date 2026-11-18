"""Orthojets EP24 - CLASSIFIED #7 - Neer classification of proximal humerus fractures: fix or sling?
Sources (verified on PubMed abstracts / metadata / PMC full text, 2026-11-18; PubMed is the only verification source):
  Neer CS. Displaced proximal humeral fractures. I. Classification and evaluation. J Bone Joint Surg Am 1970;52(6):1077-89.
    PMID 5455339 (no abstract in PubMed; reprinted Clin Orthop Relat Res 2006;442:77-82, PMID 16394743,
    doi 10.1097/01.blo.0000198718.91223.ca).
  Mitsuzawa S et al. J Orthop Surg Res 2025;20(1):18. PMID 39773752, PMC11705887, doi 10.1186/s13018-024-05423-2.
    Full text (PMC): "According to Neer's original classification, displacement was more than 1 cm of displacement or 45
    degrees of angulation"; the clinical version "identifies the number of displacements of the four main fragments";
    proximal humeral fracture about 6% of all fractures. Neer interobserver kappa 0.49 in their 100 cases.
  Court-Brown CM, Garg A, McQueen MM. Acta Orthop Scand 2001;72(4):365-71. PMID 11580125, doi 10.1080/000164701753542023.
    5-year prospective study, 1,027 proximal humeral fractures; highest age-specific incidence in women aged 80-89;
    commonest = B1.1 impacted valgus fracture, one-fifth of cases, a type NOT included in the Neer classification;
    in Neer, half of the fractures are minimally displaced and almost nine-tenths fall into only three categories.
  Sidor ML et al. J Bone Joint Surg Am 1993;75(12):1745-50. PMID 8258543, doi 10.2106/00004623-199312000-00002.
    50 fractures, trauma series (scapular AP, scapular lateral, axillary), 5 observers: all five agreed on the final
    classification for 32% and 30% of fractures; mean interobserver kappa 0.48 and 0.52; intraobserver mean 0.66.
  Hertel R et al. J Shoulder Elbow Surg 2004;13(4):427-33. PMID 15220884, doi 10.1016/j.jse.2004.01.034.
    100 intracapsular fractures: best predictors of head ischaemia = calcar (metaphyseal head extension) under 8 mm
    (accuracy 0.84), disrupted medial hinge (0.79); anatomic neck + short calcar + disrupted hinge combined: positive
    predictive value up to 97%.
  Rangan A et al. JAMA 2015;313(10):1037-47 (PROFHER). PMID 25756440, doi 10.1001/jama.2015.1629.
    250 adults (mean age 66), displaced fractures involving the surgical neck, 32 UK hospitals; surgery (fixation or
    head replacement) vs sling. Oxford Shoulder Score (0-48) over 2 years 39.07 vs 38.32, difference 0.75
    (95% CI -1.33 to 2.84), P = .48; MCID 5 points. Complications 30 vs 23 (P = .28); secondary shoulder surgery
    11 vs 11; mortality 9 vs 5 (P = .27).
  Diagram: RIGHT shoulder, AP view (lateral on the viewer's left), schematic. Neer's four segments: articular segment
  (head, above the anatomical neck), greater tuberosity (lateral), lesser tuberosity (drawn medially, where it projects
  in profile), shaft (below the surgical neck). The scapula / acromion / clavicle / coracoid geometry is reused from
  EP28 (checked there). Displacements are illustrative, not to scale: shaft medial (2-part surgical neck), greater
  tuberosity superior (3-part), head angulated 45 degrees + lesser tuberosity displaced (4-part).
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(49.0, [(10.8, 1.4), (13.8, 1.2), (17.6, 1.4), (22.8, 1.6), (29.4, 1.8), (35.6, 1.6), (41.6, 1.8),
                     (46.4, 1.4)])

DW, DH = 760, 900
XRAY = (170, 210, 242)
SS = 2
GOLD = (236, 196, 98)

def catmull(pts, n=10):
    P = pts + pts[:3]; out = []
    for i in range(len(pts)):
        p0, p1, p2, p3 = pts[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t ** 3
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    return out

def poly_mask(pts, smooth=True):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    pp = catmull(pts, 8) if smooth else pts
    d.polygon([(x * SS, y * SS) for x, y in pp], fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def ell_mask(c, rx, ry):
    big = Image.new('L', (DW * SS, DH * SS))
    ImageDraw.Draw(big).ellipse(((c[0] - rx) * SS, (c[1] - ry) * SS, (c[0] + rx) * SS, (c[1] + ry) * SS), fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

_yy = np.arange(DH, dtype=np.float32)[:, None] * np.ones((1, DW), np.float32)
_xx = np.ones((DH, 1), np.float32) * np.arange(DW, dtype=np.float32)[None, :]
FADE = np.clip((DH - 10 - _yy) / 170.0, 0, 1) * np.clip((DW - 5 - _xx) / 90.0, 0, 1)
def faded(m, f=FADE): return Image.fromarray((np.asarray(m, np.float32) * f).astype(np.uint8))
def union(ms):
    u = ms[0]
    for m in ms[1:]: u = ImageChops.lighter(u, m)
    return u
def arr(m): return np.asarray(m, np.float32) / 255
def tomask(a): return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))

# ------------------------------------------------------------------ geometry (right shoulder, AP, lateral = viewer's left)
CLAV = [(372, 200), (480, 190), (580, 168), (680, 138), (770, 124), (770, 170), (680, 184), (580, 214), (480, 236), (380, 240)]
ACRO = [(178, 236), (222, 206), (300, 198), (378, 204), (400, 226), (382, 250), (300, 254), (222, 260), (184, 252)]
SCAP = [(428, 298), (455, 282), (520, 284), (620, 298), (740, 318), (744, 372), (712, 500), (676, 640), (640, 780),
        (612, 880), (580, 884), (556, 820), (526, 700), (496, 590), (468, 520), (440, 486), (424, 466), (416, 428),
        (414, 388), (418, 344)]
SPINE = [(396, 236), (520, 262), (744, 318), (744, 346), (520, 292), (400, 258)]
CORA = [(438, 300), (428, 262), (444, 234), (482, 228), (504, 246), (488, 264), (466, 268), (462, 300)]
META = [(204, 398), (215, 500), (255, 560), (262, 620), (338, 620), (345, 560), (385, 495), (410, 450)]
SHAFT = [(262, 600), (338, 600), (334, 910), (268, 910)]
HEAD_C, HEAD_R = (335, 385), 100
GT_C, GT_RX, GT_RY = (232, 426), 42, 56
AN_A, AN_B = (276, 300), (410, 452)          # anatomical neck chord (superolateral -> inferomedial)
SN_A, SN_B = (226, 566), (372, 578)          # surgical neck
LT_C, LT_RX, LT_RY = (372, 506), 32, 40      # lesser tuberosity, drawn medially (in profile)

def _side(A, B):
    return (B[0] - A[0]) * (_yy - A[1]) - (B[1] - A[1]) * (_xx - A[0])

def build():
    scap_ms = [faded(poly_mask(p)) for p in (SCAP, CLAV, ACRO, CORA)]
    sp = faded(poly_mask(SPINE))
    hum = union([ell_mask(HEAD_C, HEAD_R, HEAD_R), poly_mask(META, False), poly_mask(SHAFT, False), ell_mask(GT_C, GT_RX, GT_RY)])
    hum = arr(faded(hum))
    sa = np.clip(_side(AN_A, AN_B) / -1.5, 0, 1)              # 1 on the articular (medial-superior) side
    ss = np.clip(_side(SN_A, SN_B) / 1.5 + 0.5, 0, 1)         # 1 below the surgical neck
    gap = 1.0                                                  # thin fracture gaps between segments
    head = hum * sa
    shaft = hum * ss * (1 - sa)
    block = hum * (1 - sa) * (1 - ss)
    lt = block * arr(ell_mask(LT_C, LT_RX, LT_RY))
    gt = block - lt
    def crisp(a):                                              # erode 1 px so fracture lines read as dark gaps
        return tomask(a).filter(ImageFilter.MinFilter(3))
    segs = {'head': crisp(head), 'gt': crisp(gt), 'lt': crisp(lt), 'shaft': crisp(shaft)}
    soft_m = union(scap_ms + [tomask(hum)]).filter(ImageFilter.MaxFilter(45)).filter(ImageFilter.GaussianBlur(20))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)))
    base = Image.new('RGBA', (DW, DH))
    base.alpha_composite(volume(soft_m, (64, 98, 140), alpha=0.5, r=18, relief=0.6))
    for i, m in enumerate(scap_ms): base.alpha_composite(bone_xray(m, XRAY, alpha=0.95, seed=11 + i))
    base.alpha_composite(bone_xray(sp, XRAY, alpha=0.35, seed=20))
    lay = {k: bone_xray(m, XRAY, alpha=0.97, seed=31 + i) for i, (k, m) in enumerate(segs.items())}
    return base, segs, lay

BASE, SEGM, SEGL = build()
CENT = {'head': (372, 352), 'gt': (238, 458), 'lt': (372, 506), 'shaft': (300, 760)}
TINT = {'head': SKY, 'gt': GOLD, 'lt': ICE, 'shaft': (120, 175, 216)}
NAMES = {'head': 'HEAD', 'gt': 'GREATER TUBEROSITY', 'lt': 'LESSER TUBEROSITY', 'shaft': 'SHAFT'}
ORDER = ['head', 'gt', 'lt', 'shaft']
DRAW = ['gt', 'lt', 'shaft', 'head']   # head on top

_rot = {}
def seg_layer(k, ang):
    a = round(ang / 3) * 3
    key = (k, a)
    if key not in _rot:
        im = SEGL[k]
        _rot[key] = im if a == 0 else im.rotate(a, resample=Image.BICUBIC, center=HEAD_C)
    return _rot[key]
_trot = {}
def tint_layer(k, ang, col):
    a = round(ang / 3) * 3
    key = (k, a, col)
    if key not in _trot:
        m = SEGM[k] if a == 0 else SEGM[k].rotate(a, resample=Image.BICUBIC, center=HEAD_C)
        im = Image.new('RGBA', (DW, DH), col + (0,)); im.putalpha(m.point(lambda v: int(v * 0.55)))
        _trot[key] = im
    return _trot[key]

def diagram(state, tints=None):
    """state: {seg: (dx, dy, ang)}; tints: {seg: alpha}"""
    pad = 60
    lay = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); lay.alpha_composite(BASE, (pad, pad))
    segs = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad))
    for k in DRAW:
        dx, dy, ang = state.get(k, (0, 0, 0))
        segs.alpha_composite(seg_layer(k, ang), (pad + int(round(dx)), pad + int(round(dy))))
        ta = (tints or {}).get(k, 0)
        if ta > 0.02:
            tl_ = tint_layer(k, ang, TINT[k])
            if ta < 0.98:
                tl_ = tl_.copy(); tl_.putalpha(tl_.getchannel('A').point(lambda v: int(v * ta)))
            segs.alpha_composite(tl_, (pad + int(round(dx)), pad + int(round(dy))))
    segs = drop_shadow(segs, dx=8, dy=12, blur=10, opacity=0.5)
    lay.alpha_composite(segs)
    return lay.crop((pad, pad, pad + DW, pad + DH))

_dcache = {}
def diagram_c(key, state, tints=None):
    if key not in _dcache: _dcache[key] = diagram(state, tints)
    return _dcache[key]

VIEW = (130, 190, 610, 900)                 # crop: proximal humerus + glenoid
VW, VH = VIEW[2] - VIEW[0], VIEW[3] - VIEW[1]
_vy = np.arange(VH, dtype=np.float32)[:, None]; _vx = np.arange(VW, dtype=np.float32)[None, :]
VFADE = (np.clip(_vy / 70.0, 0, 1) * np.clip(_vx / 60.0, 0, 1) * np.clip((VW - 1 - _vx) / 60.0, 0, 1)) ** 1.5
def scaled(img, s):
    im = img.crop(VIEW); a = np.asarray(im, np.float32).copy(); a[..., 3] *= VFADE
    return Image.fromarray(a.astype(np.uint8)).resize((int(VW * s), int(VH * s)), Image.LANCZOS)

def dp(q, cx, cy, s): return cx - VW * s / 2 + (q[0] - VIEW[0]) * s, cy - VH * s / 2 + (q[1] - VIEW[1]) * s

def fit(txt, font, size, col, maxw, track=0):
    im = tl(txt, font, size, col, track=track)
    while im.width > maxw and size > 28:
        size -= 2; im = tl(txt, font, size, col, track=track)
    return im

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=30):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14 if lx > tip[0] else lx + lab.width / 2 - 14
    d.line([tip, (ex, ly)], fill=col + (int(210 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=col + (int(255 * a),))
    main.alpha_composite(lay); put(main, lab, lx, ly, a)

def dashed(main, p, q, col, a, w=5, n=12, frac=1.0):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    for i in range(n):
        t0, t1 = i / n, (i + 0.6) / n
        if t0 > frac: break
        t1 = min(t1, frac)
        d.line([(p[0] + (q[0] - p[0]) * t0, p[1] + (q[1] - p[1]) * t0), (p[0] + (q[0] - p[0]) * t1, p[1] + (q[1] - p[1]) * t1)],
               fill=col + (int(255 * a),), width=w)
    main.alpha_composite(lay)

def badge(main, n, x, y, col, a):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=col + (int(240 * a),), outline=WHITE + (int(230 * a),), width=3)
    main.alpha_composite(lay); put(main, tl(str(n), 'MontBlack', 36, NAVY), x, y + 1, a)

def dim_line(main, glow, p, q, col, a, txt, lx, ly, size=36):
    """measurement bar with end ticks"""
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    d.line([p, q], fill=col + (int(255 * a),), width=5)
    L = math.hypot(q[0] - p[0], q[1] - p[1]) or 1; nx, ny = -(q[1] - p[1]) / L, (q[0] - p[0]) / L
    for e in (p, q): d.line([(e[0] - nx * 16, e[1] - ny * 16), (e[0] + nx * 16, e[1] + ny * 16)], fill=col + (int(255 * a),), width=5)
    ImageDraw.Draw(glow).line([p, q], fill=col + (int(90 * a),), width=18)
    main.alpha_composite(lay); put(main, pill(txt, 'MontBlack', size, col, (16, 30, 52, 235), padx=18, h=size + 26), lx, ly, a)

def angle_arc(main, glow, c, r, a0, a1, col, a, txt, lx, ly):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    lo, hi = min(a0, a1), max(a0, a1)
    d.arc((c[0] - r, c[1] - r, c[0] + r, c[1] + r), lo, hi, fill=col + (int(255 * a),), width=6)
    for ang in (a0, a1):
        d.line([c, (c[0] + (r + 30) * math.cos(math.radians(ang)), c[1] + (r + 30) * math.sin(math.radians(ang)))],
               fill=col + (int(220 * a),), width=4)
    ImageDraw.Draw(glow).arc((c[0] - r, c[1] - r, c[0] + r, c[1] + r), lo, hi, fill=col + (int(90 * a),), width=20)
    main.alpha_composite(lay); put(main, pill(txt, 'MontBlack', 40, col, (16, 30, 52, 235), padx=18, h=66), lx, ly, a)

XC = 515
# ------------------------------------------------------------------ scenes
ALLT = {k: 1.0 for k in ORDER}
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #7', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Displaced surgical neck, 70-year-old:', 'MontXB', 44, ICE, 880), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FIX OR SLING?', 'MontBlack', 124, WHITE, 900)
    put(glow, k1, W / 2, 650, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 650, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('NEER CLASSIFICATION', 'MontBlack', 56, SKY, 880, track=2), W / 2, 790, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('4 parts  ·  1 cm  ·  45°', 'MontXB', 44, WHITE), W / 2, 880 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); cx, cy, s = 540, 1345, 0.92
    put(main, scaled(diagram_c('cover', {}, ALLT), s), cx, cy + 20 * (1 - hq), a * hq)
    ba = a * eo(prog(t, 0.4, 0.8))
    for i, k in enumerate(ORDER):
        bx, by = dp(CENT[k], cx, cy + 20 * (1 - hq), s); badge(main, i + 1, bx, by, TINT[k], ba)

S2 = [(6.0, 'head'), (7.2, 'gt'), (8.4, 'lt'), (9.6, 'shaft')]
SUBS = {'head': 'articular segment, above the anatomical neck', 'gt': 'lateral, between the two necks',
        'lt': 'shown medially, where it projects in profile', 'shaft': 'below the surgical neck'}
CX, CY, S = 540, 900, 1.25
def s2(t, main, glow, fx):                                     # the 4 segments
    a = win(t, 3.3, 14.6, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('THE 4 PARTS', 'MontBlack', 64, WHITE, track=3), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Neer 1970  ·  right shoulder, AP view', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 3.5, 4.3)); cy = CY + 24 * (1 - bp)
    cur = None
    for t0, k in S2:
        if t >= t0: cur = k
    tints = {}
    for t0, k in S2:
        on = eo(prog(t, t0, t0 + 0.4))
        tints[k] = on * (1.0 if k == cur or t > 11.0 else 0.35)
    key = 's2_' + '_'.join(f'{round(tints[k] * 8)}' for k in ORDER)
    put(main, scaled(diagram_c(key, {}, {k: round(v * 8) / 8 for k, v in tints.items()}), S), CX, cy, a * bp)
    P = lambda q_: dp(q_, CX, cy, S)
    # fracture lines
    la = a * eo(prog(t, 4.2, 4.6)) * (1 - eo(prog(t, 10.6, 11.0)))
    dashed(main, P((AN_A[0] - 12, AN_A[1] - 14)), P((AN_B[0] + 12, AN_B[1] + 14)), WHITE, la, frac=prog(t, 4.2, 4.9))
    pointer(main, la, P((AN_A[0] - 6, AN_A[1] - 8)), 300, 410, 'ANATOMICAL NECK', ICE, 30)
    lb = a * eo(prog(t, 5.0, 5.4)) * (1 - eo(prog(t, 10.6, 11.0)))
    dashed(main, P((SN_A[0] + 26, SN_A[1] + 1)), P((SN_B[0] + 4, SN_B[1] + 1)), WHITE, lb, frac=prog(t, 5.0, 5.7))
    pointer(main, lb, P((258, 567)), 230, 1110, 'SURGICAL NECK', ICE, 30)
    # badges
    for i, (t0, k) in enumerate(S2):
        bq = eo(prog(t, t0, t0 + 0.35))
        bx, by = P(CENT[k]); badge(main, i + 1, bx, by, TINT[k], a * bq)
    # bottom panel
    pa = a * eo(prog(t, 5.9, 6.3))
    if pa > 0:
        y = 1450
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        if t < 11.0 and cur:
            i = ORDER.index(cur); t0 = S2[i][0]; q = eo(prog(t, t0, t0 + 0.35))
            put(main, pill(f'{i + 1} / 4', 'MontBlack', 36, NAVY, TINT[cur] + (255,), padx=22, h=64), 200, y - 40, pa * q)
            im = fit(NAMES[cur], 'MontBlack', 44, WHITE, 620); put(main, im, 300 + im.width / 2, y - 40, pa * q)
            im = fit(SUBS[cur], 'MontXB', 32, SKY, 790); put(main, im, 120 + im.width / 2, y + 50, pa * q)
        else:
            q = eo(prog(t, 11.0, 11.4))
            label(main, 'Count the DISPLACED segments', XC, y - 40, pa * q, WHITE, 44, 'MontBlack', 0)
            label(main, 'not the fracture lines', XC, y + 40, pa * q, SKY, 36, 'MontXB', 0)

# displacement program for s3 (diagram px; illustrative, not to scale)
def s3_state(t):
    sh = eio(prog(t, 15.4, 16.4)); g_ = eio(prog(t, 18.4, 19.2)); h = eio(prog(t, 20.6, 21.8))
    return {'shaft': (80 * sh, 6 * sh, 0), 'gt': (-30 * g_, -50 * g_, 0),
            'lt': (36 * h, 10 * h, 0), 'head': (30 * h, 62 * h, 45 * h)}, sh, g_, h

STEP3 = [(14.4, 'THE RULE', 'a segment is a PART only if displaced'),
         (15.4, '2-PART', 'surgical neck: shaft displaced over 1 cm'),
         (18.4, '3-PART', '+ greater tuberosity displaced'),
         (20.6, '4-PART', '+ head angulated over 45° + lesser tuberosity')]
def s3(t, main, glow, fx):                                     # the 1 cm / 45 degree rule, 2 -> 3 -> 4 parts
    a = win(t, 14.4, 24.2, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 14.5, 15.0))
    put(main, tl('WHEN IS IT A PART?', 'MontBlack', 60, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    rq = eo(prog(t, 14.6, 15.1))
    put(main, pill('over 1 cm  OR  over 45°', 'MontBlack', 40, NAVY, SKY + (255,), padx=30, h=72), XC, 345, a * rq)
    st, sh, g_, h = s3_state(t)
    ks = (round(sh * 10), round(g_ * 10), round(h * 15))
    state = {'shaft': (80 * ks[0] / 10, 6 * ks[0] / 10, 0), 'gt': (-30 * ks[1] / 10, -50 * ks[1] / 10, 0),
             'lt': (36 * ks[2] / 15, 10 * ks[2] / 15, 0), 'head': (30 * ks[2] / 15, 62 * ks[2] / 15, 45 * ks[2] / 15)}
    tints = {'shaft': 1.0 if sh > 0 else 0.25, 'gt': 1.0 if g_ > 0 else 0.25, 'lt': 1.0 if h > 0 else 0.25,
             'head': 1.0 if h > 0 else 0.25}
    key = 's3_%d_%d_%d' % ks
    put(main, scaled(diagram_c(key, state, tints), S), CX, CY, a)
    P = lambda q_: dp(q_, CX, CY, S)
    # 1 cm on the shaft
    m1 = a * eo(prog(t, 16.0, 16.4)) * (1 - eo(prog(t, 18.2, 18.5)))
    if m1 > 0:
        y = SN_B[1] + 40
        e = P((338 + 80 * sh, y)); dim_line(main, glow, P((338, y)), e, SKY, m1, 'over 1 cm', e[0] + 120, e[1], size=30)
    m2 = a * eo(prog(t, 18.9, 19.3)) * (1 - eo(prog(t, 20.4, 20.7)))
    if m2 > 0:
        x = 168
        p0, p1 = P((x, 470)), P((x, 470 - 50 * g_))
        dim_line(main, glow, p0, p1, SKY, m2, 'over 1 cm', p0[0] + 40, p0[1] + 55, size=30)
    m3 = a * eo(prog(t, 21.4, 21.8))
    if m3 > 0:
        c = P((HEAD_C[0] + 30 * h, HEAD_C[1] + 62 * h))
        angle_arc(main, glow, c, 90, -90, -90 + 45 * h, WHITE, m3, 'over 45°', c[0] + 20, c[1] - 165)
    # bottom panel
    pa = a * eo(prog(t, 14.6, 15.0))
    if pa > 0:
        y = 1450; cur = 0
        for i, (t0, _, _) in enumerate(STEP3):
            if t >= t0: cur = i
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        t0, nm, sub = STEP3[cur]; q = eo(prog(t, t0, t0 + 0.35))
        col = SKY
        im = fit(nm, 'MontBlack', 52, col, 700); put(main, im, 125 + im.width / 2, y - 40, pa * q)
        im = fit(sub, 'MontXB', 32, WHITE, 790); put(main, im, 125 + im.width / 2, y + 50, pa * q)
        label(main, 'Neer 1970  ·  Mitsuzawa et al. 2025', XC, 1610, pa, SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # reality check
    a = win(t, 24.0, 30.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REALITY CHECK', XC, 262, a * eo(prog(t, 24.1, 24.6)), WHITE, 62, 'MontBlack', 3)
    label(main, '1,027 fractures in 5 years  ·  Edinburgh', XC, 340, a * eo(prog(t, 24.3, 24.8)), SKY, 34, 'MontXB', 0)
    rows = [(24.7, '1 in 2', 'minimally displaced', 'one-part in the Neer system', SKY),
            (25.6, '80–89', 'peak incidence: women', 'fit elderly patients', SKY),
            (26.5, '1 in 5', 'valgus impacted (AO B1.1)', 'commonest; not in Neer 1970', SKY),
            (27.4, '32%', 'all 5 observers agreed', 'mean kappa 0.48 (Sidor 1993)', WARN)]
    for i, (t0, num, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 540 + i * 255 + 20 * (1 - q)
        put(main, card(880, 225, outline=col + (150,)), XC, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        nb = fit(num, 'MontBlack', 74, col, 260); put(main, nb, 125 + nb.width / 2, y, a * q * clamp(pk), clamp(pk, 0, 1.2))
        x0 = 125 + max(nb.width, 150) + 26
        im = fit(l1, 'MontBlack', 38, WHITE, 905 - x0); put(main, im, x0 + im.width / 2, y - 30, a * q)
        im = fit(l2, 'MontXB', 30, SUB, 905 - x0); put(main, im, x0 + im.width / 2, y + 34, a * q)
    label(main, 'Acta Orthop Scand 2001  ·  JBJS Am 1993', XC, 1560, a * eo(prog(t, 28.0, 28.5)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # PROFHER
    a = win(t, 30.2, 36.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'SURGERY vs SLING', XC, 262, a * eo(prog(t, 30.3, 30.8)), WHITE, 62, 'MontBlack', 3)
    label(main, 'PROFHER RCT  ·  250 adults  ·  mean age 66', XC, 340, a * eo(prog(t, 30.5, 31.0)), SKY, 34, 'MontXB', 0)
    label(main, 'displaced fractures involving the surgical neck', XC, 395, a * eo(prog(t, 30.6, 31.1)), ICE, 30, 'MontXB', 0)
    q = eo(prog(t, 30.9, 31.4))
    if q > 0:
        put(main, card(880, 470), XC, 690, a * q)
        label(main, 'Oxford Shoulder Score over 2 years (0–48)', XC, 510, a * q, SUB, 30, 'MontXB', 0)
        g_ = eio(prog(t, 31.2, 32.2)); d = ImageDraw.Draw(main)
        for i, (nm, v, col) in enumerate([('SURGERY', 39.07, SKY), ('SLING', 38.32, (120, 175, 216))]):
            y = 610 + i * 150; x0, x1 = 330, 880
            label(main, nm, 212, y, a * q, col, 36, 'MontBlack', 1)
            d.rounded_rectangle((x0, y - 34, x1, y + 34), 18, fill=(60, 90, 125, int(200 * a * q)))
            xm = x0 + (x1 - x0) * v / 48 * g_
            d.rounded_rectangle((x0, y - 34, max(x0 + 40, xm), y + 34), 18, fill=col + (int(255 * a * q),))
            put(main, tl(f'{v * g_:.2f}', 'MontBlack', 44, NAVY), max(x0 + 80, xm - 90), y, a * q)
        qq = eo(prog(t, 32.4, 32.9))
        label(main, 'difference 0.75 (needed: 5)  ·  P = .48', XC, 870, a * q * qq, WHITE, 34, 'MontBlack', 0)
    rows = [(33.2, '11 vs 11', 'secondary shoulder surgery'), (33.8, '30 vs 23', 'complications (P = .28)')]
    for i, (t0, num, txt) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 1070 + i * 160 + 15 * (1 - q)
        put(main, card(880, 130, outline=SKY + (140,)), XC, y, a * q)
        nb = tl(num, 'MontBlack', 50, SKY); put(main, nb, 125 + nb.width / 2, y, a * q)
        im = fit(txt, 'MontXB', 34, WHITE, 900 - (125 + nb.width + 30)); put(main, im, 125 + nb.width + 30 + im.width / 2, y, a * q)
    label(main, 'Rangan et al. JAMA 2015', XC, 1420, a * eo(prog(t, 34.2, 34.7)), SUB, 28, 'MontXB', 0)

TIPS = [('Surgical neck? Sling is an option', 'PROFHER: NO SIGNIFICANT DIFFERENCE AT 2 YEARS', 'Rangan et al. JAMA 2015'),
        ('Calcar under 8 mm + broken medial hinge', '+ ANATOMIC NECK: ISCHAEMIA PPV UP TO 97%', 'Hertel et al. J Shoulder Elbow Surg 2004'),
        ('Classify on a full trauma series', 'SCAPULAR AP + SCAPULAR LATERAL + AXILLARY', 'Sidor et al. JBJS Am 1993')]
def s6(t, main, glow, fx):
    a = win(t, 36.2, 42.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS & TRICKS', XC, 262, a * eo(prog(t, 36.3, 36.8)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 36.8 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
        put(main, card(880, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 40, WHITE, 660); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 740, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 28, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s7(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 42.4, 49.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 360, a * eo(prog(t, 42.5, 43.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 42.7, 43.3))
    big = fit('1 cm  ·  45°', 'MontBlack', 120, WHITE, 880)
    put(glow, big, XC, 500, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 500, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'or it is not a part', XC, 615, a * eo(prog(t, 43.1, 43.6)), SKY, 40, 'MontBlack', 0)
    label(main, 'PROFHER: sling vs surgery, no difference', XC, 690, a * eo(prog(t, 43.4, 43.9)), WHITE, 36, 'MontXB', 0)
    q = eo(prog(t, 43.8, 44.4))
    refs = ['Neer CS. JBJS Am 1970;52:1077–89', 'Rangan A, et al. JAMA 2015;313:1037–47',
            'Court-Brown CM, et al. Acta Orthop Scand 2001;72:365–71', 'Sidor ML, et al. JBJS Am 1993;75:1745–50',
            'Hertel R, et al. J Shoulder Elbow Surg 2004;13:427–33', 'Mitsuzawa S, et al. J Orthop Surg Res 2025;20:18']
    put(main, card(880, 460, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1000, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 820), XC, 840 + i * 64, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1320, a * eo(prog(t, 44.4, 45.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
