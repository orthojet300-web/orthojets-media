"""Orthojets EP28 - DON'T MISS #7 - Posterior shoulder dislocation: the light bulb sign and why it is missed.
Sources (verified on PubMed abstracts / metadata, 2026-11-17; PubMed is the only verification source):
  Hawkins RJ, Neer CS, Pianta RM, Mendoza FX. J Bone Joint Surg Am 1987;69(1):9-18. PMID 3805075 (no DOI in PubMed).
    40 patients, 41 locked posterior dislocations: diagnosis missed by the initial physician in the majority; causes:
    motor-vehicle accident, seizure, alcohol-related injury, electroshock therapy; average interval injury to diagnosis
    ONE YEAR; an axillary radiograph confirmed the diagnosis in ALL shoulders and showed the size of the impression defect;
    undisplaced proximal humerus fracture in 20 of the 41 shoulders.
  Rowe CR, Zarins B. J Bone Joint Surg Am 1982;64(4):494-505. PMID 7068692. Chronic unreduced dislocations (at least
    3 weeks): 14 posterior; 79% of the posterior dislocations had not been recognized by the initial treating physician.
  Robinson CM, Seah M, Akhtar MA. J Bone Joint Surg Am 2011;93(17):1605-13. PMID 21915575, doi 10.2106/JBJS.J.00973.
    112 patients, 120 posterior dislocations. 1.1 per 100,000 per year; peaks in men 20-49 y and in patients over 70 y;
    67% traumatic accident, most of the remainder seizures. Recurrent instability 17.7% within the first year; predictors:
    age under 40, dislocation during a seizure, large reverse Hill-Sachs lesion (over 1.5 cm3).
  Magnussen AP et al. BMJ Open Qual 2020;9(1):e000550. PMID 32193195, doi 10.1136/bmjoq-2018-000550. Missed more often
    with a concurrent proximal humerus fracture; AP and scapular Y views not always diagnostic; axial views more reliable;
    protocol: routine axial / modified trauma axial view for every ED shoulder injury, low threshold for CT.
  Espag MP et al. Ann R Coll Surg Engl 2002;84(5):334-7. PMID 12398127, doi 10.1308/003588402760452466. Survey of 40
    casualty officers: none aware of the radiographic signs of posterior dislocation; claimed always to request two views
    but did not in 75% of cases; all would accept one view to exclude dislocation.
  Mimura T et al. J Orthop Surg (Hong Kong) 2006;14(3):336-9. PMID 17200541, doi 10.1177/230949900601400321: light-bulb
    sign on the AP view; overlap of head and glenoid rim on a true AP; multidirectional radiographs + CT recommended.
  Koutserimpas C et al. Cureus 2023;15(10):e47800. PMID 38022099, doi 10.7759/cureus.47800: after a seizure, arm locked in
    internal rotation, initial X-rays without evident malalignment but showing the light-bulb sign; CT: reverse Hill-Sachs.
  Postacchini F, Facchini M. Ital J Orthop Traumatol 1987;13(1):15-26. PMID 3692794: posterior dislocation suspected
    clinically by the limb in internal rotation and loss of external rotation; easily missed on AP films; definite
    diagnosis only with the lateral (axillary) projection.
  Diagrams: RIGHT shoulder. AP view: lateral on the viewer's left; acromion above the head, clavicle to the AC joint,
  coracoid projected superomedial to the glenoid, glenoid facing laterally, scapular body medial. Normal rotation: head
  medial, greater tuberosity in profile laterally (asymmetric "cane"); internal rotation: tuberosity turned en face, head
  symmetric on the neck = "light bulb". Axillary view: anterior (coracoid) up, posterior down, humeral shaft lateral;
  dislocated head behind the posterior glenoid rim with an anterior head impaction (reverse Hill-Sachs) locked on the rim.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(49.0, [(7.8, 1.6), (13.2, 1.2), (17.6, 1.4), (22.6, 1.6), (29.4, 1.8), (35.6, 1.6), (41.6, 1.8),
                     (46.4, 1.4)])

DW, DH = 760, 900
XRAY = (170, 210, 242)
SS = 2

def catmull(pts, n=10, closed=True):
    P = pts + pts[:3] if closed else [pts[0]] + pts + [pts[-1], pts[-1]]
    rng_ = range(len(pts)) if closed else range(len(pts) - 1)
    out = []
    for i in rng_:
        if closed: p0, p1, p2, p3 = pts[i - 1], P[i], P[i + 1], P[i + 2]
        else: p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t ** 3
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    if not closed: out.append(pts[-1])
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
FADE_AP = np.clip((DH - 10 - _yy) / 170.0, 0, 1) * np.clip((DW - 5 - _xx) / 90.0, 0, 1)
FADE_AX = np.clip((_xx - 10) / 150.0, 0, 1) * np.clip((DW - 5 - _xx) / 110.0, 0, 1)
def faded(m, f): return Image.fromarray((np.asarray(m, np.float32) * f).astype(np.uint8))
def union(ms):
    u = ms[0]
    for m in ms[1:]: u = ImageChops.lighter(u, m)
    return u

def finish(bone_layers, soft_m):
    soft = volume(soft_m, (64, 98, 140), alpha=0.5, r=18, relief=0.6)
    lay = Image.new('RGBA', (DW, DH)); lay.alpha_composite(soft)
    for b in bone_layers: lay.alpha_composite(b)
    pad = 40
    big = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); big.alpha_composite(lay, (pad, pad))
    big = drop_shadow(big, dx=10, dy=16, blur=14, opacity=0.55)
    return big.crop((pad, pad, pad + DW, pad + DH))

# ------------------------------------------------------------------ AP view, right shoulder (lateral = viewer's left)
CLAV = [(372, 200), (480, 190), (580, 168), (680, 138), (770, 124), (770, 170), (680, 184), (580, 214), (480, 236), (380, 240)]
ACRO = [(178, 236), (222, 206), (300, 198), (378, 204), (400, 226), (382, 250), (300, 254), (222, 260), (184, 252)]
SCAP = [(428, 298), (455, 282), (520, 284), (620, 298), (740, 318), (744, 372), (712, 500), (676, 640), (640, 780),
        (612, 880), (580, 884), (556, 820), (526, 700), (496, 590), (468, 520), (440, 486), (424, 466), (416, 428),
        (414, 388), (418, 344)]
SPINE = [(396, 236), (520, 262), (744, 318), (744, 346), (520, 292), (400, 258)]
CORA = [(438, 300), (428, 262), (444, 234), (482, 228), (504, 246), (488, 264), (466, 268), (462, 300)]
GLEN_RIM = [(428, 298), (418, 344), (414, 388), (416, 428), (424, 466)]
META_N = [(204, 398), (215, 500), (255, 560), (262, 620), (338, 620), (345, 560), (385, 495), (410, 450)]
META_R = [(214, 452), (240, 512), (260, 562), (262, 620), (338, 620), (340, 562), (360, 512), (386, 452)]
SHAFT = [(262, 600), (338, 600), (334, 910), (268, 910)]
def lerp(a, b, r): return a + (b - a) * r
def lp(p, q, r): return (lerp(p[0], q[0], r), lerp(p[1], q[1], r))

def ap_geom(r):
    head_c = lp((335, 385), (305, 388), r); head_r = 100
    gt_c = lp((232, 426), (300, 442), r); gt_rx, gt_ry = lerp(42, 48, r), lerp(56, 40, r)
    meta = [lp(p, q, r) for p, q in zip(META_N, META_R)]
    return head_c, head_r, gt_c, gt_rx, gt_ry, meta

_SC = None
def scap_layers():
    global _SC
    if _SC is None:
        ms = [faded(poly_mask(p), FADE_AP) for p in (SCAP, CLAV, ACRO, CORA)]
        sp = faded(poly_mask(SPINE), FADE_AP)
        _SC = (ms, [bone_xray(m, XRAY, alpha=0.95, seed=11 + i) for i, m in enumerate(ms)] +
               [bone_xray(sp, XRAY, alpha=0.35, seed=20)])
    return _SC

_AP = {}
def ap_diagram(r):
    k = round(r * 12) / 12
    if k in _AP: return _AP[k]
    head_c, head_r, gt_c, gt_rx, gt_ry, meta = ap_geom(k)
    hm = union([ell_mask(head_c, head_r, head_r), poly_mask(meta + [], False), faded(poly_mask(SHAFT, False), FADE_AP)])
    hm = faded(union([hm, ell_mask(gt_c, gt_rx, gt_ry)]) if k < 0.5 else hm, FADE_AP)
    gtm = ell_mask(gt_c, gt_rx, gt_ry)
    ms, layers = scap_layers()
    soft_m = union(ms + [hm]).filter(ImageFilter.MaxFilter(45)).filter(ImageFilter.GaussianBlur(20))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)), FADE_AP)
    hum = [bone_xray(hm, XRAY, alpha=0.96, seed=31)]
    if k >= 0.5: hum.append(bone_xray(gtm, XRAY, alpha=0.18 + 0.06 * k, seed=32))   # tuberosity en face: overlapping density
    _AP[k] = finish(layers + hum, soft_m)
    return _AP[k]

# ------------------------------------------------------------------ axillary view (anterior up, lateral = viewer's left)
AX_SCAP = [(446, 328), (474, 318), (530, 348), (610, 388), (770, 404), (770, 456), (610, 472), (530, 502), (474, 532),
           (446, 522), (433, 474), (428, 426), (433, 378)]
AX_CORA = [(470, 326), (456, 272), (432, 220), (402, 192), (380, 202), (396, 228), (420, 278), (446, 330)]
HEAD_N = (322, 425)
HEAD_D = (352, 618)
NOTCH = (436, 556)                 # reverse Hill-Sachs: anterior head impaction locked on the posterior glenoid rim
POST_RIM = (446, 522)

def build_ax():
    sm = faded(poly_mask(AX_SCAP), FADE_AX); cm = faded(poly_mask(AX_CORA), FADE_AX)
    hb = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(hb)
    cx, cy, rr = HEAD_D[0] * SS, HEAD_D[1] * SS, 100 * SS
    d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=255)
    d.rectangle((0, (HEAD_D[1] - 36) * SS, (HEAD_D[0] - 40) * SS, (HEAD_D[1] + 36) * SS), fill=255)    # shaft, lateral
    d.polygon([((HEAD_D[0] - 120) * SS, (HEAD_D[1] - 60) * SS), ((HEAD_D[0] - 40) * SS, (HEAD_D[1] - 80) * SS),
               ((HEAD_D[0] - 40) * SS, (HEAD_D[1] + 80) * SS), ((HEAD_D[0] - 120) * SS, (HEAD_D[1] + 60) * SS)], fill=255)
    nx, ny = NOTCH
    d.polygon([((nx - 44) * SS, (ny - 52) * SS), ((nx + 60) * SS, (ny - 70) * SS), ((nx + 50) * SS, (ny + 40) * SS),
               ((nx - 16) * SS, (ny + 14) * SS)], fill=0)                                             # wedge impaction
    hm = faded(hb.resize((DW, DH), Image.LANCZOS), FADE_AX)
    soft_m = union([sm, cm, hm]).filter(ImageFilter.MaxFilter(45)).filter(ImageFilter.GaussianBlur(20))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)), FADE_AX)
    return finish([bone_xray(sm, XRAY, alpha=0.95, seed=41), bone_xray(cm, XRAY, alpha=0.95, seed=42),
                   bone_xray(hm, XRAY, alpha=0.96, seed=43)], soft_m)

AXD = build_ax()

_dc = {}
def scaled(img, s, key):
    k = (key, round(s, 3))
    if k not in _dc:
        _dc[k] = img.resize((int(DW * s), int(DH * s)), Image.LANCZOS) if abs(s - 1) > 1e-3 else img
    return _dc[k]

def dp(q, cx, cy, s): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

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
    d.line([tip, (ex, ly)], fill=ICE + (int(210 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay); put(main, lab, lx, ly, a)

def bulb_outline(main, glow, r, cx, cy, s, a, col):
    """glowing outline of the humeral head + neck (the 'light bulb')"""
    if a <= 0: return
    head_c, head_r, gt_c, gt_rx, gt_ry, meta = ap_geom(r)
    hc = dp(head_c, cx, cy, s); R_ = head_r * s
    lay = aa_image(W, H); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    d.arc((hc[0] - R_, hc[1] - R_, hc[0] + R_, hc[1] + R_), 128, 412, fill=col + (int(255 * a),), width=6)
    l0, r0 = dp(meta[0], cx, cy, s), dp(meta[-1], cx, cy, s)
    l1, r1 = dp(meta[2], cx, cy, s), dp(meta[5], cx, cy, s)
    d.line([l0, l1], fill=col + (int(255 * a),), width=6); d.line([r0, r1], fill=col + (int(255 * a),), width=6)
    g.ellipse((hc[0] - R_, hc[1] - R_, hc[0] + R_, hc[1] + R_), outline=col + (int(130 * a),), width=24)
    main.alpha_composite(lay)

def arc_arrow(main, glow, c, r, a0, a1, col, a, w=9):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    n = 24; pts = [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
                   for i in range(n + 1)]
    d.line(pts[:-2], fill=col + (int(255 * a),), width=w, joint='curve')
    g.line(pts, fill=col + (int(110 * a),), width=24)
    (x0, y0), (x1, y1) = pts[-4], pts[-1]; L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
    bx, by = x1 - ux * 34, y1 - uy * 34
    d.polygon([(x1 + ux * 6, y1 + uy * 6), (bx - uy * 22, by + ux * 22), (bx + uy * 22, by - ux * 22)], fill=col + (int(255 * a),))
    main.alpha_composite(lay)

def dashed_circle(main, c, R_, col, a, w=4, n=30):
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    for i in range(0, n, 2):
        d.arc((c[0] - R_, c[1] - R_, c[0] + R_, c[1] + R_), 360 * i / n, 360 * (i + 1) / n, fill=col + (int(220 * a),), width=w)
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
XC = 515
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #7", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Seizure, shoulder pain, X-ray "normal":', 'MontXB', 42, ICE, 880), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('DISLOCATED?', 'MontBlack', 130, WHITE, 880)
    put(glow, k1, W / 2, 650, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 650, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('POSTERIOR SHOULDER DISLOCATION', 'MontBlack', 50, SKY, 880, track=1), W / 2, 790, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('the light bulb sign on the AP view', 'MontXB', 42, WHITE), W / 2, 880 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); cx, cy, s = 500, 1310, 0.72
    put(main, scaled(ap_diagram(1.0), s, 'ap1'), cx, cy + 20 * (1 - hq), a * hq)
    pul = 0.65 + 0.35 * math.sin(t * 6)
    bulb_outline(main, glow, 1.0, cx, cy + 20 * (1 - hq), s, a * eo(prog(t, 0.3, 0.7)) * pul, SKY)

def s2(t, main, glow, fx):                                     # who: Robinson 2011
    a = win(t, 3.3, 9.0, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('WHO IS AT RISK?', 'MontBlack', 62, WHITE, 840, track=3), XC, 262, a * p, 1, 10 * (1 - p))
    label(main, '120 posterior dislocations  ·  Edinburgh', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    q = eo(prog(t, 3.8, 4.3))
    if q > 0:
        y = 520
        put(main, card(880, 230), XC, y, a * q)
        nb = tl('1.1', 'MontBlack', 110, SKY); put(main, nb, 125 + nb.width / 2, y, a * q)
        x0 = 125 + nb.width + 30
        im = fit('per 100,000 per year', 'MontBlack', 42, WHITE, 900 - x0); put(main, im, x0 + im.width / 2, y - 30, a * q)
        im = fit('rare, so nobody expects it', 'MontXB', 32, SUB, 900 - x0); put(main, im, x0 + im.width / 2, y + 36, a * q)
    q = eo(prog(t, 4.6, 5.1))
    if q > 0:
        y = 860; d = ImageDraw.Draw(main)
        label(main, 'CAUSE', XC, y - 120, a * q, ICE, 36, 'MontBlack', 4)
        g_ = eio(prog(t, 4.8, 5.8))
        x0, x1 = 120, 910; xm = x0 + (x1 - x0) * 0.67 * g_
        d.rounded_rectangle((x0, y - 40, x1, y + 40), 22, fill=(60, 90, 125, int(200 * a * q)))
        d.rounded_rectangle((x0, y - 40, max(x0 + 44, xm), y + 40), 22, fill=SKY + (int(255 * a * q),))
        put(main, tl(f'{round(67 * g_)}%', 'MontBlack', 50, NAVY), x0 + 90, y, a * q)
        label(main, 'traumatic accident', x0 + 180, y + 90, a * q, SKY, 34, 'MontXB', 0)
        qq = eo(prog(t, 5.8, 6.3))
        label(main, 'rest: mostly SEIZURES', x1 - 200, y + 150, a * q * qq, WARN, 32, 'MontBlack', 1)
    q = eo(prog(t, 6.4, 6.9))
    if q > 0:
        y = 1240 + 20 * (1 - q)
        put(main, card(880, 260, outline=SKY + (150,)), XC, y, a * q)
        label(main, 'Peaks:', XC, y - 76, a * q, SKY, 36, 'MontXB', 0)
        label(main, 'men aged 20–49', XC, y - 10, a * q, WHITE, 48, 'MontBlack', 0)
        label(main, 'and patients over 70', XC, y + 60, a * q, WHITE, 48, 'MontBlack', 0)
    label(main, 'Robinson et al.  ·  JBJS Am 2011', XC, 1440, a * eo(prog(t, 6.6, 7.1)), SUB, 28, 'MontXB', 0)

CX, CY, S = 500, 860, 0.86
STEPS = [(9.2, 'NORMAL AP', 'tuberosity in profile, head medial: a "cane"'),
         (13.2, 'LOCKED IN INTERNAL ROTATION', 'no external rotation: suspect it at the bedside'),
         (15.0, 'LIGHT BULB SIGN', 'internally rotated head on the AP view'),
         (18.6, 'AXILLARY VIEW', 'head behind the glenoid: it confirms it')]
def s3(t, main, glow, fx):                                     # centrepiece: AP light bulb -> axillary view
    a = win(t, 8.8, 24.2, 0.35, 0.45)
    if a <= 0: return
    ax = eio(prog(t, 18.4, 19.2))
    q = eo(prog(t, 8.9, 9.4))
    put(main, tl('READ THE X-RAY', 'MontBlack', 62, WHITE, track=3), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Right shoulder  ·  AP view' if ax < 0.5 else 'Right shoulder  ·  axillary view', XC, 335,
          a * eo(prog(t, 9.1, 9.6)) * abs(1 - 2 * ax), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 9.0, 9.9)); cy = CY + 24 * (1 - bp)
    r = eio(prog(t, 13.3, 14.6))
    if ax < 1:
        put(main, scaled(ap_diagram(r), S, f'ap{round(r * 12)}'), CX, cy, a * bp * (1 - ax))
    if ax > 0:
        put(main, scaled(AXD, S, 'ax'), CX, cy + 20 * (1 - ax), a * ax)
    P = lambda q_: dp(q_, CX, cy, S)
    # step 1 labels (normal)
    la = a * eo(prog(t, 9.6, 10.0)) * (1 - eo(prog(t, 12.8, 13.2)))
    pointer(main, la, P((196, 450)), 390, 1290, 'GREATER TUBEROSITY', ICE, 30)
    pointer(main, la, P((250, 350)), 215, 810, 'HUMERAL HEAD', ICE, 28)
    pointer(main, la, P((418, 400)), 860, 1040, 'GLENOID', ICE, 30)
    pointer(main, la, P((300, 222)), 820, 520, 'ACROMION', ICE, 30)
    # step 2: internal rotation arrow around the shaft axis
    ia = a * eo(prog(t, 13.2, 13.6)) * (1 - eo(prog(t, 15.0, 15.4)))
    if ia > 0:
        arc_arrow(main, glow, P((300, 560)), 120 * S, 200, 340, WARN, ia)
        label(main, 'INTERNAL ROTATION', XC, 1290, ia, WARN, 36, 'MontBlack', 2)
    # step 3: light bulb outline
    ba = a * eo(prog(t, 15.0, 15.5)) * (1 - ax)
    if ba > 0:
        pul = 0.7 + 0.3 * math.sin(t * 5)
        bulb_outline(main, glow, 1.0, CX, cy, S, ba * pul, SKY)
        pointer(main, ba, P((305, 288)), 230, 520, 'LIGHT BULB', SKY, 34)
    # step 4: axillary labels
    xa = a * eo(prog(t, 19.0, 19.5))
    if xa > 0:
        dashed_circle(main, P(HEAD_N), 100 * S, ICE, xa * 0.8)
        label(main, 'normal', P(HEAD_N)[0], P(HEAD_N)[1], xa * 0.9, ICE, 30, 'MontXB', 0)
        pointer(main, xa, P((380, 196)), 760, 470, 'CORACOID', ICE, 30)
        pointer(main, xa, P((450, 470)), 860, 640, 'GLENOID', ICE, 30)
        label(main, 'ANTERIOR', 180, P((0, 250))[1], xa, SUB, 30, 'MontXB', 2)
        label(main, 'POSTERIOR', 190, P((0, 860))[1], xa, SUB, 30, 'MontXB', 2)
        na = a * eo(prog(t, 20.2, 20.6))
        if na > 0:
            pul = 0.6 + 0.4 * math.sin(t * 6); nc = P((NOTCH[0] + 8, NOTCH[1] - 14))
            dd = ImageDraw.Draw(main)
            dd.ellipse((nc[0] - 34, nc[1] - 34, nc[0] + 34, nc[1] + 34), outline=WARN + (int(255 * na * pul),), width=6)
            ImageDraw.Draw(glow).ellipse((nc[0] - 40, nc[1] - 40, nc[0] + 40, nc[1] + 40), fill=WARN + (int(90 * na),))
            pointer(main, na, (nc[0] + 8, nc[1] + 8), 760, 1160, 'REVERSE HILL-SACHS', WARN, 30)
            label(main, 'locked on the rim', 800, 1205, na, WARN, 28, 'MontXB', 0)
    # bottom panel
    pa = a * eo(prog(t, 9.6, 10.0))
    if pa > 0:
        y = 1450; cur = 0
        for i, (t0, _, _) in enumerate(STEPS):
            if t >= t0: cur = i
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        t0, nm, sub = STEPS[cur]; q = eo(prog(t, t0, t0 + 0.35))
        put(main, pill(f'{cur + 1} / 4', 'MontBlack', 36, NAVY, SKY + (255,), padx=22, h=64), 170, y - 40, pa * q)
        im = fit(nm, 'MontBlack', 44, WHITE, 650); put(main, im, 270 + im.width / 2, y - 40, pa * q)
        im = fit(sub, 'MontXB', 32, SKY, 790); put(main, im, 120 + im.width / 2, y + 50, pa * q)
        if cur == 3:
            label(main, 'Hawkins et al. JBJS Am 1987  ·  Postacchini 1987', XC, 1610, pa * q, SUB, 28, 'MontXB', 0)
        elif cur >= 1:
            label(main, 'Postacchini 1987  ·  Mimura 2006  ·  Koutserimpas 2023', XC, 1610, pa * q, SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # why it is missed
    a = win(t, 24.0, 30.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'WHY IT IS MISSED', XC, 262, a * eo(prog(t, 24.1, 24.6)), WHITE, 62, 'MontBlack', 3)
    label(main, 'the numbers behind the miss', XC, 340, a * eo(prog(t, 24.3, 24.8)), SKY, 36, 'MontXB', 0)
    rows = [(24.7, '79%', 'missed by the first doctor', 'chronic cases (Rowe 1982)', WARN),
            (25.6, '1 year', 'average injury-to-diagnosis', 'locked cases (Hawkins 1987)', WARN),
            (26.5, '0 / 40', 'ED doctors knew the X-ray signs', 'survey (Espag 2002)', WARN),
            (27.4, '75%', 'of cases: no 2nd view', 'though they claimed 2 (Espag)', WARN)]
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
    label(main, 'JBJS Am 1982  ·  JBJS Am 1987  ·  Ann R Coll Surg Engl 2002', XC, 1560,
          a * eo(prog(t, 28.0, 28.5)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # after reduction: recurrence
    a = win(t, 30.2, 36.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'AFTER REDUCTION', XC, 262, a * eo(prog(t, 30.3, 30.8)), WHITE, 62, 'MontBlack', 3)
    label(main, 'recurrent instability within 1 year', XC, 340, a * eo(prog(t, 30.5, 31.0)), SKY, 36, 'MontXB', 0)
    q = eo(prog(t, 30.8, 31.3))
    if q > 0:
        g_ = eio(prog(t, 31.0, 32.0)); cx, cy, R_ = XC, 640, 170
        lay = aa_image(W, H); d = ImageDraw.Draw(lay)
        d.ellipse((cx - R_, cy - R_, cx + R_, cy + R_), outline=(60, 90, 125, int(230 * a * q)), width=34)
        if g_ > 0.01:
            d.arc((cx - R_, cy - R_, cx + R_, cy + R_), -90, -90 + 360 * 0.177 * g_, fill=WARN + (int(255 * a * q),), width=34)
        main.alpha_composite(lay)
        ImageDraw.Draw(glow).arc((cx - R_, cy - R_, cx + R_, cy + R_), -90, -90 + 360 * 0.177 * g_, fill=WARN + (int(90 * a * q),), width=50)
        put(main, tl(f'{17.7 * g_:.1f}%', 'MontBlack', 84, WHITE), cx, cy, a * q)
    label(main, 'Higher risk if:', XC, 900, a * eo(prog(t, 32.0, 32.4)), SKY, 40, 'MontBlack', 1)
    risks = [(32.4, 'age under 40'), (33.0, 'dislocated during a seizure'), (33.6, 'large reverse Hill-Sachs, over 1.5 cm³')]
    for i, (t0, txt) in enumerate(risks):
        q = eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 1030 + i * 150 + 15 * (1 - q)
        put(main, card(880, 120, outline=WARN + (140,)), XC, y, a * q)
        dd = ImageDraw.Draw(main); dot(dd, 160, y, 14, WARN, a * q)
        im = fit(txt, 'MontBlack', 40, WHITE, 700); put(main, im, 200 + im.width / 2, y, a * q)
    label(main, 'Robinson et al.  ·  JBJS Am 2011  ·  120 dislocations', XC, 1500, a * eo(prog(t, 34.0, 34.5)), SUB, 28, 'MontXB', 0)

TIPS = [('Seizure? Think posterior', 'ARM LOCKED IN INTERNAL ROTATION', 'Postacchini 1987  ·  Koutserimpas 2023'),
        ('Always get the axial view', 'AP + SCAPULAR Y NOT ALWAYS DIAGNOSTIC', 'Magnussen et al. BMJ Open Qual 2020'),
        ('Proximal humerus fracture? Look again', 'MISSED MORE OFTEN  ·  LOW CT THRESHOLD', 'Magnussen 2020  ·  Hawkins 1987: 20 of 41')]
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
        lab = fit(h_, 'MontBlack', 40, WHITE, 690); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 28, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s7(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 42.4, 49.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 380, a * eo(prog(t, 42.5, 43.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 42.7, 43.3))
    big = fit('AXILLARY VIEW', 'MontBlack', 110, WHITE, 880)
    put(glow, big, XC, 520, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 520, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'confirmed all 41 locked dislocations', XC, 640, a * eo(prog(t, 43.1, 43.6)), GOOD, 40, 'MontBlack', 0)
    label(main, 'Light bulb on AP  ·  no external rotation', XC, 710, a * eo(prog(t, 43.4, 43.9)), SKY, 36, 'MontXB', 0)
    q = eo(prog(t, 43.8, 44.4))
    refs = ['Hawkins RJ, et al. JBJS Am 1987;69:9–18', 'Rowe CR, Zarins B. JBJS Am 1982;64:494–505',
            'Robinson CM, et al. JBJS Am 2011;93:1605–13', 'Magnussen AP, et al. BMJ Open Qual 2020;9:e000550',
            'Espag MP, et al. Ann R Coll Surg Engl 2002;84:334–7', 'Mimura T, et al. J Orthop Surg 2006;14:336–9',
            'Koutserimpas C, et al. Cureus 2023;15:e47800', 'Postacchini F, et al. Ital J Orthop Traumatol 1987;13:15–26']
    put(main, card(880, 580, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1070, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 820), XC, 846 + i * 64, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1440, a * eo(prog(t, 44.4, 45.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
