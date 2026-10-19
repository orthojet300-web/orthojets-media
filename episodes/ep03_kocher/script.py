"""Orthojets EP03 - CLINICAL RULES #1 - Kocher criteria: septic arthritis vs transient synovitis of the hip in children.
Sources (verified on PubMed, abstracts):
  Kocher MS, Zurakowski D, Kasser JR. JBJS Am 1999;81(12):1662-70. PMID 10608376, doi 10.2106/00004623-199912000-00002
    Retrospective, tertiary-care children's hospital, 1979-1996. Four independent multivariate predictors: history of
    fever, non-weight-bearing, ESR at least 40 mm/h, serum WBC over 12,000 cells/mm3. Predicted probability of septic
    arthritis: 0 predictors under 0.2%, 1 = 3.0%, 2 = 40.0%, 3 = 93.1%, 4 = 99.6%. Diagnoses defined by joint-fluid WBC,
    joint-fluid and blood cultures, and clinical course. Substantial overlap of single variables in intermediate ranges.
  Kocher MS et al. JBJS Am 2004;86(8):1629-35. PMID 15292409, doi 10.2106/00004623-200408000-00005
    Prospective validation 1997-2002: 51 septic arthritis, 103 transient synovitis; same 4 predictors; AUC 0.86 vs 0.96
    in the original population ("diminished, but nevertheless very good").
  Caird MS et al. JBJS Am 2006;88(6):1251-7. PMID 16757758, doi 10.2106/JBJS.E.00216
    Prospective, 48 children who underwent hip aspiration. Fever (oral temperature over 38.5 C) was the best predictor;
    CRP over 2.0 mg/dL (over 20 mg/L) a strong independent risk factor.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(9.0, 1.8), (18.4, 1.5), (19.9, 1.0), (27.3, 2.0), (29.6, 2.2), (37.9, 2.2), (44.6, 2.6)])

# ------------------------------------------------------------------ paediatric right hip, AP schematic
# local units 700 x 700; viewer's left = lateral. Head centre HC, ossified epiphysis radius 90.
SS = 3; LW, LH = 700, 700
HC = (330.0, 330.0); HR = 90; AR = 104
def pol(r, deg, c=HC): return (c[0] + r * math.cos(math.radians(deg)), c[1] + r * math.sin(math.radians(deg)))
def S(p): return (p[0] * SS, p[1] * SS)

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

NECK = (-math.cos(math.radians(40)), math.sin(math.radians(40)))      # head -> shaft, inferolateral
PERP = (NECK[1], -NECK[0])                                              # unused helper direction

def pelvis_poly():
    arc = [pol(AR, a) for a in range(200, 401, 5)]                      # acetabular roof, lateral -> medial-inferior
    outer = [arc[-1], (396, 520), (392, 620), (432, 680), (505, 690), (598, 652), (640, 565), (622, 474), (700, 436),
             (700, 362), (604, 334), (562, 252), (598, 172), (622, 92), (562, 44), (432, 20), (300, 26), (202, 62),
             (176, 122), (224, 198), (214, 242), arc[0]]
    return arc + catmull(outer, 8, False)[1:-1]

def epiphysis_poly():
    return [pol(HR, a) for a in range(230, 411, 4)]                     # cap beyond the physis chord (50..230 deg)

def femur_poly():
    c = (HC[0] + 13 * NECK[0], HC[1] + 13 * NECK[1])
    sup = pol(84, 230, c); inf = pol(84, 50, c)
    pts = [inf, (338, 446), (296, 482), (268, 502), (288, 528), (258, 562), (250, 610), (248, 700),
           (152, 700), (150, 560), (140, 452), (118, 384), (126, 322), (156, 306), (198, 330), (238, 306), sup]
    return catmull(pts, 8, False)

def capsule_poly(sw):
    """joint capsule: acetabular rim -> femoral neck base. sw 0..1 = effusion distension"""
    b = 34 * sw
    pts = [pol(AR + 16, 202), (238 - 0.4 * b, 300 - 0.6 * b), (212 - 0.5 * b, 342 - 0.2 * b), (226 - 0.3 * b, 420 + 0.3 * b),
           (276, 474 + 0.8 * b), (338 + 0.4 * b, 452 + 0.7 * b), (394 + 0.6 * b, 420 + 0.4 * b), pol(AR + 16, 40)]
    return catmull(pts, 10, False)

def poly_mask(pts, size=(LW * SS, LH * SS)):
    m = Image.new('L', size); ImageDraw.Draw(m).polygon([S(p) for p in pts], fill=255); return m

BONE = (226, 238, 250)
def bone_layer(mask, edge_w=4, a=235):
    lay = Image.new('RGBA', mask.size, BONE + (0,))
    inner = mask.filter(ImageFilter.GaussianBlur(10 * SS / 3))
    al = np.asarray(mask, np.float32) / 255
    core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.62 + 0.38 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(edge_w * SS // 3 * 2 + 1))
    e = Image.new('RGBA', mask.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: 255 if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

def build_hip():
    size = (LW * SS, LH * SS)
    pm = poly_mask(pelvis_poly())
    ImageDraw.Draw(pm).ellipse([S((466, 500)), S((580, 628))], fill=0)          # obturator foramen
    pel = bone_layer(pm)
    d = ImageDraw.Draw(pel)
    # triradiate cartilage (radiolucent Y in the acetabular floor)
    tri = [(pol(AR, 330)), (480, 318), (548, 292)]
    d.line([S(p) for p in tri], fill=(20, 34, 54, 255), width=6 * SS // 2, joint='curve')
    d.line([S((480, 318)), S((488, 392))], fill=(20, 34, 54, 255), width=6 * SS // 2)
    # femur: metaphysis/neck/shaft + separate epiphysis with a physis gap; greater trochanter apophysis line
    fem = bone_layer(poly_mask(femur_poly()))
    fd = ImageDraw.Draw(fem)
    fd.line([S((206, 334)), S((176, 372)), S((132, 384))], fill=(20, 34, 54, 255), width=5 * SS // 2, joint='curve')
    epi = bone_layer(poly_mask(epiphysis_poly()))
    return pel, fem, epi

PEL, FEM, EPI = build_hip()
BONEMASK = np.maximum(np.asarray(poly_mask(femur_poly()), np.float32), np.asarray(poly_mask(epiphysis_poly()), np.float32)) / 255
HIP_SCALE = 1.15
HW_, HH_ = int(LW * HIP_SCALE), int(LH * HIP_SCALE)
small = lambda im: im.resize((HW_, HH_), Image.LANCZOS)
PEL_S, FEM_S, EPI_S = small(PEL), small(FEM), small(EPI)
HIP_CX, HIP_CY = 515, 900
def hp(p): return HIP_CX - HW_ / 2 + p[0] * HIP_SCALE, HIP_CY - HH_ / 2 + p[1] * HIP_SCALE

_caps = {}
def capsule_layer(sw):
    k = round(sw, 2)
    if k in _caps: return _caps[k]
    size = (LW * SS, LH * SS)
    pts = capsule_poly(k)
    fill = Image.new('RGBA', size)
    d = ImageDraw.Draw(fill)
    d.polygon([S(p) for p in pts + [HC]], fill=WARN + (int(150 * k),))
    fa = np.asarray(fill.getchannel('A'), np.float32) * (1 - BONEMASK)
    fill.putalpha(Image.fromarray(fa.astype(np.uint8))); d = ImageDraw.Draw(fill)
    d.line([S(p) for p in pts], fill=(SKY if k < 0.5 else WARN) + (230,), width=5 * SS // 2, joint='curve')
    out = fill.resize((HW_, HH_), Image.LANCZOS)
    _caps[k] = out
    return out

HIP_LABELS = [('ACETABULUM', pol(AR, 250), (200, 560), 4.7),
              ('FEMORAL HEAD', pol(45, 290), (330, 475), 5.1),
              ('PHYSIS', (HC[0] + 12 * NECK[0] + 40 * math.cos(math.radians(50)), HC[1] + 12 * NECK[1] + 40 * math.sin(math.radians(50))), (190, 1120), 5.5),
              ('TRIRADIATE CARTILAGE', (548, 292), (760, 440), 5.9)]

def hip(t, main, glow, a, t0, swell):
    pa = eo(prog(t, t0, t0 + 0.7)); fa = eo(prog(t, t0 + 0.3, t0 + 1.0)); ea = eo(prog(t, t0 + 0.6, t0 + 1.2))
    put(main, PEL_S, HIP_CX, HIP_CY, a * pa * 0.8)
    if swell > 0:
        pulse = 0.8 + 0.2 * math.sin(t * 5.0)
        cl = capsule_layer(swell)
        put(main, cl, HIP_CX, HIP_CY, a * pulse); put(glow, cl, HIP_CX, HIP_CY, a * 0.35 * swell)
    else:
        put(main, capsule_layer(0.0), HIP_CX, HIP_CY, a * eo(prog(t, t0 + 1.0, t0 + 1.5)) * 0.8)
    put(main, FEM_S, HIP_CX, HIP_CY + 30 * (1 - fa), a * fa)
    put(main, EPI_S, HIP_CX, HIP_CY, a * ea)

def hip_labels(t, main, a, labels, hide=()):
    d = ImageDraw.Draw(main)
    for name, p, anc, t0 in labels:
        if name in hide: continue
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        x, y = hp(p); lx, ly = anc
        col = WARN if name == 'EFFUSION' else ICE
        dot(d, x, y, 7, col, a * q)
        d.line([(x, y), (x + (lx - x) * q, y + (ly - y) * q)], fill=col + (int(220 * a * q),), width=3)
        lab = tl(name, 'MontXB', 30 if name != 'EFFUSION' else 40, col, track=2)
        lx2 = min(max(lx, lab.width / 2 + 90), 950 - lab.width / 2)
        ly2 = ly + (-26 if ly < y else 28)
        put(main, card(lab.width - 10, lab.height - 6, fill=(12, 24, 42, 200), outline=(0, 0, 0, 0), r=14), lx2, ly2, a * q)
        put(main, lab, lx2, ly2, a * q)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #1', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 480, a * eo(prog(t, 0.0, 0.3)))
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('KOCHER', 'MontBlack', 176, WHITE)
    put(glow, k1, W / 2, 690, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 690, a * clamp(p), clamp(p, 0, 1.2))
    p2 = R.eback(prog(t, 0.2, 0.65))
    k2 = tl('CRITERIA', 'MontBlack', 150, SKY)
    put(main, k2, W / 2, 870, a * clamp(p2), clamp(p2, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('Septic hip or transient synovitis?', 'MontXB', 44, WHITE), W / 2, 1060 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, tl('4 PREDICTORS DECIDE', 'MontBlack', 48, SKY, track=4), W / 2, 1160 + 25 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # context + hip diagram
    a = win(t, 3.3, 12.2, 0.35, 0.45)
    if a <= 0: return
    p = eo(prog(t, 3.4, 4.0))
    put(main, tl('The irritable hip in a child', 'MontBlack', 58, WHITE), W / 2, 250, a * p, 1, 10 * (1 - p))
    label(main, 'RIGHT HIP  ·  AP VIEW  ·  SCHEMATIC', W / 2, 335, a * eo(prog(t, 3.6, 4.2)), SKY, 30, 'MontXB', 4)
    swell = eio(prog(t, 6.6, 7.6))
    hip(t, main, glow, a, 3.6, swell)
    hip_labels(t, main, a, HIP_LABELS + [('EFFUSION', (330, 470), (520, 1360), 7.4)])
    q = eo(prog(t, 7.9, 8.4))
    put(main, rich([('Septic arthritis', 'MontBlack', WARN), ('   or   ', 'MontXB', ICE), ('transient synovitis?', 'MontBlack', GOOD)], 37),
        W / 2, 1480, a * q)
    label(main, 'Single tests overlap.  Combine them.', W / 2, 1580, a * eo(prog(t, 8.4, 8.9)), WHITE, 40, 'MontBlack', 1)

def thermometer(d, x, y, col, a):
    c = col + (int(255 * a),)
    d.rounded_rectangle((x - 11, y - 52, x + 11, y + 18), 11, outline=c, width=6)
    d.ellipse((x - 22, y + 8, x + 22, y + 52), fill=c)
    d.rectangle((x - 4, y - 20, x + 4, y + 14), fill=c)
def foot(d, x, y, col, a):
    c = col + (int(255 * a),)
    d.ellipse((x - 20, y - 30, x + 14, y + 38), fill=c)
    for i, (dx, dy, r) in enumerate([(-14, -48, 9), (0, -52, 7), (12, -48, 6), (22, -40, 5)]):
        d.ellipse((x + dx - r, y + dy - r, x + dx + r, y + dy + r), fill=c)
    cross(d, x + 26, y + 24, 46, WARN, a, 8)
def tube(d, x, y, col, a):
    c = col + (int(255 * a),)
    d.rounded_rectangle((x - 16, y - 54, x + 16, y + 54), 16, outline=c, width=6)
    d.rectangle((x - 9, y + 5, x + 9, y + 44), fill=WARN + (int(255 * a),))
    d.line([(x - 26, y - 54), (x + 26, y - 54)], fill=c, width=6)
def wbc(d, x, y, col, a):
    c = col + (int(255 * a),)
    d.ellipse((x - 46, y - 46, x + 46, y + 46), outline=c, width=6)
    for dx, dy in [(-14, -10), (12, -14), (2, 14)]:
        d.ellipse((x + dx - 13, y + dy - 13, x + dx + 13, y + dy + 13), fill=SKY + (int(255 * a),))

def ge40(size, col):
    return rich([('ESR   ', 'MontBlack', col), ('≥', 'DejaVuB', col), ('   40 mm/h', 'MontBlack', col)], size)

PRED = [(thermometer, lambda: tl('History of fever', 'MontBlack', 46, WHITE), 'FEVER REPORTED BY HISTORY'),
        (foot, lambda: tl('Non-weight-bearing', 'MontBlack', 46, WHITE), 'REFUSES TO WALK ON THE LEG'),
        (tube, lambda: ge40(46, WHITE), 'ERYTHROCYTE SEDIMENTATION RATE'),
        (wbc, lambda: tl('WBC  >  12,000 /mm³', 'MontBlack', 46, WHITE), 'SERUM WHITE BLOOD-CELL COUNT')]
def s3(t, main, glow, fx):                                     # the four predictors
    a = win(t, 12.0, 22.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'KOCHER CRITERIA', W / 2, 250, a * eo(prog(t, 12.1, 12.6)), SKY, 60, 'MontBlack', 8)
    label(main, 'Kocher et al.  ·  JBJS Am 1999', W / 2, 335, a * eo(prog(t, 12.3, 12.8)), WHITE, 36, 'MontXB', 1)
    d = ImageDraw.Draw(main)
    for i, (icon, txt, sub) in enumerate(PRED):
        t0 = 13.0 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 530 + i * 250 + 30 * (1 - q)
        put(main, card(860, 210), 515, y, a * q)
        put(main, tl(str(i + 1), 'MontBlack', 70, SKY), 150, y, a * q)
        icon(d, 260, y, ICE, a * q)
        im = txt(); put(main, im, 340 + im.width / 2 - 20, y - 30, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 340 + lab.width / 2 - 20, y + 42, a * q)
        ck = eo(prog(t, t0 + 0.5, t0 + 0.8))
        if ck > 0: check(d, 910, y - 58, 32, GOOD, a * ck, 6)
    label(main, 'Retrospective  ·  children\'s hospital  ·  1979–1996', W / 2, 1560, a * eo(prog(t, 19.4, 19.9)), SUB, 28, 'MontXB', 0)
    label(main, 'Each one is an independent predictor', W / 2, 1630, a * eo(prog(t, 19.5, 20.0)), WHITE, 36, 'MontBlack', 0)

PROB = [(0, 0.2, '<0.2%'), (1, 3.0, '3.0%'), (2, 40.0, '40.0%'), (3, 93.1, '93.1%'), (4, 99.6, '99.6%')]
BX0, BX1, BY0, BH = 150, 900, 1180, 560
def s4(t, main, glow, fx):                                     # probability by number of predictors + validation
    a = win(t, 22.4, 33.2, 0.4, 0.45)
    if a <= 0: return
    label(main, 'COUNT THE PREDICTORS', W / 2, 250, a * eo(prog(t, 22.5, 23.0)), SKY, 50, 'MontBlack', 5)
    label(main, 'Predicted probability of septic arthritis', W / 2, 335, a * eo(prog(t, 22.7, 23.2)), WHITE, 36, 'MontXB', 1)
    g = eo(prog(t, 22.9, 23.4))
    d = ImageDraw.Draw(main); gd = ImageDraw.Draw(glow)
    d.line([(BX0 - 20, BY0), (BX1 + 10, BY0)], fill=SUB + (int(200 * a * g),), width=4)
    slot = (BX1 - BX0) / 5; bw = slot * 0.62
    for i, (n, v, txt) in enumerate(PROB):
        t0 = 23.4 + 0.75 * i; q = eio(prog(t, t0, t0 + 0.6))
        cx = BX0 + slot * (i + 0.5)
        label(main, str(n), cx, BY0 + 50, a * g, WHITE, 46, 'MontBlack', 0)
        if q <= 0: continue
        col = GOOD if v < 10 else (SKY if v < 50 else WARN)
        h = max(6, BH * v / 100 * q)
        d.rounded_rectangle((cx - bw / 2, BY0 - h, cx + bw / 2, BY0 - 2), 12, fill=col + (int(235 * a),))
        if v > 50: gd.rounded_rectangle((cx - bw / 2, BY0 - h, cx + bw / 2, BY0 - 2), 12, fill=col + (int(90 * a),))
        label(main, txt, cx, BY0 - h - 40, a * eo(prog(t, t0 + 0.4, t0 + 0.7)), col, 40 if i else 36, 'MontBlack', 0)
    label(main, 'NUMBER OF PREDICTORS PRESENT', W / 2, BY0 + 115, a * g, SUB, 28, 'MontXB', 3)
    q = eo(prog(t, 27.4, 28.0))
    put(main, card(860, 320, outline=SKY + (150,)), 515, 1495, a * q)
    label(main, 'PROSPECTIVE VALIDATION  ·  KOCHER 2004', 515, 1380, a * q, SKY, 30, 'MontBlack', 2)
    label(main, '51 septic  vs  103 transient synovitis', 515, 1450, a * eo(prog(t, 27.7, 28.2)), WHITE, 40, 'MontBlack', 0)
    label(main, 'AUC 0.96 derivation  ·  0.86 validation', 515, 1525, a * eo(prog(t, 28.3, 28.8)), ICE, 36, 'MontXB', 0)
    label(main, 'Lower, but still very good', 515, 1585, a * eo(prog(t, 29.0, 29.5)), SUB, 30, 'MontXB', 0)

TIPS = [('Add CRP > 2.0 mg/dL', '= 20 mg/L  ·  STRONG INDEPENDENT FACTOR', 'Caird et al. JBJS Am 2006'),
        ('Fever > 38.5 °C', 'BEST SINGLE PREDICTOR IN CAIRD\'S SERIES', 'Oral temperature'),
        ('The aspirate decides', 'JOINT-FLUID CELL COUNT  +  CULTURE', 'Reference standard in these studies')]
def s5(t, main, glow, fx):
    a = win(t, 33.0, 41.5, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 33.1, 33.6)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 33.8 + 1.8 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        put(main, tl(f'0{i + 1}', 'MontBlack', 74, SKY), 870, y, a * q)
        label(main, h_, 460, y - 52, a * q, WHITE, 48, 'MontBlack', 0)
        label(main, sub, 460, y + 22, a * q, SUB, 27, 'MontXB', 0)
        if ref: label(main, ref, 460, y + 82, a * q, SKY, 27, 'MontXB', 1)

def s6(t, main, glow, fx):
    a = win(t, 41.3, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 420, a * eo(prog(t, 41.4, 41.9)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 41.6, 42.2))
    big = tl('3 of 4', 'MontBlack', 170, WHITE)
    put(glow, big, W / 2, 600, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 600, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '= 93.1% SEPTIC', W / 2, 760, a * eo(prog(t, 42.0, 42.5)), WARN, 60, 'MontBlack', 3)
    label(main, 'None of 4  =  under 0.2%', W / 2, 860, a * eo(prog(t, 42.4, 42.9)), GOOD, 44, 'MontBlack', 1)
    q = eo(prog(t, 42.9, 43.5))
    put(main, card(860, 330, fill=(24, 44, 70, 160), outline=(120, 175, 216, 80)), 515, 1150, a * q)
    for i, (au, ref) in enumerate([('Kocher MS, et al.', 'J Bone Joint Surg Am. 1999;81(12):1662–70'),
                                   ('Kocher MS, et al.', 'J Bone Joint Surg Am. 2004;86(8):1629–35'),
                                   ('Caird MS, et al.', 'J Bone Joint Surg Am. 2006;88(6):1251–7')]):
        y = 1040 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1420, a * eo(prog(t, 43.6, 44.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
