"""Orthojets EP02 — Acute compartment syndrome: Delta P < 30 mmHg.
Sources (PubMed):
  McQueen MM, Court-Brown CM. JBJS Br 1996;78(1):99-104 (PMID 8898137): 116 tibial diaphyseal fractures, 24 h
    continuous anterior compartment monitoring; dP = diastolic - compartment pressure; dP < 30 -> decompress, no missed ACS;
    absolute 30 mmHg threshold -> 50 pts (43%) fasciotomy; 40 mmHg -> 27 (23%) unnecessary fasciotomy.
  Kakar S et al. J Orthop Trauma 2007;21(2):99-103 (doi 10.1097/BOT.0b013e318032c4f4): 242 tibia fractures, IM nail
    under GA; intra-op DBP 18 +/- 13 mmHg lower than pre-op -> intra-op dP may be spuriously low.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(37.0, [(5.9, 1.0), (7.6, 2.0), (13.6, 2.5), (16.6, 3.0), (21.0, 1.5), (23.4, 3.0),
                     (27.6, 1.2), (29.0, 1.2), (30.6, 2.8), (35.6, 3.0)])

# ------------------------------------------------------------------ leg cross-section (right leg, mid-shaft, viewed from above)
SS = 3; LW, LH = 620, 540
import math as _m
LEG_C, LEG_RX, LEG_RY = (300, 290), 230, 225
def ell(deg): return (LEG_C[0] + LEG_RX * _m.cos(_m.radians(deg)), LEG_C[1] + LEG_RY * _m.sin(_m.radians(deg)))
SKIN_PTS = [ell(a) for a in range(0, 360, 3)]
A_, M_, L_ = (205, 95), (125, 170), (255, 180)          # tibial crest, medial (posteromedial) border, interosseous border
def mid(p, q, bulge=0.0, c=(195, 148)):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    return (mx + (mx - c[0]) * bulge, my + (my - c[1]) * bulge)
TIBIA = [A_, mid(A_, L_, 0.12), L_, mid(L_, M_, 0.10), M_, mid(M_, A_, 0.12)]
FIB_C, FIB_R = (455, 215), 21
SEEDS = {'ANTERIOR': (345, 150), 'LATERAL': (497, 240), 'DEEP POSTERIOR': (300, 235), 'SUPERFICIAL POSTERIOR': (300, 400)}

def smooth(pts, n=8):
    P = pts + pts[:2]; out = []
    for i in range(len(pts)):
        p0, p1, p2, p3 = pts[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t ** 3
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    return out

def S(p): return (p[0] * SS, p[1] * SS)

def build_section():
    size = (LW * SS, LH * SS)
    skin = SKIN_PTS; tib = smooth(TIBIA, 6)
    walls = Image.new('L', size); d = ImageDraw.Draw(walls)
    d.rectangle((0, 0) + size, fill=255)
    d.polygon([S(p) for p in skin], fill=0)
    d.polygon([S(p) for p in tib], fill=255)
    fx, fy = FIB_C; d.ellipse([S((fx - FIB_R, fy - FIB_R)), S((fx + FIB_R, fy + FIB_R))], fill=255)
    septa = [[L_, (345, 192), (436, 210)],                           # interosseous membrane
             [(458, 195), (455, 150), ell(-52)],                      # anterior intermuscular septum
             [(468, 233), (492, 300), ell(22)],                       # posterior intermuscular septum
             [(128, 176), (210, 262), (330, 282), (447, 233)],        # transverse (deep) fascia
             [A_, ell(245)], [M_, ell(205)]]                          # subcutaneous tibia closures
    for s_ in septa: d.line([S(p) for p in s_], fill=255, width=4 * SS, joint='curve')
    regions = {}
    for k, seed in SEEDS.items():
        r = walls.copy(); ImageDraw.floodfill(r, S(seed), 128)
        regions[k] = r.point(lambda v: 255 if v == 128 else 0).filter(ImageFilter.GaussianBlur(1.5))
    base = Image.new('RGBA', size)
    tint = {'ANTERIOR': (120, 175, 216), 'LATERAL': (100, 155, 205), 'DEEP POSTERIOR': (90, 140, 195), 'SUPERFICIAL POSTERIOR': (110, 165, 210)}
    for k, m in regions.items():
        lay = Image.new('RGBA', size, tint[k] + (0,)); lay.putalpha(m.point(lambda v: int(v * 0.30))); base.alpha_composite(lay)
    ld = ImageDraw.Draw(base)
    ld.line([S(p) for p in skin + [skin[0]]], fill=(200, 225, 248, 230), width=4 * SS, joint='curve')
    for s_ in septa[:4]: ld.line([S(p) for p in s_], fill=(170, 205, 235, 210), width=3 * SS, joint='curve')
    # bones: bright cortex, darker canal
    ld.polygon([S(p) for p in tib], fill=(232, 242, 252, 255))
    ld.polygon([S(((x - 195) * 0.45 + 195, (y - 148) * 0.45 + 148)) for x, y in tib], fill=(120, 150, 185, 255))
    ld.ellipse([S((fx - FIB_R, fy - FIB_R)), S((fx + FIB_R, fy + FIB_R))], fill=(232, 242, 252, 255))
    ld.ellipse([S((fx - 9, fy - 9)), S((fx + 9, fy + 9))], fill=(120, 150, 185, 255))
    # fracture crack in the tibia
    ld.line([S(p) for p in [(206, 104), (196, 128), (210, 146), (192, 170)]], fill=(20, 32, 50, 255), width=3 * SS)
    return base, regions

SEC, REG = build_section()
SEC_SCALE = 1.5
SEC_W, SEC_H = int(LW * SEC_SCALE), int(LH * SEC_SCALE)
SEC_SMALL = SEC.resize((SEC_W, SEC_H), Image.LANCZOS)
ANT = REG['ANTERIOR'].resize((SEC_W, SEC_H), Image.LANCZOS)
SEC_CX, SEC_CY = W / 2, 900
def sec_pt(p): return SEC_CX - SEC_W / 2 + p[0] * SEC_SCALE, SEC_CY - SEC_H / 2 + p[1] * SEC_SCALE

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):
    a = win(t, 0, 3.3, 0.25, 0.35)
    if a <= 0: return
    put(main, pill('ORTHO TIP  #02', 'MontXB', 30, NAVY, SKY + (255,), h=64), W / 2, 470, a * eo(prog(t, 0.15, 0.6)))
    p = R.eback(prog(t, 0.3, 1.0))
    big = rich([('Δ', 'DejaVuB', WHITE), ('P < 30', 'MontBlack', WHITE)], 210)
    put(glow, big, W / 2, 800, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 800, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'mmHg', W / 2, 945, a * eo(prog(t, 0.9, 1.3)), SKY, 60, 'MontXB', 4)
    fx['flash'] += 0.22 * math.exp(-((t - 0.75) / 0.08) ** 2)
    p = eo(prog(t, 1.3, 1.9)); put(main, tl('The number that decides', 'MontXB', 60, WHITE), W / 2, 1120 + 30 * (1 - p), a * p, 1, 10 * (1 - p))
    p = eo(prog(t, 1.6, 2.2)); put(main, tl('a fasciotomy', 'MontXB', 60, SKY), W / 2, 1205 + 30 * (1 - p), a * p, 1, 10 * (1 - p))

LABELS = [('ANTERIOR', (345, 150), (800, 450), 4.3), ('LATERAL', (497, 240), (960, 640), 4.7),
          ('DEEP POSTERIOR', (300, 232), None, 5.1), ('SUPERFICIAL POSTERIOR', (300, 400), (540, 1395), 5.5)]
def section(t, main, glow, a, highlight):
    put(main, SEC_SMALL, SEC_CX, SEC_CY, a)
    if highlight > 0:
        pulse = 0.55 + 0.25 * math.sin(t * 5.5)
        lay = Image.new('RGBA', (SEC_W, SEC_H), WARN + (0,)); lay.putalpha(ANT.point(lambda v: int(v * pulse * highlight)))
        put(main, lay, SEC_CX, SEC_CY, a); put(glow, lay, SEC_CX, SEC_CY, a * 0.6)
    d = ImageDraw.Draw(main)
    for name, p, anc, t0 in LABELS:
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        x, y = sec_pt(p)
        col = WARN if (name == 'ANTERIOR' and highlight > 0.5) else ICE
        if anc is None:
            put(main, tl(name, 'MontXB', 27, WHITE, track=2), x, y + 6, a * q)
            continue
        lx, ly = anc
        dot(d, x, y, 7, col, a * q)
        ex, ey = x + (lx - x) * q, y + (ly - y) * q
        d.line([(x, y), (ex, ey)], fill=col + (int(220 * a * q),), width=3)
        lab = tl(name, 'MontXB', 28, col, track=2)
        lx2 = min(max(lx, lab.width / 2 + 10), W - lab.width / 2 - 10)
        put(main, lab, lx2, ly + (-24 if ly < y else 26), a * q)

def s2(t, main, glow, fx):
    a = win(t, 3.0, 10.2, 0.35, 0.45)
    if a <= 0: return
    p = eo(prog(t, 3.1, 3.7))
    put(main, tl('Acute Compartment Syndrome', 'MontBlack', 58, WHITE), W / 2, 250, a * p, 1, 10 * (1 - p))
    label(main, 'LEG  ·  TIBIAL SHAFT FRACTURE', W / 2, 335, a * eo(prog(t, 3.4, 4.0)), SKY, 30, 'MontXB', 5)
    sp = eo(prog(t, 3.4, 4.2))
    section(t, main, glow, a * sp, eo(prog(t, 6.3, 6.9)))
    q = eo(prog(t, 6.6, 7.2))
    label(main, 'Anterior compartment:', W / 2, 1500, a * q, ICE, 38, 'MontXB', 1)
    label(main, 'MOST COMMONLY AFFECTED', W / 2, 1565, a * q, WARN, 44, 'MontBlack', 3)
    label(main, 'Pressure rises inside a closed fascial space', W / 2, 1650, a * eo(prog(t, 7.0, 7.6)), SUB, 32, 'MontXB', 1)

# delta-P gauge: 0..60 mmHg
GX0, GX1, GY = 150, 930, 1060
def gx(v): return GX0 + (GX1 - GX0) * v / 60
def gauge(main, glow, a, value=None, vcol=WHITE, vp=0.0):
    d = ImageDraw.Draw(main)
    d.rounded_rectangle((GX0, GY - 22, gx(30), GY + 22), 22, fill=WARN + (int(110 * a),))
    d.rounded_rectangle((gx(30), GY - 22, GX1, GY + 22), 22, fill=GOOD + (int(90 * a),))
    d.line([(gx(30), GY - 48), (gx(30), GY + 48)], fill=WHITE + (int(255 * a),), width=5)
    label(main, '30', gx(30), GY - 80, a, WHITE, 40, 'MontBlack', 1)
    for v in (0, 60): label(main, str(v), gx(v), GY + 70, a * 0.9, SUB, 30, 'MontXB', 1)
    label(main, 'DECOMPRESS', (GX0 + gx(30)) / 2, GY + 70, a, WARN, 28, 'MontBlack', 2)
    label(main, 'MONITOR', (gx(30) + GX1) / 2, GY + 70, a, GOOD, 28, 'MontBlack', 2)
    if value is not None and vp > 0:
        x = gx(value * vp)
        d.polygon([(x, GY - 30), (x - 18, GY - 58), (x + 18, GY - 58)], fill=vcol + (int(255 * a),))
        d.ellipse((x - 14, GY - 14, x + 14, GY + 14), fill=vcol + (int(255 * a),))
        ImageDraw.Draw(glow).ellipse((x - 20, GY - 20, x + 20, GY + 20), fill=vcol + (int(200 * a),))

def example(main, a, y, dbp, icp, good, q, r):
    put(main, card(900, 250), W / 2, y, a * q)
    label(main, 'EXAMPLE', W / 2, y - 90, a * q, SUB, 26, 'MontXB', 6)
    label(main, f'DBP {dbp}   −   ICP {icp}   =', W / 2 - 80, y - 25, a * q, WHITE, 46, 'MontBlack', 1)
    v = dbp - icp
    label(main, str(v), W / 2 + 290, y - 25, a * q, GOOD if good else WARN, 60, 'MontBlack', 1)
    if r > 0:
        d = ImageDraw.Draw(main)
        if good:
            label(main, 'Monitor  &  re-measure', W / 2 - 20, y + 60, a * r, GOOD, 40, 'MontBlack', 1)
            check(d, W / 2 + 290, y + 58, 40, GOOD, a * r)
        else:
            label(main, 'Fasciotomy', W / 2 - 20, y + 60, a * r, WARN, 44, 'MontBlack', 2)
            arrow(d, W / 2 - 300, y + 60, W / 2 - 170, WARN, a * r)

def s3(t, main, glow, fx):
    a = win(t, 10.0, 19.2, 0.4, 0.4)
    if a <= 0: return
    p = eo(prog(t, 10.1, 10.7))
    f1 = rich([('Δ', 'DejaVuB', WHITE), ('P  =  Diastolic BP  −  ICP', 'MontBlack', WHITE)], 54)
    put(main, f1, W / 2, 270, a * p, 1, 10 * (1 - p))
    label(main, 'ICP = intracompartmental pressure', W / 2, 355, a * eo(prog(t, 10.4, 11.0)), SUB, 32, 'MontXB', 1)
    g = eo(prog(t, 10.9, 11.5))
    ex2 = eo(prog(t, 14.4, 14.9))
    val, col, vp = (25, WARN, eio(prog(t, 12.6, 13.3))) if t < 14.4 else (40, GOOD, eio(prog(t, 15.2, 15.9)))
    gauge(main, glow, a * g, val, col, vp)
    label(main, 'mmHg', GX1 - 10, GY - 80, a * g, SUB, 28, 'MontXB', 2)
    if t < 14.4:
        example(main, a * (1 - prog(t, 14.1, 14.4)), 1400, 70, 45, False, eo(prog(t, 11.8, 12.4)), eo(prog(t, 13.2, 13.6)))
    else:
        example(main, a, 1400, 80, 40, True, ex2, eo(prog(t, 15.8, 16.2)))
        label(main, 'ICP of 40 mmHg  —  yet no fasciotomy indicated', W / 2, 1610, a * eo(prog(t, 16.1, 16.6)), ICE, 32, 'MontXB', 1)
    label(main, 'Illustrative values', W / 2, 1690, a * g * 0.9, SUB, 26, 'MontXB', 2)

def s4(t, main, glow, fx):
    a = win(t, 19.0, 26.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'THE EVIDENCE', W / 2, 260, a * eo(prog(t, 19.1, 19.6)), SKY, 54, 'MontBlack', 8)
    label(main, 'McQueen & Court-Brown  ·  JBJS Br 1996', W / 2, 345, a * eo(prog(t, 19.3, 19.8)), WHITE, 36, 'MontXB', 1)
    q = eo(prog(t, 19.7, 20.3))
    put(main, card(900, 210), W / 2, 560, a * q)
    label(main, '116 tibial shaft fractures', W / 2, 520, a * q, WHITE, 46, 'MontBlack', 1)
    label(main, '24 h continuous anterior compartment monitoring', W / 2, 600, a * q, SUB, 29, 'MontXB', 0)
    r = eo(prog(t, 20.3, 20.9))
    put(main, card(900, 250, outline=GOOD + (150,)), W / 2, 830, a * r)
    th = rich([('Δ', 'DejaVuB', GOOD), ('P < 30 threshold', 'MontBlack', GOOD)], 48)
    put(main, th, W / 2, 785, a * r)
    label(main, 'NO MISSED CASES OF ACS', W / 2, 870, a * r, WHITE, 40, 'MontBlack', 2)
    u = eo(prog(t, 21.6, 22.2))
    put(main, card(900, 470, outline=WARN + (150,)), W / 2, 1245, a * u)
    label(main, 'If ABSOLUTE pressure were used:', W / 2, 1065, a * u, ICE, 34, 'MontXB', 1)
    d = ImageDraw.Draw(main)
    for i, (thr, pct, txt) in enumerate([(30, '43%', 'would have had a fasciotomy'), (40, '23%', 'unnecessary fasciotomies')]):
        y = 1160 + i * 170; v = eo(prog(t, 22.2 + 0.5 * i, 22.7 + 0.5 * i))
        label(main, f'{thr} mmHg', W / 2 - 170, y, a * v, WHITE, 44, 'MontBlack', 1)
        arrow(d, W / 2 - 25, y, W / 2 + 55, WARN, a * v)
        label(main, pct, W / 2 + 160, y, a * v, WARN, 64, 'MontBlack', 1)
        label(main, txt, W / 2, y + 62, a * v, ICE, 30, 'MontXB', 0)
    label(main, 'Only 3 of 116 actually had ACS', W / 2, 1530, a * eo(prog(t, 23.1, 23.6)), SUB, 32, 'MontXB', 1)

TIPS = [('Under GA, DBP drops ~18 mmHg', 'INTRA-OP DELTA P CAN BE FALSELY LOW', 'Kakar et al. JOT 2007'),
        ('Pain out of proportion', 'PAIN ON PASSIVE STRETCH  =  EARLIEST SIGN', None),
        ('Pulses do not exclude ACS', 'PULSELESSNESS IS A LATE SIGN', None)]
def s5(t, main, glow, fx):
    a = win(t, 26.0, 33.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 26.1, 26.6)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 26.8 + 1.4 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(920, 280), W / 2, y, a * q)
        put(main, tl(f'0{i + 1}', 'MontBlack', 88, SKY), W / 2 + 375, y, a * q)
        label(main, h_, W / 2 - 60, y - 52, a * q, WHITE, 46, 'MontBlack', 0)
        label(main, sub, W / 2 - 60, y + 22, a * q, SUB, 24 if len(sub) > 40 else 27, 'MontXB', 1)
        if ref: label(main, ref, W / 2 - 60, y + 82, a * q, SKY, 26, 'MontXB', 1)

def s6(t, main, glow, fx):
    a = win(t, 33.0, 37.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 560, a * eo(prog(t, 33.1, 33.6)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 33.3, 33.9))
    big = rich([('Δ', 'DejaVuB', WHITE), ('P < 30', 'MontBlack', WHITE)], 170)
    put(glow, big, W / 2, 760, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 760, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'mmHg  =  DECOMPRESS', W / 2, 900, a * eo(prog(t, 33.6, 34.1)), WARN, 46, 'MontBlack', 3)
    q = eo(prog(t, 34.0, 34.6))
    put(main, card(920, 230, fill=(24, 44, 70, 160), outline=(120, 175, 216, 80)), W / 2, 1180, a * q)
    label(main, 'McQueen MM, Court-Brown CM.', W / 2, 1110, a * q, ICE, 30, 'MontXB', 0)
    label(main, 'J Bone Joint Surg Br. 1996;78(1):99–104', W / 2, 1155, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Kakar S, et al.', W / 2, 1210, a * q, ICE, 30, 'MontXB', 0)
    label(main, 'J Orthop Trauma. 2007;21(2):99–103', W / 2, 1255, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1440, a * eo(prog(t, 34.6, 35.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES)
