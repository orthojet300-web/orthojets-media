"""Orthojets EP13 - DON'T MISS #3 - Acute Achilles tendon rupture: the calf squeeze test, the Matles test, the X-ray clue.
Sources (verified on PubMed abstracts / metadata, 2026-11-02):
  Maffulli N. The clinical diagnosis of subcutaneous tear of the Achilles tendon. A prospective study in 174 patients.
    Am J Sports Med 1998;26(2):266-70. PMID 9548122, doi 10.1177/03635465980260021801.
    174 complete ruptures + 28 suspected but intact. Sensitivity: calf squeeze 0.96, Matles 0.88, Copeland 0.80,
    O'Brien 0.80, palpable gap 0.73 (0.81 under anaesthesia). Specificity: calf squeeze 0.93, gap 0.89, Matles 0.85.
    At least two tests were positive in every patient with a tear.
  Kiely PD et al. Achilles tendon rupture must be excluded in the neutral, non-fractured ankle X-ray study.
    J Emerg Med 2010;41(6):718-22. PMID 20580519, doi 10.1016/j.jemermed.2010.05.015.
    27 operated ruptures vs 27 intact controls. Tibio-first metatarsal angle on the lateral ankle X-ray: mean 88 deg
    (70-120) in rupture vs 125 deg (104-146) intact, p under 0.001. Neutral ankle = angle of 100 deg or less.
    "Significant potential for missing the diagnosis on initial presentation."
  Park YH et al. The influence of ankle position on radiographic diagnosis of acute Achilles tendon rupture.
    J Foot Ankle Surg 2022;61(4):845-9. PMID 34974982, doi 10.1053/j.jfas.2021.12.003.
    154 lateral radiographs: Kager's triangle, Toygar's angle, Arner's sign and tendon thickening are all more
    visible in plantar flexion than dorsiflexion; avoid neutral / dorsiflexion for suspected rupture.
  Diagrams (schematic, labelled): 1) lateral view of a prone patient's lower leg, feet over the edge of the couch:
  gastrocnemius-soleus, Achilles tendon to the calcaneal tuberosity; calf squeeze -> passive plantar flexion if intact,
  no movement if ruptured. 2) Matles test: prone, knees flexed 90 deg; the intact foot rests plantarflexed, the injured
  foot falls to neutral. 3) Lateral ankle X-ray schematic: tibial axis vs first metatarsal axis, angle 125 vs 88 deg.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(7.9, 1.4), (11.7, 1.2), (14.2, 2.0), (20.6, 1.4), (23.0, 1.6), (29.6, 1.8), (36.9, 2.0),
                     (42.6, 1.6), (45.6, 1.8)])

XC = 510
LIMB = (58, 92, 132)
MUSC = (92, 128, 170)
TEND = (236, 226, 204)
BONE = (200, 222, 242)
SS = 2

def rot(pts, th, cx, cy):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]

def shape_layer(w, h, polys, fill, outline=ICE, ow=5):
    """filled union of polygons with a smooth outline drawn around the union (supersampled)"""
    m = Image.new('L', (w * SS, h * SS)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x * SS, y * SS) for x, y in p], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(1.2))
    edge = ImageChops.subtract(m, m.filter(ImageFilter.MinFilter(ow * SS + 1 - (ow * SS) % 2)))
    im = Image.new('RGBA', (w * SS, h * SS), fill + (0,)); im.putalpha(m)
    e = Image.new('RGBA', im.size, outline + (0,)); e.putalpha(edge); im.alpha_composite(e)
    return im.resize((w, h), Image.LANCZOS)

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=1)
    ey = ly + (lab.height / 2 - 14 if tip[1] > ly else -lab.height / 2 + 14)
    d.line([tip, (lx, ey)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def chip(main, a, x, y, txt, col, ok, w=860):
    if a <= 0: return
    fill = (24, 60, 60, 200) if ok else (66, 32, 38, 200)
    put(main, card(w, 96, fill=fill, outline=col + (220,), r=30), x, y, a)
    d = ImageDraw.Draw(main)
    (check if ok else cross)(d, x - w / 2 + 60, y, 40, col, a, 8)
    lab = tl(txt, 'MontBlack', 36, WHITE); put(main, lab, x - w / 2 + 105 + lab.width / 2 - 20, y, a)

# ------------------------------------------------------------------ diagram 1: prone lower leg, lateral view
LW, LH = 900, 640
LX0, LY0 = 60, 500                     # screen position of the local canvas
ANK = (250, 330)
TABLE = (330, 428, 860, 470)
FOOT = [(-5, -48), (-55, -50), (-82, -20), (-84, 40), (-74, 120), (-62, 190), (-50, 236), (-20, 248), (10, 236),
        (22, 180), (36, 110), (48, 50), (40, 10)]
INS = (-26, -40)                       # Achilles insertion on the calcaneal tuberosity (foot-local)
MTJ = (520, 262)                       # musculotendinous junction

def leg_poly(sq):
    top = [(860, 250), (800, 215 + 6 * sq), (720, 198 + 14 * sq), (640, 205 + 12 * sq), (560, 235 + 6 * sq), (470, 270),
           (380, 292), (300, 298)]
    return top + [(262, 320), (256, 360), (270, 392), (300, 404), (450, 412), (650, 418), (860, 425)]

def muscle_poly(sq):
    return [(845, 262), (790, 230 + 6 * sq), (720, 214 + 13 * sq), (640, 220 + 11 * sq), (560, 246 + 5 * sq), (MTJ[0], MTJ[1]),
            (600, 282), (700, 290), (845, 288)]

_lc = {}
def leg_layer(th, sq, gap):
    key = (round(th, 1), round(sq, 2), round(gap, 2))
    if key in _lc: return _lc[key]
    im = Image.new('RGBA', (LW, LH)); d = ImageDraw.Draw(im)
    d.rounded_rectangle(TABLE, 10, fill=(70, 94, 126, 255))
    foot = rot(FOOT, th, *ANK)
    im.alpha_composite(shape_layer(LW, LH, [leg_poly(sq), foot], LIMB))
    im.alpha_composite(shape_layer(LW, LH, [muscle_poly(sq)], MUSC, ICE, 2))
    # tendon: musculotendinous junction -> calcaneal insertion; a gap opens 4-6 cm above the insertion
    ins = rot([INS], th, *ANK)[0]
    p0 = (MTJ[0], MTJ[1] + 6)
    big = Image.new('RGBA', (LW * SS, LH * SS)); bd = ImageDraw.Draw(big)
    def seg(u0, u1, col, w):
        a_ = (p0[0] + (ins[0] - p0[0]) * u0, p0[1] + (ins[1] - p0[1]) * u0)
        b_ = (p0[0] + (ins[0] - p0[0]) * u1, p0[1] + (ins[1] - p0[1]) * u1)
        bd.line([(a_[0] * SS, a_[1] * SS), (b_[0] * SS, b_[1] * SS)], fill=col + (255,), width=w * SS)
        for q in (a_, b_): bd.ellipse(((q[0] - w / 2) * SS, (q[1] - w / 2) * SS, (q[0] + w / 2) * SS, (q[1] + w / 2) * SS), fill=col + (255,))
    if gap <= 0:
        seg(0, 1, TEND, 17)
    else:
        seg(0, 0.58 - 0.16 * gap, TEND, 17); seg(0.72 + 0.04 * gap, 1, TEND, 17)
    im.alpha_composite(big.resize((LW, LH), Image.LANCZOS))
    if len(_lc) > 60: _lc.clear()
    _lc[key] = im
    return im

def L2S(p): return (LX0 + p[0], LY0 + p[1])

def foot_angle(t):
    """plantar flexion (deg) of the foot: rest 12; squeeze 1 (intact) -> 42; tendon cut; squeeze 2 -> none"""
    up = eio(prog(t, 6.6, 7.2)); down = eio(prog(t, 8.6, 9.2))
    return 12 + 30 * up * (1 - down) - 4 * eo(prog(t, 9.6, 10.2))

def squeeze(t):
    s1 = math.sin(math.pi * prog(t, 6.3, 8.6)); s2 = math.sin(math.pi * prog(t, 10.3, 12.6))
    return max(s1 if 6.3 < t < 8.6 else 0, s2 if 10.3 < t < 12.6 else 0)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #3", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 450, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Ankle pain, X-ray normal:', 'MontXB', 50, WHITE), W / 2, 580 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('SPRAIN?', 'MontBlack', 150, SKY)
    put(glow, k1, W / 2, 760, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 760, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('or a ruptured Achilles?', 'MontXB', 56, WHITE), W / 2, 940 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('THE 96% TEST', 'MontBlack', 38, NAVY, GOOD + (255,), h=76), W / 2, 1110 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # centrepiece: calf squeeze test
    a = win(t, 3.3, 17.0, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('CALF SQUEEZE TEST', 'MontBlack', 54, WHITE, track=2), XC, 255, a * q, 1, 8 * (1 - q))
    label(main, 'Simmonds-Thompson  ·  prone, feet off the couch', XC, 325, a * eo(prog(t, 3.6, 4.1)), SKY, 31, 'MontXB', 0)
    bp = eo(prog(t, 3.5, 4.5))
    th = foot_angle(t); sq = squeeze(t); gap = eio(prog(t, 9.2, 9.8))
    put(main, leg_layer(th, sq, gap), LX0 + LW / 2, LY0 + LH / 2 + 24 * (1 - bp), a * bp)
    # gap highlight
    if gap > 0:
        ins = rot([INS], th, *ANK)[0]; u = 0.65
        gx, gy = L2S((MTJ[0] + (ins[0] - MTJ[0]) * u, MTJ[1] + 6 + (ins[1] - MTJ[1] - 6) * u))
        pul = 0.75 + 0.25 * math.sin(t * 5)
        ImageDraw.Draw(glow).ellipse((gx - 46, gy - 46, gx + 46, gy + 46), fill=WARN + (int(150 * gap * pul * a),))
    # anatomy labels
    la = a * (1 - eo(prog(t, 6.0, 6.4)))
    pointer(main, la * eo(prog(t, 4.4, 4.9)), L2S((700, 236)), 770, 600, 'GASTROC-SOLEUS', ICE, 29)
    pointer(main, la * eo(prog(t, 4.8, 5.3)), L2S((430, 286)), 420, 640, 'ACHILLES', TEND, 30)
    pointer(main, la * eo(prog(t, 5.2, 5.7)), L2S((200, 300)), 175, 590, 'CALCANEUS', SUB, 29)
    # squeeze arrows on the calf
    if sq > 0.02:
        d = ImageDraw.Draw(main)
        for x in (640, 720, 800):
            xs, y1 = LX0 + x, LY0 + 200 + 12 * sq
            al = int(255 * a * min(1, sq * 3))
            d.line([(xs, y1 - 110), (xs, y1 - 26)], fill=SKY + (al,), width=8)
            d.polygon([(xs, y1 - 4), (xs - 18, y1 - 30), (xs + 18, y1 - 30)], fill=SKY + (al,))
        label(main, 'SQUEEZE', LX0 + 720, LY0 + 50, a * min(1, sq * 3), SKY, 34, 'MontBlack', 3)
    # foot-movement cue
    if 6.6 < t < 9.0:
        ImageDraw.Draw(glow).ellipse((LX0 + 60, LY0 + 380, LX0 + 220, LY0 + 560), fill=GOOD + (int(60 * a * squeeze(t)),))
    if gap > 0.5 and t < 9.9 + 0.01:
        pass
    ins_ = rot([INS], th, *ANK)[0]
    pointer(main, a * win(t, 9.4, 13.0, 0.3, 0.3), L2S((MTJ[0] + (ins_[0] - MTJ[0]) * 0.65, MTJ[1] + 6 + (ins_[1] - MTJ[1] - 6) * 0.65)), 300, 640, 'TORN', WARN, 34)
    chip(main, a * win(t, 7.0, 16.9, 0.35, 0.4), XC, 1235, 'Intact: foot plantarflexes', GOOD, True)
    chip(main, a * eo(prog(t, 11.0, 11.4)), XC, 1350, 'Ruptured: foot does not move', WARN, False)
    sq_ = a * eo(prog(t, 12.6, 13.1))
    if sq_ > 0:
        y = 1525 + 20 * (1 - sq_ / max(a, 1e-3))
        put(main, card(860, 190), XC, y, sq_)
        p = R.eback(prog(t, 12.8, 13.3))
        label(main, 'SENSITIVITY', 295, y - 50, sq_, SUB, 30, 'MontXB', 2)
        put(main, tl('96%', 'MontBlack', 80, GOOD), 295, y + 22, sq_ * clamp(p), clamp(p, 0, 1.15))
        label(main, 'SPECIFICITY', 725, y - 50, sq_, SUB, 30, 'MontXB', 2)
        p = R.eback(prog(t, 13.2, 13.7))
        put(main, tl('93%', 'MontBlack', 80, GOOD), 725, y + 22, sq_ * clamp(p), clamp(p, 0, 1.15))
    label(main, 'Maffulli 1998  ·  174 ruptures + 28 controls', XC, 1660, a * eo(prog(t, 13.3, 13.8)), SUB, 28, 'MontXB', 0)

TESTS = [('Calf squeeze', 96, GOOD), ('Matles', 88, GOOD), ('Copeland', 80, SKY), ("O'Brien", 80, SKY), ('Palpable gap', 73, WARN)]
def s3(t, main, glow, fx):                                     # sensitivity of each test
    a = win(t, 16.8, 24.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'WHICH TEST TO TRUST?', XC, 262, a * eo(prog(t, 16.9, 17.4)), WHITE, 54, 'MontBlack', 2)
    label(main, 'sensitivity  ·  174 complete ruptures', XC, 335, a * eo(prog(t, 17.1, 17.6)), SKY, 34, 'MontXB', 0)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    x0, x1 = 110, 760
    for i, (nm, v, col) in enumerate(TESTS):
        t0 = 17.5 + 0.55 * i; q = a * eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 500 + 175 * i
        lab = tl(nm, 'MontBlack', 42, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 48, q)
        d.rounded_rectangle((x0, y + 4, x1, y + 40), 18, fill=(60, 90, 125, int(200 * q)))
        p = eio(prog(t, t0 + 0.1, t0 + 0.9))
        xe = x0 + (x1 - x0) * v / 100 * p
        if p > 0:
            d.rounded_rectangle((x0, y, max(x0 + 40, xe), y + 44), 22, fill=col + (int(255 * q),))
            g.rounded_rectangle((x0, y, max(x0 + 40, xe), y + 44), 22, fill=col + (int(70 * q),))
        vt = tl(f'{int(round(v * p))}%', 'MontBlack', 56, col); put(main, vt, 860, y + 22, q * clamp(p * 2))
    gq = a * eo(prog(t, 21.2, 21.7))
    if gq > 0:
        y = 1430 + 20 * (1 - gq / max(a, 1e-3))
        put(main, card(860, 230, fill=(66, 32, 38, 190), outline=WARN + (200,)), XC, y, gq)
        label(main, 'NO GAP? NOT RULED OUT', XC, y - 60, gq, WHITE, 42, 'MontBlack', 1)
        label(main, 'The gap is missed in 1 of 4 ruptures', XC, y + 5, gq, WARN, 36, 'MontBlack', 0)
        label(main, 'Specificity: squeeze 93%, gap 89%', XC, y + 65, gq, SUB, 30, 'MontXB', 0)

# ------------------------------------------------------------------ diagram 2: Matles test (prone, knees at 90 deg)
MW, MH = 520, 640
MANK = (300, 175)
MFOOT = [(-55, -10), (-52, -44), (0, -52), (100, -50), (188, -44), (204, -30), (194, -14), (120, 2), (50, 24), (18, 30), (-30, 24)]
def shank_poly():
    return [(262, 560), (238, 440), (240, 340), (262, 250), (276, 180), (300, 150), (326, 180), (336, 300), (342, 560)]
_mc = {}
def matles_leg(th):
    key = round(th, 1)
    if key in _mc: return _mc[key]
    im = Image.new('RGBA', (MW, MH)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 590, 500, 630), 10, fill=(70, 94, 126, 255))
    thigh = [(40, 500), (300, 500), (342, 530), (338, 588), (40, 588), (24, 544)]
    foot = rot(MFOOT, th, *MANK)
    im.alpha_composite(shape_layer(MW, MH, [thigh, shank_poly(), foot], LIMB))
    calf = [(250, 470), (240, 400), (250, 330), (268, 268), (290, 300), (284, 400), (270, 470)]
    im.alpha_composite(shape_layer(MW, MH, [calf], MUSC, ICE, 2))
    ImageDraw.Draw(im).line([(268, 270), (270, 205), (MANK[0] - 52 * math.cos(math.radians(th)) + 40 * math.sin(math.radians(th)),
                                                       MANK[1] - 52 * math.sin(math.radians(th)) - 40 * math.cos(math.radians(th)) + 30)],
                            fill=TEND + (255,), width=12, joint='curve')
    _mc[key] = im
    return im

def s4(t, main, glow, fx):
    a = win(t, 23.8, 31.0, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 23.9, 24.4))
    put(main, tl('MATLES TEST', 'MontBlack', 58, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'prone, both knees flexed to 90 degrees', XC, 335, a * eo(prog(t, 24.1, 24.6)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 24.2, 25.0))
    drop = eio(prog(t, 25.6, 26.6))
    left_th = -28                                   # uninjured: rests plantarflexed (toes up)
    right_th = -28 + 30 * drop                      # injured: falls to neutral
    y0 = 430
    put(main, matles_leg(left_th), 325, y0 + MH / 2 + 20 * (1 - bp), a * bp)
    put(main, matles_leg(right_th), 695, y0 + MH / 2 + 20 * (1 - bp), a * bp)
    d = ImageDraw.Draw(main)
    # neutral reference (dashed) at the injured foot
    rq = a * eo(prog(t, 26.4, 26.8))
    if rq > 0:
        yy = y0 + MANK[1] - 20
        for x in range(740, 930, 26):
            if x < 940: d.line([(x, yy), (x + 13, yy)], fill=WARN + (int(255 * rq),), width=4)
    label(main, 'UNINJURED', 320, y0 + 690, a * eo(prog(t, 25.0, 25.4)), GOOD, 36, 'MontBlack', 2)
    label(main, 'rests plantarflexed', 320, y0 + 745, a * eo(prog(t, 25.1, 25.5)), ICE, 30, 'MontXB', 0)
    label(main, 'INJURED', 700, y0 + 690, a * eo(prog(t, 25.0, 25.4)), WARN, 36, 'MontBlack', 2)
    label(main, 'drops to neutral', 700, y0 + 745, a * eo(prog(t, 26.4, 26.8)), ICE, 30, 'MontXB', 0)
    sq_ = a * eo(prog(t, 27.6, 28.1))
    if sq_ > 0:
        y = 1430 + 20 * (1 - sq_ / max(a, 1e-3))
        put(main, card(860, 210), XC, y, sq_)
        label(main, 'SENSITIVITY', 295, y - 55, sq_, SUB, 30, 'MontXB', 2)
        p = R.eback(prog(t, 27.8, 28.3))
        put(main, tl('88%', 'MontBlack', 80, GOOD), 295, y + 18, sq_ * clamp(p), clamp(p, 0, 1.15))
        label(main, 'SPECIFICITY', 725, y - 55, sq_, SUB, 30, 'MontXB', 2)
        p = R.eback(prog(t, 28.2, 28.7))
        put(main, tl('85%', 'MontBlack', 80, GOOD), 725, y + 18, sq_ * clamp(p), clamp(p, 0, 1.15))
    label(main, 'Maffulli  ·  Am J Sports Med 1998', XC, 1600, a * eo(prog(t, 28.4, 28.9)), SUB, 28, 'MontXB', 0)

# ------------------------------------------------------------------ diagram 3: lateral ankle X-ray, tibio-first MT angle
XW, XH = 900, 700
XX0, XY0 = 60, 420
PIV = (380, 430)
def xray_layer(phi):
    im = Image.new('RGBA', (XW, XH))
    tib = [(330, 0), (420, 0), (424, 300), (452, 360), (460, 392), (380, 384), (300, 392), (300, 360), (322, 300)]
    fib = [(322, 0), (352, 0), (356, 340), (350, 420), (326, 446), (314, 400), (318, 340)]
    talus = [(PIV[0] + 62 * math.cos(math.radians(a_)), PIV[1] + 50 * math.sin(math.radians(a_))) for a_ in range(180, 361, 15)]
    talus += [(450, 470), (410, 490), (340, 486), (318, 450)]
    talus = rot([(x - PIV[0], y - PIV[1]) for x, y in talus], phi * 0.5, *PIV)
    calc = rot([(x - PIV[0], y - PIV[1]) for x, y in
                [(180, 470), (215, 445), (300, 462), (380, 490), (430, 520), (420, 560), (330, 572), (220, 560), (180, 520)]], phi, *PIV)
    nav = rot([(x - PIV[0], y - PIV[1]) for x, y in [(450, 478), (500, 470), (520, 500), (500, 528), (452, 524)]], phi, *PIV)
    cun = rot([(x - PIV[0], y - PIV[1]) for x, y in [(522, 478), (575, 480), (582, 528), (524, 530)]], phi, *PIV)
    mt1 = rot([(x - PIV[0], y - PIV[1]) for x, y in [(588, 486), (640, 492), (790, 498), (830, 494), (840, 512), (830, 528),
                                                      (790, 524), (640, 528), (588, 532)]], phi, *PIV)
    im.alpha_composite(shape_layer(XW, XH, [tib], BONE, ICE, 4))
    fl = shape_layer(XW, XH, [fib], (150, 180, 212), ICE, 3); fl.putalpha(fl.getchannel('A').point(lambda v: int(v * 0.6)))
    im.alpha_composite(fl)
    for p in (calc, talus, nav, cun, mt1):
        im.alpha_composite(shape_layer(XW, XH, [p], BONE, ICE, 4))
    # soft tissue: Achilles shadow from the calf down to the tuberosity
    tub = rot([(185 - PIV[0], 478 - PIV[1])], phi, *PIV)[0]
    big = Image.new('RGBA', (XW * SS, XH * SS))
    bd = ImageDraw.Draw(big)
    if phi > 10:
        bd.line([(232 * SS, 0), (214 * SS, 250 * SS), (tub[0] * SS, tub[1] * SS)], fill=TEND + (210,), width=20 * SS, joint='curve')
    else:   # rupture: tendon broken above the insertion
        mx = lambda u: ((214 + (tub[0] - 214) * u) * SS, (250 + (tub[1] - 250) * u) * SS)
        bd.line([(232 * SS, 0), (214 * SS, 250 * SS), mx(0.3)], fill=TEND + (210,), width=20 * SS, joint='curve')
        bd.line([mx(0.58), mx(1.0)], fill=TEND + (210,), width=20 * SS)
    im.alpha_composite(big.resize((XW, XH), Image.LANCZOS))
    return im

def s5(t, main, glow, fx):
    a = win(t, 30.8, 38.4, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 30.9, 31.4))
    put(main, tl('THE X-RAY CLUE', 'MontBlack', 58, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'lateral ankle film  ·  no fracture', XC, 335, a * eo(prog(t, 31.1, 31.6)), SKY, 34, 'MontXB', 0)
    m = eio(prog(t, 33.6, 35.0))
    ang = 125 + (88 - 125) * m
    phi = ang - 90
    bp = eo(prog(t, 31.2, 32.0))
    put(main, xray_layer(phi), XX0 + XW / 2, XY0 + XH / 2 + 20 * (1 - bp), a * bp)
    # axes: tibial shaft (vertical) and first metatarsal
    aq = a * eo(prog(t, 32.0, 32.5))
    if aq > 0:
        d = ImageDraw.Draw(main)
        col = GOOD if m < 0.5 else WARN
        tx = XX0 + 375
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        p0 = rot([(715 - PIV[0], 510 - PIV[1])], phi, *PIV)[0]
        tt = (375 - p0[0]) / c
        ix, iy = XX0 + 375, XY0 + p0[1] + tt * s
        for k in range(0, 34):
            y = XY0 + 10 + k * 22
            if y < iy - 6: d.line([(tx, y), (tx, min(y + 12, iy))], fill=SKY + (int(255 * aq),), width=4)
        ex, ey = XX0 + p0[0] + 160 * c, XY0 + p0[1] + 160 * s
        d.line([(ix, iy), (ex, ey)], fill=SKY + (int(255 * aq),), width=4)
        r = 90
        d.arc((ix - r, iy - r, ix + r, iy + r), -90, phi, fill=col + (int(255 * aq),), width=8)
        dot(d, ix, iy, 8, col, aq)
    pointer(main, a * eo(prog(t, 32.2, 32.7)), (XX0 + 224, XY0 + 170), 165, 700, 'ACHILLES', TEND, 29)
    vq = a * eo(prog(t, 32.4, 32.9))
    col = GOOD if m < 0.5 else WARN
    put(main, tl(f'{int(round(ang))}°', 'MontBlack', 96, col), 790, 470, vq)
    label(main, 'INTACT (mean)' if m < 0.5 else 'RUPTURE (mean)', 790, 555, vq, col, 32, 'MontBlack', 1)
    label(main, 'tibia vs 1st metatarsal', 790, 600, vq, SUB, 28, 'MontXB', 0)
    nq = a * eo(prog(t, 35.4, 35.9))
    if nq > 0:
        y = 1380 + 20 * (1 - nq / max(a, 1e-3))
        put(main, card(860, 250, fill=(66, 32, 38, 190), outline=WARN + (200,)), XC, y, nq)
        label(main, 'Ankle resting in neutral?', XC, y - 70, nq, WHITE, 42, 'MontBlack', 0)
        label(main, 'Angle 100° or less: test the Achilles', XC, y - 5, nq, WARN, 37, 'MontBlack', 0)
        label(main, 'Kiely et al  ·  J Emerg Med 2010  ·  27 vs 27', XC, y + 62, nq, SUB, 30, 'MontXB', 0)
    label(main, 'Rupture: 70-120°  ·  intact: 104-146°', XC, 1580, a * eo(prog(t, 35.8, 36.3)), SUB, 30, 'MontXB', 0)

TIPS = [('Squeeze the calf, prone', 'NO PLANTAR FLEXION = RUPTURE', 'Sensitivity 96%, specificity 93%'),
        ('No gap? Not ruled out', 'PALPABLE GAP: SENSITIVITY ONLY 73%', 'Maffulli, Am J Sports Med 1998'),
        ('Lateral film in plantar flexion', "TOYGAR AND ARNER SIGNS SHOW BEST", 'Park et al, J Foot Ankle Surg 2022')]
def s6(t, main, glow, fx):
    a = win(t, 38.2, 44.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 38.3, 38.8)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 38.9 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = [('Maffulli N.', 'Am J Sports Med. 1998;26(2):266–70'),
        ('Kiely PD et al.', 'J Emerg Med. 2010;41(6):718–22'),
        ('Park YH et al.', 'J Foot Ankle Surg. 2022;61(4):845–9')]
def s7(t, main, glow, fx):
    a = win(t, 44.2, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 300, a * eo(prog(t, 44.3, 44.7)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 44.4, 44.9))
    big = tl('2 TESTS', 'MontBlack', 140, WHITE)
    put(glow, big, XC, 455, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 455, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'At least 2 positive in all 174 ruptures', XC, 590, a * eo(prog(t, 44.6, 45.0)), SKY, 38, 'MontXB', 0)
    label(main, 'Squeeze + Matles before "sprain"', XC, 655, a * eo(prog(t, 44.8, 45.2)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 44.9, 45.3))
    put(main, card(860, 400, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 940, a * q)
    for i, (au, ref) in enumerate(REFS):
        y = 820 + i * 115
        label(main, au, XC, y, a * q, ICE, 31, 'MontXB', 0)
        label(main, ref, XC, y + 42, a * q, SUB, 29, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1260, a * eo(prog(t, 45.1, 45.5)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
