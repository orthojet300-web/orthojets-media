"""Orthojets EP20 - CLINICAL RULES #5 - Ottawa knee rule: does this knee need an X-ray?
Sources (verified on PubMed abstracts / metadata, 2026-11-10):
  Stiell IG et al. Derivation of a decision rule for the use of radiography in acute knee injuries.
    Ann Emerg Med 1995;26(4):405-13. PMID 7574120, doi 10.1016/s0196-0644(95)70106-0. 1047 adults. Rule: (1) age 55 years
    or older, (2) tenderness at the head of the fibula, (3) isolated tenderness of the patella, (4) inability to flex to
    90 degrees, (5) inability to bear weight both immediately and in the ED (four steps). One or more = radiography.
    68 fractures: sensitivity 1.0 (95% CI 0.95-1.0), specificity 0.54. Radiography 68.6% -> 49.4% (28.0% relative reduction).
  Stiell IG et al. Prospective validation of a decision rule for the use of radiography in acute knee injuries.
    JAMA 1996;275(8):611-5. PMID 8594242. 1096 adults, 63 clinically important fractures: sensitivity 1.0 (95% CI 0.94-1.0);
    potential relative reduction in radiography 28%; probability of fracture if the rule is negative 0% (95% CI 0-0.4%).
  Stiell IG et al. Implementation of the Ottawa Knee Rule. JAMA 1997;278(23):2075-9. PMID 9403421. 3907 adults:
    knee radiography 77.6% -> 57.1% in the intervention group (relative reduction 26.4%); sensitivity 1.0 for 58 fractures;
    time in ED 85.7 vs 118.8 min, charges US$80 vs US$183; no missed fractures.
  Bachmann LM et al. Ann Intern Med 2004;140(2):121-4. PMID 14734335, doi 10.7326/0003-4819-140-5-200403020-00013.
    6 studies, 4249 adults: pooled sensitivity 98.5% (93.2-100), specificity 48.6% (43.4-51.0), negative LR 0.05.
  Bulloch B et al. Validation of the Ottawa Knee Rule in children. Ann Emerg Med 2003;42(1):48-55. PMID 12827123,
    doi 10.1067/mem.2003.196. 750 children aged 2-16 y, 70 fractures: sensitivity 100% (94.9-100), specificity 42.8%;
    radiographs reduced by 31.2%.
  Seaberg DC et al. Ann Emerg Med 1998;32(1):8-13. PMID 9656942, doi 10.1016/s0196-0644(98)70092-7. Pittsburgh rule
    sensitivity 99%, specificity 60%; Ottawa rule sensitivity 97%, specificity 27% (more specific Pittsburgh rule).
  Diagram (schematic, labelled): right knee, AP view, lateral side on the viewer's left. Distal femur with condyles and
  intercondylar notch, patella projected over the distal femur above the joint line, tibial plateau with spines, fibular
  head below the lateral tibial plateau (proximal tibiofibular joint), fibular shaft lateral to the tibia.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(6.4, 0.9), (8.4, 0.9), (10.4, 0.9), (12.6, 0.9), (14.8, 0.9), (17.0, 1.4), (21.0, 1.4),
                     (25.0, 1.6), (29.6, 1.2), (32.8, 1.4), (38.8, 1.6), (43.4, 1.4)])

XC = 510
BONE = (200, 222, 242)
SS = 2

def catmull(pts, n=8):
    P = pts + pts[:3]; out = []
    for i in range(len(pts)):
        p0, p1, p2, p3 = pts[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t ** 3
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    return out

# ------------------------------------------------------------------ right knee, AP (screen coordinates)
KX = 400
FEMUR = [(345, 360), (455, 360), (458, 560), (490, 620), (528, 665), (538, 698), (528, 724), (492, 738), (452, 732),
         (424, 716), (400, 704), (376, 716), (348, 732), (306, 738), (272, 724), (262, 698), (272, 665), (310, 620),
         (342, 560)]
TIBIA = [(266, 750), (300, 742), (360, 744), (384, 726), (398, 738), (414, 724), (438, 744), (500, 742), (534, 750),
         (530, 780), (505, 822), (462, 872), (454, 905), (454, 1100), (346, 1100), (346, 905), (338, 872), (296, 822),
         (270, 780)]
FIBULA = [(236, 818), (246, 790), (276, 778), (302, 788), (312, 816), (302, 848), (291, 870), (290, 1100), (264, 1100),
          (264, 870), (248, 850)]
PAT_C, PAT_R = (400, 640), (60, 76)
FIB_HEAD = (276, 816)
OX, OY, KW, KH = 120, 340, 520, 780          # local canvas for the knee

def _mask(pts):
    m = Image.new('L', (KW * SS, KH * SS))
    ImageDraw.Draw(m).polygon([((x - OX) * SS, (y - OY) * SS) for x, y in catmull(pts)], fill=255)
    return m

_kc = {}
def knee_layer(build=1.0):
    key = round(build, 2)
    if key in _kc: return _kc[key]
    big = Image.new('RGBA', (KW * SS, KH * SS))
    # vertical fade at the shaft ends
    yy = (np.arange(KH * SS) / SS + OY)
    fade = np.clip((yy - 360) / 120, 0, 1) * np.clip((1100 - yy) / 140, 0, 1)
    fade = Image.fromarray(np.repeat((fade[:, None] * 255), KW * SS, 1).astype(np.uint8))
    parts = [(FIBULA, 0.0), (TIBIA, 0.15), (FEMUR, 0.3)]
    for pts, t0 in parts:
        a = clamp((build - t0) * 2.2)
        if a <= 0: continue
        m = _mask(pts)
        m = Image.fromarray((np.asarray(m, np.float32) * np.asarray(fade, np.float32) / 255).astype(np.uint8))
        fill = Image.new('RGBA', big.size, BONE + (0,)); fill.putalpha(m.point(lambda v: int(v * 0.42 * a)))
        edge = Image.fromarray(np.clip(np.asarray(m, np.int16) - np.asarray(m.filter(ImageFilter.MinFilter(7)), np.int16), 0, 255).astype(np.uint8))
        e = Image.new('RGBA', big.size, ICE + (0,)); e.putalpha(edge.point(lambda v: int(v * 0.95 * a)))
        big.alpha_composite(fill); big.alpha_composite(e)
    # patella: projected over the distal femur
    a = clamp((build - 0.55) * 2.5)
    if a > 0:
        d = ImageDraw.Draw(big)
        (cx, cy), (rx, ry) = PAT_C, PAT_R
        bb = ((cx - rx - OX) * SS, (cy - ry - OY) * SS, (cx + rx - OX) * SS, (cy + ry - OY) * SS)
        d.ellipse(bb, fill=BONE + (int(70 * a),), outline=WHITE + (int(230 * a),), width=3 * SS)
    big = big.filter(ImageFilter.GaussianBlur(0.6 * SS))
    im = big.resize((KW, KH), Image.LANCZOS)
    if len(_kc) > 40: _kc.clear()
    _kc[key] = im
    return im

def draw_knee(main, glow, a, t, build=1.0, hi_fib=0.0, hi_pat=0.0, sc=1.0, cx=None, cy=None):
    """hi_*: 0..1 highlight (coral glow + pulsing press point)"""
    im = knee_layer(build)
    if cx is None: cx, cy = OX + KW / 2, OY + KH / 2
    if sc != 1.0: im = im.resize((int(KW * sc), int(KH * sc)), Image.LANCZOS)
    put(main, im, cx, cy, a)
    def S(p): return (cx + (p[0] - OX - KW / 2) * sc, cy + (p[1] - OY - KH / 2) * sc)
    g = ImageDraw.Draw(glow); d = ImageDraw.Draw(main)
    pul = 0.6 + 0.4 * math.sin(t * 6)
    if hi_fib > 0:
        x, y = S(FIB_HEAD); r = 44 * sc
        g.ellipse((x - r, y - r, x + r, y + r), fill=WARN + (int(150 * hi_fib * pul * a),))
        d.ellipse((x - 30 * sc, y - 34 * sc, x + 30 * sc, y + 34 * sc), outline=WARN + (int(255 * hi_fib * a),), width=max(2, int(5 * sc)))
    if hi_pat > 0:
        x, y = S(PAT_C); rx, ry = PAT_R[0] * sc, PAT_R[1] * sc
        g.ellipse((x - rx - 12, y - ry - 12, x + rx + 12, y + ry + 12), fill=WARN + (int(140 * hi_pat * pul * a),))
        d.ellipse((x - rx, y - ry, x + rx, y + ry), outline=WARN + (int(255 * hi_pat * a),), width=max(2, int(6 * sc)))
    return S

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=1)
    ex = lx + lab.width / 2 - 14 if tip[0] > lx else lx - lab.width / 2 + 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def crit_row(main, a, y, n, txt, active):
    if a <= 0: return
    fill = (70, 34, 44, 210) if active else (30, 54, 84, 170)
    oc = WARN if active else SKY
    put(main, card(860, 88, fill=fill, outline=oc + (220,), r=30), XC, y, a)
    nm = tl(str(n), 'MontBlack', 46, oc); put(main, nm, 125, y, a)
    lab = tl(txt, 'MontBlack', 35, WHITE); put(main, lab, 175 + lab.width / 2 - 20, y, a)

# ------------------------------------------------------------------ right-hand panel visuals (x 650-950)
PX = 800
def panel_age(main, glow, a, t, p):
    if a <= 0: return
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    r = 112
    d.ellipse((PX - r, 640 - r, PX + r, 640 + r), outline=WARN + (int(255 * a),), width=8)
    g.ellipse((PX - r, 640 - r, PX + r, 640 + r), outline=WARN + (int(90 * a),), width=20)
    big = tl('55+', 'MontBlack', 92, WHITE); put(main, big, PX, 632, a, clamp(R.eback(p), 0, 1.15))
    label(main, 'YEARS', PX, 800, a, SUB, 30, 'MontXB', 2)

def panel_tender(main, a, t, what):
    if a <= 0: return
    # a fingertip pressing (simple outline), then the word
    d = ImageDraw.Draw(main)
    k = 0.5 + 0.5 * math.sin(t * 5)
    y0 = 560 + 16 * k
    d.rounded_rectangle((PX - 34, y0 - 120, PX + 34, y0), 34, outline=ICE + (int(230 * a),), width=6)
    d.arc((PX - 22, y0 - 40, PX + 22, y0 - 6), 200, 340, fill=ICE + (int(160 * a),), width=4)
    for i in range(6):
        ang = math.pi * (0.15 + 0.14 * i); r0, r1 = 40, 62 + 10 * k
        d.line([(PX + r0 * math.cos(ang), y0 + 10 + r0 * math.sin(ang) - 30), (PX + r1 * math.cos(ang), y0 + 10 + r1 * math.sin(ang) - 30)],
               fill=WARN + (int(255 * a),), width=5)
    label(main, 'TENDER', PX, 730, a, WARN, 44, 'MontBlack', 2)
    label(main, what, PX, 790, a, WHITE, 30, 'MontXB', 0)

def panel_flex(main, glow, a, t, p):
    """side view: thigh fixed, shank flexes; it stops at 60 deg, target 90 deg (dashed ghost)"""
    if a <= 0: return
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    K = (790, 560); LT, LS = 140, 190
    hip = (K[0] - LT, K[1])
    target = 90; reach = 60 * eio(clamp(p * 1.3))
    def end(th): return (K[0] + LS * math.cos(math.radians(th)), K[1] + LS * math.sin(math.radians(th)))
    # ghost 90
    e9 = end(target)
    for k in range(0, 10, 2):
        u0, u1 = k / 10, (k + 1) / 10
        d.line([(K[0] + (e9[0] - K[0]) * u0, K[1] + (e9[1] - K[1]) * u0), (K[0] + (e9[0] - K[0]) * u1, K[1] + (e9[1] - K[1]) * u1)],
               fill=GOOD + (int(220 * a),), width=6)
    label(main, '90°', e9[0] + 2, e9[1] + 40, a, GOOD, 36, 'MontBlack', 0)
    # arc of the reached flexion
    rr = 70
    if reach > 1:
        d.arc((K[0] - rr, K[1] - rr, K[0] + rr, K[1] + rr), 0, reach, fill=WARN + (int(255 * a),), width=7)
    d.line([hip, K], fill=BONE + (int(255 * a),), width=26)
    es = end(reach)
    d.line([K, es], fill=BONE + (int(255 * a),), width=22)
    g.line([K, es], fill=WARN + (int(70 * a),), width=30)
    d.ellipse((K[0] - 20, K[1] - 20, K[0] + 20, K[1] + 20), fill=WARN + (int(255 * a),))
    label(main, 'THIGH', hip[0] + 60, K[1] - 46, a, SUB, 27, 'MontXB', 1)
    lab = tl(f'{reach:.0f}°', 'MontBlack', 40, WARN)
    put(main, lab, K[0] + 118, K[1] + 50, a)
    label(main, 'STUCK SHORT', PX, 860, a * clamp(p * 2 - 0.8), WARN, 36, 'MontBlack', 1)
    label(main, 'of 90° flexion', PX, 910, a * clamp(p * 2 - 0.8), WHITE, 30, 'MontXB', 0)

def foot(d, x, y, a, col, left):
    s = -1 if left else 1
    d.ellipse((x - 22, y - 34, x + 22, y + 40), fill=col + (int(255 * a),))
    for i, (dx, dy, r) in enumerate([(-14, -50, 9), (-2, -56, 8), (9, -53, 7), (17, -46, 6), (22, -37, 5)]):
        d.ellipse((x + s * dx - r, y + dy - r, x + s * dx + r, y + dy + r), fill=col + (int(255 * a),))

def panel_steps(main, glow, a, t, p):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    pos = [(760, 960), (840, 860), (760, 760), (840, 660)]
    n = p * 4.6
    for i, (x, y) in enumerate(pos):
        q = clamp(n - i)
        if q <= 0: continue
        foot(d, x, y, a * q, WARN if i < 4 else GOOD, i % 2 == 0)
        put(main, tl(str(i + 1), 'MontBlack', 34, WHITE), x + (-62 if i % 2 == 0 else 62), y, a * q)
    label(main, '4 STEPS', PX, 545, a, WARN, 48, 'MontBlack', 2)
    label(main, 'unable both at injury', PX, 1022, a * clamp(p * 2 - 0.6), WHITE, 28, 'MontXB', 0)
    label(main, 'and in the ED', PX, 1062, a * clamp(p * 2 - 0.6), WHITE, 28, 'MontXB', 0)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #5', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Twisted knee, limping', 'MontXB', 48, WHITE), W / 2, 522 + 20 * (1 - q), a * q)
    put(main, tl('into the ED:', 'MontXB', 48, WHITE), W / 2, 586 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('X-RAY?', 'MontBlack', 150, SKY)
    put(glow, k1, W / 2, 745, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 745, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.45, 0.9))
    put(main, pill('OTTAWA KNEE RULE', 'MontBlack', 36, WHITE, WARN + (255,), h=74), W / 2, 895 + 20 * (1 - q), a * q)
    q = eo(prog(t, 0.1, 0.6)); g = eio(prog(t, 0.6, 1.4))
    draw_knee(main, glow, a * q, t, 1.0, g, g, sc=0.62, cx=W / 2 + 10, cy=1250 + 20 * (1 - q))

CRIT = ['Age 55 or older', 'Tender fibular head', 'Isolated patellar tenderness', "Can't flex to 90°", "Can't bear weight: 4 steps"]
T_C = [5.4, 7.4, 9.4, 11.4, 13.6]
def s2(t, main, glow, fx):                                     # centrepiece: the knee + the 5 criteria
    a = win(t, 3.3, 18.6, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('OTTAWA KNEE RULE', 'MontBlack', 60, WHITE, track=2), XC, 240, a * q, 1, 8 * (1 - q))
    label(main, 'X-ray if ANY 1 of these 5', XC, 308, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    build = prog(t, 3.5, 5.0)
    act = -1
    for i in range(5):
        end_ = T_C[i + 1] - 0.3 if i < 4 else 16.0
        if T_C[i] - 0.3 <= t < end_: act = i
    hf = win(t, T_C[1] - 0.2, T_C[2] - 0.3, 0.3, 0.3)
    hp = win(t, T_C[2] - 0.2, T_C[3] - 0.3, 0.3, 0.3)
    S = draw_knee(main, glow, a, t, build, hf, hp)
    la = a * eo(prog(t, 4.6, 5.2))
    label(main, 'RIGHT KNEE', 170, 392, la, SUB, 27, 'MontXB', 1)
    label(main, 'AP VIEW', 170, 428, la, SUB, 27, 'MontXB', 1)
    pointer(main, la, (PAT_C[0] - 40, PAT_C[1] - 40), 180, 520, 'PATELLA', ICE, 29)
    if la > 0:
        ImageDraw.Draw(main).line([(FIB_HEAD[0] - 26, FIB_HEAD[1] + 8), (205, 880)], fill=ICE + (int(200 * la),), width=3)
        dot(ImageDraw.Draw(main), FIB_HEAD[0] - 26, FIB_HEAD[1] + 8, 6, ICE, la)
        label(main, 'FIBULAR', 140, 908, la, ICE, 29, 'MontXB', 1)
        label(main, 'HEAD', 140, 944, la, ICE, 29, 'MontXB', 1)
    pointer(main, la, (440, 1000), 560, 1060, 'TIBIA', ICE, 27)
    label(main, 'LATERAL', 165, 690, la * 0.9, SUB, 27, 'MontXB', 1)
    # right panel per criterion
    pa = lambda i: a * win(t, T_C[i] - 0.3, (T_C[i + 1] - 0.3) if i < 4 else 16.0, 0.3, 0.3)
    panel_age(main, glow, pa(0), t, prog(t, T_C[0] - 0.3, T_C[0] + 0.3))
    panel_tender(main, pa(1), t, 'fibular head')
    panel_tender(main, pa(2), t, 'patella only')
    panel_flex(main, glow, pa(3), t, prog(t, T_C[3] - 0.2, T_C[3] + 1.0))
    panel_steps(main, glow, pa(4), t, prog(t, T_C[4] - 0.3, T_C[4] + 1.2))
    for i, txt in enumerate(CRIT):
        ra = a * eo(prog(t, T_C[i] - 0.3, T_C[i] + 0.2))
        crit_row(main, ra, 1150 + i * 100, i + 1, txt, act == i)
    fq = a * eo(prog(t, 16.0, 16.5))
    if fq > 0:
        p = R.eback(prog(t, 16.0, 16.5))
        big = tl('ANY 1 = X-RAY', 'MontBlack', 60, SKY)
        put(glow, big, XC, 1660, fq * 0.3, clamp(p, 0, 1.1)); put(main, big, XC, 1660, fq, clamp(p, 0, 1.1))

def hbar(main, glow, a, y, name, v, col, p, x0=110, x1=910):
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    lab = tl(name, 'MontBlack', 40, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 54, a)
    d.rounded_rectangle((x0, y, x1, y + 40), 20, fill=(60, 90, 125, int(200 * a)))
    xe = x0 + (x1 - x0) * v / 100 * p
    if p > 0:
        d.rounded_rectangle((x0, y - 4, max(x0 + 40, xe), y + 44), 24, fill=col + (int(255 * a),))
        g.rounded_rectangle((x0, y - 4, max(x0 + 40, xe), y + 44), 24, fill=col + (int(80 * a),))
    vt = tl(f'{v * p:.1f}%', 'MontBlack', 44, col); put(main, vt, x1 - vt.width / 2 + 20, y - 54, a * clamp(p * 2))

def stat_card(main, a, y, head, l1, l2, col):
    if a <= 0: return
    put(main, card(860, 220), XC, y, a)
    lab = tl(head, 'MontBlack', 38, col); put(main, lab, 115 + lab.width / 2 - 20, y - 62, a)
    lab = tl(l1, 'MontXB', 32, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y + 2, a)
    lab = tl(l2, 'MontXB', 30, SUB); put(main, lab, 115 + lab.width / 2 - 20, y + 60, a)

def s3(t, main, glow, fx):                                     # evidence
    a = win(t, 18.4, 26.4, 0.4, 0.45)
    if a <= 0: return
    p = R.eback(prog(t, 18.5, 19.0))
    big = tl('100%', 'MontBlack', 120, GOOD)
    put(glow, big, XC, 270, a * 0.3 * clamp(p), clamp(p, 0, 1.15)); put(main, big, XC, 270, a * clamp(p), clamp(p, 0, 1.15))
    label(main, 'sensitivity for knee fractures', XC, 370, a * eo(prog(t, 18.7, 19.2)), SKY, 34, 'MontXB', 0)
    stat_card(main, a * eo(prog(t, 19.2, 19.6)), 540, 'DERIVATION  ·  1995', '1047 adults  ·  68 fractures',
              'Sensitivity 100%  ·  Specificity 54%', SKY)
    stat_card(main, a * eo(prog(t, 19.8, 20.2)), 790, 'VALIDATION  ·  1996', '1096 adults  ·  63 fractures',
              'Rule negative: fracture risk 0% (CI 0-0.4%)', SKY)
    hq = a * eo(prog(t, 21.6, 22.0))
    label(main, 'Meta-analysis  ·  6 studies  ·  4249 adults', XC, 990, hq, WHITE, 34, 'MontBlack', 0)
    hbar(main, glow, a * eo(prog(t, 21.9, 22.3)), 1140, 'Sensitivity', 98.5, GOOD, eio(prog(t, 22.1, 22.9)))
    hbar(main, glow, a * eo(prog(t, 22.6, 23.0)), 1330, 'Specificity', 48.6, WARN, eio(prog(t, 22.8, 23.6)))
    cq = a * eo(prog(t, 23.8, 24.2))
    label(main, 'Negative LR 0.05', XC, 1490, cq, GOOD, 46, 'MontBlack', 0)
    label(main, 'Stiell, Ann Emerg Med 1995 · JAMA 1996', XC, 1570, cq, SUB, 28, 'MontXB', 0)
    label(main, 'Bachmann et al, Ann Intern Med 2004', XC, 1615, cq, SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # what it changed (implementation)
    a = win(t, 26.2, 33.6, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 26.3, 26.8))
    put(main, tl('FEWER X-RAYS', 'MontBlack', 64, WHITE, track=3), XC, 245, a * q, 1, 8 * (1 - q))
    label(main, 'implementation trial  ·  3907 adults', XC, 315, a * eo(prog(t, 26.5, 27.0)), SKY, 32, 'MontXB', 0)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    base, hmax = 1020, 560
    for i, (x, v, col, nm) in enumerate([(330, 77.6, WARN, 'BEFORE'), (690, 57.1, GOOD, 'WITH THE RULE')]):
        t0 = 26.9 + 0.8 * i; qq = a * eo(prog(t, t0, t0 + 0.4))
        if qq <= 0: continue
        p = eio(prog(t, t0 + 0.1, t0 + 1.0))
        top = base - hmax * v / 100 * p
        d.rounded_rectangle((x - 100, top, x + 100, base), 26, fill=col + (int(235 * qq),))
        g.rounded_rectangle((x - 100, top, x + 100, base), 26, fill=col + (int(60 * qq),))
        put(main, tl(f'{v * p:.1f}%', 'MontBlack', 54, col), x, top - 50, qq)
        label(main, nm, x, base + 46, qq, WHITE, 32, 'MontBlack', 1)
    label(main, 'patients sent for knee X-ray', XC, 1130, a * eo(prog(t, 27.6, 28.0)), SUB, 30, 'MontXB', 0)
    cq = a * eo(prog(t, 28.6, 29.0))
    if cq > 0:
        y = 1380
        put(main, card(880, 360, fill=(24, 60, 60, 190), outline=GOOD + (220,)), XC, y, cq)
        label(main, '26.4% relative reduction', XC, y - 120, cq, GOOD, 44, 'MontBlack', 0)
        label(main, 'Sensitivity 100% for 58 fractures', XC, y - 50, cq, WHITE, 34, 'MontXB', 0)
        label(main, 'Time in ED 85.7 vs 118.8 min', XC, y + 10, a * eo(prog(t, 29.0, 29.4)), WHITE, 34, 'MontXB', 0)
        label(main, 'Charges US$80 vs US$183', XC, y + 70, a * eo(prog(t, 29.2, 29.6)), WHITE, 34, 'MontXB', 0)
        label(main, 'Stiell et al, JAMA 1997', XC, y + 132, cq, SKY, 28, 'MontXB', 0)

TIPS = [('Children too (2 to 16 y)', '750 KIDS: SENS 100%, SPEC 42.8%', 'Bulloch et al, Ann Emerg Med 2003'),
        ('Weight-bearing = 4 steps', 'UNABLE AT INJURY AND IN THE ED', 'Stiell et al, Ann Emerg Med 1995'),
        ('Want more specificity?', 'PITTSBURGH RULE: SPEC 60% VS 27%', 'Seaberg et al, Ann Emerg Med 1998')]
def s5(t, main, glow, fx):
    a = win(t, 33.4, 40.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 33.5, 34.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 34.1 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 29, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = ['Stiell et al  ·  Ann Emerg Med 1995', 'Stiell et al  ·  JAMA 1996', 'Stiell et al  ·  JAMA 1997',
        'Bachmann et al  ·  Ann Intern Med 2004', 'Bulloch et al  ·  Ann Emerg Med 2003', 'Seaberg et al  ·  Ann Emerg Med 1998']
def s6(t, main, glow, fx):
    a = win(t, 40.0, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 270, a * eo(prog(t, 40.1, 40.5)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 40.2, 40.7))
    big = tl('ANY 1 = X-RAY', 'MontBlack', 92, WHITE)
    put(glow, big, XC, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'None of the 5: no X-ray needed', XC, 515, a * eo(prog(t, 40.4, 40.8)), GOOD, 36, 'MontXB', 0)
    label(main, '55+  ·  fibular head  ·  patella', XC, 580, a * eo(prog(t, 40.6, 41.0)), SUB, 32, 'MontXB', 0)
    label(main, 'flex to 90°  ·  4 steps', XC, 630, a * eo(prog(t, 40.6, 41.0)), SUB, 32, 'MontXB', 0)
    q = eo(prog(t, 41.2, 41.6))
    put(main, card(880, 470, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 980, a * q)
    for i, r in enumerate(REFS):
        label(main, r, XC, 810 + i * 68, a * q, ICE if i % 2 == 0 else SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1300, a * eo(prog(t, 41.8, 42.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
