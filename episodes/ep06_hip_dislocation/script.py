"""Orthojets EP06 - ORTHO NUMBERS #3 - Traumatic hip dislocation: reduce within 6 hours.
Sources (verified on PubMed abstracts / metadata, 2026-10-24):
  Hougaard K, Thomsen PB. Arch Orthop Trauma Surg 1986;106(1):32-5. PMID 3566493, doi 10.1007/BF00435649
    98 adults, 100 traumatic posterior hip dislocations, follow-up at least 5 years. AVN of the femoral head
    4.8% in hips reduced within 6 h vs 52.9% in hips reduced more than 6 h after injury. Stewart-Milford grade III
    and IV: significantly more AVN than grade I and II.
  Hougaard K, Thomsen PB. J Bone Joint Surg Am 1987;69(5):679-83. PMID 3597467 (no DOI in PubMed)
    100 hips, average follow-up 14 years. Excellent/good result 88% if reduced within 6 h vs 42% if reduced
    after 6 h. Grade III: 31% excellent/good vs 90% for grade I.
  Kellam P, Ostrum RF. J Orthop Trauma 2016;30(1):10-6. PMID 26849386, doi 10.1097/BOT.0000000000000419
    Meta-analysis: odds ratio of AVN 5.627 for dislocations reduced after 12 h vs before 12 h.
  Gautier E et al. J Bone Joint Surg Br 2000;82(5):679-83. PMID 10963165, doi 10.1302/0301-620x.82b5.10426
    The primary blood supply of the femoral head is the deep branch of the medial femoral circumflex artery;
    intracapsular part runs along the posterosuperior neck as retinacular vessels.
  Cornwall R, Radomisli TE. Clin Orthop Relat Res 2000;(377):84-91. PMID 10943188, doi 10.1097/00003086-200008000-00012
    Nerve injury in about 10% of adults (5% of children); sciatic nerve (usually the peroneal division) most often.
  Mandell JC et al. Arthroscopy 2018;34(3):917-927. PMID 29146169, doi 10.1016/j.arthro.2017.08.295
    CT sensitivity 87.3% for intra-articular fragments after traumatic hip dislocation.
  Diagram: left hip, AP pelvis schematic (medial = viewer's left). Posterior dislocation: head superior to the
  acetabulum, femur adducted and internally rotated, Shenton's line broken (classic radiographic signs).
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(42.4, [(6.8, 1.5), (10.4, 1.8), (13.2, 1.5), (17.6, 1.6), (21.6, 2.0), (25.6, 1.8), (28.4, 1.8),
                     (36.4, 2.2), (41.4, 1.8)])

# ------------------------------------------------------------------ diagram (local canvas DW x DH)
DW, DH = 760, 1000
A = (330, 330)                       # acetabulum centre = femoral head centre when reduced
XRAY = (165, 208, 242)

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

PELVIS = [(60, 64), (150, 18), (300, 2), (440, 30), (540, 104), (530, 150), (478, 200), (440, 246), (404, 284),
          (396, 380), (380, 480), (360, 580), (330, 646), (260, 662), (160, 652), (56, 640), (36, 590), (44, 528),
          (140, 474), (232, 424), (272, 394), (200, 320), (130, 230), (78, 150)]
FORAMEN = [(305, 430), (326, 474), (314, 544), (266, 596), (196, 614), (160, 590), (172, 530), (228, 470)]
SHENTON_P = [(168, 548), (186, 510), (226, 470), (300, 428), (345, 404)]          # roof of the obturator foramen
SHENTON_F = [(178, 206), (198, 220), (220, 238), (240, 262), (250, 290)]          # inferomedial neck (femur-local)
VESSEL_F = [(236, 330), (275, 300), (305, 245), (312, 196), (290, 166), (250, 142), (212, 112)]   # MFCA deep branch (femur-local)

def xray(mask):
    m = mask
    M = np.asarray(m, np.float32) / 255
    B1 = np.asarray(m.filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
    rim = np.clip((1 - B1) * 3.0, 0, 1)
    rng = np.random.default_rng(5)
    n1 = np.asarray(Image.fromarray((rng.random((m.height // 4, m.width // 4)) * 255).astype(np.uint8)).resize(m.size, Image.BICUBIC), np.float32) / 255
    dens = (0.36 + 0.56 * rim) * (0.9 + 0.2 * n1)
    inner = np.clip(M * dens, 0, 1.2)
    img = Image.fromarray(np.clip(np.stack([inner * c for c in XRAY], -1), 0, 255).astype(np.uint8)).convert('RGBA')
    img.putalpha(Image.fromarray(np.clip(inner * 255 * 1.05, 0, 255).astype(np.uint8)))
    mb = m.point(lambda v: 255 if v > 128 else 0)
    edge = ImageChops.subtract(mb, mb.filter(ImageFilter.MinFilter(5)))
    e = Image.new('RGBA', m.size, ICE + (0,)); e.putalpha(edge.point(lambda v: 200 if v > 40 else 0))
    img.alpha_composite(e)
    return img

def pelvis_layer():
    SS = 3
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    d.polygon([(x * SS, y * SS) for x, y in catmull(PELVIS, 8)], fill=255)
    d.polygon([(x * SS, y * SS) for x, y in catmull(FORAMEN, 8)], fill=0)
    m = big.resize((DW, DH), Image.LANCZOS)
    im = xray(m)
    d = ImageDraw.Draw(im)
    # acetabular roof (sourcil) and teardrop
    r = 72
    d.arc((A[0] - r, A[1] - r, A[0] + r, A[1] + r), 192, 346, fill=(235, 245, 255, 235), width=6)
    d.arc((A[0] - r - 10, A[1] - r - 10, A[0] + r + 10, A[1] + r + 10), 200, 336, fill=XRAY + (120,), width=10)
    d.line(catmull([(262, 300), (252, 352), (262, 388), (282, 372), (280, 330)], 8, False), fill=(235, 245, 255, 200), width=4)
    return im

def femur_layer():
    img, edge = FEM.render()
    img = img.crop((0, 0, FEM.BW, 760)); edge = edge.crop((0, 0, FEM.BW, 760))
    e = Image.new('RGBA', img.size, ICE + (0,)); e.putalpha(edge.point(lambda v: 210 if v > 40 else 0))
    img.alpha_composite(e)
    f = np.clip(np.linspace(9, 0, img.height), 0, 1)[:, None] * np.ones((1, img.width))   # fade the cut shaft end
    img.putalpha(Image.fromarray((np.asarray(img.getchannel('A'), np.float32) * np.clip(f * 1.0, 0, 1) ** 0.5).astype(np.uint8)))
    return img

PELV = pelvis_layer()
FEMI = femur_layer()
HC = FEM.HEAD_C

def fmat(p):
    """femur-local -> canvas map for dislocation progress p (0 reduced, 1 posterior dislocation)"""
    phi = math.radians(17 * p); sx, sy = 1 - 0.14 * p, 1 - 0.07 * p
    R2 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]) @ np.diag([sx, sy])
    c = np.array([A[0] + 34 * p, A[1] - 150 * p])
    return R2, c

def fpt(p, q):
    R2, c = fmat(p)
    v = R2 @ (np.array(q, float) - np.array(HC, float)) + c
    return (float(v[0]), float(v[1]))

_dc = {}
def diagram(p, s):
    key = (round(p, 2), round(s, 3))
    if key in _dc: return _dc[key]
    R2, c = fmat(key[0])
    Ri = np.linalg.inv(R2)
    # output (X,Y) -> input: h + Ri @ ((X,Y) - c)
    a_, b_ = Ri[0]; d_, e_ = Ri[1]
    cc = HC[0] - (a_ * c[0] + b_ * c[1]); ff = HC[1] - (d_ * c[0] + e_ * c[1])
    fl = FEMI.transform((DW, DH), Image.AFFINE, (a_, b_, cc, d_, e_, ff), Image.BICUBIC)
    im = PELV.copy(); im.alpha_composite(fl)
    out = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    if len(_dc) > 80: _dc.clear()
    _dc[key] = out
    return out

def dp(cx, cy, s, q):  # canvas -> screen
    return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def lput(main, im, x0, y, a):
    put(main, im, x0 + im.width / 2 - 20, y, a)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('ORTHO NUMBERS  #3', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 480, a * eo(prog(t, 0.0, 0.3)))
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('6 HOURS', 'MontBlack', 158, WHITE)
    put(glow, k1, W / 2, 700, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 700, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.3, 0.8))
    put(main, tl('to reduce a dislocated hip', 'MontXB', 54, SKY), W / 2, 880 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.8, 1.3))
    yv = 1060 + 25 * (1 - q)
    put(main, tl('AVN 4.8%', 'MontBlack', 64, GOOD), 330, yv, a * q)
    put(main, tl('vs', 'MontXB', 46, ICE), 560, yv + 4, a * q)
    put(main, tl('52.9%', 'MontBlack', 64, WARN), 720, yv, a * q)
    q = eo(prog(t, 1.1, 1.6))
    put(main, tl('reduced within vs after 6 h', 'MontXB', 36, SUB), W / 2, 1160 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # centrepiece: AP hip, dislocation, blood supply, reduction
    a = win(t, 3.3, 18.9, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('POSTERIOR HIP DISLOCATION', 'MontBlack', 46, WHITE, track=2), W / 2, 266, a * q, 1, 8 * (1 - q))
    label(main, 'AP pelvis  ·  left hip', W / 2, 340, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    # dislocation progress: out 7.3-8.6, back in 16.0-17.0
    pd = eio(prog(t, 7.3, 8.6)) * (1 - eio(prog(t, 16.0, 17.0)))
    cx, cy, s = 425, 900, 0.95
    q = eo(prog(t, 3.6, 4.4))
    put(main, diagram(pd, s), cx, cy + 20 * (1 - q), a * q)
    if 7.3 < t < 8.8: fx['flash'] += 0.12 * math.exp(-((t - 8.2) / 0.15) ** 2)
    if 16.6 < t < 17.4: fx['flash'] += 0.10 * math.exp(-((t - 17.0) / 0.12) ** 2)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    # Shenton's line
    sq = eo(prog(t, 5.4, 6.2))
    if sq > 0:
        col = GOOD if pd < 0.15 else (WARN if t < 16.5 else GOOD)
        pp = [dp(cx, cy, s, q_) for q_ in catmull(SHENTON_P, 8, False)]
        fp = [dp(cx, cy, s, fpt(pd, q_)) for q_ in catmull(SHENTON_F, 8, False)]
        n = int(len(pp) * min(1, sq * 1.6))
        if n > 1: d.line(pp[:n], fill=col + (int(255 * a),), width=7, joint='curve')
        if sq > 0.6:
            m = int(len(fp) * clamp((sq - 0.6) / 0.4))
            if m > 1:
                d.line(fp[:m], fill=col + (int(255 * a),), width=7, joint='curve')
                g.line(fp[:m], fill=col + (int(110 * a),), width=14)
        g.line(pp[:max(n, 2)], fill=col + (int(110 * a),), width=14)
        if col == WARN:
            gx, gy = pp[-1]; pul = 0.5 + 0.5 * math.sin(t * 7)
            g.ellipse((gx - 26, gy - 26, gx + 26, gy + 26), fill=WARN + (int(160 * a * pul),))
    # labels on the right
    pointer(main, a * eo(prog(t, 4.6, 5.1)), dp(cx, cy, s, (372, 262)), 790, 600, 'ACETABULUM')
    hq = a * eo(prog(t, 5.0, 5.5))
    pointer(main, hq, dp(cx, cy, s, fpt(pd, (196, 110))), 790, 470, 'FEMORAL HEAD')
    # status line (Shenton)
    st_ = 'INTACT' if t < 7.8 else ('BROKEN' if t < 16.6 else 'RESTORED')
    scol = GOOD if st_ != 'BROKEN' else WARN
    sa = a * eo(prog(t, 5.8, 6.3))
    put(main, rich([("SHENTON'S LINE  ", 'MontBlack', WHITE), (st_, 'MontBlack', scol)], 40), W / 2, 1430, sa)
    # posture text
    pq = a * win(t, 8.8, 13.6, 0.45, 0.35)
    label(main, 'Head above and behind the socket', W / 2, 1520, pq, ICE, 34, 'MontXB', 0)
    label(main, 'Leg short, flexed, adducted, internally rotated', W / 2, 1585, pq, SKY, 32, 'MontXB', 0)
    # MFCA deep branch (moves with the femur)
    vq = eo(prog(t, 10.9, 12.0)) * (1 - eo(prog(t, 15.6, 16.0)))
    if vq > 0:
        vp = [dp(cx, cy, s, fpt(pd, q_)) for q_ in catmull(VESSEL_F, 8, False)]
        n = max(2, int(len(vp) * clamp(vq * 1.2)))
        pul = 0.75 + 0.25 * math.sin(t * 5)
        d.line(vp[:n], fill=(232, 120, 140, int(255 * a * vq)), width=8, joint='curve')
        g.line(vp[:n], fill=(232, 120, 140, int(170 * a * vq * pul)), width=18)
        pointer(main, a * vq * eo(prog(t, 11.6, 12.0)), vp[3], 830, 900, 'MFCA', (240, 150, 165))
        label(main, 'deep branch:', 830, 950, a * vq * eo(prog(t, 11.8, 12.2)), ICE, 28, 'MontXB', 0)
        label(main, 'main supply', 830, 992, a * vq * eo(prog(t, 11.8, 12.2)), ICE, 28, 'MontXB', 0)
        label(main, 'of the head', 830, 1034, a * vq * eo(prog(t, 11.8, 12.2)), ICE, 28, 'MontXB', 0)
    # clock 0 -> 6 h
    cq = a * win(t, 13.7, 18.9, 0.4, 0.45)
    if cq > 0:
        ccx, ccy, r = 830, 1220, 78
        d.ellipse((ccx - r, ccy - r, ccx + r, ccy + r), outline=SKY + (int(150 * cq),), width=5)
        sw = eio(prog(t, 14.0, 15.6)) * 6
        if sw > 0:
            d.arc((ccx - r + 12, ccy - r + 12, ccx + r - 12, ccy + r - 12), -90, -90 + 60 * sw, fill=WARN + (int(255 * cq),), width=14)
        put(main, tl(f'{sw:.0f} h', 'MontBlack', 44, WHITE), ccx, ccy, cq)
        label(main, 'Reduce it as an emergency', W / 2, 1520, cq * eo(prog(t, 14.0, 14.5)), WHITE, 40, 'MontBlack', 0)
        label(main, 'ideally within 6 hours', W / 2, 1585, cq * eo(prog(t, 14.3, 14.8)), SKY, 36, 'MontXB', 0)
    nq = a * win(t, 4.4, 7.4, 0.4, 0.3)
    put(main, pill('NORMAL', 'MontBlack', 30, NAVY, GOOD + (255,), h=62), 770, 1040, nq)
    if t > 16.8:
        rq = a * eo(prog(t, 16.9, 17.3))
        put(main, pill('REDUCED', 'MontBlack', 34, NAVY, GOOD + (255,), h=62), 770, 1040, rq)

def bar(main, glow, x, ybase, hmax, frac, col, a):
    if a <= 0 or frac <= 0: return
    h = hmax * frac
    put(main, card(170, max(40, int(h)), fill=col + (220,), outline=col + (255,), r=18), x, ybase - h / 2, a)
    g = ImageDraw.Draw(glow); g.rounded_rectangle((x - 85, ybase - h, x + 85, ybase), 18, fill=col + (int(60 * a),))

def s3(t, main, glow, fx):                                     # AVN 4.8% vs 52.9%
    a = win(t, 18.7, 23.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'AVASCULAR NECROSIS', W / 2, 260, a * eo(prog(t, 18.8, 19.3)), WHITE, 56, 'MontBlack', 2)
    label(main, 'of the femoral head', W / 2, 345, a * eo(prog(t, 19.0, 19.5)), SKY, 38, 'MontXB', 0)
    yb, hmax = 1260, 720
    d = ImageDraw.Draw(main)
    d.line([(150, yb), (880, yb)], fill=ICE + (int(160 * a),), width=4)
    for x, v, col, lab1, t0 in [(320, 4.8, GOOD, 'WITHIN 6 H', 19.6), (710, 52.9, WARN, 'AFTER 6 H', 20.2)]:
        p = eio(prog(t, t0, t0 + 1.0))
        bar(main, glow, x, yb, hmax * (52.9 / 55), p * v / 52.9, col, a)
        hy = yb - hmax * (52.9 / 55) * (p * v / 52.9)
        if p > 0:
            put(main, tl(f'{v * p:.1f}%', 'MontBlack', 84, col), x, hy - 70, a * clamp(p * 3))
        label(main, lab1, x, yb + 60, a * eo(prog(t, t0, t0 + 0.4)), WHITE, 40, 'MontBlack', 2)
        label(main, 'reduced', x, yb + 112, a * eo(prog(t, t0, t0 + 0.4)), SUB, 30, 'MontXB', 0)
    label(main, 'Hougaard & Thomsen  ·  Arch Orthop Trauma Surg 1986', W / 2, 1520, a * eo(prog(t, 20.6, 21.1)), ICE, 27, 'MontXB', 0)
    label(main, '100 posterior dislocations  ·  5 years or more', W / 2, 1575, a * eo(prog(t, 20.8, 21.3)), SUB, 28, 'MontXB', 0)

def ring(main, glow, cx, cy, r, frac, col, a, txt):
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(60, 90, 125, int(200 * a)), width=26)
    if frac > 0:
        d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * frac, fill=col + (int(255 * a),), width=26)
        g.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * frac, fill=col + (int(120 * a),), width=40)
    put(main, tl(txt, 'MontBlack', 70, col), cx, cy, a)

def s4(t, main, glow, fx):                                     # long-term result + meta-analysis
    a = win(t, 23.2, 30.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'LONG-TERM RESULT', W / 2, 260, a * eo(prog(t, 23.3, 23.8)), WHITE, 60, 'MontBlack', 3)
    label(main, 'excellent or good hip  ·  14 years on average', W / 2, 340, a * eo(prog(t, 23.5, 24.0)), SKY, 32, 'MontXB', 0)
    for x, v, col, lab1, t0 in [(300, 88, GOOD, 'WITHIN 6 H', 23.9), (730, 42, WARN, 'AFTER 6 H', 24.5)]:
        p = eio(prog(t, t0, t0 + 1.0))
        q = eo(prog(t, t0, t0 + 0.4))
        if q > 0:
            ring(main, glow, x, 600, 150, p * v / 100, col, a * q, f'{int(round(v * p))}%')
            label(main, lab1, x, 815, a * q, WHITE, 38, 'MontBlack', 2)
    label(main, 'Hougaard & Thomsen  ·  JBJS Am 1987', W / 2, 905, a * eo(prog(t, 24.9, 25.4)), SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 26.2, 26.7))
    if q > 0:
        y = 1210 + 30 * (1 - q)
        put(main, card(860, 380), 515, y, a * q)
        label(main, 'META-ANALYSIS', 515, y - 130, a * q, SKY, 30, 'MontXB', 4)
        label(main, 'Reduced after 12 hours', 515, y - 65, a * q, WHITE, 42, 'MontBlack', 0)
        p = R.eback(prog(t, 26.6, 27.1))
        put(main, tl('AVN odds x5.6', 'MontBlack', 84, WARN), 515, y + 30, a * q * clamp(p), clamp(p, 0, 1.2))
        label(main, 'Kellam & Ostrum  ·  J Orthop Trauma 2016', 515, y + 128, a * q, SUB, 28, 'MontXB', 0)

TIPS = [('Check the sciatic nerve', 'BEFORE AND AFTER REDUCTION', 'Injured in about 10% of adults (Cornwall 2000)'),
        ('CT after reduction', 'LOOK FOR FRAGMENTS AND INCONGRUITY', 'CT sensitivity for fragments 87% (Mandell)'),
        ('Grade III-IV: more AVN', 'STEWART-MILFORD GRADE PREDICTS IT', 'Good result 31% grade III vs 90% grade I')]
def s5(t, main, glow, fx):
    a = win(t, 30.4, 37.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 30.5, 31.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 31.1 + 1.7 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 44, WHITE); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s6(t, main, glow, fx):
    a = win(t, 37.4, 42.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 380, a * eo(prog(t, 37.5, 38.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 37.7, 38.3))
    big = tl('UNDER 6 H', 'MontBlack', 140, WHITE)
    put(glow, big, W / 2, 540, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 540, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Reduce the hip, save the head', W / 2, 680, a * eo(prog(t, 38.1, 38.6)), SKY, 42, 'MontXB', 0)
    q = a * eo(prog(t, 38.4, 38.9))
    put(main, tl('AVN 4.8%', 'MontBlack', 46, GOOD), 370, 790, q)
    put(main, tl('vs', 'MontXB', 36, ICE), 540, 793, q)
    put(main, tl('52.9%', 'MontBlack', 46, WARN), 665, 790, q)
    q = eo(prog(t, 38.8, 39.4))
    put(main, card(860, 330, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 1080, a * q)
    for i, (au, ref) in enumerate([('Hougaard K, Thomsen PB.', 'Arch Orthop Trauma Surg. 1986;106(1):32–5'),
                                   ('Hougaard K, Thomsen PB.', 'J Bone Joint Surg Am. 1987;69(5):679–83'),
                                   ('Kellam P, Ostrum RF.', 'J Orthop Trauma. 2016;30(1):10–6')]):
        y = 970 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1360, a * eo(prog(t, 39.4, 40.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
