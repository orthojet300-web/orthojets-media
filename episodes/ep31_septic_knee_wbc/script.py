"""Orthojets EP31 - ORTHO NUMBERS #7 - Hot swollen knee: synovial WBC 50,000 and the likelihood of septic arthritis.
Sources (verified on PubMed abstracts / metadata, 2026-11-25; PubMed is the only verification source):
  Margaretten ME, Kohlwes J, Moore D, Bent S. Does this adult patient have septic arthritis? JAMA 2007;297(13):1478-88.
    PMID 17405973, doi 10.1001/jama.297.13.1478. Rational clinical examination review of nongonococcal bacterial arthritis:
    14 studies, 6242 patients, 653 met the gold standard for septic arthritis.
    Sensitivities: joint pain 85%, history of joint swelling 78%, fever 57%, sweats 27%, rigors 19%
    (only pain, swelling and fever occur in more than 50% of patients).
    Risk factors that significantly increase probability: age, diabetes mellitus, rheumatoid arthritis, joint surgery,
    hip or knee prosthesis, skin infection, HIV-1 infection.
    Synovial fluid WBC summary likelihood ratios: under 25,000/uL LR 0.32 (0.23-0.43); 25,000 or more LR 2.9 (2.5-3.4);
    over 50,000 LR 7.7 (5.7-11.0); over 100,000 LR 28.0 (12.0-66.0).
    PMN 90% or more: LR 3.4 (2.8-4.2); PMN under 90%: LR 0.34 (0.25-0.47).
    The 2 most powerful readily available lab findings are synovial WBC and % PMN from arthrocentesis; they are required to
    assess the likelihood of septic arthritis before Gram stain and culture results are known.
  Ghanem E, Parvizi J, et al. Cell count and differential of aspirated fluid in the diagnosis of infection at the site of
    total knee arthroplasty. J Bone Joint Surg Am 2008;90(8):1637-43. PMID 18676892, doi 10.2106/JBJS.G.00470.
    429 revision TKA knees (161 infected). Optimal cut-offs: fluid leukocyte count over 1,100 cells/uL and neutrophils over 64%.
    Both below cut-off: NPV 98.2%; both above: infection confirmed in 98.6%.
  Diagram: RIGHT knee, AP (anterior) view: lateral on the viewer's LEFT (fibular head lateral and below the lateral tibial
  plateau). Femoral shaft, condyles with intercondylar notch, tibial plateau and spines, fibula lateral. Patella projected
  over the distal femur. Effusion distends the suprapatellar pouch above the patella and the joint either side.
  Aspiration needle enters from the lateral side at the level of the superior pole of the patella, into the suprapatellar pouch.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(49.5, [(8.4, 1.4), (13.0, 1.2), (17.3, 1.2), (21.0, 1.4), (29.2, 1.6), (36.4, 1.6), (42.6, 1.6),
                     (48.2, 0.8)])

DW, DH = 760, 900
XRAY = (170, 210, 242)
FLUID = SKY                    # synovial fluid / effusion (series accent, brand palette)
SS = 2
XC = 515

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
FADE = np.clip((_yy - 4) / 150.0, 0, 1) * np.clip((DH - 6 - _yy) / 150.0, 0, 1)
def faded(m, f=FADE): return Image.fromarray((np.asarray(m, np.float32) * f).astype(np.uint8))
def union(ms):
    u = ms[0]
    for m in ms[1:]: u = ImageChops.lighter(u, m)
    return u
def lerp(a, b, r): return a + (b - a) * r
def scale_pts(pts, c, s): return [(c[0] + (x - c[0]) * s, c[1] + (y - c[1]) * s) for x, y in pts]

# ------------------------------------------------------------------ right knee, AP view (lateral = viewer's left)
FEM = [(312, 0), (452, 0), (458, 200), (476, 330), (520, 410), (548, 470), (552, 530), (530, 572), (470, 576),
       (420, 552), (392, 520), (366, 520), (338, 552), (290, 574), (232, 570), (212, 532), (218, 470), (248, 410),
       (290, 330), (306, 200)]
TIB = [(214, 604), (250, 588), (330, 592), (364, 590), (376, 572), (388, 572), (400, 590), (440, 590), (520, 588),
       (552, 606), (548, 640), (506, 676), (470, 730), (456, 820), (452, 900), (318, 900), (316, 820), (300, 730),
       (262, 676), (220, 642)]
FIB = [(176, 648), (212, 636), (240, 652), (244, 684), (230, 714), (226, 800), (226, 900), (200, 900), (198, 800),
       (192, 712), (174, 684)]
PAT_C, PAT_RX, PAT_RY = (384, 440), 74, 92
# capsule / suprapatellar pouch: normal and distended (effusion)
POUCH_N = [(300, 300), (384, 268), (466, 300), (520, 420), (556, 520), (560, 600), (384, 600), (206, 600),
           (210, 520), (246, 420)]
POUCH_E = [(270, 170), (384, 120), (498, 170), (572, 330), (602, 480), (590, 610), (384, 616), (178, 610),
           (166, 480), (196, 330)]
NEEDLE_TIP = (268, 340)        # lateral, level of the superior pole of the patella (pole at y 348), inside the pouch

def pouch_pts(r): return [(lerp(a[0], b[0], r), lerp(a[1], b[1], r)) for a, b in zip(POUCH_N, POUCH_E)]

_K = {}
def knee(r, fill=0.0):
    """r: effusion 0..1 ; fill: how turbid/coloured the fluid looks"""
    k = (round(r * 10) / 10, round(fill * 4) / 4)
    if k in _K: return _K[k]
    fm = faded(poly_mask(FEM)); tm = faded(poly_mask(TIB)); bm = faded(poly_mask(FIB))
    pm = ell_mask(PAT_C, PAT_RX, PAT_RY)
    cm = faded(poly_mask(pouch_pts(k[0])))
    soft_m = union([fm, tm, bm, cm]).filter(ImageFilter.MaxFilter(41)).filter(ImageFilter.GaussianBlur(20))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)))
    lay = Image.new('RGBA', (DW, DH))
    lay.alpha_composite(volume(soft_m, (64, 98, 140), alpha=0.5, r=18, relief=0.6))
    fc = tuple(int(lerp(c1, c2, k[1])) for c1, c2 in zip((120, 170, 215), FLUID))
    lay.alpha_composite(volume(cm, fc, alpha=0.45 + 0.35 * k[0], r=14, relief=0.7))
    lay.alpha_composite(bone_xray(bm, XRAY, alpha=0.85, seed=14))
    lay.alpha_composite(bone_xray(tm, XRAY, alpha=0.96, seed=13))
    lay.alpha_composite(bone_xray(fm, XRAY, alpha=0.96, seed=11))
    lay.alpha_composite(bone_xray(pm, XRAY, alpha=0.55, seed=12))      # patella superimposed on the femur
    pad = 40
    big = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); big.alpha_composite(lay, (pad, pad))
    big = drop_shadow(big, dx=10, dy=16, blur=14, opacity=0.55)
    _K[k] = big.crop((pad, pad, pad + DW, pad + DH))
    return _K[k]

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

def syringe(main, cx, cy, s, a, ins, draw_r):
    """needle + syringe coming from the lateral side (viewer's left); ins 0..1 insertion, draw_r 0..1 fluid drawn"""
    if a <= 0: return
    tip = dp(NEEDLE_TIP, cx, cy, s)
    ang = math.radians(8)                       # slight downward course toward the tip
    ux, uy = math.cos(ang), math.sin(ang)
    back = 80 * (1 - ins) * s
    tx, ty = tip[0] - ux * back, tip[1] - uy * back
    nl = 130 * s; bl = 190 * s; bw = 56 * s
    hx, hy = tx - ux * nl, ty - uy * nl          # hub
    ex, ey = hx - ux * bl, hy - uy * bl          # barrel end
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    al = int(255 * a)
    d.line([(hx, hy), (tx, ty)], fill=(220, 228, 238, al), width=max(3, int(6 * s)))
    nx, ny = -uy * bw / 2, ux * bw / 2
    barrel = [(hx + nx, hy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (hx - nx, hy - ny)]
    d.polygon(barrel, fill=(210, 230, 248, int(70 * a)), outline=(230, 240, 250, al))
    if draw_r > 0:
        fx_, fy_ = lerp(hx, ex, 0.04), lerp(hy, ey, 0.04)
        gx_, gy_ = lerp(hx, ex, 0.04 + 0.7 * draw_r), lerp(hy, ey, 0.04 + 0.7 * draw_r)
        n2x, n2y = nx * 0.8, ny * 0.8
        d.polygon([(fx_ + n2x, fy_ + n2y), (gx_ + n2x, gy_ + n2y), (gx_ - n2x, gy_ - n2y), (fx_ - n2x, fy_ - n2y)],
                  fill=FLUID + (int(235 * a),))
    px, py = lerp(hx, ex, 0.04 + 0.7 * draw_r + 0.02), lerp(hy, ey, 0.04 + 0.7 * draw_r + 0.02)
    d.line([(px + nx * 0.9, py + ny * 0.9), (px - nx * 0.9, py - ny * 0.9)], fill=(240, 245, 250, al), width=max(4, int(8 * s)))
    d.line([(px, py), (ex - ux * 60 * s, ey - uy * 60 * s)], fill=(240, 245, 250, al), width=max(4, int(8 * s)))
    pe = (ex - ux * 60 * s, ey - uy * 60 * s)
    d.line([(pe[0] + nx * 0.8, pe[1] + ny * 0.8), (pe[0] - nx * 0.8, pe[1] - ny * 0.8)], fill=(240, 245, 250, al), width=max(5, int(10 * s)))
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('ORTHO NUMBERS  #7', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Hot swollen knee: septic or not?', 'MontXB', 44, ICE, 880), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('50,000', 'MontBlack', 180, WHITE, 880)
    put(glow, k1, W / 2, 675, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 675, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('THE SYNOVIAL WBC NUMBER', 'MontBlack', 54, SKY, 880, track=1), W / 2, 830, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('one tap shifts the odds of a septic joint', 'MontXB', 40, WHITE), W / 2, 915 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); cx, cy, s = 540, 1330, 0.72
    put(main, scaled(knee(1.0, 1.0), s, 'h'), cx, cy + 20 * (1 - hq), a * hq)
    syringe(main, cx, cy + 20 * (1 - hq), s, a * eo(prog(t, 0.2, 0.6)), 1.0, 0.25 + 0.5 * eio(prog(t, 0.4, 2.4)))

def s2(t, main, glow, fx):                                     # history & exam can't settle it (Margaretten)
    a = win(t, 3.3, 9.6, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('THE EXAM WON\'T SETTLE IT', 'MontBlack', 60, WHITE, 880, track=2), XC, 262, a * p, 1, 10 * (1 - p))
    label(main, '14 studies  ·  6,242 patients  ·  653 septic', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    label(main, 'SENSITIVITY IN SEPTIC ARTHRITIS', XC, 420, a * eo(prog(t, 3.7, 4.2)), SUB, 30, 'MontXB', 3)
    rows = [(3.9, 85, 'Joint pain', SKY), (4.6, 78, 'History of swelling', SKY), (5.3, 57, 'Fever', WARN),
            (6.0, 27, 'Sweats', WARN), (6.7, 19, 'Rigors', WARN)]
    for i, (t0, v, name, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 540 + i * 190
        put(main, card(880, 160, outline=col + (130,)), XC, y, a * q)
        label(main, name, 125 + tl(name, 'MontBlack', 40, WHITE).width / 2, y - 34, a * q, WHITE, 40, 'MontBlack', 0)
        g = eio(prog(t, t0 + 0.15, t0 + 0.95))
        d = ImageDraw.Draw(main); x0, x1 = 125, 760
        d.rounded_rectangle((x0, y + 18, x1, y + 46), 14, fill=(60, 86, 120, int(200 * a * q)))
        xv = x0 + (x1 - x0) * v / 100 * g
        if xv > x0 + 28: d.rounded_rectangle((x0, y + 18, xv, y + 46), 14, fill=col + (int(240 * a * q),))
        put(main, tl(f'{round(v * g)}%', 'MontBlack', 58, col), 840, y + 4, a * q)
    q = eo(prog(t, 7.3, 7.8))
    label(main, 'Fever: missing in many septic joints', XC, 1500, a * q, WARN, 38, 'MontBlack', 0)
    label(main, 'Margaretten et al.  ·  JAMA 2007', XC, 1580, a * eo(prog(t, 7.4, 7.9)), ICE, 34, 'MontXB', 0)

CX, CY, S = 520, 830, 0.86
STEPS = [(9.6, 'HOT SWOLLEN KNEE', 'effusion distends the suprapatellar pouch'),
         (13.6, 'TAP IT: ARTHROCENTESIS', 'needle into the pouch, lateral side'),
         (17.6, 'COUNT THE CELLS', 'synovial WBC + % PMN, before Gram and culture')]
def s3(t, main, glow, fx):                                     # centrepiece: AP knee, effusion, aspiration
    a = win(t, 9.4, 21.6, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 9.5, 10.0))
    put(main, tl('TAP THE KNEE', 'MontBlack', 62, WHITE, track=3), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Right knee  ·  AP view', XC, 335, a * eo(prog(t, 9.7, 10.2)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 9.6, 10.5)); cy = CY + 24 * (1 - bp)
    r = eio(prog(t, 10.8, 12.4)); fl = eio(prog(t, 16.0, 17.4))
    put(main, scaled(knee(r, fl), S, f'k{round(r * 10)}_{round(fl * 4)}'), CX, cy, a * bp)
    P = lambda q_: dp(q_, CX, cy, S)
    la = a * eo(prog(t, 10.2, 10.6)) * (1 - eo(prog(t, 13.2, 13.6)))
    pointer(main, la, P((380, 120)), 820, 470, 'FEMUR', ICE, 32)
    pointer(main, la, P((420, 470)), 860, 760, 'PATELLA', ICE, 30)
    pointer(main, la, P((390, 800)), 820, 1200, 'TIBIA', ICE, 32)
    pointer(main, la, P((212, 760)), 160, 1210, 'FIBULA', ICE, 30)
    label(main, 'LATERAL', 165, 1290, la, SUB, 28, 'MontXB', 2)
    label(main, 'MEDIAL', 870, 1290, la, SUB, 28, 'MontXB', 2)
    ea = a * eo(prog(t, 12.0, 12.5)) * (1 - eo(prog(t, 13.2, 13.6)))
    if ea > 0:
        pointer(main, ea, P((190, 330)), 190, 500, 'EFFUSION', FLUID, 32)
        ImageDraw.Draw(glow).ellipse((P((200, 140))[0], P((200, 140))[1], P((580, 330))[0], P((580, 330))[1]),
                                     fill=FLUID + (int(40 * ea),))
    ins = eio(prog(t, 13.8, 14.9)); dr = eio(prog(t, 15.4, 17.2))
    sa = a * eo(prog(t, 13.6, 14.0))
    syringe(main, CX, cy, S, sa, ins, dr)
    na = a * eo(prog(t, 14.9, 15.3)) * (1 - eo(prog(t, 17.2, 17.6)))
    if na > 0:
        pointer(main, na, P((540, 260)), 800, 560, 'SUPRAPATELLAR', WHITE, 30)
        label(main, 'POUCH', 860, 600, na, WHITE, 30, 'MontXB', 2)
    ta = a * eo(prog(t, 17.6, 18.1))
    if ta > 0:
        put(main, card(420, 120, outline=FLUID + (180,)), 300, 1265, ta)
        label(main, 'WBC', 200, 1265, ta, SUB, 30, 'MontXB', 2)
        v = eio(prog(t, 18.0, 19.6))
        label(main, f'{int(round(62000 * v, -3)):,}', 360, 1265, ta, WHITE, 50, 'MontBlack', 0)
        put(main, card(330, 120, outline=FLUID + (180,)), 765, 1265, ta)
        label(main, 'PMN', 680, 1265, ta, SUB, 30, 'MontXB', 2)
        label(main, f'{round(92 * v)}%', 810, 1265, ta, WHITE, 50, 'MontBlack', 0)
        label(main, 'example sample  ·  WBC per µL', XC, 1345, ta * eo(prog(t, 18.4, 18.8)), SUB, 28, 'MontXB', 0)
    pa = a * eo(prog(t, 10.2, 10.6))
    if pa > 0:
        y = 1490; cur = 0
        for i, (t0, _, _) in enumerate(STEPS):
            if t >= t0: cur = i
        put(main, card(880, 200, outline=SKY + (160,)), XC, y, pa)
        t0, nm, sub = STEPS[cur]; q = eo(prog(t, t0, t0 + 0.35))
        put(main, pill(f'{cur + 1} / 3', 'MontBlack', 36, NAVY, SKY + (255,), padx=22, h=64), 170, y - 36, pa * q)
        im = fit(nm, 'MontBlack', 44, WHITE, 650); put(main, im, 270 + im.width / 2, y - 36, pa * q)
        im = fit(sub, 'MontXB', 32, SKY, 800); put(main, im, 120 + im.width / 2, y + 46, pa * q)
        if cur == 2: label(main, 'Margaretten et al.  ·  JAMA 2007', XC, 1640, pa * q, ICE, 32, 'MontXB', 0)

LRS = [(22.4, 'UNDER 25,000', 0.32, GOOD), (23.4, '25,000 OR MORE', 2.9, SKY), (24.4, 'OVER 50,000', 7.7, WARN),
       (25.4, 'OVER 100,000', 28.0, WARN)]
def s4(t, main, glow, fx):                                     # the WBC likelihood-ratio ladder
    a = win(t, 21.4, 30.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'READ THE COUNT', XC, 262, a * eo(prog(t, 21.5, 22.0)), WHITE, 62, 'MontBlack', 3)
    label(main, 'synovial WBC per µL  ·  likelihood ratio', XC, 340, a * eo(prog(t, 21.7, 22.2)), SKY, 36, 'MontXB', 0)
    # log scale 0.1 .. 100, LR 1 line
    xv = lambda v: 400 + 200 * math.log10(v)
    q = eo(prog(t, 22.0, 22.5))
    if q > 0:
        d = ImageDraw.Draw(main)
        d.line([(xv(1), 430), (xv(1), 1310)], fill=WHITE + (int(170 * a * q),), width=4)
        label(main, 'LR 1 = no change  ·  log scale', xv(1), 1350, a * q, SUB, 28, 'MontXB', 0)
        label(main, 'LOWER ODDS', 250, 1410, a * q, GOOD, 30, 'MontBlack', 2)
        label(main, 'HIGHER ODDS', 700, 1410, a * q, WARN, 30, 'MontBlack', 2)
    for i, (t0, name, lr, col) in enumerate(LRS):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 520 + i * 210
        nmi = tl(name, 'MontBlack', 40, WHITE, track=1); put(main, nmi, 440 + nmi.width / 2, y - 46, a * q)
        g = eio(prog(t, t0 + 0.15, t0 + 0.9))
        d = ImageDraw.Draw(main)
        xe = lerp(xv(1), xv(lr), g)
        lo, hi = sorted((xv(1), xe))
        if hi - lo > 4:
            d.rounded_rectangle((lo + 5, y + 8, hi + 5, y + 52), 16, fill=(8, 18, 32, int(150 * a * q)))
            d.rounded_rectangle((lo, y, hi, y + 44), 16, fill=col + (int(235 * a * q),))
            if hi - lo > 20: d.rounded_rectangle((lo + 6, y + 6, hi - 6, y + 16), 5, fill=(255, 255, 255, int(70 * a * q)))
            ImageDraw.Draw(glow).rounded_rectangle((lo, y, hi, y + 44), 16, fill=col + (int(70 * a * q),))
        txt = f'LR {lr:g}' if lr != 28.0 else 'LR 28'
        lab = tl(txt, 'MontBlack', 46, col)
        lx = (lo - 20 - lab.width / 2) if lr < 1 else hi + 20 + lab.width / 2
        put(main, lab, lx, y + 22, a * q * eo(prog(t, t0 + 0.6, t0 + 0.9)))
    label(main, 'Margaretten et al.  ·  JAMA 2007', XC, 1500, a * eo(prog(t, 26.4, 26.9)), ICE, 34, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # 50,000 and the % PMN
    a = win(t, 29.8, 37.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'THE CUT-OFF', XC, 262, a * eo(prog(t, 29.9, 30.4)), WHITE, 62, 'MontBlack', 3)
    label(main, 'synovial WBC over 50,000 per µL', XC, 340, a * eo(prog(t, 30.1, 30.6)), SKY, 36, 'MontXB', 0)
    pk = R.eback(prog(t, 30.4, 31.0))
    big = tl('LR 7.7', 'MontBlack', 150, WARN)
    put(glow, big, XC, 500, a * 0.3 * clamp(pk), clamp(pk, 0, 1.2)); put(main, big, XC, 500, a * clamp(pk), clamp(pk, 0, 1.2))
    label(main, 'markedly raises the odds of a septic joint', XC, 610, a * eo(prog(t, 30.9, 31.4)), WHITE, 36, 'MontXB', 0)
    label(main, 'SAME SAMPLE: % PMN', XC, 735, a * eo(prog(t, 31.6, 32.1)), SKY, 44, 'MontBlack', 3)
    rows = [(32.0, '90% OR MORE', 'LR 3.4', 'raises the likelihood', WARN),
            (33.0, 'UNDER 90%', 'LR 0.34', 'lowers the likelihood', GOOD)]
    for i, (t0, h_, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 900 + i * 280 + 20 * (1 - q)
        put(main, card(880, 240, outline=col + (170,)), XC, y, a * q)
        label(main, h_, XC, y - 66, a * q, col, 44, 'MontBlack', 3)
        im = fit(l1, 'MontBlack', 64, WHITE, 820); put(main, im, XC, y + 4, a * q)
        im = fit(l2, 'MontXB', 32, SUB, 820); put(main, im, XC, y + 70, a * q)
    q = eo(prog(t, 34.2, 34.7))
    label(main, 'Count + % PMN guide you before', XC, 1440, a * q, ICE, 34, 'MontXB', 0)
    label(main, 'Gram stain and culture are back', XC, 1490, a * q, ICE, 34, 'MontXB', 0)
    label(main, 'Margaretten et al.  ·  JAMA 2007', XC, 1590, a * eo(prog(t, 34.4, 34.9)), ICE, 34, 'MontXB', 0)

TIPS = [('Under 25,000 is not a rule-out', 'LR 0.32: IT LOWERS THE ODDS, NOT TO ZERO', 'Margaretten et al. JAMA 2007'),
        ('Know who is at risk', 'DIABETES, RA, PROSTHESIS, SKIN INFECTION', 'Margaretten et al. JAMA 2007'),
        ('Knee replacement? Far lower cut-off', 'INFECTED TKA: WBC OVER 1,100, PMN OVER 64%', 'Ghanem et al. JBJS Am 2008  ·  429 knees')]
def s6(t, main, glow, fx):
    a = win(t, 36.8, 43.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS & TRICKS', XC, 262, a * eo(prog(t, 36.9, 37.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 37.4 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
        put(main, card(880, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 40, WHITE, 690); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 32, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s7(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 43.0, 49.5, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 360, a * eo(prog(t, 43.1, 43.6)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 43.3, 43.9))
    big = fit('WBC OVER 50,000', 'MontBlack', 104, WHITE, 880)
    put(glow, big, XC, 490, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 490, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '= LR 7.7  ·  over 100,000 = LR 28', XC, 600, a * eo(prog(t, 43.7, 44.2)), WARN, 42, 'MontBlack', 0)
    label(main, 'under 25,000 = LR 0.32, not a rule-out', XC, 670, a * eo(prog(t, 44.0, 44.5)), GOOD, 36, 'MontXB', 0)
    q = eo(prog(t, 44.4, 45.0))
    refs = ['Margaretten ME, Kohlwes J, Moore D, Bent S.', 'JAMA 2007;297(13):1478–88',
            'Ghanem E, Parvizi J, et al.', 'J Bone Joint Surg Am 2008;90(8):1637–43']
    put(main, card(880, 330, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 930, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 32, ICE if i % 2 == 0 else SUB, 820), XC, 820 + i * 72, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1260, a * eo(prog(t, 45.0, 45.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
