"""Orthojets EP07 - DON'T MISS #1 - Suspected scaphoid fracture: the 3 clinical signs and the next test.
Sources (verified on PubMed abstracts / metadata, 2026-10-25):
  Parvizi J, Wayman J, Kelly P, Moran CG. J Hand Surg Br 1998;23(3):324-7. PMID 9665518, doi 10.1016/s0266-7681(98)80050-8
    215 consecutive patients with suspected scaphoid fracture. Anatomical snuffbox tenderness, scaphoid tubercle
    tenderness and pain on longitudinal compression of the thumb were each 100% sensitive, specificity 9%, 30% and
    48%. Combined within the first 24 h: sensitivity 100%, specificity 74%. Used alone they are inadequate.
  Yin ZG, Zhang JB, Kan SL, Wang XG. J Bone Joint Surg Br 2012;94(8):1077-85. PMID 22844049, doi 10.1302/0301-620X.94B8.28998
    Meta-analysis of 30 studies + latent class analysis. Sensitivity / specificity: MRI 97.7% / 99.8%,
    CT 85.2% / 99.5%, bone scintigraphy 97.8% / 93.5%, follow-up radiographs 91.1% / 99.8%. MRI is the most accurate.
  Backer HC, Wu CH, Strauch RJ. J Wrist Surg 2020;9(1):81-89. PMID 32025360, doi 10.1055/s-0039-1693147
    Scaphoid = about 15% of acute wrist fractures; clinical exam + plain X-rays may miss up to 16% of fractures.
  Mallee WH et al. Cochrane Database Syst Rev 2015;(6):CD010023. PMID 26045406, doi 10.1002/14651858.CD010023.pub2
    Median prevalence of a true fracture among clinically suspected fractures = 20%; early diagnosis reduces the
    risk of non-union and minimises unnecessary cast immobilisation.
  Diagram: right wrist, PA view (radial / thumb side on the viewer's left). Proximal row scaphoid-lunate-triquetrum
  (+ pisiform), distal row trapezium-trapezoid-capitate-hamate; scaphoid oblique from the proximal pole (radius
  scaphoid fossa, next to the lunate) to the distal pole / tubercle (next to trapezium and trapezoid); waist fracture.
  The anatomical snuffbox lies over the scaphoid waist (dorsoradial), the tubercle is palpated volarly at the distal pole.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(42.6, [(7.4, 1.6), (10.6, 1.8), (13.4, 1.6), (16.2, 1.6), (20.2, 2.0), (25.4, 1.8),
                     (31.6, 2.0), (38.2, 1.6), (41.8, 2.0)])

# ------------------------------------------------------------------ diagram (local canvas DW x DH)
DW, DH = 760, 1060
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

def tube(c0, c1, w0, w1, wm, head=1.0):
    """long bone (metacarpal / phalanx) as a polygon: base c0 width w0, head c1 width w1, waist wm"""
    x0, y0 = c0; x1, y1 = c1; L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L; nx, ny = -uy, ux
    prof = [(0.0, w0), (0.12, w0 * 0.92), (0.35, wm), (0.7, wm * 1.02), (0.9, w1 * 0.95), (1.0, w1 * 0.7)]
    left = [(x0 + ux * L * s + nx * w / 2, y0 + uy * L * s + ny * w / 2) for s, w in prof]
    right = [(x0 + ux * L * s - nx * w / 2, y0 + uy * L * s - ny * w / 2) for s, w in prof]
    tip = (x1 + ux * w1 * 0.18 * head, y1 + uy * w1 * 0.18 * head)
    return left + [tip] + right[::-1]

RADIUS = [(178, 652), (210, 672), (262, 690), (330, 702), (400, 710), (456, 716), (472, 744), (462, 792), (446, 862),
          (440, 1060), (290, 1060), (278, 900), (240, 800), (198, 724)]
ULNA = [(490, 748), (512, 730), (560, 724), (590, 712), (604, 682), (622, 704), (628, 760), (612, 830), (598, 1060),
        (502, 1060), (494, 860), (484, 792)]
SCAPHOID = [(362, 655), (352, 684), (318, 694), (278, 678), (245, 640), (216, 596), (198, 552), (204, 523), (228, 509),
            (256, 518), (274, 544), (290, 577), (318, 610), (346, 630)]
LUNATE = [(375, 625), (400, 602), (440, 598), (470, 615), (468, 660), (452, 702), (410, 706), (374, 692), (364, 662)]
TRIQ = [(482, 612), (520, 588), (560, 592), (578, 626), (552, 664), (506, 680), (480, 656)]
TRAPEZIUM = [(150, 455), (180, 425), (228, 428), (250, 462), (238, 494), (200, 503), (162, 490)]
TRAPEZOID = [(258, 455), (280, 428), (318, 430), (330, 465), (318, 500), (276, 504), (260, 485)]
CAPITATE = [(337, 440), (352, 405), (400, 398), (430, 420), (432, 480), (428, 540), (410, 586), (376, 594), (346, 574),
            (332, 530), (340, 490)]
HAMATE = [(442, 425), (482, 415), (530, 440), (560, 490), (555, 558), (520, 584), (476, 588), (442, 560), (438, 490)]
MC = [tube((178, 418), (92, 222), 50, 44, 30), tube((76, 196), (34, 66), 40, 32, 26),            # 1st MC + thumb P1
      tube((290, 420), (268, 60), 50, 44, 30), tube((386, 396), (392, 30), 54, 46, 32),
      tube((488, 414), (504, 70), 48, 42, 28), tube((560, 444), (618, 136), 46, 40, 27)]
FRACT = [(232, 638), (252, 616), (262, 607), (274, 590), (294, 566)]                               # scaphoid waist
SNUFF = (262, 602)            # anatomical snuffbox overlies the scaphoid waist
TUBER = (214, 528)            # scaphoid tubercle (distal pole, palpated volarly)
THUMB_AX = ((40, 70), (170, 410))   # thumb axis for longitudinal compression

def xray(mask, seed=5, dim=1.0):
    m = mask
    M = np.asarray(m, np.float32) / 255
    B1 = np.asarray(m.filter(ImageFilter.GaussianBlur(7)), np.float32) / 255
    rim = np.clip((1 - B1) * 3.0, 0, 1)
    rng = np.random.default_rng(seed)
    n1 = np.asarray(Image.fromarray((rng.random((m.height // 4, m.width // 4)) * 255).astype(np.uint8)).resize(m.size, Image.BICUBIC), np.float32) / 255
    dens = (0.34 + 0.58 * rim) * (0.9 + 0.2 * n1) * dim
    inner = np.clip(M * dens, 0, 1.2)
    img = Image.fromarray(np.clip(np.stack([inner * c for c in XRAY], -1), 0, 255).astype(np.uint8)).convert('RGBA')
    img.putalpha(Image.fromarray(np.clip(inner * 255 * 1.05, 0, 255).astype(np.uint8)))
    mb = m.point(lambda v: 255 if v > 128 else 0)
    edge = ImageChops.subtract(mb, mb.filter(ImageFilter.MinFilter(5)))
    e = Image.new('RGBA', m.size, ICE + (0,)); e.putalpha(edge.point(lambda v: int(190 * dim) if v > 40 else 0))
    img.alpha_composite(e)
    return img

def poly_mask(pts, smooth=True):
    SS = 3
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    pp = catmull(pts, 8) if smooth else pts
    d.polygon([(x * SS, y * SS) for x, y in pp], fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def build():
    parts = [('radius', RADIUS, True), ('ulna', ULNA, True), ('lunate', LUNATE, True), ('triq', TRIQ, True),
             ('trapezium', TRAPEZIUM, True), ('trapezoid', TRAPEZOID, True), ('capitate', CAPITATE, True),
             ('hamate', HAMATE, True)] + [(f'mc{i}', p, False) for i, p in enumerate(MC)]
    masks = {k: poly_mask(p, s) for k, p, s in parts}
    masks['mc'] = masks.pop('mc0')
    for i in range(1, len(MC)): masks['mc'] = ImageChops.lighter(masks['mc'], masks.pop(f'mc{i}'))
    sc = poly_mask(SCAPHOID)
    union = sc
    for m in masks.values(): union = ImageChops.lighter(union, m)
    # soft tissue silhouette of the hand / wrist
    st = union.filter(ImageFilter.MaxFilter(41)).filter(ImageFilter.GaussianBlur(22))
    soft = Image.new('RGBA', (DW, DH), (110, 150, 190, 0)); soft.putalpha(st.point(lambda v: int(v * 0.22)))
    bones = Image.new('RGBA', (DW, DH))
    for i, (k, m) in enumerate(masks.items()):
        bones.alpha_composite(xray(m, seed=3 + i))
    # pisiform (overlaps the triquetrum, fainter) and hook of hamate ring
    pm = Image.new('L', (DW, DH)); ImageDraw.Draw(pm).ellipse((516, 586, 580, 646), fill=255)
    bones.alpha_composite(xray(pm, seed=40, dim=0.6))
    d = ImageDraw.Draw(bones)
    d.ellipse((488, 482, 520, 514), outline=(235, 245, 255, 150), width=4)
    scaph = xray(sc, seed=77)
    # fade the cut ends (finger tips at the top, forearm shaft at the bottom)
    for im in (bones, soft):
        yy = np.arange(DH)[:, None] * np.ones((1, DW))
        f = np.clip(yy / 130.0, 0, 1) * np.clip((DH - yy) / 160.0, 0, 1)
        im.putalpha(Image.fromarray((np.asarray(im.getchannel('A'), np.float32) * f).astype(np.uint8)))
    return soft, bones, scaph, sc

SOFT, BONES, SCAPH, SCMASK = build()
SCGLOW = Image.new('RGBA', (DW, DH), SKY + (0,)); SCGLOW.putalpha(SCMASK.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * 0.85)))

_dc = {}
def diagram(s, build_p, hl):
    key = (round(s, 3), round(build_p, 2), round(hl, 2))
    if key in _dc: return _dc[key]
    im = Image.new('RGBA', (DW, DH))
    im.alpha_composite(SOFT)
    im.alpha_composite(BONES); im.alpha_composite(SCAPH)
    if hl > 0:
        g = SCGLOW.copy(); g.putalpha(g.getchannel('A').point(lambda v: int(v * hl * 0.55)))
        im.alpha_composite(g)
    if build_p < 1:
        im.putalpha(im.getchannel('A').point(lambda v: int(v * build_p)))
    out = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS) if abs(s - 1) > 0.001 else im
    if len(_dc) > 60: _dc.clear()
    _dc[key] = out
    return out

CX, CY, S = 412, 905, 0.9
def dp(q, cx=CX, cy=CY, s=S):
    return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def press(main, glow, p, t, a, col):
    """pulsing press marker (examiner's fingertip) at screen point p"""
    if a <= 0: return
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    for k in range(2):
        ph = ((t * 1.1 + k * 0.5) % 1.0)
        r = 14 + 46 * ph
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), outline=col + (int(230 * a * (1 - ph)),), width=4)
    d.ellipse((p[0] - 13, p[1] - 13, p[0] + 13, p[1] + 13), fill=col + (int(255 * a),))
    g.ellipse((p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30), fill=col + (int(150 * a),))

def badge(main, a, p, txt, dx, dy):
    if a <= 0: return
    put(main, pill(txt, 'MontBlack', 34, NAVY, SKY + (255,), padx=18, h=56), p[0] + dx, p[1] + dy, a)

XC = 510   # centre of the safe zone x 60-960

def stat_chip(main, a, y, num, title, sens, spec, specol=WARN):
    if a <= 0: return
    put(main, card(860, 200), XC, y, a)
    nm = tl(num, 'MontBlack', 60, SKY); put(main, nm, 128, y - 36, a)
    lab = tl(title, 'MontBlack', 40, WHITE); put(main, lab, 180 + lab.width / 2 - 20, y - 38, a)
    s1_ = tl('Sensitivity ' + sens, 'MontXB', 34, GOOD); put(main, s1_, 180 + s1_.width / 2 - 20, y + 40, a)
    s2_ = tl('Specificity ' + spec, 'MontXB', 34, specol); put(main, s2_, 905 - s2_.width / 2 + 20, y + 40, a)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #1", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 470, a * eo(prog(t, 0.0, 0.3)))
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('UP TO 16%', 'MontBlack', 140, WHITE)
    put(glow, k1, W / 2, 680, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 680, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.3, 0.8))
    put(main, tl('of scaphoid fractures', 'MontXB', 56, SKY), W / 2, 860 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('missed on plain X-rays', 'MontXB', 56, WHITE), W / 2, 945 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, pill('3 SIGNS + 1 TEST CATCH THEM', 'MontBlack', 36, NAVY, GOOD + (255,), h=74), W / 2, 1120 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # centrepiece: PA wrist, scaphoid, fracture, 3 signs
    a = win(t, 3.3, 21.4, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('SUSPECTED SCAPHOID FRACTURE', 'MontBlack', 44, WHITE, track=1), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Right wrist  ·  PA view', XC, 330, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 3.5, 4.6))
    hl = eo(prog(t, 4.8, 5.4)) * (0.75 + 0.25 * math.sin(t * 3))
    put(main, diagram(S, 1.0, round(hl, 2)), CX, CY + 24 * (1 - bp), a * bp)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    # labels (right column)
    pointer(main, a * eo(prog(t, 4.9, 5.4)) * (1 - eo(prog(t, 7.9, 8.3))), dp((318, 640)), 830, 560, 'SCAPHOID', SKY, 31)
    pointer(main, a * eo(prog(t, 5.2, 5.7)) * (1 - eo(prog(t, 7.9, 8.3))), dp((425, 655)), 830, 680, 'LUNATE')
    pointer(main, a * eo(prog(t, 5.4, 5.9)) * (1 - eo(prog(t, 7.9, 8.3))), dp((360, 820)), 830, 800, 'RADIUS')
    pointer(main, a * eo(prog(t, 5.6, 6.1)) * (1 - eo(prog(t, 7.9, 8.3))), dp((385, 500)), 830, 440, 'CAPITATE')
    # phase 1: first X-ray looks normal, then the hidden waist fracture
    fq = eo(prog(t, 6.3, 7.1))
    if fq > 0:
        pts = [dp(q_) for q_ in FRACT]
        n = max(2, int(len(pts) * clamp(fq * 1.3)))
        pul = 0.6 + 0.4 * math.sin(t * 6)
        d.line(pts[:n], fill=WARN + (int(255 * a),), width=8, joint='curve')
        g.line(pts[:n], fill=WARN + (int(140 * a * pul),), width=16)
    nq = a * win(t, 4.8, 6.4, 0.3, 0.3)
    put(main, pill('X-RAY: NORMAL?', 'MontBlack', 32, NAVY, GOOD + (255,), h=62), XC, 1490, nq)
    wq = a * win(t, 6.4, 8.2, 0.3, 0.35)
    label(main, 'Waist fracture: can hide on day 1', XC, 1490, wq, WARN, 40, 'MontBlack', 0)
    label(main, 'Up to 16% missed on plain X-rays', XC, 1555, wq, ICE, 34, 'MontXB', 0)
    # phase 2: the three clinical signs (Parvizi 1998)
    hq = a * eo(prog(t, 8.2, 8.7))
    label(main, 'THE 3 CLINICAL SIGNS', XC, 1395, hq, SKY, 40, 'MontBlack', 4)
    # sign 1: snuffbox (over the waist)
    s1a = a * eo(prog(t, 8.5, 9.0))
    press(main, glow, dp(SNUFF), t, s1a * (0.55 if t > 11.2 else 1.0), SKY)
    badge(main, s1a, dp(SNUFF), '1', 78, 0)
    stat_chip(main, a * win(t, 8.7, 11.3, 0.4, 0.3), 1540, '1', 'Snuffbox tenderness', '100%', '9%')
    # sign 2: scaphoid tubercle (distal pole, volar)
    s2a = a * eo(prog(t, 11.3, 11.8))
    press(main, glow, dp(TUBER), t + 0.3, s2a * (0.55 if t > 14.0 else 1.0), SKY)
    badge(main, s2a, dp(TUBER), '2', -60, -40)
    stat_chip(main, a * win(t, 11.5, 14.1, 0.4, 0.3), 1540, '2', 'Scaphoid tubercle', '100%', '30%')
    # sign 3: longitudinal compression of the thumb
    s3a = a * eo(prog(t, 14.1, 14.6))
    if s3a > 0:
        (x0, y0), (x1, y1) = THUMB_AX
        push = 0.5 + 0.5 * math.sin(t * 5)
        L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
        st = dp((x0 + ux * (20 + 18 * push), y0 + uy * (20 + 18 * push)))
        en = dp((x0 + ux * (L - 90), y0 + uy * (L - 90)))
        col = SKY + (int(255 * s3a * (0.6 if t > 16.9 else 1.0)),)
        d.line([st, en], fill=col, width=9)
        hx, hy = en; bx, by = hx - ux * 34, hy - uy * 34
        d.polygon([(hx + ux * 8, hy + uy * 8), (bx - uy * 22, by + ux * 22), (bx + uy * 22, by - ux * 22)], fill=col)
        g.line([st, en], fill=SKY + (int(120 * s3a),), width=22)
        badge(main, s3a, st, '3', 30, -60)
    stat_chip(main, a * win(t, 14.3, 17.0, 0.4, 0.3), 1540, '3', 'Thumb compression', '100%', '48%')
    # phase 3: combined within 24 h
    cq = a * eo(prog(t, 17.0, 17.5))
    if cq > 0:
        y = 1540
        put(main, card(860, 200, fill=(24, 60, 60, 190), outline=GOOD + (200,)), XC, y, cq)
        label(main, 'ALL 3 COMBINED  ·  WITHIN 24 H', XC, y - 62, cq, WHITE, 38, 'MontBlack', 1)
        p = R.eback(prog(t, 17.4, 17.9))
        put(main, tl('Sens 100%  ·  Spec 74%', 'MontBlack', 50, GOOD), XC, y + 8, cq * clamp(p), clamp(p, 0, 1.15))
        label(main, 'Parvizi et al  ·  J Hand Surg Br 1998', XC, y + 70, cq * eo(prog(t, 17.8, 18.2)), SUB, 27, 'MontXB', 0)

def bar(main, glow, x, ybase, hmax, frac, col, a, wbar=150):
    if a <= 0 or frac <= 0: return
    h = hmax * frac
    put(main, card(wbar, max(40, int(h)), fill=col + (220,), outline=col + (255,), r=18), x, ybase - h / 2, a)
    g = ImageDraw.Draw(glow); g.rounded_rectangle((x - wbar / 2, ybase - h, x + wbar / 2, ybase), 18, fill=col + (int(60 * a),))

def s3(t, main, glow, fx):                                     # specificity: alone vs combined
    a = win(t, 21.2, 27.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'SPECIFICITY', XC, 262, a * eo(prog(t, 21.3, 21.8)), WHITE, 64, 'MontBlack', 4)
    label(main, 'every sign alone: sensitivity 100%', XC, 345, a * eo(prog(t, 21.5, 22.0)), SKY, 36, 'MontXB', 0)
    yb, hmax = 1180, 640
    d = ImageDraw.Draw(main)
    d.line([(75, yb), (945, yb)], fill=ICE + (int(160 * a),), width=4)
    items = [(180, 9, WARN, 'SNUFFBOX', 'sign 1', 22.1), (390, 30, WARN, 'TUBERCLE', 'sign 2', 22.6), (600, 48, WARN, 'THUMB', 'sign 3', 23.1),
             (830, 74, GOOD, 'ALL 3', 'within 24 h', 24.0)]
    for x, v, col, l1, l2, t0 in items:
        p = eio(prog(t, t0, t0 + 0.9))
        bar(main, glow, x, yb, hmax * 0.92, p * v / 74, col, a)
        hy = yb - hmax * 0.92 * (p * v / 74)
        if p > 0:
            put(main, tl(f'{int(round(v * p))}%', 'MontBlack', 70 if col == WARN else 84, col), x, hy - 60, a * clamp(p * 3))
        q = a * eo(prog(t, t0, t0 + 0.4))
        label(main, l1, x, yb + 55, q, WHITE, 31, 'MontBlack', 0)
        label(main, l2, x, yb + 100, q, SUB, 30, 'MontXB', 0)
    q = eo(prog(t, 24.8, 25.3))
    label(main, 'Alone, each sign over-calls the fracture', XC, 1400, a * q, ICE, 36, 'MontXB', 0)
    label(main, 'Combine them to cut false positives', XC, 1460, a * eo(prog(t, 25.0, 25.5)), GOOD, 36, 'MontXB', 0)
    label(main, 'Parvizi et al  ·  J Hand Surg Br 1998  ·  215 patients', XC, 1560, a * eo(prog(t, 25.2, 25.7)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # next test: MRI / CT / bone scan
    a = win(t, 27.4, 33.8, 0.4, 0.45)
    if a <= 0: return
    label(main, 'X-RAY NORMAL, SIGNS POSITIVE?', XC, 262, a * eo(prog(t, 27.5, 28.0)), WHITE, 42, 'MontBlack', 1)
    label(main, 'accuracy of the next test', XC, 335, a * eo(prog(t, 27.7, 28.2)), SKY, 36, 'MontXB', 0)
    rows = [('MRI', 97.7, 99.8, True, 28.2), ('CT', 85.2, 99.5, False, 28.9), ('BONE SCAN', 97.8, 93.5, False, 29.6)]
    for i, (nm, se, sp, best, t0) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 290 + 30 * (1 - q)
        put(main, card(860, 250, fill=(24, 60, 60, 190) if best else (30, 54, 84, 170),
                       outline=(GOOD if best else SKY) + (200 if best else 120,)), XC, y, a * q)
        lab = tl(nm, 'MontBlack', 50, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 62, a * q)
        if best:
            pl = pill('MOST ACCURATE', 'MontBlack', 27, NAVY, GOOD + (255,), h=52); put(main, pl, 915 - pl.width / 2, y - 62, a * q)
        p = eio(prog(t, t0 + 0.2, t0 + 1.1))
        for j, (lname, v) in enumerate([('Sensitivity', se), ('Specificity', sp)]):
            x0 = 115 + j * 405; yy = y + 45
            label(main, lname, x0 + 95, yy - 30, a * q, SUB, 28, 'MontXB', 0)
            col = GOOD if v > 97 else (ICE if v > 90 else WARN)
            vt = tl(f'{v * p:.1f}%', 'MontBlack', 50, col); put(main, vt, x0 + 240 + vt.width / 2 - 40, yy - 18, a * q)
            dd = ImageDraw.Draw(main)
            dd.rounded_rectangle((x0, yy + 18, x0 + 355, yy + 34), 8, fill=(60, 90, 125, int(200 * a * q)))
            dd.rounded_rectangle((x0, yy + 18, x0 + max(16, 355 * v / 100 * p), yy + 34), 8, fill=col + (int(255 * a * q),))
    label(main, 'Yin et al  ·  JBJS Br 2012  ·  meta-analysis of 30 studies', XC, 1440, a * eo(prog(t, 30.6, 31.1)), SUB, 28, 'MontXB', 0)
    label(main, 'CT misses about 1 in 7', XC, 1520, a * eo(prog(t, 30.9, 31.4)), WARN, 38, 'MontBlack', 0)

TIPS = [('Test all 3 signs early', 'COMBINED IN 24 H: SPECIFICITY 74%', 'Parvizi 1998, 215 patients'),
        ('Normal X-ray is not the end', 'UP TO 16% ARE MISSED ON PLAIN FILMS', 'Bäcker, systematic review 2020'),
        ('Only 1 in 5 is a real fracture', 'EARLY DIAGNOSIS SPARES NEEDLESS CASTS', 'Mallee, Cochrane 2015: median 20%')]
def s5(t, main, glow, fx):
    a = win(t, 33.6, 40.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 33.7, 34.2)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 34.3 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 42, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

def s6(t, main, glow, fx):
    a = win(t, 40.0, 42.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 330, a * eo(prog(t, 40.1, 40.5)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 40.2, 40.7))
    big = tl('3 SIGNS', 'MontBlack', 150, WHITE)
    put(glow, big, XC, 490, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 490, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Snuffbox  ·  Tubercle  ·  Thumb', XC, 625, a * eo(prog(t, 40.4, 40.8)), SKY, 42, 'MontXB', 0)
    label(main, 'X-ray normal? MRI is the most accurate', XC, 700, a * eo(prog(t, 40.6, 41.0)), GOOD, 36, 'MontXB', 0)
    q = eo(prog(t, 40.8, 41.3))
    put(main, card(860, 440, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1020, a * q)
    for i, (au, ref) in enumerate([('Parvizi J, Wayman J, Kelly P, Moran CG.', 'J Hand Surg Br. 1998;23(3):324–7'),
                                   ('Yin ZG, Zhang JB, Kan SL, Wang XG.', 'J Bone Joint Surg Br. 2012;94(8):1077–85'),
                                   ('Bäcker HC, Wu CH, Strauch RJ.', 'J Wrist Surg. 2020;9(1):81–9'),
                                   ('Mallee WH et al.', 'Cochrane Database Syst Rev. 2015;CD010023')]):
        y = 865 + i * 102
        label(main, au, XC, y, a * q, ICE, 29, 'MontXB', 0)
        label(main, ref, XC, y + 40, a * q, SUB, 29, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1340, a * eo(prog(t, 41.2, 41.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
