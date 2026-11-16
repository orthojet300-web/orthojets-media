"""Orthojets EP30 - CLINICAL RULES #6 - Amsterdam Wrist Rules: who needs a wrist X-ray?
Sources (verified on PubMed abstracts / metadata, 2026-11-16; PubMed is the only verification source):
  Walenkamp MMJ, Bentohami A, Slaar A, et al. BMC Musculoskelet Disord 2015;16:389. PMID 26682537,
    doi 10.1186/s12891-015-0829-2. Multicentre prospective derivation + external validation, adults with acute wrist trauma,
    5 Dutch EDs. Only 39% of patients with wrist trauma have a fracture, yet most are routinely X-rayed. 882 patients
    (487 derivation, 395 external validation). Outcome: fracture of the distal radius, distal ulna or carpal bones on X-ray.
    8 predictors: age; sex; swelling of the wrist; swelling of the anatomical snuffbox; visible deformation; distal radius
    tender to palpation; pain on radial deviation; painful axial compression of the thumb. External validation: AUC 0.81,
    sensitivity 98% (95-99), specificity 21% (15-28), negative predictive value 90% (81-99).
  Walenkamp MMJ, Rosenwasser MP, Goslings JC, Schep NWL. Eur J Trauma Emerg Surg 2016;42(2):185-90. PMID 26038045,
    doi 10.1007/s00068-015-0527-7. 924 consecutive adults with wrist trauma, 5 EDs: 99.6% were referred for radiography;
    44% distal radius fracture.
  Mulders MAM, Walenkamp MMJ, Sosef NL, et al. Eur J Trauma Emerg Surg 2020;46(3):573-582. PMID 31541258,
    doi 10.1007/s00068-019-01194-2. Implementation, before-and-after, 4 hospitals, 402 patients: absolute reduction in wrist
    radiographs 15%; one clinically irrelevant fracture missed; non-fracture patients without radiography spent 34 min less
    in the ED; physicians adhered to the AWR in 36%; 87% of patients without radiography were satisfied.
  Slaar A, Walenkamp MMJ, Bentohami A, et al. Pediatr Radiol 2016;46(1):50-60. PMID 26298555, doi 10.1007/s00247-015-3436-3.
    Amsterdam Pediatric Wrist Rules: 787 children, 6 variables, sensitivity 95.9%, specificity 37.3%; validation: 7/170 (4.1%) fractures missed; 22% absolute
    reduction of radiographs.
  Mallee WH, Walenkamp MMJ, Mulders MAM, Goslings JC, Schep NWL. Arch Orthop Trauma Surg 2020;140(4):575-581. PMID 32125528,
    doi 10.1007/s00402-020-03383-w. Scaphoid rule: 893 acute wrist injuries, 68 (7.6%) scaphoid fractures; predictors sex,
    snuffbox swelling, snuffbox tenderness, painful ulnar deviation, painful axial thumb compression; sensitivity 97%,
    specificity 20%.
  Diagram: right wrist, PA view (radial / thumb side on the viewer's left), same bone geometry as EP07 (verified there):
  distal radius + ulna, proximal row scaphoid-lunate-triquetrum (+ pisiform), distal row trapezium-trapezoid-capitate-hamate,
  metacarpals. Snuffbox over the scaphoid waist (dorsoradial). Radial deviation = hand tilts toward the thumb side.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(49.0, [(7.8, 1.6), (15.0, 1.0), (20.0, 1.0), (23.6, 1.6), (29.4, 1.8), (35.6, 1.6), (41.6, 1.8),
                     (46.4, 1.4)])

# ------------------------------------------------------------------ diagram: right wrist PA (canvas DW x DH)
DW, DH = 760, 1060
XRAY = (170, 210, 242)

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
MC = [tube((178, 418), (92, 222), 50, 44, 30), tube((76, 196), (34, 66), 40, 32, 26),
      tube((290, 420), (268, 60), 50, 44, 30), tube((386, 396), (392, 30), 54, 46, 32),
      tube((488, 414), (504, 70), 48, 42, 28), tube((560, 444), (618, 136), 46, 40, 27)]
SNUFF = (262, 602)                 # anatomical snuffbox over the scaphoid waist
RAD_T = (330, 770)                 # distal radius, palpation point
THUMB_AX = ((40, 70), (170, 410))  # thumb axis for axial compression
PIVOT = (390, 560)                 # midcarpal centre for the deviation arc

SS = 2
def poly_mask(pts, smooth=True):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    pp = catmull(pts, 8) if smooth else pts
    d.polygon([(x * SS, y * SS) for x, y in pp], fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

_yy = np.arange(DH, dtype=np.float32)[:, None] * np.ones((1, DW), np.float32)
_FADE = np.clip((_yy - 20) / 130.0, 0, 1) * np.clip((DH - 10 - _yy) / 170.0, 0, 1)
def faded(m):
    return Image.fromarray((np.asarray(m, np.float32) * _FADE).astype(np.uint8))

def build():
    polys = [RADIUS, ULNA, SCAPHOID, LUNATE, TRIQ, TRAPEZIUM, TRAPEZOID, CAPITATE, HAMATE]
    masks = [faded(poly_mask(p)) for p in polys] + [faded(poly_mask(p, False)) for p in MC]
    pm = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(pm).ellipse((516 * SS, 586 * SS, 580 * SS, 646 * SS), fill=255)
    pis = pm.resize((DW, DH), Image.LANCZOS)
    union = masks[0]
    for m in masks[1:]: union = ImageChops.lighter(union, m)
    soft_m = union.filter(ImageFilter.MaxFilter(45)).filter(ImageFilter.GaussianBlur(20))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)))
    soft = volume(soft_m, (64, 98, 140), alpha=0.55, r=18, relief=0.6)
    bones = Image.new('RGBA', (DW, DH))
    for i, m in enumerate(masks):
        bones.alpha_composite(bone_xray(m, XRAY, alpha=0.96, seed=3 + i))
    bones.alpha_composite(bone_xray(pis, XRAY, alpha=0.45, seed=60))
    d = ImageDraw.Draw(bones)
    d.ellipse((490, 484, 518, 512), outline=(235, 245, 255, 140), width=3)      # hook of hamate ring
    lay = Image.new('RGBA', (DW, DH)); lay.alpha_composite(soft); lay.alpha_composite(bones)
    pad = 40
    big = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); big.alpha_composite(lay, (pad, pad))
    big = drop_shadow(big, dx=10, dy=16, blur=14, opacity=0.55)
    return big.crop((pad, pad, pad + DW, pad + DH)), soft_m

WRIST, SOFTM = build()

_dc = {}
def diagram(s):
    k = round(s, 3)
    if k not in _dc:
        _dc[k] = WRIST.resize((int(DW * s), int(DH * s)), Image.LANCZOS) if abs(s - 1) > 1e-3 else WRIST
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

def press(main, glow, p, t, a, col, r0=13):
    if a <= 0: return
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    for k in range(2):
        ph = (t * 1.1 + k * 0.5) % 1.0; r = r0 + 44 * ph
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), outline=col + (int(230 * a * (1 - ph)),), width=4)
    d.ellipse((p[0] - r0, p[1] - r0, p[0] + r0, p[1] + r0), fill=col + (int(255 * a),))
    g.ellipse((p[0] - 30, p[1] - 30, p[0] + 30, p[1] + 30), fill=col + (int(150 * a),))

def swell(main, glow, c, rx, ry, t, a, col):
    """swelling: a soft translucent puff that breathes, with a dashed rim"""
    if a <= 0: return
    k = 1 + 0.06 * math.sin(t * 4)
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    d.ellipse((c[0] - rx * k, c[1] - ry * k, c[0] + rx * k, c[1] + ry * k), fill=col + (int(70 * a),))
    n = 28
    for i in range(0, n, 2):
        a0, a1 = 360 * i / n, 360 * (i + 1) / n
        d.arc((c[0] - rx * k, c[1] - ry * k, c[0] + rx * k, c[1] + ry * k), a0, a1, fill=col + (int(240 * a),), width=5)
    main.alpha_composite(lay)
    ImageDraw.Draw(glow).ellipse((c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry), fill=col + (int(60 * a),))

def thick_arrow(d, g, st, en, col, a, w=9):
    ux, uy = en[0] - st[0], en[1] - st[1]; L = math.hypot(ux, uy); ux, uy = ux / L, uy / L
    hx, hy = en; bx, by = hx - ux * 34, hy - uy * 34
    d.line([st, (bx, by)], fill=col + (int(255 * a),), width=w)
    d.polygon([(hx + ux * 6, hy + uy * 6), (bx - uy * 22, by + ux * 22), (bx + uy * 22, by - ux * 22)], fill=col + (int(255 * a),))
    g.line([st, en], fill=col + (int(110 * a),), width=24)

def arc_arrow(main, glow, c, r, a0, a1, col, a, w=9):
    """curved arrow on a circle around c from angle a0 to a1 (degrees, screen coords, 0 = right, 90 = down)"""
    if a <= 0: return
    lay = aa_image(W, H); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    n = 24; pts = []
    for i in range(n + 1):
        th = math.radians(a0 + (a1 - a0) * i / n); pts.append((c[0] + r * math.cos(th), c[1] + r * math.sin(th)))
    d.line(pts[:-2], fill=col + (int(255 * a),), width=w, joint='curve')
    g.line(pts, fill=col + (int(110 * a),), width=24)
    thick_arrow(d, g, pts[-4], pts[-1], col, a, w)
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
XC = 515
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #6', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Painful wrist after a fall:', 'MontXB', 44, ICE, 860), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('X-RAY OR NOT?', 'MontBlack', 124, WHITE, 860)
    put(glow, k1, W / 2, 650, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 650, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('AMSTERDAM WRIST RULES', 'MontBlack', 56, SKY, 820, track=2), W / 2, 790, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('8 predictors  ·  sensitivity 98%', 'MontXB', 42, WHITE), W / 2, 880 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); cx, cy, s = 540, 1310, 0.5
    put(main, diagram(s), cx, cy + 20 * (1 - hq), a * hq)
    pul = 0.6 + 0.4 * math.sin(t * 6)
    press(main, glow, dp(RAD_T, cx, cy + 20 * (1 - hq), s), t, a * eo(prog(t, 0.3, 0.7)) * pul, WARN, 16)

def s2(t, main, glow, fx):                                     # context: 39% fracture, 99.6% X-rayed
    a = win(t, 3.3, 9.0, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('WRIST TRAUMA IN THE ED', 'MontBlack', 56, WHITE, track=2), XC, 262, a * p, 1, 10 * (1 - p))
    label(main, 'each square = 1% of patients', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    sz, gap = 56, 14; x0 = XC - (10 * sz + 9 * gap) / 2; y0 = 440
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    for i in range(100):
        r, c = divmod(i, 10)
        qa = eo(prog(t, 3.7 + 0.012 * i, 4.2 + 0.012 * i))
        if qa <= 0: continue
        X, Y = x0 + c * (sz + gap), y0 + r * (sz + gap)
        frac = i < 39
        fq = eo(prog(t, 5.0 + 0.02 * i, 5.4 + 0.02 * i)) if frac else 0
        col = tuple(int(ICE[j] * (1 - fq) + WARN[j] * fq) for j in range(3))
        d.rounded_rectangle((X, Y, X + sz, Y + sz), 12, fill=col + (int(255 * a * qa * (0.28 + 0.72 * fq)),),
                            outline=ICE + (int(150 * a * qa),), width=2)
    main.alpha_composite(lay)
    q = eo(prog(t, 5.4, 5.9))
    if q > 0:
        y = 1250 + 20 * (1 - q)
        n1 = tl('39%', 'MontBlack', 92, WARN); put(main, n1, 120 + n1.width / 2, y, a * q)
        im = fit('have a fracture', 'MontBlack', 44, WHITE, 520); put(main, im, 120 + n1.width + 30 + im.width / 2, y, a * q)
    q = eo(prog(t, 6.3, 6.8))
    if q > 0:
        y = 1420 + 20 * (1 - q)
        put(main, card(880, 170, outline=SKY + (150,)), XC, y, a * q)
        label(main, 'yet 99.6% were X-rayed', XC, y - 28, a * q, WHITE, 46, 'MontBlack', 0)
        label(main, 'in a separate cohort of 924 ED adults', XC, y + 42, a * q, SKY, 32, 'MontXB', 0)
    label(main, 'Walenkamp: 39% BMC MSK 2015  ·  99.6% EJTES 2016', XC, 1585, a * eo(prog(t, 6.6, 7.1)), SUB, 28, 'MontXB', 0)

PRED = [  # (start, n, name, sub)
    (10.4, 1, 'AGE', 'patient factor'),
    (11.6, 2, 'SEX', 'patient factor'),
    (12.8, 3, 'WRIST SWELLING', 'look'),
    (14.2, 4, 'SNUFFBOX SWELLING', 'look: over the scaphoid'),
    (16.0, 5, 'VISIBLE DEFORMATION', 'look'),
    (17.6, 6, 'DISTAL RADIUS TENDER', 'feel: palpate the radius'),
    (19.2, 7, 'PAIN ON RADIAL DEVIATION', 'move: tilt toward the thumb'),
    (21.0, 8, 'AXIAL THUMB COMPRESSION', 'move: push along the thumb'),
]
CX, CY, S = 470, 880, 0.8
def s3(t, main, glow, fx):                                     # centrepiece: the 8 predictors on the wrist
    a = win(t, 8.8, 24.2, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 8.9, 9.4))
    put(main, tl('THE 8 PREDICTORS', 'MontBlack', 60, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Right wrist  ·  PA view  ·  adults', XC, 335, a * eo(prog(t, 9.1, 9.6)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 9.0, 9.9))
    cy = CY + 24 * (1 - bp)
    put(main, diagram(S), CX, cy, a * bp)
    P = lambda q_: dp(q_, CX, cy, S)
    la = a * eo(prog(t, 9.5, 9.9)) * (1 - eo(prog(t, 12.6, 13.0)))
    pointer(main, la, P((330, 960)), 170, 1180, 'RADIUS', ICE, 31)
    pointer(main, la, P((560, 900)), 860, 1060, 'ULNA', ICE, 31)
    pointer(main, la, P((300, 610)), 860, 660, 'CARPUS', ICE, 31)
    cur = None
    for i, (t0, n, nm, sub) in enumerate(PRED):
        t1 = PRED[i + 1][0] if i < 7 else 24.2
        if t0 <= t < t1: cur = i
    lay = aa_image(W, H); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    def act(i): return a * eo(prog(t, PRED[i][0], PRED[i][0] + 0.4)) * (1.0 if cur == i else 0.45)
    # 3 wrist swelling: large puff around carpus + distal forearm
    if t >= PRED[2][0]: swell(main, glow, P((370, 640)), 250 * S, 175 * S, t, act(2), SKY)
    # 4 snuffbox swelling
    if t >= PRED[3][0]: swell(main, glow, P(SNUFF), 58 * S, 48 * S, t + 0.4, act(3), WARN)
    # 5 visible deformation: step in the radial contour of the distal forearm
    if t >= PRED[4][0]:
        aa = act(4); pl = eio(prog(t, PRED[4][0], PRED[4][0] + 0.8))
        pts = [P(q_) for q_ in [(150, 610), (140, 690), (176, 740), (128, 800), (150, 900), (176, 990)]]
        n_ = max(2, int(len(pts) * pl + 0.999))
        d.line(pts[:n_], fill=WARN + (int(255 * aa),), width=8, joint='curve')
        g.line(pts[:n_], fill=WARN + (int(120 * aa),), width=22)
    # 6 distal radius tenderness
    if t >= PRED[5][0]: press(main, glow, P(RAD_T), t, act(5), WARN)
    # 7 pain on radial deviation: the hand swings toward the thumb side (viewer's left)
    if t >= PRED[6][0]:
        sw = 0.5 + 0.5 * math.sin((t - PRED[6][0]) * 3.2)
        arc_arrow(main, glow, P(PIVOT), 470 * S, -62, -62 - 34 * (0.55 + 0.45 * sw), SKY, act(6))
        press(main, glow, P((200, 640)), t + 0.2, act(6) * 0.9, WARN, 10)
    # 8 axial compression of the thumb
    if t >= PRED[7][0]:
        aa = act(7); (x0, y0), (x1, y1) = THUMB_AX; push = 0.5 + 0.5 * math.sin(t * 5)
        L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L
        st = P((x0 - ux * (60 - 18 * push), y0 - uy * (60 - 18 * push))); en = P((x0 + ux * (L - 110), y0 + uy * (L - 110)))
        thick_arrow(d, g, st, en, SKY, aa)
    main.alpha_composite(lay)
    # bottom panel: the current predictor
    pa = a * eo(prog(t, 10.2, 10.6))
    if pa > 0:
        y = 1450
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        if cur is not None:
            t0, n, nm, sub = PRED[cur]
            q = eo(prog(t, t0, t0 + 0.35))
            put(main, pill(f'{n} / 8', 'MontBlack', 36, NAVY, SKY + (255,), padx=22, h=64), 170, y - 40, pa * q)
            im = fit(nm, 'MontBlack', 46, WHITE, 650); put(main, im, 270 + im.width / 2, y - 40, pa * q)
            im = fit(sub, 'MontXB', 34, SKY, 760); put(main, im, 120 + im.width / 2, y + 50, pa * q)
        dd = ImageDraw.Draw(main)
        for k in range(8):
            on = t >= PRED[k][0]
            x = XC - 7 * 50 + k * 100; yy = 1625
            col = (SKY if on else (70, 95, 125)) + (int(255 * pa),)
            dd.ellipse((x - 14, yy - 14, x + 14, yy + 14), fill=col)

def gauge(main, glow, y, a, t0, t, name, v, col, note):
    q = eo(prog(t, t0, t0 + 0.4))
    if q <= 0: return
    y = y + 20 * (1 - q)
    put(main, card(880, 230), XC, y, a * q)
    im = fit(name, 'MontBlack', 42, WHITE, 560); put(main, im, 120 + im.width / 2, y - 55, a * q)
    g_ = eio(prog(t, t0 + 0.2, t0 + 1.1))
    nb = tl(f'{round(v * g_)}%', 'MontBlack', 70, col); put(main, nb, 905 - nb.width / 2, y - 55, a * q)
    d = ImageDraw.Draw(main)
    d.rounded_rectangle((120, y + 10, 910, y + 40), 15, fill=(60, 90, 125, int(200 * a * q)))
    d.rounded_rectangle((120, y + 10, 120 + max(30, 790 * v / 100 * g_), y + 40), 15, fill=col + (int(255 * a * q),))
    ImageDraw.Draw(glow).rounded_rectangle((120, y + 10, 120 + 790 * v / 100 * g_, y + 40), 15, fill=col + (int(60 * a * q),))
    im = fit(note, 'MontXB', 30, SUB, 790); put(main, im, 120 + im.width / 2, y + 82, a * q)

def s4(t, main, glow, fx):                                     # accuracy at external validation
    a = win(t, 24.0, 30.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'HOW GOOD IS IT?', XC, 262, a * eo(prog(t, 24.1, 24.6)), WHITE, 62, 'MontBlack', 3)
    label(main, 'external validation  ·  395 adults', XC, 340, a * eo(prog(t, 24.3, 24.8)), SKY, 36, 'MontXB', 0)
    gauge(main, glow, 560, a, 24.7, t, 'Sensitivity', 98, GOOD, 'misses few fractures: a rule-OUT tool')
    gauge(main, glow, 840, a, 25.6, t, 'Negative predictive value', 90, GOOD, 'rule says no X-ray: 90% truly no fracture')
    gauge(main, glow, 1120, a, 26.5, t, 'Specificity', 21, WARN, 'many positives are still fracture-free')
    q = eo(prog(t, 27.6, 28.1))
    if q > 0:
        put(main, pill('AUC 0.81', 'MontBlack', 40, NAVY, SKY + (255,), h=72), XC, 1390 + 15 * (1 - q), a * q)
    label(main, 'Walenkamp et al.  ·  BMC Musculoskelet Disord 2015', XC, 1500, a * eo(prog(t, 27.9, 28.4)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # implementation
    a = win(t, 30.2, 36.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'IN REAL LIFE', XC, 262, a * eo(prog(t, 30.3, 30.8)), WHITE, 64, 'MontBlack', 4)
    label(main, 'implementation  ·  4 hospitals  ·  402 adults', XC, 340, a * eo(prog(t, 30.5, 31.0)), SKY, 34, 'MontXB', 0)
    rows = [(30.9, '15%', 'fewer wrist X-rays', 'absolute reduction', GOOD),
            (31.8, '34 min', 'less time in the ED', 'no fracture, X-ray skipped', GOOD),
            (32.7, '1', 'fracture missed', 'clinically irrelevant', GOOD),
            (33.6, '36%', 'physician adherence', 'room to use it more', WARN)]
    for i, (t0, num, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 540 + i * 255 + 20 * (1 - q)
        put(main, card(880, 225, outline=col + (150,)), XC, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        nb = tl(num, 'MontBlack', 78, col); put(main, nb, 125 + nb.width / 2, y, a * q * clamp(pk), clamp(pk, 0, 1.2))
        x0 = 125 + max(nb.width, 150) + 30
        im = fit(l1, 'MontBlack', 42, WHITE, 900 - x0); put(main, im, x0 + im.width / 2, y - 30, a * q)
        im = fit(l2, 'MontXB', 30, SUB, 900 - x0); put(main, im, x0 + im.width / 2, y + 34, a * q)
    label(main, 'Mulders et al.  ·  Eur J Trauma Emerg Surg 2020', XC, 1560, a * eo(prog(t, 34.0, 34.5)), SUB, 28, 'MontXB', 0)

TIPS = [('Adults only: kids have their own', 'PEDIATRIC RULES: 6 ITEMS, 4.1% MISSED', 'Slaar et al. Pediatr Radiol 2016'),
        ('Snuffbox signs? Think scaphoid', 'SCAPHOID RULE: SENS 97%, SPEC 20% (INTERNAL)', 'Mallee et al. Arch Orthop Trauma Surg 2020'),
        ('Low specificity is the trade-off', 'A SCREEN FOR THE X-RAY, NOT A DIAGNOSIS', 'Walenkamp et al. BMC MSK Disord 2015')]
def s6(t, main, glow, fx):
    a = win(t, 36.2, 42.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 36.3, 36.8)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 36.8 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
        put(main, card(880, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 42, WHITE, 700); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 28, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s7(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 42.4, 49.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 380, a * eo(prog(t, 42.5, 43.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 42.7, 43.3))
    big = fit('8 PREDICTORS', 'MontBlack', 110, WHITE, 880)
    put(glow, big, XC, 520, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 520, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'pick who needs the wrist X-ray', XC, 640, a * eo(prog(t, 43.1, 43.6)), GOOD, 40, 'MontBlack', 0)
    label(main, 'Sensitivity 98%  ·  X-rays down 15% (absolute)', XC, 710, a * eo(prog(t, 43.4, 43.9)), SKY, 36, 'MontXB', 0)
    q = eo(prog(t, 43.8, 44.4))
    refs = ['Walenkamp MMJ, et al. BMC MSK Disord 2015;16:389', 'Walenkamp MMJ, et al. EJTES 2016;42:185–90',
            'Mulders MAM, et al. EJTES 2020;46:573–82', 'Slaar A, et al. Pediatr Radiol 2016;46:50–60',
            'Mallee WH, et al. Arch Orthop Trauma Surg 2020;140:575–81']
    put(main, card(880, 460, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1060, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 820), XC, 900 + i * 80, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1400, a * eo(prog(t, 44.4, 45.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
