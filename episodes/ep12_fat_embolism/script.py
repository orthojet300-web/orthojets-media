"""Orthojets EP12 - CLINICAL RULES #3 - Fat embolism syndrome: the Gurd criteria (1 major + 4 minor).
Sources (verified on PubMed metadata / abstracts / PMC full text, 2026-11-01):
  Gurd AR, Wilson RI. The fat embolism syndrome. J Bone Joint Surg Br 1974;56B(3):408-16. PMID 4547466 (no DOI,
    no abstract on PubMed). Criteria content taken from the PMC review below (Table 1).
  Miyake T, Okada H, Kanda N. Advances and uncertainties in fat embolism syndrome: a review.
    Trauma Surg Acute Care Open 2026;11(2):e001913. PMID 42375760, PMC13311676, doi 10.1136/tsaco-2025-001913.
    - classic triad: respiratory distress, neurological dysfunction, petechial rash
    - Gurd and Wilson: at least 1 major (petechiae, hypoxemia, CNS depression, pulmonary edema) + at least 4 minor
      (tachycardia, fever, anemia, thrombocytopenia, retinal changes, fat macroglobulinemia)
    - criteria not validated in prospective studies
    - petechial rash present in less than half of cases
    - FES presents 1-3 days after injury
    - mechanism: fat emboli lodge first in the pulmonary microvasculature, then reach the arterial circulation
      through right-to-left shunts (patent foramen ovale, arteriovenous shunts)
    - incidence today 1.4% to 12.5% after long bone fractures or trauma
    - 'starfield pattern' (brain MRI) in 61.5-84.6% of cerebral fat embolism
    - early fracture stabilisation within 24 h = the most effective preventive strategy
  Pinney SJ, Keating JF, Meek RN. Injury 1998;29(2):131-3. PMID 10721407, doi 10.1016/s0020-1383(97)00154-x.
    274 isolated femoral shaft fractures, 11 FES (4%); none over 35 years; of the rest, 60 nailed within 10 h: no FES;
    109 nailed after 10 h: 11 (10%) FES (p 0.027).
  Diagram: left femur, anterior view (kit/femur.py), transverse mid-shaft fracture; marrow fat droplets travel via the
  venous system to the right heart and lungs (hypoxaemia), then through a right-to-left shunt to the brain (CNS) and
  skin (petechiae). Schematic, labelled.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FM

TL = Timeline(45.0, [(7.4, 1.4), (9.8, 1.4), (15.4, 1.6), (18.0, 1.8), (21.4, 1.4), (24.6, 1.4), (27.6, 1.8),
                     (35.2, 1.8), (41.0, 1.8)])
FAT = SKY                     # series accent (gold for CLINICAL RULES) doubles as the fat colour

# ------------------------------------------------------------------ femur (X-ray), split at a mid-shaft fracture
FS = 0.62
FCX, FCY = 225, 1095                                  # screen centre of the femur image
_img, _edge = FM.render()
_e = Image.new('RGBA', _img.size, (226, 238, 250, 0)); _e.putalpha(_edge); _img.alpha_composite(_e)
FY = 640                                              # fracture level (bone space)
def _part(top):
    m = Image.new('L', _img.size, 0); d = ImageDraw.Draw(m)
    # slightly oblique-transverse line
    if top: d.polygon([(0, 0), (FM.BW, 0), (FM.BW, FY - 10), (0, FY + 10)], fill=255)
    else: d.polygon([(0, FY + 10), (FM.BW, FY - 10), (FM.BW, FM.BH), (0, FM.BH)], fill=255)
    im = Image.new('RGBA', _img.size); im.paste(_img, (0, 0), m)
    return im.resize((int(FM.BW * FS), int(FM.BH * FS)), Image.LANCZOS)
PROX, DIST = _part(True), _part(False)
def fp(q): return FCX - FM.BW * FS / 2 + q[0] * FS, FCY - FM.BH * FS / 2 + q[1] * FS
FRAC = fp((285, 640))                                 # fracture centre on screen

def femur(main, a, disp):
    if a <= 0: return
    put(main, PROX, FCX, FCY, a)
    put(main, DIST.rotate(-7 * disp, resample=Image.BICUBIC, center=(FRAC[0] - (FCX - PROX.width / 2), FRAC[1] - (FCY - PROX.height / 2))),
        FCX + 24 * disp, FCY + 22 * disp, a)
    if disp > 0:
        lay = Image.new('RGBA', (W, H)); ImageDraw.Draw(lay).line([(FRAC[0] - 45, FRAC[1] + 8), (FRAC[0] + 45, FRAC[1] - 4)], fill=WARN + (int(200 * a * disp),), width=4)
        main.alpha_composite(lay)

# ------------------------------------------------------------------ organs (schematic)
HEAD = (700, 505, 92)
LUNG_C = (700, 820)
HEART = (668, 930)
SKIN = (720, 1250, 300, 150)

def catm(pts, n=10): return FM.catmull(pts, n)
LUNG_R = [(-20, -120), (-60, -105), (-120, -40), (-155, 60), (-160, 130), (-110, 150), (-40, 120), (-25, 30)]   # viewer's left
LUNG_L = [(20, -120), (60, -105), (120, -40), (150, 60), (155, 130), (100, 150), (60, 110), (25, 30)]
def lungs(main, glow, a, hl):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    cx, cy = LUNG_C
    fillc = tuple(int(c * (1 - hl) + w * hl) for c, w in zip((120, 160, 200), WARN))
    for poly in (LUNG_R, LUNG_L):
        P = [(cx + x, cy + y) for x, y in catm(poly)]
        d.polygon(P, fill=fillc + (int((60 + 50 * hl) * a),), outline=ICE + (int(220 * a),))
    d.line([(cx, cy - 175), (cx, cy - 90)], fill=ICE + (int(220 * a),), width=8)                 # trachea
    d.line([(cx, cy - 90), (cx - 40, cy - 40)], fill=ICE + (int(220 * a),), width=6)
    d.line([(cx, cy - 90), (cx + 40, cy - 40)], fill=ICE + (int(220 * a),), width=6)
    main.alpha_composite(lay)
    if hl > 0: ImageDraw.Draw(glow).ellipse((cx - 150, cy - 110, cx + 150, cy + 140), fill=WARN + (int(40 * hl * a),))

def heart(main, glow, a):
    if a <= 0: return
    x, y = HEART
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    P = [(x + dx, y + dy) for dx, dy in catm([(0, -30), (28, -46), (52, -24), (40, 18), (0, 52), (-40, 18), (-52, -24), (-28, -46)])]
    d.polygon(P, fill=WARN + (int(200 * a),))
    main.alpha_composite(lay)

def head(main, glow, a, hl):
    if a <= 0: return
    x, y, r = HEAD
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    d.ellipse((x - r, y - r, x + r, y + r), outline=ICE + (int(220 * a),), width=6)
    d.line([(x - 34, y + r - 8), (x - 34, y + r + 40)], fill=ICE + (int(220 * a),), width=6)
    d.line([(x + 34, y + r - 8), (x + 34, y + r + 40)], fill=ICE + (int(220 * a),), width=6)
    br = Image.new('RGBA', (W, H)); bd = ImageDraw.Draw(br)
    bd.ellipse((x - r + 18, y - r + 16, x + r - 18, y + 22), fill=(170, 150, 190, int((90 + 40 * hl) * a)))
    for k in range(3):                                   # gyri hints
        bd.arc((x - 55 + k * 30, y - 62, x - 15 + k * 30, y - 10), 200, 340, fill=ICE + (int(150 * a),), width=3)
    main.alpha_composite(br); main.alpha_composite(lay)

def skin(main, glow, a, hl, t):
    if a <= 0: return
    x, y, w, h = SKIN
    put(main, card(w, h, fill=(214, 170, 150, int(70 * a) if a < 1 else 70), outline=ICE + (160,), r=30), x, y, a)
    rng = np.random.default_rng(5)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    n = int(26 * hl)
    for k in range(26):
        px = x - w / 2 + 25 + rng.random() * (w - 50); py = y - h / 2 + 22 + rng.random() * (h - 44); rr = 3 + rng.random() * 4
        if k < n: d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=WARN + (int(255 * a),))
    main.alpha_composite(lay)

# fat flow: fracture -> vein -> right heart -> lungs ; shunt -> brain ; shunt -> skin
P_VEIN = [(FRAC[0] + 20, FRAC[1]), (420, FRAC[1] - 10), (520, 1040), (HEART[0] - 30, HEART[1] + 10), (HEART[0], HEART[1] - 20),
          (LUNG_C[0] - 70, LUNG_C[1] + 20)]
P_BRAIN = [(HEART[0] - 20, HEART[1] - 30), (LUNG_C[0] - 120, LUNG_C[1] - 40), (565, 680), (590, 600), (HEAD[0] - 62, HEAD[1] + 50)]
P_SKIN = [(HEART[0] + 30, HEART[1] + 10), (HEART[0] + 150, HEART[1] + 90), (SKIN[0] + 60, SKIN[1] - 20)]

def ppos(pts, u):
    L = [math.dist(a, b) for a, b in zip(pts, pts[1:])]; tot = sum(L) * clamp(u)
    for (a, b), l in zip(zip(pts, pts[1:]), L):
        if tot <= l: f = tot / l if l else 0; return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        tot -= l
    return pts[-1]

def path_part(pts, u):
    if u >= 1: return pts
    L = [math.dist(a, b) for a, b in zip(pts, pts[1:])]; tot = sum(L) * u; out = [pts[0]]
    for (a, b), l in zip(zip(pts, pts[1:]), L):
        if tot >= l: out.append(b); tot -= l
        else:
            f = tot / l if l else 0; out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)); break
    return out

def flow(main, glow, pts, a, u, t, col=None, dashed=False, nd=7, speed=0.35):
    """draw the route progressively (u) and droplets moving along it"""
    if a <= 0 or u <= 0: return
    col = col or (150, 185, 220)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    P = path_part([(x, y) for x, y in FM.catmull(pts, 8, closed=False)] if len(pts) > 2 else pts, u)
    if dashed:
        for A_, B_ in zip(P, P[1:]):
            n = max(1, int(math.dist(A_, B_) / 18))
            for i in range(0, n, 2):
                f0, f1 = i / n, min(1, (i + 1) / n)
                d.line([(A_[0] + (B_[0] - A_[0]) * f0, A_[1] + (B_[1] - A_[1]) * f0),
                        (A_[0] + (B_[0] - A_[0]) * f1, A_[1] + (B_[1] - A_[1]) * f1)], fill=col + (int(200 * a),), width=6)
    else:
        d.line(P, fill=col + (int(170 * a),), width=10, joint='curve')
    full = [(x, y) for x, y in FM.catmull(pts, 8, closed=False)] if len(pts) > 2 else pts
    gd = ImageDraw.Draw(glow)
    if u >= 1:
        for k in range(nd):
            v = (t * speed + k / nd) % 1
            x, y = ppos(full, v); rr = 9
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=FAT + (int(255 * a),))
            gd.ellipse((x - rr - 8, y - rr - 8, x + rr + 8, y + rr + 8), fill=FAT + (int(110 * a),))
    main.alpha_composite(lay)

def droplets_at_fracture(main, glow, a, t):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); gd = ImageDraw.Draw(glow)
    rng = np.random.default_rng(3)
    for k in range(9):
        ph = rng.random(); ang = rng.random() * 2 * math.pi
        rr = 14 + 26 * ((t * 0.6 + ph) % 1)
        x = FRAC[0] + rr * math.cos(ang); y = FRAC[1] + 0.6 * rr * math.sin(ang); s = 6 + 3 * rng.random()
        d.ellipse((x - s, y - s, x + s, y + s), fill=FAT + (int(255 * a),))
        gd.ellipse((x - s - 6, y - s - 6, x + s + 6, y + s + 6), fill=FAT + (int(90 * a),))
    main.alpha_composite(lay)

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14 if lx > tip[0] else lx + lab.width / 2 - 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay)
    put(main, lab, lx, ly, a)

def fit(txt, font, size, col, maxw, track=0):
    im = tl(txt, font, size, col, track=track)
    while im.width > maxw and size > 30:
        size -= 4; im = tl(txt, font, size, col, track=track)
    return im

def leftput(main, im, x, y, a): put(main, im, x + im.width / 2, y, a)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #3', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 430, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl('Day 2 after a femur fracture:', 'MontXB', 42, ICE), W / 2, 540, a * q)
    put(main, tl('hypoxic and confused?', 'MontXB', 42, ICE), W / 2, 598, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FAT EMBOLISM', 'MontBlack', 130, WHITE, 800)
    put(glow, k1, W / 2, 760, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 760, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('Gurd: 1 major + 4 minor', 'MontBlack', 62, SKY), W / 2, 905 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('ONSET 1-3 DAYS AFTER INJURY', 'MontBlack', 33, NAVY, WARN + (255,), h=80), W / 2, 1060 + 20 * (1 - q), a * q)
    q = eo(prog(t, 1.1, 1.6))
    put(main, pill('LUNGS  ·  BRAIN  ·  SKIN', 'MontBlack', 38, NAVY, ICE + (255,), h=80), W / 2, 1170 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # centrepiece: where the marrow fat goes
    a = win(t, 3.3, 12.6, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('WHERE THE FAT GOES', 'MontBlack', 64, WHITE, 800, track=2), W / 2, 262, a * p, 1, 10 * (1 - p))
    sa = 1 - prog(t, 10.0, 10.25)
    label(main, 'Marrow fat enters the veins', W / 2, 345, a * eo(prog(t, 3.6, 4.1)) * sa, SKY, 36, 'MontXB', 0)
    label(main, 'Lungs + brain + skin = the classic triad', W / 2, 345, a * eo(prog(t, 10.3, 10.65)), SKY, 34, 'MontXB', 0)
    q = eo(prog(t, 3.6, 4.2)); disp = eio(prog(t, 4.3, 4.8))
    femur(main, a * q, disp)
    if 4.6 < t < 5.0: fx['flash'] += 0.10 * math.exp(-((t - 4.75) / 0.06) ** 2)
    pointer(main, a * eo(prog(t, 4.1, 4.6)), fp((150, 128)), 150, 650, 'FEMUR', ICE)
    droplets_at_fracture(main, glow, a * eo(prog(t, 4.8, 5.2)), t)
    pointer(main, a * eo(prog(t, 5.0, 5.4)), (FRAC[0] + 14, FRAC[1] + 14), 470, 1160, 'MARROW FAT', FAT)
    # route to the right heart and lungs
    hq = eo(prog(t, 5.3, 5.8))
    lungs(main, glow, a * hq, eo(prog(t, 6.9, 7.3)))
    heart(main, glow, a * hq)
    flow(main, glow, P_VEIN, a, eio(prog(t, 5.6, 6.8)), t)
    if hq > 0:
        q = eo(prog(t, 7.0, 7.4))
        label(main, 'LUNGS', 855, 600, a * q, WARN, 40, 'MontBlack', 2)
        label(main, 'hypoxaemia', 850, 646, a * q, WHITE, 32, 'MontXB', 0)
    # through a right-to-left shunt to the brain and the skin
    bq = eo(prog(t, 7.6, 8.0))
    head(main, glow, a * bq, eo(prog(t, 8.6, 9.0)))
    flow(main, glow, P_BRAIN, a * bq, eio(prog(t, 7.8, 8.6)), t + 0.3, col=(170, 150, 200), dashed=True, nd=4)
    q = eo(prog(t, 8.7, 9.1))
    label(main, 'BRAIN', 330, 450, a * q, WARN, 40, 'MontBlack', 2)
    label(main, 'confusion, CNS', 330, 500, a * q, WHITE, 32, 'MontXB', 0)
    label(main, 'depression', 330, 542, a * q, WHITE, 32, 'MontXB', 0)
    sq = eo(prog(t, 9.0, 9.4))
    skin(main, glow, a * sq, eo(prog(t, 9.4, 10.0)), t)
    flow(main, glow, P_SKIN, a * sq, eio(prog(t, 9.1, 9.6)), t + 0.6, col=(170, 150, 200), dashed=True, nd=3)
    q = eo(prog(t, 9.5, 9.9))
    label(main, 'SKIN', 720, 1375, a * q, WARN, 40, 'MontBlack', 2)
    label(main, 'petechial rash', 720, 1425, a * q, WHITE, 32, 'MontXB', 0)
    q = eo(prog(t, 7.9, 8.3))
    label(main, 'via R-to-L shunt (PFO)', 372, 618, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Miyake et al.  ·  Trauma Surg Acute Care Open 2026', W / 2, 1650, a * eo(prog(t, 4.0, 4.5)), SUB, 28, 'MontXB', 0)

MAJOR = ['PETECHIAL RASH', 'HYPOXAEMIA', 'PULMONARY OEDEMA', 'CNS DEPRESSION']
MINOR = ['TACHYCARDIA', 'FEVER', 'RETINAL CHANGES', 'ANAEMIA', 'THROMBOCYTOPENIA', 'FAT MACROGLOBULINAEMIA']
def s3(t, main, glow, fx):                                     # the criteria
    a = win(t, 12.4, 22.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'GURD CRITERIA', W / 2, 262, a * eo(prog(t, 12.5, 13.0)), WHITE, 70, 'MontBlack', 4)
    label(main, 'Gurd & Wilson 1974  ·  via Miyake et al. 2026', W / 2, 340, a * eo(prog(t, 12.7, 13.2)), SKY, 34, 'MontXB', 0)
    q = eo(prog(t, 13.2, 13.7))
    if q > 0:
        y = 575 + 20 * (1 - q)
        put(main, card(830, 340, outline=WARN + (170,)), 515, y, a * q)
        leftput(main, tl('MAJOR', 'MontBlack', 46, WARN, track=3), 120, y - 120, a * q)
        lab = tl('AT LEAST 1', 'MontBlack', 34, NAVY)
        put(main, pill('AT LEAST 1', 'MontBlack', 30, NAVY, WARN + (255,), h=56), 755, y - 120, a * q)
        for i, txt in enumerate(MAJOR):
            qq = eo(prog(t, 13.7 + 0.4 * i, 14.2 + 0.4 * i))
            yy = y - 50 + i * 60
            lay = Image.new('RGBA', (W, H)); dot(ImageDraw.Draw(lay), 138, yy, 10, WARN, a * qq); main.alpha_composite(lay)
            leftput(main, tl(txt, 'MontBlack', 34, WHITE, track=1), 160, yy, a * qq)
    q = eo(prog(t, 15.6, 16.1))
    if q > 0:
        y = 1010 + 20 * (1 - q)
        put(main, card(830, 470, outline=SKY + (170,)), 515, y, a * q)
        leftput(main, tl('MINOR', 'MontBlack', 46, SKY, track=3), 120, y - 185, a * q)
        put(main, pill('AT LEAST 4', 'MontBlack', 30, NAVY, SKY + (255,), h=56), 755, y - 185, a * q)
        for i, txt in enumerate(MINOR):
            qq = eo(prog(t, 16.0 + 0.3 * i, 16.4 + 0.3 * i))
            yy = y - 110 + i * 62
            lay = Image.new('RGBA', (W, H)); dot(ImageDraw.Draw(lay), 138, yy, 9, SKY, a * qq); main.alpha_composite(lay)
            leftput(main, tl(txt, 'MontXB', 34, ICE, track=1), 160, yy, a * qq)
    p = R.eback(prog(t, 18.2, 18.7))
    if p > 0:
        im = fit('1 MAJOR + 4 MINOR = FES', 'MontBlack', 60, WHITE, 800)
        put(glow, im, 515, 1365, a * 0.3 * clamp(p), clamp(p, 0, 1.15)); put(main, im, 515, 1365, a * clamp(p), clamp(p, 0, 1.15))
    q = eo(prog(t, 19.6, 20.1))
    if q > 0:
        y = 1520 + 15 * (1 - q)
        put(main, card(830, 120, outline=WARN + (150,)), 515, y, a * q)
        label(main, 'Never prospectively validated', 515, y - 20, a * q, WHITE, 38, 'MontBlack', 0)
        label(main, 'a guide, not a rule-out', 515, y + 28, a * q, SUB, 30, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # timing: onset and fixation
    a = win(t, 22.2, 30.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIMING', W / 2, 262, a * eo(prog(t, 22.3, 22.8)), WHITE, 70, 'MontBlack', 6)
    label(main, 'when it starts', W / 2, 340, a * eo(prog(t, 22.5, 23.0)), SKY, 36, 'MontXB', 0)
    X = lambda h: 130 + 760 * h / 96
    ax = eio(prog(t, 22.6, 23.3)); yb = 560
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    sq = eio(prog(t, 23.3, 24.0))
    if sq > 0:
        d.rounded_rectangle((X(24), yb - 34, X(24) + (X(72) - X(24)) * sq, yb + 34), 18, fill=WARN + (int(90 * a),), outline=WARN + (int(220 * a),), width=3)
    d.line([(X(0), yb), (X(0) + (X(96) - X(0)) * ax, yb)], fill=ICE + (int(200 * a),), width=5)
    for h in range(0, 97, 24):
        if X(h) <= X(0) + (X(96) - X(0)) * ax: d.line([(X(h), yb - 12), (X(h), yb + 12)], fill=ICE + (int(200 * a),), width=4)
    main.alpha_composite(lay)
    for h in range(0, 97, 24):
        label(main, f'{h} h', X(h), yb + 70, a * ax, SUB, 28, 'MontXB', 0)
    label(main, 'INJURY', X(0) + 10, yb - 52, a * ax, ICE, 28, 'MontBlack', 1)
    q = eo(prog(t, 23.8, 24.2))
    label(main, 'FES: day 1 to 3', (X(24) + X(72)) / 2, yb - 80, a * q, WARN, 42, 'MontBlack', 0)
    # Pinney 1998
    q = eo(prog(t, 24.8, 25.3))
    label(main, 'NAIL THE FEMUR EARLY', W / 2, 760, a * q, WHITE, 52, 'MontBlack', 2)
    label(main, '274 isolated femoral shaft fractures', W / 2, 822, a * q, SKY, 32, 'MontXB', 0)
    for i, (head_, sub, val, n, col, t0) in enumerate([('within 10 h', '0 of 60', 0.0, '0%', GOOD, 25.6),
                                                       ('after 10 h', '11 of 109', 10.1, '10%', WARN, 26.4)]):
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 960 + i * 170
        put(main, card(830, 140, outline=col + (150,)), 515, y, a * q)
        leftput(main, tl(head_, 'MontBlack', 40, WHITE), 120, y - 26, a * q)
        leftput(main, tl(sub, 'MontXB', 30, SUB), 120, y + 30, a * q)
        bw = 330 * max(0.02, val / 12) * eio(prog(t, t0 + 0.2, t0 + 0.8))
        lay = Image.new('RGBA', (W, H)); ImageDraw.Draw(lay).rounded_rectangle((440, y - 22, 440 + bw, y + 22), 12, fill=col + (int(255 * a),))
        main.alpha_composite(lay)
        put(main, tl(n, 'MontBlack', 56, col), 440 + bw + 70, y, a * q)
    q = eo(prog(t, 27.0, 27.4))
    label(main, 'patients 35 or under  ·  none over 35 had FES', W / 2, 1300, a * q, ICE, 29, 'MontXB', 0)
    q = eo(prog(t, 28.0, 28.5))
    if q > 0:
        y = 1450 + 20 * (1 - q)
        put(main, card(830, 150, outline=SKY + (160,)), 515, y, a * q)
        label(main, 'Early stabilisation within 24 h', 515, y - 26, a * q, WHITE, 40, 'MontBlack', 0)
        label(main, 'is the best prevention we have', 515, y + 30, a * q, SKY, 34, 'MontXB', 0)
    label(main, 'Pinney et al. Injury 1998  ·  Miyake et al. 2026', W / 2, 1610, a * eo(prog(t, 25.2, 25.7)), SUB, 28, 'MontXB', 0)

TIPS = [('Rash in under half', 'DON\'T WAIT FOR PETECHIAE TO SUSPECT IT', 'Miyake et al. 2026  ·  review'),
        ('Confused? Get an MRI', 'DWI STARFIELD PATTERN IN 61.5-84.6% OF CFE', 'Miyake et al. 2026  ·  review'),
        ('Treat = support', 'OXYGEN, ICU; STEROIDS ARE NOT STANDARD', 'Miyake et al. 2026  ·  review')]
def s5(t, main, glow, fx):
    a = win(t, 30.4, 37.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 30.5, 31.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 31.0 + 1.7 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(840, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s6(t, main, glow, fx):
    a = win(t, 37.2, 45.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 330, a * eo(prog(t, 37.3, 37.8)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 37.5, 38.1))
    big = fit('1 MAJOR + 4 MINOR', 'MontBlack', 92, WHITE, 860)
    put(glow, big, W / 2, 465, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 465, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'day 1 to 3 after a long-bone fracture', W / 2, 580, a * eo(prog(t, 37.9, 38.4)), WARN, 38, 'MontBlack', 0)
    label(main, 'Fix early. Suspect early.', W / 2, 660, a * eo(prog(t, 38.2, 38.7)), GOOD, 42, 'MontBlack', 0)
    q = eo(prog(t, 38.6, 39.2))
    put(main, card(850, 340, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 935, a * q)
    for i, (au, ref) in enumerate([('Gurd AR, Wilson RI.', 'J Bone Joint Surg Br. 1974;56B(3):408–16'),
                                   ('Pinney SJ, et al.', 'Injury. 1998;29(2):131–3'),
                                   ('Miyake T, et al.', 'Trauma Surg Acute Care Open. 2026;11(2):e001913')]):
        y = 815 + i * 94
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1200, a * eo(prog(t, 39.2, 39.8)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
