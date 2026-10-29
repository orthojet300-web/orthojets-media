"""Orthojets EP09 - CLASSIFIED #3 - Supracondylar humerus fractures in children: Gartland types, the anterior humeral
line, and the nerve injury patterns.
Sources (verified on PubMed abstracts / metadata, 2026-10-29):
  Gartland JJ. Surg Gynecol Obstet 1959;109(2):145-54. PMID 13675986 (original classification; no abstract, no DOI).
  Vuillermin C, May C, Kasser J. JBJS Essent Surg Tech 2018;8(2):e10. PMID 30233982, PMC6143315,
    doi 10.2106/JBJS.ST.16.00011. Gartland: type I nondisplaced; type II displaced with hinging, posterior cortex
    intact; type III completely displaced. Standard treatment for types II and III = closed reduction and
    percutaneous pinning. Divergent lateral pins: 2 for type II, 3 for type III; medial pin (mini-open) only if
    unstable with lateral-only pins.
  Wang JH, Morris WZ, Bafus BT, Liu RW. J Pediatr Orthop 2019;39(8):e578-85. PMID 31393293,
    doi 10.1097/BPO.0000000000001096. 449 children (level 1 paediatric trauma centre, 2013-15): operative management
    in 0/208 (0%) type I, 61/106 (57.0%) type II, 135/135 (100%) type III. Anterior humeral line not intersecting
    the capitellum predicted operative management (odds ratio 200, P under 0.001).
  Babal JC, Mehlman CT, Klein G. J Pediatr Orthop 2010;30(3):253-63. PMID 20357592, doi 10.1097/BPO.0b013e3181d213a6.
    Meta-analysis, 5148 patients / 5154 displaced fractures. SCHF = the most common elbow fracture in children.
    Traumatic neurapraxia 11.3%. Extension type: AIN = 34.1% of the neurapraxias (most common). Flexion type: ulnar
    = 91.3%. Iatrogenic: lateral-only pins 3.4%, with a medial pin 4.1%; medial pin significantly raises the risk of
    ulnar nerve injury.
  Jose Jerome JT, Ganesan S, Kuppusamy T. J Pediatr Soc North Am 2026;16:100377. PMID 42306145,
    doi 10.1016/j.jposna.2026.100377. 40 studies, 6,850 children: any nerve injury 11.3%; most deficits resolve within
    2-6 months; observation with serial examinations for most closed, well-perfused injuries.
  Joist A et al. J Neurosurg 1999;90(6):1053-6. PMID 10350251, doi 10.3171/jns.1999.90.6.1053. AIN syndrome: paralysis
    of FPL, FDP to the index and pronator quadratus: weak thumb IP and index DIP flexion, no sensory loss.
  Diagram: lateral elbow radiograph schematic of a child, elbow flexed 90 deg, anterior = viewer's LEFT. Humeral shaft,
  distal humerus with the olecranon fossa, separate capitellum ossification centre, radius (radial head aligned with
  the capitellum) and ulna (olecranon, trochlear notch, coronoid). Anterior humeral line along the anterior cortex.
  Extension-type fracture line runs from anterior-distal to posterior-proximal; the distal fragment (with the forearm)
  tilts / shifts posteriorly.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(44.6, [(7.3, 1.8), (11.2, 1.4), (13.8, 1.6), (16.6, 1.6), (23.4, 1.8), (30.6, 2.0), (36.6, 2.0),
                     (41.4, 2.0)])

# ------------------------------------------------------------------ diagram: lateral elbow (local units)
SS = 2; DW, DH = 640, 820
BONE = (226, 238, 250)
AHL_X = 330
CAP_C, CAP_R = (318, 624), 47
F_ANT, F_POST = (322, 486), (469, 438)          # fracture line, anterior-distal -> posterior-proximal (hinge)

HUM = [(330, -30), (330, 380), (327, 440), (316, 500), (302, 540), (318, 562), (348, 576), (384, 580), (420, 570),
       (452, 532), (471, 474), (467, 420), (460, 380), (460, -30)]
ULNA = [(427, 572), (468, 574), (492, 610), (484, 676), (446, 726), (336, 758), (0, 772), (0, 724), (250, 712),
        (300, 692), (338, 636)] + \
       [(395 + 58 * math.cos(math.radians(a)), 625 + 58 * math.sin(math.radians(a))) for a in range(165, -41, -15)]
RADIUS = [(262, 598), (262, 652), (226, 648), (212, 640), (0, 646), (0, 612), (212, 616), (226, 604)]

def S(p): return (p[0] * SS, p[1] * SS)

def mask(poly):
    m = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(m).polygon([S(p) for p in poly], fill=255); return m

def circ_mask(c, r):
    m = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(m).ellipse((S((c[0] - r, c[1] - r)) + S((c[0] + r, c[1] + r))), fill=255)
    return m

def bone_layer(m, a=235):
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(9 * SS))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.42 + 0.58 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: int(a * 0.95) if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

def side_mask(below):
    """half-plane split along the fracture line (True = distal side)"""
    yy, xx = np.mgrid[0:DH * SS, 0:DW * SS].astype(np.float32) / SS
    (x0, y0), (x1, y1) = F_ANT, F_POST
    cr = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)
    m = (cr > 0) if below else (cr <= 0)
    return Image.fromarray((m * 255).astype(np.uint8))

def mul(a, b): return Image.fromarray((np.asarray(a, np.float32) * np.asarray(b, np.float32) / 255).astype(np.uint8))

def fade_top(im):
    f = np.clip((np.arange(im.height) - 180) / 180, 0, 1)[:, None] * np.clip(np.linspace(0, 1, im.width) * 5, 0, 1)[None, :]
    al = np.asarray(im.getchannel('A'), np.float32) * f
    im.putalpha(Image.fromarray(al.astype(np.uint8))); return im

def build():
    hm = mask(HUM)
    # olecranon fossa: thin bone -> less dense
    fos = Image.new('L', hm.size); ImageDraw.Draw(fos).ellipse(S((384, 444)) + S((430, 530)), fill=150)
    fos = fos.filter(ImageFilter.GaussianBlur(4 * SS))
    hum_d = Image.fromarray(np.clip(np.asarray(hm, np.float32) - np.asarray(fos, np.float32), 0, 255).astype(np.uint8))
    prox_m, dist_m = mul(hum_d, side_mask(False)), mul(hum_d, side_mask(True))
    prox = bone_layer(prox_m)
    dist = Image.new('RGBA', hm.size)
    dist.alpha_composite(bone_layer(mask(ULNA), 140))
    dist.alpha_composite(bone_layer(mask(RADIUS), 150))
    dist.alpha_composite(bone_layer(dist_m))
    dist.alpha_composite(bone_layer(circ_mask(CAP_C, CAP_R), 245))
    # teardrop outline (fossa)
    d = ImageDraw.Draw(dist)
    d.ellipse(S((386, 448)) + S((428, 528)), outline=(245, 250, 255, 120), width=2 * SS)
    whole = Image.new('RGBA', hm.size)
    whole.alpha_composite(bone_layer(mask(ULNA), 140))
    whole.alpha_composite(bone_layer(mask(RADIUS), 150))
    whole.alpha_composite(bone_layer(hum_d))
    whole.alpha_composite(bone_layer(circ_mask(CAP_C, CAP_R), 245))
    ImageDraw.Draw(whole).ellipse(S((386, 448)) + S((428, 528)), outline=(245, 250, 255, 120), width=2 * SS)
    prox = fade_top(prox.resize((DW, DH), Image.LANCZOS))
    dist = fade_top(dist.resize((DW, DH), Image.LANCZOS))
    whole = fade_top(whole.resize((DW, DH), Image.LANCZOS))
    return prox, dist, whole

PROX, DIST, WHOLE = build()

def rot_pt(p, c, ang, tr):
    """same transform as PIL rotate(ang, center=c, translate=tr): counter-clockwise on screen"""
    a = math.radians(ang); dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) + dy * math.sin(a) + tr[0], c[1] - dx * math.sin(a) + dy * math.cos(a) + tr[1])

# displacement per type: (angle deg, centre, translate)  - positive angle = distal fragment tilts posteriorly
POSE = {'N': (0, F_POST, (0, 0)), 'I': (0, F_POST, (0, 0)), 'II': (26, F_POST, (0, 0)),
        'III': (8, F_POST, (128, -46)), 'FLEX': (-12, F_ANT, (-92, -22))}

_dc = {}
def elbow(kind, p, s):
    """composited elbow image for a fracture type at displacement progress p (0..1), scaled by s"""
    p = round(p, 2); k = (kind, p, round(s, 3))
    if k in _dc: return _dc[k]
    ang, c, tr = POSE[kind]
    im = Image.new('RGBA', (DW, DH))
    if kind in ('N', 'I'):
        im = WHOLE if s == 1 else WHOLE.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
        _dc[k] = im
        return im
    dist = DIST.rotate(ang * p, resample=Image.BICUBIC, center=c, translate=(tr[0] * p, tr[1] * p)) if p > 0 and kind != 'I' else DIST
    im.alpha_composite(dist); im.alpha_composite(PROX)
    if s != 1: im = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    if len(_dc) > 400: _dc.clear()
    _dc[k] = im
    return im

def cap_now(kind, p):
    ang, c, tr = POSE[kind]
    if kind in ('N', 'I'): return CAP_C
    return rot_pt(CAP_C, c, ang * p, (tr[0] * p, tr[1] * p))

def dp(cx, cy, s, q): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def frac_line(main, glow, cx, cy, s, a, pl, kind='I', p=0.0):
    """fracture line on the proximal fragment edge (drawn progressively)"""
    if a <= 0 or pl <= 0: return
    pts = [F_ANT, ((F_ANT[0] + F_POST[0]) / 2, (F_ANT[1] + F_POST[1]) / 2 + 6), F_POST]
    n = 12; seg = []
    for i in range(n + 1):
        u = i / n * pl
        if u <= 0.5: q = (pts[0][0] + (pts[1][0] - pts[0][0]) * u * 2, pts[0][1] + (pts[1][1] - pts[0][1]) * u * 2)
        else: q = (pts[1][0] + (pts[2][0] - pts[1][0]) * (u - .5) * 2, pts[1][1] + (pts[2][1] - pts[1][1]) * (u - .5) * 2)
        seg.append(dp(cx, cy, s, q))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    w = max(3, int(7 * s))
    for A_, B_ in zip(seg, seg[1:]):
        d.line([A_, B_], fill=(14, 22, 38, int(255 * a)), width=int(w * 1.9))
    for A_, B_ in zip(seg, seg[1:]):
        d.line([A_, B_], fill=WARN + (int(255 * a),), width=w)
        g.line([A_, B_], fill=WARN + (int(160 * a),), width=w * 3)
    main.alpha_composite(lay)

def ahl(main, glow, cx, cy, s, a, pl, kind, p):
    """anterior humeral line + capitellum ring; green if it crosses the capitellum, coral if it misses"""
    if a <= 0 or pl <= 0: return
    cc = cap_now(kind, p)
    hit = abs(cc[0] - AHL_X) < CAP_R * 0.9
    col = GOOD if hit else WARN
    y0, y1 = 200, 200 + (720 - 200) * pl
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    y = y0
    while y < y1:                                    # dashed line
        ye = min(y + 26, y1)
        d.line([dp(cx, cy, s, (AHL_X, y)), dp(cx, cy, s, (AHL_X, ye))], fill=col + (int(255 * a),), width=max(3, int(6 * s)))
        y += 40
    g.line([dp(cx, cy, s, (AHL_X, y0)), dp(cx, cy, s, (AHL_X, y1))], fill=col + (int(110 * a),), width=int(14 * s))
    if pl >= 0.95:
        X, Y = dp(cx, cy, s, cc); r = (CAP_R + 8) * s
        d.ellipse((X - r, Y - r, X + r, Y + r), outline=col + (int(255 * a),), width=max(3, int(5 * s)))
        g.ellipse((X - r, Y - r, X + r, Y + r), outline=col + (int(150 * a),), width=int(12 * s))
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

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #3', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 440, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl("Child's elbow fracture: cast or pins?", 'MontXB', 42, ICE), W / 2, 550, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('GARTLAND', 'MontBlack', 160, WHITE, 860)
    put(glow, k1, W / 2, 700, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 700, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, tl('I  ·  II  ·  III', 'MontBlack', 84, SKY, track=2), W / 2, 860, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('The lateral X-ray decides', 'MontBlack', 54, WHITE), W / 2, 1010 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, pill('I = CAST', 'MontBlack', 40, NAVY, GOOD + (255,), h=80), 285, 1150 + 20 * (1 - q), a * q)
    put(main, pill('II & III = PINS', 'MontBlack', 40, NAVY, WARN + (255,), h=80), 715, 1150 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # context: the normal lateral elbow + AHL
    a = win(t, 3.3, 9.2, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('SUPRACONDYLAR', 'MontBlack', 72, WHITE, track=3), W / 2, 262, a * p, 1, 10 * (1 - p))
    label(main, 'the most common elbow fracture in children', W / 2, 345, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    cx, cy, s = 540, 640, 1.2
    q = eo(prog(t, 3.6, 4.3))
    put(main, elbow('N', 0, s), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), dp(cx, cy, s, (460, 330)), 845, 500, 'HUMERUS')
    pointer(main, a * eo(prog(t, 4.6, 5.1)), dp(cx, cy, s, (CAP_C[0] - 30, CAP_C[1] - 25)), 210, 690, 'CAPITELLUM', SKY, 31)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), dp(cx, cy, s, (480, 590)), 830, 1060, 'OLECRANON')
    pointer(main, a * eo(prog(t, 5.1, 5.6)), dp(cx, cy, s, (120, 630)), 190, 1170, 'RADIUS')
    pointer(main, a * eo(prog(t, 5.3, 5.8)), dp(cx, cy, s, (200, 745)), 470, 1170, 'ULNA')
    pl = eio(prog(t, 6.0, 6.8))
    ahl(main, glow, cx, cy, s, a, pl, 'N', 0)
    q = eo(prog(t, 6.6, 7.1))
    if q > 0:
        y = 1420 + 20 * (1 - q)
        put(main, card(880, 170, outline=GOOD + (150,)), 515, y, a * q)
        label(main, 'ANTERIOR HUMERAL LINE', 515, y - 34, a * q, GOOD, 40, 'MontBlack', 2)
        label(main, 'normal: it crosses the capitellum', 515, y + 30, a * q, WHITE, 34, 'MontXB', 0)

TYPES = [('I', 'Nondisplaced', 'OPERATED: 0%', GOOD, 9.4),
         ('II', 'Displaced, posterior cortex intact', 'OPERATED: 57%', SKY, 11.6),
         ('III', 'Completely displaced', 'OPERATED: 100%', WARN, 14.4)]
def s3(t, main, glow, fx):                                     # centrepiece: the 3 types drawn one by one
    a = win(t, 9.0, 19.6, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = 530, 640, 1.2
    for i, (rom, desc, op, col, t0) in enumerate(TYPES):
        t1 = TYPES[i + 1][4] if i < 2 else 19.6
        ta = a * win(t, t0 - 0.3, t1 + 0.05, 0.3 if i else 0.35, 0.25 if i < 2 else 0.4)
        if ta <= 0: continue
        pdisp = eio(prog(t, t0 + 0.7, t0 + 1.5)) if rom != 'I' else 0
        put(main, elbow(rom, pdisp, s), cx, cy, ta)
        if rom == 'I':
            frac_line(main, glow, cx, cy, s, ta, eio(prog(t, t0 + 0.2, t0 + 0.9)))
        else:
            frac_line(main, glow, cx, cy, s, ta, 0.86 if rom == 'II' else 1.0)
        ahl(main, glow, cx, cy, s, ta * eo(prog(t, t0 + 1.3, t0 + 1.7)), 1.0, rom, pdisp)
        if rom == 'II':                                   # mark the hinge on the intact posterior cortex
            hq = eo(prog(t, t0 + 1.4, t0 + 1.9))
            if hq > 0:
                X, Y = dp(cx, cy, s, F_POST)
                pul = 0.6 + 0.4 * math.sin(t * 6)
                ImageDraw.Draw(glow).ellipse((X - 34, Y - 34, X + 34, Y + 34), fill=GOOD + (int(170 * ta * hq * pul),))
                put(main, pill('HINGE', 'MontBlack', 30, NAVY, GOOD + (255,), h=56), X + 110, Y - 40, ta * hq)
        p = R.eback(prog(t, t0 + 0.05, t0 + 0.5))
        big = tl(f'TYPE {rom}', 'MontBlack', 96, WHITE)
        put(glow, big, W / 2, 265, ta * 0.25 * clamp(p)); put(main, big, W / 2, 265, ta * clamp(p), clamp(p, 0, 1.2))
        dq = eo(prog(t, t0 + 0.4, t0 + 0.9))
        put(main, fit(desc, 'MontBlack', 46, WHITE, 880), W / 2, 1230 + 15 * (1 - dq), ta * dq)
        eq = eo(prog(t, t0 + 1.0, t0 + 1.4))
        put(main, pill(op, 'MontBlack', 38, NAVY, col + (255,), h=72), W / 2, 1345 + 15 * (1 - eq), ta * eq)
    label(main, 'Wang et al.  ·  J Pediatr Orthop 2019  ·  449 children', W / 2, 1600, a * eo(prog(t, 10.0, 10.5)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # the decision: AHL
    a = win(t, 19.4, 25.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'WHAT DECIDES SURGERY?', W / 2, 262, a * eo(prog(t, 19.5, 20.0)), WHITE, 50, 'MontBlack', 1)
    label(main, 'draw the anterior humeral line', W / 2, 335, a * eo(prog(t, 19.7, 20.2)), SKY, 34, 'MontXB', 0)
    s = 0.7; cy = 650
    for kind, cx, head, col, t0 in [('N', 285, 'CROSSES', GOOD, 20.0), ('II', 745, 'MISSES', WARN, 21.0)]:
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        put(main, elbow(kind, 1.0 if kind == 'II' else 0, s), cx, cy + 20 * (1 - q), a * q)
        if kind == 'II': frac_line(main, glow, cx, cy, s, a * q, 0.86)
        ahl(main, glow, cx, cy, s, a * q, eio(prog(t, t0 + 0.3, t0 + 0.9)), kind, 1.0)
        label(main, head, cx, 1010, a * eo(prog(t, t0 + 0.8, t0 + 1.2)), col, 52, 'MontBlack', 2)
    q = eo(prog(t, 22.2, 22.7))
    if q > 0:
        y = 1250 + 25 * (1 - q)
        put(main, card(880, 300, outline=WARN + (170,)), 515, y, a * q)
        label(main, 'Line misses the capitellum', 515, y - 95, a * q, WHITE, 40, 'MontBlack', 0)
        put(main, tl('x200', 'MontBlack', 120, WARN), 515, y + 10, a * q)
        label(main, 'odds of operative treatment', 515, y + 100, a * q, ICE, 34, 'MontXB', 0)
    label(main, 'Wang et al.  ·  J Pediatr Orthop 2019', W / 2, 1600, a * eo(prog(t, 22.6, 23.1)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # nerves: extension vs flexion
    a = win(t, 25.4, 32.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'NOW CHECK THE NERVES', W / 2, 262, a * eo(prog(t, 25.5, 26.0)), WHITE, 50, 'MontBlack', 1)
    p = R.eback(prog(t, 25.7, 26.2))
    big = tl('11.3%', 'MontBlack', 110, SKY)
    put(glow, big, 300, 405, a * 0.3 * clamp(p)); put(main, big, 300, 405, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'about 1 in 9 displaced', 700, 380, a * eo(prog(t, 26.0, 26.5)), WHITE, 32, 'MontXB', 0)
    label(main, 'fractures: nerve palsy', 700, 430, a * eo(prog(t, 26.0, 26.5)), WHITE, 32, 'MontXB', 0)
    s = 0.5; cy = 760
    for kind, cx, head, nerve, pct, t0 in [('III', 285, 'EXTENSION', 'AIN', '34.1%', 26.6), ('FLEX', 745, 'FLEXION', 'ULNAR', '91.3%', 28.2)]:
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        pdisp = eio(prog(t, t0 + 0.3, t0 + 1.0))
        put(main, elbow(kind, pdisp, s), cx, cy + 20 * (1 - q), a * q)
        frac_line(main, glow, cx, cy, s, a * q, 1.0)
        label(main, head, cx, 1000, a * q, ICE, 40, 'MontBlack', 3)
        nq = eo(prog(t, t0 + 0.9, t0 + 1.3))
        if nq > 0:
            y = 1180 + 20 * (1 - nq)
            put(main, card(420, 250, outline=SKY + (150,)), cx, y, a * nq)
            label(main, nerve, cx, y - 60, a * nq, WARN, 64, 'MontBlack', 2)
            label(main, f'{pct} of palsies', cx, y + 20, a * nq, WHITE, 36, 'MontXB', 0)
            label(main, 'the most common' if kind == 'III' else 'by far the most', cx, y + 72, a * nq, SUB, 30, 'MontXB', 0)
    q = eo(prog(t, 29.8, 30.3))
    if q > 0:
        put(main, pill('AIN: weak thumb IP & index DIP', 'MontBlack', 34, NAVY, SKY + (255,), h=70), W / 2, 1420 + 15 * (1 - q), a * q)
    label(main, 'Babal et al.  ·  J Pediatr Orthop 2010  ·  5,148 children', W / 2, 1600, a * eo(prog(t, 26.4, 26.9)), SUB, 28, 'MontXB', 0)

TIPS = [('Test the AIN first', 'CAN\'T MAKE AN "OK" SIGN, SENSATION NORMAL', 'Joist et al. 1999'),
        ('Lateral pins: 2 for II, 3 for III', 'MEDIAL PIN RAISES ULNAR NERVE RISK', 'Vuillermin 2018  ·  Babal 2010'),
        ('Closed, well perfused palsy?', 'OBSERVE: MOST RECOVER IN 2-6 MONTHS', 'Jose Jerome et al. 2026')]
def s6(t, main, glow, fx):
    a = win(t, 32.0, 38.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 32.1, 32.6)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 32.6 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 38.4, 44.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 330, a * eo(prog(t, 38.5, 39.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 38.7, 39.3))
    big = tl('I CAST · II-III PINS', 'MontBlack', 84, WHITE)
    put(glow, big, W / 2, 460, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 460, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'AHL misses the capitellum: think surgery', W / 2, 580, a * eo(prog(t, 39.1, 39.6)), WARN, 35, 'MontBlack', 0)
    label(main, 'Check the AIN before you pin', W / 2, 665, a * eo(prog(t, 39.4, 39.9)), GOOD, 40, 'MontBlack', 0)
    q = eo(prog(t, 39.8, 40.4))
    put(main, card(880, 440, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 1010, a * q)
    for i, (au, ref) in enumerate([('Wang JH, et al.', 'J Pediatr Orthop. 2019;39(8):e578–85'),
                                   ('Babal JC, et al.', 'J Pediatr Orthop. 2010;30(3):253–63'),
                                   ('Vuillermin C, et al.', 'JBJS Essent Surg Tech. 2018;8(2):e10'),
                                   ('Gartland JJ.', 'Surg Gynecol Obstet. 1959;109(2):145–54')]):
        y = 855 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1330, a * eo(prog(t, 40.4, 41.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
