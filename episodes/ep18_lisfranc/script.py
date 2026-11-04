"""Orthojets EP18 - DON'T MISS #4 - Lisfranc injury: the fleck sign, the C1-M2 gap and the weight-bearing X-ray.
Sources (verified on PubMed abstracts / metadata, 2026-11-04):
  Lau S, Bozin M, Thillainadesan T. Lisfranc fracture dislocation: a review of a commonly missed injury of the midfoot.
    Emerg Med J 2017;34(1):52-6. PMID 27013521, doi 10.1136/emermed-2015-205317. "Misdiagnosed in approximately 20% of cases";
    can follow minor trauma.
  Ponkilainen VT et al. Inter- and intraobserver reliability of non-weight-bearing foot radiographs compared with CT in
    Lisfranc injuries. Arch Orthop Trauma Surg 2020;140(10):1423-9. PMID 32140830, doi 10.1007/s00402-020-03391-w.
    100 CT-confirmed sets, 6 readers: mean sensitivity 76.1%, specificity 85.3%; subtle 65.4% vs severe 87.1%;
    "a substantial number (24%) of injuries are missed if only non-weight-bearing radiographs are used".
  Chen C et al. Is the diagnostic validity of conventional radiography for Lisfranc injury acceptable?
    J Foot Ankle Res 2023;16(1):9. PMID 36855126, doi 10.1186/s13047-023-00608-0. 307 injured + 100 controls (407):
    recognition rate three-column 92.1%, intermediate-lateral column 81.5%, medial column 60.7%;
    non-displaced 76.7% vs displaced 95.5%; sensitivity 81.8%, specificity 90.0%.
  Rikken QGH et al. Novel values in the radiographic diagnosis of ligamentous Lisfranc injuries. Injury 2022;53(6):2326-32.
    PMID 35279293, doi 10.1016/j.injury.2022.02.044. 47 surgically confirmed ligamentous Lisfranc injuries, bilateral
    weight-bearing films: optimal side-to-side cut-off 2.1 mm for C1M2 diastasis; C1-M2 distance of 2.1 mm or more has a
    high specificity; M1M2 should not be used alone.
  Kennelly H et al. Utility of weight-bearing radiographs compared to CT for subtle Lisfranc injuries in the ED.
    Emerg Med Australas 2019;31(5):741-4. PMID 30780193, doi 10.1111/1742-6723.13237. 117 patients: of negative or
    equivocal weight-bearing radiographs only 12% had a positive CT; bilateral weight-bearing films advocated first line.
  Ross G et al. Plantar ecchymosis sign. J Orthop Trauma 1996;10(2):119-22. PMID 8932671, doi 10.1097/00005131-199602000-00008.
  Pourmorteza M, Vosoughi AR. Lisfranc fleck sign. Eur J Trauma Emerg Surg 2022;48(1):471-9. PMID 32561962,
    doi 10.1007/s00068-020-01415-z. "Avulsion of the Lisfranc ligament, fleck sign, results in unstable Lisfranc injuries."
  Pearsall C et al. Defining operative indications in Lisfranc injuries: a systematic review. Foot Ankle Spec
    2024;17(6):632-8 (epub 2023). PMID 37278226, doi 10.1177/19386400231175376. Diastasis of 2 mm or more was the most
    used operative cut-off (35 of 58 studies, 60.4%).
  Diagram (schematic, labelled, AP view, toes up, medial on the left): navicular, medial (C1), intermediate (C2) and
  lateral (C3) cuneiforms, cuboid, metatarsals 1-5. C2 is the shortest cuneiform, so the MT2 base is recessed between
  C1 and C3 (the keystone). Lisfranc ligament: lateral C1 to medial MT2 base. Normal: the medial border of the MT2 base
  lines up with the medial border of C2. Injury: MT2-5 shift laterally, C1-M2 gap widens, fleck fragment in the gap.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(44.0, [(7.0, 1.2), (10.0, 1.4), (14.6, 2.0), (20.0, 1.6), (23.2, 1.8), (28.0, 1.6), (31.0, 1.6),
                     (35.2, 1.2), (37.6, 1.8), (41.0, 1.4), (43.4, 1.4)])

XC = 510
BONE = (200, 222, 242)
SS = 2

def shape_layer(w, h, polys, fill, outline=ICE, ow=4):
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

# ------------------------------------------------------------------ the midfoot, AP view (local canvas 900 x 1000)
FW, FH = 900, 1000
def mt(xc, yb, yh, wb, ws, wh, dx=0):
    """metatarsal: flared base at yb, slim shaft, rounded head at yh; the shaft fans out by dx toward the head"""
    hx = xc + dx
    def sx(y, side):   # shaft edge x at height y (linear from base flare to the neck)
        u = (yb - 95 - y) / max(1, (yb - 95) - (yh + 60))
        return xc + dx * u + side * ws / 2
    pts = [(xc - wb / 2, yb), (xc + wb / 2, yb), (xc + wb / 2, yb - 40), (sx(yb - 95, 1), yb - 95), (sx(yh + 60, 1), yh + 60)]
    pts += [(hx + wh / 2 * math.cos(math.radians(a_)), yh + wh / 2 - wh / 2 * math.sin(math.radians(a_)) + 4) for a_ in range(-20, 201, 20)]
    pts += [(sx(yh + 60, -1), yh + 60), (sx(yb - 95, -1), yb - 95), (xc - wb / 2, yb - 40)]
    return pts
MT1 = mt(250, 552, 190, 110, 62, 92, -22)
MT2 = mt(368, 602, 110, 76, 40, 58, -4)          # base x 330-406: medial border in line with C2 (x 330)
MT3 = mt(468, 568, 135, 70, 38, 56)
MT4 = mt(560, 590, 170, 68, 36, 54)
MT5 = mt(650, 600, 240, 80, 38, 52)
MT5T = [(640, 600), (690, 600), (738, 612), (732, 566), (690, 540), (650, 548)]
C1 = [(195, 562), (316, 562), (318, 772), (200, 776), (190, 700)]
C2 = [(330, 612), (410, 612), (412, 768), (330, 768)]
C3 = [(424, 578), (512, 578), (514, 768), (422, 768)]
CUB = [(524, 600), (700, 608), (716, 700), (692, 838), (560, 842), (522, 776)]
NAV = [(200, 788), (518, 782), (520, 832), (470, 874), (260, 876), (205, 842)]
LAT = [MT2, MT3, MT4, MT5, MT5T]            # move laterally in the injury
ORDER = [NAV, C1, C2, C3, CUB, MT1, MT2, MT3, MT4, MT5]

def shifted(p, s): return [(x + s, y) for x, y in p]

_fc = {}
def foot_layer(build=1.0, shift=0.0, fleck=0.0, tags=True):
    key = (round(build, 2), round(shift, 1), round(fleck, 2), tags)
    if key in _fc: return _fc[key]
    im = Image.new('RGBA', (FW, FH))
    # faint talar head behind the navicular
    n = len(ORDER)
    for i, p in enumerate(ORDER):
        a = clamp(build * n - i * 0.8)
        if a <= 0: continue
        polys = [p] + ([MT5T] if p is MT5 else [])
        if any(p is q for q in LAT): polys = [shifted(q, shift) for q in polys]
        lay = shape_layer(FW, FH, polys, BONE, ICE, 4)
        if a < 1: lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
        im.alpha_composite(lay)
    if fleck > 0:
        fl = shape_layer(FW, FH, [[(319, 598), (334, 594), (339, 610), (330, 620), (318, 614)]], BONE, WARN, 3)
        fl.putalpha(fl.getchannel('A').point(lambda v: int(v * fleck)))
        im.alpha_composite(fl)
    if tags and build > 0.95:
        for txt, x, y in (('C1', 256, 670), ('C2', 371, 690), ('C3', 468, 672), ('CUBOID', 618, 722), ('NAVICULAR', 420, 832)):
            s_ = shift if txt in () else 0
            lab = tl(txt, 'MontBlack', 30 if len(txt) < 4 else 28, NAVY)
            put(im, lab, x + s_, y, 1)
        for k, (x, y) in enumerate(((247, 430), (367, 440), (469, 410), (563, 430), (655, 440))):
            lab = tl(str(k + 1), 'MontBlack', 34, NAVY)
            put(im, lab, x + (shift if k else 0), y, 1)
    if len(_fc) > 80: _fc.clear()
    _fc[key] = im
    return im

FX0, FY0 = -20, 300                       # screen offset of the full-size canvas
def F2S(p, s=0.0): return (FX0 + p[0] + s, FY0 + p[1])

def ligament(d, a, shift, torn, col):
    """Lisfranc ligament: lateral C1 -> medial MT2 base (plantar, oblique)"""
    if a <= 0: return
    p0, p1 = F2S((300, 650)), F2S((352, 618), shift)
    if torn <= 0:
        d.line([p0, p1], fill=col + (int(255 * a),), width=14)
    else:
        m0 = (p0[0] + (p1[0] - p0[0]) * 0.38, p0[1] + (p1[1] - p0[1]) * 0.38)
        m1 = (p0[0] + (p1[0] - p0[0]) * 0.70, p0[1] + (p1[1] - p0[1]) * 0.70)
        d.line([p0, m0], fill=col + (int(255 * a),), width=10); d.line([m1, p1], fill=col + (int(255 * a),), width=10)

def align_line(d, a, shift, col):
    """medial border of C2 and of the MT2 base: one straight line when normal, a step-off when injured"""
    if a <= 0: return
    x_c2 = FX0 + 330; x_m2 = FX0 + 330 + shift
    for y in range(FY0 + 500, FY0 + 780, 24):
        x = x_m2 if y < FY0 + 607 else x_c2
        d.line([(x, y), (x, min(y + 13, FY0 + 780))], fill=col + (int(255 * a),), width=5)

# ------------------------------------------------------------------ scenes
_mini = {}
def mini_foot(shift, fleck, sc):
    key = (round(shift, 1), round(fleck, 2), sc)
    if key not in _mini:
        f = foot_layer(1.0, shift, fleck, tags=False).crop((170, 90, 760, 900))
        _mini[key] = f.resize((int(f.width * sc), int(f.height * sc)), Image.LANCZOS)
        if len(_mini) > 60: _mini.clear()
    return _mini[key]

def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #4", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 430, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Twisted midfoot, X-ray "normal":', 'MontXB', 46, WHITE), W / 2, 545 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('LISFRANC?', 'MontBlack', 140, SKY)
    put(glow, k1, W / 2, 700, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 700, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.45, 0.9))
    put(main, pill('1 IN 5 IS MISSED', 'MontBlack', 38, WHITE, WARN + (255,), h=76), W / 2, 860 + 20 * (1 - q), a * q)
    # the key visual from the first second: the midfoot, gap opening at C1-M2
    g = eio(prog(t, 0.6, 1.5))
    q = eo(prog(t, 0.1, 0.6))
    mf = mini_foot(14 * g, g, 0.5)
    put(main, mf, W / 2, 1110 + 20 * (1 - q), a * q)
    if g > 0:
        gx, gy = W / 2 - mf.width / 2 + (326 - 170) * 0.5, 1110 - mf.height / 2 + (630 - 90) * 0.5
        pul = 0.75 + 0.25 * math.sin(t * 6)
        ImageDraw.Draw(glow).ellipse((gx - 34, gy - 34, gx + 34, gy + 34), fill=WARN + (int(170 * g * pul * a),))

def s2(t, main, glow, fx):                                     # centrepiece: the Lisfranc joint, normal vs injured
    a = win(t, 3.3, 17.0, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('THE LISFRANC JOINT', 'MontBlack', 54, WHITE, track=2), XC, 245, a * q, 1, 8 * (1 - q))
    label(main, 'AP foot  ·  tarsometatarsal joints', XC, 312, a * eo(prog(t, 3.6, 4.1)), SKY, 31, 'MontXB', 0)
    build = prog(t, 3.6, 5.6)
    inj = eio(prog(t, 10.6, 11.6))
    shift = 14 * inj; fleck = eo(prog(t, 11.2, 11.7))
    put(main, foot_layer(build, shift, fleck), FX0 + FW / 2, FY0 + FH / 2, a)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    # ligament
    la = a * eo(prog(t, 5.8, 6.3))
    ligament(d, la, shift, inj, SKY if inj < 0.3 else WARN)
    ligament(g, la * 0.6, shift, inj, SKY if inj < 0.3 else WARN)
    pointer(main, a * win(t, 6.0, 10.5, 0.4, 0.3), F2S((318, 640)), 300, 1235, 'LISFRANC LIGAMENT', SKY, 30)
    pointer(main, a * win(t, 6.4, 10.5, 0.4, 0.3), F2S((368, 570)), 180, 440, 'MT2 BASE', ICE, 30)
    label(main, 'the keystone', 180, 395, a * win(t, 6.5, 10.5, 0.4, 0.3), SUB, 28, 'MontXB', 0)
    # alignment line
    al = a * eo(prog(t, 8.2, 8.7))
    align_line(d, al, shift, GOOD if inj < 0.5 else WARN)
    chip(main, a * eo(prog(t, 8.6, 9.0)), XC, 1320, 'Medial MT2 in line with C2', GOOD, True)
    # injury
    if inj > 0:
        gx, gy = F2S((328, 608))
        pul = 0.75 + 0.25 * math.sin(t * 5)
        g.ellipse((gx - 40, gy - 40, gx + 40, gy + 40), fill=WARN + (int(80 * inj * pul * a),))
    pointer(main, a * eo(prog(t, 11.6, 12.1)), F2S((322, 612)), 220, 1235, 'FLECK SIGN', WARN, 30)
    pointer(main, a * eo(prog(t, 12.0, 12.5)), F2S((330, 560), shift), 180, 440, 'STEP-OFF', WARN, 30)
    label(main, 'gap widens', 180, 395, a * eo(prog(t, 12.1, 12.6)), WARN, 28, 'MontXB', 0)
    chip(main, a * eo(prog(t, 12.4, 12.8)), XC, 1430, 'Injury: C1-MT2 gap + fleck', WARN, False)
    fq = a * eo(prog(t, 13.4, 13.9))
    if fq > 0:
        y = 1590
        label(main, 'About 20% are misdiagnosed', XC, y, fq, WHITE, 38, 'MontBlack', 0)
        label(main, 'can follow minor trauma  ·  Lau et al, EMJ 2017', XC, y + 55, fq, SUB, 28, 'MontXB', 0)

BARS = [('Displaced', 95.5, GOOD), ('Three-column', 92.1, GOOD), ('Non-displaced', 76.7, SKY), ('Medial column', 60.7, WARN)]
def s3(t, main, glow, fx):                                     # why it is missed
    a = win(t, 16.8, 24.2, 0.4, 0.45)
    if a <= 0: return
    label(main, 'WHY IT GETS MISSED', XC, 250, a * eo(prog(t, 16.9, 17.4)), WHITE, 54, 'MontBlack', 2)
    label(main, 'non-weight-bearing foot X-rays', XC, 318, a * eo(prog(t, 17.1, 17.6)), SKY, 32, 'MontXB', 0)
    p = R.eback(prog(t, 17.5, 18.0))
    big = tl('24%', 'MontBlack', 170, WARN)
    put(glow, big, XC, 490, a * 0.3 * clamp(p), clamp(p, 0, 1.15)); put(main, big, XC, 490, a * clamp(p), clamp(p, 0, 1.15))
    label(main, 'of Lisfranc injuries missed', XC, 625, a * eo(prog(t, 17.9, 18.3)), WHITE, 40, 'MontBlack', 0)
    label(main, 'Ponkilainen 2020  ·  100 CT-confirmed sets', XC, 680, a * eo(prog(t, 18.1, 18.5)), SUB, 28, 'MontXB', 0)
    hq = a * eo(prog(t, 20.4, 20.8))
    label(main, 'Recognised on plain films', XC, 800, hq, SKY, 36, 'MontBlack', 1)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    x0, x1 = 110, 760
    for i, (nm, v, col) in enumerate(BARS):
        t0 = 20.6 + 0.5 * i; q = a * eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 930 + 160 * i
        lab = tl(nm, 'MontBlack', 40, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 46, q)
        d.rounded_rectangle((x0, y + 4, x1, y + 40), 18, fill=(60, 90, 125, int(200 * q)))
        p = eio(prog(t, t0 + 0.1, t0 + 0.9))
        xe = x0 + (x1 - x0) * v / 100 * p
        if p > 0:
            d.rounded_rectangle((x0, y, max(x0 + 40, xe), y + 44), 22, fill=col + (int(255 * q),))
            g.rounded_rectangle((x0, y, max(x0 + 40, xe), y + 44), 22, fill=col + (int(70 * q),))
        vt = tl(f'{v * p:.1f}%', 'MontBlack', 48, col); put(main, vt, 870, y + 22, q * clamp(p * 2))
    label(main, 'Chen et al, J Foot Ankle Res 2023  ·  407 patients', XC, 1600, a * eo(prog(t, 22.6, 23.0)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # bilateral weight-bearing films
    a = win(t, 24.0, 32.0, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 24.1, 24.6))
    put(main, tl('LOAD BOTH FEET', 'MontBlack', 58, WHITE, track=2), XC, 250, a * q, 1, 8 * (1 - q))
    label(main, 'bilateral weight-bearing AP views', XC, 318, a * eo(prog(t, 24.3, 24.8)), SKY, 32, 'MontXB', 0)
    bp = eo(prog(t, 24.4, 25.2))
    load = eio(prog(t, 25.8, 26.8))
    sc = 0.6
    d = ImageDraw.Draw(main)
    for k, (cx, sh, nm, col) in enumerate(((290, 0, 'UNINJURED', GOOD), (730, 14 * load, 'INJURED', WARN))):
        mf = mini_foot(sh, 0.0, sc)
        if k == 0: mf = mf.transpose(Image.FLIP_LEFT_RIGHT)       # left foot: mirror image, big toes face the midline
        cy = 420 + mf.height / 2 + 20 * (1 - bp)
        put(main, mf, cx, cy, a * bp)
        # C1-M2 bracket at the joint line
        y = 420 + (632 - 90) * sc
        xl = cx - mf.width / 2 + (317 - 170) * sc; xr = cx - mf.width / 2 + (330 - 170 + sh) * sc
        if k == 0: xl, xr = 2 * cx - xl, 2 * cx - xr
        mq = a * eo(prog(t, 26.6, 27.1))
        if mq > 0:
            c = col + (int(255 * mq),)
            d.line([(xl, y - 70), (xl, y + 70)], fill=c, width=3); d.line([(xr, y - 70), (xr, y + 70)], fill=c, width=3)
            d.line([(xl - 40, y + 60), (xr + 40, y + 60)], fill=c, width=4)
        label(main, nm, cx, 950, a * eo(prog(t, 25.0, 25.4)), col, 36, 'MontBlack', 2)
    label(main, 'measure C1 to the MT2 base', XC, 1010, a * eo(prog(t, 26.6, 27.1)), SUB, 30, 'MontXB', 0)
    # load arrows
    if 25.4 < t < 27.4:
        lq = a * math.sin(math.pi * prog(t, 25.4, 27.4))
        for cx in (290, 730):
            d.line([(cx, 370), (cx, 400)], fill=SKY + (int(255 * lq),), width=8)
            d.polygon([(cx, 418), (cx - 18, 394), (cx + 18, 394)], fill=SKY + (int(255 * lq),))
    cq = a * eo(prog(t, 27.2, 27.7))
    if cq > 0:
        y = 1170 + 20 * (1 - cq / max(a, 1e-3))
        put(main, card(860, 230), XC, y, cq)
        label(main, 'C1-MT2 wider by 2.1 mm or more', XC, y - 62, cq, WHITE, 40, 'MontBlack', 0)
        label(main, 'than the other foot: high specificity', XC, y + 2, cq, GOOD, 34, 'MontXB', 0)
        label(main, 'Rikken et al, Injury 2022  ·  47 operated', XC, y + 62, cq, SUB, 28, 'MontXB', 0)
    nq = a * eo(prog(t, 30.2, 30.7))
    if nq > 0:
        y = 1440 + 20 * (1 - nq / max(a, 1e-3))
        put(main, card(860, 230, fill=(24, 60, 60, 190), outline=GOOD + (200,)), XC, y, nq)
        label(main, 'WB film negative or equivocal?', XC, y - 62, nq, WHITE, 40, 'MontBlack', 0)
        label(main, 'Only 12% then had a positive CT', XC, y + 2, nq, GOOD, 34, 'MontXB', 0)
        label(main, 'Kennelly et al, EMA 2019  ·  117 patients', XC, y + 62, nq, SUB, 28, 'MontXB', 0)

TIPS = [('Bruise on the sole?', 'PLANTAR ECCHYMOSIS SIGN: THINK LISFRANC', 'Ross et al, J Orthop Trauma 1996'),
        ('Fleck in the C1-MT2 space', 'LIGAMENT AVULSION = UNSTABLE INJURY', 'Pourmorteza & Vosoughi, EJTES 2022'),
        ('Diastasis 2 mm or more', 'THE MOST USED CUT-OFF FOR SURGERY', 'Pearsall et al, Foot Ankle Spec 2024')]
def s5(t, main, glow, fx):
    a = win(t, 31.8, 38.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 31.9, 32.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 32.5 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 29, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = ['Lau et al  ·  Emerg Med J 2017', 'Ponkilainen et al  ·  Arch Orthop Trauma Surg 2020',
        'Chen et al  ·  J Foot Ankle Res 2023', 'Rikken et al  ·  Injury 2022', 'Kennelly et al  ·  Emerg Med Australas 2019',
        'Ross et al  ·  J Orthop Trauma 1996', 'Pourmorteza & Vosoughi  ·  EJTES 2022', 'Pearsall et al  ·  Foot Ankle Spec 2024']
def s6(t, main, glow, fx):
    a = win(t, 38.4, 44.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 290, a * eo(prog(t, 38.5, 38.9)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 38.6, 39.1))
    big = tl('LOAD IT', 'MontBlack', 140, WHITE)
    put(glow, big, XC, 440, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 440, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Midfoot pain + "normal" film?', XC, 575, a * eo(prog(t, 38.8, 39.2)), SKY, 38, 'MontXB', 0)
    label(main, 'Bilateral weight-bearing views', XC, 635, a * eo(prog(t, 39.0, 39.4)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 39.4, 39.8))
    put(main, card(880, 560, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1010, a * q)
    for i, r in enumerate(REFS):
        label(main, r, XC, 770 + i * 68, a * q, ICE if i % 2 == 0 else SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1370, a * eo(prog(t, 40.0, 40.4)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
