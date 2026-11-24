"""Orthojets EP26 - CLINICAL RULES #7 - NEXUS low-risk criteria: clearing the C-spine without imaging.
Sources (verified on PubMed abstracts / metadata, 2026-11-24):
  Hoffman JR et al. Validity of a set of clinical criteria to rule out injury to the cervical spine in patients with
    blunt trauma. N Engl J Med 2000;343(2):94-9. PMID 10891516, doi 10.1056/NEJM200007133430203.
    Prospective, 21 US centres, 34,069 blunt trauma patients who had C-spine radiography. Low risk = all 5 criteria:
    no midline cervical tenderness, no focal neurologic deficit, normal alertness, no intoxication, no painful
    distracting injury. Identified all but 8 of 818 injured patients: sensitivity 99.0% (95% CI 98.0-99.6),
    NPV 99.8%, specificity 12.9%, PPV 2.7%. Only 2 low-risk patients had a clinically significant injury
    (sensitivity 99.6%, NPV 99.9%); 1 of them had surgery. Imaging avoidable in 4309 (12.6%).
  Stiell IG et al. The Canadian C-spine rule versus the NEXUS low-risk criteria in patients with trauma.
    N Engl J Med 2003;349(26):2510-8. PMID 14695411, doi 10.1056/NEJMoa031375.
    8283 alert, stable patients, 169 important injuries. Sensitivity CCR 99.4% vs NEXUS 90.7%; missed 1 vs 16.
  Viccellio P et al. A prospective multicenter study of cervical spine injury in children. Pediatrics
    2001;108(2):e20. PMID 11483830, doi 10.1542/peds.108.2.e20.
    3065 NEXUS patients under 18, 30 with CSI: sensitivity 100% (95% CI 87.8-100), 603 low risk, NPV 100%.
    Only 88 children under 2 years: caution in infants and toddlers.
  Denver D et al. Falls and Implementation of NEXUS in the Elderly (FINE). J Emerg Med 2015;49(3):294-300.
    PMID 26022935, doi 10.1016/j.jemermed.2015.03.005. Over 65, fall from standing height or less, all CT:
    169 patients, sensitivity 88.9% for clinically significant CSI; NEXUS not valid in this group.
  Paykin G et al. NEXUS criteria to rule out cervical spine injury among older patients: a systematic review.
    Emerg Med Australas 2018;30(4):450-5. PMID 29232770, doi 10.1111/1742-6723.12880.
    7 studies, patients 65 or older: sensitivity 66% to 100%.
  Diagrams (schematic, labelled): 1) lateral C-spine radiograph (occiput, C1 arches, dens, C2-C7 bodies, articular
  pillars, laminae, spinous processes with C7 the longest) inside the neck soft-tissue outline, a finger pressing the
  posterior midline over the spinous processes, the cord glowing for "focal neurologic deficit"; 2) dot grid of the 818
  injured patients, 8 missed in coral; 3) tips cards; 4) takeaway with references.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(34.0, [(7.0, 1.6), (8.6, 1.4), (10.9, 1.2), (12.2, 2.4),
                     (16.8, 1.2), (18.2, 1.8), (19.3, 1.6), (20.4, 2.4),
                     (23.6, 2.0), (25.2, 2.2), (26.8, 2.6),
                     (30.6, 4.6)])

XC = 510
SS = 2
OX, OY, CW, CH = 60, 350, 900, 780          # diagram canvas (screen x 60-960, y 350-1130)

def rot(pts, th, cx, cy):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]

def ell(cx, cy, rx, ry, n=40):
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]

def rrect(u0, v0, u1, v1, r, n=6):
    pts = []
    for (cx, cy, a0) in ((u1 - r, v0 + r, -90), (u1 - r, v1 - r, 0), (u0 + r, v1 - r, 90), (u0 + r, v0 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n); pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def band(pts, w):
    """thick polyline as polygon (round-ish ends via circles added by caller)"""
    L, Rr = [], []
    for i, (x, y) in enumerate(pts):
        x0, y0 = pts[max(0, i - 1)]; x1, y1 = pts[min(len(pts) - 1, i + 1)]
        dx, dy = x1 - x0, y1 - y0; n = math.hypot(dx, dy) or 1
        nx, ny = -dy / n * w / 2, dx / n * w / 2
        L.append((x + nx, y + ny)); Rr.append((x - nx, y - ny))
    return L + Rr[::-1]

def mask_of(polys):
    m = Image.new('L', (CW * SS, CH * SS)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([((x - OX) * SS, (y - OY) * SS) for x, y in p], fill=255)
    return m.filter(ImageFilter.GaussianBlur(0.8))

def down(im): return im.resize((CW, CH), Image.LANCZOS)

# ------------------------------------------------------------------ lateral C-spine geometry (screen coordinates)
def bx(y): return 600 + 28 * math.sin(math.pi * clamp((y - 470) / 560))
def tilt(y): return -math.degrees(math.atan(28 * math.pi / 560 * math.cos(math.pi * clamp((y - 470) / 560))))
LEVELS = [('C2', 585, 120, 80, 105), ('C3', 680, 124, 62, 58), ('C4', 762, 128, 62, 62),
          ('C5', 844, 132, 62, 70), ('C6', 926, 136, 62, 92), ('C7', 1008, 140, 62, 128)]
SPIN_TIPS = {}

def vertebra(name, cy, w, h, L):
    cx = bx(cy); th = tilt(cy); P = []
    body = rrect(-w / 2, -h / 2, w / 2, h / 2, 12)
    if name == 'C2':                                   # dens rising from the C2 body
        body = rrect(-w / 2, -h / 2, w / 2, h / 2, 14)
        P.append(rot(rrect(-30, -h / 2 - 92, 22, -h / 2 + 8, 20), th, cx, cy))
    P.append(rot(body, th, cx, cy))
    ub = -w / 2
    usl = ub - 0.95 * w                                                                       # spinolaminar line
    big = name in ('C2', 'C7')
    t1 = h * (0.36 if big else 0.28)
    tipu, tipv = usl - L, 0.08 * h + L * 0.62
    post = [(ub + 2, -h * 0.36), (ub - 0.16 * w, -h * 0.40), (ub - 0.30 * w, -h * 0.62), (ub - 0.70 * w, -h * 0.40),
            (usl, -h * (0.22 if big else 0.14)), (tipu, tipv - t1 / 2), (tipu, tipv + t1 / 2),
            (usl + 6, h * (0.46 if big else 0.36)), (ub - 0.70 * w, h * 0.50), (ub - 0.45 * w, h * 0.50),
            (ub - 0.14 * w, h * 0.12), (ub + 2, h * 0.06)]
    P.append(rot(post, th, cx, cy))
    tx, ty = rot([(tipu, tipv)], th, cx, cy)[0]
    P.append(ell(tx, ty, t1 / 2 + 2, t1 / 2 + 2, 20))
    SPIN_TIPS[name] = (tx, ty)
    return P

def c1_parts():
    P = []
    P.append(ell(648, 495, 24, 30))                                  # anterior arch
    P.append(rrect(470, 468, 560, 528, 18))                           # lateral mass over the dens
    P.append(band([(476, 502), (420, 506), (370, 510)], 24))          # posterior arch
    P.append(ell(342, 510, 32, 26))                                   # posterior tubercle
    return P

def skull_parts():
    o = [(445, 440), (395, 434), (330, 424), (265, 408), (205, 390), (165, 378)]
    c = [(616, 422), (652, 404), (692, 390), (728, 380)]
    return [band(o, 28), ell(*o[0], 14, 14, 16), ell(*o[-1], 14, 14, 16), band(c, 30), ell(*c[0], 15, 15, 16),
            ell(*c[-1], 15, 15, 16)]

_cache = {}
CORD_PTS = []
def layers():
    if _cache: return _cache
    # neck soft tissue (volume, behind everything)
    post = [(190, 350), (165, 420), (148, 520), (138, 640), (132, 780), (130, 920), (134, 1060), (142, 1130)]
    ant = [(840, 1130), (822, 980), (806, 840), (796, 720), (796, 620), (812, 560), (850, 520), (884, 470), (896, 400),
           (890, 350)]
    neck = post + ant
    m = mask_of([neck]); a = np.asarray(m, np.float32)
    yy = np.linspace(0, 1, a.shape[0])[:, None]
    fade = np.clip(np.minimum(yy / 0.12, (1 - yy) / 0.10), 0, 1)
    m = Image.fromarray((a * fade).astype(np.uint8))
    _cache['neck'] = down(volume(m, (52, 84, 122), 0.75, r=40))
    # spinal cord (inside the canal, from foramen magnum to C7)
    pts = []
    for y in range(420, 1061, 20):
        yc = max(y, 540); cx = bx(yc); th = math.radians(tilt(yc))
        w = 130; u = -w / 2 - 0.475 * w
        pts.append((cx + u * math.cos(th), y + u * math.sin(th) * (1 if y >= 540 else 0)))
    CORD_PTS[:] = pts
    cm = mask_of([band(pts, 44), ell(*pts[0], 22, 22, 20), ell(*pts[-1], 22, 22, 20)])
    _cache['cord'] = down(volume(cm, (250, 214, 120), 1.0, r=10))
    # bones, one layer per level (for the build-in)
    lev = [('skull', skull_parts()), ('C1', c1_parts())] + [(n, vertebra(n, cy, w, h, L)) for n, cy, w, h, L in LEVELS]
    _cache['bones'] = []
    for name, polys in lev:
        mk = mask_of(polys)
        b = Image.blend(bone_xray(mk, (214, 226, 240), 1.0, cortex=7), volume(mk, (200, 214, 232), 1.0, r=9, relief=0.9), 0.45)
        _cache['bones'].append((name, drop_shadow(down(b), 8, 12, 12, 0.55)))
    return _cache

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #7', 'MontXB', 34, NAVY, SKY + (255,), h=70), XC, 450, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Fall, neck pain, alert patient:', 'MontXB', 48, WHITE), XC, 580 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('SCAN IT?', 'MontBlack', 150, SKY)
    put(glow, k1, XC, 755, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, XC, 755, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('or clear the neck at the bedside?', 'MontXB', 50, WHITE), XC, 935 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('5 CRITERIA  ·  34,069 PATIENTS', 'MontBlack', 32, NAVY, GOOD + (255,), h=72), XC, 1105 + 20 * (1 - q), a * q)
    q = eo(prog(t, 1.1, 1.5))
    put(main, tl('NEXUS LOW-RISK CRITERIA', 'MontBlack', 46, SKY, track=3), XC, 1245 + 20 * (1 - q), a * q)
    th_ = thumb(); q = eo(prog(t, 0.2, 0.8))
    put(main, th_, XC, 1490 + 30 * (1 - q), a * q)

_th = {}
def thumb():
    if _th: return _th['im']
    L = layers(); im = Image.new('RGBA', (CW, CH)); im.alpha_composite(L['neck'])
    for _, b in L['bones']: im.alpha_composite(b)
    _th['im'] = im.resize((int(CW * 0.4), int(CH * 0.4)), Image.LANCZOS)
    return _th['im']

CRIT = ['No midline cervical tenderness', 'No focal neurologic deficit', 'Normal alertness', 'No intoxication',
        'No painful distracting injury']
T_CRIT = [6.6, 8.2, 9.2, 9.9, 10.6]
CY0, CDY = 1170, 96

def s2(t, main, glow, fx):
    a = win(t, 3.3, 14.6, 0.35, 0.45)
    if a <= 0: return
    L = layers()
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('NEXUS: ALL 5 MUST BE ABSENT', 'MontBlack', 50, WHITE, track=2), XC, 245, a * q, 1, 8 * (1 - q))
    label(main, 'alert, blunt trauma  ·  lateral C-spine', XC, 312, a * eo(prog(t, 3.6, 4.1)), SKY, 32, 'MontXB', 0)
    # neck + bones build top to bottom
    nq = a * eo(prog(t, 3.5, 4.2))
    put(main, L['neck'], OX + CW / 2, OY + CH / 2, nq)
    cq = a * eo(prog(t, 8.2, 8.7)) * (1 - 0.6 * prog(t, 9.1, 9.6))
    for i, (name, im) in enumerate(L['bones']):
        bq = a * eo(prog(t, 3.7 + 0.22 * i, 4.15 + 0.22 * i))
        if bq > 0: put(main, im, OX + CW / 2, OY + CH / 2 + 12 * (1 - bq / max(a, 1e-3)), bq)
    if cq > 0:
        put(main, L['cord'], OX + CW / 2, OY + CH / 2, cq * (0.80 + 0.12 * math.sin(t * 6)))
        put(glow, L['cord'], OX + CW / 2, OY + CH / 2, cq * 0.55)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    lq = a * eo(prog(t, 5.6, 6.0))
    if lq > 0:
        for txt, x, y in (('C1', 720, 490), ('C2', 744, 590), ('C7', 790, 1010)):
            label(main, txt, x, y, lq, SKY, 34, 'MontBlack', 1)
        label(main, 'dens', 520, 432, lq * (1 - prog(t, 8.0, 8.3)), ICE, 28, 'MontXB', 1)
        d.line([(548, 444), (572, 470)], fill=ICE + (int(200 * lq * (1 - prog(t, 8.0, 8.3))),), width=3)
    if cq > 0.05:
        label(main, 'CORD', 300, 600, cq, (250, 214, 120), 34, 'MontBlack', 2)
        cx_, cy_ = min(CORD_PTS, key=lambda p: abs(p[1] - 640))
        d.line([(340, 615), (cx_ - 26, cy_)], fill=(250, 214, 120, int(230 * cq)), width=4)
    # finger presses the posterior midline (C5-C6 spinous processes)
    fq = a * eo(prog(t, 6.0, 6.5)) * (1 - prog(t, 8.0, 8.4))
    if fq > 0:
        push = 14 * eio(prog(t, 6.4, 6.8)) * (1 - prog(t, 7.6, 8.0))
        fx0 = 66 + push; fy = 882
        fl = aa_image(260, 120); fd = ImageDraw.Draw(fl)
        fm = Image.new('L', (520, 240)); ImageDraw.Draw(fm).rounded_rectangle((0, 60, 190, 180), 60, fill=255)
        fing = volume(fm.resize((260, 120), Image.LANCZOS), (226, 176, 150), 1.0, r=10)
        fl.alpha_composite(fing)
        fd.rounded_rectangle((60, 40, 86, 76), 12, fill=(246, 214, 200, 230))
        put(main, drop_shadow(fl, 4, 8, 8, 0.5), fx0 + 130, fy, fq)
        tx = SPIN_TIPS['C5'][0]
        pr = prog(t, 6.6, 7.6)
        for k in range(2):
            rr = 18 + 70 * ((pr + 0.5 * k) % 1); al = (1 - ((pr + 0.5 * k) % 1)) * fq
            g.ellipse((tx - 60 - rr, fy - rr, tx - 60 + rr, fy + rr), outline=WARN + (int(200 * al),), width=6)
        label(main, 'PRESS THE MIDLINE', 290, 1085, fq, WARN, 32, 'MontBlack', 1)
        d.line([(230, 625), (SPIN_TIPS['C3'][0] - 14, SPIN_TIPS['C3'][1] - 10)], fill=ICE + (int(200 * fq * eo(prog(t, 6.3, 6.7))),), width=3)
        d.line([(130, 1062), (100, 915)], fill=WARN + (int(220 * fq),), width=4)
        label(main, 'spinous processes', 235, 600, fq * eo(prog(t, 6.3, 6.7)), ICE, 28, 'MontXB', 1)
    # the 5 criteria
    for i, txt in enumerate(CRIT):
        q_ = a * eo(prog(t, T_CRIT[i], T_CRIT[i] + 0.4))
        if q_ <= 0: continue
        y = CY0 + CDY * i + 14 * (1 - q_ / max(a, 1e-3))
        on = t < (T_CRIT[i + 1] if i < 4 else 11.4)
        put(main, card(860, 80, fill=(30, 54, 84, 200), outline=(SKY + (255,)) if on else (GOOD + (170,)), r=28), XC, y, q_)
        check(d, 130, y, 36, GOOD, q_, 8)
        lab = tl(txt, 'MontBlack', 34, WHITE); put(main, lab, 170 + lab.width / 2 - 20, y, q_)
        nm = tl(str(i + 1), 'MontBlack', 34, SKY); put(main, nm, 910, y, q_)
    vq = a * eo(prog(t, 11.6, 12.1))
    if vq > 0:
        y = CY0 + CDY * 5 - 6
        put(main, card(860, 76, fill=GOOD + (70,), outline=GOOD + (255,), r=38), XC, y, vq)
        label(main, 'ALL 5 ABSENT  =  NO IMAGING', XC, y, vq, WHITE, 38, 'MontBlack', 2)
        g.rounded_rectangle((XC - 430, y - 38, XC + 430, y + 38), 38, fill=GOOD + (int(40 * vq),))

# ------------------------------------------------------------------ evidence: dot grid of the 818 injured patients
NCOL, GX, GY, GX0, GY0, DR = 33, 25, 21, 110, 500, 8
MISSED = [57, 160, 298, 371, 455, 540, 662, 777]
_grid = {}
def grid_layer():
    if _grid: return _grid['im']
    S = 3; w = NCOL * GX + 20; h = 25 * GY + 20
    im = Image.new('RGBA', (w * S, h * S)); d = ImageDraw.Draw(im)
    for i in range(818):
        r_, c_ = divmod(i, NCOL); x = 10 + c_ * GX + GX / 2; y = 10 + r_ * GY + GY / 2
        d.ellipse(((x - DR) * S, (y - DR) * S, (x + DR) * S, (y + DR) * S), fill=GOOD + (235,))
    _grid['im'] = im.resize((w, h), Image.LANCZOS)
    return _grid['im']

def s3(t, main, glow, fx):
    a = win(t, 14.4, 22.6, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 14.5, 15.0))
    put(main, tl('THE NEXUS STUDY', 'MontBlack', 60, WHITE, track=2), XC, 255, a * q, 1, 8 * (1 - q))
    label(main, 'NEJM 2000  ·  34,069 patients  ·  21 US centres', XC, 325, a * eo(prog(t, 14.7, 15.2)), SKY, 32, 'MontXB', 0)
    label(main, '818 had a C-spine injury  ·  1 dot = 1 patient', XC, 430, a * eo(prog(t, 15.0, 15.4)), ICE, 32, 'MontXB', 0)
    gl = grid_layer(); p = eio(prog(t, 15.2, 16.7))
    if p > 0:
        hh = max(1, int(gl.height * p))
        put(main, gl.crop((0, 0, gl.width, hh)), GX0 - 10 + gl.width / 2, GY0 - 10 + hh / 2, a)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    mq = a * eo(prog(t, 17.0, 17.4))
    if mq > 0:
        pul = 0.5 + 0.5 * math.sin(t * 7)
        for i in MISSED:
            r_, c_ = divmod(i, NCOL); x = GX0 + c_ * GX + GX / 2; y = GY0 + r_ * GY + GY / 2
            d.ellipse((x - DR - 1, y - DR - 1, x + DR + 1, y + DR + 1), fill=WARN + (int(255 * mq),))
            g.ellipse((x - 18, y - 18, x + 18, y + 18), fill=WARN + (int((60 + 60 * pul) * mq),))
            d.ellipse((x - 17, y - 17, x + 17, y + 17), outline=WARN + (int(200 * mq),), width=3)
        y = 1120
        put(main, tl('MISSED: 8 OF 818', 'MontBlack', 58, WARN, track=2), XC, y, mq)
    sq = a * eo(prog(t, 17.6, 18.1))
    if sq > 0:
        p2 = R.eback(prog(t, 17.6, 18.1))
        big = tl('SENSITIVITY 99.0%', 'MontBlack', 76, GOOD)
        put(glow, big, XC, 1225, sq * 0.3, clamp(p2, 0, 1.15)); put(main, big, XC, 1225, sq, clamp(p2, 0, 1.15))
    cq = a * eo(prog(t, 18.6, 19.1))
    if cq > 0:
        y = 1440 + 20 * (1 - cq / max(a, 1e-3))
        put(main, card(860, 260), XC, y, cq)
        for x, v, nm, col in ((300, 'NPV 99.8%', 'rule it out', GOOD), (720, 'SPEC 12.9%', 'most still imaged', WARN)):
            label(main, v, x, y - 50, cq, col, 54, 'MontBlack', 1)
            label(main, nm, x, y + 20, cq, SUB, 30, 'MontXB', 0)
        d.line([(XC, y - 90), (XC, y + 50)], fill=SKY + (int(90 * cq),), width=3)
        label(main, 'Imaging avoidable: 12.6% (4309)', XC, y + 88, cq * eo(prog(t, 19.4, 19.8)), WHITE, 32, 'MontXB', 0)
    fq = a * eo(prog(t, 19.9, 20.3))
    label(main, 'Clinically significant injuries missed: 2', XC, 1630, fq, ICE, 32, 'MontXB', 0)

TIPS = [('Over 65 and fell? Image it', 'SENSITIVITY ONLY 88.9% (169 PATIENTS)', 'Denver et al, J Emerg Med 2015'),
        ('Canadian rule caught more', 'NEXUS 90.7% VS CCR 99.4%, MISSED 16 VS 1', 'Stiell et al, NEJM 2003'),
        ('Children: yes. Toddlers: careful', '3065 UNDER 18: 100%, BUT ONLY 88 UNDER 2', 'Viccellio et al, Pediatrics 2001')]
def s4(t, main, glow, fx):
    a = win(t, 22.4, 29.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 22.5, 23.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 23.0 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 28, ICE, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 28, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)
    label(main, 'Older adults, 7 studies: sensitivity 66-100% (Paykin 2018)', XC, 1590,
          a * eo(prog(t, 24.0, 24.4)), ICE, 28, 'MontXB', 0)

REFS = ['Hoffman JR et al. N Engl J Med 2000;343:94-9', 'Stiell IG et al. N Engl J Med 2003;349:2510-8',
        'Viccellio P et al. Pediatrics 2001;108:e20', 'Denver D et al. J Emerg Med 2015;49:294-300',
        'Paykin G et al. Emerg Med Australas 2018;30:450-5']
def s5(t, main, glow, fx):
    a = win(t, 29.2, 34.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 380, a * eo(prog(t, 29.3, 29.7)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 29.4, 29.9))
    big = tl('ALL 5 ABSENT', 'MontBlack', 108, WHITE)
    put(glow, big, XC, 530, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 530, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '= no C-spine imaging (alert, blunt trauma)', XC, 655, a * eo(prog(t, 29.6, 30.0)), SKY, 36, 'MontXB', 0)
    label(main, 'Sensitivity 99.0%  ·  NPV 99.8%', XC, 720, a * eo(prog(t, 29.8, 30.2)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 29.9, 30.3))
    put(main, card(880, 380, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1010, a * q)
    for i, ref in enumerate(REFS):
        label(main, ref, XC, 870 + i * 70, a * q, ICE if i == 0 else SUB, 29, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1290, a * eo(prog(t, 30.2, 30.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
