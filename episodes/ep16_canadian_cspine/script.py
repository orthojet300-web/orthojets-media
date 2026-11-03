"""Orthojets EP16 - CLINICAL RULES #4 - Canadian C-spine rule: who needs neck imaging after blunt trauma.
Sources (verified on PubMed abstracts / metadata, 2026-11-03):
  Stiell IG et al. The Canadian C-spine rule for radiography in alert and stable trauma patients.
    JAMA 2001;286(15):1841-8. PMID 11597285, doi 10.1001/jama.286.15.1841.
    Derivation: 8924 adults, blunt head/neck trauma, stable vital signs, GCS 15; 151 (1.7%) important C-spine injuries.
    3 questions: (1) any high-risk factor that mandates radiography (age 65 or older, dangerous mechanism, paresthesias
    in extremities)? (2) any low-risk factor that allows safe assessment of range of motion (simple rear-end MVC,
    sitting position in ED, ambulatory at any time since injury, delayed onset of neck pain, absence of midline C-spine
    tenderness)? (3) able to actively rotate the neck 45 degrees to the left and right?
    Sensitivity 100% (95% CI 98-100), specificity 42.5%, potential radiography rate 58.2%.
  Stiell IG et al. The Canadian C-spine rule versus the NEXUS low-risk criteria in patients with trauma.
    N Engl J Med 2003;349(26):2510-8. PMID 14695411, doi 10.1056/NEJMoa031375.
    8283 alert, stable patients, 9 Canadian EDs, 394 physicians; 169 (2.0%) important injuries.
    Range of motion not evaluated as required in 845 (10.2%). CCR vs NLC: sensitivity 99.4% vs 90.7%,
    specificity 45.1% vs 36.8%, radiography rate 55.9% vs 66.6%. Missed injuries: CCR 1, NLC 16.
  Diagrams (schematic, labelled): 1) the 3-question CCR flowchart building node by node with the outcome of each
  branch; 2) top view of the patient (shoulders, head, nose) actively rotating the head 45 degrees left and right
  against a protractor; 3) CCR vs NEXUS bar comparison.
"""
import os, sys, math
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(41.0, [(6.6, 1.6), (10.2, 2.2), (13.6, 1.6), (19.2, 1.4), (21.2, 1.2), (26.8, 1.6), (28.6, 1.8),
                     (32.3, 1.2), (33.8, 1.2), (35.3, 1.6), (38.6, 2.4)])

XC = 510
LIMB = (58, 92, 132)
SKIN = (92, 128, 170)
SS = 2

def rot(pts, th, cx, cy):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]

def shape_layer(w, h, polys, fill, outline=ICE, ow=5):
    m = Image.new('L', (w * SS, h * SS)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x * SS, y * SS) for x, y in p], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(1.2))
    edge = ImageChops.subtract(m, m.filter(ImageFilter.MinFilter(ow * SS + 1 - (ow * SS) % 2)))
    im = Image.new('RGBA', (w * SS, h * SS), fill + (0,)); im.putalpha(m)
    e = Image.new('RGBA', im.size, outline + (0,)); e.putalpha(edge); im.alpha_composite(e)
    return im.resize((w, h), Image.LANCZOS)

def ell(cx, cy, rx, ry, n=48):
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]

def chip(main, a, x, y, txt, col, ok, w=860, size=34):
    if a <= 0: return
    fill = (24, 60, 60, 200) if ok else (66, 32, 38, 200)
    put(main, card(w, 92, fill=fill, outline=col + (220,), r=30), x, y, a)
    d = ImageDraw.Draw(main)
    (check if ok else cross)(d, x - w / 2 + 58, y, 38, col, a, 8)
    lab = tl(txt, 'MontBlack', size, WHITE); put(main, lab, x - w / 2 + 100 + lab.width / 2 - 20, y, a)

def varrow(d, x, y0, y1, col, a, w=6):
    d.line([(x, y0), (x, y1 - 4)], fill=col + (int(255 * a),), width=w)
    d.polygon([(x, y1), (x - 13, y1 - 22), (x + 13, y1 - 22)], fill=col + (int(255 * a),))

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #4', 'MontXB', 34, NAVY, SKY + (255,), h=70), XC, 450, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Neck pain after a crash,', 'MontXB', 50, WHITE), XC, 580 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('X-RAY?', 'MontBlack', 160, SKY)
    put(glow, k1, XC, 760, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, XC, 760, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('or clear it clinically?', 'MontXB', 56, WHITE), XC, 940 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('3 QUESTIONS  ·  99.4% SENSITIVE', 'MontBlack', 32, NAVY, GOOD + (255,), h=72), XC, 1110 + 20 * (1 - q), a * q)
    q = eo(prog(t, 1.1, 1.5))
    put(main, tl('CANADIAN C-SPINE RULE', 'MontBlack', 46, SKY, track=3), XC, 1250 + 20 * (1 - q), a * q)

# ------------------------------------------------------------------ diagram 1: the CCR flowchart
NX, NW, NH = 345, 470, 132          # node centre x, width, height
NY = [470, 690, 910]
OX = 832                             # outcome pill centre x
NODES = [('QUESTION 1', 'HIGH-RISK FACTOR?'), ('QUESTION 2', 'LOW-RISK FACTOR?'), ('QUESTION 3', 'ROTATES 45° L + R?')]
T_NODE = [3.9, 7.4, 11.0]

def outcome(main, glow, a, x, y, txt, col, w=210):
    if a <= 0: return
    put(main, card(w, 84, fill=col + (60,), outline=col + (255,), r=42), x, y, a)
    label(main, txt, x, y, a, WHITE, 34, 'MontBlack', 2)

def s2(t, main, glow, fx):
    a = win(t, 3.3, 15.0, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('CANADIAN C-SPINE RULE', 'MontBlack', 52, WHITE, track=2), XC, 255, a * q, 1, 8 * (1 - q))
    label(main, 'alert (GCS 15), stable, blunt trauma', XC, 325, a * eo(prog(t, 3.6, 4.1)), SKY, 32, 'MontXB', 0)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    active = 0 if t < T_NODE[1] else (1 if t < T_NODE[2] else 2)
    for i, (top, txt) in enumerate(NODES):
        nq = a * eo(prog(t, T_NODE[i], T_NODE[i] + 0.45))
        if nq <= 0: continue
        y = NY[i] + 16 * (1 - nq / max(a, 1e-3))
        on = (i == active)
        put(main, card(NW, NH, fill=(30, 54, 84, 200), outline=(SKY + (255,)) if on else (SKY + (110,)), r=30), NX, y, nq)
        if on:
            g.rounded_rectangle((NX - NW / 2, y - NH / 2, NX + NW / 2, y + NH / 2), 30, outline=SKY + (int(150 * nq),), width=10)
        label(main, top, NX, y - 32, nq, SKY, 28, 'MontXB', 3)
        label(main, txt, NX, y + 20, nq, WHITE, 38, 'MontBlack', 0)
    # branches
    def harrow(i, word, t0, col):
        q_ = a * eo(prog(t, t0, t0 + 0.4))
        if q_ <= 0: return
        y = NY[i]
        arrow(d, NX + NW / 2 + 8, y, OX - 105 - 6, col, q_)
        label(main, word, (NX + NW / 2 + OX - 105) / 2, y - 34, q_, col, 27, 'MontBlack', 1)
        outcome(main, glow, a * eo(prog(t, t0 + 0.2, t0 + 0.6)), OX, y, 'IMAGE', WARN)
    def down(i, word, t0, col):
        q_ = a * eo(prog(t, t0, t0 + 0.4))
        if q_ <= 0: return
        y0 = NY[i] + NH / 2 + 6; y1 = (NY[i + 1] if i < 2 else 1090) - (NH / 2 if i < 2 else 44) - 6
        varrow(d, NX, y0, y0 + (y1 - y0) * eo(prog(t, t0, t0 + 0.4)), col, q_)
        label(main, word, NX + 70, (y0 + y1) / 2, q_, col, 27, 'MontBlack', 1)
    harrow(0, 'YES', 4.8, WARN)
    down(0, 'NO', 7.0, GOOD)
    harrow(1, 'NO', 8.0, WARN)
    down(1, 'YES', 10.6, GOOD)
    harrow(2, 'UNABLE', 11.6, WARN)
    down(2, 'ABLE', 12.1, GOOD)
    fq = a * eo(prog(t, 12.4, 12.9))
    if fq > 0:
        put(main, card(420, 88, fill=GOOD + (60,), outline=GOOD + (255,), r=44), NX, 1090, fq)
        label(main, 'NO IMAGING', NX, 1090, fq, WHITE, 38, 'MontBlack', 3)
        g.rounded_rectangle((NX - 210, 1046, NX + 210, 1134), 44, fill=GOOD + (int(50 * fq),))
    # detail panel (changes with the active question)
    pq = a * eo(prog(t, 4.3, 4.8))
    if pq <= 0: return
    put(main, card(860, 440), XC, 1440, pq)
    def phase(t0, t1, head, hcol, items, size, gap):
        ph = pq * win(t, t0, t1, 0.35, 0.3)
        if ph <= 0: return
        label(main, head, XC, 1275, ph, hcol, 36, 'MontBlack', 2)
        y0 = 1355
        for k, it in enumerate(items):
            iq = ph * eo(prog(t, t0 + 0.3 + 0.35 * k, t0 + 0.7 + 0.35 * k))
            if iq <= 0: continue
            y = y0 + k * gap
            dot(d, 135, y, 9, hcol, iq)
            lab = tl(it, 'MontXB', size, WHITE); put(main, lab, 165 + lab.width / 2 - 20, y, iq)
    phase(4.4, 7.5, 'HIGH RISK  =  IMAGE', WARN,
          ['Age 65 or older', 'Dangerous mechanism', 'Paresthesias in the extremities'], 40, 82)
    phase(7.5, 11.1, 'LOW RISK  =  SAFE TO TEST ROM', GOOD,
          ['Simple rear-end collision', 'Sitting up in the ED', 'Walking at any time since injury',
           'Delayed onset of neck pain', 'No midline C-spine tenderness'], 33, 56)
    ph = pq * win(t, 11.1, 15.2, 0.35, 0.3)
    if ph > 0:
        label(main, 'ACTIVE ROTATION, 45° EACH WAY', XC, 1275, ph, SKY, 36, 'MontBlack', 1)
        label(main, 'Derivation: 8924 adults, 151 injuries', XC, 1370, ph * eo(prog(t, 11.5, 11.9)), WHITE, 36, 'MontXB', 0)
        label(main, 'Sensitivity 100%', XC, 1460, ph * eo(prog(t, 11.9, 12.3)), GOOD, 56, 'MontBlack', 1)
        label(main, 'Stiell et al  ·  JAMA 2001', XC, 1560, ph * eo(prog(t, 12.2, 12.6)), SUB, 30, 'MontXB', 0)

# ------------------------------------------------------------------ diagram 2: 45 degree rotation, top view
HX, HY = XC, 820            # head centre (top view, patient faces screen-up)
HEAD = ell(0, 0, 112, 128)
NOSE = [(-24, -122), (0, -160), (24, -122)]
EARS = [ell(-112, 6, 20, 34, 20), ell(112, 6, 20, 34, 20)]
_hc = {}
def head_layer(th):
    key = round(th, 1)
    if key in _hc: return _hc[key]
    w, h = 900, 760; ox, oy = 450, 360       # local origin = head centre
    im = Image.new('RGBA', (w, h))
    sh = [(ox + x, oy + y) for x, y in ell(0, 150, 380, 120)]
    im.alpha_composite(shape_layer(w, h, [sh], LIMB))
    parts = [rot(p, th, ox, oy) for p in [NOSE] + EARS + [HEAD]]
    im.alpha_composite(shape_layer(w, h, parts, SKIN))
    if len(_hc) > 80: _hc.clear()
    _hc[key] = im
    return im

def head_angle(t):
    """degrees, negative = patient's left (screen left, viewed from above)"""
    l = eio(prog(t, 16.4, 17.2)) - eio(prog(t, 17.8, 18.4))
    r = eio(prog(t, 18.5, 19.2)) - eio(prog(t, 20.4, 20.9))
    return -45 * l + 45 * r

def s3(t, main, glow, fx):
    a = win(t, 14.9, 22.5, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 15.0, 15.5))
    put(main, tl('THE 45° ROTATION', 'MontBlack', 58, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'only once a low-risk factor is present', XC, 335, a * eo(prog(t, 15.2, 15.7)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 15.3, 16.1))
    th = head_angle(t)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    # protractor: midline and the two 45 degree rays from the head centre
    pa = a * eo(prog(t, 15.8, 16.3))
    if pa > 0:
        R0, R1 = 150, 330
        for ang, col in ((-90, ICE), (-135, SKY), (-45, SKY)):
            c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            for k in range(0, 9):
                u0 = R0 + k * 22; u1 = u0 + 12
                if u1 > R1: break
                d.line([(HX + c * u0, HY + s * u0), (HX + c * u1, HY + s * u1)], fill=col + (int(220 * pa),), width=4)
        d.arc((HX - 300, HY - 300, HX + 300, HY + 300), -135, -45, fill=SKY + (int(140 * pa),), width=4)
        label(main, 'LEFT 45°', HX - 290, HY - 330, pa, SKY, 34, 'MontBlack', 1)
        label(main, 'RIGHT 45°', HX + 280, HY - 330, pa, SKY, 34, 'MontBlack', 1)
    put(main, head_layer(th), HX, HY - 360 + 380 + 20 * (1 - bp), a * bp)
    # live angle readout + sweep arc
    if abs(th) > 1:
        col = GOOD if abs(th) >= 44.5 else ICE
        st, en = (-90 + th, -90) if th < 0 else (-90, -90 + th)
        d.arc((HX - 210, HY - 210, HX + 210, HY + 210), st, en, fill=col + (int(255 * a),), width=10)
        g.arc((HX - 210, HY - 210, HX + 210, HY + 210), st, en, fill=col + (int(120 * a),), width=16)
    vq = a * eo(prog(t, 16.4, 16.8))
    put(main, tl(f'{int(round(abs(th)))}°', 'MontBlack', 90, GOOD if abs(th) >= 44.5 else WHITE), XC, 1170, vq)
    label(main, 'head turned by the patient', XC, 1245, vq, SUB, 28, 'MontXB', 0)
    chip(main, a * eo(prog(t, 19.3, 19.7)), XC, 1380, 'Rotates 45° both ways: no imaging', GOOD, True, size=32)
    chip(main, a * eo(prog(t, 20.4, 20.8)), XC, 1495, 'Cannot rotate 45°: image', WARN, False, size=32)
    label(main, 'Active, by the patient  ·  not passive', XC, 1615, a * eo(prog(t, 20.7, 21.1)), SUB, 30, 'MontXB', 0)

# ------------------------------------------------------------------ evidence: CCR vs NEXUS
METRICS = [('SENSITIVITY', 99.4, 90.7), ('SPECIFICITY', 45.1, 36.8), ('IMAGING RATE', 55.9, 66.6)]
def s4(t, main, glow, fx):
    a = win(t, 22.3, 31.0, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 22.4, 22.9))
    put(main, tl('CCR  vs  NEXUS', 'MontBlack', 60, WHITE, track=2), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Stiell, NEJM 2003  ·  8283 patients', XC, 335, a * eo(prog(t, 22.6, 23.1)), SKY, 34, 'MontXB', 0)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    x0, x1 = 290, 770
    for i, (nm, v1, v2) in enumerate(METRICS):
        t0 = 23.2 + 1.1 * i; qq = a * eo(prog(t, t0, t0 + 0.4))
        if qq <= 0: continue
        y = 470 + 250 * i
        lab = tl(nm, 'MontBlack', 38, WHITE); put(main, lab, 110 + lab.width / 2 - 20, y, qq)
        for k, (who, v, col) in enumerate((('CCR', v1, SKY), ('NEXUS', v2, SUB))):
            yy = y + 66 + 66 * k
            label(main, who, 200, yy, qq, col, 30, 'MontBlack', 1)
            d.rounded_rectangle((x0, yy - 18, x1, yy + 18), 18, fill=(60, 90, 125, int(180 * qq)))
            p = eio(prog(t, t0 + 0.15 + 0.15 * k, t0 + 0.95 + 0.15 * k))
            if p > 0:
                xe = x0 + (x1 - x0) * v / 100 * p
                d.rounded_rectangle((x0, yy - 18, max(x0 + 36, xe), yy + 18), 18, fill=col + (int(255 * qq),))
                if k == 0: g.rounded_rectangle((x0, yy - 18, max(x0 + 36, xe), yy + 18), 18, fill=col + (int(60 * qq),))
            vt = tl(f'{v * p:.1f}%', 'MontBlack', 38, col); put(main, vt, 875, yy, qq * clamp(p * 2))
    mq = a * eo(prog(t, 27.0, 27.5))
    if mq > 0:
        y = 1440 + 20 * (1 - mq / max(a, 1e-3))
        put(main, card(860, 300), XC, y, mq)
        label(main, 'MISSED IMPORTANT INJURIES', XC, y - 105, mq, WHITE, 38, 'MontBlack', 1)
        p = R.eback(prog(t, 27.3, 27.8))
        put(main, tl('1', 'MontBlack', 120, GOOD), 300, y + 10, mq * clamp(p), clamp(p, 0, 1.15))
        p2 = R.eback(prog(t, 27.7, 28.2))
        put(main, tl('16', 'MontBlack', 120, WARN), 720, y + 10, mq * clamp(p2), clamp(p2, 0, 1.15))
        label(main, 'CCR', 300, y + 100, mq, SKY, 32, 'MontBlack', 2)
        label(main, 'NEXUS', 720, y + 100, mq, SUB, 32, 'MontBlack', 2)
        label(main, 'of 169', XC, y + 10, mq, SUB, 32, 'MontXB', 0)

TIPS = [('Age 65+? Image it', 'HIGH-RISK FACTOR, EVEN IF THE NECK IS FINE', 'Stiell et al, JAMA 2001'),
        ("Don't skip the rotation", 'CCR INCOMPLETE IN 10.2%: ROM NOT ASSESSED', 'Stiell et al, NEJM 2003'),
        ('Alert and stable only', 'GCS 15, STABLE VITALS, BLUNT TRAUMA', 'Stiell et al, JAMA 2001')]
def s5(t, main, glow, fx):
    a = win(t, 30.8, 37.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 30.9, 31.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 31.5 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 29, ICE, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 28, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = [('Stiell IG et al.', 'JAMA. 2001;286(15):1841–8'),
        ('Stiell IG et al.', 'N Engl J Med. 2003;349(26):2510–8')]
def s6(t, main, glow, fx):
    a = win(t, 37.0, 41.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 400, a * eo(prog(t, 37.1, 37.5)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 37.2, 37.7))
    big = tl('3 QUESTIONS', 'MontBlack', 108, WHITE)
    put(glow, big, XC, 555, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 555, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'High risk?  Low risk?  Rotate 45°?', XC, 690, a * eo(prog(t, 37.4, 37.8)), SKY, 38, 'MontXB', 0)
    label(main, 'Missed 1 injury vs 16 with NEXUS', XC, 755, a * eo(prog(t, 37.6, 38.0)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 37.7, 38.1))
    put(main, card(860, 300, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1000, a * q)
    for i, (au, ref) in enumerate(REFS):
        y = 930 + i * 125
        label(main, au, XC, y, a * q, ICE, 31, 'MontXB', 0)
        label(main, ref, XC, y + 42, a * q, SUB, 29, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1270, a * eo(prog(t, 37.9, 38.3)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
