"""Orthojets EP05 - CLASSIFIED #1 - Gustilo-Anderson open fracture classification + antibiotic timing.
Sources (verified on PubMed, abstracts / PMC full text):
  Gustilo RB, Anderson JT. J Bone Joint Surg Am 1976;58(4):453-8. PMID 773941 (no DOI in PubMed)
    1,025 open long-bone fractures (673 retrospective + 352 prospective). Type III = severe soft-tissue injury,
    segmental fracture or traumatic amputation. Cephalosporin suggested as the prophylactic antibiotic of choice.
  Gustilo RB, Mendoza RM, Williams DN. J Trauma 1984;24(8):742-6. PMID 6471139, doi 10.1097/00005373-198408000-00009
    87 type III open fractures. IIIA: adequate soft-tissue coverage of the bone despite extensive laceration or flaps,
    or high-energy trauma irrespective of the size of the wound. IIIB: extensive soft-tissue loss with periosteal
    stripping and bone exposure. IIIC: arterial injury requiring repair. Wound sepsis IIIA 4%, IIIB 52%, IIIC 42%;
    amputation 0%, 16%, 42%. Type III: cephalosporin + aminoglycoside (or 3rd-generation cephalosporin).
  Lack WD et al. J Orthop Trauma 2015;29(1):1-6. PMID 25526095, doi 10.1097/BOT.0000000000000262
    137 type III open tibia fractures. Antibiotics beyond 66 min: OR 3.78 for deep infection; wound coverage beyond
    5 days: OR 7.39. Both early 1/36 = 2.8%; both delayed 17/42 = 40.5%.
  Elniel AR, Giannoudis PV. EFORT Open Rev 2018. PMID 29951271, PMC5994617, doi 10.1302/2058-5241.3.170072
    Type I: clean, skin wound under 1 cm, simple fracture pattern. Type II: wound over 1 cm, minimal soft-tissue
    injury, no flaps or avulsions. Limited inter-observer reliability (~60%).
  Brumback RJ, Jones AL. JBJS Am 1994;76(8):1162-6. PMID 8056796, doi 10.2106/00004623-199408000-00006
    245 surgeons, average agreement 60%.
  Indian J Orthop 2018. PMID 30237604, PMC6142800, doi 10.4103/ortho.IJOrtho_319_17
    "the grading is best done at the time of surgery after wound exploration and debridement."
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(47.6, [(7.8, 1.5), (13.2, 1.8), (19.2, 1.8), (28.0, 2.0), (35.6, 1.8), (42.6, 2.0), (46.6, 1.6)])

# ------------------------------------------------------------------ leg diagram (sagittal schematic, anterior = right)
# local units 400 x 900; 1 cm ~ 30 units.  Anteromedial tibia is subcutaneous: anterior cortex x=300, skin x~332.
SS = 2; DW, DH = 400, 900
CM = 30
BONE = (226, 238, 250)
TISSUE = (52, 84, 120)
DARK = (12, 20, 34)
ART = (200, 150, 160)

def catmull(pts, n=8, closed=True):
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
    return out

def S(p): return (p[0] * SS, p[1] * SS)

def bone_img(pts, a=235):
    m = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(m).polygon([S(p) for p in pts], fill=255)
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(7 * SS))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.5 + 0.5 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: 230 if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

ENV = catmull([(70, -20), (200, -20), (334, -20), (336, 300), (334, 600), (326, 920), (200, 920), (120, 920),
               (60, 760), (12, 470), (14, 250), (40, 80)], 10)
def skin_x(y):  # anterior skin line
    return 336 - 10 * max(0, (y - 600) / 320) if y > 600 else 334 + 2 * min(1, max(0, y) / 300)

FRAC = 440   # main fracture level
def tibia_parts(seg=False):
    top = [(242, -30), (300, -30), (300, FRAC - 14), (286, FRAC + 2), (262, FRAC - 8), (242, FRAC + 6)]
    if seg:
        mid = [(242, FRAC + 18), (262, FRAC + 4), (286, FRAC + 14), (300, FRAC + 2), (300, 600), (282, 612), (242, 598)]
        bot = [(242, 612), (282, 626), (300, 614), (300, 930), (242, 930)]
        return [top, mid, bot]
    bot = [(242, FRAC + 18), (262, FRAC + 4), (286, FRAC + 14), (300, FRAC + 2), (300, 930), (242, 930)]
    return [top, bot]
FIB = [(118, -30), (146, -30), (146, 930), (118, 930)]

def compose(kind):
    """kind: base | I | II | IIIA | IIIB | IIIC"""
    im = Image.new('RGBA', (DW * SS, DH * SS)); d = ImageDraw.Draw(im)
    env = [S(p) for p in ENV]
    d.polygon(env, fill=TISSUE + (170,))
    # muscle hint lines in the calf
    for k in range(4):
        y0 = 160 + 150 * k
        d.arc([S((40, y0)), S((240, y0 + 260))], 200, 300, fill=(80, 116, 156, 120), width=3 * SS)
    d.line(env + [env[0]], fill=ICE + (230,), width=5 * SS)
    seg = kind in ('IIIA', 'IIIC')
    im.alpha_composite(bone_img(FIB, 120))
    for p in tibia_parts(seg): im.alpha_composite(bone_img(p))
    # artery: posterior tibial, behind the tibia
    ay = list(range(-20, 930, 10)); apts = [S((210 + 6 * math.sin(y / 70), y)) for y in ay]
    if kind == 'IIIC':
        g1 = [p for p, y in zip(apts, ay) if y < FRAC - 40]; g2 = [p for p, y in zip(apts, ay) if y > FRAC + 50]
        for g in (g1, g2): d.line(g, fill=ART + (230,), width=7 * SS, joint='curve')
        for p in (g1[-1], g2[0]):
            d.ellipse((p[0] - 9 * SS, p[1] - 9 * SS, p[0] + 9 * SS, p[1] + 9 * SS), fill=WARN + (255,))
    else:
        d.line(apts, fill=ART + (200,), width=7 * SS, joint='curve')
    # wound
    if kind in ('I', 'II', 'IIIA', 'IIIC'):
        hw = {'I': 9, 'II': 40, 'IIIA': 150, 'IIIC': 120}[kind]
        y0, y1 = FRAC - hw, FRAC + hw
        sx = skin_x(FRAC) + 6
        if kind == 'I':
            tract = [(sx, y0), (300, FRAC - 3), (300, FRAC + 3), (sx, y1)]
        elif kind == 'II':
            tract = [(sx, y0), (306, FRAC - 18), (300, FRAC - 6), (300, FRAC + 6), (306, FRAC + 18), (sx, y1)]
        else:
            tract = [(sx, y0), (322, y0 + 30), (316, FRAC - 50), (318, FRAC), (314, FRAC + 50), (322, y1 - 30), (sx, y1)]
        d.polygon([S(p) for p in tract], fill=DARK + (255,))
        d.line([S(tract[0]), S(tract[1])], fill=WARN + (255,), width=4 * SS)
        d.line([S(tract[-2]), S(tract[-1])], fill=WARN + (255,), width=4 * SS)
        if kind in ('IIIA', 'IIIC'):   # lifted skin flap, hinged at the lower wound edge
            flap = [(sx - 4, y1), (sx + 60, y1 - 40), (sx + 70, y0 + 90), (sx + 30, y0 + 70)]
            d.polygon([S(p) for p in flap], fill=TISSUE + (220,))
            d.line([S(p) for p in flap + [flap[0]]], fill=WARN + (255,), width=4 * SS)
    if kind == 'IIIB':   # soft tissue lost over the bone: anterior cortex exposed, periosteum stripped
        y0, y1 = FRAC - 130, FRAC + 140
        hole = [(296, y0 - 20), (skin_x(y0) + 8, y0 - 30), (skin_x(y1) + 8, y1 + 30), (296, y1 + 20)]
        d.polygon([S(p) for p in hole], fill=DARK + (255,))
        d.line([S(hole[0]), S(hole[1])], fill=WARN + (255,), width=4 * SS)
        d.line([S(hole[2]), S(hole[3])], fill=WARN + (255,), width=4 * SS)
        for p in tibia_parts(False):
            im.alpha_composite(bone_img(p))
        im2 = Image.new('RGBA', im.size); d2 = ImageDraw.Draw(im2)
        d2.line([S((302, y0 - 10)), S((302, y1 + 10))], fill=WARN + (255,), width=5 * SS)
        im.alpha_composite(im2)
    im = im.resize((DW, DH), Image.LANCZOS)
    f = np.clip(np.minimum(np.linspace(0, 9, DH), np.linspace(9, 0, DH)), 0, 1)[:, None]
    al = np.asarray(im.getchannel('A'), np.float32) * f
    im.putalpha(Image.fromarray(al.astype(np.uint8)))
    return im

KINDS = ['base', 'I', 'II', 'IIIA', 'IIIB', 'IIIC']
DIAG = {k: compose(k) for k in KINDS}
_sc = {}
def diag(kind, s):
    key = (kind, round(s, 3))
    if key not in _sc: _sc[key] = DIAG[kind].resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    return _sc[key]
def dp(cx, cy, s, p):  # local -> screen
    return cx - DW * s / 2 + p[0] * s, cy - DH * s / 2 + p[1] * s

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29, left=False):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx + (lab.width / 2 - 14 if left else -lab.width / 2 + 14)
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def lput(main, im, x0, y, a):  # left-aligned put
    put(main, im, x0 + im.width / 2 - 20, y, a)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #1', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 470, a * eo(prog(t, 0.0, 0.3)))
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('GUSTILO', 'MontBlack', 160, WHITE)
    put(glow, k1, W / 2, 680, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 680, a * clamp(p), clamp(p, 0, 1.2))
    p2 = R.eback(prog(t, 0.2, 0.65))
    k2 = tl('OPEN FRACTURES', 'MontBlack', 86, SKY)
    put(main, k2, W / 2, 845, a * clamp(p2), clamp(p2, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('Grade it right:', 'MontXB', 52, WHITE), W / 2, 1020 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, tl('IIIB  =  52% WOUND SEPSIS', 'MontBlack', 50, WARN, track=2), W / 2, 1125 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.2, 1.7))
    put(main, tl('I  ·  II  ·  IIIA  ·  IIIB  ·  IIIC', 'MontBlack', 44, SKY, track=2), W / 2, 1235 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # context: what an open fracture is
    a = win(t, 3.3, 9.0, 0.35, 0.45)
    if a <= 0: return
    p = eo(prog(t, 3.4, 4.0))
    put(main, tl('OPEN FRACTURE', 'MontBlack', 76, WHITE, track=4), W / 2, 260, a * p, 1, 10 * (1 - p))
    label(main, 'the bone meets the outside world', W / 2, 350, a * eo(prog(t, 3.7, 4.2)), SKY, 38, 'MontXB', 0)
    cx, cy, s = 430, 860, 0.8
    q = eo(prog(t, 3.6, 4.3))
    kind = 'II' if t > 5.0 else 'base'
    put(main, diag('base', s), cx, cy + 20 * (1 - q), a * q)
    if t > 4.8:
        put(main, diag('II', s), cx, cy, a * eo(prog(t, 4.8, 5.3)))
        g = ImageDraw.Draw(glow); wx, wy = dp(cx, cy, s, (330, FRAC))
        pul = 0.6 + 0.4 * math.sin(t * 6)
        g.ellipse((wx - 40, wy - 50, wx + 40, wy + 50), fill=WARN + (int(150 * a * pul * eo(prog(t, 4.8, 5.3))),))
    pointer(main, a * eo(prog(t, 5.3, 5.8)), dp(cx, cy, s, (340, FRAC)), 820, 700, 'WOUND', WARN)
    pointer(main, a * eo(prog(t, 5.6, 6.1)), dp(cx, cy, s, (272, 250)), 820, 520, 'TIBIA')
    pointer(main, a * eo(prog(t, 5.9, 6.4)), dp(cx, cy, s, (334, 700)), 820, 1060, 'SKIN')
    pointer(main, a * eo(prog(t, 6.2, 6.7)), dp(cx, cy, s, (212, 780)), 820, 1200, 'ARTERY')
    label(main, 'Graded by', W / 2, 1380, a * eo(prog(t, 6.6, 7.1)), WHITE, 40, 'MontBlack', 0)
    label(main, 'Wound size  ·  Contamination  ·  Soft tissue', W / 2, 1455, a * eo(prog(t, 6.9, 7.4)), SKY, 33, 'MontXB', 0)
    label(main, 'Gustilo & Anderson 1976  ·  1,025 open fractures', W / 2, 1535, a * eo(prog(t, 7.1, 7.6)), SUB, 28, 'MontXB', 1)

def s3(t, main, glow, fx):                                     # Type I -> Type II
    a = win(t, 8.8, 20.5, 0.4, 0.45)
    if a <= 0: return
    cx, cy, s = 300, 960, 1.0
    q = eo(prog(t, 8.9, 9.5))
    put(main, diag('base', s), cx, cy + 20 * (1 - q), a * q)
    q1 = eo(prog(t, 9.6, 10.1)); q2 = eo(prog(t, 14.4, 15.0))
    put(main, diag('I', s), cx, cy, a * q1 * (1 - q2))
    put(main, diag('II', s), cx, cy, a * q2)
    d = ImageDraw.Draw(main)
    # 1 cm ruler next to the wound
    rq = eo(prog(t, 10.4, 11.0)) * (1 - eo(prog(t, 14.2, 14.6))) + eo(prog(t, 15.4, 16.0))
    if rq > 0:
        hw = 9 if t < 14.6 else 40
        bx = dp(cx, cy, s, (360, 0))[0]
        yt, yb = dp(cx, cy, s, (0, FRAC - hw))[1], dp(cx, cy, s, (0, FRAC + hw))[1]
        col = ICE + (int(240 * a * min(rq, 1)),)
        d.line([(bx, yt), (bx, yb)], fill=col, width=4)
        d.line([(bx - 10, yt), (bx + 10, yt)], fill=col, width=4); d.line([(bx - 10, yb), (bx + 10, yb)], fill=col, width=4)
        lab = 'under 1 cm' if t < 14.6 else 'over 1 cm'
        lt = tl(lab, 'MontBlack', 30, WHITE)
        put(main, card(lt.width + 10, lt.height + 4, fill=(12, 24, 42, 225), outline=SKY + (160,), r=16), bx + 10 + lt.width / 2, yb + 42, a * min(rq, 1))
        put(main, lt, bx + 10 + lt.width / 2, yb + 42, a * min(rq, 1))
    X0 = 550
    for (tag, lines, t0, t1) in [('TYPE I', [('Wound under 1 cm', WHITE), ('Clean', GOOD), ('Simple fracture', ICE)], 9.7, 14.6),
                                 ('TYPE II', [('Wound over 1 cm', WHITE), ('Minimal soft-', ICE), ('  tissue damage', ICE),
                                              ('No flaps or', ICE), ('  avulsions', ICE)], 15.0, 21.0)]:
        ta = a * win(t, t0, t1, 0.45, 0.35)
        if ta <= 0: continue
        p = R.eback(prog(t, t0, t0 + 0.5))
        big = tl(tag, 'MontBlack', 92, SKY)
        put(glow, big, X0 + big.width / 2 - 20, 470, ta * 0.3 * clamp(p)); lput(main, big, X0, 470, ta * clamp(p))
        for i, (ln, col) in enumerate(lines):
            lq = eo(prog(t, t0 + 0.6 + 0.4 * i, t0 + 1.0 + 0.4 * i))
            yy_ = 600 + 66 * i + (12 if i >= 1 else 0) + (14 if i >= 3 else 0) - (8 if i in (2, 4) and len(lines) > 3 else 0)
            lput(main, tl(ln, 'MontBlack' if i == 0 else 'MontXB', 38 if i == 0 else 36, col), X0 + 20 * (1 - lq), yy_, ta * lq)
    label(main, 'Gustilo & Anderson  ·  JBJS Am 1976', W / 2, 1530, a * eo(prog(t, 10.5, 11.0)), ICE, 31, 'MontXB', 1)

TYPES3 = [('IIIA', 'Bone covered', 'by soft tissue', '4%', '0%'),
          ('IIIB', 'Bone exposed', 'periosteum stripped', '52%', '16%'),
          ('IIIC', 'Artery injury', 'needing repair', '42%', '42%')]
def s4(t, main, glow, fx):                                     # Type III subtypes + outcomes
    a = win(t, 20.3, 29.8, 0.4, 0.45)
    if a <= 0: return
    label(main, 'TYPE III', W / 2, 250, a * eo(prog(t, 20.4, 20.9)), WHITE, 72, 'MontBlack', 6)
    label(main, 'high energy, big soft-tissue injury', W / 2, 335, a * eo(prog(t, 20.6, 21.1)), SKY, 34, 'MontXB', 0)
    s = 0.56; cy = 760
    g = ImageDraw.Draw(glow)
    for i, (tag, l1, l2, sep, amp) in enumerate(TYPES3):
        cx = 190 + 325 * i; t0 = 21.0 + 1.3 * i
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        put(main, tl(tag, 'MontBlack', 60, SKY), cx, 440, a * q)
        put(main, diag(tag, s), cx, cy + 20 * (1 - q), a * q)
        pul = 0.6 + 0.4 * math.sin(t * 6)
        hx, hy = dp(cx, cy, s, (212, FRAC) if tag == 'IIIC' else (318, FRAC))
        g.ellipse((hx - 34, hy - 50, hx + 34, hy + 50), fill=WARN + (int(120 * a * q * pul),))
        put(main, tl(l1, 'MontBlack', 31, WHITE), cx, 1050, a * q)
        put(main, tl(l2, 'MontXB', 27, ICE), cx, 1095, a * q)
    for r, (hdr, col_i, t0) in enumerate([('WOUND SEPSIS', 3, 24.9), ('AMPUTATION', 4, 26.2)]):
        hq = eo(prog(t, t0, t0 + 0.4))
        y = 1220 + 190 * r
        label(main, hdr, W / 2, y, a * hq, SKY, 32, 'MontXB', 4)
        for i, row in enumerate(TYPES3):
            nq = R.eback(prog(t, t0 + 0.2 + 0.2 * i, t0 + 0.6 + 0.2 * i))
            val = row[col_i]; col = GOOD if val in ('0%', '4%') else WARN
            put(main, tl(val, 'MontBlack', 70, col), 190 + 325 * i, y + 80, a * clamp(nq), clamp(nq, 0, 1.2))
    label(main, 'Gustilo et al.  ·  J Trauma 1984  ·  87 type III fractures', W / 2, 1640, a * eo(prog(t, 25.2, 25.7)), SUB, 28, 'MontXB', 0)

def stat(main, y, a, head, big, bigcol, sub):
    put(main, card(860, 210), 515, y, a)
    label(main, head, 515, y - 66, a, SKY, 28, 'MontXB', 2)
    label(main, big, 515, y + 2, a, bigcol, 62, 'MontBlack', 1)
    label(main, sub, 515, y + 68, a, WHITE, 30, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # timing: antibiotics + coverage
    a = win(t, 29.6, 37.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'THE CLOCK STARTS AT INJURY', W / 2, 250, a * eo(prog(t, 29.7, 30.2)), SKY, 44, 'MontBlack', 3)
    cx, cy, r = 515, 560, 170
    q = eo(prog(t, 30.0, 30.5))
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    if q > 0:
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=SKY + (int(150 * a * q),), width=6)
        for k in range(12):
            ang = math.radians(k * 30 - 90); r0 = r - 22 if k % 3 == 0 else r - 14
            d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)), (cx + (r - 4) * math.cos(ang), cy + (r - 4) * math.sin(ang))],
                   fill=ICE + (int(200 * a * q),), width=4)
    sw = eio(prog(t, 30.4, 32.4)) * 66
    if sw > 0:
        e1 = -90 + 6 * min(sw, 60)
        d.arc((cx - r + 18, cy - r + 18, cx + r - 18, cy + r - 18), -90, e1, fill=SKY + (int(255 * a),), width=22)
        if sw > 60:
            d.arc((cx - r + 18, cy - r + 18, cx + r - 18, cy + r - 18), -90, -90 + 6 * (sw - 60), fill=WARN + (int(255 * a),), width=22)
            g.arc((cx - r + 18, cy - r + 18, cx + r - 18, cy + r - 18), -90, -90 + 6 * (sw - 60), fill=WARN + (int(200 * a),), width=36)
        ang = math.radians(-90 + 6 * sw)
        d.line([(cx + 100 * math.cos(ang), cy + 100 * math.sin(ang)), (cx + (r - 40) * math.cos(ang), cy + (r - 40) * math.sin(ang))], fill=WHITE + (int(255 * a),), width=8)
        put(main, tl(f'{int(sw)}', 'MontBlack', 84, WARN if sw > 60 else WHITE), cx, cy - 10, a)
        put(main, tl('min', 'MontXB', 32, ICE), cx, cy + 58, a)
    q = eo(prog(t, 32.4, 32.9))
    stat(main, 920 + 30 * (1 - q), a * q, 'ANTIBIOTICS AFTER 66 MIN', 'Odds ratio 3.78', WARN, 'for deep infection')
    q = eo(prog(t, 33.4, 33.9))
    stat(main, 1160 + 30 * (1 - q), a * q, 'WOUND COVER AFTER 5 DAYS', 'Odds ratio 7.39', WARN, 'for deep infection')
    q = eo(prog(t, 34.4, 34.9))
    if q > 0:
        put(main, card(860, 210), 515, 1400 + 30 * (1 - q), a * q)
        y = 1400 + 30 * (1 - q)
        label(main, 'BOTH EARLY', 300, y - 60, a * q, SKY, 28, 'MontXB', 2)
        label(main, '2.8%', 300, y + 15, a * q, GOOD, 76, 'MontBlack', 1)
        label(main, 'vs', 515, y + 15, a * q, ICE, 40, 'MontXB', 0)
        label(main, 'BOTH LATE', 730, y - 60, a * q, SKY, 28, 'MontXB', 2)
        label(main, '40.5%', 730, y + 15, a * q, WARN, 76, 'MontBlack', 1)
    label(main, 'Lack et al.  ·  J Orthop Trauma 2015  ·  137 type III tibias', W / 2, 1600, a * eo(prog(t, 33.0, 33.5)), SUB, 28, 'MontXB', 0)

TIPS = [('Grade it in the OR', 'AFTER DEBRIDEMENT, NOT IN THE ED', 'Surgeons agree only 60% (Brumback 1994)'),
        ('High energy = type III', 'IRRESPECTIVE OF THE WOUND SIZE', 'Gustilo et al. J Trauma 1984'),
        ('Type III: cover Gram-neg', 'CEPHALOSPORIN + AMINOGLYCOSIDE', 'or a 3rd-generation cephalosporin')]
def s6(t, main, glow, fx):
    a = win(t, 36.8, 43.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 36.9, 37.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 37.5 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 44, WHITE); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 43.6, 47.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 380, a * eo(prog(t, 43.7, 44.2)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 43.9, 44.5))
    big = tl('I  ·  II  ·  III', 'MontBlack', 140, WHITE)
    put(glow, big, W / 2, 540, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 540, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Wound  ·  Contamination  ·  Soft tissue', W / 2, 680, a * eo(prog(t, 44.3, 44.8)), SKY, 38, 'MontXB', 0)
    label(main, 'Antibiotics early, cover early', W / 2, 790, a * eo(prog(t, 44.6, 45.1)), GOOD, 46, 'MontBlack', 0)
    q = eo(prog(t, 45.0, 45.6))
    put(main, card(860, 330, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 1080, a * q)
    for i, (au, ref) in enumerate([('Gustilo RB, Anderson JT.', 'J Bone Joint Surg Am. 1976;58(4):453–8'),
                                   ('Gustilo RB, et al.', 'J Trauma. 1984;24(8):742–6'),
                                   ('Lack WD, et al.', 'J Orthop Trauma. 2015;29(1):1–6')]):
        y = 970 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1360, a * eo(prog(t, 45.6, 46.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
