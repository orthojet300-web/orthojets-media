"""Orthojets EP08 - CLASSIFIED #2 - Salter-Harris physeal fractures: the 5 types and which ones change management.
Sources (verified on PubMed abstracts / PMC full text, 2026-10-26):
  Salter RB, Harris WR. J Bone Joint Surg Am 1963;45(3):587-622 (original classification; not indexed with an
    abstract in PubMed). Type definitions as summarised in Cepela DJ et al. Clin Orthop Relat Res 2016;474(11):2531-7,
    PMID 27206505, doi 10.1007/s11999-016-4891-3:
    I = through the physis; II = physis + metaphysis (metaphyseal / Thurston-Holland fragment);
    III = physis + epiphysis, exits through the articular surface; IV = metaphysis + physis + epiphysis (articular);
    V = compression (crush) of the physis.
    Same definitions in the abstract of Brown JH, DeLuca SA. Am Fam Physician 1992;46(4):1180-4, PMID 1414883.
  Checks 2026-10-26: design review ALL PASS (3rd round); independent fact check 13/13 PASS.
  Mann DC, Rajmaira S. J Pediatr Orthop 1990;10(6):713-6. PMID 2250054, doi 10.1097/01241398-199011000-00002
    2,650 long-bone fractures in children 0-16 y; growth plate injuries = 30%.
  Peterson HA et al. J Pediatr Orthop 1994;14(4):423-30. PMID 8077422, doi 10.1097/01241398-199407000-00002
    951 physeal fractures (Olmsted County 1979-88); Salter-Harris type II most common (54%).
  Basener CJ, Mehlman CT, DiPasquale TG. J Orthop Trauma 2009;23(9):663-7. PMID 19897989, doi 10.1097/BOT.0b013e3181a4f25b
    Meta-analysis, 16 studies, 564 distal femoral physeal fractures: growth disturbance 52% overall;
    SH I 36%, SH II 58%, SH III 49%, SH IV 64%. Displaced 65% vs non-displaced 31% (odds x4).
  Sepulveda M et al. EFORT Open Rev 2022;7(4):264-73. PMID 37931413, PMC9069856, doi 10.1530/EOR-21-0110
    Distal femoral physis = 70% of femoral growth. SH II elsewhere has a low rate of growth alteration, in the distal
    femur ~58%: inform the parents. SH III and IV affect the articular surface: anatomic reduction and fixation, CT
    recommended. Physeal damage usually appears within the first 6 months, can be evident up to 2 years: follow up
    to skeletal maturity.
  Venkatadass K et al. Indian J Orthop 2020;55(1):35-46. PMID 33569097, doi 10.1007/s43465-020-00270-9
    Paediatric ankle: low threshold for CT; intra-articular fractures displaced over 2 mm need perfect anatomical
    reduction and stabilisation.
  Diagram: generic distal long-bone end, distal (joint) DOWN, so the SALTR mnemonic reads correctly
  (A = Above = metaphysis, L = Lower = epiphysis).
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(42.6, [(7.0, 1.5), (10.4, 1.4), (13.0, 1.4), (15.6, 1.4), (18.2, 1.4), (20.8, 1.4),
                     (25.6, 1.6), (31.2, 1.8), (37.8, 1.6), (41.4, 1.8)])

# ------------------------------------------------------------------ diagram: distal end of a long bone (local units)
SS = 2; DW, DH = 600, 820
BONE = (226, 238, 250)
CX0 = 300; RX, RY = 222, 212; PHY_T = 390; PHY_B = 420

def dome_y(x):          # lower (articular) contour of the epiphysis
    u = min(abs(x - CX0) / RX, 1.0)
    return PHY_B + RY * (1 - u ** 3) ** (1 / 3)

def phy_mid(x): return 405 + 6 * math.sin(x / 40)

RIGHT = [(400, -20), (402, 150), (430, 230), (480, 310), (512, 365), (522, 392)]
def x_right(y):
    for (x0, y0), (x1, y1) in zip(RIGHT, RIGHT[1:]):
        if y0 <= y <= y1: return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return RIGHT[-1][0]
LEFT = [(2 * CX0 - x, y) for x, y in RIGHT]

def metaphysis_poly():
    return LEFT + [(x, phy_mid(x) - 15) for x in range(84, 520, 12)] + RIGHT[::-1]

def epiphysis_poly():
    xs = list(range(78, 523, 6))
    return [(x, phy_mid(x) + 15) for x in xs] + [(x, dome_y(x)) for x in xs[::-1]]

def physis_poly():
    xs = list(range(80, 521, 6))
    return [(x, phy_mid(x) - 15) for x in xs] + [(x, phy_mid(x) + 15) for x in xs[::-1]]

def opp_poly():         # opposing bone across the joint (faint)
    xs = list(range(90, 511, 6))
    return [(x, dome_y(x) + 26) for x in xs] + [(500, 840), (100, 840)]

def S(p): return (p[0] * SS, p[1] * SS)

def bone_img(pts, a=235):
    m = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(m).polygon([S(p) for p in pts], fill=255)
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(9 * SS))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.45 + 0.55 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: 230 if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

def compose():
    im = Image.new('RGBA', (DW * SS, DH * SS))
    im.alpha_composite(bone_img(opp_poly(), 110))
    im.alpha_composite(bone_img(metaphysis_poly()))
    im.alpha_composite(bone_img(epiphysis_poly()))
    d = ImageDraw.Draw(im)
    d.polygon([S(p) for p in physis_poly()], fill=SKY + (110,))
    # articular cartilage along the joint surface
    d.line([S((x, dome_y(x) + 4)) for x in range(96, 505, 6)], fill=ICE + (200,), width=5 * SS, joint='curve')
    im = im.resize((DW, DH), Image.LANCZOS)
    f = np.clip(np.linspace(0, 7, DH), 0, 1)[:, None]          # fade the cut shaft at the top
    f = f * np.clip(np.linspace(9, 0, DH), 0, 1)[:, None]      # and the opposing bone at the bottom
    al = np.asarray(im.getchannel('A'), np.float32) * f
    im.putalpha(Image.fromarray(al.astype(np.uint8)))
    return im

BASE = compose()
_sc = {}
def diag(s):
    k = round(s, 3)
    if k not in _sc: _sc[k] = BASE.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    return _sc[k]
def dp(cx, cy, s, p): return cx - DW * s / 2 + p[0] * s, cy - DH * s / 2 + p[1] * s

# fracture paths + highlighted fragment per type (local units)
JX3, JX4 = 330, 298
PATHS = {
    'I':   [(x, phy_mid(x)) for x in range(80, 521, 10)],
    'II':  [(x, phy_mid(x)) for x in range(80, 331, 10)] + [(442, 250)],
    'III': [(x, phy_mid(x)) for x in range(80, 331, 10)] + [(JX3 - 6, 520), (JX3 - 8, dome_y(JX3 - 8) + 4)],
    'IV':  [(JX4, dome_y(JX4) + 4), (JX4 + 6, 520), (JX4 + 14, phy_mid(JX4 + 14)), (360, 340), (442, 250)],
}
def frag(kind):
    if kind == 'II':
        return [(330, phy_mid(330))] + [(x_right(y), y) for y in range(250, 391, 10)] + [(522, PHY_T + 4)] + \
               [(x, phy_mid(x)) for x in range(512, 330, -10)] + [(442, 250)]
    if kind == 'III':
        xs = list(range(80, JX3 - 7, 6))
        return [(x, phy_mid(x) + 2) for x in xs] + [(JX3 - 6, 520), (JX3 - 8, dome_y(JX3 - 8))] + [(x, dome_y(x)) for x in xs[::-1]]
    if kind == 'IV':
        xs = list(range(JX4, 523, 6))
        return [(JX4, dome_y(JX4)), (JX4 + 6, 520), (JX4 + 14, phy_mid(JX4 + 14)), (360, 340), (442, 250)] + \
               [(x_right(y), y) for y in range(250, 391, 10)] + [(522, PHY_T + 10)] + [(x, dome_y(x)) for x in xs[::-1]]
    return None

def partial(pts, p):
    if p >= 1: return pts
    L = [0.0]
    for a_, b_ in zip(pts, pts[1:]): L.append(L[-1] + math.dist(a_, b_))
    tgt = L[-1] * max(p, 0); out = [pts[0]]
    for i in range(1, len(pts)):
        if L[i] <= tgt: out.append(pts[i])
        else:
            r = (tgt - L[i - 1]) / max(L[i] - L[i - 1], 1e-6)
            out.append((pts[i - 1][0] + r * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + r * (pts[i][1] - pts[i - 1][1])))
            break
    return out

def draw_type(main, glow, kind, cx, cy, s, a, p_line, p_frag, width=7):
    """overlay: fragment tint + fracture line (drawn progressively) for one Salter-Harris type"""
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    if kind == 'V':
        pts = [dp(cx, cy, s, q) for q in physis_poly()]
        d.polygon(pts, fill=WARN + (int(200 * a * p_line),))
        g.polygon(pts, fill=WARN + (int(120 * a * p_line),))
        for x in (190, 300, 410):                       # axial load arrows from above and below
            for sgn, y0 in ((1, 250), (-1, 560)):
                k = eo(clamp(p_line * 1.4))
                y1 = y0 + sgn * 90 * k
                P0, P1 = dp(cx, cy, s, (x, y0 - sgn * 30)), dp(cx, cy, s, (x, y1))
                d.line([P0, P1], fill=WARN + (int(255 * a * k),), width=max(4, int(9 * s)))
                hw = 22 * s
                d.polygon([(P1[0] - hw, P1[1] - sgn * hw), (P1[0] + hw, P1[1] - sgn * hw), (P1[0], P1[1] + sgn * hw * 0.9)],
                          fill=WARN + (int(255 * a * k),))
    else:
        fr = frag(kind)
        if fr and p_frag > 0:
            d.polygon([dp(cx, cy, s, q) for q in fr], fill=WARN + (int(95 * a * p_frag),))
        pts = [dp(cx, cy, s, q) for q in partial(PATHS[kind], p_line)]
        if len(pts) > 1:
            d.line(pts, fill=(14, 22, 38, int(255 * a)), width=int(width * 1.9 * max(s, 0.6)), joint='curve')
            d.line(pts, fill=WARN + (int(255 * a),), width=max(3, int(width * max(s, 0.6))), joint='curve')
            g.line(pts, fill=WARN + (int(170 * a),), width=int(width * 3 * max(s, 0.6)), joint='curve')
    main.alpha_composite(lay)

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29, left=False):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx + (lab.width / 2 - 14 if left else -lab.width / 2 + 14)
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
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
    put(main, pill('CLASSIFIED  #2', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 450, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl('Growth plate fracture in a child?', 'MontXB', 44, ICE), W / 2, 560, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('SALTER-HARRIS', 'MontBlack', 150, WHITE, 840)
    put(glow, k1, W / 2, 700, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 700, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, tl('I  ·  II  ·  III  ·  IV  ·  V', 'MontBlack', 70, SKY, track=2), W / 2, 850, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('Which ones go to the OR?', 'MontBlack', 56, WHITE), W / 2, 1010 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, pill('III & IV  =  INTO THE JOINT', 'MontBlack', 40, NAVY, WARN + (255,), h=80), W / 2, 1150 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # context: the growth plate
    a = win(t, 3.3, 8.6, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('THE GROWTH PLATE', 'MontBlack', 64, WHITE, track=3), W / 2, 268, a * p, 1, 10 * (1 - p))
    label(main, 'the weak link of a child\'s bone', W / 2, 345, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s = 380, 830, 0.8
    q = eo(prog(t, 3.6, 4.3))
    put(main, diag(s), cx, cy + 20 * (1 - q), a * q)
    g = ImageDraw.Draw(glow); pul = 0.6 + 0.4 * math.sin(t * 5)
    gq = eo(prog(t, 4.6, 5.1))
    if gq > 0:
        g.polygon([dp(cx, cy, s, pp) for pp in physis_poly()], fill=SKY + (int(150 * a * gq * pul),))
    pointer(main, a * eo(prog(t, 4.4, 4.9)), dp(cx, cy, s, (420, 300)), 820, 560, 'METAPHYSIS')
    pointer(main, a * eo(prog(t, 4.7, 5.2)), dp(cx, cy, s, (470, phy_mid(470))), 820, 760, 'PHYSIS', SKY, 32)
    pointer(main, a * eo(prog(t, 5.0, 5.5)), dp(cx, cy, s, (420, 520)), 820, 930, 'EPIPHYSIS')
    pointer(main, a * eo(prog(t, 5.3, 5.8)), dp(cx, cy, s, (440, dome_y(440) + 12)), 820, 1110, 'JOINT')
    q = eo(prog(t, 5.8, 6.3))
    put(main, card(860, 200), 515, 1430 + 25 * (1 - q), a * q)
    y = 1430 + 25 * (1 - q)
    put(main, tl('30%', 'MontBlack', 96, SKY), 250, y, a * q)
    label(main, 'of long-bone fractures', 640, y - 32, a * q, WHITE, 36, 'MontXB', 0)
    label(main, 'in children hit the physis', 640, y + 22, a * q, WHITE, 36, 'MontXB', 0)
    label(main, 'Mann & Rajmaira 1990  ·  2,650 fractures', W / 2, 1600, a * eo(prog(t, 6.2, 6.7)), SUB, 28, 'MontXB', 0)

TYPES = [('I', 'S', 'Straight across', 'Through the physis only', None),
         ('II', 'A', 'Above', 'Physis + metaphysis', ('MOST COMMON: 54%', GOOD)),
         ('III', 'L', 'Lower', 'Physis + epiphysis', ('INTO THE JOINT', WARN)),
         ('IV', 'T', 'Through', 'Metaphysis + physis + epiphysis', ('INTO THE JOINT', WARN)),
         ('V', 'R', 'Rammed', 'Crush of the physis', ('AXIAL COMPRESSION', ICE))]
T_TYPES = 8.6; DT = 2.6
def s3(t, main, glow, fx):                                     # centrepiece: the 5 types drawn one by one
    a = win(t, 8.4, 21.8, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = 515, 800, 0.88
    q = eo(prog(t, 8.5, 9.0))
    put(main, diag(s), cx, cy + 20 * (1 - q), a * q)
    # SALTR strip on top
    for i, (rom, letter, word, desc, extra) in enumerate(TYPES):
        t0 = T_TYPES + DT * i
        on = t >= t0
        col = SKY if (t0 <= t < t0 + DT) else (ICE if on else (90, 110, 140))
        lq = eo(prog(t, 8.6, 9.1))
        put(main, tl(letter, 'MontBlack', 64, col), 515 + (i - 2) * 120, 280, a * lq * (1.0 if on else 0.55))
    for i, (rom, letter, word, desc, extra) in enumerate(TYPES):
        t0 = T_TYPES + DT * i
        ta = a * win(t, t0, t0 + DT + 0.05, 0.3, 0.3) if i < 4 else a * win(t, t0, 21.8, 0.3, 0.4)
        if ta <= 0: continue
        pl = eio(prog(t, t0 + 0.15, t0 + 0.9)); pf = eo(prog(t, t0 + 0.9, t0 + 1.3))
        draw_type(main, glow, rom, cx, cy, s, ta, pl, pf)
        if rom in ('III', 'IV'):                         # pulse where the line breaks the joint surface
            jx = JX3 - 8 if rom == 'III' else JX4
            hx, hy = dp(cx, cy, s, (jx, dome_y(jx) + 4))
            pul = 0.6 + 0.4 * math.sin(t * 6)
            ImageDraw.Draw(glow).ellipse((hx - 40, hy - 40, hx + 40, hy + 40), fill=WARN + (int(170 * ta * pf * pul),))
        p = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        big = tl(f'TYPE {rom}', 'MontBlack', 84, WHITE)
        mn = tl(f'{letter} = {word}', 'MontXB', 40, SKY)
        rw = big.width + 40 + mn.width; bx = 515 - rw / 2 + big.width / 2; mx = 515 + rw / 2 - mn.width / 2
        put(glow, big, bx, 1300, ta * 0.25 * clamp(p)); put(main, big, bx, 1300, ta * clamp(p), clamp(p, 0, 1.2))
        mq = eo(prog(t, t0 + 0.3, t0 + 0.8))
        put(main, mn, mx, 1300, ta * mq)
        dq = eo(prog(t, t0 + 0.6, t0 + 1.1))
        put(main, fit(desc, 'MontBlack', 46, WHITE, 880), W / 2 - 25, 1410 + 15 * (1 - dq), ta * dq)
        if extra:
            eq = eo(prog(t, t0 + 1.0, t0 + 1.4))
            put(main, pill(extra[0], 'MontBlack', 34, NAVY, extra[1] + (255,), h=66), W / 2 - 25, 1515 + 15 * (1 - eq), ta * eq)
    src = 'Peterson et al. 1994  ·  951 physeal fractures' if T_TYPES + DT <= t < T_TYPES + 2 * DT else 'Salter & Harris  ·  JBJS Am 1963'
    label(main, src, W / 2, 1625, a * eo(prog(t, 9.4, 9.9)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # what changes management
    a = win(t, 21.6, 27.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'WHAT CHANGES MANAGEMENT', W / 2, 265, a * eo(prog(t, 21.7, 22.2)), WHITE, 44, 'MontBlack', 1)
    s = 0.5; cy = 610
    for i, (kind, cx, head, col, t0) in enumerate([('II', 285, 'I  ·  II', GOOD, 22.0), ('IV', 745, 'III  ·  IV', WARN, 23.4)]):
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        put(main, diag(s), cx, cy + 20 * (1 - q), a * q)
        draw_type(main, glow, kind, cx, cy, s, a * q, eio(prog(t, t0 + 0.2, t0 + 0.8)), eo(prog(t, t0 + 0.6, t0 + 1.0)), width=8)
        put(main, tl(head, 'MontBlack', 64, col), cx, 860, a * q)
    q = eo(prog(t, 22.6, 23.1))
    if q > 0:
        y = 1130 + 25 * (1 - q)
        put(main, card(400, 300, outline=GOOD + (150,)), 295, y, a * q)
        label(main, 'Joint surface', 285, y - 95, a * q, ICE, 32, 'MontXB', 0)
        label(main, 'INTACT', 285, y - 40, a * q, GOOD, 52, 'MontBlack', 2)
        label(main, 'Undisplaced:', 285, y + 40, a * q, WHITE, 34, 'MontXB', 0)
        label(main, 'cast', 285, y + 90, a * q, WHITE, 34, 'MontBlack', 0)
    q = eo(prog(t, 24.0, 24.5))
    if q > 0:
        y = 1130 + 25 * (1 - q)
        put(main, card(400, 300, outline=WARN + (170,)), 735, y, a * q)
        label(main, 'Joint surface', 745, y - 95, a * q, ICE, 32, 'MontXB', 0)
        label(main, 'BROKEN', 745, y - 40, a * q, WARN, 52, 'MontBlack', 2)
        label(main, 'Anatomic', 745, y + 40, a * q, WHITE, 34, 'MontBlack', 0)
        label(main, 'reduction + fixation', 745, y + 90, a * q, WHITE, 32, 'MontXB', 0)
    label(main, 'CT helps plan III & IV', W / 2, 1380, a * eo(prog(t, 24.6, 25.1)), SKY, 40, 'MontBlack', 1)
    label(main, 'Sepulveda et al.  ·  EFORT Open Rev 2022', W / 2, 1600, a * eo(prog(t, 24.8, 25.3)), SUB, 28, 'MontXB', 0)

BARS = [('I', 36), ('II', 58), ('III', 49), ('IV', 64)]
def s5(t, main, glow, fx):                                     # pitfall: distal femur
    a = win(t, 27.0, 32.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'THE EXCEPTION', W / 2, 275, a * eo(prog(t, 27.1, 27.6)), WARN, 46, 'MontBlack', 6)
    p = eo(prog(t, 27.2, 27.7))
    put(main, tl('DISTAL FEMUR', 'MontBlack', 92, WHITE, track=2), W / 2, 370, a * p, 1, 8 * (1 - p))
    label(main, '70% of femoral growth comes from this physis', W / 2, 465, a * eo(prog(t, 27.5, 28.0)), SKY, 32, 'MontXB', 0)
    label(main, 'GROWTH DISTURBANCE BY TYPE', W / 2, 560, a * eo(prog(t, 27.8, 28.3)), ICE, 32, 'MontXB', 3)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    base_y, hmax = 1140, 480
    q0 = eo(prog(t, 27.9, 28.3))
    d.line([(120, base_y), (910, base_y)], fill=ICE + (int(180 * a * q0),), width=3)
    for i, (rom, v) in enumerate(BARS):
        x = 215 + 200 * i; t0 = 28.2 + 0.35 * i
        k = eo(prog(t, t0, t0 + 0.7))
        if k <= 0: continue
        h = hmax * v / 70 * k
        col = WARN if rom == 'II' else SKY
        hl = rom == 'II' and t > 30.0
        d.rounded_rectangle((x - 60, base_y - h, x + 60, base_y), 14, fill=col + (int(235 * a),))
        if hl:
            g.rounded_rectangle((x - 70, base_y - h - 10, x + 70, base_y + 4), 18, fill=WARN + (int(170 * a * (0.6 + 0.4 * math.sin(t * 6))),))
        put(main, tl(f'{int(round(v * k))}%', 'MontBlack', 54, WHITE), x, base_y - h - 46, a * k)
        put(main, tl(rom, 'MontBlack', 50, col), x, base_y + 52, a * k)
    q = eo(prog(t, 29.8, 30.3))
    if q > 0:
        put(main, pill('TYPE II IS NOT BENIGN HERE', 'MontBlack', 38, NAVY, WARN + (255,), h=76), W / 2, 1300 + 20 * (1 - q), a * q)
    q = eo(prog(t, 30.4, 30.9))
    if q > 0:
        y = 1440 + 20 * (1 - q)
        label(main, 'Displaced 65%  vs  undisplaced 31%', W / 2, y, a * q, WHITE, 40, 'MontBlack', 0)
    label(main, 'Basener et al.  ·  J Orthop Trauma 2009  ·  564 fractures', 515, 1600, a * eo(prog(t, 28.6, 29.1)), SUB, 28, 'MontXB', 0)

TIPS = [('Distal femur: warn family', 'GROWTH DISTURBANCE IN 52%', 'Basener et al. 2009  ·  meta-analysis'),
        ('Joint involved? Get a CT', 'ANKLE: OVER 2 MM, ANATOMIC REDUCTION', 'Venkatadass et al. 2020'),
        ('Follow until maturity', 'ARREST USUALLY IN 6 MONTHS, UP TO 2 YRS', 'Sepulveda et al. 2022')]
def s6(t, main, glow, fx):
    a = win(t, 32.6, 38.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 32.7, 33.2)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 33.2 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 38.6, 42.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 360, a * eo(prog(t, 38.7, 39.2)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 38.9, 39.5))
    big = tl('S · A · L · T · R', 'MontBlack', 110, WHITE)
    put(glow, big, W / 2, 500, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 500, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'III & IV break the joint surface', W / 2, 630, a * eo(prog(t, 39.3, 39.8)), WARN, 42, 'MontBlack', 0)
    label(main, 'Anatomic reduction, follow to maturity', W / 2, 720, a * eo(prog(t, 39.6, 40.1)), GOOD, 38, 'MontBlack', 0)
    q = eo(prog(t, 40.0, 40.6))
    put(main, card(860, 330, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 1010, a * q)
    for i, (au, ref) in enumerate([('Peterson HA, et al.', 'J Pediatr Orthop. 1994;14(4):423–30'),
                                   ('Basener CJ, et al.', 'J Orthop Trauma. 2009;23(9):663–7'),
                                   ('Sepulveda M, et al.', 'EFORT Open Rev. 2022;7(4):264–73')]):
        y = 900 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1290, a * eo(prog(t, 40.6, 41.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
