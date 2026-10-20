"""Orthojets EP04 - CLINICAL RULES #2 - Ottawa ankle rules.
Sources (verified on PubMed, abstracts / PMC full text):
  Stiell IG et al. Ann Emerg Med 1992;21(4):384-90. PMID 1554175, doi 10.1016/s0196-0644(05)82656-3
    Derivation, 750 adults with acute blunt ankle injuries, two university EDs. Original rule: malleolar pain plus bone
    tenderness of the posterior edge or tip of either malleolus, or inability to bear weight both immediately and in the
    ED (and age 55 or more). Midfoot: tenderness at the base of the fifth metatarsal, cuboid or navicular.
  Stiell IG et al. JAMA 1993;269(9):1127-32. PMID 8433468, doi 10.1001/jama.269.9.1127
    Refinement + prospective validation. Refined rules: sensitivity 1.0 for 50 malleolar zone and 19 midfoot zone
    fractures; potential reduction in radiography 34% (ankle series) and 30% (foot series), "by one third".
  Bachmann LM et al. BMJ 2003;326(7386):417. PMID 12595378, doi 10.1136/bmj.326.7386.417
    Systematic review: 27 studies, 15,581 patients. Pooled negative LR 0.08 (ankle) and 0.08 (midfoot); children 0.07.
    At 15% fracture prevalence, under 1.4% probability of fracture when negative. Sensitivity almost 100%, modest
    specificity; should reduce unnecessary radiographs by 30-40%. Under 15% of ED ankle-sprain patients have a fracture,
    yet most undergo radiography.
  Tiemstra JD. Am Fam Physician 2012;85(12):1170-6. PMID 22962897 (refined rule as used today: pain in malleolar or
    midfoot zone and bone tenderness at lateral malleolus, medial malleolus, base of 5th metatarsal or navicular, or
    inability to bear weight for four steps immediately after the injury and in the ED).
  Carvalho MO et al. Rev Bras Ortop 2025;60(1). PMID 40041750, PMC11879485, doi 10.1055/s-0044-1800938
    (full text: malleolar tenderness at the posterior region of the distal 6 cm; four steps; rule contraindicated with
    altered mental status, drug intoxication, language barriers or significant oedema limiting bone palpation.)
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.5, [(7.8, 1.8), (14.5, 1.5), (19.6, 1.8), (26.8, 2.0), (33.8, 2.2), (41.0, 2.2), (45.6, 2.0)])

# ------------------------------------------------------------------ foot skeleton, schematic, local units 900 x 520
# heel on the left, toes on the right; ground y ~ 505.  Foot length ~ 790 units = ~26 cm, so 1 cm ~ 30 units.
SS = 3; LW, LH = 900, 520
CM = 30.0

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
def poly_mask(pts):
    m = Image.new('L', (LW * SS, LH * SS)); ImageDraw.Draw(m).polygon([S(p) for p in pts], fill=255); return m

BONE = (226, 238, 250)
def bone_layer(pts, a=235, sharp=False):
    mask = poly_mask(pts if sharp else catmull(pts))
    lay = Image.new('RGBA', mask.size, BONE + (0,))
    inner = mask.filter(ImageFilter.GaussianBlur(9 * SS / 3))
    al = np.asarray(mask, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.55 + 0.45 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', mask.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: 255 if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

# --- lateral view (right foot from the outside)
LAT = {
    'tibia':   [(268, 0), (356, 0), (360, 190), (376, 262), (378, 298), (336, 308), (292, 304), (262, 290), (258, 190)],
    'fibula':  [(214, 0), (246, 0), (248, 190), (258, 250), (266, 316), (258, 368), (240, 394), (222, 384), (208, 336), (206, 262), (210, 190)],
    'talus':   [(266, 322), (300, 304), (352, 306), (390, 322), (424, 346), (456, 356), (466, 382), (444, 400), (398, 396), (352, 410), (300, 412), (266, 396), (256, 360)],
    'calc':    [(96, 398), (72, 430), (66, 470), (86, 500), (160, 506), (252, 500), (330, 490), (372, 472), (388, 444), (366, 422), (326, 414), (296, 418), (258, 402), (200, 390), (140, 386)],
    'navic':   [(462, 360), (492, 352), (512, 374), (500, 396), (468, 394)],
    'cuboid':  [(378, 428), (444, 414), (470, 432), (466, 474), (424, 488), (384, 476)],
    'mts':     [(470, 396), (500, 372), (540, 376), (700, 430), (770, 444), (840, 462), (846, 482), (806, 488), (694, 478), (520, 446), (478, 430)],
    'mt5':     [(446, 448), (472, 436), (504, 446), (640, 476), (668, 478), (680, 494), (662, 508), (630, 504), (506, 496), (446, 506), (428, 492)],
    'ph5':     [(684, 482), (726, 488), (756, 496), (752, 508), (720, 506), (684, 504)],
}
# --- medial view (left foot from the inside: heel left, toes right)
MED = {
    'tibia':   [(240, 0), (356, 0), (360, 190), (376, 262), (376, 300), (334, 306), (300, 302), (288, 336), (268, 354), (246, 342), (236, 292), (232, 190)],
    'talus':   [(286, 318), (320, 304), (362, 306), (398, 322), (432, 348), (462, 360), (468, 388), (446, 402), (402, 398), (352, 410), (302, 412), (270, 398), (264, 364)],
    'calc':    [(96, 404), (72, 434), (68, 472), (88, 500), (162, 506), (262, 498), (330, 484), (360, 458), (342, 430), (300, 413), (242, 405), (150, 396)],
    'navic':   [(466, 356), (500, 346), (524, 362), (534, 398), (522, 424), (488, 416), (472, 396)],
    'cunei':   [(528, 356), (572, 364), (588, 408), (574, 440), (534, 432), (538, 400)],
    'mt1':     [(590, 382), (604, 372), (722, 420), (748, 440), (742, 470), (716, 472), (602, 452), (590, 436)],
    'ph1':     [(752, 440), (822, 452), (858, 470), (848, 488), (800, 484), (752, 472)],
}
DRAW_ORDER = {'lat': ['calc', 'mts', 'cuboid', 'navic', 'mt5', 'ph5', 'talus', 'tibia', 'fibula'],
              'med': ['calc', 'talus', 'navic', 'cunei', 'mt1', 'ph1', 'tibia']}
DIM = {'mts': 150, 'navic': 140, 'tibia': 210}

def build(view):
    bones = LAT if view == 'lat' else MED
    out = []
    for k in DRAW_ORDER[view]:
        out.append((k, bone_layer(bones[k], a=DIM.get(k, 235))))
    return out

SC = 0.98
VW, VH = int(LW * SC), int(LH * SC)
def shrink(im): return im.resize((VW, VH), Image.LANCZOS)
LAYERS = {v: [(k, shrink(im)) for k, im in build(v)] for v in ('lat', 'med')}
# top fade so the leg bones blend out instead of ending in a hard cut
_fade = np.clip(np.linspace(-0.2, 1.6, VH)[:, None] * np.ones((1, VW)), 0, 1).astype(np.float32)
for v in LAYERS:
    for i, (k, im) in enumerate(LAYERS[v]):
        if k in ('tibia', 'fibula'):
            a = np.asarray(im.getchannel('A'), np.float32) * _fade
            im = im.copy(); im.putalpha(Image.fromarray(a.astype(np.uint8))); LAYERS[v][i] = (k, im)

POS = {'lat': (515, 690), 'med': (515, 1225)}
def vp(view, p):
    cx, cy = POS[view]; return cx - VW / 2 + p[0] * SC, cy - VH / 2 + p[1] * SC

def foot_view(t, main, view, a, t0, pos=None):
    cx, cy = pos or POS[view]
    for i, (k, im) in enumerate(LAYERS[view]):
        q = eo(prog(t, t0 + 0.12 * i, t0 + 0.12 * i + 0.5))
        put(main, im, cx, cy + 20 * (1 - q), a * q)
    q = eo(prog(t, t0 + 0.2, t0 + 0.7))
    lab = 'LATERAL VIEW' if view == 'lat' else 'MEDIAL VIEW'
    x0, y0 = vp(view, (720, 150)); y0 += cy - POS[view][1]
    put(main, tl(lab, 'MontXB', 28, SKY, track=3), x0, y0, a * q)

# zones: posterior edge + tip of each malleolus over the distal 6 cm, base of 5th MT, navicular
LAT_EDGE = [(212, 394 - 6 * CM), (209, 260), (210, 320), (222, 370), (240, 394)]
MED_EDGE = [(236, 354 - 6 * CM), (234, 250), (238, 300), (248, 340), (268, 354)]
ZONES = {'A': ('lat', LAT_EDGE), 'B': ('med', MED_EDGE), 'C': ('lat', (440, 497)), 'D': ('med', (512, 400))}
ZLAB = {'A': ('LATERAL MALLEOLUS', (720, 300)), 'B': ('MEDIAL MALLEOLUS', (700, 250)),
        'C': ('BASE OF 5TH METATARSAL', (640, 575)), 'D': ('NAVICULAR', (680, 260))}

def zone(t, main, glow, key, a, t0):
    q = eo(prog(t, t0, t0 + 0.6))
    if q <= 0 or a <= 0: return
    view, geo = ZONES[key]
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    pulse = 0.75 + 0.25 * math.sin((t - t0) * 6)
    if isinstance(geo, list):
        pts = [vp(view, p) for p in catmull(geo, 6, closed=False)]
        n = max(2, int(len(pts) * q))
        d.line(pts[:n], fill=SKY + (int(255 * a),), width=14, joint='curve')
        g.line(pts[:n], fill=CYAN + (int(200 * a * pulse),), width=26, joint='curve')
        # 6 cm bracket behind the posterior edge
        bx = vp(view, (geo[0][0] - 40, 0))[0]
        ytop, ybot = vp(view, geo[0])[1], vp(view, geo[-1])[1]
        bq = eo(prog(t, t0 + 0.5, t0 + 1.0))
        if bq > 0:
            ya = ybot - (ybot - ytop) * bq
            d.line([(bx, ybot), (bx, ya)], fill=ICE + (int(230 * a * bq),), width=4)
            d.line([(bx - 12, ybot), (bx + 12, ybot)], fill=ICE + (int(230 * a * bq),), width=4)
            if bq > 0.98: d.line([(bx - 12, ytop), (bx + 12, ytop)], fill=ICE + (int(230 * a),), width=4)
            put(main, tl('6 cm', 'MontBlack', 34, WHITE), bx - 64, (ytop + ybot) / 2, a * bq)
        tip = pts[len(pts) // 2]
    else:
        x, y = vp(view, geo); r = 26 + 6 * pulse
        d.ellipse((x - 20, y - 20, x + 20, y + 20), fill=SKY + (int(255 * a * q),))
        g.ellipse((x - r, y - r, x + r, y + r), fill=CYAN + (int(220 * a * q * pulse),))
        tip = (x, y)
    # letter badge + label
    name, anc = ZLAB[key]; lx, ly = vp(view, anc)
    lq = eo(prog(t, t0 + 0.3, t0 + 0.8))
    if lq > 0:
        lab = tl(name, 'MontXB', 29, ICE, track=2)
        bw = lab.width + 70
        lx = min(max(lx, bw / 2 + 70), 955 - bw / 2)
        d.line([tip, (tip[0] + (lx - bw / 2 + 30 - tip[0]) * lq, tip[1] + (ly - tip[1]) * lq)], fill=ICE + (int(200 * a * lq),), width=3)
        put(main, card(bw, lab.height + 4, fill=(12, 24, 42, 215), outline=SKY + (160,), r=16), lx, ly, a * lq)
        put(main, tl(key, 'MontBlack', 34, SKY), lx - bw / 2 + 30, ly, a * lq)
        put(main, lab, lx + 20, ly, a * lq)

def sole(d, x, y, ang, col, a, s=1.0, left=False):
    """footprint: sole ellipse + 5 toes, rotated by ang (deg)"""
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    def P(u, v): return (x + (u * ca - v * sa) * s, y + (u * sa + v * ca) * s)
    c = col + (int(255 * a),)
    heel = [P(18 * math.cos(k / 20 * 2 * math.pi), 26 + 30 * math.sin(k / 20 * 2 * math.pi)) for k in range(20)]
    fore = [P(26 * math.cos(k / 20 * 2 * math.pi), -34 + 38 * math.sin(k / 20 * 2 * math.pi)) for k in range(20)]
    d.polygon(heel, fill=c); d.polygon(fore, fill=c)
    d.polygon([P(-16, 20), P(-22, -30), P(20, -30), P(14, 20)], fill=c)
    toes = [(-20, -80, 10), (-4, -86, 8), (9, -83, 7), (20, -77, 6), (28, -68, 5)]
    for u, v, r in toes:
        if left: u = -u
        cx_, cy_ = P(u, v); rr = r * s
        d.ellipse((cx_ - rr, cy_ - rr, cx_ + rr, cy_ + rr), fill=c)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLINICAL RULES  #2', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 480, a * eo(prog(t, 0.0, 0.3)))
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('OTTAWA', 'MontBlack', 162, WHITE)
    put(glow, k1, W / 2, 690, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 690, a * clamp(p), clamp(p, 0, 1.2))
    p2 = R.eback(prog(t, 0.2, 0.65))
    k2 = tl('ANKLE RULES', 'MontBlack', 112, SKY)
    put(main, k2, W / 2, 860, a * clamp(p2), clamp(p2, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('X-ray this ankle, or not?', 'MontXB', 50, WHITE), W / 2, 1040 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, tl('4 SPOTS  +  4 STEPS', 'MontBlack', 52, SKY, track=4), W / 2, 1150 + 25 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # context
    a = win(t, 3.3, 9.0, 0.35, 0.45)
    if a <= 0: return
    p = eo(prog(t, 3.4, 4.0))
    put(main, tl('The ankle sprain in the ED', 'MontBlack', 56, WHITE), W / 2, 250, a * p, 1, 10 * (1 - p))
    p = R.eback(prog(t, 3.8, 4.4))
    big = tl('UNDER 15%', 'MontBlack', 130, SKY)
    put(glow, big, W / 2, 440, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 440, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'of them have a fracture', W / 2, 560, a * eo(prog(t, 4.3, 4.8)), WHITE, 44, 'MontBlack', 1)
    foot_view(t, main, 'lat', a * 0.9, 4.6, pos=(515, 860))
    label(main, 'Yet most of them get an X-ray', W / 2, 1180, a * eo(prog(t, 6.2, 6.7)), ICE, 44, 'MontXB', 0)
    label(main, 'One clinical rule decides who needs one', W / 2, 1275, a * eo(prog(t, 6.8, 7.3)), SKY, 35, 'MontBlack', 0)
    label(main, 'Bachmann et al.  ·  BMJ 2003', W / 2, 1365, a * eo(prog(t, 7.0, 7.5)), SUB, 28, 'MontXB', 1)

def s3(t, main, glow, fx):                                     # ankle series rule
    a = win(t, 8.8, 20.5, 0.4, 0.45)
    if a <= 0: return
    label(main, 'ANKLE X-RAY  IF', W / 2, 250, a * eo(prog(t, 8.9, 9.4)), SKY, 58, 'MontBlack', 6)
    label(main, 'Pain in the malleolar zone  +  ANY of:', W / 2, 335, a * eo(prog(t, 9.1, 9.6)), WHITE, 38, 'MontXB', 0)
    va = a * (1 - eio(prog(t, 15.0, 15.6)))
    if va > 0:
        foot_view(t, main, 'lat', va, 9.2)
        foot_view(t, main, 'med', va, 9.8)
        zone(t, main, glow, 'A', va, 10.9)
        zone(t, main, glow, 'B', va, 12.3)
        label(main, 'Bone tenderness: POSTERIOR EDGE or TIP', W / 2, 1570, va * eo(prog(t, 13.5, 14.0)), WHITE, 35, 'MontBlack', 0)
        label(main, 'of either malleolus, distal 6 cm', W / 2, 1640, va * eo(prog(t, 13.7, 14.2)), SUB, 34, 'MontXB', 0)
    fa = a * eo(prog(t, 15.4, 15.9))
    if fa > 0:
        put(main, pill('OR', 'MontBlack', 44, NAVY, SKY + (255,), h=80), W / 2, 520, fa)
        label(main, "Can't bear weight", W / 2, 650, fa, WHITE, 64, 'MontBlack', 0)
        d = ImageDraw.Draw(main)
        for i in range(4):
            q = eo(prog(t, 16.0 + 0.45 * i, 16.4 + 0.45 * i))
            if q <= 0: continue
            x = 250 + i * 180; y = 1010 + (-40 if i % 2 else 40)
            sole(d, x, y + 20 * (1 - q), 8 if i % 2 else -8, ICE, fa * q, 1.25, left=not i % 2)
            put(main, tl(str(i + 1), 'MontBlack', 64, SKY), x, y + 170, fa * q)
        label(main, '4 STEPS', W / 2, 1360, fa * eo(prog(t, 17.9, 18.4)), SKY, 90, 'MontBlack', 6)
        label(main, 'right after the injury  AND  in the ED', W / 2, 1490, fa * eo(prog(t, 18.4, 18.9)), WHITE, 40, 'MontBlack', 0)
        label(main, 'Stiell et al.  ·  JAMA 1993  (refined rules)', W / 2, 1600, fa * eo(prog(t, 18.8, 19.3)), SUB, 30, 'MontXB', 1)

def s4(t, main, glow, fx):                                     # foot series rule
    a = win(t, 20.3, 28.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'FOOT X-RAY  IF', W / 2, 250, a * eo(prog(t, 20.4, 20.9)), SKY, 58, 'MontBlack', 6)
    label(main, 'Pain in the midfoot zone  +  ANY of:', W / 2, 335, a * eo(prog(t, 20.6, 21.1)), WHITE, 38, 'MontXB', 0)
    foot_view(t, main, 'lat', a, 20.5)
    foot_view(t, main, 'med', a, 20.8)
    zone(t, main, glow, 'C', a, 21.9)
    zone(t, main, glow, 'D', a, 23.2)
    label(main, 'Bone tenderness at C or D', W / 2, 1570, a * eo(prog(t, 24.3, 24.8)), WHITE, 40, 'MontBlack', 0)
    label(main, "or can't bear weight for 4 steps", W / 2, 1640, a * eo(prog(t, 24.8, 25.3)), SUB, 36, 'MontXB', 0)

def stat_card(main, y, a, head, big, bigcol, sub, sub2=None):
    put(main, card(860, 300), 515, y, a)
    label(main, head, 515, y - 105, a, SKY, 28, 'MontXB', 2)
    label(main, big, 515, y - 22, a, bigcol, 76, 'MontBlack', 1)
    label(main, sub, 515, y + 62, a, WHITE, 34, 'MontXB', 0)
    if sub2: label(main, sub2, 515, y + 112, a, SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # evidence
    a = win(t, 27.8, 35.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'DOES IT MISS FRACTURES?', W / 2, 260, a * eo(prog(t, 27.9, 28.4)), SKY, 52, 'MontBlack', 3)
    q = eo(prog(t, 28.4, 28.9))
    stat_card(main, 560 + 30 * (1 - q), a * q, 'STIELL 1993  ·  PROSPECTIVE VALIDATION', '100% sensitive', GOOD,
              '50 malleolar  +  19 midfoot fractures', 'Refined rules, adults, 2 university EDs')
    q = eo(prog(t, 30.0, 30.5))
    stat_card(main, 900 + 30 * (1 - q), a * q, 'POTENTIAL X-RAYS SAVED', 'about 1 in 3', WHITE,
              'Ankle series 34%  ·  Foot series 30%', 'Stiell et al., JAMA 1993')
    q = eo(prog(t, 31.6, 32.1))
    stat_card(main, 1240 + 30 * (1 - q), a * q, 'BACHMANN 2003  ·  15,581 PATIENTS', 'Negative LR 0.08', GOOD,
              'Rule negative: fracture under 1.4%', '27 studies  ·  assuming 15% fracture prevalence')
    label(main, 'Sensitivity almost 100%, modest specificity', W / 2, 1520, a * eo(prog(t, 32.6, 33.1)), ICE, 34, 'MontXB', 0)

TIPS = [('Press the posterior edge', 'DISTAL 6 CM OF EACH MALLEOLUS + THE TIP', 'Stiell et al. 1992 & 1993'),
        ('Works in children too', 'POOLED NEGATIVE LR 0.07 IN CHILDREN', 'Bachmann et al. BMJ 2003'),
        ('Know when not to use it', 'ALTERED MENTAL STATUS  ·  INTOXICATION', 'or swelling that blocks bone palpation')]
def s6(t, main, glow, fx):
    a = win(t, 34.8, 42.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 34.9, 35.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 35.5 + 1.7 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        put(main, tl(f'0{i + 1}', 'MontBlack', 64, SKY), 885, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 46, WHITE); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 41.8, 46.5, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 400, a * eo(prog(t, 41.9, 42.4)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 42.1, 42.7))
    big = tl('4 SPOTS', 'MontBlack', 150, WHITE)
    put(glow, big, W / 2, 560, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 560, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '+  4 STEPS', W / 2, 710, a * eo(prog(t, 42.5, 43.0)), SKY, 84, 'MontBlack', 3)
    label(main, 'All negative: fracture very unlikely', W / 2, 830, a * eo(prog(t, 42.9, 43.4)), GOOD, 42, 'MontBlack', 0)
    q = eo(prog(t, 43.3, 43.9))
    put(main, card(860, 330, fill=(24, 44, 70, 160), outline=(120, 175, 216, 80)), 515, 1120, a * q)
    for i, (au, ref) in enumerate([('Stiell IG, et al.', 'Ann Emerg Med. 1992;21(4):384–90'),
                                   ('Stiell IG, et al.', 'JAMA. 1993;269(9):1127–32'),
                                   ('Bachmann LM, et al.', 'BMJ. 2003;326(7386):417')]):
        y = 1010 + i * 100
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1400, a * eo(prog(t, 44.0, 44.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
