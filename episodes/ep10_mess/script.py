"""Orthojets EP10 - ORTHO NUMBERS #4 - MESS 7: the Mangled Extremity Severity Score and why 7+ is not a verdict.
Sources (verified on PubMed abstracts / metadata / PMC full text, 2026-10-31):
  Johansen K, Daines M, Howey T, Helfet D, Hansen ST. J Trauma 1990;30(5):568-72. PMID 2342140,
    doi 10.1097/00005373-199005000-00007. MESS = skeletal/soft-tissue damage, limb ischemia, shock and age.
    Retrospective (25 patients): mean MESS 4.88 in 17 salvaged limbs vs 9.11 in 9 amputated limbs (p under 0.01).
    Prospective (2 trauma centres): 4.00 in 14 salvaged vs 8.83 in 12 amputated ("doomed") limbs. In both series a
    MESS of 7 or more predicted amputation with 100% accuracy.
  Helfet DL, Howey T, Sanders R, Johansen K. Clin Orthop Relat Res 1990;(256):80-6. PMID 2194732 (no DOI).
    Four criteria "with increasing points for worsening prognosis": skeletal/soft-tissue injury, limb ischemia,
    shock, patient age. (Point values are NOT shown on screen: not stated in any PubMed abstract / PMC text.)
  Bosse MJ et al. (LEAP). J Bone Joint Surg Am 2001;83(1):3-14. PMID 11205855, doi 10.2106/00004623-200101000-00002.
    556 high-energy lower-extremity injuries, prospective, 5 scores (MESS, LSI, PSI, NISSSA, HFS-97). Did not
    validate the clinical utility of any score. High specificity: low scores predict limb-salvage potential. The
    converse not true: low sensitivity, so not valid as predictors of amputation. Use scores at or above the
    amputation threshold cautiously.
  Ly TV et al. J Bone Joint Surg Am 2008;90(8):1738-43. PMID 18676906, doi 10.2106/JBJS.G.00136. 407 successfully
    reconstructed limbs: none of the 5 scores predicted functional outcome (SIP) at 6 or 24 months.
  Schiro GR et al. BMC Musculoskelet Disord 2015;16:372. PMID 26796522, doi 10.1186/s12891-015-0832-7. Scores
    "cannot be used as the sole criterion by which amputation decision are made".
  (Fact check 2026-10-31: Yoneda found upper-limb MESS accuracy COMPARABLE to the lower limb -> tip says "no better".)
  Yoneda H et al. Injury 2024;55(4):111447. PMID 38417237, doi 10.1016/j.injury.2024.111447. Upper limb, 10 studies,
    338 patients: MESS pooled sensitivity 0.95, specificity 0.81; limb salvage achieved in at least 20% of patients
    above the amputation threshold; should not be used to justify amputation.
  Diagram: AP schematic of a right lower leg (tibia medial = viewer's right, fibula lateral = viewer's left, fibular
  head below the lateral tibial condyle, lateral malleolus lower than the medial malleolus). Popliteal artery
  divides below the knee into the anterior tibial artery (passes laterally, between tibia and fibula) and the
  posterior tibial artery (runs down behind the medial malleolus). Open tibial + fibular shaft fracture, wound over
  the subcutaneous anteromedial tibia, both arteries interrupted at the fracture level.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(44.6, [(9.6, 1.6), (12.0, 1.6), (16.4, 1.6), (19.4, 1.8), (25.0, 1.8), (28.6, 2.0), (35.8, 2.0),
                     (41.8, 1.8)])

# ------------------------------------------------------------------ diagram: AP lower leg (local units)
SS = 2; DW, DH = 560, 1040
BONE = (226, 238, 250)
FEMUR = [(200, -40), (200, 90), (185, 130), (150, 165), (140, 200), (160, 222), (250, 226), (280, 206), (310, 226),
         (400, 222), (420, 200), (410, 165), (375, 130), (360, 90), (360, -40)]
TIBIA = [(150, 240), (165, 290), (215, 340), (250, 390), (255, 700), (250, 860), (235, 900), (230, 935), (330, 937),
         (355, 942), (370, 978), (386, 976), (392, 930), (380, 880), (355, 840), (345, 700), (345, 400), (360, 350),
         (400, 300), (418, 268), (410, 240), (290, 238), (280, 226), (270, 232)]
FIBULA = [(138, 300), (165, 290), (190, 305), (192, 340), (178, 370), (178, 850), (190, 900), (195, 965), (180, 1012),
          (160, 1006), (150, 950), (155, 900), (160, 850), (160, 370), (140, 345)]
SKIN_L = [(95, -40), (90, 200), (70, 330), (60, 450), (80, 620), (120, 800), (140, 900), (140, 1040)]
SKIN_R = [(470, -40), (475, 200), (490, 330), (495, 430), (470, 620), (420, 800), (400, 900), (395, 1040)]
T_CUT = ((240, 540), (360, 612))                  # tibial fracture line (lateral-proximal -> medial-distal)
F_CUT = ((145, 566), (200, 590))                  # fibular fracture line
DISP = (26, 18, -3)                               # distal fragment: dx, dy, rotation deg (about the fracture)
POP = [(296, 60), (292, 120), (288, 250), (285, 330)]
ANT_T = [(285, 330), (240, 352), (218, 400), (214, 560)]
ANT_T_D = [(214, 600), (214, 860), (240, 925), (262, 990)]
POST_T = [(285, 330), (300, 420), (305, 560)]
POST_T_D = [(306, 604), (310, 800), (352, 905), (392, 1000)]

def S(p): return (p[0] * SS, p[1] * SS)

def mask(poly):
    m = Image.new('L', (DW * SS, DH * SS)); ImageDraw.Draw(m).polygon([S(p) for p in poly], fill=255); return m

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

def half(line, below):
    yy, xx = np.mgrid[0:DH * SS, 0:DW * SS].astype(np.float32) / SS
    (x0, y0), (x1, y1) = line
    cr = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)
    m = (cr > 0) if below else (cr <= 0)
    return Image.fromarray((m * 255).astype(np.uint8))

def mul(a, b): return Image.fromarray((np.asarray(a, np.float32) * np.asarray(b, np.float32) / 255).astype(np.uint8))

def fade_top(im, y0=40, y1=200):
    f = np.clip((np.arange(im.height) - y0) / (y1 - y0), 0, 1)[:, None] * np.ones((1, im.width))
    al = np.asarray(im.getchannel('A'), np.float32) * f
    im.putalpha(Image.fromarray(al.astype(np.uint8))); return im

def build():
    tm, fm = mask(TIBIA), mask(FIBULA)
    prox = Image.new('RGBA', tm.size)
    prox.alpha_composite(bone_layer(mask(FEMUR)))
    prox.alpha_composite(bone_layer(mul(fm, half(F_CUT, False)), 200))
    prox.alpha_composite(bone_layer(mul(tm, half(T_CUT, False))))
    dist = Image.new('RGBA', tm.size)
    talus = Image.new('L', tm.size); ImageDraw.Draw(talus).ellipse(S((222, 940)) + S((352, 1040)), fill=255)
    dist.alpha_composite(bone_layer(talus, 150))
    dist.alpha_composite(bone_layer(mul(fm, half(F_CUT, True)), 200))
    dist.alpha_composite(bone_layer(mul(tm, half(T_CUT, True))))
    whole = Image.new('RGBA', tm.size)
    whole.alpha_composite(bone_layer(mask(FEMUR)))
    whole.alpha_composite(bone_layer(talus, 150))
    whole.alpha_composite(bone_layer(fm, 200))
    whole.alpha_composite(bone_layer(tm))
    out = []
    for im in (prox, dist, whole):
        out.append(fade_top(im.resize((DW, DH), Image.LANCZOS)))
    return out

PROX, DIST, WHOLE = build()

def skin_layer():
    im = Image.new('RGBA', (DW * SS, DH * SS)); d = ImageDraw.Draw(im)
    d.polygon([S(p) for p in SKIN_L + SKIN_R[::-1]], fill=(150, 185, 220, 26))
    d.line([S(p) for p in SKIN_L], fill=(170, 205, 235, 110), width=3 * SS, joint='curve')
    d.line([S(p) for p in SKIN_R], fill=(170, 205, 235, 110), width=3 * SS, joint='curve')
    return fade_top(im.resize((DW, DH), Image.LANCZOS), 0, 160)
SKIN = skin_layer()

_lc = {}
def leg(p, s):
    """leg at fracture displacement p (0 = intact), scale s"""
    p = round(p, 2); k = (p, round(s, 3))
    if k in _lc: return _lc[k]
    im = Image.new('RGBA', (DW, DH)); im.alpha_composite(SKIN)
    if p <= 0:
        im.alpha_composite(WHOLE)
    else:
        dx, dy, rot = DISP
        im.alpha_composite(DIST.rotate(rot * p, resample=Image.BICUBIC, center=(300, 600), translate=(dx * p, dy * p)))
        im.alpha_composite(PROX)
    if s != 1: im = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    if len(_lc) > 300: _lc.clear()
    _lc[k] = im
    return im

def dp(cx, cy, s, q): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def polyline_part(pts, u):
    """first fraction u of a polyline"""
    if u >= 1: return pts
    L = [math.dist(a, b) for a, b in zip(pts, pts[1:])]; tot = sum(L) * u; out = [pts[0]]
    for (a, b), l in zip(zip(pts, pts[1:]), L):
        if tot >= l: out.append(b); tot -= l
        else:
            f = tot / l if l else 0; out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)); break
    return out

def arteries(main, glow, cx, cy, s, a, u, cut):
    """popliteal -> anterior + posterior tibial; drawn progressively (u); cut = 0..1 interrupts at the fracture"""
    if a <= 0 or u <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    col = WARN; w = max(4, int(8 * s))
    def draw(pts, al, dashed=False):
        P = [dp(cx, cy, s, q) for q in pts]
        if len(P) < 2: return
        if dashed:
            for A_, B_ in zip(P, P[1:]):
                n = max(1, int(math.dist(A_, B_) / 22))
                for i in range(0, n, 2):
                    f0, f1 = i / n, min(1, (i + 1) / n)
                    d.line([(A_[0] + (B_[0] - A_[0]) * f0, A_[1] + (B_[1] - A_[1]) * f0),
                            (A_[0] + (B_[0] - A_[0]) * f1, A_[1] + (B_[1] - A_[1]) * f1)], fill=col + (int(al * 255),), width=w - 2)
        else:
            d.line(P, fill=col + (int(al * 255),), width=w, joint='curve')
            g.line(P, fill=col + (int(al * 120),), width=w * 3)
    u1 = min(1, u * 2); u2 = clamp(u * 2 - 1)
    draw(polyline_part(POP, u1), a)
    if u2 > 0:
        draw(polyline_part(ANT_T, u2), a); draw(polyline_part(POST_T, u2), a)
        dist_a = a * (1 - 0.65 * cut)
        draw(polyline_part(ANT_T_D, u2), dist_a, dashed=cut > 0.5)
        draw(polyline_part(POST_T_D, u2), dist_a, dashed=cut > 0.5)
    main.alpha_composite(lay)

def wound(main, glow, cx, cy, s, a):
    if a <= 0: return
    pts = []
    for k in range(18):
        ang = k / 18 * 2 * math.pi; r = 44 + (12 if k % 2 else -6)
        pts.append(dp(cx, cy, s, (372 + r * math.cos(ang) * 0.8, 585 + r * math.sin(ang) * 1.25)))
    lay = Image.new('RGBA', (W, H)); ImageDraw.Draw(lay).polygon(pts, fill=WARN + (int(110 * a),), outline=WARN + (int(255 * a),))
    main.alpha_composite(lay)
    ImageDraw.Draw(glow).polygon(pts, fill=WARN + (int(70 * a),))

def frac_lines(main, glow, cx, cy, s, a):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    for (p0, p1) in ((T_CUT[0], T_CUT[1]), (F_CUT[0], F_CUT[1])):
        A_, B_ = dp(cx, cy, s, p0), dp(cx, cy, s, p1)
        d.line([A_, B_], fill=WARN + (int(230 * a),), width=max(3, int(5 * s)))
    main.alpha_composite(lay)

def marker(main, glow, x, y, n, a):
    if a <= 0: return
    r = 30
    lay = Image.new('RGBA', (W, H)); ImageDraw.Draw(lay).ellipse((x - r, y - r, x + r, y + r), fill=SKY + (int(255 * a),))
    main.alpha_composite(lay)
    ImageDraw.Draw(glow).ellipse((x - r - 8, y - r - 8, x + r + 8, y + r + 8), fill=SKY + (int(120 * a),))
    put(main, tl(str(n), 'MontBlack', 38, NAVY), x, y, a)

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
    put(main, pill('ORTHO NUMBERS  #4', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 440, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl('Mangled leg scores 7: amputate?', 'MontXB', 42, ICE), W / 2, 550, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('MESS 7', 'MontBlack', 190, WHITE, 880)
    put(glow, k1, W / 2, 720, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 720, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('Rule or myth?', 'MontBlack', 64, SKY), W / 2, 905 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('UNDER 7: SALVAGE', 'MontBlack', 40, NAVY, GOOD + (255,), h=80), W / 2, 1060 + 20 * (1 - q), a * q)
    q = eo(prog(t, 1.1, 1.6))
    put(main, pill('7+: NOT A VERDICT', 'MontBlack', 40, NAVY, WARN + (255,), h=80), W / 2, 1170 + 20 * (1 - q), a * q)

ROWS = [('1', 'SKELETAL / SOFT-TISSUE INJURY', 5.6), ('2', 'LIMB ISCHEMIA', 7.4), ('3', 'SHOCK', 8.2), ('4', 'AGE', 8.8)]
def s2(t, main, glow, fx):                                     # centrepiece: the mangled leg + 4 components
    a = win(t, 3.3, 12.2, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('MESS', 'MontBlack', 84, WHITE, track=6), W / 2, 262, a * p, 1, 10 * (1 - p))
    sa = 1 - prog(t, 9.7, 9.95)
    label(main, 'Mangled Extremity Severity Score', W / 2, 345, a * eo(prog(t, 3.6, 4.1)) * sa, SKY, 36, 'MontXB', 0)
    label(main, '4 factors: more points, worse prognosis', W / 2, 345, a * eo(prog(t, 10.0, 10.35)), SKY, 34, 'MontXB', 0)
    cx, cy, s = 540, 830, 0.86
    q = eo(prog(t, 3.6, 4.3))
    pdisp = eio(prog(t, 4.6, 5.3))
    put(main, leg(pdisp, s), cx, cy + 20 * (1 - q), a * q)
    frac_lines(main, glow, cx, cy, s, a * eo(prog(t, 4.5, 4.8)) * (1 - pdisp))
    wound(main, glow, cx, cy, s, a * eo(prog(t, 5.2, 5.6)))
    pointer(main, a * eo(prog(t, 4.2, 4.7)), dp(cx, cy, s, (172, 760)), 175, 980, 'FIBULA')
    pointer(main, a * eo(prog(t, 4.4, 4.9)), dp(cx, cy, s, (335, 760)), 880, 980, 'TIBIA')
    marker(main, glow, *dp(cx, cy, s, (372, 470)), 1, a * eo(prog(t, 5.5, 5.9)))
    au = eio(prog(t, 6.0, 7.0)); cut = eo(prog(t, 7.0, 7.3))
    arteries(main, glow, cx, cy, s, a, au, cut)
    if 6.95 < t < 7.4: fx['flash'] += 0.12 * math.exp(-((t - 7.1) / 0.07) ** 2)
    if au >= 1:
        pointer(main, a * eo(prog(t, 7.0, 7.4)), dp(cx, cy, s, (288, 250)), 800, 560, 'POPLITEAL A.', WARN)
    marker(main, glow, *dp(cx, cy, s, (140, 515)), 2, a * eo(prog(t, 7.3, 7.7)))
    for i, (n, txt, t0) in enumerate(ROWS):
        rq = eo(prog(t, t0, t0 + 0.45))
        if rq <= 0: continue
        y = 1335 + i * 88 + 15 * (1 - rq)
        put(main, card(850, 74, r=26), 515, y, a * rq)
        marker(main, glow, 125, y, n, a * rq)
        leftput(main, tl(txt, 'MontBlack', 34, WHITE, track=1), 175, y, a * rq)

def s3(t, main, glow, fx):                                     # the original data: number line
    a = win(t, 12.0, 21.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'THE 1990 DATA', W / 2, 262, a * eo(prog(t, 12.1, 12.6)), WHITE, 64, 'MontBlack', 3)
    label(main, 'mean MESS by outcome', W / 2, 340, a * eo(prog(t, 12.3, 12.8)), SKY, 36, 'MontXB', 0)
    X = lambda v: 130 + 65 * v
    ax = eio(prog(t, 12.4, 13.2))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    for yb in (700, 960):
        d.line([(X(0), yb), (X(0) + (X(12) - X(0)) * ax, yb)], fill=ICE + (int(170 * a),), width=4)
        for v in range(0, 13, 2):
            if X(v) <= X(0) + (X(12) - X(0)) * ax:
                d.line([(X(v), yb - 10), (X(v), yb + 10)], fill=ICE + (int(170 * a),), width=3)
    # threshold at 7
    tq = eio(prog(t, 13.2, 13.9))
    yy = 560
    while yy < 560 + 540 * tq:
        d.line([(X(7), yy), (X(7), min(yy + 24, 560 + 540 * tq))], fill=SKY + (int(255 * a),), width=6); yy += 38
    main.alpha_composite(lay)
    ImageDraw.Draw(glow).line([(X(7), 560), (X(7), 560 + 540 * tq)], fill=SKY + (int(90 * a),), width=18)
    for v in range(0, 13, 2):
        label(main, str(v), X(v), 1140, a * ax, SUB, 28, 'MontXB', 0)
    label(main, 'MESS', X(12) - 40, 1195, a * ax, SUB, 28, 'MontXB', 2)
    put(main, pill('7', 'MontBlack', 48, NAVY, SKY + (255,), padx=28, h=76), X(7), 515, a * tq)
    label(main, 'RETROSPECTIVE', 250, 600, a * eo(prog(t, 13.4, 13.8)), ICE, 30, 'MontBlack', 2)
    label(main, 'PROSPECTIVE', 230, 860, a * eo(prog(t, 15.0, 15.4)), ICE, 30, 'MontBlack', 2)
    for yb, v, n, col, t0 in [(700, 4.88, '17 salvaged', GOOD, 13.8), (700, 9.11, '9 amputated', WARN, 14.4),
                              (960, 4.00, '14 salvaged', GOOD, 15.4), (960, 8.83, '12 amputated', WARN, 16.0)]:
        q = R.eback(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        x = X(v); r = 22 * clamp(q, 0, 1.2)
        lay = Image.new('RGBA', (W, H)); ImageDraw.Draw(lay).ellipse((x - r, yb - r, x + r, yb + r), fill=col + (int(255 * a),))
        main.alpha_composite(lay)
        ImageDraw.Draw(glow).ellipse((x - r - 10, yb - r - 10, x + r + 10, yb + r + 10), fill=col + (int(140 * a),))
        label(main, f'{v:.2f}', x, yb - 58, a * clamp(q), col, 44, 'MontBlack', 0)
        label(main, n, x, yb + 56, a * clamp(q), WHITE, 30, 'MontXB', 0)
    q = eo(prog(t, 16.8, 17.3))
    if q > 0:
        y = 1370 + 20 * (1 - q)
        put(main, card(850, 250, outline=SKY + (160,)), 515, y, a * q)
        label(main, 'MESS 7+ predicted amputation', 515, y - 72, a * q, WHITE, 40, 'MontBlack', 0)
        put(main, tl('100%', 'MontBlack', 96, SKY), 515, y + 18, a * q)
        label(main, 'accuracy in both series', 515, y + 92, a * q, ICE, 32, 'MontXB', 0)
    label(main, 'Johansen et al.  ·  J Trauma 1990', W / 2, 1600, a * eo(prog(t, 13.0, 13.5)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # the twist: LEAP
    a = win(t, 20.8, 30.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'THEN CAME LEAP', W / 2, 262, a * eo(prog(t, 20.9, 21.4)), WHITE, 64, 'MontBlack', 3)
    p = R.eback(prog(t, 21.2, 21.7))
    big = tl('556', 'MontBlack', 150, SKY)
    put(glow, big, 330, 440, a * 0.3 * clamp(p)); put(main, big, 330, 440, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'high-energy leg', 735, 410, a * eo(prog(t, 21.5, 22.0)), WHITE, 36, 'MontXB', 0)
    label(main, 'injuries, prospective', 735, 462, a * eo(prog(t, 21.5, 22.0)), WHITE, 36, 'MontXB', 0)
    for cx, head, l1, l2, foot, col, ok, t0 in [(295, 'LOW SCORE', 'predicts', 'limb salvage', 'high specificity', GOOD, True, 22.3),
                                                (735, 'HIGH SCORE', 'not a valid', 'amputation predictor', 'low sensitivity', WARN, False, 23.4)]:
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 790 + 25 * (1 - q)
        put(main, card(405, 380, outline=col + (170,)), cx, y, a * q)
        lay = Image.new('RGBA', (W, H)); dd = ImageDraw.Draw(lay)
        (check if ok else cross)(dd, cx, y - 110, 80, col, a * q, w=12)
        main.alpha_composite(lay)
        label(main, head, cx, y - 15, a * q, col, 44, 'MontBlack', 2)
        label(main, l1, cx, y + 50, a * q, WHITE, 32, 'MontXB', 0)
        label(main, l2, cx, y + 95, a * q, WHITE, 32, 'MontXB', 0)
        label(main, foot, cx, y + 150, a * q, SUB, 29, 'MontXB', 0)
    q = eo(prog(t, 24.4, 24.9))
    if q > 0:
        y = 1135 + 20 * (1 - q)
        put(main, card(850, 120, outline=WARN + (150,)), 515, y, a * q)
        label(main, 'None of the 5 proved clinically useful', 515, y, a * q, WHITE, 38, 'MontBlack', 0)
    q = eo(prog(t, 26.8, 27.3))
    if q > 0:
        y = 1380 + 20 * (1 - q)
        put(main, card(850, 230, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'After a successful salvage', 515, y - 60, a * q, ICE, 34, 'MontXB', 0)
        label(main, 'no score predicted function', 515, y + 0, a * q, WHITE, 42, 'MontBlack', 0)
        label(main, 'Ly et al.  ·  JBJS Am 2008  ·  407 patients', 515, y + 66, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Bosse et al. (LEAP)  ·  JBJS Am 2001', W / 2, 1600, a * eo(prog(t, 21.8, 22.3)), SUB, 28, 'MontXB', 0)

TIPS = [('Under 7: aim to salvage', 'LOW SCORES PREDICT SALVAGE POTENTIAL', 'Bosse et al. 2001 (LEAP)'),
        ('7 or more: not a verdict', 'NEVER THE SOLE REASON TO AMPUTATE', 'Schirò et al. 2015  ·  systematic review'),
        ('Mangled arm? No better', 'AT LEAST 20% SALVAGED ABOVE THE CUT-OFF', 'Yoneda et al. 2024  ·  meta-analysis')]
def s5(t, main, glow, fx):
    a = win(t, 30.0, 37.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 30.1, 30.6)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 30.6 + 1.7 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(840, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s6(t, main, glow, fx):
    a = win(t, 36.8, 44.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 330, a * eo(prog(t, 36.9, 37.4)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 37.1, 37.7))
    big = tl('MESS 7+', 'MontBlack', 120, WHITE)
    put(glow, big, W / 2, 470, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 470, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'is a warning, not an amputation order', W / 2, 590, a * eo(prog(t, 37.5, 38.0)), WARN, 37, 'MontBlack', 0)
    label(main, 'Under 7: work to save the limb', W / 2, 670, a * eo(prog(t, 37.8, 38.3)), GOOD, 40, 'MontBlack', 0)
    q = eo(prog(t, 38.2, 38.8))
    put(main, card(850, 530, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 1022, a * q)
    for i, (au, ref) in enumerate([('Johansen K, et al.', 'J Trauma. 1990;30(5):568–72'),
                                   ('Bosse MJ, et al. (LEAP)', 'J Bone Joint Surg Am. 2001;83(1):3–14'),
                                   ('Ly TV, et al.', 'J Bone Joint Surg Am. 2008;90(8):1738–43'),
                                   ('Schirò GR, et al.', 'BMC Musculoskelet Disord. 2015;16:372'),
                                   ('Yoneda H, et al.', 'Injury. 2024;55(4):111447')]):
        y = 815 + i * 94
        label(main, au, W / 2, y, a * q, ICE, 30, 'MontXB', 0)
        label(main, ref, W / 2, y + 42, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1375, a * eo(prog(t, 38.8, 39.4)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
