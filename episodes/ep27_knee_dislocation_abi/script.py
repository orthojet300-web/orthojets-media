"""Orthojets EP27 - ORTHO NUMBERS #6 - Knee dislocation: ABI under 0.9 and the popliteal artery.
Sources (verified on PubMed abstracts / metadata, 2026-11-21; PubMed is the only verification source):
  Mills WJ, Barei DP, McNair P. J Trauma 2004;56(6):1261-5. PMID 15211135, doi 10.1097/01.ta.0000068995.63201.0b.
    Prospective, 38 knee dislocations, pulse exam + ABI. ABI lower than 0.90 -> arteriography; ABI 0.90 or higher ->
    immobilised, admitted, serial examination and delayed arterial duplex. 11 of 38 (29%) had ABI under 0.90: ALL 11 had
    arterial injury requiring surgical treatment. 27 with ABI 0.90 or higher: none had vascular injury on serial exam or
    duplex. Sensitivity, specificity, PPV and NPV 100%. Routine arteriography for all knee dislocations not supported.
  Medina O et al. Clin Orthop Relat Res 2014;472(9):2621-9. PMID 24554457, doi 10.1007/s11999-014-3511-3. Systematic
    review: 862 knee dislocations, vascular injury 171 = weighted 18%; nerve injury 25% (75/272); 80% of vascular
    injuries repaired; 12% (22/134) of vascular injuries ended in amputation; highest vascular injury: KD-IIIL 32%,
    posterior dislocation 25%.
  Green NE, Allen BL. J Bone Joint Surg Am 1977;59(2):236-9. PMID 845209 (no DOI in PubMed). 245 knee dislocations,
    popliteal artery injury 32%; vascular repair within 6, at most 8 hours of injury to avoid amputation; of patients
    not treated within that time, 86% had an amputation.
  Stannard JP et al. J Bone Joint Surg Am 2004;86(5):910-5. PMID 15118031 (no DOI in PubMed). 134 knees, flow-limiting
    popliteal artery damage 7% (9 patients); 7 of 9 were KD-IV; selective arteriography based on serial examination is
    safe; KD-IV: serial examinations for at least 48 hours.
  Sillanpaa PJ et al. J Trauma Acute Care Surg 2014;76(3):715-9. PMID 24553539, doi 10.1097/TA.0000000000000136.
    Finland, 837 knee dislocations: most common mechanism low-energy fall at the same level (46%).
  Bernhoff K et al. Eur J Vasc Endovasc Surg 2021;61(2):297-304. PMID 33303313, doi 10.1016/j.ejvs.2020.10.017.
    Sweden: popliteal artery injury after knee dislocation 3.4-8.2%; PAI after knee dislocation most frequently caused by a fall.
  Teissier V et al. Ann Vasc Surg 2019;61:238-245. PMID 31344468, doi 10.1016/j.avsg.2019.04.016. 16 knee dislocations
    with popliteal artery injury: 7 ruptures (44%), 7 dissections (44%), 2 isolated thromboses (13%); 50% posterior.
  Diagram: knee, lateral view, anterior on the viewer's LEFT. Femoral shaft + rounded condyles, patella anterior
  to the condyles, tibial plateau below with the tubercle anteriorly, fibular head posterior. The popliteal artery runs
  behind the femur and the back of the knee. Posterior dislocation = tibia displaced posteriorly (to the right) and
  proximally behind the condyles, stretching the popliteal artery around the back of the tibial plateau.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(49.5, [(8.4, 1.4), (13.0, 1.2), (17.3, 1.2), (21.0, 1.4), (29.2, 1.6), (36.4, 1.6), (42.6, 1.6),
                     (48.2, 0.8)])

DW, DH = 760, 900
XRAY = (170, 210, 242)
ART = (214, 104, 118)          # artery (normal)
SS = 2
XC = 515

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

def poly_mask(pts, smooth=True):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    pp = catmull(pts, 8) if smooth else pts
    d.polygon([(x * SS, y * SS) for x, y in pp], fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def ell_mask(c, rx, ry):
    big = Image.new('L', (DW * SS, DH * SS))
    ImageDraw.Draw(big).ellipse(((c[0] - rx) * SS, (c[1] - ry) * SS, (c[0] + rx) * SS, (c[1] + ry) * SS), fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def line_mask(pts, w):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    pp = catmull(pts, 12, closed=False)
    d.line([(x * SS, y * SS) for x, y in pp], fill=255, width=int(w * SS), joint='curve')
    for x, y in (pp[0], pp[-1]):
        d.ellipse(((x - w / 2) * SS, (y - w / 2) * SS, (x + w / 2) * SS, (y + w / 2) * SS), fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

_yy = np.arange(DH, dtype=np.float32)[:, None] * np.ones((1, DW), np.float32)
FADE = np.clip((_yy - 4) / 150.0, 0, 1) * np.clip((DH - 6 - _yy) / 150.0, 0, 1)
def faded(m, f=FADE): return Image.fromarray((np.asarray(m, np.float32) * f).astype(np.uint8))
def union(ms):
    u = ms[0]
    for m in ms[1:]: u = ImageChops.lighter(u, m)
    return u
def lerp(a, b, r): return a + (b - a) * r
def lp(p, q, r): return (lerp(p[0], q[0], r), lerp(p[1], q[1], r))
def shift(pts, dx, dy): return [(x + dx, y + dy) for x, y in pts]

# ------------------------------------------------------------------ lateral knee geometry (anterior = left)
FEM = [(300, 0), (446, 0), (452, 200), (468, 330), (492, 400), (500, 470), (482, 545), (430, 578), (360, 572),
       (306, 540), (290, 470), (298, 380), (300, 250)]
PAT = [(232, 420), (262, 392), (288, 420), (292, 490), (276, 548), (248, 556), (228, 520), (224, 470)]
TIB = [(276, 598), (310, 582), (400, 578), (488, 584), (512, 604), (500, 640), (470, 690), (452, 760), (446, 900),
       (330, 900), (322, 770), (300, 700), (272, 668), (266, 630)]
FIB = [(466, 650), (496, 640), (516, 664), (508, 706), (494, 760), (488, 900), (462, 900), (466, 760), (462, 700)]
ART_N = [(556, -20), (552, 200), (548, 400), (540, 560), (534, 640), (528, 760), (520, 920)]
DISP = (176, -70)              # posterior dislocation: tibia behind and above the condyles

def art_path(r):
    """popliteal artery centreline; with r -> 1 it is pushed back over the displaced posterior tibial plateau"""
    dx, dy = DISP[0] * r, DISP[1] * r
    return [(556, -20), (552, 200), (lerp(548, 566, r), 400), (lerp(540, 640, r), lerp(560, 500, r)),
            (lerp(534, 702, r), lerp(640, 560, r)), (lerp(528, 680, r), lerp(760, 700, r)),
            (lerp(520, 640, r), 920)]

_K = {}
def knee(r):
    k = round(r * 12) / 12
    if k in _K: return _K[k]
    dx, dy = DISP[0] * k, DISP[1] * k
    fm = faded(poly_mask(FEM)); pm = poly_mask(PAT)
    tm = faded(poly_mask(shift(TIB, dx, dy))); bm = faded(poly_mask(shift(FIB, dx, dy)))
    am = faded(line_mask(art_path(k), 30))
    soft_m = union([fm, pm, tm, bm, am]).filter(ImageFilter.MaxFilter(51)).filter(ImageFilter.GaussianBlur(22))
    soft_m = faded(soft_m.point(lambda v: 255 if v > 90 else int(v * 255 / 90)))
    lay = Image.new('RGBA', (DW, DH))
    lay.alpha_composite(volume(soft_m, (64, 98, 140), alpha=0.5, r=18, relief=0.6))
    lay.alpha_composite(bone_xray(bm, XRAY, alpha=0.8, seed=14))
    lay.alpha_composite(bone_xray(tm, XRAY, alpha=0.96, seed=13))
    lay.alpha_composite(bone_xray(fm, XRAY, alpha=0.96, seed=11))
    lay.alpha_composite(bone_xray(pm, XRAY, alpha=0.96, seed=12))
    lay.alpha_composite(volume(am, ART, alpha=0.95, r=6, relief=0.8))
    pad = 40
    big = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); big.alpha_composite(lay, (pad, pad))
    big = drop_shadow(big, dx=10, dy=16, blur=14, opacity=0.55)
    _K[k] = big.crop((pad, pad, pad + DW, pad + DH))
    return _K[k]

_dc = {}
def scaled(img, s, key):
    k = (key, round(s, 3))
    if k not in _dc:
        _dc[k] = img.resize((int(DW * s), int(DH * s)), Image.LANCZOS) if abs(s - 1) > 1e-3 else img
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

def artery_glow(main, glow, r, cx, cy, s, a, col, seg=(3, 6)):
    """glowing overlay on the stretched part of the artery"""
    if a <= 0: return
    pts = catmull(art_path(r), 12, closed=False)
    i0, i1 = seg[0] * 12, seg[1] * 12
    P = [dp(p, cx, cy, s) for p in pts[i0:i1]]
    lay = aa_image(W, H); d = ImageDraw.Draw(lay)
    d.line(P, fill=col + (int(230 * a),), width=max(6, int(16 * s)), joint='curve')
    ImageDraw.Draw(glow).line(P, fill=col + (int(140 * a),), width=int(60 * s))
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('ORTHO NUMBERS  #6', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Knee dislocation: is the artery injured?', 'MontXB', 42, ICE, 880), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('ABI 0.9', 'MontBlack', 170, WHITE, 880)
    put(glow, k1, W / 2, 670, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 670, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('THE POPLITEAL ARTERY CUT-OFF', 'MontBlack', 52, SKY, 880, track=1), W / 2, 820, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('one number decides the angiogram', 'MontXB', 42, WHITE), W / 2, 905 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); cx, cy, s = 500, 1330, 0.74
    put(main, scaled(knee(1.0), s, 'k1'), cx, cy + 20 * (1 - hq), a * hq)
    pul = 0.65 + 0.35 * math.sin(t * 6)
    artery_glow(main, glow, 1.0, cx, cy + 20 * (1 - hq), s, a * eo(prog(t, 0.3, 0.7)) * pul, WARN)

def s2(t, main, glow, fx):                                     # why it matters: Medina 2014
    a = win(t, 3.3, 9.6, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('WHY IT MATTERS', 'MontBlack', 62, WHITE, 840, track=3), XC, 262, a * p, 1, 10 * (1 - p))
    label(main, '862 patients with knee dislocation  ·  review', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    rows = [(3.9, '18%', 'vascular injury', 'weighted frequency', WARN),
            (4.9, '25%', 'nerve injury', '75 of 272', SKY),
            (5.9, '12%', 'of vascular injuries', 'ended in amputation', WARN),
            (6.9, '32%', 'vascular injury in KD-IIIL', 'posterior KDs: 25%', WARN)]
    for i, (t0, num, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 540 + i * 255 + 20 * (1 - q)
        put(main, card(880, 225, outline=col + (150,)), XC, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        nb = fit(num, 'MontBlack', 84, col, 260); put(main, nb, 125 + nb.width / 2, y, a * q * clamp(pk), clamp(pk, 0, 1.2))
        x0 = 125 + max(nb.width, 170) + 26
        im = fit(l1, 'MontBlack', 42, WHITE, 905 - x0); put(main, im, x0 + im.width / 2, y - 30, a * q)
        im = fit(l2, 'MontXB', 32, SUB, 905 - x0); put(main, im, x0 + im.width / 2, y + 34, a * q)
    label(main, 'Medina et al.  ·  Clin Orthop Relat Res 2014', XC, 1560, a * eo(prog(t, 7.4, 7.9)), ICE, 34, 'MontXB', 0)

CX, CY, S = 500, 830, 0.86
STEPS = [(9.6, 'NORMAL KNEE', 'artery runs behind the femur and the knee'),
         (13.6, 'POSTERIOR DISLOCATION', 'vascular injury in 25% of posterior KDs'),
         (17.6, 'POPLITEAL ARTERY', 'stretched over the tibia: rupture, dissection, clot')]
def s3(t, main, glow, fx):                                     # centrepiece: lateral knee, dislocation, artery
    a = win(t, 9.4, 21.6, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 9.5, 10.0))
    put(main, tl('READ THE KNEE', 'MontBlack', 62, WHITE, track=3), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Knee  ·  lateral view', XC, 335, a * eo(prog(t, 9.7, 10.2)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 9.6, 10.5)); cy = CY + 24 * (1 - bp)
    r = eio(prog(t, 13.7, 15.2))
    put(main, scaled(knee(r), S, f'k{round(r * 12)}'), CX, cy, a * bp)
    P = lambda q_: dp(q_, CX, cy, S)
    la = a * eo(prog(t, 10.2, 10.6)) * (1 - eo(prog(t, 13.3, 13.7)))
    pointer(main, la, P((380, 200)), 220, 520, 'FEMUR', ICE, 32)
    pointer(main, la, P((240, 480)), 170, 830, 'PATELLA', ICE, 30)
    pointer(main, la, P((390, 780)), 210, 1180, 'TIBIA', ICE, 32)
    pointer(main, la, P((550, 300)), 840, 560, 'POPLITEAL', ART, 30)
    label(main, 'ARTERY', 860, 600, la, ART, 30, 'MontXB', 2)
    pointer(main, la, P((482, 820)), 700, 1180, 'FIBULA', ICE, 30)
    label(main, 'ANTERIOR', 150, 1300, la, SUB, 28, 'MontXB', 2)
    label(main, 'POSTERIOR', 860, 1300, la, SUB, 28, 'MontXB', 2)
    # step 2: displacement arrow
    da = a * eo(prog(t, 13.6, 14.0)) * (1 - eo(prog(t, 17.2, 17.6)))
    if da > 0:
        y0 = P((0, 640))[1]
        ImageDraw.Draw(glow).line([(P((330, 0))[0], y0), (P((600, 0))[0], y0)], fill=WARN + (int(120 * da),), width=30)
        arrow(ImageDraw.Draw(main), P((330, 0))[0], y0, P((600, 0))[0], WARN, da, w=12)
        label(main, 'TIBIA DISPLACED POSTERIORLY', XC, 1270, da, WARN, 34, 'MontBlack', 1)
    # step 3: stretched artery glows
    ga = a * eo(prog(t, 15.4, 15.9))
    if ga > 0:
        pul = 0.6 + 0.4 * math.sin(t * 6)
        artery_glow(main, glow, 1.0, CX, cy, S, ga * pul, WARN)
    sa = a * eo(prog(t, 17.6, 18.0))
    if sa > 0:
        pointer(main, sa, P((690, 600)), 846, 1150, 'STRETCHED', WARN, 32)
        label(main, '16 injured arteries:', XC, 1240, sa, ICE, 32, 'MontXB', 0)
        label(main, '44% rupture  ·  44% dissection  ·  13% clot', XC, 1290, sa, WARN, 32, 'MontBlack', 0)
    pa = a * eo(prog(t, 10.2, 10.6))
    if pa > 0:
        y = 1450; cur = 0
        for i, (t0, _, _) in enumerate(STEPS):
            if t >= t0: cur = i
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        t0, nm, sub = STEPS[cur]; q = eo(prog(t, t0, t0 + 0.35))
        put(main, pill(f'{cur + 1} / 3', 'MontBlack', 36, NAVY, SKY + (255,), padx=22, h=64), 170, y - 40, pa * q)
        im = fit(nm, 'MontBlack', 44, WHITE, 650); put(main, im, 270 + im.width / 2, y - 40, pa * q)
        im = fit(sub, 'MontXB', 32, SKY, 790); put(main, im, 120 + im.width / 2, y + 50, pa * q)
        ref = 'Medina et al. CORR 2014' if cur < 2 else 'Teissier et al. Ann Vasc Surg 2019'
        label(main, ref, XC, 1612, pa * q, ICE, 34, 'MontXB', 0)

def limb(main, kind, cx, cy, a, cuff_a):
    """simple shaded limb with a pressure cuff: 'arm' (horizontal forearm) or 'leg' (lower leg + foot)"""
    if a <= 0: return
    w, h = 400, 300; ss = 2
    m = Image.new('L', (w * ss, h * ss)); d = ImageDraw.Draw(m)
    if kind == 'arm':
        d.rounded_rectangle((20 * ss, 110 * ss, 300 * ss, 190 * ss), 40 * ss, fill=255)
        d.ellipse((270 * ss, 100 * ss, 370 * ss, 200 * ss), fill=255)           # hand
    else:
        d.rounded_rectangle((120 * ss, 0, 210 * ss, 240 * ss), 42 * ss, fill=255)  # lower leg
        d.polygon([(118 * ss, 200 * ss), (212 * ss, 200 * ss), (360 * ss, 250 * ss), (372 * ss, 290 * ss),
                   (120 * ss, 290 * ss)], fill=255)                              # foot
        d.ellipse((110 * ss, 200 * ss, 220 * ss, 296 * ss), fill=255)            # heel
    m = m.resize((w, h), Image.LANCZOS)
    lay = Image.new('RGBA', (w + 60, h + 60)); lay.alpha_composite(volume(m, (120, 150, 185), alpha=0.95, r=10), (30, 30))
    cm = Image.new('L', (w * ss, h * ss)); dc = ImageDraw.Draw(cm)
    if kind == 'arm': dc.rounded_rectangle((70 * ss, 92 * ss, 200 * ss, 208 * ss), 18 * ss, fill=255)
    else: dc.rounded_rectangle((104 * ss, 100 * ss, 226 * ss, 186 * ss), 18 * ss, fill=255)
    cm = cm.resize((w, h), Image.LANCZOS)
    cl = volume(cm, SKY, alpha=cuff_a, r=8)
    lay.alpha_composite(cl, (30, 30))
    put(main, drop_shadow(lay, blur=10, opacity=0.45), cx, cy, a)

def s4(t, main, glow, fx):                                     # measuring the ABI
    a = win(t, 21.4, 30.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'MEASURE THE ABI', XC, 262, a * eo(prog(t, 21.5, 22.0)), WHITE, 62, 'MontBlack', 3)
    label(main, 'ankle-brachial index, injured leg', XC, 340, a * eo(prog(t, 21.7, 22.2)), SKY, 36, 'MontXB', 0)
    q1 = eo(prog(t, 22.0, 22.5)); q2 = eo(prog(t, 22.6, 23.1))
    limb(main, 'arm', 300, 560, a * q1, q1); limb(main, 'leg', 300, 900, a * q2, q2)
    v1 = eio(prog(t, 23.2, 24.2)); v2 = eio(prog(t, 24.0, 25.0))
    if q1 > 0:
        label(main, 'ARM', 740, 500, a * q1, SUB, 32, 'MontXB', 3)
        put(main, tl(f'{round(140 * v1)}', 'MontBlack', 96, WHITE), 740, 590, a * q1)
    if q2 > 0:
        label(main, 'ANKLE', 740, 840, a * q2, SUB, 32, 'MontXB', 3)
        put(main, tl(f'{round(70 * v2)}', 'MontBlack', 96, WARN), 740, 930, a * q2)
    label(main, 'systolic pressures, mmHg  ·  example', XC, 1090, a * eo(prog(t, 23.4, 23.9)), SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 25.4, 25.9))
    if q > 0:
        y = 1330 + 20 * (1 - q)
        put(main, card(880, 300, outline=SKY + (160,)), XC, y, a * q)
        label(main, 'ABI  =  ankle  /  arm', XC, y - 80, a * q, SKY, 46, 'MontBlack', 1)
        v = eio(prog(t, 26.0, 27.0))
        label(main, f'70 / 140  =  {0.5 * v:.2f}', XC, y + 10, a * q, WHITE, 66, 'MontBlack', 0)
        qq = eo(prog(t, 27.2, 27.7))
        label(main, 'UNDER 0.9: ABNORMAL', XC, y + 95, a * q * qq, WARN, 40, 'MontBlack', 2)

def s5(t, main, glow, fx):                                     # the decision: Mills 2004
    a = win(t, 29.8, 37.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'THE CUT-OFF', XC, 262, a * eo(prog(t, 29.9, 30.4)), WHITE, 62, 'MontBlack', 3)
    label(main, '38 patients with knee dislocation  ·  prospective', XC, 340, a * eo(prog(t, 30.1, 30.6)), SKY, 36, 'MontXB', 0)
    q = eo(prog(t, 30.4, 30.9))
    if q > 0:                                                  # gauge 0.5 .. 1.3 with the 0.9 line
        d = ImageDraw.Draw(main); x0, x1, y = 120, 910, 520
        xv = lambda v: x0 + (x1 - x0) * (v - 0.5) / 0.8
        d.rounded_rectangle((x0, y - 26, xv(0.9), y + 26), 22, fill=WARN + (int(220 * a * q),))
        d.rounded_rectangle((xv(0.9), y - 26, x1, y + 26), 22, fill=GOOD + (int(220 * a * q),))
        d.line([(xv(0.9), y - 70), (xv(0.9), y + 70)], fill=WHITE + (int(255 * a * q),), width=8)
        label(main, '0.9', xv(0.9), y - 110, a * q, WHITE, 52, 'MontBlack', 0)
        for v in (0.5, 1.3):
            label(main, f'{v}', xv(v) + (30 if v < 1 else -30), y + 70, a * q, SUB, 30, 'MontXB', 0)
        mv = lerp(1.2, 0.5, eio(prog(t, 30.9, 32.2)))                 # marker settles on the example ABI 0.50
        mx = xv(mv); dd = ImageDraw.Draw(main)
        dd.polygon([(mx, y - 36), (mx - 18, y - 66), (mx + 18, y - 66)], fill=WHITE + (int(255 * a * q),))
        label(main, 'example 0.50', xv(0.5) + 110, y - 100, a * q * eo(prog(t, 32.0, 32.4)), WHITE, 30, 'MontXB', 0)
    rows = [(31.2, 'UNDER 0.9', '11 of 11: arterial injury', 'all needed surgery  ·  arteriography', WARN),
            (32.4, '0.9 OR MORE', '0 of 27: vascular injury detected', 'admit, serial exams, delayed duplex', GOOD)]
    for i, (t0, h_, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 790 + i * 300 + 20 * (1 - q)
        put(main, card(880, 260, outline=col + (170,)), XC, y, a * q)
        label(main, h_, XC, y - 78, a * q, col, 44, 'MontBlack', 3)
        im = fit(l1, 'MontBlack', 46, WHITE, 820); put(main, im, XC, y - 2, a * q)
        im = fit(l2, 'MontXB', 32, SUB, 820); put(main, im, XC, y + 66, a * q)
    q = eo(prog(t, 33.8, 34.3))
    if q > 0:
        pk = R.eback(prog(t, 33.9, 34.4))
        label(main, 'Sensitivity  ·  specificity  ·  PPV  ·  NPV', XC, 1400, a * q, SKY, 34, 'MontXB', 0)
        put(main, tl('100%', 'MontBlack', 110, WHITE), XC, 1500, a * q * clamp(pk), clamp(pk, 0, 1.2))
    label(main, 'Mills et al.  ·  J Trauma 2004', XC, 1620, a * eo(prog(t, 34.2, 34.7)), ICE, 34, 'MontXB', 0)

TIPS = [('The clock: repair in 6–8 h', 'NOT REPAIRED IN TIME: 86% AMPUTATED', 'Green & Allen JBJS Am 1977  ·  245 knees'),
        ('Low energy, still dangerous', '46%: LOW-ENERGY FALL AT THE SAME LEVEL', 'Sillanpää et al. J Trauma Acute Care Surg 2014'),
        ('Normal ABI or exam? Re-examine', 'KD-IV: SERIAL EXAMS FOR AT LEAST 48 H', 'Stannard et al. JBJS Am 2004  ·  7 of 9 KD-IV')]
def s6(t, main, glow, fx):
    a = win(t, 36.8, 43.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS & TRICKS', XC, 262, a * eo(prog(t, 36.9, 37.4)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 37.4 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
        put(main, card(880, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 40, WHITE, 690); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 32, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s7(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 43.0, 49.5, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 360, a * eo(prog(t, 43.1, 43.6)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 43.3, 43.9))
    big = fit('ABI UNDER 0.9', 'MontBlack', 110, WHITE, 880)
    put(glow, big, XC, 490, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 490, a * clamp(p), clamp(p, 0, 1.2))
    label(main, '= image the artery', XC, 600, a * eo(prog(t, 43.7, 44.2)), WARN, 44, 'MontBlack', 0)
    label(main, '0.9 or more = admit, serial exams, duplex', XC, 670, a * eo(prog(t, 44.0, 44.5)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 44.4, 45.0))
    refs = ['Mills WJ, et al. J Trauma 2004;56:1261–5', 'Medina O, et al. Clin Orthop 2014;472:2621–9',
            'Green NE, Allen BL. JBJS Am 1977;59:236–9', 'Stannard JP, et al. JBJS Am 2004;86:910–5',
            'Sillanpää PJ, et al. J Trauma Acute Care Surg 2014;76:715–9',
            'Teissier V, et al. Ann Vasc Surg 2019;61:238–45']
    put(main, card(880, 460, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 978, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 820), XC, 815 + i * 65, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1360, a * eo(prog(t, 45.0, 45.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
