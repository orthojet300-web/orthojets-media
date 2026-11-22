"""Orthojets EP32 - DON'T MISS #8 - Necrotising fasciitis: LRINEC score and the finger test.
Sources (verified on PubMed abstracts / metadata / PMC full text via the PubMed connector, 2026-11-22):
  Wong CH, Chang HC, Pasupathy S, et al. Necrotizing fasciitis: clinical presentation, microbiology, and determinants of
    mortality. J Bone Joint Surg Am 2003;85(8):1454-60. PMID 12925624 (no DOI in PubMed). 89 consecutive patients:
    paucity of cutaneous findings early; only 13 of 89 had the diagnosis at admission; pre-admission antibiotics masked
    the severity; polymicrobial 53.9%; diabetes 70.8%; on multivariate analysis only a delay in surgery of more than
    24 hours was correlated with increased mortality (relative risk 9.4).
  Wong CH, Khin LW, Heng KS, et al. The LRINEC score. Crit Care Med 2004;32(7):1535-41. PMID 15241098,
    doi 10.1097/01.ccm.0000129486.35458.7d. 145 NF vs 309 severe cellulitis/abscess; 6 labs: total white cell count,
    haemoglobin, sodium, glucose, serum creatinine, CRP. Cut-off 6 points: PPV 92.0%, NPV 96.0%; AUC 0.980 / 0.976.
  Hsiao CT et al. Prospective validation of LRINEC for NF of the extremities. PLoS One 2020;15(1):e0227748.
    PMID 31978094, doi 10.1371/journal.pone.0227748. Maximum score 13 (intro). 106 NF vs 825 cellulitis (ED, limbs):
    LRINEC 6 or more sensitivity 43%, specificity 83%; AUC 0.696.
  Fernando SM et al. NSTI: diagnostic accuracy of physical examination, imaging and LRINEC. Ann Surg 2019;269(1):58-65.
    PMID 29672405, doi 10.1097/SLA.0000000000002774. 23 studies, n = 5982. Fever sens 46.0% / spec 77.0%; haemorrhagic
    bullae 25.2% / 95.8%; hypotension 21.0% / 97.7%; CT 88.5% / 93.3%; plain radiography 48.9% / 94.0%; LRINEC 6 or more
    68.2% / 84.8%; 8 or more 40.8% / 94.9%. Absence of any 1 exam feature does not rule out NSTI; LRINEC should not be
    used to rule out; high suspicion -> early surgical consultation.
  Nawijn F et al. Time is of the essence ... NSTI: systematic review and meta-analysis. World J Emerg Surg 2020;15:4.
    PMID 31921330, doi 10.1186/s13017-019-0286-6. Surgery within 6 h of presentation: mortality 19% vs 32% if delayed
    over 6 h (OR 0.43, 10 studies). Overall 1277 of 6051 died (21.1%).
  Gelbard RB et al. EAST practice management guideline: optimal timing of initial debridement for NSTI. J Trauma Acute
    Care Surg 2018;85(1):208-14. PMID 29485428, doi 10.1097/TA.0000000000001857. Recommend operative debridement within
    12 hours of suspected diagnosis; mortality 14% early vs 25.8% late.
  Kazi FN et al. Comparison of LRINEC with the finger test and histopathology. Surg J (N Y) 2022;8(1):e1-e7.
    PMID 35059495, doi 10.1055/s-0041-1740629. Finger test described by Andreasen et al. 2001; pointers of NF: lack of
    bleeding, malodorous "dish water" pus, lack of finger dissection resistance. NF = necrosis of the fascia with relative
    sparing of skin and underlying muscle. Finger test sensitivity 86.11% (40 patients).
  Diagram (schematic, labelled tissue block of a limb, front face): skin (epidermis + dermis), subcutaneous fat, deep
  fascia, muscle. Cellulitis = redness in the dermis and fat. NF = necrosis spreading along the fascial plane, wider than
  the skin change above it; muscle relatively spared.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(52.5, [(9.4, 0.8), (12.6, 1.0), (18.4, 1.2), (27.8, 1.2), (34.0, 1.0), (39.8, 1.0), (46.0, 1.4),
                     (52.0, 0.6)])

XC = 510
SS = 2
DW, DH = 680, 520
# tissue layers (diagram y)
Y_EPI, Y_DER, Y_FAT, Y_FAS, Y_MUS, Y_END = 70, 84, 140, 300, 324, 520
C_SKIN = (222, 186, 176)
C_DER = (200, 150, 150)
C_FAT = (226, 204, 146)
C_FAS = (236, 240, 250)
C_MUS = (176, 84, 96)
C_NEC = (96, 86, 78)

_xx = np.arange(DW, dtype=np.float32)[None, :] * np.ones((DH, 1), np.float32)
HFADE = np.clip((_xx - 4) / 70.0, 0, 1) * np.clip((DW - 4 - _xx) / 70.0, 0, 1)
def hfaded(m): return Image.fromarray((np.asarray(m, np.float32) * HFADE).astype(np.uint8))

def rect_mask(y0, y1, wave=0.0, seed=0):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    top = [(x * SS, (y0 + wave * math.sin(x / 47 + seed)) * SS) for x in range(0, DW + 10, 10)]
    bot = [(x * SS, (y1 + wave * math.sin(x / 53 + seed + 1.7)) * SS) for x in range(DW, -10, -10)]
    d.polygon(top + bot, fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def nec_mask(r, y0, y1, spread=1.0):
    """necrosis band centred under the skin lesion, growing sideways along the fascia with r"""
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    half = 40 + 280 * r * spread; c = DW / 2
    pts = []
    for k in range(41):
        x = c - half + 2 * half * k / 40
        e = 1 - abs((x - c) / half) ** 3
        pts.append((x * SS, (y0 - 12 * e - 3 * math.sin(x / 23)) * SS))
    for k in range(41):
        x = c + half - 2 * half * k / 40
        e = 1 - abs((x - c) / half) ** 3
        pts.append((x * SS, (y1 + 10 * e + 3 * math.sin(x / 19)) * SS))
    d.polygon(pts, fill=255)
    return big.resize((DW, DH), Image.LANCZOS).filter(ImageFilter.GaussianBlur(2))

def blob_mask(cx, cy, rx, ry):
    big = Image.new('L', (DW * SS, DH * SS))
    ImageDraw.Draw(big).ellipse(((cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS), fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def tint(mask, col, alpha):
    m = np.asarray(mask, np.float32) * alpha
    out = np.zeros((DH, DW, 4), np.uint8); out[..., :3] = col; out[..., 3] = np.clip(m, 0, 255).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')

_BASE = {}
def base_block():
    if 'b' in _BASE: return _BASE['b']
    lay = Image.new('RGBA', (DW, DH))
    mus = hfaded(rect_mask(Y_MUS, Y_END - 10, 3, 1))
    lay.alpha_composite(volume(mus, C_MUS, alpha=0.96, r=12, relief=0.5))
    st = Image.new('RGBA', (DW, DH)); ds = ImageDraw.Draw(st)               # muscle fibre striations
    for k in range(9):
        y = Y_MUS + 22 + k * 19
        ds.line([(x, y + 4 * math.sin(x / 61 + k)) for x in range(0, DW, 12)], fill=(120, 50, 62, 110), width=2)
    st.putalpha(ImageChops.multiply(st.getchannel('A'), mus)); lay.alpha_composite(st)
    fat = hfaded(rect_mask(Y_FAT, Y_FAS + 2, 4, 2))
    lay.alpha_composite(volume(fat, C_FAT, alpha=0.95, r=10, relief=0.45))
    lob = Image.new('RGBA', (DW * 3, DH * 3)); dl = ImageDraw.Draw(lob)          # fat lobules (septa), supersampled
    rng = np.random.default_rng(7)
    for k in range(46):
        x = rng.uniform(10, DW - 10); y = rng.uniform(Y_FAT + 22, Y_FAS - 18); rx = rng.uniform(22, 34); ry = rx * 0.72
        dl.ellipse(((x - rx) * 3, (y - ry) * 3, (x + rx) * 3, (y + ry) * 3), outline=(190, 160, 100, 150), width=6)
    lob = lob.resize((DW, DH), Image.LANCZOS)
    lob.putalpha(ImageChops.multiply(lob.getchannel('A'), fat)); lay.alpha_composite(lob)
    fas = hfaded(rect_mask(Y_FAS, Y_MUS, 1.5, 3))
    lay.alpha_composite(volume(fas, C_FAS, alpha=0.98, r=4, relief=0.6))
    der = hfaded(rect_mask(Y_DER, Y_FAT, 1, 4))
    lay.alpha_composite(volume(der, C_DER, alpha=0.96, r=8, relief=0.5))
    epi = hfaded(rect_mask(Y_EPI, Y_DER, 1, 5))
    lay.alpha_composite(volume(epi, C_SKIN, alpha=1.0, r=3, relief=0.7))
    _BASE['b'] = (lay, fat, fas, der, epi)
    return _BASE['b']

_K = {}
def block(cell=0.0, nec=0.0, cut=0.0):
    """cell: cellulitis redness 0..1; nec: necrosis spread along the fascia 0..1; cut: incision 0..1"""
    key = (round(cell, 2), round(nec * 16) / 16, round(cut, 2))
    if key in _K: return _K[key]
    lay0, fat, fas, der, epi = base_block()
    lay = lay0.copy()
    if cell > 0:                                                                   # cellulitis: dermis + fat
        m = ImageChops.multiply(blob_mask(DW / 2, (Y_DER + Y_FAS) / 2 - 20, 250, 120).filter(ImageFilter.GaussianBlur(30)),
                                ImageChops.lighter(der, fat))
        lay.alpha_composite(tint(m, WARN, 0.55 * cell))
    if nec > 0:                                                                    # NF: along the fascial plane
        n = nec_mask(nec, Y_FAS - 24, Y_MUS + 6)
        edge = ImageChops.subtract(n.filter(ImageFilter.MaxFilter(7)), n)
        lay.alpha_composite(tint(edge, WARN, 0.9))
        lay.alpha_composite(volume(n, C_NEC, alpha=0.97, r=6, relief=0.5))
        sk = blob_mask(DW / 2, Y_EPI + 30, 60 + 30 * nec, 26).filter(ImageFilter.GaussianBlur(10))  # small skin change
        lay.alpha_composite(tint(ImageChops.multiply(sk, ImageChops.lighter(epi, der)), WARN, 0.6 * nec))
    if cut > 0:                                                                    # small incision down to the fascia
        big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
        x = DW / 2 - 150; w = 12 * cut
        d.polygon([((x - w) * SS, (Y_EPI - 2) * SS), ((x + w) * SS, (Y_EPI - 2) * SS),
                   ((x + 3) * SS, (Y_FAS + 4) * SS), ((x - 3) * SS, (Y_FAS + 4) * SS)], fill=255)
        cm = big.resize((DW, DH), Image.LANCZOS)
        lay.alpha_composite(tint(cm, (60, 40, 46), 0.95))
    pad = 40
    big = Image.new('RGBA', (DW + 2 * pad, DH + 2 * pad)); big.alpha_composite(lay, (pad, pad))
    big = drop_shadow(big, dx=10, dy=16, blur=14, opacity=0.55)
    _K[key] = big.crop((pad, pad, pad + DW, pad + DH))
    return _K[key]

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
    ex = lx - lab.width / 2 - 8 if lx > tip[0] else lx + lab.width / 2 + 8
    d.line([tip, (ex, ly)], fill=ICE + (int(210 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay); put(main, lab, lx, ly, a)

def finger(main, a, cx, cy, s, depth, slide):
    """gloved finger entering the incision (depth 0..1) then sweeping along the fascia (slide 0..1)"""
    if a <= 0: return
    x0 = DW / 2 - 150
    pts = [(x0, -60), (x0, lerp(-60, Y_FAS - 14, depth))]
    if slide > 0: pts = [(x0, -60), (x0, Y_FAS - 40), (x0 + 30, Y_FAS - 14), (x0 + 30 + 250 * slide, Y_FAS - 12)]
    P = [dp(p, cx, cy, s) for p in pts]
    x_min = min(p[0] for p in P) - 40; y_min = min(p[1] for p in P) - 40
    wv, hv = int(max(p[0] for p in P) - x_min + 80), int(max(p[1] for p in P) - y_min + 80)
    m = Image.new('L', (wv * SS, hv * SS)); d = ImageDraw.Draw(m)
    w = 34 * s
    Q = [((x - x_min) * SS, (y - y_min) * SS) for x, y in P]
    d.line(Q, fill=255, width=int(w * SS), joint='curve')
    for q in (Q[-1],): d.ellipse((q[0] - w * SS / 2, q[1] - w * SS / 2, q[0] + w * SS / 2, q[1] + w * SS / 2), fill=255)
    m = m.resize((wv, hv), Image.LANCZOS)
    tip = Q[-1]; dd = Q[-1][0] - Q[-2][0], Q[-1][1] - Q[-2][1]; L = math.hypot(*dd) or 1; ux, uy = dd[0] / L, dd[1] / L
    vol = volume(m, (176, 206, 236), alpha=0.97, r=6, relief=0.7)
    nl = Image.new('RGBA', (wv * SS, hv * SS)); dn = ImageDraw.Draw(nl)         # nail hint on the fingertip
    nx, ny = tip[0] - ux * w * SS * 0.25 - uy * w * SS * 0.12, tip[1] - uy * w * SS * 0.25 + ux * w * SS * 0.12
    dn.ellipse((nx - w * SS * 0.28, ny - w * SS * 0.2, nx + w * SS * 0.28, ny + w * SS * 0.2), fill=(225, 238, 250, 200),
               outline=(120, 150, 185, 220), width=int(2 * SS))
    vol.alpha_composite(nl.resize((wv, hv), Image.LANCZOS))
    put(main, drop_shadow(vol, dx=6, dy=8, blur=8, opacity=0.45), x_min + wv / 2, y_min + hv / 2, a)

def lerp(a, b, r): return a + (b - a) * r

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #8", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Red, swollen leg: cellulitis or worse?', 'MontXB', 42, ICE, 880), W / 2, 520, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('NECROTISING', 'MontBlack', 120, WHITE, 900)
    k2 = fit('FASCIITIS?', 'MontBlack', 120, WHITE, 900)
    for im, y in ((k1, 655), (k2, 780)):
        put(glow, im, W / 2, y, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, im, W / 2, y, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('ONLY 13 OF 89 DIAGNOSED ON ADMISSION', 'MontBlack', 44, SKY, 900, track=1), W / 2, 905,
        a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, tl('the skin hides what the fascia shows', 'MontXB', 40, WHITE), W / 2, 975 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.35)); s = 1.12
    put(main, scaled(block(0, 0.9, 0), s, 'hook'), W / 2, 1310 + 20 * (1 - hq), a * hq)

CX, CY, S = 440, 780, 1.12
STEPS = [(3.6, 'THE LAYERS', 'skin · subcutaneous fat · deep fascia · muscle'),
         (6.6, 'CELLULITIS', 'infection in the dermis and the fat'),
         (9.6, 'NECROTISING FASCIITIS', 'necrosis tracks along the fascia, past the skin')]
def s2(t, main, glow, fx):                                     # centrepiece: tissue block
    a = win(t, 3.3, 13.4, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('WHERE IT SPREADS', 'MontBlack', 62, WHITE, track=3), XC, 262, a * q, 1, 8 * (1 - q))
    label(main, 'Soft tissue of a limb  ·  schematic', XC, 335, a * eo(prog(t, 3.6, 4.1)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 3.5, 4.3)); cy = CY + 24 * (1 - bp)
    cell = eo(prog(t, 6.8, 7.6)) * (1 - eo(prog(t, 9.4, 9.9)))
    nec = eio(prog(t, 9.8, 11.6))
    put(main, scaled(block(cell, nec, 0), S, f'b{round(cell, 2)}_{round(nec * 16)}'), CX, cy, a * bp)
    P = lambda q_: dp(q_, CX, cy, S)
    la = a * eo(prog(t, 4.2, 4.7))
    pointer(main, la, P((600, (Y_EPI + Y_FAT) / 2)), 885, P((0, (Y_EPI + Y_FAT) / 2))[1], 'SKIN', ICE, 30)
    pointer(main, la, P((620, (Y_FAT + Y_FAS) / 2)), 885, P((0, (Y_FAT + Y_FAS) / 2))[1], 'FAT', ICE, 30)
    pointer(main, la, P((560, (Y_FAS + Y_MUS) / 2)), 885, P((0, (Y_FAS + Y_MUS) / 2))[1], 'FASCIA', ICE, 30)
    pointer(main, la, P((620, 430)), 885, P((0, 430))[1], 'MUSCLE', ICE, 30)
    ga = a * eo(prog(t, 11.0, 11.5))
    if ga > 0:
        pul = 0.6 + 0.4 * math.sin(t * 6)
        y = P((0, (Y_FAS + Y_MUS) / 2))[1]
        ImageDraw.Draw(glow).line([(P((60, 0))[0], y), (P((620, 0))[0], y)], fill=WARN + (int(120 * ga * pul),), width=40)
        label(main, 'SKIN: SMALL CHANGE', CX, P((0, 20))[1], ga, WARN, 32, 'MontBlack', 1)
    pa = a * eo(prog(t, 4.2, 4.6))
    if pa > 0:
        y = 1370; cur = 0
        for i, (t0, _, _) in enumerate(STEPS):
            if t >= t0: cur = i
        put(main, card(880, 230, outline=SKY + (160,)), XC, y, pa)
        t0, nm, sub = STEPS[cur]; q = eo(prog(t, t0, t0 + 0.35))
        put(main, pill(f'{cur + 1} / 3', 'MontBlack', 36, NAVY, SKY + (255,), padx=22, h=64), 170, y - 40, pa * q)
        im = fit(nm, 'MontBlack', 44, WARN if cur == 2 else WHITE, 650); put(main, im, 270 + im.width / 2, y - 40, pa * q)
        im = fit(sub, 'MontXB', 32, SKY, 790); put(main, im, 120 + im.width / 2, y + 50, pa * q)
    put(main, fit('Fascia necrotic, skin and muscle relatively spared', 'MontXB', 30, ICE, 880), XC, 1530,
        a * eo(prog(t, 10.4, 10.9)))
    label(main, 'Kazi et al.  ·  Surg J 2022', XC, 1585, a * eo(prog(t, 10.4, 10.9)), SUB, 28, 'MontXB', 0)

def s3(t, main, glow, fx):                                     # why it is missed: Wong 2003
    a = win(t, 13.2, 19.4, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 13.3, 13.8))
    put(main, fit('WHY IT IS MISSED', 'MontBlack', 62, WHITE, 840, track=3), XC, 262, a * p, 1, 10 * (1 - p))
    label(main, '89 consecutive patients with necrotising fasciitis', XC, 335, a * eo(prog(t, 13.5, 14.0)), SKY, 32, 'MontXB', 0)
    rows = [(13.8, '13/89', 'diagnosed on admission', 'early skin findings are scarce', WARN),
            (14.9, '70.8%', 'had diabetes', 'the commonest comorbidity', SKY),
            (16.0, 'ABX', 'antibiotics before admission', 'often masked how severe it was', WARN)]
    for i, (t0, num, l1, l2, col) in enumerate(rows):
        q = eo(prog(t, t0, t0 + 0.45))
        if q <= 0: continue
        y = 560 + i * 300 + 20 * (1 - q)
        put(main, card(880, 260, outline=col + (150,)), XC, y, a * q)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        nb = fit(num, 'MontBlack', 80, col, 290); put(main, nb, 125 + nb.width / 2, y, a * q * clamp(pk), clamp(pk, 0, 1.2))
        x0 = 125 + max(nb.width, 200) + 26
        im = fit(l1, 'MontBlack', 40, WHITE, 905 - x0); put(main, im, x0 + im.width / 2, y - 32, a * q)
        im = fit(l2, 'MontXB', 30, SUB, 905 - x0); put(main, im, x0 + im.width / 2, y + 34, a * q)
    label(main, 'Wong et al.  ·  JBJS Am 2003', XC, 1520, a * eo(prog(t, 16.8, 17.3)), ICE, 34, 'MontXB', 0)

LABS = ['CRP', 'WBC', 'Hb', 'Sodium', 'Creatinine', 'Glucose']
def s4(t, main, glow, fx):                                     # LRINEC
    a = win(t, 19.2, 28.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'THE LRINEC SCORE', XC, 262, a * eo(prog(t, 19.3, 19.8)), WHITE, 62, 'MontBlack', 3)
    label(main, '6 routine admission labs  ·  0 to 13 points', XC, 340, a * eo(prog(t, 19.5, 20.0)), SKY, 34, 'MontXB', 0)
    for i, nm in enumerate(LABS):
        q = eo(prog(t, 19.8 + 0.25 * i, 20.3 + 0.25 * i))
        if q <= 0: continue
        x = 200 + (i % 3) * 310; y = 470 + (i // 3) * 115 + 14 * (1 - q)
        put(main, pill(nm, 'MontBlack', 32, WHITE, (40, 70, 108, 235), padx=18, h=80), x, y, a * q)
    q = eo(prog(t, 21.6, 22.1))
    if q > 0:                                                  # gauge 0..13, cut-off 6
        d = ImageDraw.Draw(main); x0, x1, y = 120, 910, 790
        xv = lambda v: x0 + (x1 - x0) * v / 13
        d.rounded_rectangle((x0, y - 24, xv(6), y + 24), 20, fill=GOOD + (int(200 * a * q),))
        d.rounded_rectangle((xv(6), y - 24, x1, y + 24), 20, fill=WARN + (int(220 * a * q),))
        d.line([(xv(6), y - 60), (xv(6), y + 60)], fill=WHITE + (int(255 * a * q),), width=8)
        label(main, '6', xv(6), y - 96, a * q, WHITE, 52, 'MontBlack', 0)
        label(main, '0', x0 + 18, y + 66, a * q, SUB, 30, 'MontXB', 0)
        label(main, '13', x1 - 24, y + 66, a * q, SUB, 30, 'MontXB', 0)
        label(main, '6 or more: suspect NF', XC, y + 112, a * q, ICE, 30, 'MontXB', 0)
    q = eo(prog(t, 22.4, 22.9))
    if q > 0:
        y = 1040 + 20 * (1 - q)
        put(main, card(880, 210, outline=GOOD + (170,)), XC, y, a * q)
        label(main, 'THE ORIGINAL STUDY  ·  2004', XC, y - 62, a * q, GOOD, 34, 'MontBlack', 2)
        label(main, 'PPV 92%  ·  NPV 96%', XC, y + 8, a * q, WHITE, 56, 'MontBlack', 0)
        label(main, 'Wong et al. Crit Care Med 2004  ·  145 NF vs 309', XC, y + 70, a * q, SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 24.2, 24.7))
    if q > 0:
        y = 1300 + 20 * (1 - q)
        put(main, card(880, 260, outline=WARN + (190,)), XC, y, a * q)
        label(main, 'SENSITIVITY AT 6 OR MORE', XC, y - 84, a * q, WARN, 34, 'MontBlack', 2)
        pk = R.eback(prog(t, 24.4, 24.9))
        put(main, tl('68.2%', 'MontBlack', 84, WHITE), 300, y + 4, a * q * clamp(pk), clamp(pk, 0, 1.2))
        put(main, tl('43%', 'MontBlack', 84, WARN), 720, y + 4, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'meta-analysis, 23 studies', 300, y + 82, a * q, SUB, 28, 'MontXB', 0)
        label(main, 'limbs, prospective', 720, y + 82, a * q, SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 25.8, 26.3))
    if q > 0:
        pk = R.eback(prog(t, 25.8, 26.4))
        big = fit('IT CANNOT RULE NF OUT', 'MontBlack', 60, WHITE, 880)
        put(glow, big, XC, 1510, a * 0.3 * clamp(pk), clamp(pk, 0, 1.2)); put(main, big, XC, 1510, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'Fernando et al. Ann Surg 2019  ·  Hsiao et al. PLoS One 2020', XC, 1590, a * q, ICE, 28, 'MontXB', 0)

FCX, FCY, FS = 440, 780, 1.12
SIGNS = [(31.0, 'NO BLEEDING'), (31.8, 'DISHWATER PUS'), (32.6, 'NO RESISTANCE TO THE FINGER')]
def s5(t, main, glow, fx):                                     # finger test
    a = win(t, 28.4, 35.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'THE FINGER TEST', XC, 262, a * eo(prog(t, 28.5, 29.0)), WHITE, 62, 'MontBlack', 3)
    label(main, 'bedside, small incision down to the fascia', XC, 340, a * eo(prog(t, 28.7, 29.2)), SKY, 34, 'MontXB', 0)
    bp = eo(prog(t, 28.6, 29.2)); cy = FCY + 20 * (1 - bp)
    cut = eo(prog(t, 29.0, 29.5))
    put(main, scaled(block(0, 0.9, cut), FS, f'f{round(cut, 2)}'), FCX, cy, a * bp)
    depth = eio(prog(t, 29.5, 30.3)); slide = eio(prog(t, 30.4, 31.8))
    finger(main, a * eo(prog(t, 29.4, 29.8)), FCX, cy, FS, depth, slide)
    P = lambda q_: dp(q_, FCX, cy, FS)
    la = a * eo(prog(t, 29.6, 30.0))
    pointer(main, la, P((560, (Y_FAS + Y_MUS) / 2)), 885, P((0, (Y_FAS + Y_MUS) / 2))[1], 'FASCIA', ICE, 30)
    pointer(main, la, P((620, 430)), 885, P((0, 430))[1], 'MUSCLE', ICE, 30)
    for i, (t0, txt) in enumerate(SIGNS):
        q = eo(prog(t, t0, t0 + 0.4))
        if q <= 0: continue
        y = 1150 + i * 105 + 14 * (1 - q)
        put(main, card(880, 88, outline=WARN + (150,), r=28), XC, y, a * q)
        check(ImageDraw.Draw(main), 150, y, 40, WARN, a * q)
        im = fit(txt, 'MontBlack', 40, WHITE, 700); put(main, im, 200 + im.width / 2, y, a * q)
    q = eo(prog(t, 33.2, 33.7))
    label(main, 'Test described by Andreasen et al. 2001', XC, 1490, a * q, SKY, 32, 'MontXB', 0)
    label(main, 'Kazi et al. 2022, n = 40: sensitivity 86%, specificity 25%', XC, 1550, a * q, ICE, 30, 'MontXB', 0)

def s6(t, main, glow, fx):                                     # time is fascia
    a = win(t, 34.8, 40.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'TIME IS FASCIA', XC, 262, a * eo(prog(t, 34.9, 35.4)), WHITE, 62, 'MontBlack', 3)
    label(main, 'mortality by time to surgery  ·  meta-analysis', XC, 340, a * eo(prog(t, 35.1, 35.6)), SKY, 34, 'MontXB', 0)
    base = 1050; hmax = 520
    for i, (lab, v, col, t0) in enumerate([('WITHIN 6 H', 19, GOOD, 35.6), ('AFTER 6 H', 32, WARN, 36.2)]):
        q = eio(prog(t, t0, t0 + 0.9)); x = 330 + i * 380
        if q <= 0: continue
        h = hmax * v / 32 * q
        d = ImageDraw.Draw(main)
        d.rounded_rectangle((x - 95, base - h, x + 95, base), 24, fill=col + (int(225 * a),))
        label(main, f'{round(v * q)}%', x, base - h - 60, a, WHITE, 76, 'MontBlack', 0)
        label(main, lab, x, base + 50, a, col, 36, 'MontBlack', 2)
    label(main, 'Nawijn et al. World J Emerg Surg 2020  ·  OR 0.43', XC, base + 115, a * eo(prog(t, 36.8, 37.3)), ICE, 30, 'MontXB', 0)
    q = eo(prog(t, 37.6, 38.1))
    if q > 0:
        y = 1400 + 20 * (1 - q)
        put(main, card(880, 210, outline=WARN + (170,)), XC, y, a * q)
        label(main, 'Surgery delayed over 24 h', XC, y - 52, a * q, WHITE, 42, 'MontBlack', 0)
        label(main, 'mortality relative risk 9.4', XC, y + 10, a * q, WARN, 44, 'MontBlack', 0)
        label(main, 'Wong et al. JBJS Am 2003  ·  multivariate', XC, y + 68, a * q, SUB, 28, 'MontXB', 0)

TIPS = [('No fever? Not ruled out', 'SENS: FEVER 46%, BULLAE 25.2%, LOW BP 21%', 'Fernando et al. Ann Surg 2019  ·  23 studies'),
        ('Imaging? CT beats the plain film', 'CT SENS 88.5%, SPEC 93.3%  ·  X-RAY SENS 48.9%', 'never delay the surgical consult for a scan'),
        ('Suspect it? Debride within 12 h', 'MORTALITY 14% EARLY VS 25.8% LATE', 'EAST guideline  ·  Gelbard et al. 2018')]
def s7(t, main, glow, fx):
    a = win(t, 40.4, 46.8, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS & TRICKS', XC, 262, a * eo(prog(t, 40.5, 41.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 41.0 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 350 + 30 * (1 - q)
        put(main, card(880, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 40, WHITE, 690); put(main, lab, 125 + lab.width / 2, y - 52, a * q)
        lab = fit(sub, 'MontXB', 30, ICE, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 22, a * q)
        lab = fit(ref, 'MontXB', 30, SKY, 790, track=1); put(main, lab, 125 + lab.width / 2, y + 82, a * q)

def s8(t, main, glow, fx):                                     # takeaway + references
    a = win(t, 46.6, 52.5, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 330, a * eo(prog(t, 46.7, 47.2)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 46.9, 47.5))
    big = fit('LOW LRINEC', 'MontBlack', 104, WHITE, 880)
    put(glow, big, XC, 450, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 450, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'does not rule out NF', XC, 555, a * eo(prog(t, 47.3, 47.8)), WARN, 44, 'MontBlack', 0)
    label(main, 'Suspect it: early surgical consultation', XC, 625, a * eo(prog(t, 47.6, 48.1)), GOOD, 36, 'MontXB', 0)
    q = eo(prog(t, 48.0, 48.6))
    refs = ['Wong CH, et al. JBJS Am 2003;85:1454–60', 'Wong CH, et al. Crit Care Med 2004;32:1535–41',
            'Fernando SM, et al. Ann Surg 2019;269:58–65', 'Hsiao CT, et al. PLoS One 2020;15:e0227748',
            'Nawijn F, et al. World J Emerg Surg 2020;15:4', 'Gelbard RB, et al. J Trauma Acute Care Surg 2018',
            'Kazi FN, et al. Surg J 2022;8:e1–e7']
    put(main, card(880, 520, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 975, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE if i % 2 == 0 else SUB, 820), XC, 780 + i * 65, a * q)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1350, a * eo(prog(t, 48.6, 49.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7, s8]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
