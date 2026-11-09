"""Orthojets EP22 - DON'T MISS #5 - Pyogenic flexor tenosynovitis: Kanavel's 4 signs, how good they are, and why tonight.
Sources (verified on PubMed abstracts / metadata, 2026-11-09):
  Kennedy CD, Lauder AS, Pribaz JR, Kennedy SA. Differentiation between pyogenic flexor tenosynovitis and other finger
    infections. Hand (N Y) 2017;12(6):585-90. PMID 28720000, doi 10.1177/1558944717692089. Adults with surgical consultation
    for suspected PFT: sensitivity of the Kanavel signs 91.4% to 97.1%, specificity 51.3% to 69.2% (poor on an individual
    basis). Independent predictors: tenderness along the flexor tendon sheath, pain with passive extension, duration of
    symptoms less than 5 days; combined algorithm AUC 0.91.
  Pang HN et al. Factors affecting the prognosis of pyogenic flexor tenosynovitis. J Bone Joint Surg Am 2007;89(8):1742-8.
    PMID 17671013, doi 10.2106/JBJS.F.01356. 75 patients. Group I (no subcutaneous purulence, no digital ischaemia):
    no amputations; Group II (subcutaneous purulence, no ischaemia): amputation 8%; Group III (subcutaneous purulence +
    ischaemia): amputation 59%. Risk factors: age over 43, diabetes / PVD / renal failure, subcutaneous purulence,
    digital ischaemia, polymicrobial infection.
  Idarrha F et al. Predictive factors for outcomes in pyogenic flexor tenosynovitis: a 10-year analysis of 341 patients.
    Injury 2026;57(10):113631. PMID 42721565, doi 10.1016/j.injury.2026.113631. PFT is a surgical emergency. Consultation
    delay aOR 1.13 per day and diabetes aOR 3.19 for an unfavourable functional outcome; amputation 11.4%. "The absence of
    pain on passive extension in advanced stages is a grave prognostic sign indicating tissue necrosis rather than
    improvement."
  Frenkel Rutenberg T et al. Conservative treatment for PFT. J Plast Surg Hand Surg 2020;54(1):14-8 (epub 2019).
    PMID 31453741, doi 10.1080/2000656X.2019.1657434. Gold standard is irrigation and debridement of the flexor sheath;
    46 treated with IV antibiotics, 3 eventually operated, no fingers lost; only under a hand specialist's surveillance.
  Siska RC et al. A1 pulley tenderness ... in diagnosing PFT. Plast Reconstr Surg Glob Open 2022;10(3):e4165.
    PMID 35261842, doi 10.1097/GOX.0000000000004165. 33 patients: A1 pulley tenderness had the greatest odds ratio, PPV,
    specificity and accuracy compared with all Kanavel signs.
  Yi A et al. Radiographic soft tissue thickness differentiating PFT from other finger infections. J Hand Surg Am
    2019;44(5):394-9. PMID 30797654, doi 10.1016/j.jhsa.2019.01.013. Volar minus dorsal soft tissue thickness at the
    proximal phalanx of 7 mm or more: sensitivity 84%, specificity 74%, PPV 82%.
  Diagram (schematic, labelled, lateral view of a finger, volar side down, pointing right): metacarpal, proximal, middle
  and distal phalanges (lengths about 1 : 0.63 : 0.46), soft tissue envelope thicker on the volar side, flexor tendon
  sheath along the volar side from the distal palm (A1, at the MCP) to the DIP joint. Signs: flexed resting posture,
  fusiform (uniform) swelling, tenderness along the sheath, pain on passive extension.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(6.9, 0.9), (8.7, 0.9), (10.5, 0.9), (12.7, 1.4), (15.2, 1.2), (20.6, 1.4), (24.8, 1.6),
                     (29.2, 1.2), (32.6, 1.4), (38.6, 1.6), (43.0, 1.4)])

XC = 510
BONE = (200, 222, 242)
TISSUE = (70, 112, 160)
SS = 2

# ------------------------------------------------------------------ the finger, lateral view (local canvas 960 x 640)
LW, LH = 960, 640
MCP = (290.0, 200.0)
SEG = [230.0, 145.0, 105.0]             # proximal, middle, distal phalanx
MC0 = (40.0, 182.0)                     # metacarpal shaft start (in the palm)

def chain(th):
    """joint angles (deg, + = flexion, toward volar = down) -> MCP, PIP, DIP, tip, and segment directions"""
    pts = [MCP]; dirs = []; phi = 0.0
    for L, a in zip(SEG, th):
        phi += a; c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        dirs.append((c, s)); x, y = pts[-1]; pts.append((x + L * c, y + L * s))
    return pts, dirs

def samples(th, step=6):
    """centre-line samples along metacarpal + phalanges: (x, y, u, seg, (c, s)); u = 0..1 within each segment"""
    pts, dirs = chain(th); out = []
    mx, my = MC0; L = math.hypot(MCP[0] - mx, MCP[1] - my); c, s = (MCP[0] - mx) / L, (MCP[1] - my) / L
    for k in range(int(L / step) + 1):
        u = k * step / L; out.append((mx + c * u * L, my + s * u * L, u, -1, (c, s)))
    for i in range(3):
        x0, y0 = pts[i]; c, s = dirs[i]; n = int(SEG[i] / step) + 1
        for k in range(n + 1):
            u = k / n; out.append((x0 + c * u * SEG[i], y0 + s * u * SEG[i], u, i, (c, s)))
    return out

def tissue_r(seg, u):
    r = {-1: 84, 0: 76 - 8 * u, 1: 66 - 7 * u, 2: 58 - 16 * u ** 2}[seg]
    return r

_fc = {}
def finger_layer(th, swell=0.0, sheath=1.0, infl=0.0, build=1.0):
    """th: joint angles; swell 0..1 (fusiform); sheath alpha; infl 0..1 (sheath turns WARN); build 0..1"""
    key = (tuple(round(a, 1) for a in th), round(swell, 2), round(sheath, 2), round(infl, 2), round(build, 2))
    if key in _fc: return _fc[key]
    S = samples(th)
    big = Image.new('RGBA', (LW * SS, LH * SS))
    # soft tissue (volar-biased): union of circles, centre shifted volar by 10 px
    m = Image.new('L', big.size); dm = ImageDraw.Draw(m)
    ta = clamp(build * 3)
    for x, y, u, seg, (c, s) in S:
        r = tissue_r(seg, u) * (1 + 0.30 * swell * (1 if seg >= 0 else clamp(u * 3 - 1.6)))
        vx, vy = -s, c
        cx, cy = x + vx * 12, y + vy * 12
        dm.ellipse(((cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS), fill=255)
    # fingertip pad
    x, y, u, seg, (c, s) = S[-1]
    r = 46 * (1 + 0.3 * swell)
    dm.ellipse(((x + c * 10 - r) * SS, (y + s * 10 + 6 - r) * SS, (x + c * 10 + r) * SS, (y + s * 10 + 6 + r) * SS), fill=255)
    # palm side fades out to the left
    xs = np.clip((np.arange(LW * SS) / SS - 50) / 250, 0, 1) ** 2 * 255    # palm side fades in from x 50 to 300
    fade = Image.fromarray(np.repeat(xs[None, :], LH * SS, 0).astype(np.uint8))
    m = Image.fromarray((np.asarray(m, np.float32) * np.asarray(fade, np.float32) / 255).astype(np.uint8))
    m = m.filter(ImageFilter.GaussianBlur(1.5 * SS))
    col = tuple(int(a + (b - a) * 0.55 * swell) for a, b in zip(TISSUE, WARN))
    t_im = Image.new('RGBA', big.size, col + (0,)); t_im.putalpha(m.point(lambda v: int(v * 0.62 * ta)))
    edge = Image.fromarray(np.clip(np.asarray(m, np.int16) - np.asarray(m.filter(ImageFilter.MinFilter(7)), np.int16), 0, 255).astype(np.uint8))
    e_im = Image.new('RGBA', big.size, ICE + (0,)); e_im.putalpha(edge.point(lambda v: int(v * 0.9 * ta)))
    big.alpha_composite(t_im); big.alpha_composite(e_im)
    # flexor sheath: volar to the bones, from the distal palm (A1) to just past the DIP
    if sheath > 0:
        scol = tuple(int(a + (b - a) * infl) for a, b in zip(SKY, WARN))
        pts = []
        for x, y, u, seg, (c, s) in S:
            if (seg == -1 and u < 0.66) or (seg == 2 and u > 0.28): continue
            br = 22 if seg == -1 else (20 if seg == 0 else 17 if seg == 1 else 14)
            off = br + 16 + 6 * infl
            pts.append((x - s * off, y + c * off))
        sd = ImageDraw.Draw(big)
        w = int((16 + 8 * infl) * SS)
        sd.line([(px * SS, py * SS) for px, py in pts], fill=scol + (int(230 * sheath * ta),), width=w, joint='curve')
        for px, py in (pts[0], pts[-1]):
            sd.ellipse(((px * SS - w / 2), (py * SS - w / 2), (px * SS + w / 2), (py * SS + w / 2)), fill=scol + (int(230 * sheath * ta),))
        sd.line([(px * SS, py * SS) for px, py in pts], fill=WHITE + (int(200 * sheath * ta),), width=int(4 * SS), joint='curve')
    # bones: metacarpal + 3 phalanges, shortened at the joints
    bd = Image.new('L', big.size); bdd = ImageDraw.Draw(bd)
    for x, y, u, seg, (c, s) in S:
        L = SEG[seg] if seg >= 0 else 250
        gap = 13 / L
        if (seg >= 0 and (u < gap or u > 1 - gap)) or (seg == -1 and u > 1 - gap * 1.6): continue
        base = {-1: 26, 0: 24, 1: 20, 2: 17}[seg]
        if seg == 2: r = base * (1.15 - 0.55 * u) + 5 * math.exp(-((u - 0.95) / 0.06) ** 2)  # tuft
        else: r = base * (0.78 + 0.32 * math.exp(-((u - 0.08) / 0.1) ** 2) + 0.24 * math.exp(-((u - 0.94) / 0.08) ** 2))
        if seg == -1: r = base * (0.8 + 0.4 * math.exp(-((u - 1.0) / 0.12) ** 2))
        bdd.ellipse(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), fill=255)
    bd = Image.fromarray((np.asarray(bd, np.float32) * np.asarray(fade, np.float32) / 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SS))
    ba = clamp(build * 3 - 1)
    b_im = Image.new('RGBA', big.size, BONE + (0,)); b_im.putalpha(bd.point(lambda v: int(v * ba)))
    big.alpha_composite(b_im)
    im = big.resize((LW, LH), Image.LANCZOS)
    if len(_fc) > 80: _fc.clear()
    _fc[key] = im
    return im

FY0 = 440                                       # screen y of the local canvas top (MCP at screen y 640)
def L2S(p): return (p[0], FY0 + p[1])

def sheath_points(th, infl=0.0):
    S = samples(th); out = []
    for x, y, u, seg, (c, s) in S:
        if (seg == -1 and u < 0.66) or (seg == 2 and u > 0.28): continue
        br = 22 if seg == -1 else (20 if seg == 0 else 17 if seg == 1 else 14)
        off = br + 16 + 6 * infl
        out.append(L2S((x - s * off, y + c * off)))
    return out

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

def sign_row(main, a, y, n, txt, active):
    if a <= 0: return
    fill = (70, 34, 44, 210) if active else (30, 54, 84, 170)
    oc = WARN if active else SKY
    put(main, card(860, 92, fill=fill, outline=oc + (220,), r=30), XC, y, a)
    nm = tl(str(n), 'MontBlack', 48, oc); put(main, nm, 125, y, a)
    lab = tl(txt, 'MontBlack', 36, WHITE); put(main, lab, 175 + lab.width / 2 - 20, y, a)

def lerp3(a, b, k): return tuple(x + (y - x) * k for x, y in zip(a, b))
STRAIGHT = (4.0, 8.0, 5.0)
FLEXED = (20.0, 38.0, 22.0)

# ------------------------------------------------------------------ scenes
_mini = {}
def mini_finger(swell, infl, sc):
    key = (round(swell, 2), round(infl, 2), sc)
    if key not in _mini:
        f = finger_layer(FLEXED, swell, 1.0, infl).crop((60, 60, 900, 600))
        _mini[key] = f.resize((int(f.width * sc), int(f.height * sc)), Image.LANCZOS)
        if len(_mini) > 60: _mini.clear()
    return _mini[key]

def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #5", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Thorn prick 2 days ago,', 'MontXB', 46, WHITE), W / 2, 520 + 20 * (1 - q), a * q)
    put(main, tl('finger held bent and swollen:', 'MontXB', 46, WHITE), W / 2, 582 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('WASH OUT?', 'MontBlack', 124, SKY)
    put(glow, k1, W / 2, 735, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 735, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.45, 0.9))
    put(main, pill('FLEXOR SHEATH INFECTION', 'MontBlack', 36, WHITE, WARN + (255,), h=74), W / 2, 885 + 20 * (1 - q), a * q)
    g = eio(prog(t, 0.4, 1.4))
    q = eo(prog(t, 0.1, 0.6))
    mf = mini_finger(g, g, 0.72)
    cy = 1130 + 20 * (1 - q)
    put(main, mf, W / 2, cy, a * q)
    if g > 0:
        pul = 0.6 + 0.4 * math.sin(t * 6)
        x0, y0 = W / 2 - mf.width / 2, cy - mf.height / 2
        gd = ImageDraw.Draw(glow)
        for px, py in sheath_points(FLEXED, g)[::6]:
            sx, sy = x0 + (px - 60) * 0.72, y0 + (py - FY0 - 60) * 0.72
            gd.ellipse((sx - 22, sy - 22, sx + 22, sy + 22), fill=WARN + (int(70 * g * pul * a),))

SIGNS = ['Flexed resting posture', 'Fusiform swelling', 'Tender along the sheath', 'Pain on passive extension']
T_SIGN = [6.2, 8.0, 9.8, 11.6]
def s2(t, main, glow, fx):                                     # centrepiece: the finger and the 4 signs
    a = win(t, 3.3, 17.6, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl("KANAVEL'S 4 SIGNS", 'MontBlack', 58, WHITE, track=2), XC, 245, a * q, 1, 8 * (1 - q))
    label(main, 'pyogenic flexor tenosynovitis', XC, 312, a * eo(prog(t, 3.6, 4.1)), SKY, 32, 'MontXB', 0)
    build = prog(t, 3.6, 5.2)
    k1 = eio(prog(t, T_SIGN[0], T_SIGN[0] + 0.9))
    th = lerp3(STRAIGHT, FLEXED, k1)
    swell = eio(prog(t, T_SIGN[1], T_SIGN[1] + 1.0))
    infl = eo(prog(t, T_SIGN[2], T_SIGN[2] + 0.6))
    # sign 4: passive extension pushes the finger straighter, then it springs back
    pe = math.sin(math.pi * prog(t, T_SIGN[3] + 0.1, T_SIGN[3] + 1.9)) if T_SIGN[3] < t < T_SIGN[3] + 2.0 else 0.0
    th = lerp3(th, STRAIGHT, 0.55 * pe)
    put(main, finger_layer(th, swell, a, infl, build), LW / 2, FY0 + LH / 2, a)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    sp = sheath_points(th, infl)
    # labels
    la = a * win(t, 5.0, 6.2, 0.4, 0.3)
    pointer(main, la, sp[len(sp) // 3], 300, 1030, 'FLEXOR TENDON SHEATH', SKY, 30)
    pointer(main, la, L2S(chain(th)[0][1]), 560, 455, 'PIP', ICE, 30)
    pointer(main, la, L2S(chain(th)[0][2]), 780, 455, 'DIP', ICE, 30)
    # sign 3: tenderness dots running along the sheath
    if infl > 0:
        for k, (px, py) in enumerate(sp[::5]):
            pul = 0.5 + 0.5 * math.sin(t * 7 - k * 0.8)
            g.ellipse((px - 20, py - 20, px + 20, py + 20), fill=WARN + (int(110 * infl * pul * a),))
    # sign 4: the examiner's push on the fingertip (extension = upward) and the pain burst over the sheath
    if pe > 0:
        tip = L2S(chain(th)[0][3])
        bx, by = tip[0] + 10, tip[1] + 70
        d.line([(bx, by + 60), (bx, by)], fill=SKY + (int(255 * a * clamp(pe * 3)),), width=9)
        d.polygon([(bx, by - 26), (bx - 20, by + 2), (bx + 20, by + 2)], fill=SKY + (int(255 * a * clamp(pe * 3)),))
        px, py = sp[len(sp) // 2]
        for k in range(10):
            ang = k * math.pi / 5 + t; r0, r1 = 30, 30 + 40 * pe
            d.line([(px + r0 * math.cos(ang), py + r0 * math.sin(ang)), (px + r1 * math.cos(ang), py + r1 * math.sin(ang))],
                   fill=WARN + (int(255 * a * pe),), width=6)
    # the 4 sign rows
    for i, txt in enumerate(SIGNS):
        ra = a * eo(prog(t, T_SIGN[i] - 0.3, T_SIGN[i] + 0.2))
        active = T_SIGN[i] - 0.3 <= t < (T_SIGN[i + 1] - 0.3 if i < 3 else T_SIGN[3] + 2.2)
        sign_row(main, ra, 1150 + i * 112, i + 1, txt, active)
    fq = a * eo(prog(t, 14.4, 14.9))
    if fq > 0:
        label(main, 'Pain on passive extension:', XC, 1612, fq, WHITE, 38, 'MontBlack', 0)
        label(main, 'the sheath is under pressure', XC, 1666, fq, SUB, 30, 'MontXB', 0)

def range_bar(main, glow, a, y, name, lo, hi, col, p):
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    x0, x1 = 110, 910
    lab = tl(name, 'MontBlack', 42, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 56, a)
    d.rounded_rectangle((x0, y, x1, y + 40), 20, fill=(60, 90, 125, int(200 * a)))
    xl = x0 + (x1 - x0) * lo / 100; xh = x0 + (x1 - x0) * (lo + (hi - lo) * p) / 100
    if p > 0:
        d.rounded_rectangle((xl, y - 4, max(xl + 30, xh), y + 44), 24, fill=col + (int(255 * a),))
        g.rounded_rectangle((xl, y - 4, max(xl + 30, xh), y + 44), 24, fill=col + (int(80 * a),))
    vt = tl(f'{lo:.1f} to {hi:.1f}%', 'MontBlack', 44, col); put(main, vt, x1 - vt.width / 2 + 20, y - 56, a * clamp(p * 2))
    for v in (0, 50, 100):
        tx = x0 + (x1 - x0) * v / 100
        put(main, tl(f'{v}%', 'MontXB', 27, SUB), tx, y + 72, a * 0.9)

PRED = ['Tender along the sheath', 'Pain on passive extension', 'Symptoms under 5 days']
def s3(t, main, glow, fx):                                     # how good are the signs
    a = win(t, 17.4, 25.9, 0.4, 0.45)
    if a <= 0: return
    label(main, 'SENSITIVE, NOT SPECIFIC', XC, 250, a * eo(prog(t, 17.5, 18.0)), WHITE, 52, 'MontBlack', 2)
    label(main, 'each Kanavel sign on its own', XC, 318, a * eo(prog(t, 17.7, 18.2)), SKY, 32, 'MontXB', 0)
    q1 = a * eo(prog(t, 18.0, 18.4)); range_bar(main, glow, q1, 490, 'Sensitivity', 91.4, 97.1, GOOD, eio(prog(t, 18.2, 19.0)))
    q2 = a * eo(prog(t, 18.8, 19.2)); range_bar(main, glow, q2, 720, 'Specificity', 51.3, 69.2, WARN, eio(prog(t, 19.0, 19.8)))
    label(main, 'Kennedy et al, Hand 2017', XC, 880, a * eo(prog(t, 19.6, 20.0)), SUB, 28, 'MontXB', 0)
    hq = a * eo(prog(t, 21.0, 21.4))
    label(main, 'Independent predictors of PFT', XC, 1000, hq, SKY, 38, 'MontBlack', 0)
    for i, txt in enumerate(PRED):
        chip(main, a * eo(prog(t, 21.4 + 0.5 * i, 21.8 + 0.5 * i)), XC, 1120 + i * 118, txt, GOOD, True)
    cq = a * eo(prog(t, 23.4, 23.8))
    if cq > 0:
        p = R.eback(prog(t, 23.4, 23.9))
        big = tl('AUC 0.91', 'MontBlack', 76, GOOD)
        put(glow, big, XC, 1500, cq * 0.3, clamp(p, 0, 1.15)); put(main, big, XC, 1500, cq, clamp(p, 0, 1.15))
        label(main, 'all 3 combined', XC, 1580, cq, SUB, 30, 'MontXB', 0)

GROUPS = [('I', 'No pus under skin, no ischaemia', 0, GOOD), ('II', 'Subcutaneous pus', 8, SKY), ('III', 'Extensive pus + ischaemia', 59, WARN)]
def s4(t, main, glow, fx):                                     # why tonight: prognosis + delay
    a = win(t, 25.7, 33.6, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 25.8, 26.3))
    put(main, tl('WHY TONIGHT', 'MontBlack', 62, WHITE, track=3), XC, 250, a * q, 1, 8 * (1 - q))
    label(main, 'amputation rate by Pang group', XC, 318, a * eo(prog(t, 26.0, 26.5)), SKY, 32, 'MontXB', 0)
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    x0, x1 = 110, 760
    for i, (gr, nm, v, col) in enumerate(GROUPS):
        t0 = 26.4 + 0.6 * i; qq = a * eo(prog(t, t0, t0 + 0.4))
        if qq <= 0: continue
        y = 480 + 190 * i
        lab = tl(f'Group {gr}', 'MontBlack', 42, col); put(main, lab, x0 + lab.width / 2 - 20, y - 70, qq)
        lab = tl(nm, 'MontXB', 30, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 22, qq)
        d.rounded_rectangle((x0, y + 14, x1, y + 50), 18, fill=(60, 90, 125, int(200 * qq)))
        p = eio(prog(t, t0 + 0.1, t0 + 0.9))
        xe = x0 + (x1 - x0) * v / 100 * p
        if v > 0 and p > 0:
            d.rounded_rectangle((x0, y + 10, max(x0 + 40, xe), y + 54), 22, fill=col + (int(255 * qq),))
            g.rounded_rectangle((x0, y + 10, max(x0 + 40, xe), y + 54), 22, fill=col + (int(70 * qq),))
        vt = tl(f'{v * p:.0f}%', 'MontBlack', 54, col); put(main, vt, 870, y + 32, qq * clamp(p * 2 + (0.5 if v == 0 else 0)))
    label(main, 'Pang et al, JBJS Am 2007  ·  75 patients', XC, 1040, a * eo(prog(t, 28.4, 28.8)), SUB, 28, 'MontXB', 0)
    cq = a * eo(prog(t, 29.8, 30.3))
    if cq > 0:
        y = 1290 + 20 * (1 - cq / max(a, 1e-3))
        put(main, card(880, 300, fill=(66, 32, 38, 200), outline=WARN + (220,)), XC, y, cq)
        label(main, 'Each day of delay', XC, y - 92, cq, WHITE, 42, 'MontBlack', 0)
        label(main, 'OR 1.13 for a poor result', XC, y - 26, cq, WARN, 44, 'MontBlack', 0)
        label(main, 'adjusted  ·  diabetes OR 3.19', XC, y + 38, cq, SUB, 30, 'MontXB', 0)
        label(main, 'Idarrha et al, Injury 2026  ·  341 patients', XC, y + 96, cq, SKY, 28, 'MontXB', 0)

TIPS = [('Press the A1 pulley', 'MORE SPECIFIC THAN KANAVEL SIGNS', 'Siska et al, PRS Glob Open 2022'),
        ('Lateral X-ray of the finger', 'VOLAR MINUS DORSAL 7 MM OR MORE', 'Sens 84%, spec 74%  ·  Yi, JHS Am 2019'),
        ('Extension stops hurting?', 'LATE: NECROSIS, NOT IMPROVEMENT', 'Idarrha et al, Injury 2026')]
def s5(t, main, glow, fx):
    a = win(t, 33.4, 40.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 33.5, 34.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 34.1 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 29, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = ['Kennedy et al  ·  Hand 2017', 'Pang et al  ·  JBJS Am 2007', 'Idarrha et al  ·  Injury 2026',
        'Frenkel Rutenberg et al  ·  JPSHS 2020', 'Siska et al  ·  PRS Glob Open 2022', 'Yi et al  ·  J Hand Surg Am 2019']
def s6(t, main, glow, fx):
    a = win(t, 40.0, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 270, a * eo(prog(t, 40.1, 40.5)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 40.2, 40.7))
    big = tl("DON'T WAIT", 'MontBlack', 130, WHITE)
    put(glow, big, XC, 410, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 410, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'PFT is a surgical emergency', XC, 545, a * eo(prog(t, 40.4, 40.8)), SKY, 36, 'MontXB', 0)
    label(main, 'Gold standard: irrigate + debride', XC, 605, a * eo(prog(t, 40.6, 41.0)), GOOD, 36, 'MontXB', 0)
    label(main, 'IV antibiotics alone: uncomplicated only,', XC, 665, a * eo(prog(t, 40.8, 41.2)), SUB, 30, 'MontXB', 0)
    label(main, 'inpatient, under a hand surgeon', XC, 710, a * eo(prog(t, 40.8, 41.2)), SUB, 30, 'MontXB', 0)
    q = eo(prog(t, 41.2, 41.6))
    put(main, card(880, 470, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1030, a * q)
    for i, r in enumerate(REFS):
        label(main, r, XC, 860 + i * 68, a * q, ICE if i % 2 == 0 else SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1350, a * eo(prog(t, 41.8, 42.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
