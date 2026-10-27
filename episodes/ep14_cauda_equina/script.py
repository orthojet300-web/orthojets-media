"""Orthojets EP14 - DON'T MISS #2 - Cauda equina syndrome: the red flags, the bladder scan, the 48 h window.
Sources (verified on PubMed abstracts / metadata, 2026-10-27):
  Ahn UM, Ahn NU, Buchowski JM, Garrett ES, Sieber AN, Kostuik JP. Spine 2000;25(12):1515-22. PMID 10851100,
    doi 10.1097/00007632-200006150-00010. Meta-analysis, 42 studies, 322 patients (CES from lumbar disc herniation).
    Decompression within 48 h vs after 48 h: significantly better resolution of sensory and motor deficits and of
    urinary and rectal function. No extra benefit under 24 h vs 24-48 h.
  Dionne N et al. Musculoskelet Sci Pract 2019;42:125-33. PMID 31132655, doi 10.1016/j.msksp.2019.05.004.
    Systematic review, 7 studies, 569 patients, red flags vs MRI (reduced anal tone, leg pain, back pain, saddle
    anaesthesia, urinary retention, urinary incontinence, bowel incontinence): pooled sensitivity 0.19-0.43,
    specificity 0.62-0.88. More specific than sensitive; when present, they justify prompt diagnostic work-up.
  Katzouraki G, Zubairi AJ, Hershkovich O, Grevitt MP. Bone Joint J 2020;102-B(6):677-82. PMID 32475252,
    doi 10.1302/0301-620X.102B6.BJJ-2020-0195.R1. Prospective, 260 suspected CES, bladder scan before MRI.
    Bilateral sciatica: sensitivity 32.4%, PPV 17.2%. PVR 200 ml or more: sensitivity 94.1%, specificity 66.8%,
    PPV 29.9%, NPV 98.7%.
  Boktor J et al. Bone Joint J 2023;105-B(5):487-95. PMID 37121596, doi 10.1302/0301-620X.105B5.BJJ-2022-0739.R2.
    Meta-analysis, 7 studies, 1083 patients: 200 ml = the most accurate PVR threshold (sens 82%, spec 65%); adjunct only.
  Srikandarajah N et al. Spine 2015;40(8):580-3. PMID 25646751, doi 10.1097/BRS.0000000000000813.
    200 operated CES (139 CES-I, 61 CES-R). CES-I operated within 48 h of autonomic symptoms: bladder dysfunction at
    follow-up 10/64 = 15.6%; after 48 h 42/75 = 56%. (Within 24 h 11.1% vs 46.6%.)
  Najjar E et al. Bone Joint J 2026;108-B(3):416-24. PMID 41763246, doi 10.1302/0301-620X.108B3.BJJ-2025-0986.R2.
    Over 80% of urgent MRIs for suspected CES show no cauda equina compression.
  Diagram: midline sagittal lumbar spine, T2-style (anterior = viewer's left). L1-S1 bodies with lordosis, discs,
  canal with bright CSF, the cord tapering to the conus at about L1-L2, cauda equina roots below it, laminae /
  spinous processes behind the canal. A large central L4-L5 disc prolapse fills the canal and compresses the roots.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(6.8, 1.4), (10.8, 1.8), (17.6, 1.6), (24.6, 2.0), (31.2, 2.0), (36.9, 2.0),
                     (42.7, 1.8), (45.6, 2.2)])

XC = 510
# ------------------------------------------------------------------ diagram (local canvas DW x DH)
DW, DH = 640, 1110
BONE = (165, 200, 232)
X0S, Y0S = 120, 335          # screen position of the local canvas origin

LEVELS = [('T12', -40, 70), ('L1', 105, 225), ('L2', 262, 382), ('L3', 419, 539), ('L4', 576, 696), ('L5', 735, 855)]
DISCS = [(70, 105), (225, 262), (382, 419), (539, 576), (696, 735), (855, 892)]   # T12-L1 ... L5-S1
BX0, BX1 = 110, 285          # vertebral body (anterior, posterior wall)
CA0, CA1 = 293, 368          # canal (anterior, posterior wall)

def off(y):
    """lordosis: mid-lumbar bodies shift anteriorly (left); below S1 the sacrum runs posteriorly (right)"""
    if y < 892: return -42 * math.sin(math.pi * max(0.0, y + 40) / 932)
    return (y - 892) * 0.62 + 0.0

def M(pts):
    return [(x + off(y), y) for x, y in pts]

def dense(pts, step=12):
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    return out

def rrect(x0, y0, x1, y1, r):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        pts += [(cx + r * math.cos(math.radians(a0 + k * 15)), cy + r * math.sin(math.radians(a0 + k * 15))) for k in range(7)]
    return pts

SS = 3
def mask(polys):
    big = Image.new('L', (DW * SS, DH * SS)); d = ImageDraw.Draw(big)
    for p in polys:
        d.polygon([(x * SS, y * SS) for x, y in M(dense(p))], fill=255)
    return big.resize((DW, DH), Image.LANCZOS)

def bone_tex(m, seed, dim=1.0, col=BONE):
    M_ = np.asarray(m, np.float32) / 255
    B1 = np.asarray(m.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
    rim = np.clip((1 - B1) * 3.0, 0, 1)
    rng = np.random.default_rng(seed)
    n1 = np.asarray(Image.fromarray((rng.random((DH // 4, DW // 4)) * 255).astype(np.uint8)).resize((DW, DH), Image.BICUBIC), np.float32) / 255
    dens = (0.36 + 0.55 * rim) * (0.88 + 0.24 * n1) * dim
    inner = np.clip(M_ * dens, 0, 1.2)
    img = Image.fromarray(np.clip(np.stack([inner * c for c in col], -1), 0, 255).astype(np.uint8)).convert('RGBA')
    img.putalpha(Image.fromarray(np.clip(inner * 255 * 1.05, 0, 255).astype(np.uint8)))
    mb = m.point(lambda v: 255 if v > 128 else 0)
    edge = ImageChops.subtract(mb, mb.filter(ImageFilter.MinFilter(5)))
    e = Image.new('RGBA', m.size, ICE + (0,)); e.putalpha(edge.point(lambda v: int(170 * dim) if v > 40 else 0))
    img.alpha_composite(e)
    return img

def solid(m, col, alpha=1.0):
    im = Image.new('RGBA', (DW, DH), col + (0,)); im.putalpha(m.point(lambda v: int(v * alpha))); return im

def build():
    bodies = [rrect(BX0, a, BX1, b, 16) for _, a, b in LEVELS]
    # sacrum (S1 body + sacral mass running posteriorly-down)
    sac = [(BX0 + 4, 896), (BX1, 896), (BX1 + 6, 1000), (BX1 - 10, 1110), (BX0 + 60, 1110), (BX0 + 10, 1000)]
    sac_post = [(CA1 + 2, 920), (CA1 + 70, 930), (CA1 + 60, 1110), (CA1 + 4, 1110)]
    # posterior elements per level: lamina behind the canal + spinous process pointing posteriorly-down
    post = []
    for _, a, b in LEVELS[1:]:
        c = (a + b) / 2
        post.append([(CA1 + 2, a + 6), (CA1 + 40, a - 4), (CA1 + 130, c - 10), (CA1 + 175, c + 18), (CA1 + 168, c + 52),
                     (CA1 + 120, c + 62), (CA1 + 40, b + 20), (CA1 + 2, b - 6)])
    discs = [rrect(BX0 + 6, a + 3, BX1 - 2, b - 3, 12) for a, b in DISCS]
    nuclei = [rrect(BX0 + 50, a + 9, BX1 - 50, b - 9, 9) for a, b in DISCS]
    mb = mask(bodies + [sac]); mp = mask(post + [sac_post]); md = mask(discs); mn = mask(nuclei)
    bone = Image.new('RGBA', (DW, DH))
    bone.alpha_composite(bone_tex(mb, 3))
    bone.alpha_composite(bone_tex(mp, 9, dim=0.82))
    disc = solid(md, (70, 104, 140), 0.95)
    nb = solid(mn.filter(ImageFilter.GaussianBlur(3)), (200, 225, 245), 0.85)
    disc.alpha_composite(nb)
    # thecal sac (bright CSF on T2) from the top down to about S2
    sac_pts = [(CA0 + 4, -20), (CA1 - 4, -20), (CA1 - 4, 950), (CA1 - 18, 1010), (CA0 + 30, 1012), (CA0 + 6, 960)]
    msac = mask([sac_pts])
    csf = solid(msac.filter(ImageFilter.GaussianBlur(2)), (215, 232, 248), 0.92)
    soft = solid(ImageChops.lighter(mb, mp).filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.GaussianBlur(20)), (110, 150, 190), 0.18)
    layers = [soft, disc, bone, csf]
    for im in layers:     # fade the top (T12) and the bottom of the sacrum
        yy = np.arange(DH)[:, None] * np.ones((1, DW))
        f = np.clip((yy - 10) / 110.0, 0, 1) * np.clip((1055 - yy) / 60.0, 0, 1)
        im.putalpha(Image.fromarray((np.asarray(im.getchannel('A'), np.float32) * f).astype(np.uint8)))
    return layers, msac

LAYERS, MSAC = build()
CONUS_Y = 244                       # conus tip at the L1-L2 disc
CORD_X = 333
ROOT_X = [306, 314, 322, 330, 338, 346, 354]
HERN = (696, 735)                   # L4-L5 disc

def hern_front(y, p):
    """posterior edge (x, local, before off) of the herniation at height y, growth p 0..1"""
    a, b = HERN[0] - 30, HERN[1] + 42
    if y <= a or y >= b: return None
    u = (y - a) / (b - a)
    return CA0 - 6 + p * 62 * math.sin(math.pi * u) ** 0.8

def roots_pts(i, p, sq=0.0):
    x = ROOT_X[i]; pts = []
    for y in range(CONUS_Y - 30, 1000, 10):
        f = clamp((y - CONUS_Y + 30) / 120)
        xx = CORD_X + (x - CORD_X) * f
        if y > 900: xx += (i - 3) * 2.0 * (y - 900) / 100
        hf = hern_front(y, p)
        if hf is not None:
            lo = hf + 6 + (CA1 - 8 - hf - 6) * (i / 6)     # roots squeezed into the gap behind the prolapse
            xx = max(xx, lo) if hf > CA0 else xx
        pts.append((xx + off(y), y))
    return pts

_dc = {}
def diagram(p, cmp_hl):
    key = (round(p, 2), round(cmp_hl, 2))
    if key in _dc: return _dc[key]
    soft, disc, bone, csf = LAYERS
    im = Image.new('RGBA', (DW, DH))
    for L in (soft, csf, disc, bone): im.alpha_composite(L)
    d = ImageDraw.Draw(im)
    # cord: from the top, tapering into the conus
    cpts_l, cpts_r = [], []
    for y in range(-10, CONUS_Y + 1, 8):
        w = 15 if y < CONUS_Y - 90 else 15 * ((CONUS_Y - y) / 90) ** 0.7
        cpts_l.append((CORD_X - w + off(y), y)); cpts_r.append((CORD_X + w + off(y), y))
    d.polygon(cpts_l + cpts_r[::-1], fill=(58, 84, 118, 255))
    for i in range(len(ROOT_X)):
        d.line(roots_pts(i, p), fill=(52, 78, 112, 255), width=4, joint='curve')
    if p > 0:
        pts = []
        for y in range(HERN[0] - 30, HERN[1] + 43, 4):
            hf = hern_front(y, p); pts.append(((hf if hf else CA0 - 6) + off(y), y))
        base = [(CA0 - 8 + off(y), y) for y in range(HERN[1] + 42, HERN[0] - 31, -6)]
        d.polygon(pts + base, fill=(185, 214, 240, 255))
        ym = (HERN[0] + HERN[1]) / 2          # extruded nucleus tracking through the posterior annulus
        d.polygon([(BX1 - 52 + off(ym), ym - 7 * p), (CA0 - 4 + off(ym), ym - 12 * p), (CA0 - 4 + off(ym), ym + 12 * p),
                   (BX1 - 52 + off(ym), ym + 7 * p)], fill=(185, 214, 240, 255))
        d.line(pts, fill=WARN + (int(255 * min(1, p * 1.5)),), width=4)
        if cmp_hl > 0:
            g = Image.new('L', (DW, DH)); gd = ImageDraw.Draw(g)
            cy = (HERN[0] + HERN[1]) / 2
            gd.ellipse((CA0 - 10 + off(cy), cy - 60, CA1 + 12 + off(cy), cy + 60), fill=255)
            g = g.filter(ImageFilter.GaussianBlur(14))
            im.alpha_composite(solid(g, WARN, 0.5 * cmp_hl))
    if len(_dc) > 80: _dc.clear()
    _dc[key] = im
    return im

def sp(q):       # local (pre-offset) point -> screen
    x, y = q
    return X0S + x + off(y), Y0S + y

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29, side='right'):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=1)
    ex = lx - lab.width / 2 + 14 if side == 'right' else lx + lab.width / 2 - 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def flag_chip(main, a, x, y, txt, col=WHITE):
    if a <= 0: return
    put(main, card(410, 86, fill=(60, 30, 50, 190), outline=SKY + (200,), r=26), x, y, a)
    d = ImageDraw.Draw(main)
    dot(d, x - 170, y, 9, SKY, a)
    lab = tl(txt, 'MontXB', 31, col); put(main, lab, x - 145 + lab.width / 2 - 14, y, a)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #2", 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 450, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('Back pain + numb saddle?', 'MontXB', 50, WHITE), W / 2, 580 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('48 HOURS', 'MontBlack', 138, SKY)
    put(glow, k1, W / 2, 755, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 755, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.5, 1.0))
    put(main, tl('to save the bladder', 'MontXB', 56, WHITE), W / 2, 935 + 25 * (1 - q), a * q)
    q = eo(prog(t, 0.9, 1.4))
    put(main, pill('CAUDA EQUINA SYNDROME', 'MontBlack', 36, NAVY, GOOD + (255,), h=74), W / 2, 1110 + 20 * (1 - q), a * q)

def s2(t, main, glow, fx):                                     # centrepiece: sagittal lumbar spine
    a = win(t, 3.3, 19.4, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl('CAUDA EQUINA', 'MontBlack', 50, WHITE, track=2), XC, 248, a * q, 1, 8 * (1 - q))
    label(main, 'Lumbar spine  ·  sagittal MRI (T2)', XC, 312, a * eo(prog(t, 3.6, 4.1)), SKY, 32, 'MontXB', 0)
    bp = eo(prog(t, 3.5, 4.6))
    hp = eio(prog(t, 8.0, 9.6))
    hl = eo(prog(t, 9.4, 9.9)) * (0.7 + 0.3 * math.sin(t * 4))
    put(main, diagram(round(hp, 2), round(hl, 2)), X0S + DW / 2, Y0S + DH / 2 + 24 * (1 - bp), a * bp)
    # level labels (left of the bodies)
    for k, (nm, y0, y1) in enumerate(LEVELS[1:]):
        lq = a * eo(prog(t, 4.4 + 0.12 * k, 4.8 + 0.12 * k))
        cy = (y0 + y1) / 2
        x, y = sp((BX0, cy))
        label(main, nm, x - 52, y, lq, SUB, 30, 'MontBlack', 0)
    x, y = sp((BX0 + 40, 960)); label(main, 'S1', x - 95, y - 40, a * eo(prog(t, 5.0, 5.4)), SUB, 30, 'MontBlack', 0)
    # anatomy labels (right column)
    la = a * (1 - eo(prog(t, 16.8, 17.3)))
    pointer(main, la * eo(prog(t, 4.9, 5.4)), sp((CORD_X, 120)), 805, 430, 'SPINAL CORD', ICE, 29)
    pointer(main, la * eo(prog(t, 5.3, 5.8)), sp((CORD_X, CONUS_Y - 10)), 805, 560, 'CONUS  ·  L1-L2', SKY, 29)
    pointer(main, la * eo(prog(t, 5.8, 6.3)), sp((340, 470)), 805, 760, 'CAUDA EQUINA', SKY, 29)
    pointer(main, la * eo(prog(t, 6.2, 6.7)) * (1 - eo(prog(t, 7.3, 7.7))), sp((200, 715)), 805, 1000, 'L4-L5 DISC', ICE, 29)
    nq = a * win(t, 5.4, 8.0, 0.35, 0.3)
    label(main, 'Below the conus: only nerve roots', XC, 1520, nq, ICE, 38, 'MontBlack', 0)
    label(main, 'bladder, bowel, saddle, both legs', XC, 1585, nq, SUB, 34, 'MontXB', 0)
    # prolapse
    pa = la * eo(prog(t, 9.3, 9.8))
    pointer(main, pa, sp((CA0 + 30, 715)), 790, 1000, 'PROLAPSE', WARN, 29)
    wq = a * win(t, 8.1, 11.3, 0.35, 0.35)
    label(main, 'A large central disc prolapse', XC, 1520, wq, WARN, 40, 'MontBlack', 0)
    label(main, 'squeezes all the roots at once', XC, 1585, wq, ICE, 36, 'MontXB', 0)
    # red flags
    rq = a * eo(prog(t, 11.3, 11.8))
    label(main, 'RED FLAGS', XC, 1412, rq, SKY, 42, 'MontBlack', 6)
    flags = [('Saddle numbness', 11.6), ('Bladder change', 12.4), ('Bowel change', 13.2), ('Bilateral sciatica', 14.0)]
    for i, (txt, t0) in enumerate(flags):
        fq = a * eo(prog(t, t0, t0 + 0.45))
        flag_chip(main, fq, 293 + 434 * (i % 2), 1500 + 104 * (i // 2) + 18 * (1 - fq / max(a, 1e-3)), txt)

def gauge(main, glow, y, title, lo, hi, col, a, t0, t, note):
    if a <= 0: return
    put(main, card(860, 270), XC, y + 8, a)
    lab = tl(title, 'MontBlack', 44, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 70, a)
    x0, x1 = 120, 900; yy = y + 10
    d = ImageDraw.Draw(main)
    d.rounded_rectangle((x0, yy - 14, x1, yy + 14), 14, fill=(60, 90, 125, int(220 * a)))
    p = eio(prog(t, t0, t0 + 0.9))
    if p > 0:
        xa = x0 + (x1 - x0) * lo / 100; xb = x0 + (x1 - x0) * (lo + (hi - lo) * p) / 100
        d.rounded_rectangle((xa, yy - 22, max(xa + 28, xb), yy + 22), 22, fill=col + (int(255 * a),))
        ImageDraw.Draw(glow).rounded_rectangle((xa, yy - 22, max(xa + 28, xb), yy + 22), 22, fill=col + (int(80 * a),))
    for v in (0, 50, 100):
        xv = x0 + (x1 - x0) * v / 100
        label(main, f'{v}%', min(max(xv, 150), 870), yy + 54, a, SUB, 30, 'MontXB', 0)
    vt = tl(f'{lo}-{hi}%', 'MontBlack', 54, col); put(main, vt, 905 - vt.width / 2 + 20, y - 70, a * clamp(p * 2))
    label(main, note, XC, y + 102, a * eo(prog(t, t0 + 0.6, t0 + 1.0)), col, 28, 'MontXB', 0)

def s3(t, main, glow, fx):                                     # red flags: specific, not sensitive
    a = win(t, 19.2, 25.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'RED FLAGS ARE LATE', XC, 262, a * eo(prog(t, 19.3, 19.8)), WHITE, 58, 'MontBlack', 2)
    label(main, 'pooled vs MRI  ·  7 studies, 569 patients', XC, 340, a * eo(prog(t, 19.5, 20.0)), SKY, 34, 'MontXB', 0)
    q1 = eo(prog(t, 19.9, 20.4)); gauge(main, glow, 560, 'Sensitivity', 19, 43, WARN, a * q1, 20.1, t, 'each flag alone misses most CES cases')
    q2 = eo(prog(t, 21.0, 21.5)); gauge(main, glow, 870, 'Specificity', 62, 88, GOOD, a * q2, 21.2, t, 'when a flag is present, take it seriously')
    q3 = eo(prog(t, 22.4, 22.9))
    if q3 > 0:
        y = 1160 + 25 * (1 - q3)
        put(main, card(860, 210, fill=(60, 34, 40, 190), outline=WARN + (180,)), XC, y, a * q3)
        label(main, 'BILATERAL SCIATICA', XC, y - 55, a * q3, WHITE, 40, 'MontBlack', 2)
        label(main, 'Sensitivity 32.4%  ·  PPV 17.2%', XC, y + 10, a * q3, WARN, 42, 'MontBlack', 0)
        label(main, 'Katzouraki et al  ·  Bone Joint J 2020', XC, y + 68, a * q3, SUB, 30, 'MontXB', 0)
    label(main, 'No red flag does not mean no CES', XC, 1385, a * eo(prog(t, 23.3, 23.8)), ICE, 38, 'MontBlack', 0)
    label(main, 'Dionne et al  ·  Musculoskelet Sci Pract 2019', XC, 1450, a * eo(prog(t, 23.5, 24.0)), SUB, 30, 'MontXB', 0)

def bladder(main, glow, cx, cy, fill, a, t):
    if a <= 0: return
    w, h = 300, 330
    im = Image.new('RGBA', (w * 2, h * 2)); d = ImageDraw.Draw(im)
    pts = [(0.5 + 0.46 * math.cos(math.radians(a_)) * (1 - 0.18 * max(0, math.sin(math.radians(a_)))),
            0.46 + 0.42 * math.sin(math.radians(a_))) for a_ in range(-90, 270, 10) if not (75 < a_ < 105)]
    k = next(i for i, (x, y) in enumerate(pts) if x < 0.5 and y > 0.7)
    pts = pts[:k] + [(0.57, 0.92), (0.56, 1.0), (0.44, 1.0), (0.43, 0.92)] + pts[k:]
    body = [(w * 2 * x, h * 2 * y) for x, y in pts]
    mk = Image.new('L', im.size); ImageDraw.Draw(mk).polygon(body, fill=255)
    lvl = h * 2 * (0.93 - 0.82 * fill)
    wav = Image.new('L', im.size); wd = ImageDraw.Draw(wav)
    wd.polygon([(0, lvl + 10 * math.sin(t * 3))] + [(x, lvl + 10 * math.sin(t * 3 + x / 60)) for x in range(0, w * 2 + 20, 20)] +
               [(w * 2, h * 2), (0, h * 2)], fill=255)
    urine = Image.new('RGBA', im.size, (236, 196, 98, 0)); urine.putalpha(ImageChops.multiply(mk, wav).point(lambda v: int(v * 0.8)))
    shell = Image.new('RGBA', im.size, (60, 90, 125, 0)); shell.putalpha(mk.point(lambda v: int(v * 0.55)))
    im.alpha_composite(shell); im.alpha_composite(urine)
    d.line(body + [body[0]], fill=ICE + (255,), width=8, joint='curve')
    im = im.resize((w, h), Image.LANCZOS)
    put(main, im, cx, cy, a)

def s4(t, main, glow, fx):                                     # bladder scan: PVR 200 ml
    a = win(t, 25.4, 32.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'SCAN THE BLADDER', XC, 262, a * eo(prog(t, 25.5, 26.0)), WHITE, 58, 'MontBlack', 2)
    label(main, 'post-void residual (PVR) before the MRI', XC, 340, a * eo(prog(t, 25.7, 26.2)), SKY, 34, 'MontXB', 0)
    fill = eio(prog(t, 26.2, 28.0)) * 0.78
    bq = a * eo(prog(t, 25.9, 26.4))
    bladder(main, glow, 270, 640, fill, bq, t)
    # 200 ml line
    lq = a * eo(prog(t, 27.4, 27.8))
    yl = 640 + 165 - 330 * (0.93 - 0.82 * 0.0) + 330 * (0.82 * 0.0)   # top of the shell reference
    y200 = 640 - 165 + 330 * (0.93 - 0.82 * (200 / 320 * 0.78))
    d = ImageDraw.Draw(main)
    if lq > 0:
        for x in range(90, 470, 28): d.line([(x, y200), (x + 16, y200)], fill=WARN + (int(255 * lq),), width=5)
        put(main, pill('200 ml', 'MontBlack', 32, NAVY, WARN + (255,), padx=18, h=54), 270, y200, lq)
    ml = int(round(fill / 0.78 * 320 / 10) * 10)
    label(main, f'PVR {ml} ml', 270, 860, bq, ICE, 40, 'MontBlack', 0)
    # stats (right)
    sq = a * eo(prog(t, 27.9, 28.4))
    if sq > 0:
        put(main, card(430, 470, fill=(24, 60, 60, 190), outline=GOOD + (200,)), 718, 640, sq)
        label(main, 'PVR 200 ml +', 718, 455, sq, WHITE, 38, 'MontBlack', 0)
        label(main, 'sensitivity', 718, 525, sq, SUB, 30, 'MontXB', 0)
        p = R.eback(prog(t, 28.2, 28.7))
        put(main, tl('94.1%', 'MontBlack', 74, GOOD), 718, 590, sq * clamp(p), clamp(p, 0, 1.15))
        label(main, 'NPV', 718, 675, sq, SUB, 30, 'MontXB', 0)
        p = R.eback(prog(t, 28.6, 29.1))
        put(main, tl('98.7%', 'MontBlack', 74, GOOD), 718, 740, sq * clamp(p), clamp(p, 0, 1.15))
        label(main, 'Katzouraki 2020', 718, 828, sq, SKY, 30, 'MontXB', 0)
    mq = a * eo(prog(t, 29.4, 29.9))
    if mq > 0:
        y = 1110 + 25 * (1 - mq / max(a, 1e-3))
        put(main, card(860, 250), XC, y, mq)
        label(main, 'META-ANALYSIS  ·  7 STUDIES', XC, y - 72, mq, WHITE, 36, 'MontBlack', 1)
        label(main, '200 ml = the most accurate cut-off', XC, y - 10, mq, GOOD, 38, 'MontBlack', 0)
        label(main, 'an adjunct to the exam, not a rule-out alone', XC, y + 50, mq, ICE, 30, 'MontXB', 0)
        label(main, 'Boktor et al  ·  Bone Joint J 2023', XC, y + 97, mq, SUB, 30, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # timing: 48 h
    a = win(t, 32.4, 38.4, 0.4, 0.45)
    if a <= 0: return
    label(main, 'TIME IS BLADDER', XC, 262, a * eo(prog(t, 32.5, 33.0)), WHITE, 58, 'MontBlack', 2)
    label(main, 'decompression within 48 h of onset', XC, 340, a * eo(prog(t, 32.7, 33.2)), SKY, 34, 'MontXB', 0)
    # clock bar 0 -> 72 h
    x0, x1, y = 110, 910, 520
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    q = a * eo(prog(t, 32.9, 33.3))
    d.rounded_rectangle((x0, y - 16, x1, y + 16), 16, fill=(60, 90, 125, int(220 * q)))
    p = eio(prog(t, 33.2, 34.4))
    x48 = x0 + (x1 - x0) * 48 / 72
    if p > 0:
        xe = x0 + (x48 - x0) * p
        d.rounded_rectangle((x0, y - 22, max(x0 + 30, xe), y + 22), 22, fill=GOOD + (int(255 * a),))
        g.rounded_rectangle((x0, y - 22, max(x0 + 30, xe), y + 22), 22, fill=GOOD + (int(90 * a),))
    for hv in (0, 24, 48, 72):
        xv = x0 + (x1 - x0) * hv / 72
        label(main, f'{hv} h', min(max(xv, 140), 880), y + 60, q, ICE if hv != 48 else GOOD, 32 if hv != 48 else 38, 'MontBlack', 0)
    label(main, 'Better sensory, motor, bladder and bowel recovery', XC, 650, a * eo(prog(t, 34.2, 34.7)), ICE, 30, 'MontXB', 0)
    label(main, 'Ahn et al  ·  Spine 2000  ·  42 studies, 322 patients', XC, 705, a * eo(prog(t, 34.4, 34.9)), SUB, 30, 'MontXB', 0)
    # CES-I bladder outcome bars
    hq = a * eo(prog(t, 34.9, 35.4))
    label(main, 'INCOMPLETE CES (CES-I)', XC, 830, hq, SKY, 38, 'MontBlack', 1)
    label(main, 'bladder dysfunction at follow-up', XC, 885, hq, SUB, 30, 'MontXB', 0)
    yb, hmax = 1360, 320
    d.line([(150, yb), (870, yb)], fill=ICE + (int(160 * hq),), width=4)
    for x, v, col, l1, t0 in [(330, 15.6, GOOD, 'within 48 h', 35.3), (690, 56.0, WARN, 'after 48 h', 35.8)]:
        pb = eio(prog(t, t0, t0 + 0.9))
        if pb > 0 and hq > 0:
            h_ = hmax * (v / 56) * pb
            put(main, card(190, max(40, int(h_)), fill=col + (220,), outline=col + (255,), r=18), x, yb - h_ / 2, hq)
            put(main, tl(f'{v * pb:.1f}%', 'MontBlack', 64, col), x, yb - h_ - 50, hq * clamp(pb * 3))
        label(main, l1, x, yb + 45, hq * eo(prog(t, t0, t0 + 0.4)), WHITE, 34, 'MontBlack', 0)
    label(main, 'Srikandarajah et al  ·  Spine 2015  ·  139 CES-I', XC, 1490, a * eo(prog(t, 36.2, 36.7)), SUB, 30, 'MontXB', 0)
    label(main, 'Operate before it becomes retention', XC, 1565, a * eo(prog(t, 36.4, 36.9)), GOOD, 38, 'MontBlack', 0)

TIPS = [('Ask every back pain', 'SADDLE, BLADDER, BOWEL, BOTH LEGS', 'Red flags: specific, not sensitive'),
        ('Scan the bladder', 'PVR 200 ML OR MORE: SUSPECT CES', 'Katzouraki 2020: NPV 98.7%'),
        ('Red flag = urgent MRI', 'OVER 80% OF URGENT MRIS: NO COMPRESSION', 'Najjar, Bone Joint J 2026')]
def s6(t, main, glow, fx):
    a = win(t, 38.2, 44.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 38.3, 38.8)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 38.9 + 1.5 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 42, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 27, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 27, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)

REFS = [('Ahn UM et al.', 'Spine. 2000;25(12):1515–22'),
        ('Dionne N et al.', 'Musculoskelet Sci Pract. 2019;42:125–33'),
        ('Katzouraki G et al.', 'Bone Joint J. 2020;102-B(6):677–82'),
        ('Boktor J et al.', 'Bone Joint J. 2023;105-B(5):487–95'),
        ('Srikandarajah N et al.', 'Spine. 2015;40(8):580–3'),
        ('Najjar E et al.', 'Bone Joint J. 2026;108-B(3):416–24')]
def s7(t, main, glow, fx):
    a = win(t, 44.4, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 300, a * eo(prog(t, 44.5, 44.9)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 44.6, 45.1))
    big = tl('48 H', 'MontBlack', 150, WHITE)
    put(glow, big, XC, 455, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 455, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Red flag? Bladder scan, MRI now', XC, 590, a * eo(prog(t, 44.8, 45.2)), SKY, 40, 'MontXB', 0)
    label(main, 'Decompress before retention', XC, 655, a * eo(prog(t, 45.0, 45.4)), GOOD, 38, 'MontXB', 0)
    q = eo(prog(t, 45.1, 45.5))
    put(main, card(860, 640, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 1040, a * q)
    for i, (au, ref) in enumerate(REFS):
        y = 773 + i * 99
        label(main, au, XC, y, a * q, ICE, 29, 'MontXB', 0)
        label(main, ref, XC, y + 38, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1430, a * eo(prog(t, 45.3, 45.7)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
