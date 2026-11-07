"""Orthojets EP17 - CLASSIFIED #4 - Garden classification of femoral neck fractures: fix or replace?
Sources (verified on PubMed abstracts / metadata / PMC full text, 2026-11-07):
  Garden RS. Low-angle fixation in fractures of the femoral neck. J Bone Joint Surg Br 1961;43-B:647-63 (original; not
    indexed with an abstract). Definitions quoted from Fu 2015 (below).
  Fu X, Xu GJ, Li ZJ, et al. Medicine (Baltimore) 2015;94(39):e1393. PMID 26426607, PMC4616870,
    doi 10.1097/MD.0000000000001393. Garden on the AP radiograph: stage I incomplete fracture, abducted or impacted;
    II complete without displacement; III complete with partial displacement; IV complete with full displacement.
    Some suggest collapsing into undisplaced (I-II) vs displaced (III-IV).
  Kazley JM, Banerjee S, Abousayed MM, Rosenbaum AJ. Clin Orthop Relat Res 2018;476(2):441-5. PMID 29389800,
    doi 10.1007/s11999.0000000000000066 (Classifications in Brief; no abstract).
  Bjorgul K, Reikeras O. Acta Orthop Scand 2002;73(3):307-10. PMID 12143978, doi 10.1080/000164702320155301.
    6 surgeons, 32 radiographs: Garden kappa 0.4, rising to 0.6 when divided into Garden 1-2 vs 3-4.
  Lu-Yao GL, Keller RB, Littenberg B, Wennberg JE. J Bone Joint Surg Am 1994;76(1):15-25. PMID 8288658,
    doi 10.2106/00004623-199401000-00003. Meta-analysis of 106 reports, displaced femoral neck fractures: within 2 years
    of internal fixation nonunion 33%, AVN 16%; reoperation 20-36% after IF vs 6-18% after hemiarthroplasty.
  Palm H, Gosvig K, Krasheninnikoff M, et al. Acta Orthop 2009;80(3):303-7. PMID 19634021, doi 10.3109/17453670902967281.
    113 undisplaced (Garden I-II), age 60 or older, 2 parallel implants: posterior tilt 20 deg or more -> 14/25
    reoperated vs 12/88 with less tilt (P under 0.001); only significant predictor.
  Slullitel PA, et al. J Arthroplasty 2024;39(1):111-7 (epub 2023). PMID 37380144, doi 10.1016/j.arth.2023.06.035.
    232 Garden I-II, cannulated screws: posterior tilt 20 deg or more -> subsequent arthroplasty 38.8% vs 5%.
  Dolatowski FC, et al. J Bone Joint Surg Am 2019;101(2):136-44. PMID 30653043, doi 10.2106/JBJS.18.00316. RCT,
    age 70 or older, nondisplaced: major reoperation 20% (22/110) screws vs 5% (5/108) hemiarthroplasty.
  Riaz O, et al. Ann R Coll Surg Engl 2016;98(6):376-9. PMID 27055409, doi 10.1308/rcsann.2016.0124. 251 undisplaced
    with cannulated screws: low serum albumin independently associated with fixation failure (p=0.01).
  Pietrzak JRT, et al. Orthop Rev (Pavia) 2026;18:158940. PMID 41868737, doi 10.52965/001c.158940. FNFs ~50% of hip fractures.
  Diagram: AP of the LEFT hip (medial = viewer's left), proximal femur from kit/femur.py, schematic hemipelvis
  (acetabular roof, ilium, pubic rami, ischium, obturator foramen, teardrop). Subcapital fracture line. Type I: incomplete
  line, head tilted into valgus and impacted superolaterally; II: complete line, no displacement; III: head tilted into
  varus, shaft partly migrated proximally (fragments still in contact); IV: fully displaced, shaft migrated proximally.
  Lateral schematic for posterior tilt (head tilted posteriorly on the neck axis).
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(46.0, [(7.4, 1.6), (11.4, 1.2), (14.2, 1.2), (17.0, 1.2), (20.2, 1.4), (24.0, 1.0), (26.6, 2.0),
                     (32.6, 2.0), (39.4, 1.6)])

# ------------------------------------------------------------------ diagram: AP left hip (canvas units)
DW, DH = 600, 640
OX, OY = 150, 70                                   # femur bone-space -> canvas offset
HC = (FEM.HEAD_C[0] + OX, FEM.HEAD_C[1] + OY); HR = FEM.HEAD_R
F_SUP, F_INF = (246 + OX, 148 + OY), (207 + OX, 224 + OY)   # subcapital fracture line (superior -> inferior)
BONE = (226, 238, 250)

def mask_poly(poly, ss=2):
    m = Image.new('L', (DW * ss, DH * ss)); ImageDraw.Draw(m).polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    return m.resize((DW, DH), Image.LANCZOS)

def bone_layer(m, a=235):
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(9))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.40 + 0.60 * (1 - core) ** 0.7) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    ed = m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: int(a * 0.9) if v > 20 else 0))
    lay.alpha_composite(e)
    return lay

def arc_pts(c, r, a0, a1, n=24):
    return [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]

def build_pelvis():
    cx, cy = HC
    sm = lambda pts: FEM.catmull(pts, n=10)
    roof = arc_pts(HC, HR + 8, 200, 330)
    ilium = roof + [(cx + 95, cy - 80), (cx + 112, cy - 170), (cx + 100, -80), (cx - 190, -80), (cx - 160, cy - 110),
                    (cx - 105, cy - 45)]
    pubis = [(cx - 60, cy + 10), (cx - 150, cy + 30), (cx - 330, cy + 40), (cx - 330, cy + 95), (cx - 150, cy + 90),
             (cx - 55, cy + 72)]
    isch = [(cx - 40, cy + 62), (cx + 22, cy + 72), (cx + 32, cy + 150), (cx + 12, cy + 230), (cx - 40, cy + 262),
            (cx - 150, cy + 250), (cx - 330, cy + 232), (cx - 330, cy + 190), (cx - 150, cy + 200), (cx - 72, cy + 160)]
    ms = [np.asarray(mask_poly(roof + sm(ilium[len(roof):] + [roof[0]])[:-10]), np.uint8),
          np.asarray(mask_poly(sm(pubis)), np.uint8), np.asarray(mask_poly(sm(isch)), np.uint8)]
    m = Image.fromarray(np.maximum.reduce(ms))
    hole = Image.new('L', (DW, DH)); ImageDraw.Draw(hole).ellipse((cx - 235, cy + 96, cx - 60, cy + 200), fill=255)
    hole = hole.filter(ImageFilter.GaussianBlur(4))
    m = Image.fromarray(np.clip(np.asarray(m, np.float32) - np.asarray(hole, np.float32), 0, 255).astype(np.uint8))
    lay = bone_layer(m, 120)
    d = ImageDraw.Draw(lay)
    d.line(roof, fill=(240, 248, 255, 190), width=4)                                   # sourcil (dense roof line)
    yy, xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
    a = np.asarray(lay).copy()
    a[..., 3] = (a[..., 3] * np.clip((yy - 10) / 130, 0, 1) * np.clip((xx - 10) / 150, 0, 1)).astype(np.uint8)
    return Image.fromarray(a)

FIMG, _FEDGE = FEM.render()
FEM_CANVAS = Image.new('RGBA', (DW, DH)); FEM_CANVAS.alpha_composite(FIMG.crop((0, 0, 480, DH - OY)), (OX, OY))
_yy, _xx = np.mgrid[0:DH, 0:DW].astype(np.float32)
def side(head):
    (x0, y0), (x1, y1) = F_SUP, F_INF
    cr = (x1 - x0) * (_yy - y0) - (y1 - y0) * (_xx - x0)
    m = (cr >= 0) if head else (cr < 0)
    return m

def cut(img, m):
    a = np.asarray(img).copy(); a[..., 3] = (a[..., 3] * m).astype(np.uint8); return Image.fromarray(a)

# fade the shaft out at the bottom
_fa = np.asarray(FEM_CANVAS).copy(); _fa[..., 3] = (_fa[..., 3] * np.clip((DH - 20 - _yy) / 120, 0, 1)).astype(np.uint8)
FEM_CANVAS = Image.fromarray(_fa)
HEAD = cut(FEM_CANVAS, side(True)); DIST = cut(FEM_CANVAS, side(False))
PELVIS = build_pelvis()

def rot_pt(p, c, ang, tr=(0, 0)):
    """same as PIL rotate(ang, center=c, translate=tr): counter-clockwise on screen for positive ang"""
    a = math.radians(ang); dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) + dy * math.sin(a) + tr[0], c[1] - dx * math.sin(a) + dy * math.cos(a) + tr[1])

# pose per type: head (angle, centre), distal fragment translate
POSE = {'N': (0, HC, (0, 0)), 'I': (-14, F_INF, (0, 0)), 'II': (0, HC, (0, 0)),
        'III': (20, HC, (18, -26)), 'IV': (0, HC, (44, -62))}

_dc = {}
def hip(kind, p, s):
    p = round(p, 2); k = (kind, p, round(s, 3))
    if k in _dc: return _dc[k]
    ang, c, tr = POSE[kind]
    im = Image.new('RGBA', (DW, DH)); im.alpha_composite(PELVIS)
    if kind == 'N':
        im.alpha_composite(FEM_CANVAS)
    else:
        im.alpha_composite(DIST.transform(DIST.size, Image.AFFINE, (1, 0, -tr[0] * p, 0, 1, -tr[1] * p), Image.BICUBIC))
        im.alpha_composite(HEAD.rotate(ang * p, resample=Image.BICUBIC, center=c))
    if s != 1: im = im.resize((int(DW * s), int(DH * s)), Image.LANCZOS)
    if len(_dc) > 300: _dc.clear()
    _dc[k] = im
    return im

def dp(cx, cy, s, q): return cx - DW * s / 2 + q[0] * s, cy - DH * s / 2 + q[1] * s

def frac_line(main, glow, cx, cy, s, a, pl, kind, p):
    """fracture line drawn on the head fragment edge (moves with the head)"""
    if a <= 0 or pl <= 0: return
    ang, c, tr = POSE[kind]
    A_ = rot_pt(F_SUP, c, ang * p) if kind != 'N' else F_SUP
    B_ = rot_pt(F_INF, c, ang * p) if kind != 'N' else F_INF
    mid = ((A_[0] + B_[0]) / 2 + 5, (A_[1] + B_[1]) / 2)
    pts = [A_, mid, B_]; seg = []
    n = 12
    for i in range(n + 1):
        u = i / n * pl
        if u <= 0.5: q = (pts[0][0] + (pts[1][0] - pts[0][0]) * u * 2, pts[0][1] + (pts[1][1] - pts[0][1]) * u * 2)
        else: q = (pts[1][0] + (pts[2][0] - pts[1][0]) * (u - .5) * 2, pts[1][1] + (pts[2][1] - pts[1][1]) * (u - .5) * 2)
        seg.append(dp(cx, cy, s, q))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    w = max(3, int(7 * s))
    for P_, Q_ in zip(seg, seg[1:]): d.line([P_, Q_], fill=(14, 22, 38, int(255 * a)), width=int(w * 1.9))
    for P_, Q_ in zip(seg, seg[1:]):
        d.line([P_, Q_], fill=WARN + (int(255 * a),), width=w); g.line([P_, Q_], fill=WARN + (int(160 * a),), width=w * 3)
    main.alpha_composite(lay)

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

def kappa(val, size, col):
    k = tl('κ', 'DejaVuB', int(size * 0.8), col); v = tl('= ' + val, 'MontBlack', size, col)
    out = Image.new('RGBA', (k.width + v.width - 10, max(k.height, v.height)))
    out.alpha_composite(k, (0, (out.height - k.height) // 2)); out.alpha_composite(v, (k.width - 10, (out.height - v.height) // 2))
    return out

# ------------------------------------------------------------------ lateral view schematic (posterior tilt)
LW, LH = 560, 560
L_PIV = (300, 268)                                   # head-neck junction (pivot of the tilt)
def lateral(tilt, s):
    """cross-table lateral, anterior = viewer's LEFT: shaft goes down-right, neck up-left, head on top of the neck"""
    SS = 2
    m = Image.new('L', (LW * SS, LH * SS)); d = ImageDraw.Draw(m)
    sc = lambda pts: [(x * SS, y * SS) for x, y in pts]
    d.polygon(sc(FEM.catmull([(266, 262), (334, 262), (344, 330), (352, 420), (358, 600), (278, 600), (272, 420),
                              (264, 330)], n=10)), fill=255)          # neck + proximal shaft
    neck = m.resize((LW, LH), Image.LANCZOS)
    hm = Image.new('L', (LW * SS, LH * SS)); hd = ImageDraw.Draw(hm)
    hc = (300, 190); r = 88
    hd.ellipse(sc([(hc[0] - r, hc[1] - r), (hc[0] + r, hc[1] + r)]), fill=255)
    hd.polygon(sc([(262, 272), (338, 272), (342, 230), (258, 230)]), fill=255)
    head = hm.resize((LW, LH), Image.LANCZOS)
    hl = bone_layer_l(head, 190).rotate(-tilt, resample=Image.BICUBIC, center=L_PIV)     # negative = clockwise = posterior
    im = Image.new('RGBA', (LW, LH)); im.alpha_composite(bone_layer_l(neck)); im.alpha_composite(hl)
    a = np.asarray(im).copy(); a[..., 3] = (a[..., 3] * np.clip((LH - 10 - np.arange(LH)) / 110, 0, 1)[:, None]).astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((int(LW * s), int(LH * s)), Image.LANCZOS) if s != 1 else im

def bone_layer_l(m, a=235):
    lay = Image.new('RGBA', m.size, BONE + (0,))
    inner = m.filter(ImageFilter.GaussianBlur(14))
    al = np.asarray(m, np.float32) / 255; core = np.asarray(inner, np.float32) / 255
    alpha = al * (0.30 + 0.70 * (1 - core) ** 0.8) * a / 255
    lay.putalpha(Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    er = m.filter(ImageFilter.MinFilter(5))
    ed = Image.fromarray(np.clip(np.asarray(m, np.float32) - np.asarray(er, np.float32), 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    e = Image.new('RGBA', m.size, (245, 250, 255, 0)); e.putalpha(ed.point(lambda v: int(v * a / 255)))
    lay.alpha_composite(e)
    return lay

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #4', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 400, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, tl('80-year-old, fell, hip fracture:', 'MontXB', 42, ICE), W / 2, 500, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FIX OR REPLACE?', 'MontBlack', 120, WHITE, 840)
    put(glow, k1, W / 2, 630, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 630, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, tl('GARDEN  I · II · III · IV', 'MontBlack', 64, SKY, track=2), W / 2, 770, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.7, 1.2))
    put(main, tl('The AP hip X-ray decides', 'MontBlack', 50, WHITE), W / 2, 880 + 25 * (1 - q), a * q)
    q = eo(prog(t, 1.0, 1.5))
    put(main, pill('I-II = USUALLY FIX', 'MontBlack', 36, NAVY, GOOD + (255,), h=70), W / 2, 985 + 20 * (1 - q), a * q)
    put(main, pill('III-IV = REPLACE', 'MontBlack', 36, NAVY, WARN + (255,), h=70), W / 2, 1070 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.3)); pd = eio(prog(t, 0.3, 1.4))
    put(main, hip('III', pd, 0.8), 560, 1420, a * hq)
    frac_line(main, glow, 560, 1420, 0.8, a * hq, 1.0, 'III', pd)

def s2(t, main, glow, fx):                                     # context: the normal AP hip
    a = win(t, 3.3, 9.2, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, tl('FEMORAL NECK', 'MontBlack', 76, WHITE, track=3), W / 2, 262, a * p, 1, 10 * (1 - p))
    label(main, 'about half of all hip fractures', W / 2, 345, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s = 560, 800, 1.4
    q = eo(prog(t, 3.6, 4.3))
    put(main, hip('N', 0, s), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), dp(cx, cy, s, (HC[0] - 30, HC[1] + 5)), 190, 640, 'HEAD', SKY, 34)
    pointer(main, a * eo(prog(t, 4.6, 5.1)), dp(cx, cy, s, (268 + OX, 205 + OY)), 220, 1080, 'NECK', SKY, 34)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), dp(cx, cy, s, (345 + OX, 130 + OY)), 790, 455, 'GREATER TROCH.')
    pointer(main, a * eo(prog(t, 5.1, 5.6)), dp(cx, cy, s, (242 + OX, 318 + OY)), 790, 1190, 'LESSER TROCH.')
    pointer(main, a * eo(prog(t, 5.3, 5.8)), dp(cx, cy, s, (HC[0] + 36, HC[1] - HR - 6)), 300, 455, 'ACETABULUM')
    fl = eio(prog(t, 6.0, 6.8))
    frac_line(main, glow, cx, cy, s, a, fl, 'N', 0)
    q = eo(prog(t, 6.6, 7.1))
    if q > 0:
        y = 1420 + 20 * (1 - q)
        put(main, card(880, 170, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'GARDEN: 4 STAGES', 515, y - 34, a * q, SKY, 40, 'MontBlack', 2)
        label(main, 'read on the AP radiograph', 515, y + 30, a * q, WHITE, 34, 'MontXB', 0)

TYPES = [('I', 'Incomplete, valgus impacted', 'UNDISPLACED', GOOD, 9.4),
         ('II', 'Complete, not displaced', 'UNDISPLACED', GOOD, 12.2),
         ('III', 'Complete, partially displaced', 'DISPLACED', WARN, 15.0),
         ('IV', 'Complete, fully displaced', 'DISPLACED', WARN, 17.8)]
def s3(t, main, glow, fx):                                     # centrepiece: the 4 stages drawn one by one
    a = win(t, 9.0, 21.0, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s = 560, 760, 1.4
    for i, (rom, desc, grp, col, t0) in enumerate(TYPES):
        t1 = TYPES[i + 1][4] if i < 3 else 21.0
        ta = a * win(t, t0 - 0.3, t1 + 0.05, 0.3 if i else 0.35, 0.25 if i < 3 else 0.4)
        if ta <= 0: continue
        pdisp = eio(prog(t, t0 + 0.6, t0 + 1.4)) if rom != 'II' else 0
        put(main, hip(rom, pdisp, s), cx, cy, ta)
        if rom == 'I':
            frac_line(main, glow, cx, cy, s, ta, eio(prog(t, t0 + 0.2, t0 + 0.8)) * 0.62, rom, pdisp)
        elif rom == 'II':
            frac_line(main, glow, cx, cy, s, ta, eio(prog(t, t0 + 0.2, t0 + 0.8)), rom, 0)
        else:
            frac_line(main, glow, cx, cy, s, ta, 1.0, rom, pdisp)
        p = R.eback(prog(t, t0 + 0.05, t0 + 0.5))
        big = tl(f'TYPE {rom}', 'MontBlack', 96, WHITE)
        put(glow, big, W / 2, 265, ta * 0.25 * clamp(p)); put(main, big, W / 2, 265, ta * clamp(p), clamp(p, 0, 1.2))
        dq = eo(prog(t, t0 + 0.4, t0 + 0.9))
        put(main, fit(desc, 'MontBlack', 48, WHITE, 880), W / 2, 1270 + 15 * (1 - dq), ta * dq)
        eq = eo(prog(t, t0 + 1.0, t0 + 1.4))
        put(main, pill(grp, 'MontBlack', 38, NAVY, col + (255,), h=72), W / 2, 1385 + 15 * (1 - eq), ta * eq)
    label(main, 'Garden 1961  ·  Fu et al. Medicine 2015', W / 2, 1600, a * eo(prog(t, 10.0, 10.5)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # what changes management
    a = win(t, 20.8, 27.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'WHAT CHANGES MANAGEMENT?', W / 2, 262, a * eo(prog(t, 20.9, 21.4)), WHITE, 44, 'MontBlack', 0)
    label(main, 'not 4 stages: 2 groups', W / 2, 335, a * eo(prog(t, 21.1, 21.6)), SKY, 36, 'MontXB', 0)
    # reliability: 4 stages vs 2 groups
    for cx, head, val, col, t0 in [(285, '4 STAGES', '0.4', WARN, 21.5), (745, 'I-II vs III-IV', '0.6', GOOD, 22.3)]:
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + 20 * (1 - q)
        put(main, card(420, 260, outline=col + (160,)), cx, y, a * q)
        label(main, head, cx, y - 75, a * q, ICE, 36, 'MontBlack', 1)
        pk = R.eback(prog(t, t0 + 0.1, t0 + 0.6))
        put(main, kappa(val, 96, col), cx, y + 30, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'interobserver', cx, y + 100, a * q, SUB, 28, 'MontXB', 0)
    label(main, 'Bjørgul & Reikerås  ·  Acta Orthop Scand 2002', W / 2, 735, a * eo(prog(t, 22.6, 23.1)), SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 23.4, 23.9))
    if q > 0:
        y = 1100 + 25 * (1 - q)
        put(main, card(880, 520, outline=WARN + (170,)), 515, y, a * q)
        label(main, 'DISPLACED + INTERNAL FIXATION', 515, y - 200, a * q, WHITE, 40, 'MontBlack', 0)
        for k, (num, txt) in enumerate([('33%', 'nonunion'), ('16%', 'AVN')]):
            xx = 300 + 430 * k; pk = R.eback(prog(t, 23.7 + 0.3 * k, 24.2 + 0.3 * k))
            put(main, tl(num, 'MontBlack', 110, WARN), xx, y - 70, a * q * clamp(pk), clamp(pk, 0, 1.2))
            label(main, txt, xx, y + 15, a * q * clamp(pk), ICE, 36, 'MontXB', 0)
        rq = eo(prog(t, 24.6, 25.1))
        label(main, 'Reoperation within 2 years', 515, y + 105, a * q * rq, WHITE, 36, 'MontBlack', 0)
        label(main, 'fixation 20-36%', 335, y + 165, a * q * rq, WARN, 32, 'MontBlack', 0)
        label(main, 'vs', 530, y + 165, a * q * rq, WHITE, 32, 'MontXB', 0)
        label(main, 'hemi 6-18%', 700, y + 165, a * q * rq, GOOD, 32, 'MontBlack', 0)
    label(main, 'Lu-Yao et al.  ·  JBJS Am 1994  ·  106 reports', W / 2, 1460, a * eo(prog(t, 24.0, 24.5)), SUB, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # undisplaced: check the lateral (posterior tilt)
    a = win(t, 27.4, 33.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'UNDISPLACED?', W / 2, 262, a * eo(prog(t, 27.5, 28.0)), WHITE, 60, 'MontBlack', 2)
    label(main, 'check the lateral for posterior tilt', W / 2, 340, a * eo(prog(t, 27.7, 28.2)), SKY, 36, 'MontXB', 0)
    cx, cy, s = 540, 760, 1.15
    q = eo(prog(t, 27.8, 28.4))
    tilt = 24 * eio(prog(t, 28.8, 30.2))
    put(main, lateral(tilt, s), cx, cy + 20 * (1 - q), a * q)
    lp = lambda p: (cx - LW * s / 2 + p[0] * s, cy - LH * s / 2 + p[1] * s)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    al = a * eo(prog(t, 28.4, 28.8))
    if al > 0:                                           # neck axis (dashed, fixed) and head axis (tilts)
        P0 = lp(L_PIV)
        for (ang, col) in [(0, ICE), (tilt, WARN if tilt >= 19.5 else SKY)]:
            dx, dy = math.sin(math.radians(ang)), -math.cos(math.radians(ang))
            L = 330 * s / 1.15
            for k in range(0, int(L), 34):
                d.line([(P0[0] + dx * k, P0[1] + dy * k), (P0[0] + dx * min(k + 20, L), P0[1] + dy * min(k + 20, L))],
                       fill=col + (int(255 * al),), width=5)
        if tilt > 1:
            R_ = 150
            d.arc((P0[0] - R_, P0[1] - R_, P0[0] + R_, P0[1] + R_), -90, -90 + tilt, fill=WARN + (int(255 * al),), width=6)
    main.alpha_composite(lay)
    put(main, tl(f'{int(round(tilt))}°', 'MontBlack', 64, WARN if tilt >= 19.5 else SKY), cx + 230, cy - 200, al)
    label(main, 'neck axis', cx - 150, cy - 300, al, ICE, 30, 'MontXB', 1)
    label(main, 'ANTERIOR', 200, 1150, a * q, SUB, 30, 'MontXB', 2)
    label(main, 'POSTERIOR', 830, 1150, a * q, SUB, 30, 'MontXB', 2)
    q = eo(prog(t, 30.4, 30.9))
    if q > 0:
        y = 1360 + 20 * (1 - q)
        put(main, card(880, 270, outline=WARN + (170,)), 515, y, a * q)
        label(main, 'Reoperated within 1 year', 515, y - 85, a * q, WHITE, 38, 'MontBlack', 0)
        put(main, tl('56%', 'MontBlack', 80, WARN), 300, y + 5, a * q)
        label(main, 'tilt 20° or more', 300, y + 80, a * q, ICE, 30, 'MontXB', 0)
        label(main, 'vs', 515, y + 5, a * q, WHITE, 36, 'MontXB', 0)
        put(main, tl('14%', 'MontBlack', 80, GOOD), 730, y + 5, a * q)
        label(main, 'under 20°', 730, y + 80, a * q, ICE, 30, 'MontXB', 0)
    label(main, 'Palm et al.  ·  Acta Orthop 2009  ·  113 patients', W / 2, 1600, a * eo(prog(t, 30.8, 31.3)), SUB, 28, 'MontXB', 0)

TIPS = [('Undisplaced, 70 or older?', 'SCREWS 20% VS HEMI 5% REOPERATION AT 2 Y', 'Dolatowski et al. JBJS Am 2019'),
        ('Screws + tilt 20° or more?', 'CONVERSION TO ARTHROPLASTY 38.8% VS 5%', 'Slullitel et al. J Arthroplasty 2024'),
        ('Check the serum albumin', 'LOW ALBUMIN PREDICTS SCREW FAILURE', 'Riaz et al. Ann R Coll Surg Engl 2016')]
def s6(t, main, glow, fx):
    a = win(t, 33.4, 39.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 33.5, 34.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 34.0 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 680); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = fit(ref, 'MontXB', 29, SKY, 760, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 39.4, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 290, a * eo(prog(t, 39.5, 40.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 39.7, 40.3))
    big = fit('III-IV: REPLACE', 'MontBlack', 92, WHITE, 860)
    put(glow, big, W / 2, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'I-II: fix, unless posterior tilt 20° or more', W / 2, 505, a * eo(prog(t, 40.1, 40.6)), GOOD, 34, 'MontBlack', 0)
    label(main, 'Displaced + fixation: 1 in 3 nonunion', W / 2, 575, a * eo(prog(t, 40.4, 40.9)), WARN, 34, 'MontBlack', 0)
    q = eo(prog(t, 40.8, 41.4))
    refs = ['Garden RS. J Bone Joint Surg Br 1961;43-B:647–63', 'Lu-Yao GL, et al. JBJS Am 1994;76(1):15–25',
            'Bjørgul K, et al. Acta Orthop Scand 2002;73:307–10', 'Palm H, et al. Acta Orthop 2009;80(3):303–7',
            'Dolatowski FC, et al. JBJS Am 2019;101(2):136–44', 'Slullitel PA, et al. J Arthroplasty 2024;39:111–7',
            'Riaz O, et al. Ann R Coll Surg Engl 2016;98:376–9', 'Fu X, et al. Medicine 2015;94(39):e1393']
    put(main, card(900, 620, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 990, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 29, ICE if i % 2 == 0 else SUB, 840), W / 2 - 25, 745 + i * 70, a * q)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1400, a * eo(prog(t, 41.4, 42.0)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
