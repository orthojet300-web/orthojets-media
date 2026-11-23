"""Orthojets EP33 - CLASSIFIED #8 - Vancouver classification of periprosthetic femoral fractures: fix or revise?
Sources (verified on PubMed abstracts / metadata / PMC full text, 2026-11-23; PubMed is the only verification source):
  Duncan CP, Masri BA. Fractures of the femur after hip replacement. Instr Course Lect 1995;44:293-304. PMID 7797866
    (no abstract in PubMed; original Vancouver paper).
  Mondanelli N et al. Treatment Algorithm of Periprosthetic Femoral Fractures. Geriatr Orthop Surg Rehabil 2022;13:
    21514593221097608. PMID 35573905, PMC9096211, doi 10.1177/21514593221097608. Full text (PMC): Vancouver proposed by
    Duncan and Masri in 1995. Post-operative types: A = trochanteric region, AG greater / AL lesser trochanter;
    B = around or just distal to the stem (bed of the implant), up to 80% of post-operative PFFs; B1 stable stem
    (adequate bone stock), B2 loose stem with satisfactory bone stock, B3 loose stem with severe bone loss / poor bone
    stock; C = well below the tip of the stem. Management: AL conservative; AG proximal migration under 2 cm conservative,
    surgery if over 2.5 cm; B1 osteosynthesis (MIPO or ORIF); B2 revision with a long stem bypassing the most distal
    fracture line by at least 2 femoral diameters; B3 long-stem revision, allograft-prosthetic composite (APC) or
    proximal femoral replacement; C osteosynthesis, avoid a stress riser between the tip of the stem and the plate
    (longer plate spanning the stem). "Roughly 20% of stems that were concluded to be stable were found to be loose
    intraoperatively." "Even if osteosynthesis is scheduled, it is recommended to have a revision implant available."
  Lindahl H et al. Three hundred and twenty-one periprosthetic femoral fractures. J Bone Joint Surg Am 2006;88(6):1215-22.
    PMID 16757753, doi 10.2106/JBJS.E.00457. Swedish register 1999-2000: 321 fractures; loose stem at the time of fracture
    66% (primary replacement group) and 51% (revision group); 88% Vancouver type B; difficulty with preoperative
    radiographic categorisation; recommend exploring the joint to test stem stability when it is uncertain.
  Lee S et al. Reliability and Validity of the Vancouver Classification in Periprosthetic Fractures Around Cementless
    Femoral Stems. J Arthroplasty 2019;34(7S):S277-S281. PMID 30922672, doi 10.1016/j.arth.2019.02.062. 53 Vancouver B
    fractures, 10 surgeons: interobserver kappa 0.45, all raters agreed on only 43% of radiographs; 20% (14-24%) of
    unstable B2 fractures misread as B1.
  Lewis DP et al. Management of Vancouver B2 PFFs, revision THA vs ORIF: systematic review and meta-analysis.
    J Orthop Trauma 2022;36(1):7-16. PMID 33942785, doi 10.1097/BOT.0000000000002148. 24 studies, 1621 patients
    (331 ORIF, 1280 rTHA): overall complications 24% ORIF vs 18% rTHA (P = 0.13); similar revision rate; ORIF superior
    for postoperative dislocation.
  Haider T et al. Revision arthroplasty vs ORIF of Vancouver B2 and B3 PFFs. JBJS Rev 2021;9(8). PMID 34415859,
    doi 10.2106/JBJS.RVW.21.00008. B2: no significant difference ORIF vs RA; B3: ORIF had higher revision and
    reoperation rates than RA.
  Diagram: AP LEFT femur (kit/femur.py outline, medial side on the viewer's left) with a cementless total hip: head and
  neck resected at a neck cut above the lesser trochanter, tapered stem in the canal, ball head at the native head
  centre, acetabular cup. Fracture lines: AG through the greater trochanter, AL through the lesser trochanter, B oblique
  around the distal half of the stem, C transverse-oblique well below the stem tip. B2 = radiolucent gap around a
  subsided stem; B3 = B2 plus thinned cortex and osteolysis (poor bone stock). Schematic, not to scale.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *
import femur as FEM

TL = Timeline(46.6, [(7.6, 1.4), (10.8, 1.0), (13.6, 1.0), (16.4, 1.0), (19.2, 1.0), (22.0, 1.0), (24.6, 1.2),
                     (30.0, 1.6), (35.6, 1.6), (41.2, 1.8), (45.0, 1.2)])

GOLD = (236, 196, 98)
XRAY = (168, 208, 242)
METAL = (196, 206, 220)
SS = 2
OX, OY = 40, 100                         # bone space -> canvas offset (room for cup / pelvis on the left & top)
CWD, CHT = 560, 1400
ax = lambda y: 300 - 50 * (y - 330) / 710          # femoral shaft axis (same as kit/femur.py)

def B(p): return (p[0] + OX, p[1] + OY)            # bone space -> canvas

# ------------------------------------------------------------------ masks (supersampled, downscaled with LANCZOS)
def mask_poly(pts, smooth=True, width=None):
    big = Image.new('L', (CWD * SS, CHT * SS)); d = ImageDraw.Draw(big)
    pp = FEM.catmull(pts, 8) if smooth else pts
    pp = [(B(p)[0] * SS, B(p)[1] * SS) for p in pp]
    if width: d.line(pp, fill=255, width=int(width * SS), joint='curve')
    else: d.polygon(pp, fill=255)
    return big.resize((CWD, CHT), Image.LANCZOS)

def arr(m): return np.asarray(m, np.float32) / 255
def img(a): return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

FEMUR_M = arr(mask_poly(FEM.outline()))
# neck osteotomy: remove the head and most of the neck (everything medial/proximal of the cut line)
CUT0, CUT1 = (284, 168), (238, 262)                 # superior neck -> inferior neck, above the lesser trochanter
_yy, _xx = np.mgrid[0:CHT, 0:CWD].astype(np.float32)
_bx, _by = _xx - OX, _yy - OY
_side = (CUT1[0] - CUT0[0]) * (_by - CUT0[1]) - (CUT1[1] - CUT0[1]) * (_bx - CUT0[0])
KEEP = np.clip(-_side / 40 + 0.5, 0, 1)
BONE_M = FEMUR_M * KEEP

# stem (tapered, cementless) + prosthetic neck + ball head + cup
STEM_TIP = 690
def stem_pts(dy=0):
    lat = [(ax(y) + w, y) for y, w in [(330, 22), (420, 20), (520, 16), (620, 12), (STEM_TIP - 14, 9)]]
    med = [(ax(y) - w, y) for y, w in [(STEM_TIP - 14, 9), (620, 12), (520, 15), (420, 19)]]
    pts = [(312, 186), (322, 210), (322, 262)] + lat + [(ax(STEM_TIP), STEM_TIP + 4)] + med + \
          [(258, 330), (246, 290), (240, 262), (226, 236), (214, 214), (236, 196), (270, 190)]
    return [(x, y + dy) for x, y in pts]
HEAD_C = (166, 146); BALL_R = 40
def neck_pts(dy=0):
    # trunnion from the stem shoulder toward the ball centre (neck-shaft angle about 130 degrees)
    return [(232, 222 + dy), (262, 196 + dy), (HEAD_C[0] + 14, HEAD_C[1] - 14 + dy), (HEAD_C[0] - 6, HEAD_C[1] + 10 + dy)]

def ball_mask(dy=0):
    big = Image.new('L', (CWD * SS, CHT * SS)); d = ImageDraw.Draw(big)
    cx, cy = B((HEAD_C[0], HEAD_C[1] + dy))
    d.ellipse(((cx - BALL_R) * SS, (cy - BALL_R) * SS, (cx + BALL_R) * SS, (cy + BALL_R) * SS), fill=255)
    return big.resize((CWD, CHT), Image.LANCZOS)

def cup_mask():
    big = Image.new('L', (CWD * SS, CHT * SS)); d = ImageDraw.Draw(big)
    cx, cy = B(HEAD_C); R1, R2 = 58, 46; ang = -40          # cup opening tilted ~40 degrees, facing down-laterally
    pts_o, pts_i = [], []
    for k in range(0, 181, 4):
        th = math.radians(ang - 90 - 90 + k)
        pts_o.append((cx + R1 * math.cos(th), cy + R1 * math.sin(th)))
        pts_i.append((cx + R2 * math.cos(th), cy + R2 * math.sin(th)))
    poly = pts_o + pts_i[::-1]
    d.polygon([(x * SS, y * SS) for x, y in poly], fill=255)
    return big.resize((CWD, CHT), Image.LANCZOS)

def pelvis_mask():
    # schematic acetabular roof / ilium above and medial to the cup, plus the teardrop region
    cx, cy = HEAD_C; R0 = 61
    arc = [(cx + R0 * math.cos(math.radians(a_)), cy + R0 * math.sin(math.radians(a_))) for a_ in range(135, 316, 15)]
    pts = [(214, 100), (240, 50), (252, -20), (236, -95), (60, -98), (0, -40), (8, 80), (56, 168), (98, 204)] + arc
    return arr(mask_poly(pts))

# fracture paths (bone space)
FRACT = {
    'AG': [(308, 128), (318, 170), (340, 205), (372, 228), (392, 236)],
    'AL': [(238, 282), (262, 300), (272, 330), (266, 362), (250, 374)],
    'B':  [(ax(500) + 44, 500), (ax(545) + 10, 548), (ax(590) - 22, 598), (ax(628) - 44, 628)],
    'C':  [(ax(815) + 44, 812), (ax(830), 832), (ax(848) - 44, 850)],
}
def gap(kind, w=5):
    return arr(mask_poly(FRACT[kind], smooth=True, width=w))

STEM_M = {0: arr(mask_poly(stem_pts(0))), 12: arr(mask_poly(stem_pts(12)))}
NECK_M = {0: arr(mask_poly(neck_pts(0), smooth=False)), 12: arr(mask_poly(neck_pts(12), smooth=False))}
BALL_M = {0: arr(ball_mask(0)), 12: arr(ball_mask(12))}
CUP_M = arr(cup_mask())
PELV_M = pelvis_mask()
_halo = img(STEM_M[0]).filter(ImageFilter.MaxFilter(13))
HALO_M = arr(_halo) * (1 - STEM_M[12])                      # radiolucent zone around a loose, subsided stem
_rng = np.random.default_rng(5)

def build(kind):
    """kind: intact | AG | AL | B1 | B2 | B3 | C  -> RGBA canvas (CWD x CHT) with drop shadow"""
    bm = BONE_M.copy(); dy = 12 if kind in ('B2', 'B3') else 0
    if kind in ('AG', 'AL', 'C'): bm *= 1 - gap(kind)
    if kind in ('B1', 'B2', 'B3'): bm *= 1 - gap('B')
    lay = Image.new('RGBA', (CWD, CHT))
    lay.alpha_composite(bone_xray(img(PELV_M), XRAY, alpha=0.42, seed=21, cortex=7))
    bx = bone_xray(img(bm), XRAY, alpha=0.97, seed=11, cortex=9)
    if kind in ('B2', 'B3'):
        a = np.asarray(bx).astype(np.float32)
        a[..., 3] *= 1 - 0.9 * HALO_M
        if kind == 'B3':                                        # poor bone stock: thin cortex + osteolysis around the stem
            prox = np.clip((640 + OY - _yy) / 120, 0, 1)
            n = np.asarray(Image.fromarray((_rng.random((CHT // 14, CWD // 14)) * 255).astype(np.uint8)).resize(
                (CWD, CHT), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
            lyt = np.clip((n - 0.45) * 4, 0, 1) * prox * arr(img(STEM_M[0]).filter(ImageFilter.MaxFilter(41)))
            a[..., 3] *= 1 - 0.55 * prox * (1 - STEM_M[12]) - 0.35 * lyt
        bx = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    lay.alpha_composite(bx)
    lay.alpha_composite(volume(img(CUP_M), METAL, alpha=1.0, r=6, relief=0.8))
    lay.alpha_composite(volume(img(NECK_M[dy]), METAL, alpha=1.0, r=5, relief=0.7))
    lay.alpha_composite(volume(img(STEM_M[dy]), METAL, alpha=1.0, r=7, relief=0.8))
    lay.alpha_composite(volume(img(BALL_M[dy]), (214, 222, 234), alpha=1.0, r=9, relief=0.9))
    fa = np.asarray(lay).copy()
    fa[..., 3] = (fa[..., 3] * np.clip((1150 - _yy) / 160, 0, 1)).astype(np.uint8)   # fade the distal femur out
    return drop_shadow(Image.fromarray(fa), dx=8, dy=14, blur=12, opacity=0.5)

KINDS = ['intact', 'AG', 'AL', 'B1', 'B2', 'B3', 'C']
_base = {}
def base(kind):
    if kind not in _base: _base[kind] = build(kind)
    return _base[kind]

_sc = {}
def fem(kind, s, crop_y=CHT):
    k = (kind, round(s, 3), crop_y)
    if k not in _sc:
        im = base(kind).crop((0, 0, CWD, crop_y))
        if crop_y < CHT:
            fa = np.asarray(im).copy(); fy = np.arange(crop_y)[:, None]
            fa[..., 3] = (fa[..., 3] * np.clip((crop_y - 10 - fy) / 110, 0, 1)).astype(np.uint8); im = Image.fromarray(fa)
        _sc[k] = im.resize((int(CWD * s), int(crop_y * s)), Image.LANCZOS)
        if len(_sc) > 60: _sc.clear()
    return _sc[k]

def bp(cx, cy, s, q, crop_y=CHT):
    """bone-space point -> screen for a femur drawn centred at (cx, cy) with scale s"""
    c = B(q); return cx - CWD * s / 2 + c[0] * s, cy - crop_y * s / 2 + c[1] * s

def fit(txt, font, size, col, maxw, track=0):
    im = tl(txt, font, size, col, track=track)
    while im.width > maxw and size > 30:
        size -= 4; im = tl(txt, font, size, col, track=track)
    return im

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=30):
    if a <= 0: return
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    lab = tl(txt, 'MontXB', size, col, track=2)
    ex = lx - lab.width / 2 + 14 if lx > tip[0] else lx + lab.width / 2 - 14
    d.line([tip, (ex, ly)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    main.alpha_composite(lay)
    put(main, lab, lx, ly, a)

def fracture_line(main, glow, kind, cx, cy, s, a, t, crop_y=CHT, col=WARN):
    if a <= 0: return
    pts = [bp(cx, cy, s, p, crop_y) for p in FEM.catmull(FRACT[kind], 8, False)]
    n = max(2, int(len(pts) * clamp(a * 1.4)))
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay); g = ImageDraw.Draw(glow)
    pul = 0.75 + 0.25 * math.sin(t * 6)
    d.line(pts[:n], fill=col + (int(255 * a * pul),), width=6, joint='curve')
    g.line(pts[:n], fill=col + (int(150 * a),), width=16, joint='curve')
    main.alpha_composite(lay)

def halo_line(main, glow, cx, cy, s, a, t, crop_y=CHT):
    """dashed outline around the loose stem (radiolucent gap)"""
    if a <= 0: return
    sp = stem_pts(12)[3:-7]
    pts = [bp(cx, cy, s, p, crop_y) for p in FEM.catmull(sp, 6, False)]
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    for k in range(0, len(pts) - 1, 3):
        off = 9 * s
        d.line([pts[k], pts[k + 1]], fill=GOLD + (int(230 * a),), width=3)
    main.alpha_composite(lay)

# ------------------------------------------------------------------ scenes
def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill('CLASSIFIED  #8', 'MontXB', 34, NAVY, SKY + (255,), h=70), W / 2, 410, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.25))
    put(main, fit('Hip replacement, fall, femur', 'MontXB', 44, ICE, 900), W / 2, 495, a * q)
    put(main, fit('broken around the stem:', 'MontXB', 44, ICE, 900), W / 2, 552, a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = fit('FIX OR REVISE?', 'MontBlack', 128, WHITE, 840)
    put(glow, k1, W / 2, 680, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, W / 2, 680, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    p2 = R.eback(prog(t, 0.2, 0.65))
    put(main, fit('VANCOUVER CLASSIFICATION', 'MontBlack', 54, SKY, 840, track=2), W / 2, 800, a * clamp(p2), clamp(p2, 0, 1.2))
    q = eo(prog(t, 0.6, 1.1))
    put(main, pill('THE STEM DECIDES', 'MontBlack', 38, NAVY, WARN + (255,), h=72), W / 2, 900 + 20 * (1 - q), a * q)
    hq = eo(prog(t, 0.0, 0.3)); s = 0.9; cy = 1330; cx = 540; cr = 820
    put(main, fem('B1', s, cr), cx, cy, a * hq)
    fracture_line(main, glow, 'B', cx, cy, s, a * eo(prog(t, 0.3, 1.0)), t, cr)

def s2(t, main, glow, fx):                                     # context: the hip replacement + the numbers
    a = win(t, 3.3, 9.0, 0.35, 0.4)
    if a <= 0: return
    p = eo(prog(t, 3.4, 3.9))
    put(main, fit('PERIPROSTHETIC FRACTURE', 'MontBlack', 64, WHITE, 900, track=3), W / 2, 250, a * p, 1, 10 * (1 - p))
    label(main, 'femur broken around a hip stem', W / 2, 325, a * eo(prog(t, 3.6, 4.1)), SKY, 36, 'MontXB', 0)
    cx, cy, s, cr = 520, 850, 0.98, 860
    q = eo(prog(t, 3.6, 4.3))
    put(main, fem('intact', s, cr), cx, cy + 20 * (1 - q), a * q)
    pointer(main, a * eo(prog(t, 4.3, 4.8)), bp(cx, cy, s, (300, 470), cr), 815, 760, 'STEM', SKY, 34)
    pointer(main, a * eo(prog(t, 4.6, 5.1)), bp(cx, cy, s, (HEAD_C[0] - 10, HEAD_C[1] + 30), cr), 170, 700, 'HEAD + CUP', ICE, 30)
    pointer(main, a * eo(prog(t, 4.9, 5.4)), bp(cx, cy, s, (360, 160), cr), 790, 520, 'GREATER TROCH.', ICE, 30)
    pointer(main, a * eo(prog(t, 5.2, 5.7)), bp(cx, cy, s, (244, 320), cr), 205, 880, 'LESSER TROCH.', ICE, 30)
    pointer(main, a * eo(prog(t, 5.5, 6.0)), bp(cx, cy, s, (ax(STEM_TIP), STEM_TIP), cr), 820, 1080, 'STEM TIP', GOLD, 32)
    q = eo(prog(t, 6.0, 6.5))
    if q > 0:
        y = 1470 + 20 * (1 - q)
        put(main, card(880, 330, outline=SKY + (150,)), 515, y + 10, a * q)
        label(main, 'Swedish register, 321 fractures', 515, y - 100, a * q, WHITE, 34, 'MontBlack', 0)
        for x, num, txt, col, t0 in [(290, '88%', 'type B', GOLD, 6.3), (740, '66%', 'loose stem', WARN, 6.7)]:
            pk = R.eback(prog(t, t0, t0 + 0.5))
            put(main, tl(num, 'MontBlack', 92, col), x, y + 5, a * q * clamp(pk), clamp(pk, 0, 1.2))
            label(main, txt, x, y + 90, a * q * clamp(pk), ICE, 32, 'MontXB', 0)
        label(main, 'after primary THA', 740, y + 132, a * q * eo(prog(t, 6.6, 6.9)), SUB, 28, 'MontXB', 0)
        label(main, 'Lindahl et al. JBJS Am 2006', W / 2, 1685, a * q, SUB, 28, 'MontXB', 0)

TYPES = [  # key, start, big title, subtitle, verdict, verdict colour, small line
    ('A', 8.8, 'TYPE  A', 'TROCHANTERS', 'MOSTLY NON-OP', GOOD, 'GT migrated under 2 cm  ·  LT: non-op'),
    ('B1', 11.4, 'TYPE  B1', 'AROUND THE STEM  ·  STEM STABLE', 'FIX  ·  KEEP THE STEM', GOOD, 'plate osteosynthesis'),
    ('B2', 14.2, 'TYPE  B2', 'STEM LOOSE  ·  BONE GOOD', 'REVISE  ·  LONG STEM', WARN, 'long stem, at least 2 diameters past the fracture'),
    ('B3', 17.0, 'TYPE  B3', 'STEM LOOSE  ·  BONE POOR', 'REVISE  +  RECONSTRUCT', WARN, 'long stem, APC or proximal femoral replacement'),
    ('C', 19.8, 'TYPE  C', 'WELL BELOW THE STEM', 'FIX  ·  KEEP THE STEM', GOOD, 'plate that overlaps the stem'),
]
def s3(t, main, glow, fx):                                     # centrepiece: A, B1, B2, B3, C drawn one by one
    a = win(t, 8.4, 25.6, 0.35, 0.4)
    if a <= 0: return
    cx, cy, s, cr = 520, 935, 1.02, 1100
    cur = None
    for i, ty in enumerate(TYPES):
        t1 = TYPES[i + 1][1] if i < 4 else 22.6
        if ty[1] <= t < t1 or (i == 0 and t < ty[1]): cur = (i, ty, t1)
    summ = t >= 22.6
    if cur and not summ:
        i, (key, t0, big, sub, verdict, vcol, small), t1 = cur
        kind = 'AG' if (key == 'A' and t < t0 + 1.3) else ('AL' if key == 'A' else key)
        ta = a * win(t, t0 - 0.25, t1 + 0.05, 0.3, 0.25)
        put(main, fem(kind, s, cr), cx, cy, a)
        p = R.eback(prog(t, t0, t0 + 0.45))
        put(main, tl(big, 'MontBlack', 84, WHITE, track=2), W / 2, 250, ta * clamp(p), clamp(p, 0, 1.2))
        label(main, sub, W / 2, 335, ta * eo(prog(t, t0 + 0.15, t0 + 0.55)), SKY, 36, 'MontBlack', 1)
        fa = ta * eo(prog(t, t0 + 0.2, t0 + 0.9))
        if key == 'A':
            fracture_line(main, glow, 'AG', cx, cy, s, ta * eo(prog(t, t0 + 0.2, t0 + 0.9)) * (1 if t < t0 + 1.3 else 0), t, cr)
            fracture_line(main, glow, 'AL', cx, cy, s, ta * eo(prog(t, t0 + 1.3, t0 + 1.9)), t, cr)
            if t < t0 + 1.3: pointer(main, fa, bp(cx, cy, s, (352, 200), cr), 805, 560, 'AG  ·  GREATER', WARN, 30)
            else: pointer(main, ta * eo(prog(t, t0 + 1.4, t0 + 1.8)), bp(cx, cy, s, (262, 340), cr), 175, 1000, 'AL  ·  LESSER', WARN, 30)
        elif key == 'C':
            fracture_line(main, glow, 'C', cx, cy, s, fa, t, cr)
            pointer(main, ta * eo(prog(t, t0 + 0.6, t0 + 1.0)), bp(cx, cy, s, (ax(STEM_TIP), STEM_TIP + 4), cr), 830, 1180, 'STEM TIP', GOLD, 30)
            pointer(main, ta * eo(prog(t, t0 + 0.9, t0 + 1.3)), bp(cx, cy, s, (ax(830) - 30, 838), cr), 175, 1420, 'FRACTURE', WARN, 30)
        else:
            fracture_line(main, glow, 'B', cx, cy, s, fa, t, cr)
            if key == 'B1':
                pointer(main, ta * eo(prog(t, t0 + 0.7, t0 + 1.1)), bp(cx, cy, s, (ax(440) + 18, 440), cr), 830, 760, 'STEM FIXED', GOOD, 30)
            else:
                halo_line(main, glow, cx, cy, s, ta * eo(prog(t, t0 + 0.6, t0 + 1.0)), t, cr)
                pointer(main, ta * eo(prog(t, t0 + 0.7, t0 + 1.1)), bp(cx, cy, s, (ax(440) + 30, 452), cr), 830, 760, 'LOOSE STEM', GOLD, 30)
                if key == 'B3':
                    pointer(main, ta * eo(prog(t, t0 + 1.0, t0 + 1.4)), bp(cx, cy, s, (ax(420) - 36, 420), cr), 170, 900, 'THIN BONE', WARN, 30)
        q = eo(prog(t, t0 + 0.9, t0 + 1.3))
        put(main, pill(verdict, 'MontBlack', 42, NAVY, vcol + (255,), h=80), W / 2, 1560 + 15 * (1 - q), ta * q)
        label(main, small, W / 2, 1640, ta * eo(prog(t, t0 + 1.1, t0 + 1.5)), ICE, 30, 'MontXB', 0)
        label(main, 'Duncan & Masri 1995  ·  Mondanelli et al. 2022', W / 2, 1692, ta, SUB, 28, 'MontXB', 0)
        put(main, pill(f'{i + 1} / 5', 'MontBlack', 32, NAVY, SKY + (255,), padx=20, h=58), 150, 420, ta)
    if summ:                                                   # the stem decides: fix vs revise
        sa = a * win(t, 22.6, 25.6, 0.35, 0.4)
        p = R.eback(prog(t, 22.6, 23.1))
        big = tl('THE STEM DECIDES', 'MontBlack', 78, WHITE, track=2)
        put(glow, big, W / 2, 260, sa * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 260, sa * clamp(p), clamp(p, 0, 1.2))
        for j, (kind, x, col, head, lines, t0) in enumerate([
                ('B1', 295, GOOD, 'STABLE STEM', ['B1 and C', 'FIX'], 22.9),
                ('B2', 745, WARN, 'LOOSE STEM', ['B2 and B3', 'REVISE'], 23.4)]):
            q = eo(prog(t, t0, t0 + 0.5))
            if q <= 0: continue
            put(main, card(420, 1160, outline=col + (170,)), x, 960 + 20 * (1 - q), sa * q)
            label(main, head, x, 440, sa * q, col, 40, 'MontBlack', 1)
            put(main, fem(kind, 0.62, 860), x, 860, sa * q)
            if kind == 'B2': halo_line(main, glow, x, 860, 0.62, sa * q, t, 860)
            label(main, lines[0], x, 1300, sa * q, ICE, 36, 'MontXB', 0)
            pk = R.eback(prog(t, t0 + 0.3, t0 + 0.8))
            put(main, tl(lines[1], 'MontBlack', 80, col), x, 1400, sa * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'Duncan & Masri 1995  ·  Mondanelli et al. 2022', W / 2, 1640, sa * eo(prog(t, 23.6, 24.0)), SUB, 28, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # B1 or B2? the X-ray lies
    a = win(t, 25.4, 31.0, 0.4, 0.4)
    if a <= 0: return
    label(main, 'B1 OR B2?', W / 2, 260, a * eo(prog(t, 25.5, 26.0)), WHITE, 84, 'MontBlack', 3)
    label(main, 'the X-ray can hide a loose stem', W / 2, 345, a * eo(prog(t, 25.7, 26.2)), SKY, 36, 'MontXB', 0)
    q = eo(prog(t, 26.0, 26.5))
    if q > 0:
        y = 650 + 20 * (1 - q)
        put(main, card(880, 400, outline=WARN + (170,)), 515, y, a * q)
        pk = R.eback(prog(t, 26.2, 26.7))
        put(glow, tl('1 IN 5', 'MontBlack', 150, WARN), 515, y - 50, a * q * 0.3 * clamp(pk), clamp(pk, 0, 1.2))
        put(main, tl('1 IN 5', 'MontBlack', 150, WARN), 515, y - 50, a * q * clamp(pk), clamp(pk, 0, 1.2))
        label(main, 'of "stable" stems were loose at surgery', 515, y + 85, a * q, WHITE, 38, 'MontBlack', 0)
        label(main, 'Mondanelli et al. Geriatr Orthop Surg Rehabil 2022', 515, y + 150, a * q, SUB, 27, 'MontXB', 0)
    q = eo(prog(t, 27.2, 27.7))
    if q > 0:
        y = 1080 + 20 * (1 - q)
        put(main, card(880, 320, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'Cementless stems, 10 surgeons:', 515, y - 105, a * q, WHITE, 36, 'MontBlack', 0)
        for x, num, txt, col, t0 in [(290, '20%', 'of B2 read as B1', WARN, 27.5), (740, '43%', 'full agreement', GOLD, 27.9)]:
            pk = R.eback(prog(t, t0, t0 + 0.5))
            put(main, tl(num, 'MontBlack', 96, col), x, y, a * q * clamp(pk), clamp(pk, 0, 1.2))
            label(main, txt, x, y + 90, a * q * clamp(pk), ICE, 31, 'MontXB', 0)
    q = eo(prog(t, 28.6, 29.1))
    if q > 0:
        label(main, 'kappa 0.45  ·  Lee et al. J Arthroplasty 2019', W / 2, 1300, a * q, SUB, 28, 'MontXB', 0)
        put(main, pill('UNSURE? TREAT IT AS LOOSE', 'MontBlack', 36, NAVY, GOLD + (255,), padx=26, h=68),
            W / 2, 1440 + 15 * (1 - q), a * q)
        label(main, 'explore the joint and test the stem  ·  Lindahl et al. 2006', W / 2, 1530, a * q, ICE, 28, 'MontXB', 0)

def s5(t, main, glow, fx):                                     # the evidence: B2 / B3, fix or revise?
    a = win(t, 30.8, 36.6, 0.4, 0.4)
    if a <= 0: return
    label(main, 'LOOSE STEM:', W / 2, 250, a * eo(prog(t, 30.9, 31.4)), WHITE, 70, 'MontBlack', 2)
    label(main, 'CAN YOU JUST PLATE IT?', W / 2, 335, a * eo(prog(t, 31.1, 31.6)), WARN, 56, 'MontBlack', 2)
    q = eo(prog(t, 31.5, 32.0))
    if q > 0:
        y = 760 + 20 * (1 - q)
        put(main, card(880, 680, outline=SKY + (150,)), 515, y, a * q)
        label(main, 'B2  ·  meta-analysis, 1621 patients', 515, y - 260, a * q, WHITE, 36, 'MontBlack', 0)
        label(main, 'overall complications', 515, y - 200, a * q, SUB, 30, 'MontXB', 0)
        base_y = y + 180
        for x, v, txt, col, t0 in [(330, 24, 'ORIF', WARN, 32.0), (700, 18, 'REVISION', GOOD, 32.4)]:
            g = eio(prog(t, t0, t0 + 0.7)); hgt = 220 * v / 24 * g
            lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
            d.rounded_rectangle((x - 80, base_y - hgt, x + 80, base_y), 18, fill=col + (int(235 * a * q),))
            main.alpha_composite(lay)
            if g > 0.05: label(main, f'{int(round(v * g))}%', x, base_y - hgt - 50, a * q, col, 60, 'MontBlack', 0)
            label(main, txt, x, base_y + 45, a * q, ICE, 34, 'MontBlack', 1)
        label(main, 'P = 0.13: no significant difference', 515, y + 290, a * q * eo(prog(t, 33.0, 33.4)), GOLD, 32, 'MontBlack', 0)
    label(main, 'Lewis et al. J Orthop Trauma 2022', W / 2, 1140, a * eo(prog(t, 33.2, 33.6)), SUB, 28, 'MontXB', 0)
    q = eo(prog(t, 33.8, 34.3))
    if q > 0:
        y = 1380 + 20 * (1 - q)
        put(main, card(880, 300, outline=WARN + (170,)), 515, y, a * q)
        label(main, 'B3 with ORIF:', 515, y - 85, a * q, WHITE, 40, 'MontBlack', 0)
        label(main, 'more revisions and reoperations', 515, y - 20, a * q, WARN, 38, 'MontBlack', 0)
        label(main, 'than revision arthroplasty', 515, y + 40, a * q, ICE, 34, 'MontXB', 0)
        label(main, 'Haider et al. JBJS Rev 2021', 515, y + 100, a * q, SUB, 28, 'MontXB', 0)

TIPS = [('Fixing? Have a revision stem ready', 'TEST THE STEM IN THEATRE IF IN DOUBT', 'Lindahl et al. 2006  ·  Mondanelli et al. 2022'),
        ('Revising? Use a long stem', 'AT LEAST 2 DIAMETERS PAST THE FRACTURE', 'Mondanelli et al. 2022'),
        ('Plating a C? Overlap the stem', 'NO STRESS RISER BETWEEN STEM TIP AND PLATE', 'Mondanelli et al. 2022')]
def s6(t, main, glow, fx):
    a = win(t, 36.4, 42.4, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', W / 2, 300, a * eo(prog(t, 36.5, 37.0)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 36.9 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), 515, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = fit(h_, 'MontBlack', 44, WHITE, 690); put(main, lab, 140 + lab.width / 2 - 20, y - 52, a * q)
        lab = fit(sub, 'MontXB', 31, ICE, 770, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 22, a * q)
        lab = fit(ref, 'MontXB', 28, SKY, 770, track=1); put(main, lab, 140 + lab.width / 2 - 20, y + 82, a * q)

def s7(t, main, glow, fx):
    a = win(t, 42.2, 46.6, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 280, a * eo(prog(t, 42.3, 42.8)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 42.5, 43.1))
    big = fit('THE STEM DECIDES', 'MontBlack', 92, WHITE, 880)
    put(glow, big, W / 2, 390, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, W / 2, 390, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Stable stem (B1, C): fix', W / 2, 495, a * eo(prog(t, 42.9, 43.4)), GOOD, 40, 'MontBlack', 0)
    label(main, 'Loose stem (B2, B3): revise', W / 2, 560, a * eo(prog(t, 43.2, 43.7)), WARN, 40, 'MontBlack', 0)
    q = eo(prog(t, 43.5, 44.1))
    refs = ['Duncan CP, Masri BA. Instr Course Lect 1995;44:293–304', 'Lindahl H, et al. JBJS Am 2006;88(6):1215–22',
            'Lee S, et al. J Arthroplasty 2019;34(7S):S277–81', 'Lewis DP, et al. J Orthop Trauma 2022;36(1):7–16',
            'Haider T, et al. JBJS Rev 2021;9(8)', 'Mondanelli N, et al. Geriatr Orthop Surg Rehabil 2022;13']
    put(main, card(880, 500, fill=(24, 44, 70, 160), outline=SKY + (80,)), 515, 920, a * q)
    for i, r_ in enumerate(refs):
        put(main, fit(r_, 'MontXB', 28, ICE, 780), 515, 745 + i * 70, a * q)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1290, a * eo(prog(t, 44.0, 44.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6, s7]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
