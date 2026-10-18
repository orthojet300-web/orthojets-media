"""Orthojets episode 01 — Tip-Apex Distance (TAD) < 25 mm.
Source: Baumgaertner MR et al. J Bone Joint Surg Am. 1995;77(7):1058-64 (doi 10.2106/00004623-199507000-00012)
  198 peritrochanteric fractures, 16 cut-outs; mean TAD 24 mm (healed) vs 38 mm (cut-out);
  none of 120 screws with TAD <= 25 mm cut out.
"""
import sys, os, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
import render as R
import femur as FEM

W, H, FPS = R.W, R.H, R.FPS
SCRIPT_DUR = 34.0
HOLDS = [(2.6, 1.5), (4.7, 1.0), (7.9, 2.2), (12.8, 1.5), (16.2, 3.2), (19.5, 1.5), (22.3, 3.5),
         (25.6, 1.0), (26.95, 1.0), (28.4, 2.5), (32.8, 3.0)]   # (script time, extra seconds to read)
DUR = SCRIPT_DUR + sum(h for _, h in HOLDS)
def script_time(tr):
    off = 0.0
    for T, h in HOLDS:
        if tr - off <= T: return tr - off
        if tr - off < T + h: return T
        off += h
    return tr - off
def real_time(st):
    return st + sum(h for T, h in HOLDS if T < st)
SKY, NAVY, WHITE, ICE, CYAN = R.SKY, R.NAVY, R.WHITE, R.ICE, R.CYAN
WARN = (236, 112, 104)
GOOD = (120, 210, 170)
put, tl, prog, eo, ei, eio, win, clamp = R.put, R.text_layer, R.prog, R.eo, R.ei, R.eio, R.win, R.clamp

# ------------------------------------------------------------------ AP composite (bone space -> 3x px)
SS = 3; CX0, CY0, CW, CH = 30, 40, 400, 520
def bp(x, y): return ((x - CX0) * SS, (y - CY0) * SS)

HEAD = np.array(FEM.HEAD_C, float); HR = FEM.HEAD_R
U = np.array([0.82, 0.57]); U /= np.linalg.norm(U)        # neck axis, head -> lateral cortex
N = np.array([U[1], -U[0]])                                  # perpendicular, points superior
APEX = HEAD - U * HR
MM = 24.0 / HR                                               # head Ø ≈ 48 mm  -> mm per bone unit

def lat_cortex_x(y):
    pts = [(262, 380), (320, 360), (360, 345), (400, 335), (500, 325), (600, 317)]
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        if y0 <= y <= y1: return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return pts[-1][1]

def build_ap_base():
    img, edge = FEM.render()
    crop = img.crop((CX0, CY0, CX0 + CW, CY0 + CH)).resize((CW * SS, CH * SS), Image.LANCZOS)
    e = Image.new('RGBA', crop.size, ICE + (0,))
    e.putalpha(edge.crop((CX0, CY0, CX0 + CW, CY0 + CH)).resize(crop.size, Image.LANCZOS))
    # intertrochanteric fracture line (GT base -> lesser trochanter), dark radiolucent gap
    frac = Image.new('L', crop.size); d = ImageDraw.Draw(frac)
    line = [(372, 214), (352, 232), (330, 238), (306, 256), (284, 262), (262, 284), (246, 300)]
    d.line([bp(*p) for p in line], fill=255, width=9, joint='curve')
    a = np.asarray(crop, np.float32)
    f = np.asarray(frac.filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255
    a[..., :3] *= (1 - 0.75 * f); a[..., 3:4] *= (1 - 0.6 * f)
    base = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    base.alpha_composite(e)
    return base

def implant_layer(depth=1.0, tip_s=-(HR - 10 / MM), offset=0.0, plate=1.0, size=(CW * SS, CH * SS), sp=bp):
    """lag screw along the neck axis; tip_s = signed distance of tip from head centre along U (negative = toward apex);
    offset = shift perpendicular (superior +). depth 0..1 animates insertion from lateral cortex."""
    L = Image.new('RGBA', size); d = ImageDraw.Draw(L)
    entry_s = 258.0
    cur_tip = entry_s + (tip_s - entry_s) * depth
    c = HEAD + N * offset
    P = lambda s, w: c + U * s + N * w
    metal = (236, 244, 252, 255)
    if plate > 0:
        # barrel + side plate on the lateral cortex
        bar = [P(entry_s + 12, 17), P(entry_s - 70, 17), P(entry_s - 70, -17), P(entry_s + 12, -17)]
        d.polygon([sp(*q) for q in bar], fill=(214, 228, 244, int(235 * plate)))
        ys = np.arange(268, 545, 4)
        left = [(lat_cortex_x(y) + 1, y) for y in ys]; right = [(lat_cortex_x(y) + 15, y) for y in ys[::-1]]
        d.polygon([sp(*q) for q in left + right], fill=(214, 228, 244, int(235 * plate)))
        for y in (370, 430, 490):
            x0 = lat_cortex_x(y) + 15
            d.line([sp(x0, y), sp(x0 - 120, y)], fill=(225, 236, 248, int(220 * plate)), width=4 * SS)
            for k in range(-50, 0, 8):
                d.line([sp(x0 + k * 2.2 + 8, y - 6), sp(x0 + k * 2.2 + 8, y + 6)], fill=(170, 190, 215, int(200 * plate)), width=SS)
    if depth > 0:
        thread_len = 27.0
        shank_end = cur_tip + thread_len
        sh = [P(entry_s + 10, 10.5), P(shank_end, 10.5), P(shank_end, -10.5), P(entry_s + 10, -10.5)]
        d.polygon([sp(*q) for q in sh], fill=metal)
        th = [P(shank_end, 16.5), P(cur_tip + 3, 16.5), P(cur_tip, 12), P(cur_tip, -12), P(cur_tip + 3, -16.5), P(shank_end, -16.5)]
        d.polygon([sp(*q) for q in th], fill=metal)
        s = shank_end
        while s > cur_tip + 2:
            d.line([sp(*P(s, 16.5)), sp(*P(s - 2.5, -16.5))], fill=(160, 182, 210, 255), width=SS)
            s -= 4.2
    return L

def tip_point(tip_s=-(HR - 10 / MM), offset=0.0):
    return HEAD + N * offset + U * tip_s

def camera(src, cx, cy, z, vw, vh, src_scale=SS, origin=(CX0, CY0)):
    """crop src (bone units * src_scale) around bone point (cx,cy) at z screen px per bone unit"""
    hw, hh = vw / z / 2, vh / z / 2
    x0, y0 = (cx - hw - origin[0]) * src_scale, (cy - hh - origin[1]) * src_scale
    x1, y1 = (cx + hw - origin[0]) * src_scale, (cy + hh - origin[1]) * src_scale
    return src.transform((int(vw), int(vh)), Image.EXTENT, (x0, y0, x1, y1), Image.BICUBIC)

def to_screen(px, py, cx, cy, z, scx, scy):
    return scx + (px - cx) * z, scy + (py - cy) * z

# ------------------------------------------------------------------ lateral view (schematic, X-ray look)
LW, LH = 300, 430
LHEAD = np.array([150.0, 125.0]); LD = np.array([-0.26, 0.97]); LD /= np.linalg.norm(LD)
LAPEX = LHEAD - LD * HR
def build_lateral():
    m = Image.new('L', (LW * SS, LH * SS)); d = ImageDraw.Draw(m)
    S = lambda p: (p[0] * SS, p[1] * SS)
    cx, cy = LHEAD
    d.ellipse([S((cx - HR, cy - HR)), S((cx + HR, cy + HR))], fill=255)
    nrm = np.array([LD[1], -LD[0]])
    neck = [LHEAD + LD * 40 + nrm * 40, LHEAD + LD * 150 + nrm * 50, LHEAD + LD * 150 - nrm * 50, LHEAD + LD * 40 - nrm * 40]
    d.polygon([S(p) for p in neck], fill=255)
    d.polygon([S(p) for p in [(88, 255), (196, 255), (190, 440), (100, 440)]], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(6 * SS)).point(lambda v: 255 if v > 128 else 0)
    M = np.asarray(m, np.float32) / 255
    B = np.asarray(m.filter(ImageFilter.GaussianBlur(9 * SS)), np.float32) / 255
    rng = np.random.default_rng(5)
    n1 = np.asarray(Image.fromarray((rng.random((LH, LW)) * 255).astype(np.uint8)).resize(m.size, Image.BICUBIC), np.float32) / 255
    dens = (0.42 + 0.58 * np.clip((1 - B) * 3, 0, 1)) * (0.86 + 0.24 * n1)
    inner = np.clip(M * dens, 0, 1.2)
    rgb = np.stack([inner * c for c in (165, 208, 242)], -1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert('RGBA')
    img.putalpha(Image.fromarray(np.clip(inner * 280, 0, 255).astype(np.uint8)))
    # screw along neck axis, tip 8 mm from lateral apex
    tip_s = -(HR - 8 / MM)
    L = ImageDraw.Draw(img)
    P = lambda s, w: LHEAD + LD * s + nrm * w
    L.polygon([S(p) for p in [P(tip_s + 27, 10.5), P(330, 10.5), P(330, -10.5), P(tip_s + 27, -10.5)]], fill=(236, 244, 252, 255))
    L.polygon([S(p) for p in [P(tip_s + 27, 16.5), P(tip_s + 3, 16.5), P(tip_s, 12), P(tip_s, -12), P(tip_s + 3, -16.5), P(tip_s + 27, -16.5)]], fill=(236, 244, 252, 255))
    s = tip_s + 27
    while s > tip_s + 2:
        L.line([S(P(s, 16.5)), S(P(s - 2.5, -16.5))], fill=(160, 182, 210, 255), width=SS); s -= 4.2
    L.polygon([S(p) for p in [(128, 300), (156, 300), (156, 440), (128, 440)]], fill=(214, 228, 244, 150))
    return img, LHEAD + LD * tip_s

# ------------------------------------------------------------------ helpers
def label(main, txt, x, y, a, col=ICE, size=28, font='MontXB', track=3, anchor='c'):
    L = tl(txt, font, size, col, track=track)
    dx = {'c': 0, 'l': L.width / 2 - 20, 'r': -L.width / 2 + 20}[anchor]
    put(main, L, x + dx, y, a)

def ar(main, txt, x, y, a, size=64, col=WHITE, font='CairoBlack', blur=0.0, dy=0.0):
    put(main, tl(txt, font, size, col, ar=True), x, y + dy, a, 1, blur)

def dot(d, x, y, r, col, a):
    d.ellipse((x - r, y - r, x + r, y + r), fill=col + (int(255 * a),))

def ring(d, x, y, r, col, a, w=3):
    d.ellipse((x - r, y - r, x + r, y + r), outline=col + (int(255 * a),), width=w)

def dashed(d, p0, p1, col, a, w=4, dash=14, gap=10, frac=1.0):
    p0 = np.array(p0, float); p1 = np.array(p1, float); v = p1 - p0; Ln = np.linalg.norm(v)
    if Ln < 1: return
    v /= Ln; s = 0.0; end = Ln * frac
    while s < end:
        e = min(s + dash, end)
        d.line([tuple(p0 + v * s), tuple(p0 + v * e)], fill=col + (int(255 * a),), width=w); s += dash + gap

def check(d, x, y, sz, col, a, w=7):
    d.line([(x - sz * .5, y), (x - sz * .12, y + sz * .38), (x + sz * .55, y - sz * .42)], fill=col + (int(255 * a),), width=w, joint='curve')

def cross(d, x, y, sz, col, a, w=7):
    d.line([(x - sz * .4, y - sz * .4), (x + sz * .4, y + sz * .4)], fill=col + (int(255 * a),), width=w)
    d.line([(x + sz * .4, y - sz * .4), (x - sz * .4, y + sz * .4)], fill=col + (int(255 * a),), width=w)

def pill(text, font, size, fg, bg, outline=None, padx=34, h=None):
    t = tl(text, font, size, fg, track=4)
    h = h or t.height + 4
    w = t.width + 2 * padx
    im = Image.new('RGBA', (w * 3, h * 3)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), h * 3 / 2, fill=bg, outline=outline, width=6 if outline else 0)
    im = im.resize((w, h), Image.LANCZOS)
    im.alpha_composite(t, (int(w / 2 - t.width / 2), int(h / 2 - t.height / 2)))
    return im

def card(w, h, fill=(30, 54, 84, 170), outline=SKY + (120,), r=34):
    im = Image.new('RGBA', (w * 2, h * 2)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=fill, outline=outline, width=4)
    return im.resize((w, h), Image.LANCZOS)

# ------------------------------------------------------------------ cached assets
AP = build_ap_base()
IMPL_FINAL = implant_layer()
GOOD_TIP = tip_point()
BAD_S, BAD_OFF = -(HR - 21.5 / MM) + 3, 22.0
BAD_TIP = tip_point(BAD_S, BAD_OFF)
AP_GOOD = AP.copy(); AP_GOOD.alpha_composite(IMPL_FINAL)
AP_BAD = AP.copy(); AP_BAD.alpha_composite(implant_layer(tip_s=BAD_S, offset=BAD_OFF))
LAT, LAT_TIP = build_lateral()
TAD_AP = np.linalg.norm(GOOD_TIP - APEX) * MM          # ≈ 10 mm
TAD_BAD_AP = np.linalg.norm(BAD_TIP - APEX) * MM       # ≈ 21 mm

# ------------------------------------------------------------------ scenes
def s1_hook(t, main, glow, fx):
    a = win(t, 0.0, 3.5, 0.25, 0.35)
    if a <= 0: return
    tag = pill('ORTHO TIP  #01', 'MontXB', 30, NAVY, SKY + (255,), h=64)
    put(main, tag, W / 2, 470, a * eo(prog(t, 0.15, 0.6)))
    n = int(round(25 * eo(prog(t, 0.3, 1.5))))
    big = tl(str(n), 'MontBlack', 330, WHITE)
    put(glow, big, W / 2 - 60, 820, a * 0.35)
    put(main, big, W / 2 - 60, 820, a)
    put(main, tl('mm', 'MontXB', 96, SKY), W / 2 + 270, 885, a * eo(prog(t, 1.2, 1.6)))
    fx['flash'] += 0.25 * math.exp(-((t - 1.5) / 0.08) ** 2)
    p = eo(prog(t, 1.5, 2.2))
    put(main, tl('One number that protects', 'MontXB', 60, WHITE), W / 2, 1110 + 30 * (1 - p), a * p, 1, 10 * (1 - p))
    p = eo(prog(t, 1.8, 2.5))
    put(main, tl('your hip fracture fixation', 'MontXB', 60, SKY), W / 2, 1195 + 30 * (1 - p), a * p, 1, 10 * (1 - p))

def s2_context(t, main, glow, fx):
    a = win(t, 3.3, 9.2, 0.35, 0.01)
    if a <= 0: return
    p = eo(prog(t, 3.4, 4.0))
    put(main, tl('Peritrochanteric Fractures', 'MontBlack', 64, WHITE), W / 2, 250, a * p, 1, 10 * (1 - p))
    label(main, 'AO/OTA  31-A', W / 2, 345, a * eo(prog(t, 3.7, 4.3)), (175, 205, 232), 28, 'MontLight', 7)
    # femur view (camera shared with s3 for a seamless push-in)
    depth = eio(prog(t, 5.0, 6.2)); plate = eo(prog(t, 4.8, 5.3))
    comp = AP.copy()
    if plate > 0 or depth > 0: comp.alpha_composite(implant_layer(depth=depth, plate=plate))
    z = 2.05 + 0.08 * prog(t, 3.3, 9.0)
    view = camera(comp, 255, 300, z, 900, 1060)
    put(main, view, W / 2, 950, a * eo(prog(t, 3.3, 3.9)))
    fx['flash'] += 0.12 * math.exp(-((t - 6.2) / 0.08) ** 2)
    # fracture callout
    fp = eo(prog(t, 4.1, 4.6)) * (1 - prog(t, 6.0, 6.4))
    if fp > 0:
        d = ImageDraw.Draw(main)
        x, y = to_screen(306, 256, 255, 300, z, W / 2, 950)
        ring(d, x, y, 16, WARN, a * fp)
        d.line([(x, y), (x - 120, y + 120), (x - 260, y + 120)], fill=WARN + (int(255 * a * fp),), width=3)
        label(main, 'FRACTURE LINE', x - 400, y + 98, a * fp, WARN, 26, 'MontXB', 3)
    # implant name
    ip = eo(prog(t, 6.3, 6.8))
    label(main, 'DHS  /  CEPHALOMEDULLARY NAIL', W / 2, 1520, a * ip * (1 - prog(t, 8.6, 9.0)), ICE, 28, 'MontXB', 4)
    # failure mode
    wp = eo(prog(t, 7.0, 7.6)) * (1 - prog(t, 8.7, 9.1))
    if wp > 0:
        label(main, 'Most common mode of fixation failure', W / 2, 1610, a * wp, ICE, 36, 'MontXB', 1)
        label(main, 'LAG SCREW CUT-OUT', W / 2, 1690, a * wp, WARN, 44, 'MontBlack', 4)

def s3_define(t, main, glow, fx):
    a = win(t, 9.0, 17.2, 0.01, 0.4)
    if a <= 0: return
    # push in from the s2 framing to the femoral head
    zp = eio(prog(t, 9.0, 10.4))
    cx, cy = 255 + (150 - 255) * zp, 300 + (130 - 300) * zp
    z = 2.13 + (4.4 - 2.13) * zp
    # split: AP view moves left & shrinks when the lateral view arrives
    sp = eio(prog(t, 12.9, 13.8))
    vw, vh = 900 - 440 * sp, 1060 - 400 * sp
    scx, scy = W / 2 - 230 * sp, 950 - 140 * sp
    zz = z * (1 - 0.42 * sp)
    view = camera(AP_GOOD, cx, cy, zz, vw, vh)
    put(main, view, scx, scy, a)
    p = eo(prog(t, 9.2, 9.8))
    label(main, 'TIP–APEX DISTANCE', W / 2, 250, a * p, WHITE, 64, 'MontBlack', 3)
    label(main, 'Lag screw tip  to  femoral head apex', W / 2, 340, a * eo(prog(t, 9.5, 10.1)), SKY, 36, 'MontXB', 1)
    d = ImageDraw.Draw(main)
    ax, ay = to_screen(*APEX, cx, cy, zz, scx, scy)
    tx, ty = to_screen(*GOOD_TIP, cx, cy, zz, scx, scy)
    # neck axis guide (defines the apex)
    gp = eo(prog(t, 10.4, 11.0))
    if gp > 0:
        far = to_screen(*(HEAD + U * 120), cx, cy, zz, scx, scy); near = to_screen(*(APEX - U * 25), cx, cy, zz, scx, scy)
        dashed(d, far, near, SKY, a * gp * 0.8, 3, 12, 10, gp)
    ap_ = eo(prog(t, 10.8, 11.2))
    if ap_ > 0:
        pulse = 1 + 0.25 * math.sin(t * 7) * (t < 12.5)
        dot(d, ax, ay, 9 * pulse, WHITE, a * ap_); ring(glow and ImageDraw.Draw(glow), ax, ay, 18 * pulse, SKY, a * ap_, 5)
        label(main, 'APEX', ax - 80 + 30 * sp, ay - 52, a * ap_, WHITE, 28, 'MontXB', 3)
    tp = eo(prog(t, 11.2, 11.6))
    if tp > 0:
        dot(d, tx, ty, 8, SKY, a * tp)
        label(main, 'TIP', tx + 30, ty + 52, a * tp, SKY, 28, 'MontXB', 3)
    mp = eio(prog(t, 11.6, 12.3))
    if mp > 0:
        d.line([(ax, ay), (ax + (tx - ax) * mp, ay + (ty - ay) * mp)], fill=SKY + (int(255 * a),), width=6)
        ImageDraw.Draw(glow).line([(ax, ay), (ax + (tx - ax) * mp, ay + (ty - ay) * mp)], fill=SKY + (int(200 * a),), width=10)
    vp = eo(prog(t, 12.2, 12.7))
    if vp > 0:
        label(main, f'X AP  =  {TAD_AP:.0f} mm', scx, scy + vh / 2 + 50, a * vp, WHITE, 38, 'MontXB', 2)
    # lateral panel
    if sp > 0:
        lz = 1.62
        lcx, lcy = 150, 175
        lv = camera(LAT, lcx, lcy, lz, 420, 660, SS, (0, 0))
        lsx, lsy = W / 2 + 255, 810
        put(main, lv, lsx + 120 * (1 - sp), lsy, a * sp)
        lax, lay = to_screen(*LAPEX, lcx, lcy, lz, lsx, lsy); ltx, lty = to_screen(*LAT_TIP, lcx, lcy, lz, lsx, lsy)
        lp = eo(prog(t, 13.8, 14.3))
        if lp > 0:
            dot(d, lax, lay, 7, WHITE, a * lp); dot(d, ltx, lty, 6, SKY, a * lp)
            d.line([(lax, lay), (ltx, lty)], fill=SKY + (int(255 * a * lp),), width=5)
            label(main, f'X LAT  =  8 mm', lsx, lsy + 380, a * lp, WHITE, 38, 'MontXB', 2)
        label(main, 'AP', scx, scy - vh / 2 - 34, a * sp, (175, 205, 232), 26, 'MontLight', 6)
        label(main, 'LATERAL', lsx, lsy - 362, a * sp, (175, 205, 232), 26, 'MontLight', 6)
    # formula
    fp = eo(prog(t, 14.4, 15.0))
    if fp > 0:
        label(main, 'TAD  =  X AP  +  X LAT', W / 2, 1395, a * fp, WHITE, 52, 'MontBlack', 2)
        q = eo(prog(t, 15.0, 15.5))
        label(main, '=  10  +  8  =  18 mm', W / 2 - 30, 1480, a * q, GOOD, 50, 'MontBlack', 2)
        check(d, W / 2 + 290, 1478, 46, GOOD, a * q)
        label(main, 'Corrected for magnification', W / 2, 1580, a * eo(prog(t, 15.4, 15.9)), (190, 212, 235), 36, 'MontXB', 1)
        label(main, 'X × ( D true ÷ D x-ray )  using the lag screw diameter', W / 2, 1648, a * eo(prog(t, 15.6, 16.1)), (190, 212, 235), 28, 'MontXB', 1)

def mini(img, tip, val, good, x, y, a, t0, t):
    view = camera(img, 140, 132, 3.25, 430, 470)
    c = card(450, 600, fill=(24, 44, 70, 190), outline=(GOOD if good else WARN) + (150,))
    put(main_ref[0], c, x, y + 50, a)
    put(main_ref[0], view, x, y, a)
    d = ImageDraw.Draw(main_ref[0])
    ax, ay = to_screen(*APEX, 140, 132, 3.25, x, y); tx, ty = to_screen(*tip, 140, 132, 3.25, x, y)
    mp = eio(prog(t, t0, t0 + 0.5))
    dot(d, ax, ay, 7, WHITE, a); d.line([(ax, ay), (ax + (tx - ax) * mp, ay + (ty - ay) * mp)], fill=(GOOD if good else WARN) + (int(255 * a),), width=5)
    vp = eo(prog(t, t0 + 0.4, t0 + 0.8))
    label(main_ref[0], f'TAD {val} mm', x - 26, y + 285, a * vp, GOOD if good else WARN, 44, 'MontBlack', 1)
    (check if good else cross)(d, x + 150, y + 285, 40, GOOD if good else WARN, a * vp)
    label(main_ref[0], 'CENTER – CENTER' if good else 'SUPERIOR  +  SHORT', x, y + 345, a * vp, ICE, 27, 'MontXB', 2)
main_ref = [None]

def s4_rule(t, main, glow, fx):
    a = win(t, 17.0, 24.3, 0.4, 0.4)
    if a <= 0: return
    main_ref[0] = main
    p = R.eback(prog(t, 17.1, 17.7))
    big = tl('TAD  <  25 mm', 'MontBlack', 104, WHITE, track=1)
    put(glow, big, W / 2, 300, a * 0.3 * clamp(p), clamp(p, 0, 1.2))
    put(main, big, W / 2, 300, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 17.45) / 0.08) ** 2)
    m1 = eo(prog(t, 17.8, 18.3)); m2 = eo(prog(t, 18.4, 18.9))
    if m1 > 0: mini(AP_GOOD, GOOD_TIP, 18, True, W / 2 - 235, 720 + 40 * (1 - m1), a * m1, 18.3, t)
    if m2 > 0: mini(AP_BAD, BAD_TIP, 38, False, W / 2 + 235, 720 + 40 * (1 - m2), a * m2, 18.9, t)
    q = eo(prog(t, 20.0, 20.6))
    if q > 0:
        put(main, card(900, 260), W / 2, 1390 + 20 * (1 - q), a * q)
        label(main, '120 screws with TAD of 25 mm or less', W / 2, 1345 + 20 * (1 - q), a * q, WHITE, 38, 'MontXB', 1)
        label(main, 'ZERO CUT-OUTS', W / 2, 1425 + 20 * (1 - q), a * q, GOOD, 62, 'MontBlack', 4)
    r = eo(prog(t, 21.4, 22.0))
    if r > 0:
        label(main, 'MEAN TAD', W / 2, 1565, a * r, (190, 212, 235), 28, 'MontXB', 8)
        label(main, '24 mm  healed', W / 2 - 200, 1630, a * r, GOOD, 40, 'MontBlack', 1)
        label(main, '38 mm  cut-out', W / 2 + 200, 1630, a * r, WARN, 40, 'MontBlack', 1)

TIPS = [('Center – center', 'ON  BOTH  AP  &  LATERAL  VIEWS'),
        ('Avoid superior / anterior', 'LAG  SCREW  POSITION  IN  THE  HEAD'),
        ('Measure TAD on fluoro', 'BEFORE  YOU  CLOSE')]
def s5_tips(t, main, glow, fx):
    a = win(t, 24.1, 30.6, 0.4, 0.4)
    if a <= 0: return
    p = eo(prog(t, 24.2, 24.8))
    label(main, 'TIPS  &  TRICKS', W / 2, 330, a * p, SKY, 56, 'MontBlack', 6)
    label(main, 'How to keep TAD under 25 mm', W / 2, 425, a * eo(prog(t, 24.5, 25.1)), WHITE, 44, 'MontXB', 1)
    for i, (arx, en) in enumerate(TIPS):
        t0 = 25.0 + 1.35 * i
        q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 640 + i * 290 + 30 * (1 - q)
        put(main, card(900, 240), W / 2, y, a * q)
        num = tl(f'0{i + 1}', 'MontBlack', 92, SKY)
        put(main, num, W / 2 + 362, y, a * q)
        label(main, arx, W / 2 - 75, y - 28, a * q, WHITE, 46, 'MontBlack', 1)
        label(main, en, W / 2 - 75, y + 50, a * q, (190, 212, 235), 27, 'MontXB', 2)

def s6_take(t, main, glow, fx):
    a = win(t, 30.4, 34.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', W / 2, 560, a * eo(prog(t, 30.5, 31.0)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 30.7, 31.3))
    big = tl('TAD < 25', 'MontBlack', 170, WHITE, track=1)
    put(glow, big, W / 2, 760, a * 0.35 * clamp(p), clamp(p, 0, 1.2))
    put(main, big, W / 2, 760, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'MILLIMETRES', W / 2, 900, a * eo(prog(t, 31.0, 31.5)), SKY, 36, 'MontXB', 14)
    q = eo(prog(t, 31.4, 32.0))
    put(main, card(880, 200, fill=(24, 44, 70, 160), outline=(120, 175, 216, 80)), W / 2, 1190, a * q)
    label(main, 'Baumgaertner MR, et al.', W / 2, 1140, a * q, ICE, 30, 'MontXB', 1)
    label(main, 'J Bone Joint Surg Am. 1995;77(7):1058–64', W / 2, 1192, a * q, ICE, 32, 'MontXB', 0)
    label(main, '198 fractures  ·  0 cut-outs at TAD of 25 mm or less', W / 2, 1245, a * q, (190, 212, 235), 27, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', W / 2, 1420, a * eo(prog(t, 32.0, 32.6)), WHITE, 38, 'MontXB', 1)

SCENES = [s1_hook, s2_context, s3_define, s4_rule, s5_tips, s6_take]

def zoom_for(t): return 1.0 + 0.025 * ((t % 8.5) / 8.5)

def frame(fi):
    t = fi / FPS
    main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H))
    fx = {'flash': 0.0, 'shake': 0.0}
    d = ImageDraw.Draw(main)
    for x, y, sp, sz, ph in R.P:
        px = (x * W + 20 * math.sin(t * 0.5 + ph * 6)) % W; py = (y * H - t * (20 + 60 * sp)) % H
        rr = 1 + 2.5 * sz; al = int(30 + 70 * sz * (0.6 + 0.4 * math.sin(t * 2 + ph * 9)))
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(170, 220, 255, al))
    st = script_time(t)
    for s in SCENES: s(st, main, glow, fx)
    g = glow.resize((W // 4, H // 4), Image.BILINEAR)
    g1 = np.asarray(g.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32)
    g2 = np.asarray(g.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BILINEAR), np.float32)
    m = np.asarray(main, np.float32); ma = m[..., 3:4] / 255
    out = R.bg * (1 - ma) + m[..., :3] * ma
    out += (g1[..., :3] * (g1[..., 3:4] / 255)) * 1.0 + (g2[..., :3] * (g2[..., 3:4] / 255)) * 1.2
    out = out * R.vign + R.grain[fi % 6] * 0.8
    out += fx['flash'] * 255 * np.array([0.75, 0.92, 1.0], np.float32)
    fade = min(prog(t, 0, 0.3), 1 - prog(t, DUR - 0.5, DUR))
    img = Image.fromarray(np.clip(out * fade, 0, 255).astype(np.uint8))
    z = zoom_for(t)
    cw, ch = W / z, H / z
    return img.transform((W, H), Image.EXTENT, ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2), Image.BILINEAR)

if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for tt in sys.argv[2:]: frame(int(float(tt) * FPS)).save(f'still_{tt}.jpg', quality=85)
    else:
        a, b, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i).tobytes())
        p.stdin.close(); p.wait()
