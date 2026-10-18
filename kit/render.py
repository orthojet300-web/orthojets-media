import sys, math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 25.0
NF = int(DUR * FPS)
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts') + '/'
F = lambda n, s: ImageFont.truetype(FD + n + '.ttf', s)

# brand palette from the Orthojets logo: navy #223F61, sky #78AFD8, white
NAVY = (34, 63, 97)
SKY = (120, 175, 216)
CYAN = (135, 192, 238)   # glow version of sky
ICE = (226, 238, 250)
ORANGE = SKY            # accent slot now uses brand sky
WHITE = (255, 255, 255)

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def prog(t, a, b): return clamp((t - a) / (b - a))
def eo(x): return 1 - (1 - x) ** 3
def ei(x): return x ** 3
def eio(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def eback(x):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def win(t, a, b, fi=0.4, fo=0.4):
    """alpha envelope for an element visible from a to b"""
    if t < a or t > b: return 0.0
    return min(eo(prog(t, a, a + fi)) if fi > 0 else 1, 1 - ei(prog(t, b - fo, b)) if fo > 0 else 1)

# ---------------------------------------------------------------- text cache
_tc = {}
def text_layer(txt, font, size, color=WHITE, track=0, ar=False):
    key = (txt, font, size, color, track, ar)
    if key in _tc: return _tc[key]
    f = F(font, size)
    if ar:
        bb = f.getbbox(txt, direction='rtl', language='ar')
        w, h = bb[2] - bb[0] + 40, bb[3] - bb[1] + 40
        im = Image.new('RGBA', (w, h))
        ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), txt, font=f, fill=color + (255,), direction='rtl', language='ar')
    else:
        ws = [f.getlength(c) for c in txt]
        tw = int(sum(ws) + track * (len(txt) - 1))
        asc, desc = f.getmetrics()
        im = Image.new('RGBA', (tw + 40, asc + desc + 40))
        d = ImageDraw.Draw(im); x = 20
        for c, cw in zip(txt, ws):
            d.text((x, 20), c, font=f, fill=color + (255,)); x += cw + track
        im = im.crop(im.getbbox()); im = pad(im, 20)
    _tc[key] = im
    return im

def pad(im, p):
    n = Image.new('RGBA', (im.width + 2 * p, im.height + 2 * p)); n.paste(im, (p, p)); return n

def put(dst, layer, cx, cy, alpha=1.0, scale=1.0, blur=0.0):
    if alpha <= 0.003: return
    l = layer
    if abs(scale - 1) > 0.002:
        l = l.resize((max(1, int(l.width * scale)), max(1, int(l.height * scale))), Image.BILINEAR)
    if blur > 0.3:
        l = pad(l, int(blur * 2)).filter(ImageFilter.GaussianBlur(blur))
    if alpha < 0.999:
        a = l.getchannel('A').point(lambda v: int(v * alpha)); l = l.copy(); l.putalpha(a)
    dst.alpha_composite(l, (int(cx - l.width / 2), int(cy - l.height / 2)))

# ---------------------------------------------------------------- static assets
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.45) / H) ** 2)
bg = np.zeros((H, W, 3), np.float32)
bg[..., 0] = 7 + 27 * np.exp(-r * 3.8); bg[..., 1] = 12 + 50 * np.exp(-r * 3.8); bg[..., 2] = 21 + 78 * np.exp(-r * 3.8)
vign = np.clip(1.15 - 1.25 * r ** 1.6, 0.18, 1.0)[..., None]
rng = np.random.default_rng(7)
grain = [rng.normal(0, 7, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
grain = [np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None] for g in grain]
P = rng.random((140, 5))  # x, y, speed, size, phase

import femur as FEM
def build_bone():
    img, edge = FEM.render()
    eimg = Image.new('RGBA', img.size, ICE + (0,)); eimg.putalpha(edge)
    return img, eimg

BONE, BONE_EDGE = build_bone()

def bone_icon(w=380, col=CYAN):
    h = int(w * 0.36); im = Image.new('RGBA', (w + 40, h + 40)); d = ImageDraw.Draw(im)
    r = h * 0.27; cy = 20 + h / 2
    for cx in (20 + r, 20 + w - r):
        for dy in (-r * 0.85, r * 0.85): d.ellipse((cx - r, cy + dy - r, cx + r, cy + dy + r), fill=col + (255,))
    d.rounded_rectangle((20 + r, cy - r * 0.62, 20 + w - r, cy + r * 0.62), r * 0.3, fill=col + (255,))
    return im

ICON = bone_icon()

# ---------------------------------------------------------------- scenes
def ecg_y(u):
    """normalized ECG shape over u in [0,1] per beat"""
    if 0.30 < u < 0.36: return -0.25 * math.sin((u - .30) / .06 * math.pi)
    if 0.40 < u < 0.43: return 0.25 * (u - .40) / .03
    if 0.43 <= u < 0.47: return 0.25 - 1.6 * (u - .43) / .04
    if 0.47 <= u < 0.51: return -1.35 + 1.75 * (u - .47) / .04
    if 0.51 <= u < 0.53: return 0.4 - 0.4 * (u - .51) / .02
    if 0.62 < u < 0.74: return -0.35 * math.sin((u - .62) / .12 * math.pi)
    return 0.0

def scene1(t, main, glow, fx):
    a = win(t, 0.0, 3.6, 0.3, 0.35)
    if a <= 0: return
    cy = 860; amp = 150
    head = eo(prog(t, 0.2, 2.6))
    n = 260; pts = []
    for i in range(n + 1):
        x = i / n
        if x > head: break
        beat = (x * 2.2) % 1.0
        big = 1.0 + (1.2 if 1.4 < x * 2.2 < 1.9 else 0)
        pts.append((60 + x * (W - 120), cy + ecg_y(beat) * amp * big))
    if len(pts) > 1:
        d = ImageDraw.Draw(glow); d2 = ImageDraw.Draw(main)
        col = tuple(int(c * a) for c in CYAN)
        d.line(pts, fill=col + (int(255 * a),), width=10, joint='curve')
        d2.line(pts, fill=ICE + (int(255 * a),), width=4, joint='curve')
        hx, hy = pts[-1]
        if head < 1:
            d.ellipse((hx - 22, hy - 22, hx + 22, hy + 22), fill=WHITE + (int(255 * a),))
    # spike flash
    fx['flash'] += 0.55 * math.exp(-((t - 1.95) / 0.09) ** 2)
    txt = text_layer('لكل عظمة قصة', 'CairoBlack', 118, ar=True)
    p = prog(t, 1.5, 2.4)
    put(main, txt, W / 2, 1210 + 40 * (1 - eo(p)), a * eo(p), 1.0, 14 * (1 - eo(p)))
    sub = text_layer('EVERY BONE TELLS A STORY', 'MontLight', 36, (200, 235, 250), track=9)
    p2 = prog(t, 2.0, 2.8)
    put(main, sub, W / 2, 1330, a * eo(p2) * 0.9)

def scene2(t, main, glow, fx):
    a = win(t, 3.4, 8.2, 0.3, 0.45)
    if a <= 0: return
    lt = t - 3.4
    s = 0.86 + 0.05 * lt / 4.8
    bw, bh = int(BONE.width * s), int(BONE.height * s)
    ox, oy = W // 2 - bw // 2 + 20, 1030 - bh // 2
    reveal = eio(prog(t, 3.6, 5.6))
    cut = int(bh * reveal)
    if cut > 2:
        b = BONE.resize((bw, bh), Image.BILINEAR).crop((0, 0, bw, cut))
        e = BONE_EDGE.resize((bw, bh), Image.BILINEAR).crop((0, 0, bw, cut))
        flick = 0.92 + 0.08 * math.sin(t * 37)
        put_tl(main, b, ox, oy, a * flick)
        put_tl(glow, e, ox, oy, a)
        put_tl(main, e, ox, oy, a * 0.8)
    if 0 < reveal < 1:
        sy = oy + cut; d = ImageDraw.Draw(glow)
        d.rectangle((0, sy - 3, W, sy + 3), fill=CYAN + (255,))
        ImageDraw.Draw(main).rectangle((0, sy - 1, W, sy + 1), fill=WHITE + (230,))
    # HUD labels
    LM = FEM.LANDMARKS
    labels = [('FEMORAL HEAD', -1, 150, -60, 4.3), ('GREATER TROCHANTER', 1, 80, -60, 4.55),
              ('NECK', -1, 110, 45, 4.8), ('LESSER TROCHANTER', -1, 150, 55, 5.05),
              ('SHAFT', 1, 150, -60, 5.3), ('MEDIAL CONDYLE', -1, 120, -70, 5.55), ('LATERAL CONDYLE', 1, 110, -70, 5.8)]
    d = ImageDraw.Draw(main)
    for name, side, dx, dy, t0 in labels:
        px, py = LM[name]
        p = eo(prog(t, t0, t0 + 0.5))
        if p <= 0: continue
        X, Y = ox + px * s, oy + py * s
        ex = X + side * dx * p; ey = Y + dy * p
        al = int(255 * a * p)
        d.ellipse((X - 7, Y - 7, X + 7, Y + 7), outline=ICE + (al,), width=2)
        d.line([(X, Y), (X + side * 0.4 * dx * p, ey), (ex, ey)], fill=ICE + (al,), width=2)
        lab = text_layer(name, 'MontLight', 25, ICE, track=3)
        lx = ex + side * (lab.width / 2 - 14)
        lx = min(max(lx, 24 + lab.width / 2 - 20), W - 24 - lab.width / 2 + 20)
        put(main, lab, lx, ey - 22, a * p)
    t1 = text_layer('العظام بلغة العلم', 'CairoBlack', 104, ar=True)
    p = prog(t, 4.0, 4.8)
    put(main, t1, W / 2, 250 - 30 * (1 - eo(p)), a * eo(p), 1, 12 * (1 - eo(p)))
    t2 = text_layer('ORTHOPEDICS  ·  DECODED', 'MontLight', 32, (175, 205, 232), track=8)
    put(main, t2, W / 2, 360, a * eo(prog(t, 4.5, 5.2)) * 0.9)

def put_tl(dst, im, x, y, alpha):
    put(dst, im, x + im.width / 2, y + im.height / 2, alpha)

WORDS = [('تشريح', 'ANATOMY'), ('كسور', 'FRACTURES'), ('جراحة', 'SURGERY'), ('فحص سريري', 'CLINICAL EXAM')]
def scene3(t, main, glow, fx):
    for i, (ar, en) in enumerate(WORDS):
        t0 = 8.0 + 1.25 * i; t1 = t0 + 1.25
        if not (t0 <= t < t1 + 0.01): continue
        lt = t - t0
        pin = eo(prog(lt, 0, 0.28)); pout = ei(prog(lt, 1.02, 1.25))
        a = pin * (1 - pout)
        num = text_layer('0%d' % (i + 1), 'MontBlack', 520, (48, 84, 124))
        put(main, num, W / 2 + 60 * (1 - pin) - 140 * pout, 900, a * 0.55)
        big = text_layer(ar, 'CairoBlack', 190, ar=True)
        put(main, big, W / 2 - 200 * pout, 940, a, 1.35 - 0.35 * eback(clamp(lt / 0.32)) + 0.03 * lt, 22 * (1 - pin) + 18 * pout)
        sub = text_layer(en, 'MontXB', 44, ORANGE, track=14)
        put(main, sub, W / 2 + 200 * pout, 1110, a * eo(prog(lt, 0.15, 0.45)))
        # streaks
        d = ImageDraw.Draw(glow)
        sp = prog(lt, 0, 0.35)
        if 0 < sp < 1:
            for k in range(7):
                y = 700 + k * 75 + (k * 37 % 40)
                x2 = W * (1.4 - 1.8 * eo(sp)) + k * 40
                d.line([(x2, y), (x2 + 380, y)], fill=(CYAN if k % 2 else ORANGE) + (int(200 * (1 - sp)),), width=3)
        if i == 1 and lt > 0.25:  # fracture crack
            cp = eo(prog(lt, 0.25, 0.45))
            pts = [(330, 860), (420, 930), (480, 905), (560, 990), (620, 950), (750, 1030)]
            n = max(2, int(len(pts) * cp))
            d.line(pts[:n], fill=WHITE + (int(255 * a),), width=6)
            ImageDraw.Draw(main).line(pts[:n], fill=(10, 14, 22, int(255 * a)), width=5)
        fx['flash'] += 0.18 * math.exp(-((lt - 0.05) / 0.06) ** 2)

def phone_layer(t, a, main, glow, cx, cy):
    pw, ph = 380, 680
    d = ImageDraw.Draw(glow); dm = ImageDraw.Draw(main)
    box = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2)
    dm.rounded_rectangle(box, 48, fill=(14, 26, 42, int(225 * a)))
    d.rounded_rectangle(box, 48, outline=CYAN + (int(255 * a),), width=6)
    dm.rounded_rectangle(box, 48, outline=ICE + (int(255 * a),), width=3)
    sb = BONE.resize((int(BONE.width * 0.38), int(BONE.height * 0.38)))
    put(main, sb, cx, cy - 20, a * 0.75)
    tri = [(cx - 40, cy - 60), (cx - 40, cy + 20), (cx + 35, cy - 20)]
    dm.ellipse((cx - 85, cy - 105, cx + 85, cy + 65), fill=(0, 0, 0, int(140 * a)))
    dm.polygon(tri, fill=WHITE + (int(255 * a),))
    bp = prog(t, 13.4, 17.0)
    y = cy + ph / 2 - 50
    dm.rounded_rectangle((cx - 150, y, cx + 150, y + 8), 4, fill=(255, 255, 255, int(60 * a)))
    dm.rounded_rectangle((cx - 150, y, cx - 150 + 300 * bp, y + 8), 4, fill=ORANGE + (int(255 * a),))
    d.rounded_rectangle((cx - 150, y, cx - 150 + 300 * bp, y + 8), 4, fill=ORANGE + (int(255 * a),))

def scene4(t, main, glow, fx):
    a = win(t, 13.0, 17.4, 0.35, 0.4)
    if a <= 0: return
    pin = eback(prog(t, 13.0, 13.6))
    phone_layer(t, a, main, glow, W / 2, 700 + 120 * (1 - clamp(pin)))
    l1 = text_layer('فيديوهات قصيرة', 'CairoBlack', 112, ar=True)
    p = eo(prog(t, 13.5, 14.2)); put(main, l1, W / 2, 1195 + 40 * (1 - p), a * p, 1, 12 * (1 - p))
    p = eo(prog(t, 14.1, 14.7))
    for k, (wd, x) in enumerate([('علمية', W / 2 + 230), ('دقيقة', W / 2), ('مركّزة', W / 2 - 230)]):
        pk = eo(prog(t, 14.1 + k * 0.15, 14.6 + k * 0.15))
        put(main, text_layer(wd, 'CairoBold', 62, ICE, ar=True), x, 1335 + 30 * (1 - pk), a * pk * 0.95)
    dd = ImageDraw.Draw(main)
    for x in (W / 2 + 115, W / 2 - 115):
        dd.ellipse((x - 7, 1335 - 7, x + 7, 1335 + 7), fill=ORANGE + (int(255 * a * p),))
    d = ImageDraw.Draw(main); p = eo(prog(t, 14.8, 15.4))
    d.line([(W / 2 - 160 * p, 1440), (W / 2 + 160 * p, 1440)], fill=ORANGE + (int(255 * a),), width=3)
    l3 = text_layer('لطلاب الطب وأطباء العظام', 'CairoBold', 66, ORANGE, ar=True)
    p = eo(prog(t, 15.0, 15.7)); put(main, l3, W / 2, 1540 + 30 * (1 - p), a * p)

LAYER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo_layers')
MARK_PX = 520
_mk = {}
def mark_layers():
    if not _mk:
        for n in ('ring', 'speed', 'bone'):
            _mk[n] = Image.open(os.path.join(LAYER_DIR, f'L_{n}.png')).convert('RGBA').resize((MARK_PX, MARK_PX), Image.LANCZOS)
        yy_, xx_ = np.mgrid[0:MARK_PX, 0:MARK_PX].astype(np.float32) - MARK_PX / 2
        # clockwise sweep starting at 190 deg (where the ring begins)
        _mk['ang'] = (np.degrees(np.arctan2(yy_, xx_)) - 190) % 360
        _mk['ring_a'] = np.asarray(_mk['ring'].getchannel('A'), np.float32)
        f = F('MontXB', 168); tr = -5
        full = 'orthojets'; ws = [f.getlength(c) for c in full]
        asc, desc = f.getmetrics()
        wm = Image.new('RGBA', (int(sum(ws) + tr * 8) + 40, asc + desc + 40)); d = ImageDraw.Draw(wm); x = 20
        for i_, (c, cw) in enumerate(zip(full, ws)):
            d.text((x, 20), c, font=f, fill=(WHITE if i_ < 5 else SKY) + (255,)); x += cw + tr
        _mk['wm'] = pad(wm.crop(wm.getbbox()), 20)
    return _mk

def draw_logo(t, main, glow, fx, a, cy_icon, cy_text, sc=1.0):
    M = mark_layers(); cx = W / 2
    # 1) bone launches in from lower-left along the mark's own 35 deg flight path
    land = eo(prog(t, 17.3, 18.0))
    dist = 1100 * (1 - land)
    ux, uy = math.cos(math.radians(35)), -math.sin(math.radians(35))
    bx, by = cx - ux * dist, cy_icon - uy * dist
    tail = 1 - prog(t, 18.0, 18.7)
    if tail > 0:
        d = ImageDraw.Draw(glow)
        L0 = 260 + 500 * (1 - land)
        for k, off in enumerate((-34, 0, 34)):
            nx, ny = -uy * off, ux * off   # perpendicular offset
            hx, hy = bx - ux * 120 + nx, by - uy * 120 + ny
            ln = L0 * (0.7 if k != 1 else 1.0) * tail
            d.line([(hx, hy), (hx - ux * ln, hy - uy * ln)], fill=(CYAN if k != 1 else WHITE) + (int(230 * a * tail),), width=6)
    put(glow, M['bone'], bx, by, a * 0.55)
    put(main, M['bone'], bx, by, a)
    # 2) ring draws itself clockwise around the landing bone
    rp = eio(prog(t, 17.65, 18.35))
    if rp > 0:
        alpha = M['ring_a'] * (M['ang'] <= rp * 272)
        ring = M['ring'].copy(); ring.putalpha(Image.fromarray(alpha.astype(np.uint8)))
        put(glow, ring, cx, cy_icon, a * 0.6)
        put(main, ring, cx, cy_icon, a)
    # 3) speed lines snap in after impact
    sp = eo(prog(t, 18.0, 18.35))
    if sp > 0:
        put(main, M['speed'], cx - 40 * (1 - sp) * ux, cy_icon - 40 * (1 - sp) * uy, a * sp)
    fx['flash'] += 0.55 * math.exp(-((t - 18.0) / 0.1) ** 2)
    fx['shake'] += 12 * math.exp(-((t - 18.0) / 0.12) ** 2)
    # 4) wordmark + light sweep
    Lw = M['wm']
    p = eo(prog(t, 18.15, 18.75))
    if p > 0:
        put(main, Lw, W / 2, cy_text + 30 * (1 - p), a * p, 1.06 - 0.06 * p, 16 * (1 - p))
        s_ = prog(t, 18.8, 19.7)
        if 0 < s_ < 1:
            m = np.asarray(Lw.getchannel('A'), np.float32) / 255
            xs = np.arange(Lw.width)[None, :] - np.arange(Lw.height)[:, None] * 0.4
            c = -200 + s_ * (Lw.width + 400)
            band = np.exp(-((xs - c) / 60) ** 2) * m
            hl = Image.new('RGBA', Lw.size, WHITE + (0,)); hl.putalpha(Image.fromarray((band * 255 * a).astype(np.uint8)))
            put(glow, hl, W / 2, cy_text, 1.0)

def scene5(t, main, glow, fx):
    a = win(t, 17.2, 25.0, 0.01, 0.6)
    if a <= 0: return
    up = eio(prog(t, 21.2, 22.0))
    cy_icon = 700 - 210 * up; cy_text = 1060 - 210 * up
    draw_logo(t, main, glow, fx, a, cy_icon, cy_text)
    p = eo(prog(t, 19.0, 19.7))
    tg = text_layer('طبّ العظام ببساطة', 'CairoBold', 70, ICE, ar=True)
    put(main, tg, W / 2, cy_text + 160 + 20 * (1 - p), a * p * (1 - 0.0 * up))
    en = text_layer('ORTHOPEDICS  ·  SIMPLIFIED', 'MontLight', 30, (175, 205, 232), track=8)
    put(main, en, W / 2, cy_text + 250, a * eo(prog(t, 19.4, 20.1)) * 0.9)
    # CTA — calm glass pill: soft fill, thin sky outline, follow (+) badge, gentle rise, one slow sheen
    cp = eo(prog(t, 21.8, 22.6))
    if cp > 0:
        pill = cta_pill()
        cy = 1380 + 36 * (1 - cp)
        put(glow, pill, W / 2, cy, a * cp * 0.12)
        put(main, pill, W / 2, cy, a * cp)
        sh = prog(t, 22.9, 24.0)
        if 0 < sh < 1:
            m = np.asarray(_cta['mask'], np.float32) / 255
            xs = np.arange(pill.width)[None, :] - np.arange(pill.height)[:, None] * 0.5
            c = -150 + sh * (pill.width + 300)
            band = np.exp(-((xs - c) / 55) ** 2) * m * 0.22
            hl = Image.new('RGBA', pill.size, WHITE + (0,)); hl.putalpha(Image.fromarray((band * 255 * a).astype(np.uint8)))
            put(main, hl, W / 2, cy, 1.0)
        h = text_layer('محتوى جديد كل أسبوع', 'Cairo', 40, (190, 212, 235), ar=True)
        put(main, h, W / 2, 1505, a * eo(prog(t, 22.5, 23.2)) * 0.8)

_cta = {}
def cta_pill():
    if 'img' in _cta: return _cta['img']
    pw, ph = 620, 116; P = 24
    SS = 3
    big = Image.new('RGBA', ((pw + 2 * P) * SS, (ph + 2 * P) * SS)); d = ImageDraw.Draw(big)
    box = (P * SS, P * SS, (P + pw) * SS, (P + ph) * SS)
    d.rounded_rectangle(box, ph * SS / 2, fill=(42, 74, 112, 150), outline=SKY + (200,), width=2 * SS)
    # follow badge on the reading-start side (right, RTL)
    r = 38; bx = P + pw - 16 - r; by = P + ph / 2
    d.ellipse(((bx - r) * SS, (by - r) * SS, (bx + r) * SS, (by + r) * SS), fill=SKY + (255,))
    k = 15; th = 5
    d.rounded_rectangle(((bx - k) * SS, (by - th / 2) * SS, (bx + k) * SS, (by + th / 2) * SS), th * SS / 2, fill=NAVY + (255,))
    d.rounded_rectangle(((bx - th / 2) * SS, (by - k) * SS, (bx + th / 2) * SS, (by + k) * SS), th * SS / 2, fill=NAVY + (255,))
    img = big.resize((pw + 2 * P, ph + 2 * P), Image.LANCZOS)
    tx = text_layer('تابع القناة على تيك توك', 'CairoBold', 48, WHITE, ar=True)
    tcx = P + (pw - 2 * r - 16) / 2
    img.alpha_composite(tx, (int(tcx - tx.width / 2), int(P + ph / 2 - tx.height / 2 + 3)))
    mask = Image.new('L', img.size); ImageDraw.Draw(mask).rounded_rectangle((P, P, P + pw, P + ph), ph / 2, fill=255)
    _cta['img'] = img; _cta['mask'] = mask
    return img

# ---------------------------------------------------------------- frame
def zoom_for(t):
    if t < 3.5: return 1.0 + 0.05 * t / 3.5
    if t < 8.0: return 1.0 + 0.04 * (t - 3.5) / 4.5
    if t < 13: return 1.02
    if t < 17.2: return 1.0 + 0.04 * (t - 13) / 4.2
    return 1.0 + 0.03 * (t - 17.2) / 7.8

def render(fi):
    t = fi / FPS
    main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H))
    fx = {'flash': 0.0, 'shake': 0.0}
    # dust particles
    d = ImageDraw.Draw(main)
    for x, y, sp, sz, ph in P:
        px = (x * W + 20 * math.sin(t * 0.5 + ph * 6)) % W
        py = (y * H - t * (20 + 60 * sp)) % H
        rr = 1 + 2.5 * sz; al = int(40 + 90 * sz * (0.6 + 0.4 * math.sin(t * 2 + ph * 9)))
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(170, 220, 255, al))
    scene1(t, main, glow, fx); scene2(t, main, glow, fx); scene3(t, main, glow, fx)
    scene4(t, main, glow, fx); scene5(t, main, glow, fx)
    # compose
    g = glow.resize((W // 4, H // 4), Image.BILINEAR)
    g1 = np.asarray(g.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32)
    g2 = np.asarray(g.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BILINEAR), np.float32)
    m = np.asarray(main, np.float32)
    ma = m[..., 3:4] / 255
    out = bg * (1 - ma) + m[..., :3] * ma
    out += (g1[..., :3] * (g1[..., 3:4] / 255)) * 1.1 + (g2[..., :3] * (g2[..., 3:4] / 255)) * 1.4
    # fade from / to black
    fade = min(prog(t, 0, 0.5), 1 - prog(t, 24.4, 25.0))
    out = out * vign + grain[fi % 6]
    out += fx['flash'] * 255 * np.array([0.75, 0.92, 1.0], np.float32)
    out = np.clip(out * fade, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    z = zoom_for(t); sh = fx['shake']
    if z > 1.001 or sh > 0.5:
        cw, ch = W / z, H / z
        dx = sh * math.sin(t * 91); dy = sh * math.cos(t * 77)
        x0 = (W - cw) / 2 + dx; y0 = (H - ch) / 2 + dy
        img = img.transform((W, H), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BILINEAR)
    return img

if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for tt in sys.argv[2:]:
            render(int(float(tt) * FPS)).save(f'still_{tt}.jpg', quality=85)
    else:
        a, b, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(render(i).tobytes())
        p.stdin.close(); p.wait()
