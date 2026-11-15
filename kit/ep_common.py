"""Shared helpers for Orthojets episodes (frame compositing, text, cards, reading holds)."""
import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render as R

W, H, FPS = R.W, R.H, R.FPS
SKY, NAVY, WHITE, ICE, CYAN = R.SKY, R.NAVY, R.WHITE, R.ICE, R.CYAN

# ---------------------------------------------------------------- visual themes (owner decision 2026-10-23)
# Brand constants never change (navy base, logo, Montserrat, sky in the logo/outro). Inside the episode body:
#   * the ACCENT colour (the name SKY that every script uses) follows the episode's SERIES,
#   * the BACKGROUND style and the CAMERA move change every 5 episodes (about once a week).
# Override with env ORTHO_SERIES=numbers|rules|classified|dontmiss and ORTHO_BG=0..3 if needed.
import json as _json, os as _os, re as _re
SERIES_ACCENT = {
    'numbers':    (120, 175, 216),   # sky (brand)
    'rules':      (236, 196, 98),    # warm gold
    'classified': (170, 140, 250),   # violet
    'dontmiss':   (240, 120, 200),   # rose-magenta (kept apart from the coral WARN colour)
}
BG_STYLES = ['stars', 'grid', 'waves', 'aurora']

def _detect_theme():
    series, n = _os.environ.get('ORTHO_SERIES'), None
    try:
        script = _os.path.abspath(sys.argv[0]); slug = _os.path.basename(_os.path.dirname(script))
        m = _re.match(r'ep(\d+)', slug); n = int(m.group(1)) if m else None
        if series is None:
            root = _os.path.dirname(_os.path.dirname(_os.path.dirname(script)))
            for e in _json.load(open(_os.path.join(root, 'topics.json')))['episodes']:
                if e.get('slug') == slug: series = e.get('series')
    except Exception:
        pass
    bg = int(_os.environ['ORTHO_BG']) if 'ORTHO_BG' in _os.environ else (((n - 1) // 5) % 4 if n else 0)
    return series or 'numbers', bg

THEME_SERIES, THEME_BG = _detect_theme()
SKY = SERIES_ACCENT.get(THEME_SERIES, SERIES_ACCENT['numbers'])
CYAN = tuple(min(255, int(c * 1.1 + 10)) for c in SKY)
# tint the background glow very slightly toward the accent (navy stays the base)
_acc = np.array(SKY, np.float32) / 255
BG = R.bg * (0.86 + 0.14 * (_acc / _acc.mean()))[None, None, :] if THEME_SERIES != 'numbers' else R.bg
PART_COL = tuple(int(0.55 * c + 0.45 * w) for c, w in zip(SKY, (170, 220, 255)))
WARN = (236, 112, 104)
GOOD = (120, 210, 170)
SUB = (190, 212, 235)
put, tl, prog, eo, ei, eio, win, clamp = R.put, R.text_layer, R.prog, R.eo, R.ei, R.eio, R.win, R.clamp

# ---------------------------------------------------------------- anti-aliased drawing (quality upgrade 2026-11-16)
# Pillow's ImageDraw does not anti-alias lines, circles, arcs or polygons, so diagrams had jagged ("stair-step") edges.
# Every ImageDraw.Draw(main) in an episode script now returns AADraw: each shape is drawn supersampled (2-4x) in its own
# small box, downsampled and composited in order. Scripts need no change. Text calls go straight to Pillow (already AA).
import inspect as _insp
from PIL import ImageDraw as _ID
_RawDraw = getattr(_ID, '_orthojets_raw_draw', _ID.Draw)
_ID._orthojets_raw_draw = _RawDraw
_AA_OPS = ('line', 'ellipse', 'polygon', 'arc', 'chord', 'pieslice', 'rectangle', 'rounded_rectangle')
_SIGS = {}

class AADraw:
    def __init__(self, im):
        self.im = im; self._raw = _RawDraw(im)
    def __getattr__(self, name):
        raw = getattr(self._raw, name)
        if name not in _AA_OPS: return raw
        if name not in _SIGS: _SIGS[name] = _insp.signature(getattr(_RawDraw(Image.new('L', (1, 1))), name))
        sig = _SIGS[name]
        def op(*a, **k):
            b = sig.bind(*a, **k); args = b.arguments
            try:
                pts = np.asarray(list(args['xy']) if not isinstance(args['xy'], np.ndarray) else args['xy'], np.float64).reshape(-1, 2)
            except Exception:
                return raw(*a, **k)
            if not len(pts): return
            wdt = float(args.get('width') or (1 if name == 'line' else 0))
            padd = wdt + 4
            x0 = int(max(0, math.floor(pts[:, 0].min() - padd))); y0 = int(max(0, math.floor(pts[:, 1].min() - padd)))
            x1 = int(min(self.im.width, math.ceil(pts[:, 0].max() + padd))); y1 = int(min(self.im.height, math.ceil(pts[:, 1].max() + padd)))
            if x1 <= x0 or y1 <= y0: return
            area = (x1 - x0) * (y1 - y0)
            S = 4 if area < 90_000 else (3 if area < 500_000 else 2)
            q = (pts - (x0, y0)) * S
            if name in ('line', 'polygon'): args['xy'] = [tuple(p) for p in q]
            else: args['xy'] = [float(q[0, 0]), float(q[0, 1]), float(q[-1, 0]), float(q[-1, 1])]
            if 'width' in args and args['width']: args['width'] = max(1, int(round(args['width'] * S)))
            if 'radius' in args and args['radius']: args['radius'] = args['radius'] * S
            tmp = Image.new('RGBA', ((x1 - x0) * S, (y1 - y0) * S))
            getattr(_RawDraw(tmp), name)(**args)
            t = np.asarray(tmp.reduce(S), np.float32)
            reg = self.im.crop((x0, y0, x1, y1)); m = np.asarray(reg, np.float32)
            ta, ma = t[..., 3:4] / 255, m[..., 3:4] / 255
            # colour: normal "over"; alpha: max (like Pillow's own replace mode), so semi-transparent strokes drawn as
            # many overlapping pieces do not build up darker joints
            oa = ta + ma * (1 - ta)
            rgb = (t[..., :3] * ta + m[..., :3] * ma * (1 - ta)) / np.maximum(oa, 1e-6)
            out = np.dstack([rgb, np.maximum(ta, ma) * 255])
            self.im.paste(Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), 'RGBA'), (x0, y0))
        return op

def _draw(im, mode=None):
    if mode is None and getattr(im, '_aa', False): return AADraw(im)
    return _RawDraw(im, mode)
_ID.Draw = _draw

def aa_image(w, h):
    """new transparent RGBA layer whose ImageDraw.Draw is anti-aliased"""
    im = Image.new('RGBA', (w, h)); im._aa = True; return im

def label(main, txt, x, y, a, col=ICE, size=28, font='MontXB', track=1):
    put(main, tl(txt, font, size, col, track=track), x, y, a)

def rich(parts, size):
    """compose [(text, font, color), ...] on one baseline (e.g. a DejaVu Δ + Montserrat)"""
    ims = [tl(t, f, size, c, track=1) for t, f, c in parts]
    w = sum(i.width for i in ims) - 30 * (len(ims) - 1); h = max(i.height for i in ims)
    out = Image.new('RGBA', (w, h)); x = 0
    for i in ims:
        out.alpha_composite(i, (x, (h - i.height) // 2)); x += i.width - 30
    return out

def dP(size, col=WHITE, rest='P'):
    return rich([('Δ', 'DejaVuB', col), (rest, 'MontBlack', col)], size)

def dot(d, x, y, r, col, a): d.ellipse((x - r, y - r, x + r, y + r), fill=col + (int(255 * a),))

def check(d, x, y, sz, col, a, w=7):
    d.line([(x - sz * .5, y), (x - sz * .12, y + sz * .38), (x + sz * .55, y - sz * .42)], fill=col + (int(255 * a),), width=w, joint='curve')

def cross(d, x, y, sz, col, a, w=7):
    d.line([(x - sz * .4, y - sz * .4), (x + sz * .4, y + sz * .4)], fill=col + (int(255 * a),), width=w)
    d.line([(x + sz * .4, y - sz * .4), (x - sz * .4, y + sz * .4)], fill=col + (int(255 * a),), width=w)

def arrow(d, x0, y, x1, col, a, w=6):
    d.line([(x0, y), (x1 - 4, y)], fill=col + (int(255 * a),), width=w)
    d.polygon([(x1, y), (x1 - 22, y - 13), (x1 - 22, y + 13)], fill=col + (int(255 * a),))

def pill(text, font, size, fg, bg, padx=34, h=None):
    t = tl(text, font, size, fg, track=4); h = h or t.height + 4; w = t.width + 2 * padx
    im = Image.new('RGBA', (w * 3, h * 3)); ImageDraw.Draw(im).rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), h * 3 / 2, fill=bg)
    im = im.resize((w, h), Image.LANCZOS); im.alpha_composite(t, (int(w / 2 - t.width / 2), int(h / 2 - t.height / 2)))
    return im

_cards = {}
def card(w, h, fill=(30, 54, 84, 170), outline=SKY + (120,), r=34):
    k = (w, h, fill, outline, r)
    if k not in _cards:
        im = Image.new('RGBA', (w * 2, h * 2))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), r * 2, fill=fill, outline=outline, width=4)
        _cards[k] = im.resize((w, h), Image.LANCZOS)
    return _cards[k]

class Timeline:
    def __init__(self, script_dur, holds):
        self.holds = holds; self.dur = script_dur + sum(h for _, h in holds)
    def script(self, tr):
        off = 0.0
        for T, h in self.holds:
            if tr - off <= T: return tr - off
            if tr - off < T + h: return T
            off += h
        return tr - off
    def real(self, st): return st + sum(h for T, h in self.holds if T < st)

def make_frame(fi, tlm, scenes):
    t = fi / FPS; st = tlm.script(t)
    main = aa_image(W, H); glow = Image.new('RGBA', (W, H)); bgl = aa_image(W, H)
    fx = {'flash': 0.0}
    draw_background(ImageDraw.Draw(bgl), glow, t)
    for s in scenes: s(st, main, glow, fx)
    g = glow.resize((W // 4, H // 4), Image.BILINEAR)
    g1 = np.asarray(g.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BICUBIC), np.float32)
    g2 = np.asarray(g.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BICUBIC), np.float32)
    # background (gradient + weekly pattern) gets the slow camera move; the content layer is never resampled, so text
    # and line art stay pixel-sharp (before 2026-11-16 the whole frame was re-zoomed every frame, which softened it).
    b = np.asarray(bgl, np.float32); ba = b[..., 3:4] / 255
    bgimg = Image.fromarray(np.clip(BG * (1 - ba) + b[..., :3] * ba, 0, 255).astype(np.uint8))
    if THEME_BG == 1:
        z = 1.03; dy = 18 * math.sin(t * 0.35)
    elif THEME_BG == 2:
        z = 1.02 + 0.012 * math.sin(t * 0.6); dy = 0
    else:
        z = 1.0 + 0.025 * ((t % 8.5) / 8.5); dy = 0
    cw, ch = W / z, H / z
    bgimg = bgimg.transform((W, H), Image.EXTENT, ((W - cw) / 2, (H - ch) / 2 + dy, (W + cw) / 2, (H + ch) / 2 + dy), Image.BICUBIC)
    m = np.asarray(main, np.float32); ma = m[..., 3:4] / 255
    out = np.asarray(bgimg, np.float32) * (1 - ma) + m[..., :3] * ma
    out += (g1[..., :3] * (g1[..., 3:4] / 255)) * 1.0 + (g2[..., :3] * (g2[..., 3:4] / 255)) * 1.2
    out = out * R.vign + R.grain[fi % 6] * 0.45 * (1 - ma)      # light grain on the background only (no banding, clean text)
    out += fx['flash'] * 255 * np.array([0.75, 0.92, 1.0], np.float32)
    fade = min(prog(t, 0, 0.3), 1 - prog(t, tlm.dur - 0.5, tlm.dur))
    return Image.fromarray(np.clip(out * fade, 0, 255).astype(np.uint8))

def draw_background(d, glow, t):
    """weekly background style, always subtle and behind the content"""
    if THEME_BG == 0:                                   # stars: drifting particles (original look)
        for x, y, sp, sz, ph in R.P:
            px = (x * W + 20 * math.sin(t * 0.5 + ph * 6)) % W; py = (y * H - t * (20 + 60 * sp)) % H
            rr = 1 + 2.5 * sz; al = int(30 + 70 * sz * (0.6 + 0.4 * math.sin(t * 2 + ph * 9)))
            d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=PART_COL + (al,))
    elif THEME_BG == 1:                                 # grid: fine blueprint grid drifting slowly + node dots
        step = 90; off = (t * 12) % step
        for x in range(-step, W + step, step):
            d.line([(x + off, 0), (x + off, H)], fill=PART_COL + (28,), width=1)
        for y in range(-step, H + step, step):
            d.line([(0, y + off * 0.6), (W, y + off * 0.6)], fill=PART_COL + (28,), width=1)
        for k, (x, y, sp, sz, ph) in enumerate(R.P[:40]):
            gx = (int(x * W / step) * step + off) % W; gy = (int(y * H / step) * step + off * 0.6) % H
            al = int(40 + 60 * (0.5 + 0.5 * math.sin(t * 1.5 + ph * 9)))
            d.ellipse((gx - 3, gy - 3, gx + 3, gy + 3), fill=PART_COL + (al,))
    elif THEME_BG == 2:                                 # waves: soft flowing lines
        for k in range(7):
            yb = 180 + k * 260; amp = 26 + 8 * k; ph = t * (0.35 + 0.05 * k) + k
            pts = [(x, yb + amp * math.sin(x / 170 + ph) + 10 * math.sin(x / 61 - ph * 1.7)) for x in range(-20, W + 40, 30)]
            d.line(pts, fill=PART_COL + (34 + 6 * (k % 3),), width=2, joint='curve')
    else:                                               # aurora: slow soft light blobs in the accent colour
        gd = ImageDraw.Draw(glow)
        for k, (x, y, sp, sz, ph) in enumerate(R.P[:5]):
            cx = W * (0.15 + 0.7 * x) + 120 * math.sin(t * 0.21 + ph * 6)
            cy = H * (0.1 + 0.8 * y) + 160 * math.cos(t * 0.17 + ph * 4)
            rr = 160 + 120 * sz
            gd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=SKY + (26,))
        for x, y, sp, sz, ph in R.P[:50]:
            px = (x * W) % W; py = (y * H - t * (10 + 25 * sp)) % H; rr = 1 + 1.5 * sz
            d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=PART_COL + (int(25 + 45 * sz),))

def main_cli(tlm, scenes, hook_end=None, resume=None):
    """hook_end = SCRIPT time at which the hook has been fully readable long enough; the 1 s logo flash is
    inserted there (add_brand.py --hook-first). resume = SCRIPT time where the next scene starts (after the hook has
    faded out) so the hook does not reappear after the logo. 'script.py hookend' prints both in real seconds."""
    if sys.argv[1] == 'still':
        for tt in sys.argv[2:]: make_frame(int(float(tt) * FPS), tlm, scenes).save(f'still_{tt}.png')
    elif sys.argv[1] == 'dur':
        print(tlm.dur)
    elif sys.argv[1] == 'hookend':
        h = tlm.real(hook_end) if hook_end is not None else 3.0
        r = tlm.real(resume) if resume is not None else h
        print(h, r)
    else:
        a, b, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '10', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(make_frame(i, tlm, scenes).tobytes())
        p.stdin.close(); p.wait()

# ---------------------------------------------------------------- depth & material helpers (quality upgrade 2026-11-16)
# Flat one-colour silhouettes look like clip-art. Build the shape as a greyscale MASK (any size, supersampled is fine),
# then turn it into a lit, rounded object with one of these. Light comes from the top-left in every episode.
LIGHT = (-0.55, -0.83)

def _blur_arr(m, r):
    return np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255

def volume(mask, col, alpha=1.0, r=None, relief=0.55, light=LIGHT):
    """mask (PIL 'L') -> RGBA layer of colour col shaded like a rounded 3-D body (soft tissue, organs, cards, implants)."""
    m = np.asarray(mask, np.float32) / 255
    r = r or max(3.0, min(m.shape) / 30)
    e = _blur_arr((m > 0.06).astype(np.float32), r)        # shade from the silhouette, not from alpha fades
    gy, gx = np.gradient(e)
    lit = np.clip(-(gx * light[0] + gy * light[1]) * r * 2.2, -1, 1)          # + on edges facing the light
    core = np.clip(e * 1.6, 0, 1)                                              # 0 at the rim, 1 inside
    f = (0.80 + 0.20 * core) * (1 + relief * 0.45 * lit)
    rgb = np.array(col, np.float32)[None, None, :] * f[..., None]
    rgb += (255 - rgb) * (np.clip(lit, 0, 1) ** 2 * (1 - core) * relief * 0.6)[..., None]  # soft specular rim
    out = np.dstack([np.clip(rgb, 0, 255), m * 255 * alpha]).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')

def bone_xray(mask, col=ICE, alpha=1.0, cortex=None, seed=3, light=LIGHT):
    """mask (PIL 'L') -> RGBA radiograph-style bone: bright cortical rim, darker medulla with a fine trabecular texture."""
    m = np.asarray(mask, np.float32) / 255
    h, w = m.shape; cortex = cortex or max(2.0, min(h, w) / 60)
    sil = (m > 0.06).astype(np.float32)
    e = _blur_arr(sil, cortex)
    rim = np.clip((1 - e) * 2.4, 0, 1) * sil                                   # cortex = outer band
    rng = np.random.default_rng(seed)
    n = rng.random((max(1, h // 4), max(1, w // 4))).astype(np.float32)
    tex = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    med = 0.84 + 0.08 * tex                                                    # medulla brightness with trabeculae
    gy, gx = np.gradient(_blur_arr(sil, cortex * 3))
    lit = np.clip(-(gx * light[0] + gy * light[1]) * cortex * 6, -1, 1)
    b = np.clip(med * (1 - rim) + 1.0 * rim + 0.08 * lit, 0, 1.0)
    rgb = np.array(col, np.float32)[None, None, :] * b[..., None]
    rgb += (255 - rgb) * (rim * 0.45 + np.clip(lit, 0, 1) * 0.25)[..., None]     # cortex glows whiter
    a = m * alpha
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), 'RGBA')

def drop_shadow(layer, dx=10, dy=18, blur=16, opacity=0.5):
    """returns layer with a soft shadow underneath (same size; keep 30+ px transparent margin around the art)"""
    a = layer.getchannel('A').filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * opacity))
    sh = Image.new('RGBA', layer.size, (2, 6, 14, 0)); sh.putalpha(a)
    out = Image.new('RGBA', layer.size); out.alpha_composite(sh, (dx, dy)); out.alpha_composite(layer)
    return out
