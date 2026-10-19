#!/usr/bin/env python3
"""Orthojets weekly Instagram carousel ("This week's pearls").

usage: python3 tools/build_carousel.py OUTDIR SLUG [SLUG ...] [--week "Oct 20 - 22"]

Reads each episode's `carousel` block (and `series`/`part`) from topics.json and writes
OUTDIR/slide01.jpg ... : cover, one slide per episode, then a follow slide. 1080x1350 (4:5).
The `carousel` block in topics.json:
  {"headline": "TAD < 25 mm", "sub": "one short line", "points": ["...", "...", "..."], "ref": "Author et al., Journal Year"}
Every point must be one of the episode's verified on-screen claims (no new facts on the carousel).
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kit'))
import render as R
from ep_common import pill, card, check, arrow

W, H = 1080, 1350
SKY, NAVY, WHITE, ICE = R.SKY, R.NAVY, R.WHITE, R.ICE
SUB = (190, 212, 235)
GOOD = (120, 210, 170)
F = R.F

# ---------------------------------------------------------------- text helpers
def seg_text(txt, font, size, col, track=1):
    """text layer that swaps Δ to DejaVu (Montserrat has no Δ)"""
    if 'Δ' not in txt:
        return R.text_layer(txt, font, size, col, track=track)
    parts, buf = [], ''
    for c in txt:
        if c == 'Δ':
            if buf: parts.append((buf, font)); buf = ''
            parts.append(('Δ', 'DejaVuB'))
        else:
            buf += c
    if buf: parts.append((buf, font))
    ims = [R.text_layer(t, f, size, col, track=track) for t, f in parts]
    w = sum(i.width for i in ims) - 34 * (len(ims) - 1); h = max(i.height for i in ims)
    out = Image.new('RGBA', (w, h)); x = 0
    for i in ims:
        out.alpha_composite(i, (x, (h - i.height) // 2)); x += i.width - 34
    return out

def wrap(txt, font, size, maxw):
    f = F(font, size); words = txt.split(); lines, cur = [], ''
    for w_ in words:
        t = (cur + ' ' + w_).strip()
        if f.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w_
    if cur: lines.append(cur)
    return lines

def put_c(img, layer, cx, cy): img.alpha_composite(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
def put_l(img, layer, x, cy): img.alpha_composite(layer, (int(x - 20), int(cy - layer.height / 2)))

def fit_size(txt, font, start, maxw, minsize=60):
    s = start
    while s > minsize and F(font, s).getlength(txt.replace('Δ', 'D')) > maxw: s -= 6
    return s

# ---------------------------------------------------------------- background
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H * 0.42) / H) ** 2)
BG = np.stack([7 + 27 * np.exp(-_r * 3.6), 12 + 50 * np.exp(-_r * 3.6), 21 + 78 * np.exp(-_r * 3.6)], -1)
VIGN = np.clip(1.15 - 1.2 * _r ** 1.6, 0.25, 1.0)[..., None]
GRAIN = np.random.default_rng(7).normal(0, 5, (H, W, 1)).astype(np.float32)

def finish(main, glow):
    g = glow.filter(ImageFilter.GaussianBlur(26)); a = np.asarray(g, np.float32)
    m = np.asarray(main, np.float32); ma = m[..., 3:4] / 255
    out = BG.copy() + a[..., :3] * (a[..., 3:4] / 255) * 1.3
    out = out * (1 - ma) + m[..., :3] * ma
    out = out * VIGN + GRAIN
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def deco(main):
    """faint brand orbit in the corner + speed lines"""
    d = ImageDraw.Draw(main)
    d.arc((W - 330, -250, W + 250, 330), 100, 300, fill=SKY + (38,), width=10)
    for k, (y, ln) in enumerate([(78, 70), (104, 110), (130, 70)]):
        d.line([(60, y), (60 + ln, y)], fill=SKY + (150 if k == 1 else 90,), width=6)

def brand_small(main, y=1285):
    M = R.mark_layers()
    wm = M['wm'].resize((int(M['wm'].width * 0.26), int(M['wm'].height * 0.26)), Image.LANCZOS)
    mk = Image.new('RGBA', M['ring'].size)
    for n in ('ring', 'speed', 'bone'): mk.alpha_composite(M[n])
    mk = mk.resize((76, 76), Image.LANCZOS)
    total = mk.width + 14 + wm.width; x0 = W / 2 - total / 2
    main.alpha_composite(mk, (int(x0), int(y - mk.height / 2)))
    main.alpha_composite(wm, (int(x0 + mk.width + 14), int(y - wm.height / 2)))

def counter(main, i, n):
    put_c(main, R.text_layer(f'{i}/{n}', 'MontXB', 28, SUB, track=2), W - 110, 104)

# ---------------------------------------------------------------- slides
def cover(n_eps, n, week):
    main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H)); deco(main)
    put_c(main, pill('THIS WEEK ON ORTHOJETS', 'MontXB', 30, NAVY, SKY + (255,), h=66), W / 2, 330)
    big = R.text_layer(str(n_eps), 'MontBlack', 330, WHITE)
    put_c(glow, R.text_layer(str(n_eps), 'MontBlack', 330, SKY), W / 2, 560); put_c(main, big, W / 2, 560)
    put_c(main, R.text_layer('ORTHO PEARLS', 'MontBlack', 96, WHITE, track=4), W / 2, 780)
    put_c(main, R.text_layer('you should know by heart', 'MontXB', 46, SKY), W / 2, 870)
    if week: put_c(main, R.text_layer(week.upper(), 'MontXB', 30, SUB, track=4), W / 2, 960)
    sw = R.text_layer('SWIPE', 'MontXB', 34, WHITE, track=6); d = ImageDraw.Draw(main)
    put_c(main, sw, W / 2 - 40, 1110); arrow(d, W / 2 + 40, 1110, W / 2 + 130, SKY, 1.0)
    brand_small(main); counter(main, 1, n)
    return finish(main, glow)

def pearl(ep, idx, n):
    c = ep['carousel']; main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H)); deco(main)
    tag = f"{ep.get('series_tag', 'ORTHO TIP')}  #{ep.get('part', ep['n'])}"
    put_c(main, pill(tag, 'MontXB', 30, NAVY, SKY + (255,), h=66), W / 2, 230)
    hs = fit_size(c['headline'], 'MontBlack', 150, 920)
    hl = seg_text(c['headline'], 'MontBlack', hs, WHITE)
    put_c(glow, seg_text(c['headline'], 'MontBlack', hs, SKY), W / 2, 380); put_c(main, hl, W / 2, 380)
    y = 400 + hl.height / 2 + 40
    for ln in wrap(c['sub'], 'MontXB', 44, 900):
        put_c(main, seg_text(ln, 'MontXB', 44, SKY), W / 2, y); y += 58
    # points card
    rows = [wrap(p, 'MontXB', 36, 760) for p in c['points']]
    ch = sum(len(r) * 50 + 46 for r in rows) + 40
    top = max(y + 30, 600); main.alpha_composite(card(960, int(ch)), (60, int(top)))
    d = ImageDraw.Draw(main); yy_ = top + 44
    for r in rows:
        check(d, 118, yy_ + 18, 38, GOOD, 1.0)
        for k, ln in enumerate(r):
            put_l(main, seg_text(ln, 'MontXB', 36, WHITE), 158, yy_ + 18 + k * 50)
        yy_ += len(r) * 50 + 46
    refs = wrap(c['ref'], 'MontXB', 28, 960)
    ref_y = min(top + ch + 60, 1205 - 40 * (len(refs) - 1))
    for k, ln in enumerate(refs):
        put_c(main, R.text_layer(ln, 'MontXB', 28, SUB), W / 2, ref_y + k * 40)
    brand_small(main); counter(main, idx, n)
    return finish(main, glow)

def follow(n):
    main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H)); deco(main)
    M = R.mark_layers(); mk = Image.new('RGBA', M['ring'].size)
    for nm in ('ring', 'speed', 'bone'): mk.alpha_composite(M[nm])
    mk = mk.resize((300, 300), Image.LANCZOS); put_c(glow, mk, W / 2, 390); put_c(main, mk, W / 2, 390)
    put_c(main, M['wm'].resize((int(M['wm'].width * 0.62), int(M['wm'].height * 0.62)), Image.LANCZOS), W / 2, 640)
    put_c(main, R.text_layer('Ortho pearls at jet speed', 'MontXB', 48, SKY), W / 2, 750)
    for k, (t, col) in enumerate([('SAVE this post', WHITE), ('SHARE it with a colleague', WHITE), ('FOLLOW for new pearls', WHITE)]):
        y = 880 + k * 92; d = ImageDraw.Draw(main); check(d, 250, y, 40, GOOD, 1.0)
        put_l(main, R.text_layer(t, 'MontXB', 42, col), 300, y)
    put_c(main, pill('NEW TIPS  SUNDAY TO THURSDAY', 'MontXB', 30, NAVY, SKY + (255,), h=66), W / 2, 1180)
    counter(main, n, n)
    return finish(main, glow)

def main():
    args = [a for a in sys.argv[1:]]
    week = ''
    if '--week' in args:
        i = args.index('--week'); week = args[i + 1]; del args[i:i + 2]
    out, slugs = args[0], args[1:]
    topics = json.load(open(os.path.join(ROOT, 'topics.json')))
    series = topics.get('series', {})
    eps = []
    for s in slugs:
        ep = next(e for e in topics['episodes'] if e['slug'] == s)
        if 'carousel' not in ep: sys.exit(f'{s}: no carousel block in topics.json')
        if ep.get('series') in series: ep['series_tag'] = series[ep['series']]['tag']
        eps.append(ep)
    os.makedirs(out, exist_ok=True); n = len(eps) + 2
    slides = [cover(len(eps), n, week)] + [pearl(e, i + 2, n) for i, e in enumerate(eps)] + [follow(n)]
    for i, im in enumerate(slides, 1):
        p = os.path.join(out, f'slide{i:02d}.jpg'); im.save(p, quality=95, subsampling=0); print(p)

if __name__ == '__main__':
    main()
