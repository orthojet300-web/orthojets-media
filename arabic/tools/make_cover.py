#!/usr/bin/env python3
"""Cover for «عظامك ببساطة» episodes (owner 2026-10-24: the cover must carry «عظامك ببساطة · كل يوم سبت»).

usage: python3 make_cover.py STILL.png OUT.jpg N "سطر السؤال الأول" "سطر السؤال الثاني"
STILL = a frame of the hook scene (its single caption word sits at y ~320-560 and is painted over here).
Layout (top -> bottom): series badge (navy capsule «عظامك ببساطة» + sky tab «كل يوم سبت» with a calendar icon,
episode number), the episode's English topic pill from the still, the two-line Arabic question, the hook art.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts', 'Cairo.ttf')
NAVY, SKY, WHITE, CORAL, GOLD = (34, 63, 97), (120, 175, 216), (255, 255, 255), (226, 96, 86), (255, 214, 102)

def F(size, wght=900):
    f = ImageFont.truetype(FONT, size); f.set_variation_by_axes([wght]); return f

def blend_rows(a, y0, y1):
    top, bot = a[y0 - 1].copy(), a[y1 + 1].copy()
    for y in range(y0, y1 + 1):
        k = (y - y0) / (y1 - y0); a[y] = top * (1 - k) + bot * k

def tsize(d, txt, f):
    bb = d.textbbox((0, 0), txt, font=f, direction='rtl', language='ar'); return bb, bb[2] - bb[0], bb[3] - bb[1]

def main():
    still, out, n, l1, l2 = sys.argv[1:6]
    im = Image.open(still).convert('RGB'); W, H = im.size
    a = np.asarray(im).astype(np.float32)
    blend_rows(a, 60, 200)        # room for the series badge
    blend_rows(a, 315, 560)       # remove the single caption word
    im = Image.fromarray(a.astype(np.uint8)).convert('RGBA')

    # ---- series badge: navy capsule (series name) joined to a sky tab (every Saturday)
    lay = Image.new('RGBA', (W, H)); d = ImageDraw.Draw(lay)
    name, when = 'عظامك ببساطة', 'كل يوم سبت'
    fn, fw = F(54), F(40, 800)
    bbn, wn, hn = tsize(d, name, fn); bbw, ww, hw = tsize(d, when, fw)
    icon = 46; pad = 34; gap = 22
    tab_w = ww + icon + gap + 2 * pad; cap_w = wn + 2 * pad + 20; h = 96
    total = cap_w + tab_w; x0 = (W - total) / 2; y0 = 92
    # shadow
    sh = Image.new('RGBA', (W, H)); ds = ImageDraw.Draw(sh)
    ds.rounded_rectangle((x0, y0 + 10, x0 + total, y0 + h + 10), radius=h / 2, fill=(20, 40, 70, 90))
    lay.alpha_composite(sh.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(lay)
    # RTL: series name on the RIGHT (read first), "every Saturday" tab on the LEFT
    d.rounded_rectangle((x0, y0, x0 + total, y0 + h), radius=h / 2, fill=SKY + (255,))
    d.rounded_rectangle((x0 + tab_w - 6, y0, x0 + total, y0 + h), radius=h / 2, fill=NAVY + (255,))
    d.rectangle((x0 + tab_w - 6, y0, x0 + tab_w + h / 2, y0 + h), fill=NAVY + (255,))
    # series name
    cx = x0 + tab_w + cap_w / 2
    d.text((cx - wn / 2 - bbn[0], y0 + h / 2 - hn / 2 - bbn[1]), name, font=fn, fill=WHITE, direction='rtl', language='ar')
    # "every Saturday" + calendar icon (icon on the right of the text, RTL)
    tx = x0 + pad; ty = y0 + h / 2 - hw / 2 - bbw[1]
    d.text((tx - bbw[0], ty), when, font=fw, fill=NAVY, direction='rtl', language='ar')
    ix = x0 + pad + ww + gap; iy = y0 + (h - icon) / 2
    d.rounded_rectangle((ix, iy + 6, ix + icon, iy + icon), radius=8, outline=NAVY, width=5)
    d.rectangle((ix, iy + 6, ix + icon, iy + 17), fill=NAVY)
    for k in (0.28, 0.72): d.line((ix + icon * k, iy, ix + icon * k, iy + 12), fill=NAVY, width=5)
    for r in range(2):
        for c in range(3):
            px, py = ix + 9 + c * 12, iy + 24 + r * 11; d.rectangle((px, py, px + 5, py + 5), fill=NAVY)
    # episode number medal
    m = 62; mx, my = x0 + total - m * 0.55, y0 - m * 0.35
    d.ellipse((mx, my, mx + m, my + m), fill=GOLD + (255,), outline=WHITE, width=5)
    fnum = F(36); bb, w_, h_ = tsize(d, str(n), fnum)
    d.text((mx + m / 2 - w_ / 2 - bb[0], my + m / 2 - h_ / 2 - bb[1]), str(n), font=fnum, fill=NAVY)
    im.alpha_composite(lay)

    # ---- the question (two lines)
    d = ImageDraw.Draw(im)
    for y, txt, size, fill, stroke, sw in ((318, l1, 94, WHITE, NAVY, 7), (445, l2, 70, CORAL, WHITE, 5)):
        f = F(size); bb, w_, h_ = tsize(d, txt, f)
        while w_ > W - 80: size -= 4; f = F(size); bb, w_, h_ = tsize(d, txt, f)
        d.text(((W - w_) / 2 - bb[0], y), txt, font=f, fill=fill, direction='rtl', language='ar', stroke_width=sw, stroke_fill=stroke)
    im.convert('RGB').save(out, quality=93)
    print(out)

if __name__ == '__main__':
    main()
