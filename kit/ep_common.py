"""Shared helpers for Orthojets episodes (frame compositing, text, cards, reading holds)."""
import math, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render as R

W, H, FPS = R.W, R.H, R.FPS
SKY, NAVY, WHITE, ICE, CYAN = R.SKY, R.NAVY, R.WHITE, R.ICE, R.CYAN
WARN = (236, 112, 104)
GOOD = (120, 210, 170)
SUB = (190, 212, 235)
put, tl, prog, eo, ei, eio, win, clamp = R.put, R.text_layer, R.prog, R.eo, R.ei, R.eio, R.win, R.clamp

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
    main = Image.new('RGBA', (W, H)); glow = Image.new('RGBA', (W, H))
    fx = {'flash': 0.0}
    d = ImageDraw.Draw(main)
    for x, y, sp, sz, ph in R.P:
        px = (x * W + 20 * math.sin(t * 0.5 + ph * 6)) % W; py = (y * H - t * (20 + 60 * sp)) % H
        rr = 1 + 2.5 * sz; al = int(30 + 70 * sz * (0.6 + 0.4 * math.sin(t * 2 + ph * 9)))
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(170, 220, 255, al))
    for s in scenes: s(st, main, glow, fx)
    g = glow.resize((W // 4, H // 4), Image.BILINEAR)
    g1 = np.asarray(g.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR), np.float32)
    g2 = np.asarray(g.filter(ImageFilter.GaussianBlur(18)).resize((W, H), Image.BILINEAR), np.float32)
    m = np.asarray(main, np.float32); ma = m[..., 3:4] / 255
    out = R.bg * (1 - ma) + m[..., :3] * ma
    out += (g1[..., :3] * (g1[..., 3:4] / 255)) * 1.0 + (g2[..., :3] * (g2[..., 3:4] / 255)) * 1.2
    out = out * R.vign + R.grain[fi % 6] * 0.8
    out += fx['flash'] * 255 * np.array([0.75, 0.92, 1.0], np.float32)
    fade = min(prog(t, 0, 0.3), 1 - prog(t, tlm.dur - 0.5, tlm.dur))
    img = Image.fromarray(np.clip(out * fade, 0, 255).astype(np.uint8))
    z = 1.0 + 0.025 * ((t % 8.5) / 8.5); cw, ch = W / z, H / z
    return img.transform((W, H), Image.EXTENT, ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2), Image.BILINEAR)

def main_cli(tlm, scenes):
    if sys.argv[1] == 'still':
        for tt in sys.argv[2:]: make_frame(int(float(tt) * FPS), tlm, scenes).save(f'still_{tt}.jpg', quality=85)
    elif sys.argv[1] == 'dur':
        print(tlm.dur)
    else:
        a, b, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                              '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', outp], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(make_frame(i, tlm, scenes).tobytes())
        p.stdin.close(); p.wait()
