"""Anatomical left femur, anterior view (medial side on viewer's left), rendered as an X-ray.
Proportions follow adult femur ratios: head Ø ≈ 1.7× mid-shaft width, bicondylar width ≈ 3× shaft,
neck–shaft angle ≈ 125°, shaft inclined medially toward the knee, medial condyle larger."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

BW, BH = 480, 1300
HEAD_C, HEAD_R = (160, 140), 64

def catmull(pts, n=14, closed=True):
    if closed:
        P = pts + pts[:2]; rng_ = range(len(pts)); pre = lambda i: P[i - 1] if i else pts[-1]
    else:
        P = pts + [pts[-1], pts[-1]]; rng_ = range(len(pts) - 1); pre = lambda i: P[i - 1] if i else pts[0]
    out = []
    for i in rng_:
        p0, p1, p2, p3 = pre(i), P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n; t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    return out

def head_arc(a0, a1, step=12):
    cx, cy = HEAD_C; out = []
    a = a0
    while a <= a1 + 1e-6:
        r = HEAD_R
        # fovea capitis: small pit on the medial head surface
        if 188 <= a <= 200: r -= 4 * math.sin((a - 188) / 12 * math.pi)
        out.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
        a += step if not (180 <= a <= 205) else 3
    return out

def outline():
    # shaft axis tilts medially toward the knee
    ax = lambda y: 300 - 50 * (y - 330) / 710
    shaft_hw = {400: 40, 500: 37, 600: 36, 700: 36, 800: 38, 870: 43}
    lat = [(ax(y) + h, y) for y, h in shaft_hw.items()]
    med = [(ax(y) - h, y) for y, h in reversed(shaft_hw.items())]
    pts = []
    pts += [(224, 136), (246, 148), (266, 162), (284, 172), (300, 176)]                     # superior neck (concave)
    pts += [(309, 162), (315, 138), (322, 117), (340, 108), (362, 116), (379, 138)]          # greater trochanter
    pts += [(387, 175), (388, 215), (380, 262), (360, 320), (345, 360)]                      # lateral trochanter → subtrochanteric
    pts += lat
    pts += [(310, 930), (320, 975), (338, 1020), (354, 1060), (359, 1102), (357, 1130)]                   # lateral epicondyle
    pts += [(350, 1158), (334, 1190), (311, 1210), (288, 1213), (268, 1203), (253, 1195)]   # lateral condyle → notch
    pts += [(236, 1206), (212, 1223), (187, 1226), (164, 1215), (149, 1192)]                 # medial condyle (larger, lower)
    pts += [(141, 1157), (140, 1118), (145, 1080), (153, 1050), (166, 1018)]                 # medial epicondyle / adductor tubercle
    pts += [(192, 975), (208, 930)]
    pts += med
    pts += [(256, 362), (246, 344), (236, 330), (232, 312), (236, 296), (242, 282)]  # lesser trochanter (rounded)
    pts += [(244, 262), (226, 240), (205, 222), (186, 207)]                                  # inferior neck
    pts += head_arc(74, 352)                                                                 # femoral head
    return pts

def render():
    SS = 3
    big = Image.new('L', (BW * SS, BH * SS)); d = ImageDraw.Draw(big)
    poly = catmull(outline(), n=8)
    d.polygon([(x * SS, y * SS) for x, y in poly], fill=255)
    m = big.resize((BW, BH), Image.LANCZOS)
    M = np.asarray(m, np.float32) / 255
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    rng = np.random.default_rng(11)

    # cortex: distance-to-edge falloff (bright rim, darker medullary canal in the shaft)
    B1 = np.asarray(m.filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
    B2 = np.asarray(m.filter(ImageFilter.GaussianBlur(22)), np.float32) / 255
    rim = np.clip((1 - B1) * 3.0, 0, 1)
    canal = np.clip((B2 - 0.82) * 6, 0, 1)
    shaft_zone = np.clip((yy - 380) / 80, 0, 1) * np.clip((900 - yy) / 80, 0, 1)
    dens = 0.42 + 0.58 * rim - 0.32 * canal * shaft_zone

    # trabecular bone in the epiphyses (head, neck, trochanters, condyles)
    ends = np.clip((360 - yy) / 120, 0, 1) + np.clip((yy - 950) / 120, 0, 1)
    n1 = np.asarray(Image.fromarray((rng.random((BH // 3, BW // 3)) * 255).astype(np.uint8)).resize((BW, BH), Image.BICUBIC), np.float32) / 255
    n2 = np.asarray(Image.fromarray((rng.random((BH // 9, BW // 9)) * 255).astype(np.uint8)).resize((BW, BH), Image.BICUBIC), np.float32) / 255
    dens *= (1 - 0.28 * ends * (n1 - 0.5) * 2 * 0.6) * (0.92 + 0.16 * n2)

    # compressive trabeculae arch through the neck (bright band head → medial cortex)
    def band(p0, p1, w, k):
        x0, y0 = p0; x1, y1 = p1; vx, vy = x1 - x0, y1 - y0; L2 = vx * vx + vy * vy
        t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / L2, 0, 1)
        dd = np.hypot(xx - (x0 + t * vx), yy - (y0 + t * vy))
        return k * np.exp(-(dd / w) ** 2)
    dens += band((150, 128), (250, 330), 14, 0.16) + band((170, 100), (330, 150), 10, 0.08)

    # overlapping structures seen on AP film: lesser trochanter, greater trochanter, patella
    def blob(cx, cy, rx, ry, k):
        return k * np.clip(1 - ((xx - cx) / rx) ** 2 - ((yy - cy) / ry) ** 2, 0, 1) ** 0.6
    dens += blob(244, 314, 18, 18, 0.12) + blob(345, 150, 40, 48, 0.07)
    dens += blob(256, 1070, 58, 74, 0.08)                       # patella superimposed on distal femur
    # faint lines: intertrochanteric line, physeal scars, intercondylar notch roof
    lines = Image.new('L', (BW, BH)); ld = ImageDraw.Draw(lines)
    ld.line(catmull([(308, 160), (306, 215), (288, 268), (260, 310)], 10, False), fill=255, width=3)
    ld.arc((108, 92, 220, 196), 300, 60, fill=255, width=2)                          # head physeal scar
    ld.line([(165, 1130), (205, 1122), (250, 1128), (300, 1120), (345, 1126)], fill=255, width=2)   # distal physeal scar
    ld.arc((222, 1150, 290, 1235), 200, 340, fill=255, width=3)                    # notch roof (Blumensaat)
    ld.ellipse((198, 1000, 314, 1142), outline=255, width=2)                        # patella outline
    L = np.asarray(lines.filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255
    dens += 0.18 * L

    inner = np.clip(M * dens, 0, 1.2)
    rgb = np.stack([inner * c for c in (165, 208, 242)], -1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert('RGBA')
    img.putalpha(Image.fromarray(np.clip(inner * 255 * 1.1, 0, 255).astype(np.uint8)))
    mb = m.point(lambda v: 255 if v > 128 else 0)
    edge = ImageChops.subtract(mb, mb.filter(ImageFilter.MinFilter(5)))
    return img, edge

# anatomical landmarks in bone space for HUD labels
LANDMARKS = {
    'FEMORAL HEAD': (150, 128), 'NECK': (238, 192), 'GREATER TROCHANTER': (350, 140),
    'LESSER TROCHANTER': (238, 314), 'SHAFT': (316, 640),
    'MEDIAL CONDYLE': (168, 1190), 'LATERAL CONDYLE': (335, 1170),
}

if __name__ == '__main__':
    img, edge = render()
    bg = Image.new('RGBA', img.size, (10, 18, 30, 255)); bg.alpha_composite(img)
    e = Image.new('RGBA', img.size, (226, 238, 250, 0)); e.putalpha(edge); bg.alpha_composite(e)
    d = ImageDraw.Draw(bg)
    for k, (x, y) in LANDMARKS.items(): d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 120, 80, 255))
    bg.convert('RGB').save('femur_test.png')
