"""Orthojets EP11 - DON'T MISS #6 - SCFE: Klein's line, the modified Klein's line and the lateral view.
Sources (verified on PubMed abstracts / metadata, 2026-11-14):
  Green DW, Mogekwu N, Scher DM, Handler S, Chalmers P, Widmann RF. A modification of Klein's line to improve sensitivity
    of the anterior-posterior radiograph in SCFE. J Pediatr Orthop 2009;29(5):449-53. PMID 19568015,
    doi 10.1097/BPO.0b013e3181aa20cd. Klein's line = line drawn parallel to the superior edge of the femoral neck on the AP
    view; lack of intersection with the epiphysis confirms a slip. Classic definition identified only 40.3% of slips
    (30 unilateral SCFE, 5 observers). Modified: measure the width of epiphysis lateral to Klein's line; a difference of
    2 mm between hips indicating a slip improved sensitivity to 79%.
  Rebich EJ, Lee SS, Schlechter JA. The S sign: a new radiographic tool to aid in the diagnosis of SCFE.
    J Emerg Med 2018;54(6):835-43. PMID 29550284, doi 10.1016/j.jemermed.2018.01.023. 20 observers, 35 SCFE patients.
    Klein's line (AP) sensitivity 68.3%, specificity 89.0%; S-sign (frog-leg lateral) 89.0% / 95.2%;
    combination 96.5% / 85.0%.
  Loder RT, Richards BS, Shapiro PS, Reznick LR, Aronson DD. Acute SCFE: the importance of physeal stability.
    J Bone Joint Surg Am 1993;75(8):1134-40. PMID 8354671, doi 10.2106/00004623-199308000-00002. Unstable = pain so severe
    that weight-bearing was not possible even with crutches. AVN in 14 of 30 unstable hips (47%) vs 0 of 25 stable hips.
  Ortegren J, Osterman J, Tiderius CJ. Patients' delay is the major cause for late diagnosis of SCFE.
    J Pediatr Orthop B 2021;30(2):105-10. PMID 32371650, doi 10.1097/BPB.0000000000000738. 54 stable SCFE. Knee-pain-dominant
    children: doctors' delay 14 vs 4 weeks (P=0.002). Total delay correlated with slip severity (R=0.59).
  Shnaekel AW et al. The atypical patient with SCFE may be at increased risk for a missed contralateral slip.
    Orthopedics 2020;43(2):e114-8. PMID 31930409, doi 10.3928/01477447-20200107-04. 56 patients, 5 missed contralateral
    slips (8.9%); Klein's line false-negative rate 40%.
  Diagram (schematic, labelled): adolescent proximal femora on an AP pelvis, heads pointing to the midline, neck-shaft angle
  about 135 degrees, greater and lesser trochanter, capital epiphysis separated from the neck by the physis (lucent line),
  acetabular roof. Klein's line runs along the superior neck border; normally it cuts a cap of the lateral epiphysis.
  In SCFE the epiphysis is displaced posteriorly / inferiorly, which on the AP view appears as medial-inferior displacement.
"""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kit'))
from ep_common import *

TL = Timeline(46.0, [(6.6, 0.8), (8.4, 1.0), (11.2, 1.2), (14.6, 1.6), (19.6, 1.4), (24.4, 1.6),
                     (29.6, 1.6), (32.6, 1.2), (38.9, 1.8), (43.4, 1.4)])

XC = 510
BONE = (206, 224, 242)
SS = 2

# ------------------------------------------------------------------ proximal femur, AP view (local canvas, head points LEFT)
LW, LH = 440, 600
HC = (150.0, 205.0)          # head centre
HR = 66.0                    # head radius
U = (math.sqrt(.5), math.sqrt(.5))      # neck axis, head -> shaft (neck-shaft angle ~135 deg)
N = (math.sqrt(.5), -math.sqrt(.5))     # superior (lateral) normal of the neck
NH = 43.0                    # neck half-width
P0 = 12.0                    # physis position along the neck axis (from the head centre)

def ax(a, o): return (HC[0] + a * U[0] + o * N[0], HC[1] + a * U[1] + o * N[1])

def meta_poly():
    pts = [ax(a, NH) for a in range(int(P0) + 2, 152, 10)]
    pts += [(295, 266), (308, 240), (326, 226), (346, 232), (360, 255), (368, 295), (368, 345), (360, 390)]   # greater troch
    pts += [(357, 450), (354, 520), (352, 600), (262, 600), (262, 520), (264, 455)]                           # shaft
    pts += [(256, 432), (240, 422), (232, 405), (240, 392), (248, 382)]                                       # lesser troch
    pts += [(236, 366), (218, 344), (204, 326)]                                                               # calcar
    pts += [ax(a, -NH) for a in range(110, int(P0) + 1, -10)]
    return pts

def halfplane(a_lo, a_hi):
    """mask (local px) of the strip a_lo <= axial coordinate <= a_hi, as a polygon"""
    far = 900
    return [ax(a_lo, -far), ax(a_lo, far), ax(a_hi, far), ax(a_hi, -far)]

_hc = {}
def hip_layer(slip=0.0, build=1.0):
    key = (round(slip, 1), round(build, 2))
    if key in _hc: return _hc[key]
    big = (LW * SS, LH * SS)
    sc = lambda pts, dx=0, dy=0: [((x + dx) * SS, (y + dy) * SS) for x, y in pts]
    # metaphysis = neck/shaft polygon + head part beyond the physis
    m = Image.new('L', big); dm = ImageDraw.Draw(m)
    dm.polygon(sc(meta_poly()), fill=255)
    hm = Image.new('L', big); dh = ImageDraw.Draw(hm)
    dh.ellipse(((HC[0] - HR) * SS, (HC[1] - HR) * SS, (HC[0] + HR) * SS, (HC[1] + HR) * SS), fill=255)
    hp = Image.new('L', big); ImageDraw.Draw(hp).polygon(sc(halfplane(P0 + 4, 400)), fill=255)
    # the metaphysis never projects above the superior neck border (Klein's line is tangent to it)
    far = 900
    sup = Image.new('L', big); ImageDraw.Draw(sup).polygon(sc([ax(-far, NH), ax(far, NH), ax(far, -far), ax(-far, -far)]), fill=255)
    hmeta = np.minimum(np.minimum(np.asarray(hm), np.asarray(hp)), np.asarray(sup))
    m = Image.fromarray(np.maximum(np.asarray(m), hmeta))
    # epiphysis = head part before the physis, displaced medially-inferiorly (-N) by slip
    dx, dy = -N[0] * slip, -N[1] * slip
    em = Image.new('L', big); ImageDraw.Draw(em).ellipse(((HC[0] - HR + dx) * SS, (HC[1] - HR + dy) * SS,
                                                          (HC[0] + HR + dx) * SS, (HC[1] + HR + dy) * SS), fill=255)
    ep = Image.new('L', big); ImageDraw.Draw(ep).polygon(sc(halfplane(-400, P0 - 4), dx, dy), fill=255)
    e = Image.fromarray(np.minimum(np.asarray(em), np.asarray(ep)))
    out = Image.new('RGBA', big)
    ba = clamp(build * 2)
    for mask, k in ((m, 1.0), (e, 1.0)):
        mk = mask.filter(ImageFilter.GaussianBlur(1.2 * SS))
        fill = Image.new('RGBA', big, BONE + (0,)); fill.putalpha(mk.point(lambda v: int(v * 0.42 * ba * k)))
        edge = Image.fromarray(np.clip(np.asarray(mk, np.int16) - np.asarray(mk.filter(ImageFilter.MinFilter(9)), np.int16), 0, 255).astype(np.uint8))
        ed = Image.new('RGBA', big, WHITE + (0,)); ed.putalpha(edge.point(lambda v: int(v * 0.95 * ba)))
        out.alpha_composite(fill); out.alpha_composite(ed)
    # medullary canal (darker) in the shaft
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((286 * SS, 430 * SS, 330 * SS, 640 * SS), 20 * SS, fill=(34, 63, 97, int(110 * ba)))
    # acetabular roof (fixed, the pelvis does not move)
    aa = clamp(build * 2 - 0.6)
    r = HR + 13
    d.arc(((HC[0] - r) * SS, (HC[1] - r) * SS, (HC[0] + r) * SS, (HC[1] + r) * SS), 140, 318, fill=BONE + (int(200 * aa),), width=9 * SS)
    im = out.resize((LW, LH), Image.LANCZOS)
    if len(_hc) > 60: _hc.clear()
    _hc[key] = im
    return im

def cap_layer(slip, col):
    """the part of the epiphysis lateral to Klein's line (offset > NH), filled with col"""
    big = (LW * SS, LH * SS)
    dx, dy = -N[0] * slip, -N[1] * slip
    em = Image.new('L', big); ImageDraw.Draw(em).ellipse(((HC[0] - HR + dx) * SS, (HC[1] - HR + dy) * SS,
                                                          (HC[0] + HR + dx) * SS, (HC[1] + HR + dy) * SS), fill=255)
    ep = Image.new('L', big); ImageDraw.Draw(ep).polygon([((x + dx) * SS, (y + dy) * SS) for x, y in halfplane(-400, P0 - 4)], fill=255)
    far = 900
    lp = Image.new('L', big); ImageDraw.Draw(lp).polygon([(x * SS, y * SS) for x, y in
                                                          (ax(-far, NH), ax(far, NH), ax(far, far), ax(-far, far))], fill=255)
    msk = Image.fromarray(np.minimum(np.minimum(np.asarray(em), np.asarray(ep)), np.asarray(lp)))
    im = Image.new('RGBA', big, col + (0,)); im.putalpha(msk.point(lambda v: int(v * 0.85)))
    return im.resize((LW, LH), Image.LANCZOS)

class Hip:
    """a placed hip: screen centre, scale, mirror (True = head points right = patient's right hip on an AP pelvis)"""
    def __init__(self, cx, cy, sc, mirror):
        self.cx, self.cy, self.sc, self.mirror = cx, cy, sc, mirror
    def S(self, p):
        x, y = p
        if self.mirror: x = LW - x
        return (self.cx + (x - LW / 2) * self.sc, self.cy + (y - LH / 2) * self.sc)
    def draw(self, main, a, slip=0.0, build=1.0):
        im = hip_layer(slip, build)
        if self.mirror: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        put(main, im, self.cx, self.cy, a, self.sc)
    def cap(self, main, glow, a, slip, col):
        if a <= 0: return
        im = cap_layer(slip, col)
        if self.mirror: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        put(main, im, self.cx, self.cy, a, self.sc); put(glow, im, self.cx, self.cy, a * 0.5, self.sc)
    def klein(self, main, glow, a, p, col, w=6):
        """Klein's line along the superior neck border, drawn from the neck toward and past the head (p = 0..1)"""
        if a <= 0 or p <= 0: return
        a0, a1 = 175, -120
        am = a0 + (a1 - a0) * p
        p0, p1 = self.S(ax(a0, NH)), self.S(ax(am, NH))
        d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
        d.line([p0, p1], fill=col + (int(255 * a),), width=w)
        g.line([p0, p1], fill=col + (int(120 * a),), width=w + 8)
    def width_bar(self, main, a, slip, col, letter=None, lcol=WHITE):
        """bracket from Klein's line to the lateral edge of the epiphysis, perpendicular to the line"""
        if a <= 0: return
        w = HR - slip - NH
        p0 = self.S(ax(-14, NH)); p1 = self.S(ax(-14, NH + max(w, 0)))
        d = ImageDraw.Draw(main)
        d.line([p0, p1], fill=col + (int(255 * a),), width=6)
        for p in (p0, p1):
            ux, uy = U if not self.mirror else (-U[0], U[1])
            d.line([(p[0] - 12 * ux, p[1] - 12 * uy), (p[0] + 12 * ux, p[1] + 12 * uy)], fill=col + (int(255 * a),), width=5)
        if letter:
            lp = self.S(ax(-14, NH + max(w, 0) + 46))
            put(main, tl(letter, 'MontBlack', 44, lcol), lp[0], lp[1], a)

def pointer(main, a, tip, lx, ly, txt, col=ICE, size=29):
    if a <= 0: return
    d = ImageDraw.Draw(main)
    lab = tl(txt, 'MontXB', size, col, track=1)
    ey = ly + (lab.height / 2 - 16 if tip[1] > ly else -lab.height / 2 + 16)
    d.line([tip, (lx, ey)], fill=ICE + (int(200 * a),), width=3)
    d.ellipse((tip[0] - 6, tip[1] - 6, tip[0] + 6, tip[1] + 6), fill=ICE + (int(255 * a),))
    put(main, lab, lx, ly, a)

def hbar(main, glow, a, y, name, v, col, p, x0=110, x1=760):
    d = ImageDraw.Draw(main); g = ImageDraw.Draw(glow)
    lab = tl(name, 'MontBlack', 38, WHITE); put(main, lab, x0 + lab.width / 2 - 20, y - 50, a)
    d.rounded_rectangle((x0, y, x1, y + 40), 20, fill=(60, 90, 125, int(200 * a)))
    xe = x0 + (x1 - x0) * v / 100 * p
    if p > 0:
        d.rounded_rectangle((x0, y - 4, max(x0 + 40, xe), y + 44), 24, fill=col + (int(255 * a),))
        g.rounded_rectangle((x0, y - 4, max(x0 + 40, xe), y + 44), 24, fill=col + (int(70 * a),))
    vt = tl(f'{v * p:.1f}%', 'MontBlack', 50, col); put(main, vt, 868, y + 20, a * clamp(p * 2))

# ------------------------------------------------------------------ scenes
SLIP = 30.0          # clear slip: classic Klein's line misses the epiphysis
MILD = 11.0          # mild slip: the line still cuts it, only the width is smaller

def s1(t, main, glow, fx):                                     # HOOK / cover
    a = win(t, 0, 3.4, 0.0, 0.4)
    if a <= 0: return
    put(main, pill("DON'T MISS  #6", 'MontXB', 34, NAVY, SKY + (255,), h=70), XC, 420, a * eo(prog(t, 0.0, 0.3)))
    q = eo(prog(t, 0.0, 0.35))
    put(main, tl('13 y, limping with knee pain,', 'MontXB', 46, WHITE), XC, 520 + 20 * (1 - q), a * q)
    put(main, tl('knee X-ray normal:', 'MontXB', 46, WHITE), XC, 582 + 20 * (1 - q), a * q)
    p = R.eback(prog(t, 0.05, 0.5))
    k1 = tl('X-RAY THE HIP?', 'MontBlack', 92, SKY)
    put(glow, k1, XC, 725, a * 0.3 * clamp(p), clamp(p, 0, 1.2)); put(main, k1, XC, 725, a * clamp(p), clamp(p, 0, 1.2))
    fx['flash'] += 0.2 * math.exp(-((t - 0.45) / 0.08) ** 2)
    q = eo(prog(t, 0.45, 0.9))
    put(main, pill('SLIPPED CAPITAL FEMORAL EPIPHYSIS', 'MontBlack', 28, WHITE, WARN + (255,), h=66), XC, 862 + 20 * (1 - q), a * q)
    q = eo(prog(t, 0.1, 0.6))
    L, Rh = Hip(305, 1110 + 20 * (1 - q), 0.64, True), Hip(715, 1110 + 20 * (1 - q), 0.64, False)
    sl = SLIP * eio(prog(t, 0.5, 1.3))
    L.draw(main, a * q); Rh.draw(main, a * q, sl)
    kp = eo(prog(t, 0.3, 1.0))
    L.klein(main, glow, a * q, kp, GOOD, 5); Rh.klein(main, glow, a * q, kp, WARN, 5)
    L.cap(main, glow, a * q * kp, 0, GOOD)
    pul = 0.6 + 0.4 * math.sin(t * 6)
    if sl > 20:
        hx, hy = Rh.S((HC[0] - N[0] * sl, HC[1] - N[1] * sl))
        ImageDraw.Draw(glow).ellipse((hx - 60, hy - 60, hx + 60, hy + 60), fill=WARN + (int(60 * pul * a),))

def s2(t, main, glow, fx):                                     # centrepiece: Klein's line, normal vs slip
    a = win(t, 3.3, 15.4, 0.35, 0.45)
    if a <= 0: return
    q = eo(prog(t, 3.4, 3.9))
    put(main, tl("KLEIN'S LINE", 'MontBlack', 66, WHITE, track=2), XC, 258, a * q, 1, 8 * (1 - q))
    label(main, 'AP pelvis: along the superior femoral neck', XC, 325, a * eo(prog(t, 3.6, 4.1)), SKY, 31, 'MontXB', 0)
    build = prog(t, 3.6, 5.0)
    L, Rh = Hip(300, 760, 0.95, True), Hip(735, 760, 0.95, False)
    sl = SLIP * eio(prog(t, 8.6, 9.6))
    L.draw(main, a, 0, build); Rh.draw(main, a, sl, build)
    # anatomy labels on the normal hip
    la = a * win(t, 5.0, 6.8, 0.4, 0.3)
    pointer(main, la, L.S(ax(-30, 10)), 290, 470, 'EPIPHYSIS', ICE, 29)
    pointer(main, la, L.S(ax(P0, -30)), 530, 840, 'PHYSIS', SKY, 29)
    pointer(main, la, L.S(ax(75, -25)), 530, 960, 'NECK', ICE, 29)
    # normal: the line cuts a cap of the epiphysis
    kp = eo(prog(t, 5.6, 6.4))
    L.klein(main, glow, a, kp, GOOD)
    L.cap(main, glow, a * eo(prog(t, 6.2, 6.6)), 0, GOOD)
    # slip: epiphysis slides, the line misses it
    kp2 = eo(prog(t, 9.7, 10.5))
    Rh.klein(main, glow, a, kp2, WARN)
    if t > 8.6 and t < 9.8:
        hx, hy = Rh.S((HC[0] - N[0] * sl, HC[1] - N[1] * sl))
        ImageDraw.Draw(glow).ellipse((hx - 70, hy - 70, hx + 70, hy + 70), fill=WARN + (int(70 * a * math.sin(math.pi * prog(t, 8.6, 9.8))),))
    qa = a * eo(prog(t, 6.4, 6.8))
    label(main, 'NORMAL', 300, 1180, qa, GOOD, 40, 'MontBlack', 2)
    label(main, 'line cuts the epiphysis', 300, 1232, qa, SUB, 28, 'MontXB', 0)
    qb = a * eo(prog(t, 10.3, 10.7))
    label(main, 'SCFE', 735, 1180, a * eo(prog(t, 8.5, 8.9)), WARN, 40, 'MontBlack', 2)
    label(main, 'line misses it', 735, 1232, qb, SUB, 28, 'MontXB', 0)
    cq = a * eo(prog(t, 12.6, 13.1))
    if cq > 0:
        put(main, card(880, 230, fill=(66, 32, 38, 200), outline=WARN + (220,)), XC, 1440, cq)
        label(main, 'Epiphysis slips posteriorly', XC, 1385, cq, WHITE, 40, 'MontBlack', 0)
        label(main, 'on the AP it drops medially', XC, 1440, cq, SUB, 32, 'MontXB', 0)
        label(main, 'Klein: no intersection = slip', XC, 1497, cq, WARN, 34, 'MontBlack', 0)

def s3(t, main, glow, fx):                                     # the classic line misses most slips; modified Klein
    a = win(t, 15.2, 25.4, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 15.3, 15.8))
    put(main, tl('MILD SLIP?', 'MontBlack', 66, WHITE, track=2), XC, 258, a * q, 1, 8 * (1 - q))
    label(main, 'the line still cuts it', XC, 325, a * eo(prog(t, 15.5, 16.0)), SKY, 32, 'MontXB', 0)
    L, Rh = Hip(290, 660, 0.9, True), Hip(745, 660, 0.9, False)
    sl = MILD * eio(prog(t, 15.8, 16.6))
    L.draw(main, a); Rh.draw(main, a, sl)
    L.klein(main, glow, a, 1, GOOD, 5); Rh.klein(main, glow, a, 1, SKY, 5)
    L.cap(main, glow, a, 0, GOOD); Rh.cap(main, glow, a, sl, SKY)
    wq = a * eo(prog(t, 17.0, 17.5))
    L.width_bar(main, wq, 0, WHITE, 'A', GOOD); Rh.width_bar(main, wq, sl, WHITE, 'B', SKY)
    label(main, 'A: normal hip', 290, 968, wq, GOOD, 30, 'MontBlack', 0)
    label(main, 'B: mild slip', 745, 968, wq, SKY, 30, 'MontBlack', 0)
    mq = a * eo(prog(t, 17.8, 18.3))
    if mq > 0:
        put(main, card(860, 120, fill=(24, 60, 60, 200), outline=GOOD + (220,), r=30), XC, 1080, mq)
        label(main, 'A minus B: a 2 mm difference = slip', XC, 1080, mq, WHITE, 38, 'MontBlack', 0)
    label(main, 'Sensitivity for SCFE on the AP', XC, 1190, a * eo(prog(t, 19.8, 20.2)), SKY, 34, 'MontXB', 0)
    q1 = a * eo(prog(t, 20.0, 20.4)); hbar(main, glow, q1, 1310, 'Classic Klein', 40.3, WARN, eio(prog(t, 20.2, 21.0)))
    q2 = a * eo(prog(t, 21.2, 21.6)); hbar(main, glow, q2, 1460, 'Modified Klein, 2 mm', 79.0, GOOD, eio(prog(t, 21.4, 22.2)))
    label(main, 'Green et al, J Pediatr Orthop 2009', XC, 1590, a * eo(prog(t, 22.4, 22.8)), SUB, 28, 'MontXB', 0)
    label(main, 'classic definition missed 60% of slips', XC, 1645, a * eo(prog(t, 22.6, 23.0)), WARN, 30, 'MontXB', 0)

def s4(t, main, glow, fx):                                     # add the lateral view: S-sign + Klein
    a = win(t, 25.2, 33.4, 0.4, 0.45)
    if a <= 0: return
    q = eo(prog(t, 25.3, 25.8))
    put(main, tl('ADD THE LATERAL', 'MontBlack', 66, WHITE, track=2), XC, 258, a * q, 1, 8 * (1 - q))
    label(main, 'frog-leg lateral: the S-sign', XC, 325, a * eo(prog(t, 25.5, 26.0)), SKY, 32, 'MontXB', 0)
    label(main, 'Sensitivity', XC, 430, a * eo(prog(t, 25.8, 26.2)), SKY, 36, 'MontBlack', 2)
    rows = [("Klein's line (AP)", 68.3, WARN), ('S-sign (lateral)', 89.0, SKY), ('Both together', 96.5, GOOD)]
    for i, (nm, v, col) in enumerate(rows):
        t0 = 26.2 + 0.9 * i
        qq = a * eo(prog(t, t0, t0 + 0.4))
        if qq > 0: hbar(main, glow, qq, 590 + 170 * i, nm, v, col, eio(prog(t, t0 + 0.1, t0 + 0.9)))
    sq = a * eo(prog(t, 29.0, 29.4))
    if sq > 0:
        put(main, card(860, 200, fill=(24, 44, 70, 170), outline=SKY + (120,), r=30), XC, 1180, sq)
        label(main, 'Specificity', XC, 1140, sq, SKY, 34, 'MontBlack', 2)
        label(main, 'Klein 89.0%  ·  S-sign 95.2%  ·  both 85.0%', XC, 1215, sq, WHITE, 31, 'MontXB', 0)
    label(main, 'Rebich et al, J Emerg Med 2018', XC, 1360, a * eo(prog(t, 29.6, 30.0)), SUB, 28, 'MontXB', 0)
    label(main, '35 SCFE patients  ·  20 observers', XC, 1410, a * eo(prog(t, 29.7, 30.1)), SUB, 28, 'MontXB', 0)
    cq = a * eo(prog(t, 30.6, 31.0))
    if cq > 0:
        p = R.eback(prog(t, 30.6, 31.1))
        big = tl('2 VIEWS, BOTH HIPS', 'MontBlack', 62, GOOD)
        put(glow, big, XC, 1560, cq * 0.3, clamp(p, 0, 1.12)); put(main, big, XC, 1560, cq, clamp(p, 0, 1.12))

TIPS = [('Knee pain in a teen?', "DOCTORS' DELAY 14 VS 4 WEEKS", 'Örtegren et al, JPO B 2021'),
        ("Can't walk even on crutches?", 'UNSTABLE: AVN 47% VS 0% STABLE', 'Loder et al, JBJS Am 1993'),
        ('Check the other hip', 'MISSED IN 8.9% OF PATIENTS (5/56)', 'Shnaekel et al, Orthopedics 2020')]
def s5(t, main, glow, fx):
    a = win(t, 33.2, 41.2, 0.4, 0.4)
    if a <= 0: return
    label(main, 'TIPS  &  TRICKS', XC, 300, a * eo(prog(t, 33.3, 33.8)), SKY, 56, 'MontBlack', 6)
    for i, (h_, sub, ref) in enumerate(TIPS):
        t0 = 33.9 + 1.6 * i; q = eo(prog(t, t0, t0 + 0.5))
        if q <= 0: continue
        y = 560 + i * 330 + 30 * (1 - q)
        put(main, card(860, 280), XC, y, a * q)
        nm = tl(f'0{i + 1}', 'MontBlack', 64, SKY); put(main, nm, 905 - nm.width / 2 + 20, y - 70, a * q)
        lab = tl(h_, 'MontBlack', 40, WHITE); put(main, lab, 115 + lab.width / 2 - 20, y - 52, a * q)
        lab = tl(sub, 'MontXB', 29, SUB, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 22, a * q)
        lab = tl(ref, 'MontXB', 29, SKY, track=1); put(main, lab, 115 + lab.width / 2 - 20, y + 82, a * q)
    fq = a * eo(prog(t, 39.0, 39.4))
    label(main, 'Longer delay, bigger slip (R 0.59)', XC, 1590, fq, WARN, 34, 'MontBlack', 0)

REFS = ['Green et al  ·  J Pediatr Orthop 2009', 'Rebich et al  ·  J Emerg Med 2018', 'Loder et al  ·  JBJS Am 1993',
        'Örtegren et al  ·  J Pediatr Orthop B 2021', 'Shnaekel et al  ·  Orthopedics 2020']
def s6(t, main, glow, fx):
    a = win(t, 41.0, 46.0, 0.4, 0.45)
    if a <= 0: return
    label(main, 'REMEMBER', XC, 270, a * eo(prog(t, 41.1, 41.5)), SKY, 48, 'MontBlack', 14)
    p = R.eback(prog(t, 41.2, 41.7))
    big = tl('IMAGE THE HIP', 'MontBlack', 106, WHITE)
    put(glow, big, XC, 400, a * 0.35 * clamp(p), clamp(p, 0, 1.2)); put(main, big, XC, 400, a * clamp(p), clamp(p, 0, 1.2))
    label(main, 'Knee pain in a teen: think SCFE', XC, 530, a * eo(prog(t, 41.4, 41.8)), SKY, 36, 'MontXB', 0)
    label(main, 'AP + lateral of both hips', XC, 590, a * eo(prog(t, 41.6, 42.0)), GOOD, 36, 'MontXB', 0)
    label(main, 'Measure Klein on both sides', XC, 650, a * eo(prog(t, 41.8, 42.2)), SUB, 32, 'MontXB', 0)
    q = eo(prog(t, 42.2, 42.6))
    put(main, card(880, 410, fill=(24, 44, 70, 160), outline=SKY + (80,)), XC, 960, a * q)
    for i, r in enumerate(REFS):
        label(main, r, XC, 820 + i * 70, a * q, ICE if i % 2 == 0 else SUB, 28, 'MontXB', 0)
    label(main, 'Save it  ·  Share it with a colleague', XC, 1260, a * eo(prog(t, 42.8, 43.2)), WHITE, 38, 'MontXB', 1)

SCENES = [s1, s2, s3, s4, s5, s6]
if __name__ == '__main__':
    main_cli(TL, SCENES, hook_end=2.8, resume=3.4)
