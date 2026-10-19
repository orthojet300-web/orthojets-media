#!/usr/bin/env python3
"""Orthojets: wrap an episode body with the channel branding.

usage:
  python3 add_brand.py BODY.mp4 OUT.mp4 --hook-first SECONDS [--resume SECONDS] [--light]   (current format, from EP03)
  python3 add_brand.py BODY.mp4 OUT.mp4 [--no-intro] [--no-outro] [--light]   (legacy: 3 s intro + 6 s outro)

Current format (retention-first):
  video: [hook: body 0..HOOK] -> soft 1 s logo (Orthojets_sting.mp4) -> [body RESUME..end] -> [3 s outro]
         RESUME (default = HOOK) is where the next scene starts, so the hook's own fade-out is skipped and the
         hook does not reappear after the logo.
  audio: the body's soundtrack (background music) plays CONTINUOUSLY under hook, logo and content (no cut at
         the logo); the logo's soft chime is mixed on top; then a cross-fade into the outro's own sound.
  Dissolves of 0.3 s into and out of the logo, 0.35 s into the outro.

- Any input size/orientation: non-9:16 is fitted over a blurred, darkened copy of itself.
- Inputs without sound get a silent track.
- --light also writes a 720p preview copy.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTRO, OUTRO = os.path.join(HERE, 'Orthojets_intro.mp4'), os.path.join(HERE, 'Orthojets_outro.mp4')
STING, OUTRO3 = os.path.join(HERE, 'Orthojets_sting.mp4'), os.path.join(HERE, 'Orthojets_outro3.mp4')
W, H, FPS = 1080, 1920, 30
XS, XO = 0.3, 0.35

def probe(p):
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                                   '-of', 'json', p], capture_output=True, text=True, check=True).stdout)
    return float(j['format']['duration']), any(s['codec_type'] == 'audio' for s in j['streams'])

def vprep(i):
    return (f'[{i}:v]fps={FPS},split[a{i}][b{i}];'
            f'[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2,eq=brightness=-0.18[bg{i}];'
            f'[b{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{i}];'
            f'[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{i}]')

def aprep(i, has_a, d, label):
    if has_a:
        return f'[{i}:a]aresample=44100,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,apad,atrim=0:{d:.3f}[{label}]'
    return f'anullsrc=r=44100:cl=stereo,atrim=0:{d:.3f}[{label}]'

def encode(cmd, fil, vout, aout, dst):
    cmd += ['-filter_complex', ';'.join(fil), '-map', f'[{vout}]', '-map', f'[{aout}]',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst]
    subprocess.run(cmd, check=True)

def hook_first(src, dst, hook, resume=None):
    resume = hook if resume is None else resume
    bdur, b_has_a = probe(src)
    sdur, s_has_a = probe(STING)
    odur, o_has_a = probe(OUTRO3)
    rest = bdur - resume
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-t', f'{hook:.3f}', '-i', src,                 # 0 hook (video)
           '-i', STING,                                    # 1 logo
           '-ss', f'{resume:.3f}', '-i', src,              # 2 rest of body (video)
           '-i', OUTRO3,                                   # 3 outro
           '-i', src]                                      # 4 body soundtrack (continuous)
    fil = [vprep(i) for i in range(4)]
    t1 = hook - XS
    fil.append(f'[v0][v1]xfade=transition=fade:duration={XS}:offset={t1:.3f}[x1]')
    t2 = t1 + sdur - XS
    fil.append(f'[x1][v2]xfade=transition=fade:duration={XS}:offset={t2:.3f}[x2]')
    content_len = t2 + rest
    fil.append(f'[x2][v3]xfade=transition=fade:duration={XO}:offset={content_len - XO:.3f}[vout]')
    fil.append(aprep(4, b_has_a, content_len, 'bodya'))
    d_ms = int(t1 * 1000)
    if s_has_a:
        fil.append(f'[1:a]aresample=44100,aformat=channel_layouts=stereo,adelay={d_ms}|{d_ms},apad,atrim=0:{content_len:.3f}[chime]')
        fil.append('[bodya][chime]amix=inputs=2:normalize=0:duration=first[ac]')
    else:
        fil.append('[bodya]anull[ac]')
    fil.append(aprep(3, o_has_a, odur, 'oa'))
    fil.append(f'[ac][oa]acrossfade=d={XO}[aout]')
    encode(cmd, fil, 'vout', 'aout', dst)
    return content_len + odur - XO

def legacy(src, dst, flags):
    clips = ([INTRO] if '--no-intro' not in flags else []) + [src] + ([OUTRO] if '--no-outro' not in flags else [])
    cmd = ['ffmpeg', '-y', '-loglevel', 'error']; fil = []; durs = []
    for i, c in enumerate(clips):
        d, has_a = probe(c); durs.append(d); cmd += ['-i', c]
        fil.append(vprep(i)); fil.append(aprep(i, has_a, d, f's{i}'))
    v, a, t = 'v0', 's0', durs[0]
    for i in range(1, len(clips)):
        fil.append(f'[{v}][v{i}]xfade=transition=fade:duration={XO}:offset={t - XO:.3f}[vx{i}]')
        fil.append(f'[{a}][s{i}]acrossfade=d={XO}[ax{i}]')
        v, a, t = f'vx{i}', f'ax{i}', t + durs[i] - XO
    encode(cmd, fil, v, a, dst)
    return t

def main():
    argv = sys.argv[1:]
    flags = {x for x in argv if x.startswith('--')}
    hook = float(argv[argv.index('--hook-first') + 1]) if '--hook-first' in argv else None
    resume = float(argv[argv.index('--resume') + 1]) if '--resume' in argv else None
    pos = [x for i, x in enumerate(argv) if not x.startswith('--') and not (i > 0 and argv[i - 1] in ('--hook-first', '--resume'))]
    if not pos: sys.exit(__doc__)
    src = pos[0]
    dst = pos[1] if len(pos) > 1 else os.path.splitext(os.path.basename(src))[0] + '_orthojets.mp4'
    total = hook_first(src, dst, hook, resume) if hook is not None else legacy(src, dst, flags)
    print(f'{dst}  ({total:.1f}s)')
    if '--light' in flags:
        lt = os.path.splitext(dst)[0] + '_light.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', dst, '-vf', 'scale=720:1280:flags=lanczos',
                        '-c:v', 'libx264', '-profile:v', 'main', '-preset', 'slow', '-crf', '25', '-maxrate', '2.5M',
                        '-bufsize', '5M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', lt], check=True)
        print(lt)

if __name__ == '__main__':
    main()
