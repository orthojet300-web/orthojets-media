#!/usr/bin/env python3
"""Orthojets: wrap an episode body with the channel branding.

usage:
  python3 add_brand.py BODY.mp4 OUT.mp4 --hook-first SECONDS [--light]   (current format, from EP03)
  python3 add_brand.py BODY.mp4 OUT.mp4 [--no-intro] [--no-outro] [--light]   (legacy: 3 s intro + 6 s outro)

Current format (retention-first):
  [hook: body 0..SECONDS] -> [1 s logo flash] -> [rest of body] -> [3 s outro]
  The episode opens directly on its hook; branding never occupies the first seconds.

- Any input size/orientation: non-9:16 is fitted over a blurred, darkened copy of itself.
- Inputs without sound get a silent track so the joins work.
- Cross-fades: 0.18 s around the logo flash, 0.35 s into the outro.
- --light also writes a 720p preview copy.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTRO, OUTRO = os.path.join(HERE, 'Orthojets_intro.mp4'), os.path.join(HERE, 'Orthojets_outro.mp4')
STING, OUTRO3 = os.path.join(HERE, 'Orthojets_sting.mp4'), os.path.join(HERE, 'Orthojets_outro3.mp4')
W, H, FPS = 1080, 1920, 30

def probe(p):
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                                   '-of', 'json', p], capture_output=True, text=True, check=True).stdout)
    return float(j['format']['duration']), any(s['codec_type'] == 'audio' for s in j['streams'])

def main():
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith('--')}
    hook = None
    if '--hook-first' in argv:
        hook = float(argv[argv.index('--hook-first') + 1])
    pos = [a for i, a in enumerate(argv) if not a.startswith('--') and not (i > 0 and argv[i - 1] == '--hook-first')]
    if not pos: sys.exit(__doc__)
    src = pos[0]
    dst = pos[1] if len(pos) > 1 else os.path.splitext(os.path.basename(src))[0] + '_orthojets.mp4'
    bdur, _ = probe(src)

    # each segment: (path, start, duration or None, crossfade INTO this segment)
    if hook is not None:
        segs = [(src, 0.0, hook, 0), (STING, 0.0, None, 0.18), (src, hook, bdur - hook, 0.18), (OUTRO3, 0.0, None, 0.35)]
    else:
        segs = ([(INTRO, 0.0, None, 0)] if '--no-intro' not in flags else []) + [(src, 0.0, None, 0.35 if '--no-intro' not in flags else 0)] \
               + ([(OUTRO, 0.0, None, 0.35)] if '--no-outro' not in flags else [])
        segs[0] = (segs[0][0], segs[0][1], segs[0][2], 0)

    cmd = ['ffmpeg', '-y', '-loglevel', 'error']
    fil, durs = [], []
    for i, (p, ss, du, _) in enumerate(segs):
        full, has_a = probe(p)
        d = du if du is not None else full - ss
        durs.append(d)
        cmd += ['-ss', f'{ss:.3f}', '-t', f'{d:.3f}', '-i', p]
        fil.append(f'[{i}:v]fps={FPS},split[a{i}][b{i}];'
                   f'[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2,eq=brightness=-0.18[bg{i}];'
                   f'[b{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{i}];'
                   f'[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{i}]')
        if has_a:
            fil.append(f'[{i}:a]aresample=44100,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[s{i}]')
        else:
            fil.append(f'anullsrc=r=44100:cl=stereo,atrim=0:{d:.3f}[s{i}]')
    v, a, t = 'v0', 's0', durs[0]
    for i in range(1, len(segs)):
        xf = segs[i][3]
        fil.append(f'[{v}][v{i}]xfade=transition=fade:duration={xf}:offset={t - xf:.3f}[vx{i}]')
        fil.append(f'[{a}][s{i}]acrossfade=d={xf}[ax{i}]')
        v, a, t = f'vx{i}', f'ax{i}', t + durs[i] - xf
    cmd += ['-filter_complex', ';'.join(fil), '-map', f'[{v}]', '-map', f'[{a}]',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst]
    subprocess.run(cmd, check=True)
    print(f'{dst}  ({t:.1f}s)')
    if '--light' in flags:
        lt = os.path.splitext(dst)[0] + '_light.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', dst, '-vf', 'scale=720:1280:flags=lanczos',
                        '-c:v', 'libx264', '-profile:v', 'main', '-preset', 'slow', '-crf', '25', '-maxrate', '2.5M',
                        '-bufsize', '5M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', lt], check=True)
        print(lt)

if __name__ == '__main__':
    main()
