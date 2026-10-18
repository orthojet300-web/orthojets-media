#!/usr/bin/env python3
"""Orthojets: wrap a short video with the channel intro + outro.

usage: python3 add_brand.py INPUT.mp4 [OUTPUT.mp4] [--no-intro] [--no-outro] [--light]

- Any input size/orientation: 9:16 is scaled to 1080x1920; other shapes are fitted
  on a blurred, darkened copy of themselves (no black bars).
- Inputs without sound get a silent track so the joins work.
- 0.35 s cross-fades (picture and sound) between intro -> video -> outro.
- --light also writes a 720p preview copy (small file, opens fast on phones).
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTRO, OUTRO = os.path.join(HERE, 'Orthojets_intro.mp4'), os.path.join(HERE, 'Orthojets_outro.mp4')
W, H, FPS, XF = 1080, 1920, 30, 0.35

def probe(p):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
                          '-of', 'json', p], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return float(j['format']['duration']), any(s['codec_type'] == 'audio' for s in j['streams'])

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    if not args: sys.exit(__doc__)
    src = args[0]
    dst = args[1] if len(args) > 1 else os.path.splitext(os.path.basename(src))[0] + '_orthojets.mp4'
    clips = ([INTRO] if '--no-intro' not in flags else []) + [src] + ([OUTRO] if '--no-outro' not in flags else [])

    cmd = ['ffmpeg', '-y', '-loglevel', 'error']
    durs, fil = [], []
    for i, c in enumerate(clips):
        d, has_a = probe(c); durs.append(d)
        cmd += ['-i', c]
        # picture: fit inside 1080x1920 over a blurred fill of itself
        fil.append(f'[{i}:v]fps={FPS},split[a{i}][b{i}];'
                   f'[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2,eq=brightness=-0.18[bg{i}];'
                   f'[b{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{i}];'
                   f'[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v{i}]')
        if has_a:
            fil.append(f'[{i}:a]aresample=44100,aformat=channel_layouts=stereo[s{i}]')
        else:
            fil.append(f'anullsrc=r=44100:cl=stereo,atrim=0:{d:.3f}[s{i}]')
    v, a, t = 'v0', 's0', durs[0]
    for i in range(1, len(clips)):
        off = t - XF
        fil.append(f'[{v}][v{i}]xfade=transition=fade:duration={XF}:offset={off:.3f}[vx{i}]')
        fil.append(f'[{a}][s{i}]acrossfade=d={XF}[ax{i}]')
        v, a, t = f'vx{i}', f'ax{i}', t + durs[i] - XF
    cmd += ['-filter_complex', ';'.join(fil), '-map', f'[{v}]', '-map', f'[{a}]',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-maxrate', '10M', '-bufsize', '20M',
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst]
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
