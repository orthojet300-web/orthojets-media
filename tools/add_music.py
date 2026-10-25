#!/usr/bin/env python3
"""Lay a background music track under an episode body (before the intro/outro are added).

usage: python3 tools/add_music.py BODY.mp4 TRACK.mp3 OUT.mp4 [--lufs -26]

- Measures the track's loudness and sets it to a quiet background level (default -26 LUFS;
  the owner asked for calm, unobtrusive music — the channel's intro/outro sound stays louder).
- Loops the track if it is shorter than the body, fades in 1.5 s and out 2.5 s.
- Replaces any existing audio in BODY (the body is rendered silent by design).
"""
import json, re, subprocess, sys

def dur(p):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                                capture_output=True, text=True, check=True).stdout.strip())

def lufs(p):
    err = subprocess.run(['ffmpeg', '-i', p, '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True).stderr
    vals = re.findall(r'^\s+I:\s+(-?[\d.]+) LUFS', err, re.M)
    return float(vals[-1])

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    target = -32.0   # owner 2026-10-25: music was distracting; was -26
    if '--lufs' in sys.argv: target = float(sys.argv[sys.argv.index('--lufs') + 1])
    body, track, out = args[:3]
    d = dur(body); gain = target - lufs(track)
    af = (f'volume={gain:.2f}dB,atrim=0:{d:.3f},asetpts=PTS-STARTPTS,'
          f'afade=t=in:d=1.5,afade=t=out:st={max(0, d - 2.5):.3f}:d=2.5,'
          'aresample=44100,aformat=channel_layouts=stereo')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', body, '-stream_loop', '-1', '-i', track,
                    '-map', '0:v', '-map', '1:a', '-af', af, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
                    '-t', f'{d:.3f}', out], check=True)
    print(out, f'(track gain {gain:+.1f} dB -> {target} LUFS)')

if __name__ == '__main__':
    main()
