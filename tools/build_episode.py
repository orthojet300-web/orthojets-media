#!/usr/bin/env python3
"""Render an Orthojets episode end-to-end.

Music: if the private repo orthojets-music is cloned next to this repo (../orthojets-music) or MUSIC_DIR
is set, a background track is laid under the body (rotating by episode number, -26 LUFS, fades).
Pass --no-music to keep the body silent.

usage: python3 tools/build_episode.py episodes/epNN_slug/script.py SLUG
output: out/SLUG_1080p.mp4 (upload copy WITH music, for TikTok + Instagram),
        out/SLUG_youtube_1080p.mp4 (same video, music-free body + synthesized motion sound effects (tools/add_sfx.py), for YouTube: YouTube Content ID claimed our
        Pixabay tracks and blocked EP01/EP02 worldwide, 2026-10-20) and out/SLUG_light.mp4 (720p preview)

Body is rendered silent (channel rule); intro/outro keep their sound (add_brand.py).
Rendering runs in 2 parallel segments; each segment is a separate process so the
job survives long renders. Expect ~15-20 min for a 60 s body on 2 cores.
"""
import glob, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, **kw):
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

def main():
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    script, slug = os.path.abspath(pos[0]), pos[1]
    wd = os.path.dirname(script)
    dur = float(subprocess.run([sys.executable, script, 'dur'], cwd=wd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    nf = int(round(dur * 30)); half = nf // 2
    out = os.path.join(ROOT, 'out'); os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=slug + '_')
    a, b = os.path.join(tmp, 'a.mp4'), os.path.join(tmp, 'b.mp4')
    p1 = subprocess.Popen([sys.executable, script, 'seg', '0', str(half), a], cwd=wd)
    p2 = subprocess.Popen([sys.executable, script, 'seg', str(half), str(nf), b], cwd=wd)
    if p1.wait() or p2.wait(): sys.exit('segment render failed')
    lst = os.path.join(tmp, 'l.txt'); open(lst, 'w').write(f"file '{a}'\nfile '{b}'\n")
    body = os.path.join(tmp, 'body.mp4')
    silent_body = body
    run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-shortest', body])
    mdir = os.environ.get('MUSIC_DIR') or os.path.join(os.path.dirname(ROOT), 'orthojets-music')
    tracks = sorted(glob.glob(os.path.join(mdir, '*.mp3')))
    if tracks and '--no-music' not in sys.argv:
        m = re.match(r'ep(\d+)', slug); n = int(m.group(1)) if m else 1
        track = tracks[(n - 1) % len(tracks)]
        withm = os.path.join(tmp, 'body_music.mp4')
        run([sys.executable, os.path.join(ROOT, 'tools', 'add_music.py'), body, track, withm])
        body = withm
        print('MUSIC', os.path.basename(track))
    else:
        print('MUSIC none (no tracks found in', mdir, ')')
    full = os.path.join(tmp, 'full.mp4')
    hv = subprocess.run([sys.executable, script, 'hookend'], cwd=wd, capture_output=True, text=True,
                        check=True).stdout.strip().splitlines()[-1].split()
    hook_end = float(hv[0]); resume = float(hv[1]) if len(hv) > 1 else hook_end
    print('HOOK_END', hook_end, 'RESUME', resume, '-> cover frame ~', int((hook_end - 0.4) * 1000), 'ms')
    run([sys.executable, os.path.join(ROOT, 'kit', 'add_brand.py'), body, full, '--hook-first', f'{hook_end:.2f}',
         '--resume', f'{resume:.2f}', '--light'])
    # Best-quality upload copy: x264 slow, high bitrate (~10 Mbps, about 2x the old 4.5) so text stays crisp after
    # TikTok re-compression. The bitrate is capped from the duration so the file stays under ~90 MB
    # (GitHub's per-file limit is 100 MB).
    final = os.path.join(out, f'{slug}_1080p.mp4')
    dur_full = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', full],
                                    capture_output=True, text=True, check=True).stdout)
    kbps = int(min(10000, (90e6 * 8 / dur_full) / 1000 - 200))
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', full, '-vf', 'hqdn3d=0.8:0.6:1.2:1.2', '-c:v', 'libx264',
         '-preset', 'slow', '-tune', 'animation', '-profile:v', 'high', '-level', '4.2',
         '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.2)}k', '-bufsize', f'{kbps * 2}k',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', final])
    os.replace(os.path.join(tmp, 'full_light.mp4'), os.path.join(out, f'{slug}_light.mp4'))
    # YouTube copy: identical picture, audio = silent body + logo chime + outro sound (no library music).
    yt_full = os.path.join(tmp, 'yt_full.mp4')
    run([sys.executable, os.path.join(ROOT, 'kit', 'add_brand.py'), silent_body, yt_full, '--hook-first', f'{hook_end:.2f}',
         '--resume', f'{resume:.2f}'])
    yt = os.path.join(out, f'{slug}_youtube_1080p.mp4')
    yt_mux = os.path.join(tmp, 'yt_mux.mp4')
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', final, '-i', yt_full, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
         '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', yt_mux])
    # Original synthesized sound effects on every motion (owner approved 2026-10-20, YouTube only).
    # Skip the logo sting (it has its chime) and the outro (own sound).
    run([sys.executable, os.path.join(ROOT, 'tools', 'add_sfx.py'), yt_mux, yt,
         '--skip', f'{hook_end - 0.45:.2f}-{hook_end + 0.85:.2f}', '--skip', f'{dur_full - 3.4:.2f}-{dur_full:.2f}'])
    print('YOUTUBE', yt)
    print('DONE', os.path.join(out, f'{slug}_1080p.mp4'))

if __name__ == '__main__':
    main()
