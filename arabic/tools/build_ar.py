#!/usr/bin/env python3
"""Build one «عظامك ببساطة» (Arabic, general-public) episode end to end.

usage: python3 arabic/tools/build_ar.py arabic/episodes/arNN_slug SLUG [--skip-tts] [--skip-render]
output: out/SLUG_1080p.mp4 (same file for TikTok, Instagram and YouTube: voice + our own synthesized music, no
        third-party audio -> no Content ID), out/SLUG_light.mp4, and prints the cover frame time.
Steps: Piper voice (gen_audio + assemble) -> music.py -> Playwright render -> mix (voice -16 LUFS, music -30 LUFS,
loudnorm each SEPARATELY, then amix) -> kit/add_brand.py --hook-first H --resume H+0.4 -> 10 Mbps x264 slow encode.
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
def run(c): print('+', ' '.join(c), flush=True); subprocess.run(c, check=True)
def dur(p): return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p], capture_output=True, text=True, check=True).stdout)

def main():
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    ep, slug = os.path.abspath(pos[0]), pos[1]
    if '--skip-tts' not in sys.argv:
        run([sys.executable, os.path.join(HERE, 'gen_audio.py'), ep]); run([sys.executable, os.path.join(HERE, 'assemble.py'), ep])
    run([sys.executable, os.path.join(HERE, 'music.py'), ep])
    if '--skip-render' not in sys.argv or not os.path.exists(os.path.join(ep, 'video_noaudio.mp4')):
        run([sys.executable, os.path.join(HERE, 'render.py'), ep])
    TL = json.load(open(os.path.join(ep, 'timeline.json'))); H = TL['hook_end']
    tmp = tempfile.mkdtemp(prefix=slug + '_')
    v, m, body = os.path.join(tmp, 'v.wav'), os.path.join(tmp, 'm.wav'), os.path.join(tmp, 'body.mp4')
    run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(ep, 'voice.wav'), '-af', 'loudnorm=I=-16:TP=-1.5', '-ar', '44100', '-ac', '2', v])
    run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(ep, 'music.wav'), '-af', 'loudnorm=I=-36:TP=-10', '-ar', '44100', '-ac', '2', m])
    run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(ep, 'video_noaudio.mp4'), '-i', v, '-i', m, '-filter_complex',
         '[1:a][2:a]amix=inputs=2:normalize=0:duration=first[a]', '-map', '0:v', '-map', '[a]', '-c:v', 'copy',
         '-c:a', 'aac', '-b:a', '192k', body])
    full = os.path.join(tmp, 'full.mp4')
    run([sys.executable, os.path.join(ROOT, 'kit', 'add_brand.py'), body, full, '--hook-first', f'{H:.2f}', '--resume', f'{H + 0.4:.2f}'])
    out = os.path.join(ROOT, 'out'); os.makedirs(out, exist_ok=True)
    final = os.path.join(out, f'{slug}_1080p.mp4'); D = dur(full)
    kbps = int(min(10000, (90e6 * 8 / D) / 1000 - 200))
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', full, '-c:v', 'libx264', '-preset', 'slow', '-tune', 'animation',
         '-profile:v', 'high', '-level', '4.2', '-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.2)}k', '-bufsize', f'{kbps * 2}k',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', final])
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', final, '-vf', 'scale=720:1280:flags=lanczos', '-c:v', 'libx264', '-preset', 'slow',
         '-crf', '25', '-maxrate', '2.5M', '-bufsize', '5M', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
         os.path.join(out, f'{slug}_light.mp4')])
    print('HOOK_END', H, 'DURATION', round(D, 2), 'kbps', kbps); print('DONE', final)

if __name__ == '__main__':
    main()
