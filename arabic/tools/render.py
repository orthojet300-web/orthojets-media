#!/usr/bin/env python3
"""Render EPDIR/anim.html (render(t) per frame, 30 fps, 1080x1920) with Playwright -> EPDIR/video_noaudio.mp4.
usage: python3 render.py EPDIR   (about 8-10 min)"""
import asyncio, json, os, subprocess, sys
from playwright.async_api import async_playwright
ep = os.path.abspath(sys.argv[1])
TL = json.load(open(os.path.join(ep, 'timeline.json'))); FPS = 30; N = int(TL['total'] * FPS)
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'image2pipe', '-framerate', str(FPS), '-c:v', 'mjpeg', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p',
                       os.path.join(ep, 'video_noaudio.mp4')], stdin=subprocess.PIPE)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1080, 'height': 1920})
        await pg.goto('file://' + os.path.join(ep, 'anim.html'))
        await pg.evaluate('x=>setTL(x)', TL)
        await pg.evaluate("Promise.all([document.fonts.load('900 50px Cairo'),document.fonts.load('700 50px Mont'),"
                          "document.fonts.load('900 50px Mont'),document.fonts.load('600 50px Mont'),document.fonts.load('800 50px Mont')])")
        for i in range(N):
            await pg.evaluate(f'render({i / FPS})')
            ff.stdin.write(await pg.screenshot(type='jpeg', quality=97))
            if i % 300 == 0: print(i, N, flush=True)
        await b.close()
asyncio.run(main()); ff.stdin.close(); ff.wait(); print('done')
