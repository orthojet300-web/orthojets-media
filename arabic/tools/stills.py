#!/usr/bin/env python3
"""usage: python3 stills.py EPDIR t1 t2 ...  -> EPDIR/st/<t>.png (check frames before the full render)"""
import asyncio, json, os, sys
from playwright.async_api import async_playwright
ep = os.path.abspath(sys.argv[1]); TL = json.load(open(os.path.join(ep, 'timeline.json')))
times = [float(x) for x in sys.argv[2:]]; os.makedirs(os.path.join(ep, 'st'), exist_ok=True)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width': 1080, 'height': 1920})
        await pg.goto('file://' + os.path.join(ep, 'anim.html')); await pg.evaluate('x=>setTL(x)', TL)
        await pg.evaluate("Promise.all([document.fonts.load('900 50px Cairo'),document.fonts.load('700 50px Mont'),document.fonts.load('900 50px Mont')])")
        for t in times:
            await pg.evaluate(f'render({t})'); await pg.screenshot(path=os.path.join(ep, 'st', f'{t:06.2f}.png'))
        await b.close()
asyncio.run(main())
