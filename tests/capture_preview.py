"""Render the included CPU-generated 768-unit telemetry, not an AI mockup.
A capture is evidence of rendering, NOT of successful learning. Network testing
is in transport_check.py; this uses the public ingest API in a browser.
"""
from pathlib import Path
import json,os
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1600,'height':1050},device_scale_factor=1)
    page.set_content((ROOT/'viewer/FlyFight_Viewer.html').read_text(),wait_until='load')
    page.wait_for_function('!!window.FlyFight')
    frames=json.loads((ROOT/'evidence/full_model_frames.json').read_text())
    page.evaluate('(ms)=>{for(const m of ms)window.FlyFight.ingest(m);window.FlyFight.setPaused(true)}',frames)
    page.locator('[data-camera="duel"]').dispatch_event('click')
    page.locator('#pixelToggle').dispatch_event('click')
    page.wait_for_timeout(300)
    page.screenshot(path=str(ROOT/'evidence/FlyFight_Native_preview.png'),full_page=True)
    browser.close()
