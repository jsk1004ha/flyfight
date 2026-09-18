"""Render captured real backend messages and test viewer contracts/UI.

Local navigation is restricted in this CI; set_content is intentional, not an
attempt to bypass that policy. Network transport is tested separately with an
actual aiohttp client by transport_check.py. No JS/WebGL GPU-speed claim.
"""
from pathlib import Path
import copy,json,os
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def ok(name,test):
    assert test,name
    checks.append({'name':name,'passed':True})

def main():
    frames=json.loads((ROOT/'evidence/transport_frames.json').read_text(encoding='utf-8'))
    html=(ROOT/'viewer/FlyFight_Viewer.html').read_text(encoding='utf-8')
    with sync_playwright() as p:
        browser_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium')
        browser_args=['--no-sandbox','--disable-dev-shm-usage']
        if 'CHROMIUM_PATH' not in os.environ:
            browser_args += ['--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader']
        b=p.chromium.launch(executable_path=browser_path,headless=True,args=browser_args)
        page=b.new_page(viewport={'width':1600,'height':1050},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.set_content(html,wait_until='load');page.wait_for_function('!!window.FlyFight',timeout=10000)
        page.wait_for_timeout(3500);fps_sample=page.evaluate('window.FlyFight.getState().fps')
        ok('render loop reports live FPS',fps_sample>0)
        page.evaluate('window.FlyFight.setPaused(true)')
        state=page.evaluate('window.FlyFight.getState()')
        ok('standalone loads expanded 64x48 map with 16 obstacles',state['map']=={'width':64,'depth':48,'obstacles':16})
        ok('standalone is clearly scripted not native',not state['native'] and state['mode']=='showcase')
        for side in ['A', 'B']:
            index = 0 if side == 'A' else 1
            before_zoom = page.evaluate('window.FlyFight.getState().neuralView.zoom')[index]
            zoom = page.locator('#neuralZoomIn'+side)
            zoom.dispatch_event('click')
            ok('neural zoom in '+side, page.evaluate('window.FlyFight.getState().neuralView.zoom')[index] > before_zoom)
            page.locator('#neuralZoomOut'+side).dispatch_event('click')
            ok('neural zoom reset '+side, abs(page.evaluate('window.FlyFight.getState().neuralView.zoom')[index]-before_zoom)<1e-6)
            edges = page.locator('#neuralEdges'+side)
            initial_edges = edges.inner_text()
            edges.dispatch_event('click')
            ok('neural edges control '+side, edges.inner_text() != initial_edges)
            edges.dispatch_event('click');edges.dispatch_event('click')
            size = page.locator('#neuralSize'+side)
            initial_size = size.inner_text()
            size.dispatch_event('click')
            ok('neural point size control '+side, size.inner_text() != initial_size)
            size.dispatch_event('click');size.dispatch_event('click')
        rect = page.locator('#brainA').bounding_box()
        selected = None
        for fx,fy in [(0.5,0.45),(0.5,0.40),(0.4,0.4),(0.6,0.4),(0.5,0.5)]:
            page.mouse.click(rect['x']+rect['width']*fx,rect['y']+rect['height']*fy)
            selected = page.evaluate('window.FlyFight.getState().neuralView.selected[0]')
            if selected is not None:
                break
        ok('click pins a real neuron', selected is not None and selected['id'].startswith('syn-'))
        ok('pinned neuron preserves signed activity', abs(abs(selected['signed'])-selected['magnitude'])<1e-6)
        ok('probe shows selected neuron identity', selected['id'] in page.locator('#neuralProbeA').inner_text())
        ok('text state exposes coordinate system and speed',json.loads(page.evaluate('window.render_game_to_text()'))['speed']==1)
        before=state['t'];page.evaluate('window.advanceTime(100)')
        ok('deterministic time hook advances standalone simulation',page.evaluate('window.FlyFight.getState().t')>before)
        page.locator('#speedModeBtn').dispatch_event('click')
        ok('speed mode button enables 4x',page.evaluate('window.FlyFight.getState().speed')==4 and page.locator('#speedModeBtn').evaluate('e=>e.classList.contains("active")'))
        page.keyboard.press('b');ok('B shortcut returns to 1x',page.evaluate('window.FlyFight.getState().speed')==1)
        page.locator('#speed').select_option('8');ok('8x turbo option works',page.evaluate('window.FlyFight.getState().speed')==8)
        page.evaluate('window.FlyFight.setSpeed(1)')
        for cam in ['duel','orbit','followA','followB','top']:
            page.locator(f'[data-camera="{cam}"]').dispatch_event('click')
            ok('camera '+cam,page.locator(f'[data-camera="{cam}"]').evaluate('e=>e.classList.contains("active")'))
        page.locator('[data-camera="orbit"]').dispatch_event('click');page.wait_for_timeout(250)
        page.screenshot(path=str(ROOT/'evidence/large_map_standalone.png'),full_page=True)
        page.evaluate('(messages)=>{for(let m of messages)window.FlyFight.ingest(m);window.FlyFight.setPaused(true)}',frames)
        page.wait_for_timeout(200)
        state=page.evaluate('window.FlyFight.getState()')
        ok('native frame activates Python mode',state['native'] and state['external'])
        ok('native map agrees with backend',state['map']['width']==frames[0]['map']['width'])
        last=next(m for m in reversed(frames) if m['type']=='frame')
        ok('positions come from actual backend',state['agents'][0]['position']==last['agents'][0]['position'])
        ok('activity is exact streamed model state',abs(state['agents'][0]['mean']-sum(last['agents'][0]['activity'])/len(last['agents'][0]['activity']))<1e-6)
        if 'signed_activity' in last['agents'][0]:
            signed = page.evaluate('window.FlyFight.getNeuralSample(0).signedActivity')
            ok('signed neural state matches backend', all(abs(x-y)<1e-6 for x,y in zip(signed,last['agents'][0]['signed_activity'][:10])))
        ok('training metrics are distinct from spectator score',state['training']['update']==last['training']['update'])
        ok('decoded actual RGB pixels displayed',page.locator('#pixelsA').evaluate('e=>e.width')==last['agents'][0]['observation']['width'])
        page.evaluate('window.FlyFight.setPaused(false)')
        previous_headshots=page.evaluate('window.FlyFight.getState().agents[0].headshots')
        event_frame=copy.deepcopy(last);event_frame['t']+=1;event_frame['events']=[{'type':'shot','agent':'A','start':[0,1,0],'end':[1,2,3],'hit':True,'headshot':True,'damage':100}]
        page.evaluate('(f)=>window.FlyFight.ingest(f)',event_frame)
        ok('headshot telemetry updates viewer state',page.evaluate('window.FlyFight.getState().agents[0].headshots')==previous_headshots+1)
        page.locator('#pixelToggle').dispatch_event('click')
        ok('pretty render explicitly not input',page.locator('#pixelMetaA').inner_text()=='상세 3D 재구성 · 학습 입력과 다름')
        page.locator('#pixelToggle').dispatch_event('click')
        page.locator('[data-camera="duel"]').dispatch_event('click');page.wait_for_timeout(250)
        page.screenshot(path=str(ROOT/'evidence/native_live_cpu.png'),full_page=True)
        page.locator('[data-camera="orbit"]').dispatch_event('click');page.wait_for_timeout(250)
        page.screenshot(path=str(ROOT/'evidence/large_map_cpu.png'),full_page=True)
        page.evaluate('window.FlyFight.setPaused(false)')
        invalid=copy.deepcopy(event_frame);invalid['t']-=2
        ok('reject backwards stream time',page.evaluate('(f)=>{try{window.FlyFight.ingest(f);return false}catch(e){return true}}',invalid))
        invalid=copy.deepcopy(event_frame);invalid['t']+=1;invalid['agents'][0]['observation']['rgb_base64']='AA=='
        ok('reject malformed RGB before mutating state',page.evaluate('(f)=>{try{window.FlyFight.ingest(f);return false}catch(e){return true}}',invalid) and page.evaluate('window.FlyFight.getState().t')==event_frame['t'])
        page.evaluate('window.FlyFight.setPaused(true)')
        page.locator('#infoBtn').dispatch_event('click');ok('help opens',page.locator('#info').evaluate('e=>e.open'))
        page.locator('#closeInfo').dispatch_event('click');ok('help closes',not page.locator('#info').evaluate('e=>e.open'))
        page.set_viewport_size({'width':1440,'height':900});page.wait_for_timeout(250)
        box=page.evaluate("({footer:document.querySelector('.footer').getBoundingClientRect().top,readouts:[...document.querySelectorAll('.readouts')].map(e=>e.getBoundingClientRect().bottom),overflow:document.documentElement.scrollWidth>innerWidth})")
        ok('desktop no horizontal overflow',not box['overflow'])
        ok('desktop panels above footer',max(box['readouts'])<=box['footer']+1)
        page.set_viewport_size({'width':430,'height':932});page.wait_for_timeout(250)
        ok('mobile no horizontal overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        page.screenshot(path=str(ROOT/'evidence/mobile.png'),full_page=True)
        ok('no JavaScript errors',not errors)
        b.close()
    report={'passed':len(checks),'checks':checks,'render_fps_sample':fps_sample,'browser':'Chromium / WebGL2 '+('system Chrome' if 'CHROMIUM_PATH' in os.environ else 'SwiftShader'),
            'load_method':'complete bundled HTML via page.set_content; actual backend frames injected through documented ingest API',
            'limitations':['Direct browser localhost navigation is administratively restricted in this runtime.','WebSocket transport was tested separately by aiohttp, not by a real browser socket.','CPU only; no RTX 5090 speedup or skill-improvement verification.']}
    (ROOT/'evidence/browser_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
