"""Actual HTTP/WebSocket integration test; independent of browser rendering.
Runs a tiny CPU trainer in a temporary directory, tests auth, telemetry and
control separation, then gracefully stops it. No user experiment is modified.
"""
from __future__ import annotations
import asyncio,json,re,signal,socket,subprocess,sys,tempfile,time
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parents[1]

async def run_checks(base: str, run: Path) -> dict:
    checks=[]
    def ok(name,value):
        assert value,name
        checks.append({'name':name,'passed':True})
    async with aiohttp.ClientSession() as s:
        for _ in range(300):
            try:
                async with s.get(base+'/health') as r:
                    status=await r.json()
                if status.get('view_ready'):break
                if status.get('error'):raise RuntimeError(status['error'])
            except (aiohttp.ClientConnectionError,asyncio.TimeoutError):pass
            await asyncio.sleep(.1)
        else:raise TimeoutError('Server did not become ready')
        async def health():
            async with s.get(base+'/health') as r:return await r.json()
        async with s.get(base+'/') as r:
            html=await r.text();ok('bundled HTML served over actual HTTP',r.status==200)
        meta=json.loads(re.search(r'window.FLYFIGHT_NATIVE=(\{.*?\});',html).group(1))
        for name,url,headers in [
            ('reject missing token and origin',base+'/ws',{}),
            ('reject foreign origin',base+'/ws?token='+meta['token'],{'Origin':'http://untrusted.invalid'}),
            ('reject wrong token',base+'/ws?token=incorrect',{'Origin':base})]:
            try:
                ws=await s.ws_connect(url,headers=headers);await ws.close();ok(name,False)
            except aiohttp.WSServerHandshakeError as exc:ok(name,exc.status==403)
        async with s.get(base+'/health',headers={'Host':'untrusted.invalid'}) as r:ok('reject unexpected Host',r.status==403)
        async with s.ws_connect(base+'/ws?token='+meta['token'],headers={'Origin':base}) as ws:
            async def response(kind):
                for _ in range(100):
                    msg=await ws.receive_json(timeout=10)
                    if msg['type']==kind:return msg
                raise TimeoutError(kind)
            hello=await response('hello');frames=[hello]
            ok('map 64x48 and real neuron metadata transmitted',hello['map']['width']==64 and hello['map']['depth']==48 and len(hello['neurons'])==48)
            for _ in range(23):frames.append(await response('frame'))
            f=frames[-1]
            ok('actual CPU training metrics transmitted',f['training']['device']=='cpu' and f['training']['update']>0)
            ok('both agents transmit activity and input RGB',all(len(a['activity'])==48 and a['observation']['rgb_base64'] for a in f['agents']))
            async def command(c,**values):
                await ws.send_json({'type':'command','command':c,**values});return await response('ack')
            a=await command('train_pause');ok('train pause acknowledged',a['train_paused'])
            await asyncio.sleep(.7);u=(await health())['update'];await asyncio.sleep(.5)
            ok('learner actually pauses',u==(await health())['update'])
            await command('checkpoint')
            for _ in range(100):
                if (run/'latest.pt').exists():break
                await asyncio.sleep(.1)
            ok('checkpoint saved while paused',(run/'latest.pt').is_file())
            a=await command('train_resume');ok('train resume acknowledged',not a['train_paused'])
            a=await command('view_pause');ok('viewer pause does not pause learner',a['viewer_paused'] and not a['train_paused'])
            await asyncio.sleep(.8);ok('learner advances with viewer paused',(await health())['update']>u)
            await command('view_resume')
            a=await command('view_speed',value=8);ok('8x viewer speed acknowledged without changing training',a['viewer_speed']==8 and not a['train_paused'])
            ok('health reports viewer speed',(await health())['viewer_speed']==8)
            await ws.send_json({'type':'command','command':'exec','value':'not permitted'})
            ok('reject arbitrary command',(await response('error'))['message']=='Unsupported command')
        u=(await health())['update'];await asyncio.sleep(.7)
        ok('learner advances without any connected browser',(await health())['update']>u)
    (ROOT/'evidence/transport_frames.json').write_text(json.dumps(frames),encoding='utf-8')
    report={'passed':len(checks),'checks':checks,'transport':'actual loopback HTTP + aiohttp WebSocket client','model':'tiny CPU model: envs4, hidden48, RGB12x8, horizon8, 3-second episodes','limitation':'This is functional integration, not a browser socket end-to-end or skill/GPU benchmark.'}
    (ROOT/'evidence/transport_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def main():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='flyfight-test-') as tmp:
        run=Path(tmp)/'run'
        with (ROOT/'evidence/transport_server.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'run.py','--device','cpu','--envs','4','--hidden','48','--width','12','--height','8','--horizon','8','--epochs','1','--minibatch-envs','4','--threads','2','--episode-seconds','3','--publish-every','2','--save-every','10000','--run-dir',str(run),'--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
            try:report=asyncio.run(run_checks('http://127.0.0.1:'+str(port),run))
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform=='win32' else signal.SIGINT)
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
        assert (run/'latest.pt').exists(),'graceful shutdown checkpoint'
        report['graceful_shutdown_checkpoint']=True
        (ROOT/'evidence/transport_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
