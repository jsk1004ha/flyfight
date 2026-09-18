import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
import pytest

from cloud import cloud_config
from flyfight.server import create_app, training_ready
from flyfight.training import Config


def test_cloud_resume_preserves_saved_learning_config(tmp_path):
    values = dict(action_steps=2001,turn_speed=1.234,headshot_bonus=.375,updates=1)
    (tmp_path/'config.json').write_text(json.dumps(values),encoding='utf-8')
    (tmp_path/'latest.pt').touch()
    cfg = cloud_config(tmp_path)
    assert cfg.action_steps == 2001 and cfg.turn_speed == 1.234 and cfg.headshot_bonus == .375
    assert cfg.updates == 0 and cfg.resume == str(tmp_path/'latest.pt')


def test_cloud_refuses_missing_checkpoint(tmp_path):
    (tmp_path/'config.json').write_text('{}',encoding='utf-8')
    with pytest.raises(RuntimeError,match='no checkpoint'):
        cloud_config(tmp_path)


def test_public_viewer_denies_control_and_foreign_origin():
    async def check():
        origin='https://flyfight.example.org'
        app=create_app(Config(envs=1,hidden=24),public_origin=origin)
        app.on_startup.clear(); app.on_cleanup.clear()
        async with TestClient(TestServer(app)) as client:
            response=await client.get('/',headers={'Host':'flyfight.example.org'})
            html=await response.text()
            assert response.status == 200
            settings=json.loads(html.split('window.FLYFIGHT_NATIVE=')[1].split(';</script>')[0])
            assert settings['readOnly']
            assert (await client.get('/',headers={'Host':'attacker.example.org'})).status == 403
            assert (await client.get('/ws',headers={'Host':'flyfight.example.org','Origin':'https://attacker.example.org'})).status == 403
            ws=await client.ws_connect('/ws?token='+settings['token'],headers={'Host':'flyfight.example.org','Origin':origin})
            await ws.send_json({'type':'command','command':'train_pause'})
            message=await ws.receive_json()
            assert message['type']=='error'
            health=await (await client.get('/health',headers={'Host':'flyfight.example.org'})).json()
            assert not health['train_paused']
            await ws.close()
    asyncio.run(check())


def test_training_readiness_requires_live_trainer_and_model():
    assert training_ready({'state':'training'},True,None,True)
    for state,view,error,alive in [('starting',True,None,True),('error',True,None,True),
                                  ('stopped',True,None,True),('training',False,None,True),
                                  ('training',True,'failed',True),('training',True,None,False)]:
        assert not training_ready({'state':state},view,error,alive)


def test_pod_probes_are_minimal_and_do_not_relax_viewer_host_checks():
    async def check():
        app=create_app(Config(envs=1,hidden=24),public_origin='https://flyfight.example.org')
        app.on_startup.clear(); app.on_cleanup.clear()
        async with TestClient(TestServer(app)) as client:
            headers={'Host':'10.42.1.7:8765'}
            response=await client.get('/health/live',headers=headers)
            assert response.status == 200 and await response.json() == {'alive':True}
            response=await client.get('/health/ready',headers=headers)
            assert response.status == 503 and await response.json() == {'ready':False}
            for path in ('/','/health','/ws'):
                assert (await client.get(path,headers=headers)).status == 403
    asyncio.run(check())
