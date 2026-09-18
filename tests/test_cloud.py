import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
import pytest

from cloud import cloud_config
from flyfight.server import create_app
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
