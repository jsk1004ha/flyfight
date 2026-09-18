"""Embed the shared map and viewer sources into a standalone HTML file."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
src=ROOT/"viewer"/"src"
html=(src/"index.html").read_text(encoding="utf-8")
spec=json.loads((ROOT/"maps"/"arena_large.json").read_text())
html=html.replace("<!--MAP_DATA-->",'<script>window.FLYFIGHT_MAP='+json.dumps(spec,separators=(",",":"))+';</script>')
for name in ("engine.js","app.js"):
    js=(src/name).read_text(encoding="utf-8").replace("</script","<\\/script")
    html=html.replace(f'<script src="{name}"></script>',f'<script>\n{js}\n</script>')
out=ROOT/"viewer"/"FlyFight_Viewer.html"
out.write_text(html,encoding="utf-8")
print(out, out.stat().st_size)
