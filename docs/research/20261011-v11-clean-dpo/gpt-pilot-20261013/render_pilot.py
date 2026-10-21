"""Render all five existing assembled scenes without modifying their geometry."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import bpy

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SOURCE = OUT.parent / "roomgenbench_clean_final"
SCENES = ("gym", "restaurant", "livingroom", "bedroom", "bathroom")
renderer = runpy.run_path(str(REPO / "RoomGenBench/bench/render.py"))
render_scene = renderer["render_scene"]
original_reset = render_scene.__globals__["reset"]


def reset():
    original_reset()
    bpy.context.scene.render.film_transparent = False


render_scene.__globals__["reset"] = reset
records = []
for name in SCENES:
    meta_path, = (SOURCE / f"{name}_assembled").glob("fastfill_*.json")
    mesh = meta_path.with_suffix(".glb")
    meta = json.loads(meta_path.read_text())
    prefix = OUT / "renders" / name
    render_scene(mesh, meta, prefix, 16, (960, 720))
    images = [Path(f"{prefix}_{view}.png") for view in ("persp", "top")]
    files = [mesh, meta_path, SOURCE / name / "condition.json", *images]
    records.append({
        "scene": name,
        "files": [{"path": str(p.relative_to(OUT.parent)),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                   "bytes": p.stat().st_size} for p in files],
    })
    print("PILOT_RENDER_COMPLETE", name, flush=True)
(OUT / "render-manifest.json").write_text(json.dumps({
    "blender_version": bpy.app.version_string,
    "renderer_sha256": hashlib.sha256((REPO / "RoomGenBench/bench/render.py").read_bytes()).hexdigest(),
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "engine": "Cycles CPU", "samples": 16, "resolution": [960, 720],
    "background": "opaque white", "geometry_modified": False,
    "views": ["perspective dollhouse; near walls hidden", "orthographic top"],
    "records": records,
}, indent=2) + "\n")
sys.stdout.flush()
os._exit(0)
