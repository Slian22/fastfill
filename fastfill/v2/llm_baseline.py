"""Prompt-engineering baseline: an OpenAI-compatible chat model places the three-field request.

Keeps the evaluation rows whose room is a boundary-known rectangle and projects them to the same
three-field condition the structured model is selected on (``evaluate.project_minimal``). The LLM sees
room type, room extent and the furniture list (id, category, description) only, and answers every
object's width/depth/height, bottom centre and facing angle. Writes ``rows.jsonl`` (the projected rows)
and ``predictions.jsonl`` (one row each, ``layout`` null on failure) for::

    python -m fastfill.v2.evaluate --data <out>/rows.jsonl --predictions <out>/predictions.jsonl --output <new dir>

Credentials come from an env file (OPENAI_BASE_URL, OPENAI_API_KEY, optional OPENAI_MODEL), never argv.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import time
import urllib.request

from fastfill.v2.evaluate import project_minimal
from fastfill.v2.io import read_samples, safe_output
from fastfill.v2.schema import validate_layout

SYSTEM = ("You are an interior layout designer. Place every listed object in the room and answer with JSON only. "
          "Coordinates are metres in the room frame: x and y on the floor, z up. width_m is the object's "
          "left-right extent, depth_m its front-back extent, height_m its vertical extent. x, y, z is the centre "
          "of the object's bottom face (z = height of the surface it stands on). facing_deg is the direction its "
          "front faces, counter-clockwise from +x (0 faces +x, 90 faces +y). Use realistic sizes, keep objects "
          "inside the room, avoid overlaps except objects resting on others.")


def load_env(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def prompt(condition):
    room = condition["room"]
    xs, ys = [p[0] for p in room["floor_polygon_xy_m"]], [p[1] for p in room["floor_polygon_xy_m"]]
    height = room.get("height_m")
    objects = [{"id": o["id"], "category": o["category"], "description": o["description"]} for o in condition["objects"]]
    return (f"Room type: {room.get('room_type', 'unknown')}\n"
            f"Floor: x from {min(xs):.3f} to {max(xs):.3f}, y from {min(ys):.3f} to {max(ys):.3f}, floor z = {room['floor_z_m']}\n"
            f"Ceiling height: {f'{height:.3f} m' if height else 'unknown'}\n"
            f"Objects ({len(objects)}):\n{json.dumps(objects, ensure_ascii=False)}\n"
            'Answer {"objects": [{"id", "width_m", "depth_m", "height_m", "x", "y", "z", "facing_deg"}, ...]} '
            "with exactly one entry per id.")


def to_layout(reply, condition):
    """LLM JSON -> fastfill.v2 layout: local +X is the front, so size = [depth, width, height] and yaw = facing."""
    text = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()
    text = text[text.find("{"):text.rfind("}") + 1]
    answers = {str(o["id"]): o for o in json.loads(text)["objects"]}
    objects = []
    for request in condition["objects"]:
        o = answers[request["id"]]
        facing = math.radians(float(o["facing_deg"]))
        objects.append({"id": request["id"],
                        "target_size_local_m": [float(o["depth_m"]), float(o["width_m"]), float(o["height_m"])],
                        "bottom_center_m": [float(o["x"]), float(o["y"]), float(o["z"])],
                        "yaw_rad": (facing + math.pi) % (2 * math.pi) - math.pi})
    layout = {"schema_version": "fastfill.v2", "objects": objects}
    validate_layout(layout, condition)
    return layout


def chat(env, model, user, *, timeout=300.):
    body = json.dumps({"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}).encode()
    request = urllib.request.Request(env["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {env['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def run(data, output, env_file, *, model=None, workers=8, max_samples=None, retries=1, ask=chat):
    target = safe_output(output, create=True)
    env = load_env(env_file)
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise ValueError("pass --model or set OPENAI_MODEL in the env file")
    rows = [p for p in map(project_minimal, read_samples(data)) if p is not None][:max_samples]

    def one(row):
        start, error = time.perf_counter(), None
        for _ in range(retries + 1):
            try:
                layout = to_layout(ask(env, model, prompt(row["condition"])), row["condition"])
                return {"layout": layout, "error": None, "latency_s": time.perf_counter() - start}
            except Exception as exc:  # malformed or incomplete answers are failures, never dropped
                error = f"{type(exc).__name__}: {exc}"[:500]
        return {"layout": None, "error": error, "latency_s": time.perf_counter() - start}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, rows))
    with (target / "rows.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with (target / "predictions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    summary = {"model": model, "rows": len(rows), "failed": sum(r["layout"] is None for r in results),
               "mean_latency_s": sum(r["latency_s"] for r in results) / max(1, len(results)), "data": str(data)}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True, help="file with OPENAI_BASE_URL / OPENAI_API_KEY [/ OPENAI_MODEL]")
    p.add_argument("--model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-samples", type=int)
    a = p.parse_args(argv)
    print(json.dumps(run(a.data, a.output, a.env, model=a.model, workers=a.workers, max_samples=a.max_samples)))


if __name__ == "__main__":
    main()
