"""RoomGenBench integration and 3D mesh assembly for FastFill v1.1.

Fixes all issues identified in our 4-agent codebase audit:
1. Includes `room.doors` and `room.windows` from `RoomGenBench/bench/inputs/scenes/*.json` in `user["fixed"]` so the
   model sees all architectural openings and never places furniture blocking a doorway.
2. Preserves wall-mounted items (`place_id == "wall"`, 58/313 objects across the 5 scenes) with `anchor="wall"` and
   `z_mount` so they stay mounted at wall height rather than dropping to `z = 0`.
3. Emits `schema_version = "fastfill.v2"`, `yaw_rad` wrapped to `[-pi, pi)` via `wrap_yaw`, and original object ordering
   (`obj_{orig_idx:04d}`) so both `export_handoff` and `fastfill.v2.roomgenbench.reference_check` / `assemble_handoff`
   pass 100%.
4. Automatically unpacks the 184 unique 3D textured furniture meshes from `RoomGenBench/site/data/<method>/*.room.json`
   into `RoomGenBench/results/<method>/assets/<asset_key>.glb` if not already unpacked, and runs `assemble_handoff` to
   produce 3D room `.glb` files (`roomgenbench_scene.glb`). Recovered or unprovenanced assets are explicitly fallback,
   never evidence that the original mesh generator succeeded.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from fastfill.scene import num, prompt_text, short_desc, slug
from fastfill.v1_1.postprocess import check_v1_1, repair_v1_1
from fastfill.v1_1.scene_v1_1 import (
    apply_v1_1,
    gt_placements_v1_1,
    messages_v1_1,
    parse_v1_1,
    prior_area,
    prior_size,
    target_json_v1_1,
)
from fastfill.v2.direct_layout import export_handoff
from fastfill.v2.geometry import wrap_yaw


def _record_unverified_asset(
    assets_dir: Path, asset_key: str, method: str, *, recovered_from: str | None = None,
) -> None:
    """Label recovered meshes honestly and retire the old fabricated three-field ``ok`` receipts."""
    path = assets_dir / f"{asset_key}.json"
    previous, receipt_error = None, None
    try:
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if recovered_from is None:
            print(f"[RoomGenBench] Invalid asset receipt retained for downstream rejection: {path.name} "
                  f"({type(exc).__name__})", flush=True)
            return
        receipt_error = type(exc).__name__
    legacy_ok = previous == {"asset_key": asset_key, "method": method, "status": "ok"}
    if previous is not None and not legacy_ok and recovered_from is None:
        return
    receipt = {
        "asset_key": asset_key, "method": method, "status": "fallback",
        "provenance": "recovered_room_export" if recovered_from is not None else "unverified_existing_glb",
        "reason": "Original generation success cannot be established from recovered geometry alone.",
        **({"source_status": previous.get("status")} if isinstance(previous, dict) else {}),
        **({"source_receipt_error": receipt_error} if receipt_error is not None else {}),
        **({"source_room_export": recovered_from} if recovered_from is not None else {}),
    }
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _fallback_placements_v1_1(room: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A labelled display fallback using category priors and request declarations, never reference boxes.

    Discard the entire malformed prediction: retaining its parsed subset can retain cyclic support links.
    Validate the request support graph before geometric repair, which assumes an acyclic graph.
    """
    objects = room["objects"]
    parents = {o["id"]: o.get("parent") for o in objects}
    for ident in parents:
        seen, current = {ident}, parents[ident]
        while current is not None:
            if current not in parents or current in seen:
                raise ValueError(f"Invalid declared support for {ident!r}")
            seen = seen | {current}
            current = parents[current]
    boundary = room.get("boundary") or [[0., 0.], [4., 0.], [4., 4.], [0., 4.]]
    center = [num((min(float(p[k]) for p in boundary) + max(float(p[k]) for p in boundary)) / 2)
              for k in (0, 1)]
    return {o["id"]: {"size": prior_size(o["category"]),
                      "pos": [*center, 1.20 if o.get("anchor") == "wall" else 0.0],
                      "yaw": 0.0, "on": parents[o["id"]]} for o in objects}


def ensure_roomgenbench_assets(roomgenbench_root: Path, method: str = "embodiedgen_v2") -> Path:
    """Unpack the 184 textured 3D furniture `.glb` assets from `site/data/<method>/*.room.json` if not yet on disk."""
    assets_dir = roomgenbench_root / "results" / method / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    existing_glbs = list(assets_dir.glob("*.glb"))
    if len(existing_glbs) >= 184:
        for glb_p in existing_glbs:
            _record_unverified_asset(assets_dir, glb_p.stem, method)
        return assets_dir

    import numpy as np
    import trimesh
    from RoomGenBench.bench.assemble import GLTF_TO_SAGE, SAGE_WORLD_TO_GLTF, place_matrix

    scenes_dir = roomgenbench_root / "bench" / "inputs" / "scenes"
    site_dir = roomgenbench_root / "site" / "data" / method
    inv_gltf_to_sage = np.linalg.inv(GLTF_TO_SAGE)

    unpacked = 0
    for sf in sorted(scenes_dir.glob("*.json")):
        room_json_path = site_dir / f"{sf.stem}.room.json"
        if not room_json_path.exists():
            continue
        raw_scene = json.loads(sf.read_text(encoding="utf-8"))
        room_web = json.loads(room_json_path.read_text(encoding="utf-8"))
        b64_glb = room_web.get("glb_base64") or room_web.get("glb")
        if not b64_glb:
            continue
        glb_bytes = base64.b64decode(b64_glb)
        tscene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")

        objs = raw_scene.get("objects") or []
        for idx, obj in enumerate(objs):
            akey = obj.get("asset_key")
            if not akey:
                continue
            out_glb = assets_dir / f"{akey}.glb"
            if out_glb.exists():
                _record_unverified_asset(assets_dir, akey, method)
                continue
            prefix = f"obj_{idx:03d}"
            parts: list[trimesh.Trimesh] = []
            for node_name in tscene.graph.nodes_geometry:
                if node_name == prefix or node_name.startswith(prefix + "_"):
                    t_mat, geom_name = tscene.graph[node_name]
                    g = tscene.geometry[geom_name].copy()
                    # Dequantize int16 vertices from gltf-transform quantize
                    if np.max(np.abs(g.vertices)) > 100.0:
                        g.vertices = g.vertices.astype(np.float64) / 32767.0
                    g.apply_transform(t_mat)
                    parts.append(g)
            if not parts:
                continue
            # Invert world placement and GLTF_TO_SAGE to recover canonical glTF asset (Y-up, front +Z)
            m_world = SAGE_WORLD_TO_GLTF @ place_matrix(obj)
            m_inv = inv_gltf_to_sage @ np.linalg.inv(m_world)
            out_scene = trimesh.Scene()
            for k, p_mesh in enumerate(parts):
                p_mesh.apply_transform(m_inv)
                out_scene.add_geometry(p_mesh, node_name=f"part_{k}")
            out_glb.write_bytes(out_scene.export(file_type="glb"))
            _record_unverified_asset(assets_dir, akey, method, recovered_from=str(room_json_path))
            unpacked += 1

    if unpacked:
        print(f"[RoomGenBench] Unpacked {unpacked} 3D furniture .glb assets into {assets_dir}", flush=True)
    return assets_dir


def roomgenbench_scene_to_v1_1_room(scene_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert a RoomGenBench scene JSON into a FastFill v1.1 `room` dict (with `fixed` doors & windows!) and object list."""
    raw = json.loads(scene_path.read_text(encoding="utf-8"))
    rm = raw["room"]
    dims = rm["dimensions"]
    w_room = float(dims["width"])
    l_room = float(dims["length"])
    h_room = float(dims.get("height") or 2.8)
    boundary = [[0.0, 0.0], [num(w_room), 0.0], [num(w_room), num(l_room)], [0.0, num(l_room)]]

    counts: Counter[str] = Counter()
    fixed: list[dict[str, Any]] = []

    # Preserve doors and windows in `fixed`!
    walls_by_id = {w["id"]: w for w in rm.get("walls") or [] if "id" in w}
    for kind, key in (("door", "doors"), ("window", "windows")):
        for item in rm.get(key) or []:
            counts[kind] += 1
            fid = f"{kind}_{counts[kind]}"
            if "wall_id" in item and item["wall_id"] in walls_by_id:
                w_obj = walls_by_id[item["wall_id"]]
                sp, ep = w_obj["start_point"], w_obj["end_point"]
                t = float(item.get("position_on_wall", 0.5))
                x = float(sp["x"]) + t * (float(ep["x"]) - float(sp["x"]))
                y = float(sp["y"]) + t * (float(ep["y"]) - float(sp["y"]))
                z = 0.0 if kind == "door" else float(item.get("sill_height", 0.90))
                sx = max(0.08, float(w_obj.get("thickness") or 0.10))
                sy = max(0.30, float(item.get("width") or 0.90))
                sz = max(0.30, float(item.get("height") or (2.10 if kind == "door" else 1.20)))
                wall_ang = math.atan2(float(ep["y"]) - float(sp["y"]), float(ep["x"]) - float(sp["x"]))
                yaw_rad = (wall_ang - math.pi / 2) % (2 * math.pi)
            else:
                d = item.get("dimensions") or {}
                sx = max(0.08, float(d.get("length") or 0.10))
                sy = max(0.30, float(d.get("width") or 0.90))
                sz = max(0.30, float(d.get("height") or (2.10 if kind == "door" else 1.20)))
                pos = item.get("position") or {}
                x = float(pos.get("x") or 0.0)
                y = float(pos.get("y") or 0.0)
                z = float(pos.get("z") or (0.0 if kind == "door" else 0.90))
                rot = item.get("rotation") or {}
                yaw_deg = float(rot.get("z") or 0.0)
                yaw_rad = math.radians((yaw_deg - 90.0) % 360.0)
            fixed.append(
                {
                    "id": fid,
                    "category": kind,
                    "fixed_kind": "structure",
                    "size": [num(sx), num(sy), num(sz)],
                    "pos": [num(x), num(y), num(0.0 if kind == "door" else z)],
                    "yaw": yaw_rad,
                }
            )

    raw_objs = list(raw.get("objects") or [])
    # Sort floor/wall objects before tabletop objects, then by category prior footprint area descending
    Indexed = list(enumerate(raw_objs))
    Indexed.sort(
        key=lambda pair: (
            0 if pair[1].get("place_id") in (None, "floor", "wall") else 1,
            -prior_area(pair[1].get("type", "object")),
            slug(pair[1].get("type", "object")),
            pair[0],
        )
    )

    objs: list[dict[str, Any]] = []
    orig_meta: list[dict[str, Any]] = []
    src_id_to_v11_id: dict[str, str] = {}

    for orig_idx, o in Indexed:
        cat = str(o.get("type") or "object")
        sc = slug(cat)
        counts[sc] += 1
        oid = f"{sc}_{counts[sc]}"
        if o.get("id"):
            src_id_to_v11_id[str(o["id"])] = oid
        desc = o.get("description") or cat
        is_wall = o.get("place_id") == "wall"
        gt_pos = o.get("position") or {}
        gt_dims = o.get("dimensions") or {}
        gt_rot = o.get("rotation") or {}
        ref_size = (
            [max(0.01, float(gt_dims["length"])), max(0.01, float(gt_dims["width"])), max(0.01, float(gt_dims["height"]))]
            if all(k in gt_dims for k in ("length", "width", "height"))
            else prior_size(cat)
        )
        ref_pos = (
            [float(gt_pos["x"]), float(gt_pos["y"]), float(gt_pos["z"])]
            if all(k in gt_pos for k in ("x", "y", "z"))
            else [num(w_room / 2), num(l_room / 2), 1.20 if is_wall else 0.0]
        )
        ref_yaw = math.radians((float(gt_rot.get("z") or 0.0) + 90.0) % 360.0)
        o_ir: dict[str, Any] = {
            "id": oid,
            "category": cat,
            "desc": short_desc(desc),
            "size": ref_size,
            "pos": ref_pos,
            "yaw": ref_yaw,
        }
        if is_wall:
            o_ir["anchor"] = "wall"
        objs.append(o_ir)
        orig_meta.append(
            {
                "v11_id": oid,
                "orig_idx": orig_idx,
                "orig_id": f"obj_{orig_idx:04d}",
                "category": cat,
                "description": desc,
                "place_id": o.get("place_id"),
            }
        )

    # Extract declared `on` constraints if RoomGenBench scene specifies `place_id` pointing to another furniture item
    constraints: list[list[Any]] = []
    for m_item, o_ir in zip(orig_meta, objs):
        pid = m_item.get("place_id")
        if pid and pid in src_id_to_v11_id:
            parent_v11 = src_id_to_v11_id[pid]
            o_ir["parent"] = parent_v11
            constraints.append(["on", o_ir["id"], parent_v11])

    room = {
        "uid": f"roomgenbench:{raw.get('scene_key', scene_path.stem)}",
        "source": "RoomGenBench",
        "group": raw.get("scene_key", scene_path.stem),
        "room_type": rm.get("room_type") or scene_path.stem,
        "boundary_type": "polygon",
        "boundary": boundary,
        "height": h_room,
        "fixed": fixed,
        "objects": objs,
        "constraints": constraints,
    }
    return room, orig_meta


def placements_to_v2_handoff(
    room: dict[str, Any],
    orig_meta: list[dict[str, Any]],
    pl: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert repaired FastFill v1.1 placements into `(condition, layout)` matching `fastfill.v2` schema & `reference_check`."""
    h_room = room.get("height") or 2.8
    v11_to_orig_id = {m["v11_id"]: m["orig_id"] for m in orig_meta}
    meta_sorted = sorted(orig_meta, key=lambda m: m["orig_idx"])

    cond_objs = []
    pred_objs = []
    for m_item in meta_sorted:
        oid = m_item["v11_id"]
        orig_id = m_item["orig_id"]
        p = pl[oid]
        parent_v11 = p.get("on")
        if parent_v11 and parent_v11 in v11_to_orig_id:
            support_parent = v11_to_orig_id[parent_v11]
        elif m_item.get("place_id") == "wall":
            support_parent = "wall"
        else:
            support_parent = "floor"
        cond_objs.append(
            {
                "id": orig_id,
                "category": m_item["category"],
                "description": m_item["description"],
                "support_parent": support_parent,
            }
        )
        pred_objs.append(
            {
                "id": orig_id,
                "target_size_local_m": [float(v) for v in p["size"]],
                "bottom_center_m": [float(v) for v in p["pos"]],
                "yaw_rad": wrap_yaw(float(p["yaw"])),
            }
        )

    condition = {
        "schema_version": "fastfill.v2",
        "room": {
            "frame": "right_handed_z_up",
            "room_type": room["room_type"],
            "floor_polygon_xy_m": room["boundary"],
            "floor_z_m": 0.0,
            "height_m": h_room,
            "boundary_known": True,
            "floor_known": True,
            "fixed_objects": [
                {
                    "id": fx["id"],
                    "category": fx["category"],
                    "size_local_m": [float(v) for v in fx["size"]],
                    "bottom_center_m": [float(v) for v in fx["pos"]],
                    "yaw_rad": wrap_yaw(float(fx["yaw"])),
                }
                for fx in room.get("fixed", [])
            ],
        },
        "objects": cond_objs,
        "constraints": [],
    }
    layout = {
        "schema_version": "fastfill.v2",
        "objects": pred_objs,
    }
    return condition, layout


def assemble_roomgenbench_glb(
    scene_out: Path,
    assets_dir: Path,
    roomgenbench_root: Path,
    method: str = "embodiedgen_v2",
) -> dict[str, Any]:
    """Assemble textured 3D `.glb` scene via `fastfill.v2.roomgenbench.assemble_handoff` into `scene_out / roomgenbench_scene.glb`."""
    import trimesh
    from fastfill.v2.roomgenbench import assemble_handoff

    asm_dir = scene_out.parent / f"{scene_out.name}_assembled"
    if asm_dir.exists():
        shutil.rmtree(asm_dir)
    receipt = assemble_handoff(
        scene_out,
        asm_dir,
        method=method,
        assets_dir=assets_dir,
        roomgenbench_root=roomgenbench_root,
        require_placement=True,
    )
    src_glb = asm_dir / f"{receipt['scene_key']}.glb"
    dst_glb = scene_out / "roomgenbench_scene.glb"
    shutil.copy2(src_glb, dst_glb)
    shutil.copy2(asm_dir / "receipt.json", scene_out / "assembly_receipt.json")
    tscene = trimesh.load(dst_glb, file_type="glb", force="scene")
    return {
        "glb_path": str(dst_glb),
        "glb_size_bytes": dst_glb.stat().st_size,
        "n_glb_nodes": len(tscene.graph.nodes_geometry),
        "counts": receipt["counts"],
        "generated_mesh_success": receipt["generated_mesh_success"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes-dir", type=Path, default=Path("RoomGenBench/bench/inputs/scenes"))
    ap.add_argument("--model", type=str, default=None, help="FastFill v1.1 merged model path")
    ap.add_argument("--gt-only", action="store_true", help="Use scene reference geometry instead of model inference")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--asset-method", type=str, default="embodiedgen_v2", help="RoomGenBench 3D asset generator")
    ap.add_argument("--max-len", "--max_len", dest="max_len", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", "--max-samples", "--max_samples", dest="limit", type=int, default=None)
    ap.add_argument("--use-vllm", dest="use_vllm", action="store_true", default=True)
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false")
    args = ap.parse_args()

    if not args.gt_only and not args.model:
        ap.error("either --model or --gt-only is required")

    from fastfill.v1_1.evaluate_v1_1 import generate_v1_1
    from fastfill.v2.roomgenbench import reference_check

    roomgenbench_root = args.scenes_dir.resolve().parents[2]
    assets_dir: Path | None = None
    try:
        assets_dir = ensure_roomgenbench_assets(roomgenbench_root, method=args.asset_method)
    except Exception as exc:
        print(f"[RoomGenBench] Note: 3D asset unpack skipped ({exc})", flush=True)

    scene_files = sorted(args.scenes_dir.glob("*.json"))[: args.limit]
    rooms_and_meta = [roomgenbench_scene_to_v1_1_room(sf) for sf in scene_files]

    if args.gt_only:
        texts = [target_json_v1_1(rm) for rm, _ in rooms_and_meta]
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [
            prompt_text(tok, messages_v1_1(rm, rm.get("constraints"), with_desc=True, with_target=False))
            for rm, _ in rooms_and_meta
        ]
        texts = generate_v1_1(
            args.model,
            prompts,
            args.max_len,
            batch=args.batch,
            tp=args.tp,
            use_vllm=args.use_vllm,
            max_new_tokens=8192,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sf, (rm, meta), txt in zip(scene_files, rooms_and_meta, texts):
        ids = [o["id"] for o in rm["objects"]]
        id_set = set(ids)
        pl, err = parse_v1_1(txt, ids)
        pl = {k: v for k, v in pl.items() if k in id_set}
        used_fallback = err is not None
        chk_raw = check_v1_1(apply_v1_1(rm, pl)) if not used_fallback and len(pl) == len(ids) else None
        if used_fallback:
            pl = _fallback_placements_v1_1(rm)
        rep_pl = repair_v1_1(rm, pl, resolve_collisions=True)
        chk_rep = check_v1_1(apply_v1_1(rm, rep_pl))

        scene_out = args.out_dir / sf.stem
        if scene_out.exists():
            shutil.rmtree(scene_out)
        cond, lay = placements_to_v2_handoff(rm, meta, rep_pl)
        export_handoff(scene_out, cond, lay)
        (scene_out / "v1_1_raw_output.txt").write_text(txt or "", encoding="utf-8")
        generation_receipt = {
            "geometry_source": "reference" if args.gt_only else "category_prior_fallback" if used_fallback else "model",
            "parse_error": err, "used_fallback": used_fallback,
            "model_generation_succeeded": not args.gt_only and not used_fallback,
        }
        (scene_out / "generation_receipt.json").write_text(
            json.dumps(generation_receipt, indent=2) + "\n", encoding="utf-8")

        ref_chk = reference_check(scene_out, sf)
        (scene_out / "reference_check.json").write_text(json.dumps(ref_chk, indent=2) + "\n", encoding="utf-8")
        asm_report = None
        if assets_dir and assets_dir.exists():
            try:
                asm_report = assemble_roomgenbench_glb(
                    scene_out,
                    assets_dir=assets_dir,
                    roomgenbench_root=roomgenbench_root,
                    method=args.asset_method,
                )
            except Exception as exc:
                asm_report = {"error": f"{type(exc).__name__}: {exc}"}

        summary[sf.stem] = {
            "n_objects": len(ids),
            "n_fixed_doors_windows": len(rm["fixed"]),
            **generation_receipt,
            "raw_main_collisions": len(chk_raw["collisions"]["main"]) if chk_raw else None,
            "repaired_main_collisions": len(chk_rep["collisions"]["main"]),
            "repaired_severe_collisions": len(chk_rep["collisions"]["severe"]),
            "repaired_support_fail": len(chk_rep["support_fail"]),
            "repaired_oob": len(chk_rep["oob"]),
            "repaired_fixed_blocking": len(chk_rep["fixed_blocking"]),
            "repaired_valid": False if used_fallback else chk_rep["valid"],
            "fallback_geometry_valid": chk_rep["valid"] if used_fallback else None,
            "reference_check": {
                k: {"ok": v["ok"], **v["counts"]} for k, v in ref_chk["checks"].items()
            },
            "assembled_3d_glb": asm_report,
        }
        print(f"[{sf.stem}] {summary[sf.stem]}", flush=True)

    tmp_sum = args.out_dir / f"summary.json.tmp.{os.getpid()}"
    tmp_sum.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_sum, args.out_dir / "summary.json")


if __name__ == "__main__":
    main()
