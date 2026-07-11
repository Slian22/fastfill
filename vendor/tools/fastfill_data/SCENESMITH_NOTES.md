# scenesmith released scenes — converter audit notes (T2.4-lite)

Audit of `/…/data/scenesmith_scenes/` for `convert_scenesmith.py`.
Verified 2026-07 against the local download (HF-cached release).

## Inventory (local reality)

| Subset | Scenes | Rooms | Furniture | Manipulands (free) | Notes |
|---|---|---|---|---|---|
| Room | scene_100, scene_150 | pet_store; pantry | 11; 1 | 98; 31 | full pipeline w/ critic |
| House | scene_186 | bedroom, bathroom | 10; 5 | 19; 8 | one 2-room house |
| NoCritic | scene_100 | pet_store | 11 | 143 | critic disabled → denser clutter, more tilt |
| NotGenerated | scene_100 | pet_store | 11 | 126 | **retrieved** (HSSD) assets, not generated |

**Total: 5 scene dirs, 6 rooms.** The briefing's "convert >= 20 real
samples / >= 10 real rooms" is impossible for this source — only 6 rooms
exist locally. The converter emits all 6 (one FastFillSample per room);
tests assert >= 5 rooms and >= 100 extracted surface objects instead.

## Directory layout (identical across subsets)

```
scene_N/
  package.xml                      # ROS package for package://scene/ URIs
  combined_house/
    house.dmd.yaml                 # THE pose source (Drake model directives)
    house_furniture_welded.dmd.yaml  # same, furniture welded (unused)
    house.blend                    # visualization only
  room_geometry/room_geometry_<room>.sdf   # floor + wall collision boxes
  room_<room>/generated_assets/{furniture,manipuland,wall_mounted,
    ceiling_mounted}/sdf/<asset_dir>/      # per-asset SDF + meshes
  floor_plans/<room>/{floors,walls,windows}/  # glTF visuals only
  mujoco/scene.xml                 # duplicate MJCF export (unused)
  materials/                       # textures
```

## Which files carry what

- **Poses**: only `combined_house/house.dmd.yaml`. Per room an
  `add_frame room_<name>_frame`; furniture + manipulands are `add_model`
  entries with `default_free_body_pose` (translation + `!AngleAxis`
  rotation) expressed in the **room frame** (`base_frame:
  room_<name>_frame`), so house-level placement never enters the sample.
  Wall/ceiling-mounted objects are `add_weld`ed with `X_PC` instead.
  Writer matched in source: `scenesmith/agent_utils/house.py::
  HouseScene.assemble` → `_generate_combined_directive` →
  `scenesmith/agent_utils/room.py::RoomScene.to_drake_directive`.
- **Room shell**: `room_geometry_<room>.sdf` — an axis-aligned
  `floor_collision` box (all 6 rooms are rectangles centered on the room
  frame origin) plus per-wall collision boxes whose top edge gives the
  ceiling height (2.7–3.6 m here). Windows exist only as glTF visuals;
  no machine-readable door/window spec, so `RoomContext.doors/windows`
  stay empty.
- **Dimensions**: nothing persisted. Derived as the union AABB of each
  asset SDF's collision OBJ meshes (× per-mesh `<scale>`); OBJ files are
  in the Z-up model frame (Drake reads OBJ verbatim; the visual glTF is
  Y-up and would need a rotation — collision meshes avoid that entirely).
  Multi-link articulated SDFs (e.g. NotGenerated's
  `aquarium_stand_…/model_shoe_cabinet_4.sdf`, free-body key `E_body_1`
  instead of `base_link`) are unioned across links, joints ignored.
- **Categories**: not persisted; recovered from the directive model name
  (`pet_store_dog_ball_toy_0_f0_1` → `dog_ball_toy`).

## Coordinate frame (confirmed)

scenesmith is **Z-up, right-handed, meters** (Drake) — no Y-up mapping
applied. Asset canonicalization (`scenesmith/agent_utils/
mesh_canonicalization.py::canonicalize_mesh`) guarantees up=+Z,
**front=+Y**, bottom at z=0, XY-centered origin — i.e. exactly the
FastFill contract, so directive yaw maps 1:1 to `yaw_deg` (extracted from
the AngleAxis rotation matrix as `atan2(R10, R00)`, normalized to
[-180, 180)). Empirically verified: collision AABBs are XY-centered to
<1 mm with z_min ≈ 0; furniture translations have |z| < 7 mm.

## Support relations: NOT persisted → geometrically inferred

`HouseScene.assemble` also writes `house_state.json` /
`sceneeval_state.json`, whose `SceneObject.placement_info.
parent_surface_id` (room.py `PlacementInfo`) IS the explicit support
relation — **but these files are not shipped in the release** (no *.json
anywhere in the scene dirs). All manipulands appear as free bodies in the
room frame.

The converter therefore infers support geometrically (flagged
`support=inferred_geometric` in provenance.notes): a manipuland is
assigned to the smallest-footprint furniture whose yawed footprint
(+5 cm margin) contains its XY and whose z-range contains its resting
height; resting heights per parent are clustered into shelf levels
(4 cm merge) → one `SupportSurfaceSpec` (TOP if within 6 cm of the parent
top, else SHELF; `polygon_local` = parent footprint rect) + one FREE
`SurfaceObjectGroup` per level. Result on this data: **every upright
manipuland found a parent** (0 floor-standing, 0 orphans, 327 placed),
and levels match visible shelf spacing (e.g. retail shelf: 0.10 / 0.40 /
0.70 / 1.02 / 1.33 / 1.70 m).

## Quality flags per subset

- **Room / House**: cleanest layouts (critic-reviewed), all upright
  furniture, modest clutter.
- **NoCritic**: same shell as Room/scene_100, ~1.5× the manipulands,
  ~2× the tilted-object rate — usable but noisier surface layer.
- **NotGenerated**: fully populated (the name means *assets* not
  generated, not "scene empty") — furniture/manipuland dirs are 40-hex
  SHA-1 ids, matching HSSD mesh ids
  (`scenesmith/agent_utils/hssd_retrieval/data_loader.py::
  construct_hssd_mesh_path`). Tagged `LicenseTag.CC_BY_NC` +
  `assets=hssd_retrieved_sha1_ids` note, **contradicting the briefing's
  blanket "license PERMISSIVE"** (铁律 2: HSSD is CC BY-NC).

## Skips (counted, never silent) — full-run stats

`{'converted': 6, 'skipped': {'welded_wall_mounted_object': 53,
'welded_ceiling_mounted_object': 63, 'manipuland_tilted_pose': 98,
'welded_manipuland_object': 6}}`

- wall/ceiling-mounted decor: welded, outside the floor+surface contract;
- tilted manipulands (R[2,2] < 0.9 ≈ >25° off vertical: balls, potatoes,
  fallen bags): unrepresentable in the yaw-only schema — skipped rather
  than re-uprighted (98/425 ≈ 23%);
- 6 welded manipulands = aquarium gravel mats (NoCritic), fixed decor.

## Could NOT find

- explicit support/parent relations (see above), room task text, style
  hints, door specs, per-asset bbox metadata, `scene_state.json` in any
  `room_*/` dir. `RoomContext.task` / `expected_*` left empty per T2.6.

## Reality vs briefing contradictions (summary)

1. Only 6 rooms locally → ">= 20 samples / >= 10 rooms" unreachable.
2. No `scene_state.json` / object-metadata JSON is shipped at all.
3. NotGenerated is NOT unusable/empty — but it is HSSD-derived, so it is
   the one subset that cannot default to a permissive license tag.
