"""End-to-end contract test: EmbodiedGen FastFillBackend (its real request builder, OpenAI client, cache and
ff_to_ours) -> fastfill.serve over real HTTP -> interface.py -> an ORACLE "model" whose text is the grid placer's
layout written in the model frame. If every conversion is right, the fastfill room equals the grid room.
Field names in the backend request are renamed to Appendix B here (id->subject, near->max_gap_m), simulating the
two-line backend change we propose; nothing in EmbodiedGen is edited."""
import json, math, os, sys, tempfile, threading
from http.server import HTTPServer

sys.path[:0] = [os.environ.get("EMBODIEDGEN", "../EmbodiedGen"), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from embodied_gen.world import backend
from embodied_gen.world.spec import AttachmentSpec, EntityReq, RelationReq, SemanticSpec
from dataclasses import replace
from fastfill.interface import request_to_room
from fastfill.serve import make_handler
from fastfill.scene import apply
from fastfill.validate import check

orig_request = backend.fastfill_request
def appendix_b_request(spec, attach):
    r = orig_request(spec, attach)
    for c in r["layout_constraints"]:
        if "id" in c:
            c["subject"] = c.pop("id")
        if "max_distance_m" in c:
            del c["max_distance_m"]; c["max_gap_m"] = 0.5
    return r
backend.fastfill_request = appendix_b_request

STASH, PROMPTS, MODE = {}, [], {"oracle": True}
orig_realize = backend.FastFillBackend.realize
def stash_realize(self, spec, attach, *, seed=0):
    STASH.update(spec=spec, attach=attach, seed=seed)
    return orig_realize(self, spec, attach, seed=seed)
backend.FastFillBackend.realize = stash_realize

OOB = []
def oracle(msgs):
    """The grid's layout, as the model would have to write it: model ids, model frame, integer degrees."""
    PROMPTS.append(msgs[1]["content"])
    if not MODE["oracle"]:
        return "sorry, no layout"
    spec, attach, seed = STASH["spec"], STASH["attach"], STASH["seed"]
    room, ctx = request_to_room(backend.fastfill_request(spec, attach))
    grid = {p["id"]: p for p in backend.ours_to_ff(backend.GridBackend().realize(spec, attach, seed=seed).instances)["placements"]}
    m = {o["ext_id"]: o["id"] for o in room["objects"]}
    out = []
    for o in room["objects"]:
        if o["ext_id"] not in grid:     # grid could not fit it; the model has to place all, so this answer fails
            continue
        p = grid[o["ext_id"]]
        q = {"id": o["id"]}
        if p["support_parent"] in m:
            q["on"] = m[p["support_parent"]]
        x, y, z = p["position_m"]
        q["pos"] = [x - ctx["x0"], y - ctx["y0"], z - ctx["floor_z"]]
        q["yaw"] = round(math.degrees(p["yaw_rad"] + ctx["front_k"][o["ext_id"]] * math.pi / 2)) % 360
        out.append(q)
    pl = {q["id"]: {"pos": q["pos"], "yaw": math.radians(q["yaw"]), "on": q.get("on")} for q in out}
    OOB.extend(check(apply(room, pl))["oob"])          # grid's valid room vs our padded keepout notch
    return json.dumps({"placements": out})

srv = HTTPServer(("127.0.0.1", 0), make_handler(oracle, "fastfill"))
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ.update(WORLDEDGE_FASTFILL_URL=f"http://127.0.0.1:{srv.server_port}/v1", WORLDEDGE_FASTFILL_MODEL="fastfill",
                  WORLDEDGE_CACHE=tempfile.mkdtemp())

ATTACH = AttachmentSpec(parent_room_id="room_0", entry_wall="east", entry_anchor=1.2, entry_width=0.9,
                        available_envelope=(4.0, 3.5), entry_keepout=((0.0, 0.75, 1.0, 2.25),))
KITCHEN = SemanticSpec(task="wipe down the counter", room_type="kitchen",
                       required_entities=((EntityReq("kitchen_counter", "furniture"), EntityReq("sink", "receptacle")),),
                       required_relations=(RelationReq("sink", "on", "kitchen_counter"),))
DINING = SemanticSpec(task="lay the table", room_type="dining_room",
                      required_entities=((EntityReq("dining_table", "furniture"), EntityReq("chair", "furniture", count=3)),))
LIVING = SemanticSpec(task="sit down", room_type="living_room", required_entities=((EntityReq("sofa", "furniture"),),))

overlaps = lambda a, b: not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
n = bad = 0
REFUSED = []
for spec in (KITCHEN, DINING, LIVING):
    for wall in ("north", "south", "east", "west"):
        for keep, env in ((ATTACH.entry_keepout, (4.0, 3.5)), (((0.0, 0.4, 1.6, 3.1),), (6.0, 5.0)), (ATTACH.entry_keepout, (7.0, 6.0))):
            for seed in (1, 2, 3, 4, 5):
                att = replace(ATTACH, entry_wall=wall, entry_keepout=keep, available_envelope=env)
                gr = backend.realize_room("grid", spec, att, seed=seed)
                n += 1
                try:
                    ff = backend.realize_room("fastfill", spec, att, seed=seed)
                except backend.BackendError as e:
                    asked = len(backend.fastfill_request(spec, att)["objects_to_place"])
                    if len(gr.instances) < asked:
                        REFUSED.append((spec.room_type, wall, seed, asked, len(gr.instances)))
                        continue
                    raise
                g = {i.instance_id: i for i in gr.instances}
                f = {i.instance_id: i for i in ff.instances}
                errs = []
                if set(f) != set(g):
                    errs.append(f"ids {sorted(set(f) ^ set(g))}")
                for k in set(f) & set(g):
                    a, b = f[k], g[k]
                    dy = abs(math.remainder(a.pose[3] - b.pose[3], math.tau))
                    if max(abs(a.pose[i] - b.pose[i]) for i in range(3)) > 1e-6 or dy > math.radians(0.5) + 1e-9 \
                            or a.parent != b.parent or a.extent != b.extent:
                        errs.append(f"{k}: ff {a.pose} {a.parent} {a.extent} vs grid {b.pose} {b.parent} {b.extent}")
                for i in ff.instances:
                    if not i.parent and any(overlaps(backend.footprint_xy(i), r) for r in keep):
                        errs.append(f"{i.instance_id} in keepout")
                if ff.relations != gr.relations or ff.actual_entry != gr.actual_entry:
                    errs.append(f"relations/entry {ff.relations} {ff.actual_entry} vs {gr.relations} {gr.actual_entry}")
                if errs:
                    bad += 1
                    print(spec.room_type, wall, keep, seed, errs[:3])
print(f"{n - bad - len(REFUSED)}/{n} rooms: fastfill path == grid (ids, pose, parent, extent, relations, entry), "
      f"none in keepout; {len(REFUSED)} refused because the shopping list did not fit (grid dropped objects), e.g. {REFUSED[:2]}")
print("grid objects our padded keepout notch would call out-of-bounds:", len(OOB))
print("one prompt the model sees:", PROMPTS[0][:700])

MODE["oracle"] = False              # a model answer that does not parse must refuse the room, not commit an empty one
try:
    backend.realize_room("fastfill", KITCHEN, replace(ATTACH, entry_anchor=1.3), seed=99)
    print("FAIL: unparsable answer produced a room")
except backend.BackendError as e:
    print("unparsable answer -> BackendError (room refused):", str(e)[:120])
backend.fastfill_request = orig_request  # the backend as it is today: id / max_distance_m
try:
    backend.realize_room("fastfill", KITCHEN, replace(ATTACH, entry_anchor=1.4), seed=98)
    print("today's field names: accepted?!")
except backend.BackendError as e:
    print("today's field names -> BackendError:", str(e)[:160])
