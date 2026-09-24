"""v1.1 minus its constraints must equal v1 byte for byte (same uids, order, inputs, answers).  usage: V1_DIR V11_DIR"""
import json, sys
a, b = sys.argv[1:3]
n = diff = s2 = 0
for la, lb in zip(open(f"{a}/train.jsonl"), open(f"{b}/train.jsonl"), strict=True):
    ra, rb = json.loads(la), json.loads(lb); n += 1
    ub = json.loads(rb["messages"][1]["content"])
    s2 += "constraints" in ub
    ub.pop("constraints", None)
    same = (ra["uid"] == rb["uid"] and ra["messages"][0] == rb["messages"][0] and ra["messages"][2] == rb["messages"][2]
            and json.loads(ra["messages"][1]["content"]) == ub)
    diff += not same
print(f"train rows {n}, v1.1 rows with constraints {s2}, rows that differ once constraints are removed: {diff}")
