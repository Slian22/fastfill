"""OpenAI-compatible endpoint that speaks the WorldEdge request (scope doc Appendix B, fields in interface.py).

    python -m fastfill.serve --model outputs/ff-v1-merged --port 8001
    # EmbodiedGen side: WORLDEDGE_FASTFILL_URL=http://<host>:8001/v1 WORLDEDGE_FASTFILL_MODEL=fastfill

The last user message is the request JSON. The reply's content is placements_from_text's JSON
({"placements": [{"id", "position_m", "yaw_rad", "support_parent", "support_surface"?}], "raw_placements": [...],
"validation": {"raw", "repaired"}, "error": null, "unsupported": [...]}). HTTP 422 and no layout when the request is
rejected, the answer does not parse (a missing or extra object included), or the repaired layout still fails the
canonical checks: FastFill places every object validly or fails back to the caller, it never drops one.
Greedy decoding, one request at a time.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastfill.interface import placements_from_text, request_to_room
from fastfill.scene import messages, prompt_text


def answer(req, generate):
    """Request dict -> (HTTP status, body). generate(model messages) -> model text."""
    try:
        room, ctx = request_to_room(req)
    except ValueError as e:
        return 422, {"error": {"message": str(e), "type": "invalid_request_error"}}
    out = placements_from_text(generate(messages(room, room["constraints"], with_target=False)), room, ctx)
    if out["error"]:
        return 422, {"error": {"message": f"model answer rejected: {out['error']}", "type": "invalid_layout"}}
    if not out["validation"]["repaired"]["ok"]:     # every object must be placed validly; else the caller decides
        return 422, {"error": {"message": "layout fails validation even after repair", "type": "invalid_layout",
                               "raw_placements": out["raw_placements"], "validation": out["validation"]}}
    return 200, out


def make_handler(generate, name):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                return self.reply(404, {"error": {"message": "POST .../v1/chat/completions"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                req = json.loads(body["messages"][-1]["content"])
            except (ValueError, KeyError, IndexError, TypeError) as e:
                return self.reply(400, {"error": {"message": f"bad body: {e!r}", "type": "invalid_request_error"}})
            try:
                status, out = answer(req, generate)
            except Exception as e:      # a bug or a dead model: say so instead of dropping the connection
                return self.reply(500, {"error": {"message": repr(e), "type": "server_error"}})
            if status == 200:
                out = {"id": f"fastfill-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                       "model": name, "choices": [{"index": 0, "finish_reason": "stop",
                                                   "message": {"role": "assistant", "content": json.dumps(out)}}],
                       "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            self.reply(status, out)

        def reply(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir")
    ap.add_argument("--name", default="fastfill", help="model name echoed in replies")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=8192)
    tok, sp = llm.get_tokenizer(), SamplingParams(temperature=0, max_tokens=a.max_new_tokens)
    generate = lambda msgs: llm.generate([prompt_text(tok, msgs)], sp, use_tqdm=False)[0].outputs[0].text
    print(f"FastFill endpoint: http://{a.host}:{a.port}/v1  model {a.model}", flush=True)
    HTTPServer((a.host, a.port), make_handler(generate, a.name)).serve_forever()


if __name__ == "__main__":
    main()
