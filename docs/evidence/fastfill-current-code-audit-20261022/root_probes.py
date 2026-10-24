"""Read-only FastFill audit repros; local loopback and synthetic rooms only."""
import copy
import json
import socket
import threading
from http.server import HTTPServer

from fastfill.evaluate import score
from fastfill.scene import messages
from fastfill.serve import make_handler
from fastfill.tools.qa_report import _train_metrics


def qa_probe():
    room = {
        "uid": "audit-qa", "source": "synthetic", "group": "audit",
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "boundary_type": "polygon", "height": 3,
        "objects": [
            {"id": "chair_1", "category": "chair", "size": [.5, .5, .9],
             "pos": [.25, .22, 0], "yaw": 0, "parent": None},
            {"id": "box_1", "category": "box", "size": [.06, .06, .3],
             "pos": [.60, .03, 0], "yaw": 0, "parent": None},
        ],
        "constraints": [["faces", "chair_1", "box_1"]],
    }
    row = {"messages": messages(room, room["constraints"])}
    qa = _train_metrics([row])
    runtime = score(copy.deepcopy(room), None)
    result = {
        "qa_inside": qa["train reference inside walls within 1 mm"],
        "qa_constraint_reference_validation": qa["constraint_reference_validation"],
        "runtime_inside_raw": runtime["inside_1mm"],
        "runtime_inside_repaired": runtime["inside_1mm_repaired"],
        "runtime_constraints_repaired": runtime["constraints_repaired"],
        "runtime_accepted": runtime["accepted"],
    }
    assert result["qa_inside"]["repaired"] == "1/1", result
    assert result["runtime_inside_repaired"] is False, result
    return result


def http_probe():
    entered = threading.Event()
    base = make_handler(lambda _: "unused", "audit", "audit")

    class Handler(base):
        def do_POST(self):
            entered.set()
            return super().do_POST()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    first = second = None
    try:
        first = socket.create_connection(server.server_address, timeout=2)
        first.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Length: 4\r\n\r\n{")
        assert entered.wait(2)
        second = socket.create_connection(server.server_address, timeout=2)
        second.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
        second.settimeout(.3)
        blocked = False
        try:
            second.recv(1024)
        except socket.timeout:
            blocked = True
        first.sendall(b"}  ")
        first.recv(4096)
        second.settimeout(2)
        response = second.recv(4096).split(b"\r\n", 1)[0].decode()
        result = {"second_request_blocked_until_first_finished": blocked,
                  "second_status_after_unblocking": response,
                  "handler_socket_timeout": Handler.timeout if hasattr(Handler, "timeout") else None}
        assert blocked and "400" in response, result
        return result
    finally:
        if first:
            first.close()
        if second:
            second.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


if __name__ == "__main__":
    print(json.dumps({"qa_constraint_omission": qa_probe(), "http_stall": http_probe()}, indent=2))
