"""Real loopback HTTP requests: bounded input reads and serialized model calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time

import pytest

from fastfill import serve
from fastfill.tests.test_runtime_repairs import model_text, request


@contextmanager
def running_server(monkeypatch, generate):
    monkeypatch.setattr(serve, 'REQUEST_READ_TIMEOUT', .15, raising=False)
    ready = threading.Event()
    base = serve.make_handler(generate, 'test', '/test/model')

    class Handler(base):
        def setup(self):
            super().setup()
            ready.set()

        def log_message(self, *args):
            pass

    server = serve.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield server, ready
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def post(server):
    body = json.dumps({'messages': [{'role': 'user', 'content': json.dumps(request())}]}).encode()
    connection = http.client.HTTPConnection(*server.server_address, timeout=.8)
    try:
        connection.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize('phase', ['headers', 'body'])
@pytest.mark.parametrize('drip', [False, True])
def test_incomplete_request_cannot_block_next_request(monkeypatch, phase, drip):
    with running_server(monkeypatch, lambda _: model_text()) as (server, ready):
        first = socket.create_connection(server.server_address, timeout=2)
        stop = threading.Event()
        writer = None
        try:
            prefix = b'POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\n'
            first.sendall(prefix + (b'X-Slow: ' if phase == 'headers' else b'Content-Length: 100000\r\n\r\n{'))
            assert ready.wait(1)
            if drip:
                def keep_sending():
                    while not stop.wait(.03):
                        try:
                            first.sendall(b' ')
                        except OSError:
                            return
                writer = threading.Thread(target=keep_sending, daemon=True)
                writer.start()
            status, response = post(server)
            assert status == 200
            assert response['model'] == 'test'
            if phase == 'body':
                assert b'408' in first.recv(4096).split(b'\r\n', 1)[0]
        finally:
            stop.set()
            first.close()
            if writer:
                writer.join(1)


def test_input_deadline_does_not_limit_generation_or_parallelize_model(monkeypatch):
    active = peak = calls = 0

    def generate(_):
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        time.sleep(.25)  # Longer than the input deadline; the complete request remains valid.
        calls += 1
        active -= 1
        return model_text()

    with running_server(monkeypatch, generate) as (server, _):
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: post(server), range(2)))
    assert [r[0] for r in results] == [200, 200]
    assert (calls, peak) == (2, 1)
