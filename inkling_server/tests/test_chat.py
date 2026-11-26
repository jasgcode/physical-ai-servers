import json

from server import inference
from server.config import settings
from tests.conftest import AUTH


def _chat(**extra):
    return {"model": "anything", "messages": [{"role": "user", "content": "hi"}], **extra}


def test_health_and_models_need_no_key(client):
    assert client.get("/health").json() == {"status": "ok"}
    model = client.get("/v1/models").json()["data"][0]
    assert model["id"] == settings.served_model_name
    assert "vision" in model["capabilities"]


def test_chat_requires_a_key(client, engine):
    assert client.post("/v1/chat/completions", json=_chat()).status_code == 401
    bad = {"Authorization": "Bearer sk-INKLING-wrong"}
    assert client.post("/v1/chat/completions", json=_chat(), headers=bad).status_code == 401
    assert engine.requests == []


def test_chat_is_forwarded_with_the_served_model_name(client, engine):
    r = client.post("/v1/chat/completions", json=_chat(temperature=0.2), headers=AUTH)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hello"
    sent = engine.requests[-1]
    assert sent["model"] == settings.served_model_name
    assert sent["temperature"] == 0.2
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_stream_is_relayed_as_server_sent_events(client, engine):
    r = client.post("/v1/chat/completions", json=_chat(stream=True), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    data = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]"
    assert "".join(json.loads(d)["choices"][0]["delta"]["content"] for d in data[:-1]) == "hello"


def test_bad_bodies_are_client_errors(client, engine, monkeypatch):
    assert client.post("/v1/chat/completions", content=b"{nope", headers=AUTH).status_code == 400
    assert client.post("/v1/chat/completions", json=[1, 2], headers=AUTH).status_code == 400
    monkeypatch.setattr(settings, "max_request_mb", 0)
    assert client.post("/v1/chat/completions", json=_chat(), headers=AUTH).status_code == 413
    assert engine.requests == []


def test_engine_down_is_503(client, engine):
    engine.down = True
    assert client.post("/v1/chat/completions", json=_chat(), headers=AUTH).status_code == 503


def test_action_routes_are_501(client):
    assert client.post("/v1/actions/generations", headers=AUTH).status_code == 501
    assert client.post("/v1/actions/reset", headers=AUTH).status_code == 501


def test_engine_command_matches_the_supported_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model", str(tmp_path))
    monkeypatch.setattr(settings, "engine_extra_args", "--enforce-eager")
    cmd = inference.engine_command()
    assert cmd[1:3] == ["serve", str(tmp_path)]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == str(settings.engine_port)
    assert cmd[cmd.index("--reasoning-parser") + 1] == "inkling"
    assert json.loads(cmd[cmd.index("--speculative-config") + 1])["method"] == "dspark"
    assert cmd[-1] == "--enforce-eager"
    # A local snapshot directory is served without touching the network.
    assert inference.engine_env()["HF_HUB_OFFLINE"] == "1"


def test_hub_id_is_not_forced_offline(monkeypatch):
    monkeypatch.setattr(settings, "model", "thinkingmachines/Inkling-2-Mini")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert "HF_HUB_OFFLINE" not in inference.engine_env()


def test_engine_that_dies_during_startup_fails_loud(monkeypatch, tmp_path):
    class DeadEngine:
        pid = 0

        def poll(self):
            return 1

    monkeypatch.setattr(inference, "engine_log_path", lambda: tmp_path / "vllm.log")
    monkeypatch.setattr(inference.subprocess, "Popen", lambda *a, **k: DeadEngine())
    monkeypatch.setattr(settings, "engine_port", 1)  # nothing listens here
    try:
        inference.init_policy()
    except RuntimeError as error:
        assert "exited with code 1" in str(error)
    else:
        raise AssertionError("init_policy returned with a dead engine")
    assert inference._engine is None
