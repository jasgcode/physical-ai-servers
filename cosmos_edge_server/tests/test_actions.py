"""HTTP contract tests: auth, validation, response shape. Model-agnostic —
every request is built from the loaded policy's spec."""

import json

import numpy as np

from server import inference

from .conftest import AUTH, build_request, png

URL = "/v1/actions/generations"


def _post(client, data, files, headers=AUTH):
    return client.post(URL, data=data, files=files, headers=headers)


def test_health_and_models_need_no_key(client):
    assert client.get("/health").json() == {"status": "ok"}
    model = client.get("/v1/models").json()["data"][0]
    assert model["capabilities"] == ["action-generation"]
    assert model["policy"] == inference.spec().model_dump()


def test_missing_bearer_token_is_401(client):
    data, files = build_request(inference.spec())
    assert _post(client, data, files, headers={}).status_code == 401


def test_wrong_key_is_401(client):
    data, files = build_request(inference.spec())
    response = _post(client, data, files, headers={"Authorization": "Bearer sk-nope"})
    assert response.status_code == 401


def test_valid_request_returns_one_chunk_per_action_key(client):
    spec = inference.spec()
    data, files = build_request(spec)
    response = _post(client, data, files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action_horizon"] == spec.action_horizon
    assert set(body["actions"]) == set(spec.action)
    for key, dim in spec.action.items():
        chunk = np.asarray(body["actions"][key])
        assert chunk.shape[0] == spec.action_horizon
        if dim is not None:
            assert chunk.shape[1] == dim
        assert np.isfinite(chunk).all()
    assert body["timing_ms"]["inference"] >= 0


def test_missing_camera_is_400(client):
    spec = inference.spec()
    data, files = build_request(spec)
    dropped = f"image.{spec.cameras[0]}"
    files = [f for f in files if f[0] != dropped]
    response = _post(client, data, files)
    assert response.status_code == 400
    assert dropped in response.json()["detail"]


def test_unknown_camera_is_400(client):
    data, files = build_request(inference.spec())
    files.append(("image.not_a_camera", ("x.png", png(), "image/png")))
    assert _post(client, data, files).status_code == 400


def test_corrupt_image_is_400(client):
    spec = inference.spec()
    data, files = build_request(spec)
    files[0] = (files[0][0], ("x.png", b"not an image", "image/png"))
    assert _post(client, data, files).status_code == 400


def test_empty_instruction_is_400(client):
    data, files = build_request(inference.spec(), instruction="   ")
    assert _post(client, data, files).status_code == 400


def test_state_must_be_json_with_exact_keys(client):
    spec = inference.spec()
    data, files = build_request(spec, state="{not json")
    assert _post(client, data, files).status_code == 400
    data, files = build_request(spec, state=json.dumps({"bogus": [0.0]}))
    assert _post(client, data, files).status_code == 400


def test_non_finite_state_is_400(client):
    spec = inference.spec()
    data, files = build_request(spec)
    state = json.loads(data["state"])
    key = next(iter(state))
    state[key][0][0] = float("nan")
    data["state"] = json.dumps(state)
    assert _post(client, data, files).status_code == 400


def test_wrong_state_width_is_400_when_spec_declares_one(client):
    spec = inference.spec()
    declared = {k: d for k, d in spec.state.items() if d is not None}
    if not declared:
        return
    key, dim = next(iter(declared.items()))
    data, files = build_request(spec)
    state = json.loads(data["state"])
    state[key] = [[0.0] * (dim + 1) for _ in range(spec.state_horizon)]
    data["state"] = json.dumps(state)
    response = _post(client, data, files)
    assert response.status_code == 400
    assert key in response.json()["detail"]


def test_non_finite_action_is_500_not_a_command(client, monkeypatch):
    spec = inference.spec()

    async def bad_predict(obs):
        return {k: np.full((spec.action_horizon, d or 7), np.nan) for k, d in spec.action.items()}, {}

    monkeypatch.setattr(inference, "predict", bad_predict)
    data, files = build_request(spec)
    assert _post(client, data, files).status_code == 500


def test_reset_requires_key(client):
    assert client.post("/v1/actions/reset").status_code == 401
    assert client.post("/v1/actions/reset", headers=AUTH).json() == {"status": "ok"}
