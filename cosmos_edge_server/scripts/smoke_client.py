"""Smoke a RUNNING server end to end, and the reference client.

    pixi run smoke                                   # http://127.0.0.1:8000
    pixi run smoke --url https://<host>.ts.net -n 20

Reads the layout from /v1/models, sends synthetic observations (gray frames,
zero state), and checks every returned chunk has the declared shape and is
finite. Zero state is not a meaningful robot pose: this proves the plumbing
and measures latency, it says nothing about policy quality.
"""

import argparse
import io
import json
import statistics
import sys
import time

import httpx
from PIL import Image


def _frame(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(127, 127, 127)).save(buf, format="JPEG")
    return buf.getvalue()


def build_request(policy: dict, instruction: str, width: int, height: int):
    files = [
        (f"image.{camera}", (f"{camera}.jpg", _frame(width, height), "image/jpeg"))
        for camera in policy["cameras"]
        for _ in range(policy["video_horizon"])
    ]
    unknown = [k for k, d in policy["state"].items() if d is None]
    if unknown:
        print(f"[smoke] WARN: no declared width for state {unknown}; guessing 7")
    state = {
        key: [[0.0] * (dim or 7) for _ in range(policy["state_horizon"])]
        for key, dim in policy["state"].items()
    }
    return {"instruction": instruction, "state": json.dumps(state)}, files


def check(policy: dict, body: dict) -> list[str]:
    problems = []
    for key, dim in policy["action"].items():
        chunk = body["actions"].get(key)
        if chunk is None:
            problems.append(f"missing action '{key}'")
            continue
        if len(chunk) != policy["action_horizon"]:
            problems.append(f"{key}: {len(chunk)} steps, expected {policy['action_horizon']}")
        if dim is not None and any(len(row) != dim for row in chunk):
            problems.append(f"{key}: row width != {dim}")
        if any(v != v or abs(v) == float("inf") for row in chunk for v in row):
            problems.append(f"{key}: non-finite values")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=None, help="default: first key in .env")
    parser.add_argument("-n", type=int, default=5, help="timed requests after one discarded warmup")
    parser.add_argument("--instruction", default="pick up the block")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    key = args.key
    if key is None:
        from server.config import settings

        key = settings.api_key_list[0] if settings.api_key_list else ""

    # One client = one kept-alive connection, which is how a control loop
    # should call this: a fresh TLS handshake per request costs more than a
    # small policy's forward pass.
    with httpx.Client(base_url=args.url, timeout=120.0,
                      headers={"Authorization": f"Bearer {key}"}) as client:
        model = client.get("/v1/models").raise_for_status().json()["data"][0]
        policy = model["policy"]
        print(f"[smoke] {model['id']}: {json.dumps(policy)}")

        latencies, server_ms = [], []
        for i in range(args.n + 1):
            data, files = build_request(policy, args.instruction, args.width, args.height)
            start = time.perf_counter()
            response = client.post("/v1/actions/generations", data=data, files=files)
            elapsed = (time.perf_counter() - start) * 1000.0
            if response.status_code != 200:
                print(f"[smoke] FAIL: HTTP {response.status_code}: {response.text[:500]}")
                return 1
            body = response.json()
            problems = check(policy, body)
            if problems:
                print(f"[smoke] FAIL: {problems}")
                return 1
            if i > 0:
                latencies.append(elapsed)
                server_ms.append(body["timing_ms"].get("inference", 0.0))

    if latencies:
        latencies.sort()
        p90 = latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
        print(f"[smoke] OK: {len(latencies)} requests, end-to-end p50={statistics.median(latencies):.1f}ms "
              f"p90={p90:.1f}ms, server inference p50={statistics.median(server_ms):.1f}ms")
    else:
        print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
