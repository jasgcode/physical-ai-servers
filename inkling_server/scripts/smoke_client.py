"""Smoke-test a running inkling_server: text, image and streaming requests.

    pixi run smoke                                  # http://127.0.0.1:8000
    pixi run smoke --url https://<host>.ts.net

Proves plumbing and speed. It says nothing about answer quality.
"""

import argparse
import base64
import io
import json
import sys
import time

import httpx
from PIL import Image


def _image_url() -> str:
    image = Image.new("RGB", (256, 256), (255, 255, 255))
    image.paste((220, 30, 30), (64, 64, 192, 192))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _chat(client: httpx.Client, model: str, content, max_tokens: int) -> tuple[str, float]:
    started = time.perf_counter()
    r = client.post("/v1/chat/completions", json={
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }).raise_for_status().json()
    return (r["choices"][0]["message"]["content"] or "").strip(), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=None, help="default: first key in .env")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    key = args.key
    if key is None:
        from server.config import settings

        key = settings.api_key_list[0] if settings.api_key_list else ""

    ok = True
    with httpx.Client(base_url=args.url, timeout=600,
                      headers={"Authorization": f"Bearer {key}"}) as client:
        model = client.get("/v1/models").raise_for_status().json()["data"][0]
        print(f"[smoke] model: {json.dumps(model)}")

        answer, took = _chat(client, model["id"],
                             "What is 17 * 23? Answer with the number only.", args.max_tokens)
        print(f"[smoke] text  {took:5.2f}s: {answer[:200]!r}")
        ok &= "391" in answer

        answer, took = _chat(client, model["id"], [
            {"type": "image_url", "image_url": {"url": _image_url()}},
            {"type": "text", "text": "What color is the square in the middle? One word."},
        ], args.max_tokens)
        print(f"[smoke] image {took:5.2f}s: {answer[:200]!r}")
        ok &= "red" in answer.lower()

        started = time.perf_counter()
        first = None
        chunks = 0
        with client.stream("POST", "/v1/chat/completions", json={
            "model": model["id"], "max_tokens": 512, "stream": True,
            "messages": [{"role": "user",
                          "content": "Explain how a robot arm grasps a cup, in about 200 words."}],
        }) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices") or [{}]
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = first or time.perf_counter()
                    chunks += 1
        done = time.perf_counter()
        if first is None:
            print("[smoke] FAIL: stream produced no tokens")
            return 1
        print(f"[smoke] stream: first token {(first - started) * 1000:.0f} ms, "
              f"{chunks} chunks in {done - started:.2f}s")

    print("[smoke] OK" if ok else "[smoke] FAIL: an answer was wrong (expected 391 and red)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
