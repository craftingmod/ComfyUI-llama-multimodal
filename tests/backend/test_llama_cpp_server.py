import json
import math
from threading import RLock

import pytest

from backend.backends.llama_cpp_server import (
    LlamaCppServerSession,
    OwnedLlamaCppServerSession,
    list_server_models,
    parse_models_response,
)
from backend.core import BackendError, MediaBundle
from backend.core.media import MediaItem
from backend.llama_cpp.llama_cpp_session_cleanup import close_tracked_sessions


def test_server_session_parses_models_generates_multimodal_and_retries_unload(
    monkeypatch,
):
    calls = []
    completion_payloads = []
    unload_attempts = 0
    import backend.llama_cpp.llama_cpp_session_cleanup as cleanup_module

    tracked = set()
    monkeypatch.setattr(cleanup_module, "_sessions", tracked)
    monkeypatch.setattr(cleanup_module, "_sessions_lock", RLock())

    def transport(url, method, body, timeout):
        nonlocal unload_attempts
        calls.append((url, method, body, timeout))
        if url.endswith("/health"):
            assert method == "GET"
            if "unhealthy" in url:
                return 200, b'{"status":"loading"}'
            return 200, b'{"status":"ok"}'
        if url.endswith("/models"):
            return 200, b'{"data":[{"id":"model-a"},{"id":"model-b"}]}'
        if url.endswith("/v1/chat/completions"):
            payload = json.loads(body)
            completion_payloads.append(payload)
            parts = payload["messages"][-1]["content"]
            assert payload["model"] == "model-a"
            assert payload["stream"] is False
            if "cache_prompt" not in payload:
                assert [part["type"] for part in parts] == [
                    "text",
                    "image_url",
                    "input_audio",
                    "input_video",
                ]
            else:
                assert [part["type"] for part in parts] == [
                    "image_url",
                    "input_audio",
                    "input_video",
                    "text",
                ]
                assert isinstance(payload["cache_prompt"], bool)
            return 200, json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "answer",
                                "reasoning_content": "thought",
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                    "timings": {"predicted_n": 1},
                }
            ).encode()
        assert url.endswith("/models/unload")
        assert method == "POST"
        assert json.loads(body) == {"model": "model-a"}
        unload_attempts += 1
        if unload_attempts == 1:
            return 503, b'{"error":{"message":"server busy"}}'
        return 200, b'{"success":true}'

    assert list_server_models(url="http://localhost:8080", transport=transport) == [
        "model-a",
        "model-b",
    ]
    with pytest.raises(BackendError):
        parse_models_response(b'{"data":[{"id":4}]}')

    with pytest.raises(BackendError, match="health check failed"):
        LlamaCppServerSession(
            url="http://unhealthy:8080",
            model="model-a",
            transport=transport,
        )
    assert tracked == set()

    session = LlamaCppServerSession(
        url="http://localhost:8080",
        model="model-a",
        transport=transport,
    )
    media = MediaBundle(
        (
            MediaItem("image", 0, "image/png", b"png", {}),
            MediaItem("audio", 1, "audio/wav", b"wav", {}),
            MediaItem("video", 2, "video/mp4", b"mp4", {}),
        )
    )
    result = session.generate(
        system="system",
        prompt="prompt",
        media=media,
        max_tokens=8,
        seed=-1,
        stop="",
    )
    session.generate(
        system="system",
        prompt="cached prompt",
        media=media,
        max_tokens=8,
        seed=-1,
        stop="",
        media_before_prompt=True,
        reuse_kv_cache=True,
    )
    session.generate(
        system="system",
        prompt="uncached prompt",
        media=media,
        max_tokens=8,
        seed=-1,
        stop="",
        media_before_prompt=True,
        reuse_kv_cache=False,
    )
    assert "cache_prompt" not in completion_payloads[0]
    assert completion_payloads[1]["cache_prompt"] is True
    assert completion_payloads[2]["cache_prompt"] is False
    assert (result.response, result.thinking) == ("answer", "thought")
    assert result.metrics["server_timings"] == {"predicted_n": 1}
    assert result.media_diagnostics["mtmd"]["verification"] == "unverified_remote"

    assert session in tracked
    close_tracked_sessions()
    assert session.closed is False
    assert session in tracked
    close_tracked_sessions()
    assert session.closed is True
    session.close()
    assert unload_attempts == 2
    assert session not in tracked


def test_server_decide_cache_flag_and_media_first_order():
    pytest.importorskip("makoto_decision")
    applied_messages = []
    completion_requests = []
    server_marker = "<__media_server_random_marker__>"
    vocab_requests = 0

    def transport(url, method, body, timeout):
        nonlocal vocab_requests
        if url.endswith("/health"):
            return 200, b'{"status":"ok"}'
        if url.endswith("/v1/models"):
            vocab_requests += 1
            return 200, json.dumps(
                {"data": [{"id": "model-a", "meta": {"n_vocab": 128}}]}
            ).encode()
        if url.endswith("/tokenize"):
            token = 11 if json.loads(body)["content"] == "A" else 12
            return 200, json.dumps({"tokens": [token]}).encode()
        if url.endswith("/apply-template"):
            messages = json.loads(body)["messages"]
            applied_messages.append(messages)
            assert messages[0]["role"] == "system"
            assert "Return exactly one choice label" in messages[0]["content"]
            parts = messages[-1]["content"]
            prompt = "".join(
                part["text"] if part["type"] == "text" else server_marker
                for part in parts
            )
            return 200, json.dumps({"prompt": prompt}).encode()
        if url.endswith("/completion"):
            request = json.loads(body)
            prompt = request["prompt"]
            assert "<__media__>" not in prompt["prompt_string"]
            assert prompt["prompt_string"].count(server_marker) == len(
                prompt["multimodal_data"]
            )
            completion_requests.append(request)
            return 200, json.dumps(
                {
                    "content": "A",
                    "completion_probabilities": [
                        {
                            "id": 11,
                            "token": "A",
                            "logprob": math.log(0.3),
                            "top_logprobs": [
                                {"id": 11, "token": "A", "logprob": math.log(0.3)},
                                {
                                    "id": 100,
                                    "token": "<|channel|>",
                                    "logprob": math.log(1.519e-5),
                                },
                                {"id": 12, "token": "B", "logprob": math.log(0.1)},
                            ],
                        }
                    ],
                }
            ).encode()
        assert url.endswith("/models/unload")
        return 200, b'{"success":true}'

    session = LlamaCppServerSession(
        url="http://localhost:8080",
        model="model-a",
        transport=transport,
    )
    media = MediaBundle(
        (
            MediaItem("image", 0, "image/png", b"image", {}),
            MediaItem("audio", 1, "audio/wav", b"audio", {}),
            MediaItem("video", 2, "video/mp4", b"video", {}),
        )
    )
    model_profile = {
        "temperature": 0.7,
        "top_k": 1,
        "top_p": 0.1,
        "min_p": 0.5,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "recommended_reasoning_mode": "auto",
    }
    results = []
    try:
        for index, (reuse_kv_cache, media_before_prompt) in enumerate(
            ((None, False), (True, True), (False, True))
        ):
            results.append(
                session.decide(
                    question=f"Question {index}",
                    context="Shared context",
                    answers=["first", "second"],
                    media=media,
                    model_profile=model_profile,
                    reuse_kv_cache=reuse_kv_cache,
                    media_before_prompt=media_before_prompt,
                )
            )
    finally:
        session.close()

    assert vocab_requests == 1
    assert "cache_prompt" not in completion_requests[0]
    assert completion_requests[1]["cache_prompt"] is True
    assert completion_requests[2]["cache_prompt"] is False
    for request in completion_requests:
        assert request["n_probs"] == 128
        assert request["post_sampling_probs"] is False
        assert request["backend_sampling"] is False
        assert request["top_k"] == 1
        assert request["top_p"] == 0.1
        assert request["min_p"] == 0.5
    assert results[0].selected == "first"
    assert results[0].probabilities == pytest.approx({"first": 0.75, "second": 0.25})
    assert completion_requests[1]["prompt"]["multimodal_data"] == [
        "aW1hZ2U=",
        "YXVkaW8=",
        "dmlkZW8=",
    ]
    assert [part["type"] for part in applied_messages[0][-1]["content"]] == [
        "text",
        "image_url",
        "input_audio",
        "input_video",
        "text",
    ]
    for messages in applied_messages[1:]:
        assert [part["type"] for part in messages[-1]["content"]] == [
            "image_url",
            "input_audio",
            "input_video",
            "text",
            "text",
        ]
    for request in completion_requests:
        prompt = request["prompt"]["prompt_string"]
        assert prompt.index("Shared context") < prompt.index("Question")
    assert completion_requests[0]["prompt"]["prompt_string"].index(
        "Shared context"
    ) < completion_requests[0]["prompt"]["prompt_string"].index(server_marker)
    assert completion_requests[1]["prompt"]["prompt_string"].index(
        server_marker
    ) < completion_requests[1]["prompt"]["prompt_string"].index("Shared context")


def test_server_decide_requires_probability_for_every_choice():
    pytest.importorskip("makoto_decision")

    def transport(url, method, body, timeout):
        if url.endswith("/health"):
            return 200, b'{"status":"ok"}'
        if url.endswith("/v1/models"):
            return 200, b'{"data":[{"id":"model-a","meta":{"n_vocab":128}}]}'
        if url.endswith("/tokenize"):
            token = 11 if json.loads(body)["content"] == "A" else 12
            return 200, json.dumps({"tokens": [token]}).encode()
        if url.endswith("/apply-template"):
            return 200, b'{"prompt":"formatted"}'
        if url.endswith("/completion"):
            return 200, json.dumps(
                {
                    "content": "A",
                    "probs": [
                        {
                            "id": 11,
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"id": 11, "token": "A", "logprob": -0.2},
                                {"id": 100, "token": "<|channel|>", "logprob": -3.0},
                            ],
                        }
                    ],
                }
            ).encode()
        return 200, b'{"status":"ok","success":true}'

    session = LlamaCppServerSession(
        url="http://localhost:8080", model="model-a", transport=transport
    )
    try:
        with pytest.raises(BackendError) as caught:
            session.decide(
                question="Private question",
                context="Private context",
                answers=["first", "second"],
            )
        message = str(caught.value)
        assert "generated='A'" in message
        assert "missing_targets=['B']" in message
        assert "expected_tokens={'A': 11, 'B': 12}" in message
        assert "Private" not in message
    finally:
        session.close()


def test_server_decide_rejects_all_zero_probability_choice_tokens():
    pytest.importorskip("makoto_decision")

    zero_logprob = -3.4028234663852886e38

    def transport(url, method, body, timeout):
        if url.endswith("/health"):
            return 200, b'{"status":"ok"}'
        if url.endswith("/v1/models"):
            return 200, b'{"data":[{"id":"model-a","meta":{"n_vocab":128}}]}'
        if url.endswith("/tokenize"):
            token = 11 if json.loads(body)["content"] == "A" else 12
            return 200, json.dumps({"tokens": [token]}).encode()
        if url.endswith("/apply-template"):
            return 200, b'{"prompt":"formatted"}'
        if url.endswith("/completion"):
            return 200, json.dumps(
                {
                    "content": "A",
                    "probs": [
                        {
                            "id": 11,
                            "logprob": zero_logprob,
                            "top_logprobs": [
                                {"id": 11, "token": "A", "logprob": zero_logprob},
                                {"id": 12, "token": "B", "logprob": zero_logprob},
                            ],
                        }
                    ],
                }
            ).encode()
        return 200, b'{"status":"ok","success":true}'

    session = LlamaCppServerSession(
        url="http://localhost:8080", model="model-a", transport=transport
    )
    try:
        with pytest.raises(BackendError, match="zero probability mass for every"):
            session.decide(
                question="Private question",
                context="Private context",
                answers=["first", "second"],
            )
    finally:
        session.close()


def test_owned_server_session_retries_shutdown_and_cleanup_idempotently(monkeypatch):
    import backend.llama_cpp.llama_cpp_session_cleanup as cleanup_module

    tracked = set()
    monkeypatch.setattr(cleanup_module, "_sessions", tracked)
    monkeypatch.setattr(cleanup_module, "_sessions_lock", RLock())

    events = []
    requests = []

    class Process:
        close_calls = 0

        def close(self):
            self.close_calls += 1
            events.append("process")
            if self.close_calls == 1:
                raise RuntimeError("shutdown failed")

    process = Process()
    cleanup_calls = 0

    def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        events.append("cleanup")
        if cleanup_calls == 1:
            raise RuntimeError("cleanup failed")

    def transport(url, method, body, timeout):
        requests.append((url, method))
        assert url.endswith("/health")
        return 200, b'{"status":"ok"}'

    session = OwnedLlamaCppServerSession(
        url="http://localhost:8080",
        model="model-a",
        process=process,
        cleanup=cleanup,
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="shutdown failed"):
        session.close()
    assert session in tracked
    assert cleanup_calls == 0

    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()
    assert session in tracked
    assert session.closed is False

    session.close()
    assert session.closed is True
    assert session not in tracked
    session.close()

    assert process.close_calls == 2
    assert cleanup_calls == 2
    assert events == ["process", "process", "cleanup", "cleanup"]
    assert requests == [("http://localhost:8080/health", "GET")]


def test_owned_server_session_closes_at_prompt_end(monkeypatch):
    import backend.llama_cpp.llama_cpp_session_cleanup as cleanup_module

    tracked = set()
    monkeypatch.setattr(cleanup_module, "_sessions", tracked)
    monkeypatch.setattr(cleanup_module, "_sessions_lock", RLock())

    class Process:
        closed = False

        def close(self):
            self.closed = True

    process = Process()
    session = OwnedLlamaCppServerSession(
        url="http://localhost:8080",
        model="model-a",
        process=process,
        transport=lambda *_args: (200, b'{"status":"ok"}'),
    )

    assert session in tracked
    close_tracked_sessions()
    assert process.closed is True
    assert session.closed is True
    assert session not in tracked
