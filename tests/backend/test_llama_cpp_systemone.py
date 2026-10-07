import json
import math

import pytest

from backend.backends.llama_cpp_server import (
    LlamaCppServerSession,
    OwnedLlamaCppServerSession,
    normalize_systemone_question,
)
from backend.core import BackendError, InputNormalizationError, MediaBundle
from backend.core.media import MediaItem
from tests.backend.test_extension_registration import import_llama_cpp_decision_nodes


def _transport_for(response, *, status=200):
    calls = []

    def transport(url, method, body, timeout):
        if url.endswith("/health"):
            return 200, b'{"status":"ok"}'
        if url.endswith("/v1/systemone"):
            request = json.loads(body)
            calls.append((url, method, request, timeout))
            response_body = (
                response
                if isinstance(response, bytes)
                else json.dumps(response, allow_nan=True).encode("utf-8")
            )
            return status, response_body
        if url.endswith("/models/unload"):
            calls.append((url, method, json.loads(body), timeout))
            return 200, b'{"success":true}'
        raise AssertionError(f"unexpected llama.cpp request: {method} {url}")

    return transport, calls


def _new_session(
    response, *, status=200, session_type=LlamaCppServerSession, **kwargs
):
    transport, calls = _transport_for(response, status=status)
    session = session_type(
        url="http://localhost:8080",
        model="model-a",
        transport=transport,
        **kwargs,
    )
    return session, calls


def _response(answer, *, usage=None):
    return {
        "model": "model-a",
        "answers": {"question": answer},
        "usage": usage or {"input_tokens": 12, "output_tokens": 0},
    }


def test_systemone_question_builders_and_legacy_choice_payload(monkeypatch):
    nodes = import_llama_cpp_decision_nodes(monkeypatch)

    choice = nodes.LlamaCppBuildQuestionNode.execute(
        ["Pick one"], ["first", "second"], ["choice"]
    )[0]
    score = nodes.LlamaCppBuildQuestionNode.execute(
        ["Rate it"], ["low", "middle", "high"], ["score"]
    )[0]

    assert choice == {
        "type": "choice",
        "question": "Pick one",
        "answer": ["first", "second"],
    }
    assert score == {
        "type": "score",
        "question": "Rate it",
        "answer": ["low", "middle", "high"],
    }
    assert nodes.LlamaCppBuildNoulNode.execute("Is it ready?")[0] == {
        "type": "noul",
        "question": "Is it ready?",
    }
    assert normalize_systemone_question(
        {"question": "Legacy?", "answer": ["yes", "no"]}
    ) == {
        "type": "choice",
        "question": "Legacy?",
        "answer": ["yes", "no"],
    }
    assert len(
        normalize_systemone_question(
            {
                "type": "choice",
                "question": "Pick",
                "answer": [f"option {index}" for index in range(26)],
            }
        )["answer"]
    ) == 26
    assert len(
        normalize_systemone_question(
            {
                "type": "score",
                "question": "Rate",
                "answer": [f"level {index}" for index in range(10)],
            }
        )["answer"]
    ) == 10

    for values in (
        {"type": "choice", "question": "Pick", "answer": ["one"]},
        {"type": "score", "question": "Rate", "answer": ["same", "same"]},
        {"type": "score", "question": "Rate", "answer": ["one"] * 11},
        {"type": "choice", "question": "Pick", "answer": ["one", "two"] * 14},
        {"type": [], "question": "Pick", "answer": ["one", "two"]},
    ):
        with pytest.raises(InputNormalizationError):
            normalize_systemone_question(values)

    with pytest.raises(InputNormalizationError, match="exactly one STRING"):
        nodes.LlamaCppBuildQuestionNode.execute(
            ["Pick", "another"], ["one", "two"], ["choice"]
        )
    with pytest.raises(InputNormalizationError, match="flat ComfyUI STRING list"):
        nodes.LlamaCppBuildQuestionNode.execute(
            ["Pick"], [["one"], ["two"]], ["choice"]
        )
    with pytest.raises(InputNormalizationError, match="exactly one COMBO"):
        nodes.LlamaCppBuildQuestionNode.execute(
            ["Pick"], ["one", "two"], ["choice", "score"]
        )


def test_systemone_choice_request_order_image_and_diagnostics():
    raw_answer = {
        "type": "choice",
        "choice": "second",
        "probabilities": {"second": 0.7, "first": 0.3},
        "confidence": 0.4,
        "extra": "retained",
    }
    session, calls = _new_session(
        _response(raw_answer, usage={"input_tokens": 12, "output_tokens": 0})
    )
    media = MediaBundle((MediaItem("image", 0, "image/png", b"image", {}),))
    question = {"question": "Which one?", "answer": ["first", "second"]}
    try:
        answer, metrics, diagnostics = session.systemone(
            state="System:\nrules\n\nContext:\nfacts", question=question, media=media
        )
    finally:
        session.close()

    request = calls[0][2]
    assert calls[0][0].endswith("/v1/systemone")
    assert calls[0][1] == "POST"
    assert request["model"] == "model-a"
    assert request["state"] == "System:\nrules\n\nContext:\nfacts"
    assert request["questions"] == {
        "question": {
            "type": "choice",
            "instructions": "Which one?",
            "criteria": {"first": None, "second": None},
        }
    }
    assert request["images"] == ["data:image/png;base64,aW1hZ2U="]
    assert answer == {
        "type": "choice",
        "selected": "second",
        "value": 0.7,
        "probabilities": [0.3, 0.7],
        "result": raw_answer,
    }
    assert metrics["operation"] == "systemone"
    assert metrics["model"] == "model-a"
    assert metrics["usage"] == {"input_tokens": 12, "output_tokens": 0}
    assert metrics["decision_seconds"] >= 0
    assert diagnostics["requested"]["image_count"] == 1
    assert diagnostics["mtmd"]["verification"] == "unverified_remote"
    assert diagnostics["mtmd"]["all_media_evaluated"] is False


@pytest.mark.parametrize(
    ("question", "raw_answer", "expected"),
    [
        (
            {"type": "score", "question": "Rate", "answer": ["low", "high"]},
            {
                "type": "score",
                "score": 0.25,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"1": 0.25, "0": 0.75},
                "confidence": 0.5,
            },
            ("", 0.25, [0.75, 0.25]),
        ),
        (
            {"type": "noul", "question": "Is it ready?"},
            {"type": "noul", "noul": 0.8},
            ("", 0.8, [1.0 - 0.8, 0.8]),
        ),
    ],
)
def test_systemone_score_and_noul_mapping(question, raw_answer, expected):
    session, calls = _new_session(_response(raw_answer))
    try:
        answer, _, _ = session.systemone(state="context", question=question)
    finally:
        session.close()

    sent_question = calls[0][2]["questions"]["question"]
    assert sent_question["type"] == question["type"]
    assert sent_question["instructions"] == question["question"]
    if question["type"] == "score":
        assert sent_question["criteria"] == ["low", "high"]
        assert answer["result"]["legend"] == raw_answer["legend"]
    else:
        assert "criteria" not in sent_question
    assert (answer["selected"], answer["value"], answer["probabilities"]) == expected


@pytest.mark.parametrize(
    ("question", "raw_answer"),
    [
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {"type": "score", "choice": "a", "probabilities": {"a": 1, "b": 0}},
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {"type": "choice", "choice": "missing", "probabilities": {"a": 1, "b": 0}},
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {"type": "choice", "choice": "a", "probabilities": {"a": 1}},
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": True, "b": 0},
            },
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.2},
            },
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 1, "b": 10**1000},
            },
        ),
        (
            {"type": "choice", "question": "Pick", "answer": ["a", "b"]},
            {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": math.nan, "b": 0},
            },
        ),
        (
            {"type": "score", "question": "Rate", "answer": ["low", "high"]},
            {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "low", "1": "wrong"},
                "probabilities": {"0": 0.5, "1": 0.5},
            },
        ),
        (
            {"type": "score", "question": "Rate", "answer": ["low", "high"]},
            {
                "type": "score",
                "score": 1.1,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.5, "1": 0.5},
            },
        ),
        (
            {"type": "noul", "question": "Ready?"},
            {"type": "noul", "noul": True},
        ),
        (
            {"type": "noul", "question": "Ready?"},
            {"type": "noul", "noul": 1.1},
        ),
    ],
)
def test_systemone_rejects_malformed_responses(question, raw_answer):
    session, _ = _new_session(_response(raw_answer))
    try:
        with pytest.raises(BackendError):
            session.systemone(state="context", question=question)
    finally:
        session.close()


@pytest.mark.parametrize("status", [404, 501])
def test_systemone_http_errors_do_not_fall_back(status):
    session, calls = _new_session(b'{"error":"unsupported"}', status=status)
    try:
        with pytest.raises(BackendError, match=f"HTTP {status}"):
            session.systemone(
                state="context",
                question={"type": "noul", "question": "Ready?"},
            )
    finally:
        session.close()

    paths = [call[0] for call in calls]
    assert paths == [
        "http://localhost:8080/v1/systemone",
        "http://localhost:8080/models/unload",
    ]
    assert not any(
        path.endswith(("/apply-template", "/tokenize", "/v1/chat/completions"))
        for path in paths
    )


def test_systemone_closed_session_rejects_request():
    session, calls = _new_session(_response({"type": "noul", "noul": 0.5}))
    session.close()

    with pytest.raises(BackendError, match="already been unloaded"):
        session.systemone(
            state="context", question={"type": "noul", "question": "Ready?"}
        )
    assert [call[0] for call in calls] == ["http://localhost:8080/models/unload"]


def test_extract_answer_returns_atomic_copy_and_checks_schema_and_consistency(
    monkeypatch,
):
    nodes = import_llama_cpp_decision_nodes(monkeypatch)
    schema = nodes.LlamaCppExtractAnswerNode.define_schema()
    assert getattr(schema, "is_input_list", False) is False
    assert [field.name for field in schema.outputs] == [
        "selected",
        "value",
        "probabilities",
        "result_json",
    ]
    probabilities_field = schema.outputs[2]
    assert probabilities_field.data_type == "float"
    assert probabilities_field.options["is_output_list"] is True

    payload = {
        "type": "choice",
        "selected": "second",
        "value": 0.7,
        "probabilities": [0.3, 0.7],
        "result": {
            "type": "choice",
            "choice": "second",
            "probabilities": {"second": 0.7, "first": 0.3},
            "confidence": 0.6,
        },
    }
    extracted = nodes.LlamaCppExtractAnswerNode.execute([payload])
    assert extracted[0] == "second"
    assert extracted[1] == 0.7
    assert extracted[2] == [0.3, 0.7]
    assert json.loads(extracted[3]) == payload["result"]
    extracted[2].append(0.0)
    assert payload["probabilities"] == [0.3, 0.7]

    bad_payloads = [
        {**payload, "type": []},
        {
            "type": "choice",
            "selected": "second",
            "value": 0.7,
            "probabilities": [0.3, 0.7],
            "result": {
                "type": "choice",
                "choice": "second",
                "probabilities": {"second": 0.6, "first": 0.4},
            },
        },
        {
            "type": "choice",
            "selected": "second",
            "value": 0.7,
            "probabilities": [0.4, 0.6],
            "result": {
                "type": "choice",
                "choice": "second",
                "probabilities": {"second": 0.7, "first": 0.3},
            },
        },
        {
            "type": "score",
            "selected": "",
            "value": 1.0,
            "probabilities": [0.8, 0.2],
            "result": {
                "type": "score",
                "score": 1.0,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.2, "1": 0.8},
            },
        },
        {
            "type": "noul",
            "selected": "",
            "value": 10**1000,
            "probabilities": [0.0, 1.0],
            "result": {"type": "noul", "noul": 1.0},
        },
    ]
    for bad_payload in bad_payloads:
        with pytest.raises(InputNormalizationError):
            nodes.LlamaCppExtractAnswerNode.execute([bad_payload])

    with pytest.raises(InputNormalizationError, match="exactly one value"):
        nodes.LlamaCppExtractAnswerNode.execute([payload, payload])


def test_systemone_unload_uses_connect_and_runtime_ownership(monkeypatch):
    nodes = import_llama_cpp_decision_nodes(monkeypatch)
    raw_answer = {
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 0.9, "no": 0.1},
    }
    response = _response(raw_answer)

    connect, connect_calls = _new_session(response)
    try:
        result = nodes.LlamaCppDecideSystemOneNode.execute(
            [connect],
            ["rules"],
            ["context"],
            [{"type": "choice", "question": "Ready?", "answer": ["yes", "no"]}],
            session_unload=[True],
        )
        assert json.loads(result[1])["session"]["unload_required"] is False
        assert result[2]["model_unloaded_after_response"] is True
        assert connect.closed is True
        assert connect_calls[-1][0].endswith("/models/unload")
    finally:
        connect.close()

    class Process:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    process = Process()
    runtime, runtime_calls = _new_session(
        response, session_type=OwnedLlamaCppServerSession, process=process
    )
    try:
        result = nodes.LlamaCppDecideSystemOneNode.execute(
            [runtime],
            ["rules"],
            ["context"],
            [{"type": "choice", "question": "Ready?", "answer": ["yes", "no"]}],
            session_unload=[True],
        )
        assert runtime.closed is True
        assert process.close_calls == 1
        assert not any(call[0].endswith("/models/unload") for call in runtime_calls)
        assert json.loads(result[1])["model_unloaded"] is True
    finally:
        runtime.close()

    with pytest.raises(InputNormalizationError, match="Native sessions are unsupported"):
        nodes.LlamaCppDecideSystemOneNode.execute(
            [object()], [""], [""], [{"type": "noul", "question": "Ready?"}]
        )
