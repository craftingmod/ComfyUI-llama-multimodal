from __future__ import annotations

import base64
import json
import math
import re
import socket
import time
from collections.abc import Mapping
from threading import Lock
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ..core import BackendError, InputNormalizationError, MediaBundle
from ..llama_cpp.llama_cpp_session_cleanup import track_session, untrack_session
from .llama_cpp import (
    LlamaCppDecisionResult,
    LlamaCppResult,
    _data_uri,
    _extract_response,
)

Transport = Callable[[str, str, bytes | None, float], tuple[int, bytes]]
_BASE64_RUN = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/])"
)
_REQUEST_TIMEOUT_SECONDS = 300.0
_FLOAT32_LOWEST_LOG_PROBABILITY = -3.4028234663852886e38


class _ClosableProcess(Protocol):
    def close(self) -> None: ...


def _validated_url(value: str) -> SplitResult:
    if not isinstance(value, str) or not value.strip():
        raise InputNormalizationError("llama.cpp server URL must be a nonempty string.")
    text = value.strip()
    try:
        parsed = urlsplit(text)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise InputNormalizationError("llama.cpp server URL is invalid.") from exc
    if any(character.isspace() for character in text):
        raise InputNormalizationError("llama.cpp server URL cannot contain whitespace.")
    if parsed.scheme not in {"http", "https"}:
        raise InputNormalizationError("llama.cpp server URL must use http or https.")
    if not hostname:
        raise InputNormalizationError("llama.cpp server URL must include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise InputNormalizationError(
            "llama.cpp server URL cannot include credentials."
        )
    if parsed.query or parsed.fragment:
        raise InputNormalizationError(
            "llama.cpp server URL cannot include a query string or fragment."
        )
    return parsed


def _endpoint_url(value: str, endpoint: str) -> str:
    parsed = _validated_url(value)
    path = parsed.path.rstrip("/") + endpoint
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _redact_text(value: str) -> str:
    return _BASE64_RUN.sub("<redacted-base64>", value)[:1000]


def _http_error_message(status: int, body: bytes) -> str:
    detail = ""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            detail = str(error.get("message", ""))
        elif isinstance(error, str):
            detail = error
    elif body:
        detail = body.decode("utf-8", errors="replace")
    detail = _redact_text(detail.strip())
    return f"llama.cpp server returned HTTP {status}" + (
        f": {detail}" if detail else "."
    )


def _default_transport(
    url: str,
    method: str,
    body: bytes | None,
    timeout: float,
    *,
    api_key: str | None = None,
) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise BackendError(
            f"Could not reach llama.cpp server: {_redact_text(str(reason))}"
        ) from exc


def _request(
    *,
    url: str,
    method: str,
    body: bytes | None = None,
    timeout_seconds: float,
    api_key: str | None = None,
    transport: Transport | None = None,
) -> bytes:
    if timeout_seconds <= 0:
        raise InputNormalizationError("timeout_seconds must be greater than zero.")
    if transport is None:
        status, response_body = _default_transport(
            url, method, body, float(timeout_seconds), api_key=api_key
        )
    else:
        status, response_body = transport(url, method, body, float(timeout_seconds))
    if status < 200 or status >= 300:
        raise BackendError(_http_error_message(status, response_body))
    return response_body


def parse_models_response(response_body: bytes) -> list[str]:
    try:
        parsed = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendError(
            "llama.cpp returned an invalid model-list response."
        ) from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("data"), list):
        raise BackendError(
            "llama.cpp model-list response did not contain a data array."
        )

    models: list[str] = []
    seen: set[str] = set()
    for item in parsed["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise BackendError("llama.cpp model-list entries must contain a string id.")
        model = item["id"].strip()
        if not model:
            raise BackendError("llama.cpp model-list entries cannot have an empty id.")
        if model not in seen:
            models.append(model)
            seen.add(model)
    return models


def list_server_models(
    *,
    url: str,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
    transport: Transport | None = None,
) -> list[str]:
    body = _request(
        url=_endpoint_url(url, "/models"),
        method="GET",
        timeout_seconds=timeout_seconds,
        api_key=api_key,
        transport=transport,
    )
    return parse_models_response(body)


def _build_messages(
    system: str,
    prompt: str,
    media: MediaBundle,
    *,
    media_before_prompt: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if not media.items:
        messages.append({"role": "user", "content": prompt})
        return messages

    prompt_part = {"type": "text", "text": prompt}
    content: list[dict[str, Any]] = [] if media_before_prompt else [prompt_part]
    for item in media.items:
        if item.kind == "image":
            part = {
                "type": "image_url",
                "image_url": {"url": _data_uri(item.mime_type, item.payload)},
            }
        elif item.kind == "audio":
            part = {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(item.payload).decode("ascii"),
                    "format": "wav",
                },
            }
        elif item.kind == "video":
            part = {
                "type": "input_video",
                "input_video": {"data": base64.b64encode(item.payload).decode("ascii")},
            }
        else:
            raise InputNormalizationError(
                f"The llama.cpp server node does not support {item.kind} media."
            )
        content.append(part)
    if media_before_prompt:
        content.append(prompt_part)
    messages.append({"role": "user", "content": content})
    return messages


def _systemone_probability_values(
    value: Any, expected_keys: list[str], label: str
) -> list[float]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise BackendError(
            f"System One {label} must contain exactly the expected keys."
        )
    values: list[float] = []
    for key in expected_keys:
        probability = value[key]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0.0 <= probability <= 1.0
        ):
            raise BackendError(
                f"System One {label} values must be finite probabilities."
            )
        values.append(float(probability))
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise BackendError(f"System One {label} probabilities must sum to 1.")
    return values


def normalize_systemone_question(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputNormalizationError("question must be a System One question object.")
    fields = set(value)
    if fields == {"question", "answer"}:
        question_type = "choice"
    else:
        question_type = value.get("type")
        expected_fields = (
            {"type", "question"}
            if question_type == "noul"
            else {"type", "question", "answer"}
        )
        if (
            not isinstance(question_type, str)
            or question_type not in {"choice", "score", "noul"}
            or fields != expected_fields
        ):
            raise InputNormalizationError("question is not a valid System One payload.")

    instructions = value.get("question")
    if not isinstance(instructions, str) or not instructions.strip():
        raise InputNormalizationError("question must be a non-empty string.")
    if question_type == "noul":
        return {"type": "noul", "question": instructions}

    answers = value.get("answer")
    maximum = 26 if question_type == "choice" else 10
    if not isinstance(answers, (list, tuple)) or not 2 <= len(answers) <= maximum:
        label = "answers" if question_type == "choice" else "score levels"
        raise InputNormalizationError(
            f"{label} must contain between 2 and {maximum} values."
        )
    if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise InputNormalizationError("answer values must be non-empty strings.")
    if len(set(answers)) != len(answers):
        raise InputNormalizationError("answer values must be unique.")
    return {
        "type": question_type,
        "question": instructions,
        "answer": list(answers),
    }


def _systemone_answer_payload(
    answer: Any, question: Mapping[str, Any]
) -> dict[str, Any]:
    question_type = question["type"]
    if not isinstance(answer, dict) or answer.get("type") != question_type:
        raise BackendError(
            "System One response answer has an invalid or mismatched type."
        )

    if question_type == "choice":
        options = list(question["criteria"])
        probabilities = _systemone_probability_values(
            answer.get("probabilities"), options, "choice probabilities"
        )
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in options:
            raise BackendError("System One response selected an unknown choice.")
        value = probabilities[options.index(selected)]
    elif question_type == "score":
        levels = question["criteria"]
        keys = [str(index) for index in range(len(levels))]
        probabilities = _systemone_probability_values(
            answer.get("probabilities"), keys, "score probabilities"
        )
        legend = answer.get("legend")
        if (
            not isinstance(legend, dict)
            or set(legend) != set(keys)
            or any(
                legend[key] != level for key, level in zip(keys, levels, strict=True)
            )
        ):
            raise BackendError(
                "System One response score legend does not match its levels."
            )
        value = answer.get("score")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= len(levels) - 1
        ):
            raise BackendError("System One response score is outside its level range.")
        value = float(value)
        selected = ""
    else:
        value = answer.get("noul")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            raise BackendError(
                "System One response noul value must be a finite probability."
            )
        value = float(value)
        probabilities = [1.0 - value, value]
        selected = ""

    return {
        "type": question_type,
        "selected": selected,
        "value": value,
        "probabilities": list(probabilities),
        "result": dict(answer),
    }


class LlamaCppServerSession:
    """A local handle for a model managed by a llama.cpp HTTP server."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        _endpoint_url(url, "/v1/chat/completions")
        if not isinstance(model, str) or not model.strip():
            raise InputNormalizationError("model cannot be empty.")

        health_body = _request(
            url=_endpoint_url(url, "/health"),
            method="GET",
            timeout_seconds=10.0,
            api_key=api_key,
            transport=transport,
        )
        try:
            health = json.loads(health_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "llama.cpp server health check returned invalid JSON."
            ) from exc
        if not isinstance(health, dict) or health.get("status") != "ok":
            status = health.get("status") if isinstance(health, dict) else None
            raise BackendError(
                "llama.cpp server health check failed: expected status 'ok', "
                f"received {_redact_text(str(status))!r}."
            )

        self.url = url.strip()
        self.model = model
        self._api_key = api_key
        self._transport = transport
        self._closed = False
        self._close_lock = Lock()
        self._execution_count = 0
        self._decision_token_ids: dict[str, int] = {}
        self._decision_vocab_size: int | None = None
        track_session(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        media: MediaBundle,
        max_tokens: int,
        seed: int,
        stop: str,
        model_profile: dict[str, Any] | None = None,
        reuse_kv_cache: bool | None = None,
        media_before_prompt: bool = False,
    ) -> LlamaCppResult:
        if self._closed:
            raise BackendError(
                "The llama.cpp server session has already been unloaded."
            )
        if not self.model.strip():
            raise InputNormalizationError("model cannot be empty.")
        if max_tokens <= 0:
            raise InputNormalizationError("max_tokens must be greater than zero.")
        if reuse_kv_cache is not None and not isinstance(reuse_kv_cache, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        if not isinstance(media_before_prompt, bool):
            raise InputNormalizationError("media_before_prompt must be a boolean.")

        request: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(
                system,
                prompt,
                media,
                media_before_prompt=media_before_prompt,
            ),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if reuse_kv_cache is not None:
            request["cache_prompt"] = reuse_kv_cache
        if model_profile is not None:
            # ponytail: Handler/template are launch-bound; changes need a restart.
            request.update(
                temperature=model_profile["temperature"],
                top_p=model_profile["top_p"],
                top_k=model_profile["top_k"],
                min_p=model_profile["min_p"],
                presence_penalty=model_profile["presence_penalty"],
                repeat_penalty=model_profile["repeat_penalty"],
            )
            if model_profile["recommended_reasoning_mode"] != "auto":
                request["chat_template_kwargs"] = {
                    "enable_thinking": model_profile["recommended_reasoning_mode"]
                    == "on"
                }
        if seed >= 0:
            request["seed"] = int(seed)
        if stop:
            request["stop"] = [stop]
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )

        started = time.perf_counter()
        response_body = _request(
            url=_endpoint_url(self.url, "/v1/chat/completions"),
            method="POST",
            body=body,
            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
            api_key=self._api_key,
            transport=self._transport,
        )
        elapsed = time.perf_counter() - started
        try:
            raw = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "llama.cpp server returned an invalid chat-completion response."
            ) from exc
        if not isinstance(raw, dict):
            raise BackendError("llama.cpp server returned a non-object response.")
        try:
            response, thinking = _extract_response(raw)
        except BackendError as exc:
            message = str(exc).replace("llama-cpp-python", "llama.cpp server")
            raise BackendError(message) from None

        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
        execution_index = self._execution_count
        self._execution_count += 1
        metrics = {
            "load_seconds": 0.0,
            "generation_seconds": elapsed,
            "cleanup_seconds": 0.0,
            "total_seconds": elapsed,
            "usage": usage,
            "server_timings": timings,
            "model_unloaded": False,
            "session": {
                "execution_index": execution_index,
                "model_reused": execution_index > 0,
                "unload_required": True,
                "remote": True,
            },
        }
        manifest = media.manifest()
        has_media = bool(media.items)
        media_diagnostics = {
            "schema_version": 1,
            "backend": "llama.cpp-server",
            "model": self.model,
            "handler": "server",
            "capabilities": {"vision": False, "audio": False, "video": False},
            "requested": manifest,
            "evaluated": {
                "media_count": 0,
                "image_count": 0,
                "audio_count": 0,
                "video_count": 0,
            },
            "mtmd": {
                "strict_pipeline": False,
                "completion_succeeded": True,
                "all_media_evaluated": not has_media,
                "verification": "unverified_remote" if has_media else "no_media",
            },
            "model_unloaded_after_response": False,
        }
        return LlamaCppResult(
            response=response,
            thinking=thinking,
            raw=raw,
            metrics=metrics,
            media_diagnostics=media_diagnostics,
        )

    def decide(
        self,
        *,
        question: str,
        context: str,
        answers: list[str],
        model_profile: dict[str, Any] | None = None,
        seed: int = -1,
        media: MediaBundle | None = None,
        reuse_kv_cache: bool | None = None,
        media_before_prompt: bool = False,
    ) -> LlamaCppDecisionResult:
        if self._closed:
            raise BackendError(
                "The llama.cpp server session has already been unloaded."
            )
        media = media or MediaBundle()
        if reuse_kv_cache is not None and not isinstance(reuse_kv_cache, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        if not isinstance(media_before_prompt, bool):
            raise InputNormalizationError("media_before_prompt must be a boolean.")
        if not isinstance(question, str) or not question.strip():
            raise InputNormalizationError("question must be a non-empty string.")
        if not isinstance(context, str):
            raise InputNormalizationError("context must be a string.")
        if not isinstance(answers, list) or not 2 <= len(answers) <= 26:
            raise InputNormalizationError(
                "answers must contain between 2 and 26 items."
            )
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise InputNormalizationError("answers must be non-empty strings.")
        if len(set(answers)) != len(answers):
            raise InputNormalizationError("answers must be unique.")

        try:
            from makoto_decision import Choices
        except ImportError as exc:
            raise BackendError(
                "makoto-decision is required for Llama.cpp decision sessions. "
                "Install the optional llama dependencies and restart ComfyUI."
            ) from exc

        try:
            choices = Choices.letters(*answers)
            vocab_size = self._get_decision_vocab_size()
            token_to_answer: dict[int, str] = {}
            for choice in choices:
                token_id = self._decision_token_id(choice.target)
                if token_id in token_to_answer:
                    raise BackendError(
                        "Decision choice targets must map to distinct server token IDs."
                    )
                token_to_answer[token_id] = choice.value

            prompt_sections = [question]
            prompt_sections.extend(
                (
                    "Choices:\n"
                    + "\n".join(
                        f"{choice.target}: {choice.value}" for choice in choices
                    ),
                    "Respond with exactly one of: "
                    + ", ".join(choice.target for choice in choices),
                )
            )
            context_sections = [context] if context else []
            question_prompt = "\n\n".join(prompt_sections)
            context_sections.append(question_prompt)
            decision_prompt = "\n\n".join(context_sections)
            grammar = "root ::= " + " | ".join(
                f'"{choice.target}"' for choice in choices
            )
            template_request: dict[str, Any] = {
                "messages": [{"role": "user", "content": decision_prompt}]
            }
            if media.items:
                media_parts = _build_messages("", "", media)[-1]["content"][1:]
                context_parts = (
                    [{"type": "text", "text": context + "\n\n"}] if context else []
                )
                content = (
                    media_parts + context_parts
                    if media_before_prompt
                    else context_parts + media_parts
                )
                content.append({"type": "text", "text": question_prompt})
                # Let the server insert its active, possibly randomized media marker.
                template_request["messages"][0]["content"] = content
            if (
                model_profile is not None
                and model_profile["recommended_reasoning_mode"] != "auto"
            ):
                template_request["chat_template_kwargs"] = {
                    "enable_thinking": model_profile["recommended_reasoning_mode"]
                    == "on"
                }
            template_body = json.dumps(
                template_request, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            template_response = _request(
                url=_endpoint_url(self.url, "/apply-template"),
                method="POST",
                body=template_body,
                timeout_seconds=10.0,
                api_key=self._api_key,
                transport=self._transport,
            )
            try:
                template = json.loads(template_response.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackendError(
                    "llama.cpp server returned an invalid decision template response."
                ) from exc
            prompt = template.get("prompt") if isinstance(template, dict) else None
            if not isinstance(prompt, str) or not prompt:
                raise BackendError(
                    "llama.cpp server did not return a formatted decision prompt."
                )

            request = {
                "prompt": prompt,
                "n_predict": 1,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "xtc_probability": 0.0,
                "mirostat": 0,
                # Request raw pre-sampling log probabilities for the entire
                # vocabulary: post-sampling top-N can contain non-choice tokens
                # and omit a canonical choice after grammar/sampler processing.
                "n_probs": vocab_size,
                "post_sampling_probs": False,
                "backend_sampling": False,
                "grammar": grammar,
            }
            if media.items:
                request["prompt"] = {
                    "prompt_string": prompt,
                    "multimodal_data": [
                        base64.b64encode(item.payload).decode("ascii")
                        for item in media.items
                    ],
                }
            if reuse_kv_cache is not None:
                request["cache_prompt"] = reuse_kv_cache
            if seed >= 0:
                request["seed"] = int(seed)
            if model_profile is not None:
                request.update(
                    temperature=model_profile["temperature"],
                    top_k=model_profile["top_k"],
                    top_p=model_profile["top_p"],
                    min_p=model_profile["min_p"],
                    repeat_penalty=model_profile["repeat_penalty"],
                    presence_penalty=model_profile["presence_penalty"],
                )
            started = time.perf_counter()
            response_body = _request(
                url=_endpoint_url(self.url, "/completion"),
                method="POST",
                body=json.dumps(
                    request, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8"),
                timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
                api_key=self._api_key,
                transport=self._transport,
            )
            elapsed = time.perf_counter() - started
            try:
                response = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackendError(
                    "llama.cpp server returned an invalid decision response."
                ) from exc
            if not isinstance(response, dict):
                raise BackendError(
                    "llama.cpp server returned a non-object decision response."
                )

            generated = response.get("content")
            if not isinstance(generated, str) or generated.strip() not in {
                choice.target for choice in choices
            }:
                raise BackendError(
                    "llama.cpp server did not generate a valid decision label."
                )

            probability_steps = response.get("probs")
            if probability_steps is None:
                probability_steps = response.get("completion_probabilities")
            if not isinstance(probability_steps, list) or len(probability_steps) != 1:
                raise BackendError(
                    "llama.cpp server decision response must contain one token "
                    "probability step."
                )
            top_logprobs = probability_steps[0]
            if isinstance(top_logprobs, dict):
                top_logprobs = top_logprobs.get("top_logprobs")
            if not isinstance(top_logprobs, list):
                raise BackendError(
                    "llama.cpp server decision response is missing top_logprobs."
                )

            choice_logprobs: dict[int, float] = {}
            seen_token_ids: set[int] = set()
            for item in top_logprobs:
                if not isinstance(item, dict):
                    raise BackendError(
                        "llama.cpp server returned an invalid decision probability."
                    )
                token_id = item.get("id")
                logprob = item.get("logprob")
                if type(token_id) is not int or not 0 <= token_id < vocab_size:
                    raise BackendError(
                        "llama.cpp server returned an invalid decision token ID."
                        f" generated={generated!r}, vocab_size={vocab_size},"
                        f" candidate={item!r}"
                    )
                if token_id in seen_token_ids:
                    raise BackendError(
                        "llama.cpp server returned a duplicate decision token."
                    )
                if (
                    isinstance(logprob, bool)
                    or not isinstance(logprob, (int, float))
                    or not math.isfinite(logprob)
                    or logprob > 0.0
                ):
                    raise BackendError(
                        "llama.cpp server returned an invalid decision probability."
                    )
                seen_token_ids.add(token_id)
                if token_id in token_to_answer:
                    choice_logprobs[token_id] = float(logprob)

            missing_targets = [
                choice.target
                for choice in choices
                if self._decision_token_ids[choice.target] not in choice_logprobs
            ]
            if missing_targets:
                expected_tokens = {
                    choice.target: self._decision_token_ids[choice.target]
                    for choice in choices
                }
                raise BackendError(
                    "llama.cpp server did not return probability values for every "
                    "decision choice."
                    f" generated={generated!r}, missing_targets={missing_targets!r},"
                    f" expected_tokens={expected_tokens!r},"
                    f" received_candidates={len(top_logprobs)}, vocab_size={vocab_size}"
                )

            ordered_choice_logprobs = [
                choice_logprobs[self._decision_token_ids[choice.target]]
                for choice in choices
            ]
            if all(
                logprob == _FLOAT32_LOWEST_LOG_PROBABILITY
                for logprob in ordered_choice_logprobs
            ):
                raise BackendError(
                    "llama.cpp server returned zero probability mass for every "
                    "decision choice."
                )
            max_logprob = max(ordered_choice_logprobs)
            weights = [
                math.exp(logprob - max_logprob) for logprob in ordered_choice_logprobs
            ]
            total = math.fsum(weights)
            if not math.isfinite(total) or total <= 0.0:
                raise BackendError(
                    "llama.cpp server returned no usable probability mass for "
                    "the decision choices."
                )
            probabilities = {
                choice.value: weight / total
                for choice, weight in zip(choices, weights, strict=True)
            }

            selected = max(probabilities, key=probabilities.__getitem__)
            execution_index = self._execution_count
            self._execution_count += 1
            manifest = media.manifest()
            has_media = bool(media.items)
            return LlamaCppDecisionResult(
                selected=selected,
                probabilities=probabilities,
                metrics={
                    "decision_seconds": elapsed,
                    "server_timings": response.get("timings", {}),
                    "model_unloaded": False,
                    "session": {
                        "execution_index": execution_index,
                        "model_reused": execution_index > 0,
                        "unload_required": True,
                        "remote": True,
                    },
                },
                media_diagnostics={
                    "schema_version": 1,
                    "backend": "llama.cpp-server",
                    "model": self.model,
                    "handler": "server",
                    "capabilities": {
                        "vision": False,
                        "audio": False,
                        "video": False,
                    },
                    "requested": manifest,
                    "evaluated": {
                        "media_count": 0,
                        "image_count": 0,
                        "audio_count": 0,
                        "video_count": 0,
                    },
                    "mtmd": {
                        "strict_pipeline": False,
                        "completion_succeeded": True,
                        "all_media_evaluated": not has_media,
                        "verification": "unverified_remote"
                        if has_media
                        else "no_media",
                    },
                    "model_unloaded_after_response": False,
                },
            )
        except (BackendError, InputNormalizationError):
            raise
        except Exception as exc:
            raise BackendError(f"llama.cpp server decision failed: {exc}") from exc

    def systemone(
        self,
        *,
        state: str,
        question: Mapping[str, Any],
        media: MediaBundle | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise BackendError(
                "The llama.cpp server session has already been unloaded."
            )
        if not isinstance(state, str):
            raise InputNormalizationError("state must be a string.")
        question = normalize_systemone_question(question)
        media = MediaBundle() if media is None else media
        if not isinstance(media, MediaBundle) or any(
            item.kind != "image" for item in media.items
        ):
            raise InputNormalizationError("System One accepts images only.")

        question_type = question.get("type")
        instructions = question["question"]
        if question_type == "choice":
            request_question = {
                "type": question_type,
                "instructions": instructions,
                "criteria": {answer: None for answer in question["answer"]},
            }
        elif question_type == "score":
            request_question = {
                "type": question_type,
                "instructions": instructions,
                "criteria": list(question["answer"]),
            }
        elif question_type == "noul":
            request_question = {"type": question_type, "instructions": instructions}
        else:
            raise InputNormalizationError("System One question type is unsupported.")

        request: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {"question": request_question},
        }
        if media.items:
            request["images"] = [
                _data_uri(item.mime_type, item.payload) for item in media.items
            ]
        started = time.perf_counter()
        response_body = _request(
            url=_endpoint_url(self.url, "/v1/systemone"),
            method="POST",
            body=json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
            timeout_seconds=_REQUEST_TIMEOUT_SECONDS,
            api_key=self._api_key,
            transport=self._transport,
        )
        elapsed = time.perf_counter() - started
        try:
            response = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "llama.cpp server returned invalid System One JSON."
            ) from exc
        if not isinstance(response, dict):
            raise BackendError(
                "llama.cpp server returned a non-object System One response."
            )
        answers = response.get("answers")
        if not isinstance(answers, dict) or not isinstance(
            answers.get("question"), dict
        ):
            raise BackendError(
                "llama.cpp System One response is missing answers.question."
            )
        answer_payload = _systemone_answer_payload(
            answers["question"], request_question
        )

        execution_index = self._execution_count
        self._execution_count += 1
        manifest = media.manifest()
        has_media = bool(media.items)
        metrics = {
            "operation": "systemone",
            "model": self.model,
            "decision_seconds": elapsed,
            "usage": dict(response.get("usage", {}))
            if isinstance(response.get("usage"), dict)
            else {},
            "model_unloaded": False,
            "session": {
                "execution_index": execution_index,
                "model_reused": execution_index > 0,
                "unload_required": True,
                "remote": True,
            },
        }
        media_diagnostics = {
            "schema_version": 1,
            "backend": "llama.cpp-server",
            "model": self.model,
            "handler": "server",
            "capabilities": {"vision": False, "audio": False, "video": False},
            "requested": manifest,
            "evaluated": {
                "media_count": 0,
                "image_count": 0,
                "audio_count": 0,
                "video_count": 0,
            },
            "mtmd": {
                "strict_pipeline": False,
                "completion_succeeded": True,
                "all_media_evaluated": not has_media,
                "verification": "unverified_remote" if has_media else "no_media",
            },
            "model_unloaded_after_response": False,
        }
        return answer_payload, metrics, media_diagnostics

    def _get_decision_vocab_size(self) -> int:
        if self._decision_vocab_size is not None:
            return self._decision_vocab_size

        response_body = _request(
            url=_endpoint_url(self.url, "/v1/models"),
            method="GET",
            timeout_seconds=10.0,
            api_key=self._api_key,
            transport=self._transport,
        )
        try:
            response = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "llama.cpp server returned an invalid decision model-info response."
            ) from exc
        models = response.get("data") if isinstance(response, dict) else None
        if not isinstance(models, list) or not models:
            raise BackendError(
                "llama.cpp server model-info response did not contain a data array."
            )
        matching_models = [
            item
            for item in models
            if isinstance(item, dict) and item.get("id") == self.model
        ]
        if len(matching_models) == 1:
            model_info = matching_models[0]
        elif not matching_models and len(models) == 1 and isinstance(models[0], dict):
            # The direct llama-server contract exposes exactly one loaded model;
            # accept its sole entry if it reports a different alias than this handle.
            model_info = models[0]
        else:
            raise BackendError(
                "llama.cpp server model-info response did not identify this session's "
                "model unambiguously."
            )
        metadata = model_info.get("meta")
        vocab_size = metadata.get("n_vocab") if isinstance(metadata, dict) else None
        if type(vocab_size) is not int or not 0 < vocab_size <= 2_147_483_647:
            raise BackendError(
                "llama.cpp server model-info response is missing a valid meta.n_vocab."
            )
        self._decision_vocab_size = vocab_size
        return vocab_size

    def _decision_token_id(self, target: str) -> int:
        cached = self._decision_token_ids.get(target)
        if cached is not None:
            return cached
        body = json.dumps(
            {"content": target, "add_special": False}, separators=(",", ":")
        ).encode("utf-8")
        response_body = _request(
            url=_endpoint_url(self.url, "/tokenize"),
            method="POST",
            body=body,
            timeout_seconds=10.0,
            api_key=self._api_key,
            transport=self._transport,
        )
        try:
            response = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendError(
                "llama.cpp server returned an invalid decision tokenization response."
            ) from exc
        tokens = response.get("tokens") if isinstance(response, dict) else None
        if (
            not isinstance(tokens, list)
            or len(tokens) != 1
            or type(tokens[0]) is not int
            or tokens[0] < 0
        ):
            raise BackendError(
                f"Decision choice target {target!r} must tokenize to exactly one "
                "valid server token."
            )
        self._decision_token_ids[target] = tokens[0]
        return tokens[0]

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                body = json.dumps({"model": self.model}, separators=(",", ":")).encode(
                    "utf-8"
                )
                response_body = _request(
                    url=_endpoint_url(self.url, "/models/unload"),
                    method="POST",
                    body=body,
                    timeout_seconds=30.0,
                    api_key=self._api_key,
                    transport=self._transport,
                )
                if response_body:
                    try:
                        response = json.loads(response_body.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise BackendError(
                            "llama.cpp server returned an invalid model-unload response."
                        ) from exc
                    if not isinstance(response, dict):
                        raise BackendError(
                            "llama.cpp server returned an invalid model-unload response."
                        )
                    if response.get("success") is False or "error" in response:
                        raise BackendError(
                            "llama.cpp server rejected the model unload request."
                        )
            except Exception:
                # Prompt-end cleanup clears the registry before closing; retain failed
                # unloads so a later cleanup or explicit retry can try again.
                track_session(self)
                raise
            self._closed = True
            untrack_session(self)


class OwnedLlamaCppServerSession(LlamaCppServerSession):
    """A server session whose process lifetime is owned by this handle."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        process: _ClosableProcess,
        cleanup: Callable[[], None] | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not callable(getattr(process, "close", None)):
            raise InputNormalizationError("process must provide a close() method.")
        self._process = process
        self._process_closed = False
        self._cleanup = cleanup
        super().__init__(url=url, model=model, api_key=api_key, transport=transport)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            try:
                if not self._process_closed:
                    self._process.close()
                    self._process_closed = True
                if self._cleanup is not None:
                    self._cleanup()
                    self._cleanup = None
            except Exception:
                track_session(self)
                raise
            self._closed = True
            untrack_session(self)


__all__ = [
    "LlamaCppServerSession",
    "OwnedLlamaCppServerSession",
    "list_server_models",
    "parse_models_response",
]
