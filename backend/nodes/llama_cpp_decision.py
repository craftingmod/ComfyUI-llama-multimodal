from __future__ import annotations

import json
import math
from collections.abc import Mapping
from time import perf_counter
from typing import Any

try:
    from comfy_api.v0_0_2 import io
except ImportError:  # pragma: no cover - compatibility with newer ComfyUI builds
    from comfy_api.latest import io

from ..backends.llama_cpp import LlamaCppDecisionResult, LlamaCppSession
from ..backends.llama_cpp_server import (
    LlamaCppServerSession,
    normalize_systemone_question,
)
from ..core import (
    InputNormalizationError,
    normalize_media,
    unwrap_optional_scalar,
    unwrap_required_scalar,
)
from .llama_cpp_compact import (
    BASE_CATEGORY,
    LlamaCppModelProfileType,
    _sequential_media_bundles,
    normalize_compact_model_profile,
)
from .llama_cpp_diagnostics import LlamaCppMediaDiagnosticsType
from .llama_cpp_session import LlamaCppSessionType

LlamaCppQuestionType = io.Custom("OLLAMA_IMAGE_LIST_LLAMA_CPP_QUESTION")
LlamaCppAnswerType = io.Custom("OLLAMA_IMAGE_LIST_LLAMA_CPP_ANSWER")
MAX_ANSWERS = 26


def make_question_payload(question: Any, answers: Any) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise InputNormalizationError("question must be a non-empty string.")
    if not isinstance(answers, (list, tuple)):
        raise InputNormalizationError("answer must be a list of strings.")
    if not 2 <= len(answers) <= MAX_ANSWERS:
        raise InputNormalizationError("answer must contain between 2 and 26 values.")
    if any(not isinstance(answer, str) for answer in answers):
        raise InputNormalizationError("answer values must be strings.")
    if any(not answer.strip() for answer in answers):
        raise InputNormalizationError("answer values must be non-empty strings.")
    if len(set(answers)) != len(answers):
        raise InputNormalizationError("answer values must be unique.")
    return {"question": question, "answer": list(answers)}


def _question_from_input_lists(question: Any, answers: Any) -> dict[str, Any]:
    if not isinstance(question, list) or len(question) != 1:
        raise InputNormalizationError(
            "question must resolve to exactly one STRING value."
        )
    if not isinstance(answers, list) or any(
        isinstance(value, (list, tuple)) for value in answers
    ):
        raise InputNormalizationError("answer must be one flat ComfyUI STRING list.")
    return make_question_payload(question[0], answers)


def make_systemone_question_payload(
    question: Any, answers: Any, question_type: Any = "choice"
) -> dict[str, Any]:
    return normalize_systemone_question(
        {"type": question_type, "question": question, "answer": answers}
    )


def _systemone_question_from_input_lists(
    question: Any, answers: Any, question_type: Any
) -> dict[str, Any]:
    if not isinstance(question, list) or len(question) != 1:
        raise InputNormalizationError(
            "question must resolve to exactly one STRING value."
        )
    if not isinstance(question_type, list) or len(question_type) != 1:
        raise InputNormalizationError("type must resolve to exactly one COMBO value.")
    if not isinstance(answers, list) or any(
        isinstance(value, (list, tuple)) for value in answers
    ):
        raise InputNormalizationError("answer must be one flat ComfyUI STRING list.")
    return make_systemone_question_payload(question[0], answers, question_type[0])


def _validated_answer_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "type",
        "selected",
        "value",
        "probabilities",
        "result",
    }:
        raise InputNormalizationError("answer must be a System One Answer payload.")
    answer_type = value["type"]
    selected = value["selected"]
    number = value["value"]
    probabilities = value["probabilities"]
    result = value["result"]
    if not isinstance(answer_type, str) or answer_type not in {
        "choice",
        "score",
        "noul",
    }:
        raise InputNormalizationError("answer type is unsupported.")
    if not isinstance(selected, str):
        raise InputNormalizationError("answer selected must be a string.")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise InputNormalizationError("answer value must be a finite number.")
    try:
        number = float(number)
    except OverflowError as exc:
        raise InputNormalizationError(
            "answer value must be a finite number."
        ) from exc
    if not math.isfinite(number):
        raise InputNormalizationError("answer value must be a finite number.")
    if not isinstance(probabilities, list) or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not 0.0 <= item <= 1.0
        for item in probabilities
    ):
        raise InputNormalizationError("answer probabilities must be a finite list.")
    if len(probabilities) < 2 or not math.isclose(
        math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise InputNormalizationError("answer probabilities must sum to 1.")
    if not isinstance(result, Mapping):
        raise InputNormalizationError("answer result must be an object.")

    result_type = result.get("type")
    if result_type != answer_type:
        raise InputNormalizationError("answer result type does not match its payload.")
    if answer_type == "choice":
        result_probabilities = result.get("probabilities")
        if (
            not selected
            or len(probabilities) > 26
            or result.get("choice") != selected
            or not isinstance(result_probabilities, Mapping)
            or any(not isinstance(key, str) for key in result_probabilities)
            or selected not in result_probabilities
            or len(result_probabilities) != len(probabilities)
        ):
            raise InputNormalizationError("choice Answer fields are inconsistent.")
        raw_probabilities = list(result_probabilities.values())
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not 0.0 <= item <= 1.0
            for item in raw_probabilities
        ) or not math.isclose(
            math.fsum(raw_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise InputNormalizationError("choice Answer fields are inconsistent.")
        selected_probability = result_probabilities[selected]
        if (
            isinstance(selected_probability, bool)
            or not isinstance(selected_probability, (int, float))
            or not 0.0 <= selected_probability <= 1.0
            or float(selected_probability) != number
            or sorted(float(item) for item in raw_probabilities)
            != sorted(float(item) for item in probabilities)
        ):
            raise InputNormalizationError("choice Answer fields are inconsistent.")
    elif answer_type == "score":
        expected_keys = [str(index) for index in range(len(probabilities))]
        result_probabilities = result.get("probabilities")
        legend = result.get("legend")
        result_score = result.get("score")
        if (
            selected
            or len(probabilities) > 10
            or not isinstance(result_probabilities, Mapping)
            or set(result_probabilities) != set(expected_keys)
            or not isinstance(legend, Mapping)
            or set(legend) != set(expected_keys)
            or any(
                not isinstance(legend[key], str) or not legend[key].strip()
                for key in expected_keys
            )
            or len(set(legend.values())) != len(expected_keys)
            or not 0.0 <= number <= len(probabilities) - 1
            or isinstance(result_score, bool)
            or not isinstance(result_score, (int, float))
            or not 0.0 <= result_score <= len(probabilities) - 1
            or float(result_score) != number
        ):
            raise InputNormalizationError("score Answer fields are inconsistent.")
        raw_probabilities = [result_probabilities[key] for key in expected_keys]
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not 0.0 <= item <= 1.0
            for item in raw_probabilities
        ) or [float(item) for item in raw_probabilities] != [
            float(item) for item in probabilities
        ]:
            raise InputNormalizationError("score Answer fields are inconsistent.")
    else:
        result_noul = result.get("noul")
        if (
            selected
            or len(probabilities) != 2
            or not 0.0 <= number <= 1.0
            or isinstance(result_noul, bool)
            or not isinstance(result_noul, (int, float))
            or not 0.0 <= result_noul <= 1.0
            or float(result_noul) != number
            or float(probabilities[0]) != 1.0 - number
            or float(probabilities[1]) != number
        ):
            raise InputNormalizationError("noul Answer fields are inconsistent.")

    try:
        result_json = json.dumps(
            dict(result), ensure_ascii=False, indent=2, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise InputNormalizationError(
            "answer result must contain JSON values."
        ) from exc
    return {
        "selected": selected,
        "value": number,
        "noul": number >= 0.5 if answer_type == "noul" else None,
        "probabilities": [float(item) for item in probabilities],
        "result_json": result_json,
    }


def _validated_question_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {"question", "answer"}:
        raise InputNormalizationError(
            "question must come from a [llama.cpp] Build Question (Prefill) node."
        )
    return make_question_payload(payload["question"], payload["answer"])


def _decision_inputs(*, include_reuse_kv_cache: bool = False) -> list[Any]:
    inputs = [
        LlamaCppSessionType.Input("session"),
        io.String.Input("system", default="", multiline=True, dynamic_prompts=False),
        io.String.Input("context", default="", multiline=True, dynamic_prompts=False),
        LlamaCppQuestionType.Input("question"),
        io.Image.Input("images", optional=True),
        io.Audio.Input("audio", optional=True),
        io.Video.Input("video", optional=True),
        io.Boolean.Input("video_with_audio", default=False),
        io.Int.Input(
            "seed",
            default=-1,
            min=-1,
            max=0xFFFFFFFF,
            step=1,
            tooltip="Runtime completion seed; Native prefill scoring is deterministic.",
        ),
        LlamaCppModelProfileType.Input(
            "model_profile",
            optional=True,
            tooltip=(
                "Optional override; server sessions apply sampling and explicit "
                "on/off reasoning settings per request."
            ),
        ),
    ]
    if include_reuse_kv_cache:
        inputs.append(
            io.Boolean.Input(
                "reuse_kv_cache",
                default=True,
                tooltip="Attempt common-prefix KV reuse across independent decisions.",
            )
        )
    inputs.append(
        io.Boolean.Input(
            "session_unload",
            default=False,
            label_on="Unload",
            label_off="Keep",
            tooltip=(
                "Unload the session after the decision sequence completes."
                if include_reuse_kv_cache
                else "Unload the session after the decision completes."
            ),
        )
    )
    return inputs


def _decision_outputs(*, is_output_list: bool = False) -> list[Any]:
    return [
        io.String.Output("selected", is_output_list=is_output_list),
        io.String.Output(
            "probabilities_json",
            display_name="probabilities_json",
            is_output_list=is_output_list,
        ),
        io.String.Output(
            "metrics_json", display_name="metrics", is_output_list=is_output_list
        ),
        LlamaCppMediaDiagnosticsType.Output(
            "media_diagnostics",
            display_name="media_diagnostics",
            is_output_list=is_output_list,
        ),
        LlamaCppSessionType.Output("session", display_name="session"),
    ]


def _flat_contexts(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise InputNormalizationError("context must contain at least one string.")
    if any(not isinstance(item, str) for item in values):
        raise InputNormalizationError("context must be a flat list of strings.")
    return list(values)


def _flat_question_payloads(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise InputNormalizationError("question must contain at least one payload.")
    return [_validated_question_payload(item) for item in values]


def _flat_systemone_question_payloads(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise InputNormalizationError("question must contain at least one payload.")
    return [normalize_systemone_question(item) for item in values]


def _broadcast_values(values: list[Any], count: int, name: str) -> list[Any]:
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return values
    raise InputNormalizationError(
        f"{name} must contain one value or exactly {count} values."
    )


def _decision_context(system: str, context: str) -> str:
    context_parts = [
        f"System:\n{system}" if system else "",
        f"Context:\n{context}" if context else "",
    ]
    return "\n\n".join(part for part in context_parts if part)


def _systemone_inputs() -> list[Any]:
    return [
        LlamaCppSessionType.Input("session"),
        io.String.Input("system", default="", multiline=True, dynamic_prompts=False),
        io.String.Input("context", default="", multiline=True, dynamic_prompts=False),
        LlamaCppQuestionType.Input("question"),
        io.Image.Input("images", optional=True),
        io.Boolean.Input(
            "session_unload",
            default=False,
            label_on="Unload",
            label_off="Keep",
            tooltip="Unload the session after all System One requests.",
        ),
    ]


def _systemone_outputs(*, is_output_list: bool = False) -> list[Any]:
    return [
        LlamaCppAnswerType.Output("answer", is_output_list=is_output_list),
        io.String.Output(
            "metrics_json", display_name="metrics", is_output_list=is_output_list
        ),
        LlamaCppMediaDiagnosticsType.Output(
            "media_diagnostics",
            display_name="media_diagnostics",
            is_output_list=is_output_list,
        ),
        LlamaCppSessionType.Output("session", display_name="session"),
    ]


def _systemone_output_values(
    result: tuple[dict[str, Any], dict[str, Any], dict[str, Any]], *, unloaded: bool
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    answer, metrics, media_diagnostics = result
    metrics = dict(metrics)
    metrics["model_unloaded"] = unloaded
    if isinstance(metrics.get("session"), dict):
        metrics["session"] = dict(metrics["session"])
        metrics["session"]["unload_required"] = not unloaded
    media_diagnostics = dict(media_diagnostics)
    media_diagnostics["model_unloaded_after_response"] = unloaded
    return answer, json.dumps(metrics, ensure_ascii=False, indent=2), media_diagnostics


def _run_decision(
    session: Any,
    payload: dict[str, Any],
    context: str,
    media: Any,
    seed: int,
    profile: Any,
    *,
    reuse_kv_cache: bool | None = None,
    media_before_prompt: bool | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = {
        "question": payload["question"],
        "context": context,
        "answers": payload["answer"],
        "media": media,
    }
    if isinstance(session, LlamaCppServerSession) and seed >= 0:
        request["seed"] = seed
    if profile is not None:
        request["model_profile"] = profile
    if reuse_kv_cache is not None:
        request["reuse_kv_cache"] = reuse_kv_cache
    if media_before_prompt is not None:
        request["media_before_prompt"] = media_before_prompt

    started = perf_counter()
    decision = session.decide(**request)
    elapsed_ms = (perf_counter() - started) * 1000
    if isinstance(decision, LlamaCppDecisionResult):
        selected = decision.selected
        probabilities = decision.probabilities
        metrics = dict(decision.metrics)
        media_diagnostics = dict(decision.media_diagnostics)
    else:
        selected, probabilities = decision
        metrics = {"decision_duration_ms": elapsed_ms}
        media_diagnostics = {
            "schema_version": 1,
            "backend": "llama.cpp-decision",
            "requested": media.manifest(),
            "evaluated": {"image_count": 0, "audio_count": 0, "video_count": 0},
            "mtmd": {
                "completion_succeeded": True,
                "all_media_evaluated": not media.items,
                "verification": "no_media" if not media.items else "unverified",
            },
            "model_unloaded_after_response": False,
        }
    ordered_probabilities = {
        answer: probabilities[answer] for answer in payload["answer"]
    }
    return selected, ordered_probabilities, metrics, media_diagnostics


def _decision_output_values(
    result: tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]],
    seed: int,
    *,
    unloaded: bool,
) -> tuple[Any, str, str, dict[str, Any]]:
    selected, probabilities, metrics, media_diagnostics = result
    metrics.update(operation="decide", seed=seed, model_unloaded=unloaded)
    if isinstance(metrics.get("session"), dict):
        metrics["session"]["unload_required"] = not unloaded
    media_diagnostics["model_unloaded_after_response"] = unloaded
    return (
        selected,
        json.dumps(probabilities, ensure_ascii=False, indent=2),
        json.dumps(metrics, ensure_ascii=False, indent=2),
        media_diagnostics,
    )


def _decision_common_values(
    system: Any, seed: Any, model_profile: Any, session_unload: Any
) -> tuple[str, int, Any, bool]:
    resolved_system = unwrap_required_scalar("system", system)
    if not isinstance(resolved_system, str):
        raise InputNormalizationError("system must be a string.")
    resolved_seed = int(unwrap_required_scalar("seed", seed))
    if resolved_seed < -1 or resolved_seed > 0xFFFFFFFF:
        raise InputNormalizationError("seed must be between -1 and 4294967295.")
    profile_value = unwrap_optional_scalar("model_profile", model_profile, None)
    profile = (
        normalize_compact_model_profile(profile_value)
        if profile_value is not None
        else None
    )
    unload = bool(unwrap_required_scalar("session_unload", session_unload))
    return resolved_system, resolved_seed, profile, unload


def _sequential_decision_output(
    session: Any,
    results: list[tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]],
    seed: int,
    *,
    unload: bool,
) -> io.NodeOutput:
    if unload:
        session.close()
    output_values = [
        _decision_output_values(
            result, seed, unloaded=unload and index == len(results) - 1
        )
        for index, result in enumerate(results)
    ]
    return io.NodeOutput(
        [values[0] for values in output_values],
        [values[1] for values in output_values],
        [values[2] for values in output_values],
        [values[3] for values in output_values],
        session,
    )


class LlamaCppCreateQuestionFromInputNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_CreateQuestionFromInput",
            display_name="[llama.cpp] Build Question (Prefill)",
            category=f"{BASE_CATEGORY}/decision/prefill",
            description=(
                "Combines one STRING question with a flat ComfyUI STRING list of answers."
            ),
            is_input_list=True,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "question",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                    force_input=True,
                ),
                io.String.Input("answer", force_input=True),
            ],
            outputs=[LlamaCppQuestionType.Output("question")],
        )

    @classmethod
    def execute(cls, question: Any, answer: Any) -> io.NodeOutput:
        return io.NodeOutput(_question_from_input_lists(question, answer))


class LlamaCppBuildQuestionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_BuildQuestion",
            display_name="[llama.cpp] Build Question",
            category=f"{BASE_CATEGORY}/decision/system_one",
            description="Builds a choice or ordered score question for System One.",
            is_input_list=True,
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "question",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                    force_input=True,
                ),
                io.String.Input(
                    "answer",
                    force_input=True,
                    tooltip=(
                        "Choice requires 2–26 unique options; score requires 2–10 "
                        "unique levels in lowest-to-highest order."
                    ),
                ),
                io.Combo.Input("type", options=["choice", "score"], default="choice"),
            ],
            outputs=[LlamaCppQuestionType.Output("question")],
        )

    @classmethod
    def execute(cls, question: Any, answer: Any, type: Any) -> io.NodeOutput:
        return io.NodeOutput(
            _systemone_question_from_input_lists(question, answer, type)
        )


class LlamaCppBuildNoulNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_BuildNoul",
            display_name="[llama.cpp] Build Noul",
            category=f"{BASE_CATEGORY}/decision/system_one",
            description="Builds a true-or-false probability question for System One.",
            is_experimental=True,
            inputs=[
                io.String.Input(
                    "question",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                ),
            ],
            outputs=[LlamaCppQuestionType.Output("question")],
        )

    @classmethod
    def execute(cls, question: Any) -> io.NodeOutput:
        payload = normalize_systemone_question(
            {"type": "noul", "question": unwrap_required_scalar("question", question)}
        )
        return io.NodeOutput(payload)


class LlamaCppDecideSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_Decide",
            display_name="[llama.cpp] Prefill Decide",
            category=f"{BASE_CATEGORY}/decision/prefill",
            description=(
                "Scores letter-token choices with a Native Session prefill or a Runtime "
                "or Connect Session's constrained llama-server completion. Optional media "
                "is added to the context; question and answers remain text-only."
            ),
            is_input_list=True,
            not_idempotent=True,
            is_experimental=True,
            inputs=_decision_inputs(),
            outputs=_decision_outputs(),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        seed: Any = -1,
        model_profile: Any = None,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise InputNormalizationError(
                "Decide requires [llama.cpp] Create Native Session or "
                "a llama.cpp server session."
            )
        validated = _validated_question_payload(
            unwrap_required_scalar("question", question)
        )
        resolved_system = unwrap_required_scalar("system", system)
        resolved_context = unwrap_required_scalar("context", context)
        if not isinstance(resolved_system, str) or not isinstance(
            resolved_context, str
        ):
            raise InputNormalizationError("system and context must be strings.")
        profile_value = unwrap_optional_scalar("model_profile", model_profile, None)
        profile = (
            normalize_compact_model_profile(profile_value)
            if profile_value is not None
            else None
        )
        decision_context = _decision_context(resolved_system, resolved_context)
        media = normalize_media(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=bool(
                unwrap_required_scalar("video_with_audio", video_with_audio)
            ),
            audio_sample_rate=16_000,
            audio_channels=1,
        )
        resolved_seed = int(unwrap_required_scalar("seed", seed))
        if resolved_seed < -1 or resolved_seed > 0xFFFFFFFF:
            raise InputNormalizationError("seed must be between -1 and 4294967295.")
        result = _run_decision(
            resolved_session,
            validated,
            decision_context,
            media,
            resolved_seed,
            profile,
        )
        unloaded = bool(unwrap_required_scalar("session_unload", session_unload))
        if unloaded:
            resolved_session.close()
        return io.NodeOutput(
            *_decision_output_values(result, resolved_seed, unloaded=unloaded),
            resolved_session,
        )


class LlamaCppDecideSystemOneNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_DecideSystemOne",
            display_name="[llama.cpp] System One Decide",
            category=f"{BASE_CATEGORY}/decision/system_one",
            description=(
                "Answers one typed choice, score, or noul question with a Runtime or "
                "Connect Session using llama.cpp's System One API."
            ),
            not_idempotent=True,
            is_experimental=True,
            inputs=_systemone_inputs(),
            outputs=_systemone_outputs(),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, LlamaCppServerSession):
            raise InputNormalizationError(
                "System One Decide requires a Runtime or Connect Session; "
                "Native sessions are unsupported."
            )
        resolved_system = unwrap_required_scalar("system", system)
        resolved_context = unwrap_required_scalar("context", context)
        if not isinstance(resolved_system, str) or not isinstance(
            resolved_context, str
        ):
            raise InputNormalizationError("system and context must be strings.")
        payload = normalize_systemone_question(
            unwrap_required_scalar("question", question)
        )
        unload = unwrap_required_scalar("session_unload", session_unload)
        if not isinstance(unload, bool):
            raise InputNormalizationError("session_unload must be a boolean.")

        result = resolved_session.systemone(
            state=_decision_context(resolved_system, resolved_context),
            question=payload,
            media=normalize_media(images=images),
        )
        if unload:
            resolved_session.close()
        return io.NodeOutput(
            *_systemone_output_values(result, unloaded=unload),
            resolved_session,
        )


def _run_systemone_sequence(
    session: LlamaCppServerSession,
    system: str,
    contexts: list[str],
    questions: list[dict[str, Any]],
    media_bundles: list[Any],
    unload: bool,
) -> io.NodeOutput:
    contexts = _broadcast_values(contexts, len(media_bundles), "context")
    questions = _broadcast_values(questions, len(media_bundles), "question")
    try:
        results = [
            session.systemone(
                state=_decision_context(system, item_context),
                question=payload,
                media=media,
            )
            for media, item_context, payload in zip(
                media_bundles, contexts, questions, strict=True
            )
        ]
    finally:
        if unload:
            session.close()

    outputs = [
        _systemone_output_values(
            result, unloaded=unload and index == len(results) - 1
        )
        for index, result in enumerate(results)
    ]
    return io.NodeOutput(
        [values[0] for values in outputs],
        [values[1] for values in outputs],
        [values[2] for values in outputs],
        session,
    )


class LlamaCppDecideSystemOneMediaSequentialNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_DecideSystemOneMediaSequential",
            display_name="[llama.cpp] System One Decide (Media Sequential)",
            category=f"{BASE_CATEGORY}/decision/system_one",
            description=(
                "Runs one System One decision per image, sharing one context and "
                "question or accepting one for each image."
            ),
            is_input_list=True,
            not_idempotent=True,
            is_experimental=True,
            inputs=_systemone_inputs(),
            outputs=_systemone_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, LlamaCppServerSession):
            raise InputNormalizationError(
                "System One requires a Runtime or Connect Session; "
                "Native sessions are unsupported."
            )
        resolved_system = unwrap_required_scalar("system", system)
        if not isinstance(resolved_system, str):
            raise InputNormalizationError("system must be a string.")
        unload = unwrap_required_scalar("session_unload", session_unload)
        if not isinstance(unload, bool):
            raise InputNormalizationError("session_unload must be a boolean.")

        return _run_systemone_sequence(
            resolved_session,
            resolved_system,
            _flat_contexts(context),
            _flat_systemone_question_payloads(question),
            _sequential_media_bundles(images=images),
            unload,
        )


class LlamaCppDecideSystemOnePromptSequentialNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_DecideSystemOnePromptSequential",
            display_name="[llama.cpp] System One Decide (Prompt Sequential)",
            category=f"{BASE_CATEGORY}/decision/system_one",
            description=(
                "Runs paired context and question lists independently against the same "
                "complete image bundle."
            ),
            is_input_list=True,
            not_idempotent=True,
            is_experimental=True,
            inputs=_systemone_inputs(),
            outputs=_systemone_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, LlamaCppServerSession):
            raise InputNormalizationError(
                "System One requires a Runtime or Connect Session; "
                "Native sessions are unsupported."
            )
        resolved_system = unwrap_required_scalar("system", system)
        if not isinstance(resolved_system, str):
            raise InputNormalizationError("system must be a string.")
        unload = unwrap_required_scalar("session_unload", session_unload)
        if not isinstance(unload, bool):
            raise InputNormalizationError("session_unload must be a boolean.")

        contexts = _flat_contexts(context)
        questions = _flat_systemone_question_payloads(question)
        count = max(len(contexts), len(questions))
        return _run_systemone_sequence(
            resolved_session,
            resolved_system,
            contexts,
            questions,
            [normalize_media(images=images)] * count,
            unload,
        )


class LlamaCppExtractAnswerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_ExtractAnswer",
            display_name="[llama.cpp] Extract Answer",
            category=f"{BASE_CATEGORY}/decision",
            description=(
                "Value is the selected choice probability, the expected 0-based score "
                "level, or P(true) for noul. Probabilities follow input-answer order; "
                "noul uses [false, true]."
            ),
            is_experimental=True,
            inputs=[LlamaCppAnswerType.Input("answer")],
            outputs=[
                io.String.Output(
                    "selected",
                    tooltip="Selected choice option; empty for score and noul.",
                ),
                io.Float.Output(
                    "value",
                    tooltip=(
                        "Choice: selected option probability. Score: expected "
                        "0-based level index. Noul: probability of true."
                    ),
                ),
                io.Boolean.Output(
                    "noul",
                    tooltip=(
                        "For noul answers, true when P(true) is at least 0.5; "
                        "None for choice and score answers."
                    ),
                ),
                io.Float.Output(
                    "probabilities",
                    is_output_list=True,
                    tooltip=(
                        "Input-answer order for choice/score; [false, true] for noul."
                    ),
                ),
                io.String.Output(
                    "result_json",
                    tooltip="Complete raw per-question System One answer JSON.",
                ),
            ],
        )

    @classmethod
    def execute(cls, answer: Any) -> io.NodeOutput:
        values = _validated_answer_payload(unwrap_required_scalar("answer", answer))
        return io.NodeOutput(
            values["selected"],
            values["value"],
            values["noul"],
            values["probabilities"],
            values["result_json"],
        )


class LlamaCppDecideMediaSequentialNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_DecideMediaSequential",
            display_name="[llama.cpp] Prefill Decide (Media Sequential)",
            category=f"{BASE_CATEGORY}/decision/prefill",
            description=(
                "Runs one decision per atomic IMAGE, AUDIO, or VIDEO bundle in modality "
                "order. A single context or question is shared, or provide one per item."
            ),
            is_input_list=True,
            not_idempotent=True,
            is_experimental=True,
            inputs=_decision_inputs(),
            outputs=_decision_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        seed: Any = -1,
        model_profile: Any = None,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise InputNormalizationError(
                "Decide requires [llama.cpp] Create Native Session or "
                "a llama.cpp server session."
            )
        contexts = _flat_contexts(context)
        questions = _flat_question_payloads(question)
        resolved_system, resolved_seed, profile, unload = _decision_common_values(
            system, seed, model_profile, session_unload
        )
        bundles = _sequential_media_bundles(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=bool(
                unwrap_required_scalar("video_with_audio", video_with_audio)
            ),
        )
        contexts = _broadcast_values(contexts, len(bundles), "context")
        questions = _broadcast_values(questions, len(bundles), "question")
        results = [
            _run_decision(
                resolved_session,
                payload,
                _decision_context(resolved_system, item_context),
                bundle,
                resolved_seed,
                profile,
            )
            for bundle, item_context, payload in zip(
                bundles, contexts, questions, strict=True
            )
        ]
        return _sequential_decision_output(
            resolved_session,
            results,
            resolved_seed,
            unload=unload,
        )


class LlamaCppDecidePromptSequentialNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_DecidePromptSequential",
            display_name="[llama.cpp] Prefill Decide (Prompt Sequential)",
            category=f"{BASE_CATEGORY}/decision/prefill",
            description=(
                "Runs paired context and question lists independently against the same "
                "complete media bundle. Common-prefix KV reuse is attempted by default."
            ),
            is_input_list=True,
            not_idempotent=True,
            is_experimental=True,
            inputs=_decision_inputs(include_reuse_kv_cache=True),
            outputs=_decision_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        context: Any,
        question: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        seed: Any = -1,
        model_profile: Any = None,
        reuse_kv_cache: Any = True,
        session_unload: Any = False,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise InputNormalizationError(
                "Decide requires [llama.cpp] Create Native Session or "
                "a llama.cpp server session."
            )
        contexts = _flat_contexts(context)
        questions = _flat_question_payloads(question)
        count = max(len(contexts), len(questions))
        contexts = _broadcast_values(contexts, count, "context")
        questions = _broadcast_values(questions, count, "question")
        resolved_system, resolved_seed, profile, unload = _decision_common_values(
            system, seed, model_profile, session_unload
        )
        reuse_value = unwrap_required_scalar("reuse_kv_cache", reuse_kv_cache)
        if not isinstance(reuse_value, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        video_audio = bool(unwrap_required_scalar("video_with_audio", video_with_audio))
        media = normalize_media(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=video_audio,
            audio_sample_rate=16_000,
            audio_channels=1,
        )
        results = [
            _run_decision(
                resolved_session,
                payload,
                _decision_context(resolved_system, item_context),
                media,
                resolved_seed,
                profile,
                reuse_kv_cache=reuse_value,
                media_before_prompt=True,
            )
            for item_context, payload in zip(contexts, questions, strict=True)
        ]
        return _sequential_decision_output(
            resolved_session,
            results,
            resolved_seed,
            unload=unload,
        )


__all__ = [
    "LlamaCppBuildNoulNode",
    "LlamaCppBuildQuestionNode",
    "LlamaCppCreateQuestionFromInputNode",
    "LlamaCppDecideSystemOneNode",
    "LlamaCppDecideSystemOneMediaSequentialNode",
    "LlamaCppDecideSystemOnePromptSequentialNode",
    "LlamaCppDecideMediaSequentialNode",
    "LlamaCppDecidePromptSequentialNode",
    "LlamaCppDecideSessionNode",
    "LlamaCppExtractAnswerNode",
    "LlamaCppAnswerType",
    "LlamaCppQuestionType",
    "MAX_ANSWERS",
    "make_systemone_question_payload",
    "make_question_payload",
]
