from __future__ import annotations

import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from comfy_api.v0_0_2 import io
except ImportError:  # pragma: no cover - compatibility with newer ComfyUI builds
    from comfy_api.latest import io

from ..backends.llama_cpp import HANDLER_NAMES, LlamaCppSession, _resolve_file
from ..backends.llama_cpp_server import (
    LlamaCppServerSession,
    OwnedLlamaCppServerSession,
    list_server_models,
)
from ..core import (
    InputNormalizationError,
    normalize_media,
    unwrap_optional_scalar,
    unwrap_required_scalar,
)
from ..llama_cpp.llama_cpp_runtime import (
    _llama_child_environment,
    _resolve_llama_executable,
    start_owned_llama_server,
)
from .llama_cpp_compact import (
    BASE_CATEGORY,
    DEFAULT_COMPACT_HARDWARE_PROFILE,
    LlamaCppHardwareRuntimeProfileType,
    LlamaCppModelProfileType,
    LlamaCppReasoningConfigType,
    LlamaCppSpeculativeConfigType,
    _sequential_media_bundles,
    _sequential_prompts,
    build_compact_session_kwargs,
    normalize_compact_hardware_profile,
    normalize_compact_model_profile,
    normalize_compact_speculative,
    normalize_reasoning_config,
)
from .llama_cpp_diagnostics import LlamaCppMediaDiagnosticsType
from .llama_cpp_generate import (
    NO_MMPROJ_OPTION,
    _gguf_options,
    _resolve_gguf_selection,
)
from .llama_cpp_prefill import (
    LlamaCppPrefillProfileType,
    normalize_prefill_profile,
)

LlamaCppSessionType = io.Custom("OLLAMA_IMAGE_LIST_LLAMA_CPP_SESSION")


def _session_generate_inputs() -> list[Any]:
    return [
        LlamaCppSessionType.Input("session"),
        io.String.Input("system", default="", multiline=True, dynamic_prompts=False),
        io.String.Input("prompt", default="", multiline=True, dynamic_prompts=False),
        io.Int.Input("max_tokens", default=512, min=1, max=131_072, step=1),
        io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFF, step=1),
        io.String.Input("stop", default="", advanced=True),
        io.Image.Input("images", optional=True),
        io.Audio.Input("audio", optional=True),
        io.Video.Input("video", optional=True),
        io.Boolean.Input("video_with_audio", default=False),
        io.Boolean.Input(
            "session_unload",
            default=False,
            label_on="Unload",
            label_off="Keep",
            tooltip="Unload the session after generation completes.",
        ),
        LlamaCppModelProfileType.Input(
            "model_profile",
            optional=True,
            tooltip=(
                "Optional per-request sampling and reasoning override. The "
                "session handler is selected at creation and is not changed."
            ),
        ),
    ]


def _session_generate_outputs(*, is_output_list: bool = False) -> list[Any]:
    return [
        io.String.Output(
            "response", display_name="response", is_output_list=is_output_list
        ),
        io.String.Output(
            "thinking", display_name="thinking", is_output_list=is_output_list
        ),
        io.String.Output(
            "raw_json", display_name="raw_json", is_output_list=is_output_list
        ),
        io.String.Output(
            "metrics_json", display_name="metrics", is_output_list=is_output_list
        ),
        LlamaCppMediaDiagnosticsType.Output(
            "media_diagnostics",
            display_name="media_diagnostics",
            is_output_list=is_output_list,
        ),
    ]


def _session_prompt_generate_inputs() -> list[Any]:
    inputs = _session_generate_inputs()
    inputs.insert(
        -2,
        io.Boolean.Input(
            "reuse_kv_cache",
            default=True,
            tooltip="Attempt common-prefix KV reuse across independent prompts.",
        ),
    )
    return inputs


def _session_sequence_outputs(
    session: Any, results: list[Any], *, unload_after_sequence: bool
) -> io.NodeOutput:
    if unload_after_sequence:
        session.close()
        results[-1].metrics["model_unloaded"] = True
        results[-1].metrics["session"]["unload_required"] = False
        results[-1].media_diagnostics["model_unloaded_after_response"] = True
    return io.NodeOutput(
        [result.response for result in results],
        [result.thinking for result in results],
        [json.dumps(result.raw, ensure_ascii=False, indent=2) for result in results],
        [
            json.dumps(result.metrics, ensure_ascii=False, indent=2)
            for result in results
        ],
        [result.media_diagnostics for result in results],
    )


def _runtime_server_arguments(
    *,
    model_path: Any,
    mmproj_path: Any,
    model_profile: Any = None,
    hardware_profile: Any = None,
    prefill_profile: Any = None,
    reasoning: Any = None,
    speculative: Any = None,
    custom_chat_template: Any = None,
    n_ctx: Any,
    verbose: Any,
    reasoning_preserve_supported: bool,
) -> tuple[list[str], str | None, str]:
    model = _resolve_file(
        _resolve_gguf_selection(
            str(unwrap_required_scalar("model_path", model_path)),
            label="model GGUF",
            required=True,
        ),
        label="model GGUF",
        required=True,
    )
    mmproj_selection = unwrap_optional_scalar(
        "mmproj_path", mmproj_path, NO_MMPROJ_OPTION
    )
    mmproj = _resolve_file(
        _resolve_gguf_selection(
            str(mmproj_selection), label="mmproj GGUF", required=False
        ),
        label="mmproj GGUF",
        required=False,
    )
    if model is None:
        raise InputNormalizationError("model GGUF is required.")

    profile = None
    profile_reasoning_mode = None
    if model_profile is not None:
        profile = normalize_compact_model_profile(
            unwrap_required_scalar("model_profile", model_profile)
        )
        profile_reasoning_mode = profile.pop("recommended_reasoning_mode")
        profile.pop("handler")
    selected_chat_template = unwrap_optional_scalar(
        "custom_chat_template", custom_chat_template, ""
    )
    if not isinstance(selected_chat_template, str):
        raise InputNormalizationError("custom_chat_template must be a string.")

    hardware = normalize_compact_hardware_profile(
        DEFAULT_COMPACT_HARDWARE_PROFILE
        if hardware_profile is None
        else unwrap_required_scalar("hardware_profile", hardware_profile)
    )
    capacity = (
        normalize_prefill_profile(
            unwrap_required_scalar("prefill_profile", prefill_profile)
        )
        if prefill_profile is not None
        else None
    )
    reasoning_config = normalize_reasoning_config(
        {
            "reasoning_mode": "auto",
            "reasoning_effort": "auto",
            "max_reasoning_tokens": 0,
            "preserve_thinking": False,
        }
        if reasoning is None
        else unwrap_required_scalar("reasoning", reasoning)
    )
    reasoning_mode = reasoning_config["reasoning_mode"]
    if (
        profile_reasoning_mode is not None
        and profile_reasoning_mode != "auto"
        and reasoning_mode != "auto"
        and reasoning_mode != profile_reasoning_mode
    ):
        raise InputNormalizationError(
            "The selected [llama.cpp] Model Profile requires reasoning_mode="
            f"{profile_reasoning_mode}, but [llama.cpp] Thinking / Reasoning Profile "
            f"requests reasoning_mode={reasoning_mode}."
        )
    if reasoning_mode == "auto" and profile_reasoning_mode is not None:
        reasoning_mode = profile_reasoning_mode

    speculative_config = normalize_compact_speculative(
        {"kind": "off"}
        if speculative is None
        else unwrap_required_scalar("speculative", speculative)
    )
    resolved_ctx = unwrap_required_scalar("n_ctx", n_ctx)
    if type(resolved_ctx) is not int or not 512 <= resolved_ctx <= 1_048_576:
        raise InputNormalizationError(
            "n_ctx must be an integer between 512 and 1048576."
        )
    verbose_value = unwrap_optional_scalar("verbose", verbose, False)
    if not isinstance(verbose_value, bool):
        raise InputNormalizationError("verbose must be a boolean.")

    arguments = ["--model", model, "--ctx-size", str(resolved_ctx)]
    if mmproj:
        arguments.extend(("--mmproj", mmproj))
    if profile is not None:
        for name, option in (
            ("temperature", "--temp"),
            ("top_p", "--top-p"),
            ("top_k", "--top-k"),
            ("min_p", "--min-p"),
            ("presence_penalty", "--presence-penalty"),
            ("repeat_penalty", "--repeat-penalty"),
        ):
            arguments.extend((option, str(profile[name])))
    if capacity is not None:
        arguments.extend(("--batch-size", str(capacity["n_batch"])))
        if capacity["n_ubatch"] > 0:
            arguments.extend(("--ubatch-size", str(capacity["n_ubatch"])))
    kv_cache_types = {"FP16": "f16", "Q8_0": "q8_0", "Q4_0": "q4_0"}
    arguments.extend(
        (
            "--cache-type-k",
            kv_cache_types[hardware["type_k"]],
            "--cache-type-v",
            kv_cache_types[hardware["type_v"]],
        )
    )
    gpu_layers = {"all": "all", "auto": "auto", "cpu": "0"}[hardware["gpu_layers"]]
    arguments.extend(
        (
            "--gpu-layers",
            gpu_layers,
            "--main-gpu",
            str(hardware["main_gpu"]),
            "--flash-attn",
            {"auto": "auto", "enabled": "on", "disabled": "off"}[
                hardware["flash_attention"]
            ],
            "--load-mode",
            "mmap" if hardware["use_mmap"] else "none",
            "--reasoning",
            reasoning_mode,
        )
    )
    if hardware["n_threads"] > 0:
        arguments.extend(("--threads", str(hardware["n_threads"])))
    if reasoning_mode != "off":
        if reasoning_config["reasoning_effort"] != "auto":
            arguments.extend(
                ("--reasoning-effort", reasoning_config["reasoning_effort"])
            )
        if reasoning_config["max_reasoning_tokens"] > 0:
            arguments.extend(
                ("--reasoning-budget", str(reasoning_config["max_reasoning_tokens"]))
            )
    if reasoning is not None:
        arguments.extend(
            (
                "--chat-template-kwargs",
                json.dumps(
                    {"preserve_thinking": reasoning_config["preserve_thinking"]}
                ),
            )
        )
        if reasoning_preserve_supported:
            arguments.append(
                "--reasoning-preserve"
                if reasoning_config["preserve_thinking"]
                else "--no-reasoning-preserve"
            )
    if capacity is not None:
        for name in ("image_min_tokens", "image_max_tokens"):
            value = capacity[name]
            if value > 0:
                option = (
                    "--image-min-tokens"
                    if name == "image_min_tokens"
                    else "--image-max-tokens"
                )
                arguments.extend((option, str(value)))

    if speculative_config["kind"] == "native":
        config = speculative_config["config"]
        spec_type = config["spec_type"]
        if spec_type != "none":
            arguments.extend(
                (
                    "--spec-type",
                    spec_type,
                    "--spec-draft-n-max",
                    str(config["draft_n_max"]),
                    "--spec-draft-p-min",
                    str(config["draft_p_min"]),
                    "--spec-draft-ngl",
                    str(config["draft_n_gpu_layers"]),
                    "--spec-draft-backend-sampling"
                    if config["draft_backend_sampling"]
                    else "--no-spec-draft-backend-sampling",
                )
            )
            draft_required = spec_type in {"draft-dflash", "draft-dspark"} or (
                spec_type == "draft-mtp" and config["mtp_provider"] == "external"
            )
            if draft_required:
                draft = _resolve_file(
                    _resolve_gguf_selection(
                        config["draft_model"], label="draft model GGUF", required=True
                    ),
                    label="draft model GGUF",
                    required=True,
                )
                if draft is None:
                    raise InputNormalizationError("draft model GGUF is required.")
                arguments.extend(("--spec-draft-model", draft))
    elif speculative_config["kind"] == "ngram":
        config = speculative_config["config"]
        ngram_mode = f"ngram-map-{config['ngram_mode']}"
        option_prefix = f"--spec-{ngram_mode}"
        arguments.extend(
            (
                "--spec-type",
                ngram_mode,
                f"{option_prefix}-size-n",
                str(config["ngram_size"]),
                f"{option_prefix}-size-m",
                str(config["num_pred_tokens"]),
                f"{option_prefix}-min-hits",
                str(config["ngram_min_hits"]),
            )
        )

    if verbose_value:
        arguments.append("--verbose")
    return arguments, selected_chat_template or None, model


def _supports_reasoning_preserve(executable: str, *, internal: bool) -> bool:
    try:
        result = subprocess.run(
            [executable, "server", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=False,
            env=_llama_child_environment(Path(executable), internal=internal),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return b"--reasoning-preserve" in (result.stdout or b"")


class LlamaCppConnectSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_ConnectSession",
            display_name="[llama.cpp] Connect Session",
            category=f"{BASE_CATEGORY}/session",
            description=(
                "Creates a session handle for a llama.cpp server. Connect fetches model IDs; "
                "the saved model string remains the value used for generation."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "url",
                    default="http://127.0.0.1:9931",
                    tooltip="llama.cpp server base URL. Only HTTP and HTTPS are accepted.",
                ),
                io.Combo.Input(
                    "available_models",
                    options=[],
                    default="",
                    tooltip="Models reported by the configured llama.cpp server.",
                ),
                io.String.Input(
                    "model",
                    default="",
                    tooltip=(
                        "Saved model ID used for generation. Selecting available_models "
                        "copies its ID here."
                    ),
                ),
                io.String.Input(
                    "api_key",
                    display_name="api_key",
                    default="",
                    tooltip="Optional API key for the configured llama.cpp server.",
                ),
            ],
            outputs=[LlamaCppSessionType.Output("session", display_name="session")],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs: Any) -> int:
        return time.monotonic_ns()

    @classmethod
    def validate_inputs(cls, available_models: str) -> bool:
        del available_models
        return True

    @classmethod
    def execute(
        cls, url: Any, available_models: Any, model: Any, api_key: Any = ""
    ) -> io.NodeOutput:
        resolved_url = unwrap_required_scalar("url", url)
        resolved_model = unwrap_required_scalar("model", model)
        resolved_api_key = unwrap_optional_scalar("api_key", api_key, "")
        if not isinstance(resolved_api_key, str):
            raise InputNormalizationError("api_key must be a string.")
        del available_models
        return io.NodeOutput(
            LlamaCppServerSession(
                url=str(resolved_url),
                model=str(resolved_model),
                api_key=resolved_api_key or None,
            )
        )


class LlamaCppCreateSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_options, mmproj_options = _gguf_options()
        return io.Schema(
            node_id="LlamaCppMtmd_CreateSession",
            display_name="[llama.cpp] Create Native Session",
            category=f"{BASE_CATEGORY}/session",
            description=(
                "Keeps one Llama.cpp model resident until Llama.cpp Unload Session. "
                "Sessions left open during an interrupted workflow unload at prompt end. "
                "Place this before Start Loop and carry its session through End Loop."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=[
                io.Combo.Input(
                    "model_path", options=model_options, default=model_options[0]
                ),
                io.Combo.Input(
                    "mmproj_path", options=mmproj_options, default=NO_MMPROJ_OPTION
                ),
                LlamaCppModelProfileType.Input("model_profile", optional=True),
                io.Combo.Input(
                    "custom_handler",
                    options=list(HANDLER_NAMES),
                    default="auto",
                    tooltip=(
                        "A specific handler overrides Model Profile; auto keeps the "
                        "profile's handler."
                    ),
                ),
                io.String.Input(
                    "custom_chat_template",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional custom Jinja template for this native session. "
                        "When disconnected, the GGUF metadata template is used."
                    ),
                ),
                LlamaCppHardwareRuntimeProfileType.Input(
                    "hardware_profile", optional=True
                ),
                LlamaCppReasoningConfigType.Input("reasoning", optional=True),
                LlamaCppSpeculativeConfigType.Input("speculative", optional=True),
                LlamaCppPrefillProfileType.Input("prefill_profile", optional=True),
                io.Int.Input("n_ctx", default=8_192, min=512, max=1_048_576, step=512),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[LlamaCppSessionType.Output("session", display_name="session")],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs: Any) -> int:
        return time.monotonic_ns()

    @classmethod
    def execute(cls, **values: Any) -> io.NodeOutput:
        custom_handler = unwrap_required_scalar(
            "custom_handler", values.pop("custom_handler", "auto")
        )
        if custom_handler not in HANDLER_NAMES:
            raise InputNormalizationError(
                f"custom_handler must be one of {', '.join(HANDLER_NAMES)}."
            )
        configuration = build_compact_session_kwargs(**values)
        if custom_handler != "auto":
            configuration["handler"] = custom_handler
        return io.NodeOutput(LlamaCppSession(**configuration))


class LlamaCppCreateRuntimeSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_options, mmproj_options = _gguf_options()
        return io.Schema(
            node_id="LlamaCppMtmd_CreateRuntimeSession",
            display_name="[llama.cpp] Create Runtime Session",
            category=f"{BASE_CATEGORY}/session",
            description=(
                "Starts a workflow-owned local llama.cpp server and keeps its model "
                "loaded until Unload Session or prompt-end cleanup. The session "
                "owns its server API key."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=[
                io.Combo.Input(
                    "model_path", options=model_options, default=model_options[0]
                ),
                io.Combo.Input(
                    "mmproj_path", options=mmproj_options, default=NO_MMPROJ_OPTION
                ),
                LlamaCppModelProfileType.Input("model_profile", optional=True),
                io.String.Input(
                    "custom_chat_template",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Optional custom Jinja template for this server session. "
                        "When connected, it overrides the GGUF metadata template; "
                        "when disconnected, the GGUF template is used."
                    ),
                ),
                LlamaCppHardwareRuntimeProfileType.Input(
                    "hardware_profile", optional=True
                ),
                LlamaCppReasoningConfigType.Input("reasoning", optional=True),
                LlamaCppSpeculativeConfigType.Input("speculative", optional=True),
                LlamaCppPrefillProfileType.Input("prefill_profile", optional=True),
                io.Int.Input("n_ctx", default=8_192, min=512, max=1_048_576, step=512),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[LlamaCppSessionType.Output("session", display_name="session")],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs: Any) -> int:
        return time.monotonic_ns()

    @classmethod
    def execute(cls, **values: Any) -> io.NodeOutput:
        executable, source = _resolve_llama_executable()
        if executable is None:
            raise RuntimeError(
                "Could not find a 'llama' executable on PATH or in the completed "
                "Internal llama.cpp runtime installation."
            )

        internal = source == "internal"
        arguments, custom_chat_template, model = _runtime_server_arguments(
            **values,
            reasoning_preserve_supported=_supports_reasoning_preserve(
                executable, internal=internal
            ),
        )
        api_key = secrets.token_urlsafe(32)
        template_path: Path | None = None
        if custom_chat_template:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".jinja",
                delete=False,
            ) as template_file:
                template_file.write(custom_chat_template)
                template_path = Path(template_file.name)
            arguments.extend(("--jinja", "--chat-template-file", str(template_path)))

        def cleanup_template() -> None:
            if template_path is not None:
                template_path.unlink(missing_ok=True)

        process = None
        try:
            process = start_owned_llama_server(
                executable, arguments, internal=internal, api_key=api_key
            )
            session = OwnedLlamaCppServerSession(
                url=process.url,
                model=model,
                api_key=api_key,
                process=process,
                cleanup=cleanup_template,
            )
        except BaseException:
            if process is not None:
                process.close()
            cleanup_template()
            raise

        try:
            models = list_server_models(url=process.url, api_key=api_key)
            if len(models) != 1:
                raise RuntimeError(
                    "The workflow-owned llama.cpp server must report exactly one model."
                )
            session.model = models[0]
        except BaseException:
            session.close()
            raise
        return io.NodeOutput(session)


class LlamaCppSessionGenerateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_Generate",
            display_name="[llama.cpp] Generate",
            category=f"{BASE_CATEGORY}/generate",
            description=(
                "Runs one request on a local or server-backed llama.cpp session and "
                "carries it forward."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=_session_generate_inputs(),
            outputs=_session_generate_outputs(),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        prompt: Any,
        max_tokens: Any,
        seed: Any,
        stop: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        session_unload: Any = False,
        model_profile: Any = None,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise TypeError(
                "session must be a Llama.cpp Create or Connect Session output."
            )
        profile_value = unwrap_optional_scalar("model_profile", model_profile, None)
        profile = (
            normalize_compact_model_profile(profile_value)
            if profile_value is not None
            else None
        )
        bundle = normalize_media(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=bool(
                unwrap_required_scalar("video_with_audio", video_with_audio)
            ),
            audio_sample_rate=16_000,
            audio_channels=1,
        )
        request = {
            "system": str(unwrap_required_scalar("system", system)),
            "prompt": str(unwrap_required_scalar("prompt", prompt)),
            "media": bundle,
            "max_tokens": int(unwrap_required_scalar("max_tokens", max_tokens)),
            "seed": int(unwrap_required_scalar("seed", seed)),
            "stop": str(unwrap_required_scalar("stop", stop)),
        }
        if profile is not None:
            request["model_profile"] = profile
        result = resolved_session.generate(**request)
        if bool(unwrap_required_scalar("session_unload", session_unload)):
            resolved_session.close()
            result.metrics["model_unloaded"] = True
            result.metrics["session"]["unload_required"] = False
            result.media_diagnostics["model_unloaded_after_response"] = True
        return io.NodeOutput(
            result.response,
            result.thinking,
            json.dumps(result.raw, ensure_ascii=False, indent=2),
            json.dumps(result.metrics, ensure_ascii=False, indent=2),
            result.media_diagnostics,
        )


class LlamaCppSessionSequentialGenerateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_GenerateSequential",
            display_name="[llama.cpp] Generate (Media Sequential)",
            category=f"{BASE_CATEGORY}/generate",
            description=(
                "Runs one independent IMAGE, AUDIO, or VIDEO request at a time on "
                "the connected llama.cpp session and returns parallel output lists. "
                "A single prompt is shared across items; a prompt list pairs with them."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=_session_generate_inputs(),
            outputs=_session_generate_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        prompt: Any,
        max_tokens: Any,
        seed: Any,
        stop: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        session_unload: Any = False,
        model_profile: Any = None,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise TypeError(
                "session must be a Llama.cpp Create or Connect Session output."
            )

        bundles = _sequential_media_bundles(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=bool(
                unwrap_required_scalar("video_with_audio", video_with_audio)
            ),
        )
        raw_prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        if not any(bundle.items for bundle in bundles) and len(raw_prompts) > 1:
            bundles *= len(raw_prompts)
        prompts = _sequential_prompts(prompt, len(bundles))

        profile_value = unwrap_optional_scalar("model_profile", model_profile, None)
        profile = (
            normalize_compact_model_profile(profile_value)
            if profile_value is not None
            else None
        )
        unload_after_sequence = bool(
            unwrap_required_scalar("session_unload", session_unload)
        )
        common_request = {
            "system": str(unwrap_required_scalar("system", system)),
            "max_tokens": int(unwrap_required_scalar("max_tokens", max_tokens)),
            "seed": int(unwrap_required_scalar("seed", seed)),
            "stop": str(unwrap_required_scalar("stop", stop)),
        }
        if profile is not None:
            common_request["model_profile"] = profile

        results = [
            resolved_session.generate(
                **common_request,
                prompt=item_prompt,
                media=bundle,
            )
            for bundle, item_prompt in zip(bundles, prompts, strict=True)
        ]
        return _session_sequence_outputs(
            resolved_session,
            results,
            unload_after_sequence=unload_after_sequence,
        )


class LlamaCppSessionPromptSequentialGenerateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_GeneratePromptSequential",
            display_name="[llama.cpp] Generate (Prompt Sequential)",
            category=f"{BASE_CATEGORY}/generate",
            description=(
                "Runs each prompt independently against the same complete media "
                "bundle and returns five parallel output lists. Common-prefix KV "
                "reuse is attempted by default."
            ),
            is_input_list=True,
            not_idempotent=True,
            inputs=_session_prompt_generate_inputs(),
            outputs=_session_generate_outputs(is_output_list=True),
        )

    @classmethod
    def execute(
        cls,
        session: Any,
        system: Any,
        prompt: Any,
        max_tokens: Any,
        seed: Any,
        stop: Any,
        images: Any = None,
        audio: Any = None,
        video: Any = None,
        video_with_audio: Any = False,
        reuse_kv_cache: Any = True,
        session_unload: Any = False,
        model_profile: Any = None,
    ) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise TypeError(
                "session must be a Llama.cpp Create or Connect Session output."
            )

        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        if not prompts:
            raise InputNormalizationError(
                "Prompt Sequential Generate requires at least one prompt."
            )
        if any(not isinstance(value, str) for value in prompts):
            raise InputNormalizationError(
                "Prompt Sequential Generate prompts must be flat strings."
            )

        bundle = normalize_media(
            images=images,
            audio=audio,
            video=video,
            video_with_audio=bool(
                unwrap_required_scalar("video_with_audio", video_with_audio)
            ),
            audio_sample_rate=16_000,
            audio_channels=1,
        )
        profile_value = unwrap_optional_scalar("model_profile", model_profile, None)
        profile = (
            normalize_compact_model_profile(profile_value)
            if profile_value is not None
            else None
        )
        reuse_value = unwrap_required_scalar("reuse_kv_cache", reuse_kv_cache)
        if not isinstance(reuse_value, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        unload_after_sequence = bool(
            unwrap_required_scalar("session_unload", session_unload)
        )
        common_request = {
            "system": str(unwrap_required_scalar("system", system)),
            "max_tokens": int(unwrap_required_scalar("max_tokens", max_tokens)),
            "seed": int(unwrap_required_scalar("seed", seed)),
            "stop": str(unwrap_required_scalar("stop", stop)),
        }
        if profile is not None:
            common_request["model_profile"] = profile

        results = [
            resolved_session.generate(
                **common_request,
                prompt=item_prompt,
                media=bundle,
                reuse_kv_cache=reuse_value,
                media_before_prompt=True,
            )
            for item_prompt in prompts
        ]
        return _session_sequence_outputs(
            resolved_session,
            results,
            unload_after_sequence=unload_after_sequence,
        )


class LlamaCppUnloadSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlamaCppMtmd_UnloadSession",
            display_name="[llama.cpp] Unload Session",
            category=f"{BASE_CATEGORY}/session",
            description=(
                "Unloads a local or server-backed Llama.cpp session. Connect session and "
                "a final End Loop result to timing so unloading runs after the loop."
            ),
            is_input_list=True,
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                LlamaCppSessionType.Input("session"),
                io.Custom("*").Input(
                    "timing",
                    optional=True,
                    tooltip="Optional execution dependency passed through unchanged.",
                ),
            ],
            outputs=[io.Custom("*").Output("timing")],
        )

    @classmethod
    def execute(cls, session: Any, timing: Any = None) -> io.NodeOutput:
        resolved_session = unwrap_required_scalar("session", session)
        if not isinstance(resolved_session, (LlamaCppSession, LlamaCppServerSession)):
            raise TypeError(
                "session must be a Llama.cpp Create or Connect Session output."
            )
        resolved_session.close()
        return io.NodeOutput(timing)


__all__ = [
    "LlamaCppConnectSessionNode",
    "LlamaCppCreateSessionNode",
    "LlamaCppCreateRuntimeSessionNode",
    "LlamaCppSessionGenerateNode",
    "LlamaCppSessionType",
    "LlamaCppUnloadSessionNode",
]
