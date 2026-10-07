from __future__ import annotations

import base64
import gc
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import Any

from ..core import BackendError, InputNormalizationError, MediaBundle
from ..llama_cpp.llama_cpp_session_cleanup import track_session, untrack_session

HANDLER_NAMES = (
    "auto",
    "generic",
    "gemma4",
    "qwen3_vl",
    "qwen25_vl",
    "qwen3_asr",
    "qwen35",
)
REASONING_STRENGTHS = ("auto", "low", "medium", "high", "xhigh")
_MAX_REASONING_BUDGET = 65536
_HANDLER_CLASSES = {
    "generic": "GenericMTMDChatHandler",
    "gemma4": "Gemma4ChatHandler",
    "qwen3_vl": "Qwen3VLChatHandler",
    "qwen25_vl": "Qwen25VLChatHandler",
    "qwen3_asr": "Qwen3ASRChatHandler",
    "qwen35": "Qwen35ChatHandler",
}
_FLASH_ATTN_TYPES = {"auto": -1, "disabled": 0, "enabled": 1}
_KV_CACHE_TYPE_IDS = {"FP16": 1, "Q8_0": 8, "Q4_0": 2}
_SPECULATIVE_TYPES = {"draft-dflash", "draft-dspark"}
_MTP_PROVIDERS = {"off", "external", "internal"}
_DEFAULT_N_UBATCH = 512
_NATIVE_EXECUTION_LOCK = RLock()
_DECISION_OUTPUT_RULE = (
    "Return exactly one choice label (A, B, ...). "
    "Do not include explanations or any other text."
)
_LOGGER = logging.getLogger(__name__)
_JAMEPENG_RELEASES_URL = "https://github.com/JamePeng/llama-cpp-python/releases/"
_VISION_INSTALL_GUIDE_URL = (
    "https://github.com/goodguy1963/ComfyUI-ThinkingLLM/blob/main/docs/"
    "LLAMA_CPP_PYTHON_VISION_INSTALL.md"
)
_NATIVE_SPECULATIVE_RELEASE_URL = _JAMEPENG_RELEASES_URL


def _fork_install_hint() -> str:
    return (
        "This node targets JamePeng's multimodal llama-cpp-python fork.\n"
        f"Installation guide: {_VISION_INSTALL_GUIDE_URL}\n"
        f"Prebuilt wheels: {_JAMEPENG_RELEASES_URL}"
    )


def _kv_cache_type_id(name: str, value: str) -> int:
    try:
        return _KV_CACHE_TYPE_IDS[value]
    except (KeyError, TypeError) as exc:
        raise InputNormalizationError(f"{name} must be FP16, Q8_0, or Q4_0.") from exc


@dataclass(frozen=True, slots=True)
class LlamaCppBindings:
    llama_class: type
    handlers: dict[str, type]
    jinja_formatter_class: type | None = None
    chat_formatter_to_handler: Any | None = None


@dataclass(frozen=True, slots=True)
class NativeSpeculativeBindings:
    spec_config: type
    speculative_type: type


@dataclass(frozen=True, slots=True)
class LlamaCppResult:
    response: str
    thinking: str
    raw: dict[str, Any]
    metrics: dict[str, Any]
    media_diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LlamaCppDecisionResult:
    selected: str
    probabilities: dict[str, float]
    metrics: dict[str, Any]
    media_diagnostics: dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        yield self.selected
        yield self.probabilities


def _close_resources(resources: tuple[Any, ...]) -> list[Exception]:
    """Close only explicitly caller-owned resources through public close()."""
    errors: list[Exception] = []
    for resource in resources:
        if resource is None:
            continue
        close_resource = getattr(resource, "close", None)
        if not callable(close_resource):
            continue
        try:
            close_resource()
        except Exception as exc:  # pragma: no cover - resource-specific failure
            errors.append(exc)
    return errors


@dataclass(frozen=True, slots=True)
class _HandlerCapabilities:
    vision: bool
    audio: bool
    video: bool
    callable: bool


class _LlamaPublicAdapter:
    """Expose the small public Llama surface used by this backend."""

    def __init__(
        self,
        llama: Any,
        *,
        before_completion: Any | None = None,
        close_callback: Any | None = None,
    ):
        self._llama = llama
        self._before_completion = before_completion
        self._close_callback = close_callback

    @property
    def metadata(self) -> Any:
        return self._llama.metadata

    @property
    def chat_handler(self) -> Any:
        return self._llama.chat_handler

    @chat_handler.setter
    def chat_handler(self, value: Any) -> None:
        old_value = self._llama.chat_handler
        if old_value is not None and old_value is not value:
            _close_resources((old_value,))
        self._llama.chat_handler = value

    @property
    def last_speculative_stats(self) -> dict[str, Any]:
        return dict(self._llama.last_speculative_stats or {})

    def n_layer_nextn(self) -> int:
        return int(self._llama.n_layer_nextn())

    def token_ids(self) -> dict[str, int]:
        def token_id(method: Any) -> int:
            try:
                return int(method())
            except Exception:
                return -1

        return {
            "eos_token": token_id(self._llama.token_eos),
            "bos_token": token_id(self._llama.token_bos),
            "eot_token": token_id(self._llama.token_eot),
            "sep_token": token_id(self._llama.token_sep),
            "nl_token": token_id(self._llama.token_nl),
            "pad_token": token_id(self._llama.token_pad),
            "mask_token": token_id(self._llama.token_mask),
        }

    def token_text(self, token_id: int) -> str:
        if token_id == -1:
            return ""
        try:
            value = self._llama.detokenize([token_id], special=True)
        except Exception:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def handler_capabilities(handler: Any) -> _HandlerCapabilities:
        """Read documented MTMD capability flags from a handler instance."""
        if handler is None:
            return _HandlerCapabilities(False, False, False, False)
        return _HandlerCapabilities(
            vision=bool(handler.is_support_vision),
            audio=bool(handler.is_support_audio),
            video=bool(handler.is_support_video),
            callable=callable(handler),
        )

    def create_chat_completion(
        self, *, reuse_kv_cache: bool = False, **kwargs: Any
    ) -> Any:
        if self._before_completion is not None and not reuse_kv_cache:
            self._before_completion()
        return self._llama.create_chat_completion(**kwargs)

    def close(self) -> None:
        if self._close_callback is not None:
            self._close_callback()
            return
        self._llama.close()


class _SequentialLlamaSession:
    def __init__(self, llama_class: type):
        self._llama_class = llama_class
        self.llm: Any | None = None
        self.reset_count = 0

    def create(self, **kwargs: Any) -> _LlamaPublicAdapter:
        if self.llm is None:
            self.llm = self._llama_class(**kwargs)
            transient_resources: tuple[Any, ...] = ()
        else:
            if "chat_handler" in kwargs:
                new_handler = kwargs["chat_handler"]
                old_handler = self.llm.chat_handler
                if new_handler is not old_handler:
                    _close_resources((old_handler,))
                    self.llm.chat_handler = new_handler
            transient_resources = ()

        def reset() -> None:
            assert self.llm is not None
            self.llm.reset()
            self.reset_count += 1

        def close_transient_resources() -> None:
            errors = _close_resources(transient_resources)
            if errors:
                raise errors[0]

        return _LlamaPublicAdapter(
            self.llm,
            before_completion=reset,
            close_callback=close_transient_resources,
        )

    def close(self) -> None:
        if self.llm is None:
            return
        try:
            self.llm.close()
        finally:
            self.llm = None
            gc.collect()


def _model_profile_overrides(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    overrides = dict(value)
    reasoning_mode = overrides.pop("recommended_reasoning_mode")
    overrides["thinking"] = {
        "auto": None,
        "off": False,
        "on": True,
    }[reasoning_mode]
    return overrides


class LlamaCppSession:
    """Retain one llama.cpp model across independent ComfyUI executions."""

    def __init__(
        self,
        *,
        bindings: LlamaCppBindings | None = None,
        **configuration: Any,
    ) -> None:
        native = bindings or _import_bindings()
        self._configuration = dict(configuration)
        self._native_session = _SequentialLlamaSession(native.llama_class)
        self._bindings = LlamaCppBindings(
            llama_class=self._native_session.create,
            handlers=native.handlers,
            jinja_formatter_class=native.jinja_formatter_class,
            chat_formatter_to_handler=native.chat_formatter_to_handler,
        )
        self._closed = False
        self._execution_count = 0
        track_session(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def generate(self, **request: Any) -> LlamaCppResult:
        if self._closed:
            raise BackendError("The Llama.cpp session has already been unloaded.")
        reuse_kv_cache = request.pop("reuse_kv_cache", False)
        media_before_prompt = request.pop("media_before_prompt", False)
        if not isinstance(reuse_kv_cache, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        if not isinstance(media_before_prompt, bool):
            raise InputNormalizationError("media_before_prompt must be a boolean.")
        model_profile = request.pop("model_profile", None)
        configuration = self._configuration
        if model_profile is not None:
            overrides = _model_profile_overrides(model_profile)
            overrides.pop("handler", None)
            configuration = {
                **self._configuration,
                **overrides,
            }
        max_tokens = int(request.get("max_tokens", 0))
        reasoning_budget = int(configuration.get("reasoning_budget", 0))
        if reasoning_budget > max_tokens:
            raise InputNormalizationError(
                "Session reasoning_budget cannot exceed Generate max_tokens."
            )
        try:
            result = run_chat(
                bindings=self._bindings,
                reuse_kv_cache=reuse_kv_cache,
                media_before_prompt=media_before_prompt,
                **configuration,
                **request,
            )
        except Exception:
            self.close()
            raise
        self._execution_count += 1
        result.metrics["model_unloaded"] = False
        result.metrics["session"] = {
            "execution_index": self._execution_count - 1,
            "model_reused": self._execution_count > 1,
            "unload_required": True,
        }
        result.media_diagnostics["model_unloaded_after_response"] = False
        return result

    def decide(
        self,
        question: str,
        context: str,
        answers: list[str],
        model_profile: dict[str, Any] | None = None,
        media: MediaBundle | None = None,
        *,
        reuse_kv_cache: bool = False,
        media_before_prompt: bool = False,
    ) -> LlamaCppDecisionResult:
        if self._closed:
            raise BackendError("The Llama.cpp session has already been unloaded.")
        if not isinstance(reuse_kv_cache, bool):
            raise InputNormalizationError("reuse_kv_cache must be a boolean.")
        if not isinstance(media_before_prompt, bool):
            raise InputNormalizationError("media_before_prompt must be a boolean.")
        media = media or MediaBundle()
        try:
            from makoto_decision import Choices, Decision
            from makoto_decision.evaluators import (
                LlamaCppEvaluator,
                MultiTokenChoiceError,
            )
        except ImportError as exc:
            self.close()
            raise BackendError(
                "makoto-decision is required for Llama.cpp decision sessions. "
                "Install the optional llama dependencies and restart ComfyUI."
            ) from exc

        try:
            if not isinstance(question, str) or not question.strip():
                raise InputNormalizationError("question must be a non-empty string.")
            if not isinstance(context, str):
                raise InputNormalizationError("context must be a string.")
            if not isinstance(answers, list) or not 2 <= len(answers) <= 26:
                raise InputNormalizationError(
                    "answers must contain between 2 and 26 items."
                )
            if any(
                not isinstance(answer, str) or not answer.strip() for answer in answers
            ):
                raise InputNormalizationError("answers must be non-empty strings.")
            if len(set(answers)) != len(answers):
                raise InputNormalizationError("answers must be unique.")

            configuration = {
                **self._configuration,
                **_model_profile_overrides(model_profile),
            }
            configuration["handler"] = self._configuration.get("handler", "auto")
            choices = Choices.letters(*answers)
            decision = Decision(
                choices=choices,
                context=context,
                question=question,
            )
            with _NATIVE_EXECUTION_LOCK:
                fallback_handler, model_path, mmproj_path = (
                    self._initialize_for_decision(configuration, media)
                )
                llama = self._native_session.llm
                if llama is None:
                    raise BackendError("The Llama.cpp decision session has no model.")
                if not callable(getattr(llama, "tokenize", None)):
                    raise BackendError(
                        "The installed llama-cpp-python fork must expose "
                        "Llama.tokenize() for decision sessions."
                    )
                if not callable(getattr(llama, "create_chat_prefill", None)):
                    raise BackendError(
                        "The installed llama-cpp-python fork must expose "
                        "Llama.create_chat_prefill() for decision sessions."
                    )

                token_ids: dict[int, str] = {}
                for choice in choices:
                    tokens = llama.tokenize(
                        choice.target.encode("utf-8"),
                        add_bos=False,
                        special=False,
                    )
                    if len(tokens) != 1:
                        raise BackendError(
                            f"Choice target {choice.target!r} must tokenize to one "
                            f"token; got {len(tokens)}."
                        )
                    token_id = int(tokens[0])
                    if token_id < 0:
                        raise BackendError(
                            f"Choice target {choice.target!r} produced an invalid "
                            "token ID."
                        )
                    previous_target = token_ids.get(token_id)
                    if previous_target is not None:
                        raise BackendError(
                            f"Choice targets {previous_target!r} and {choice.target!r} "
                            "map to the same token ID."
                        )
                    token_ids[token_id] = choice.target

                if not reuse_kv_cache:
                    llama.reset()
                started = time.perf_counter()
                evaluator_llama = _DecisionMediaPrefill(
                    llama,
                    context,
                    media,
                    media_before_prompt=media_before_prompt,
                )
                result = LlamaCppEvaluator(evaluator_llama).evaluate(decision)
                if result.selected is None:
                    raise BackendError("Llama.cpp decision did not select an answer.")
                probabilities = result.probabilities
                ordered_probabilities = {
                    answer: float(probabilities[answer]) for answer in answers
                }
                elapsed = time.perf_counter() - started
                execution_index = self._execution_count
                self._execution_count += 1
                return LlamaCppDecisionResult(
                    selected=result.selected,
                    probabilities=ordered_probabilities,
                    metrics={
                        "decision_seconds": elapsed,
                        "model_unloaded": False,
                        "session": {
                            "execution_index": execution_index,
                            "model_reused": execution_index > 0,
                            "unload_required": True,
                        },
                    },
                    media_diagnostics=_capture_media_diagnostics(
                        llm=llama,
                        fallback_handler=fallback_handler,
                        media=media,
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                    ),
                )
        except Exception as exc:
            self.close()
            if isinstance(exc, (BackendError, InputNormalizationError)):
                raise
            if isinstance(exc, MultiTokenChoiceError):
                raise BackendError(str(exc)) from exc
            if isinstance(exc, NotImplementedError):
                raise BackendError(
                    "The selected llama.cpp chat handler does not support prefill."
                ) from exc
            raise BackendError(f"llama.cpp decision failed: {exc}") from exc

    def _initialize_for_decision(
        self,
        configuration: dict[str, Any] | None = None,
        media: MediaBundle | None = None,
    ) -> tuple[Any | None, str, str | None]:
        configuration = self._configuration if configuration is None else configuration
        media = media or MediaBundle()
        has_media = bool(media.items)
        model_path = _resolve_file(
            str(configuration.get("model_path", "")),
            label="model_path",
            required=True,
        )
        n_ctx = int(configuration.get("n_ctx", 8192))
        n_batch = int(configuration.get("n_batch", 512))
        mmproj_path = _resolve_file(
            str(configuration.get("mmproj_path", "")),
            label="mmproj_path",
            required=has_media,
        )
        gpu_layers = str(configuration.get("gpu_layers", "all"))
        if gpu_layers not in {"auto", "all", "cpu"}:
            raise InputNormalizationError("gpu_layers must be auto, all, or cpu.")
        flash_attention = str(configuration.get("flash_attention", "auto"))
        if flash_attention not in _FLASH_ATTN_TYPES:
            raise InputNormalizationError(
                "flash_attention must be auto, enabled, or disabled."
            )
        verbose = bool(configuration.get("verbose", False))
        thinking = configuration.get("thinking", False)
        reasoning_strength = str(configuration.get("reasoning_strength", "auto"))
        effective_reasoning_strength = _effective_reasoning_strength(
            bool(thinking),
            reasoning_strength,
        )
        handler = str(configuration.get("handler", "auto"))
        custom_chat_template = str(configuration.get("custom_chat_template", ""))
        resolved_draft = _resolve_file(
            str(configuration.get("draft_model_path", "")),
            label="draft_model_path",
            required=False,
        )
        spec_type = configuration.get("spec_type")
        if spec_type is None and resolved_draft is not None:
            raise InputNormalizationError(
                "draft_model_path requires an explicit spec_type; use the official "
                "JamePeng SpecConfig path instead of the removed implicit "
                "Experimental API."
            )
        native_configuration = _normalize_native_speculative(
            resolved_draft=resolved_draft,
            spec_type="none" if spec_type is None else str(spec_type),
            spec_n_max=int(configuration.get("spec_n_max", 2)),
            spec_n_min=int(configuration.get("spec_n_min", 0)),
            spec_p_min=float(configuration.get("spec_p_min", 0.0)),
            mtp_provider=str(configuration.get("mtp_provider", "off")),
            draft_n_gpu_layers=configuration.get("draft_n_gpu_layers", "all"),
            draft_backend_sampling=configuration.get("draft_backend_sampling", True),
            verbose=verbose,
            has_media=has_media,
            gpu_layers=gpu_layers,
            n_ctx=n_ctx,
        )
        ngram_configuration = normalize_ngram_speculative(
            configuration.get("ngram_speculative")
        )
        if (
            native_configuration is not None
            and ngram_configuration["speculative_mode"] == "ngram"
        ):
            raise InputNormalizationError(
                "Native draft GGUF and N-gram speculative decoding cannot be enabled together."
            )

        native_speculative_api = configuration.get("speculative_api")
        if (
            native_configuration is not None
            or ngram_configuration["speculative_mode"] == "ngram"
        ):
            native_speculative_api = (
                native_speculative_api or _import_native_speculative_bindings()
            )

        n_ubatch_override = _optional_positive_override(
            "n_ubatch",
            bool(configuration.get("override_n_ubatch", False)),
            int(configuration.get("n_ubatch", _DEFAULT_N_UBATCH)),
        )
        image_min_tokens_override = _optional_positive_override(
            "image_min_tokens",
            bool(configuration.get("override_image_min_tokens", False)),
            int(configuration.get("image_min_tokens", 1024)),
        )
        image_max_tokens_override = _optional_positive_override(
            "image_max_tokens",
            bool(configuration.get("override_image_max_tokens", False)),
            int(configuration.get("image_max_tokens", 1120)),
        )
        model_kwargs, chat_handler = _native_model_kwargs(
            self._bindings,
            model_path=model_path,
            mmproj_path=mmproj_path,
            has_media=has_media,
            handler=handler,
            verbose=verbose,
            thinking=thinking,
            effective_reasoning_strength=effective_reasoning_strength,
            preserve_thinking=bool(configuration.get("preserve_thinking", False)),
            custom_chat_template=custom_chat_template,
            n_ctx=n_ctx,
            n_batch=n_batch,
            gpu_layers=gpu_layers,
            main_gpu=int(configuration.get("main_gpu", 0)),
            n_threads=int(configuration.get("n_threads", 0)),
            flash_attention=flash_attention,
            use_mmap=bool(configuration.get("use_mmap", True)),
            type_k=str(configuration.get("type_k", "FP16")),
            type_v=str(configuration.get("type_v", "FP16")),
            n_ubatch_override=n_ubatch_override,
            image_min_tokens_override=image_min_tokens_override,
            image_max_tokens_override=image_max_tokens_override,
            native_configuration=native_configuration,
            ngram_configuration=ngram_configuration,
            native_speculative_api=native_speculative_api,
        )
        if has_media:
            _validate_multimodal_batch_settings(
                handler=handler,
                media=media,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch_override,
                image_min_tokens=image_min_tokens_override,
                image_max_tokens=image_max_tokens_override,
            )
        adapter = self._native_session.create(**model_kwargs)
        if not has_media:
            _install_text_template_handler(
                self._bindings,
                adapter,
                thinking=thinking,
                reasoning_strength=effective_reasoning_strength,
                custom_chat_template=custom_chat_template,
            )
        if (
            native_configuration is not None
            and native_configuration["mtp_provider"] == "internal"
        ):
            try:
                mtp_n_layer_nextn = adapter.n_layer_nextn()
            except (AttributeError, TypeError) as exc:
                raise BackendError(
                    "The installed llama-cpp-python fork does not expose "
                    "Llama.n_layer_nextn(); reinstall the matching Native MTP wheel."
                ) from exc
            if mtp_n_layer_nextn <= 0:
                raise BackendError(
                    "Selected Qwen 3.5+ target GGUF has no usable embedded NextN/MTP "
                    "layers."
                )
        return chat_handler, model_path, mmproj_path

    def close(self) -> None:
        if self._closed:
            return
        try:
            with _NATIVE_EXECUTION_LOCK:
                self._native_session.close()
                self._closed = True
        finally:
            untrack_session(self)


def _import_bindings() -> LlamaCppBindings:
    try:
        import_module("llama_cpp")
    except (ImportError, OSError) as exc:
        raise BackendError(
            "llama-cpp-python could not be imported. Install a wheel compatible with "
            "ComfyUI's Python, platform, and native backend, then restart ComfyUI. "
            + _fork_install_hint()
        ) from exc

    try:
        from llama_cpp import llama_multimodal as handler_module
    except (ImportError, OSError):
        try:
            from llama_cpp import llama_chat_format as handler_module
        except (ImportError, OSError):
            handler_module = None

    handlers: dict[str, type] = {}
    if handler_module is not None:
        for name, class_name in _HANDLER_CLASSES.items():
            handler_class = getattr(handler_module, class_name, None)
            if handler_class is not None:
                handlers[name] = handler_class

    try:
        from llama_cpp import llama_chat_format as chat_format_module
    except (ImportError, OSError):
        chat_format_module = None
    jinja_formatter_class = (
        getattr(chat_format_module, "Jinja2ChatFormatter", None)
        if chat_format_module is not None
        else None
    )
    chat_formatter_to_handler = (
        getattr(chat_format_module, "chat_formatter_to_chat_completion_handler", None)
        if chat_format_module is not None
        else None
    )

    try:
        from llama_cpp import Llama as llama_class
    except (ImportError, OSError) as exc:
        raise BackendError(
            "The installed llama-cpp-python package does not expose Llama. "
            + _fork_install_hint()
        ) from exc
    return LlamaCppBindings(
        llama_class=llama_class,
        handlers=handlers,
        jinja_formatter_class=jinja_formatter_class,
        chat_formatter_to_handler=chat_formatter_to_handler,
    )


def _import_native_speculative_bindings() -> NativeSpeculativeBindings:
    try:
        from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
    except (ImportError, OSError) as exc:
        raise BackendError(
            "Native speculative decoding is not installed in the Python environment "
            "that runs ComfyUI. The JamePeng backend requires the official "
            "llama_cpp.llama_speculative.SpecConfig/SpeculativeType API. No model was "
            "loaded.\n"
            f"Release and installation notes: {_NATIVE_SPECULATIVE_RELEASE_URL}\n"
            "Install a release wheel compatible with ComfyUI's exact Python, platform, "
            "CUDA runtime, and native DLLs, then restart ComfyUI."
        ) from exc

    return NativeSpeculativeBindings(
        spec_config=SpecConfig,
        speculative_type=SpeculativeType,
    )


def require_native_speculative() -> NativeSpeculativeBindings:
    """Require the JamePeng SpecConfig API before native model loading."""
    return _import_native_speculative_bindings()


def _native_speculative_stats(llm: _LlamaPublicAdapter) -> dict[str, Any]:
    """Expose the official Llama stats with the node's stable metric aliases."""
    stats = llm.last_speculative_stats
    try:
        drafted = int(stats.get("drafted", stats.get("drafted_tokens", 0)) or 0)
        accepted = int(
            stats.get(
                "accepted_draft_tokens",
                stats.get("accepted_tokens", stats.get("accepted", 0)),
            )
            or 0
        )
        draft_calls = int(stats.get("draft_calls", 0) or 0)
    except (TypeError, ValueError):
        drafted = 0
        accepted = 0
        draft_calls = 0

    acceptance_rate = stats.get("draft_token_acceptance_rate")
    try:
        acceptance_rate = float(acceptance_rate)
    except (TypeError, ValueError):
        acceptance_rate = accepted / drafted if drafted else 0.0
    mean_accepted = accepted / draft_calls if draft_calls else 0.0
    return {
        **stats,
        "draft_calls": draft_calls,
        "accept_calls": int(stats.get("accept_calls", 0) or 0),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": acceptance_rate,
        "mean_accepted_tokens": mean_accepted,
        "mean_accepted_per_call": mean_accepted,
    }


def _normalize_native_speculative(
    *,
    resolved_draft: str | None,
    spec_type: str,
    spec_n_max: int,
    spec_n_min: int,
    spec_p_min: float,
    mtp_provider: str,
    draft_n_gpu_layers: str | int,
    draft_backend_sampling: bool,
    verbose: bool,
    has_media: bool,
    gpu_layers: str,
    n_ctx: int,
) -> dict[str, Any] | None:
    provider = str(mtp_provider)
    if provider not in _MTP_PROVIDERS:
        raise InputNormalizationError(
            "mtp_provider must be off, external, or internal."
        )
    if (
        isinstance(draft_n_gpu_layers, bool)
        or not (
            (
                isinstance(draft_n_gpu_layers, str)
                and draft_n_gpu_layers in {"all", "auto"}
            )
            or isinstance(draft_n_gpu_layers, int)
        )
        or (isinstance(draft_n_gpu_layers, int) and draft_n_gpu_layers < -2)
    ):
        raise InputNormalizationError(
            "draft_n_gpu_layers must be all, auto, or an integer >= -2."
        )
    if not isinstance(draft_backend_sampling, bool):
        raise InputNormalizationError("draft_backend_sampling must be a boolean.")

    if spec_type == "none":
        if provider != "off":
            raise InputNormalizationError(
                "mtp_provider must be off when spec_type is none."
            )
        return None

    if has_media:
        raise InputNormalizationError(
            "Native speculative decoding currently supports text-only generation; "
            "disconnect IMAGE, AUDIO, and VIDEO inputs."
        )

    if provider != "off":
        if spec_type != "draft-mtp":
            raise InputNormalizationError(
                "spec_type must be draft-mtp when an MTP provider is selected."
            )
        n_max = int(spec_n_max)
        n_min = int(spec_n_min)
        p_min = float(spec_p_min)
        if n_max < 1:
            raise InputNormalizationError("spec_n_max must be at least 1.")
        if n_min < 0 or n_min > n_max:
            raise InputNormalizationError(
                "spec_n_min must be between 0 and spec_n_max."
            )
        if not 0.0 <= p_min <= 1.0:
            raise InputNormalizationError("spec_p_min must be between 0.0 and 1.0.")
        if gpu_layers != "all":
            raise InputNormalizationError(
                "Native MTP requires gpu_layers=all for CUDA all-layer offload."
            )
        if int(n_ctx) < n_max + 1:
            raise InputNormalizationError(
                "n_ctx must be at least spec_n_max + 1 for Native MTP."
            )
        if provider == "external" and resolved_draft is None:
            raise InputNormalizationError(
                "External MTP requires a draft GGUF in draft_model."
            )
        if provider == "internal" and resolved_draft is not None:
            raise InputNormalizationError(
                "Qwen 3.5+ internal MTP uses embedded NextN layers; leave draft_model "
                "unselected."
            )
        return {
            "spec_type": "draft-mtp",
            "mtp_provider": provider,
            "draft_model_path": resolved_draft if provider == "external" else None,
            "draft_n_max": n_max,
            "draft_n_min": n_min,
            "draft_p_min": p_min,
            "draft_n_gpu_layers": draft_n_gpu_layers,
            "draft_backend_sampling": draft_backend_sampling,
            "verbose": bool(verbose),
        }

    if spec_type == "draft-mtp":
        raise InputNormalizationError(
            "draft-mtp requires mtp_provider external or internal."
        )
    if spec_type not in _SPECULATIVE_TYPES:
        raise InputNormalizationError("spec_type must be draft-dflash or draft-dspark.")
    if resolved_draft is None:
        raise InputNormalizationError(
            f"{spec_type} requires a compatible draft GGUF in draft_model."
        )
    if int(spec_n_max) < 1:
        raise InputNormalizationError("spec_n_max must be at least 1.")
    if int(spec_n_min) < 0 or int(spec_n_min) > int(spec_n_max):
        raise InputNormalizationError("spec_n_min must be between 0 and spec_n_max.")
    if not 0.0 <= float(spec_p_min) <= 1.0:
        raise InputNormalizationError("spec_p_min must be between 0.0 and 1.0.")
    return {
        "spec_type": spec_type,
        "mtp_provider": "off",
        "draft_model_path": resolved_draft,
        "draft_n_max": int(spec_n_max),
        "draft_n_min": int(spec_n_min),
        "draft_p_min": float(spec_p_min),
        "draft_n_gpu_layers": draft_n_gpu_layers,
        "draft_backend_sampling": draft_backend_sampling,
        "verbose": False,
    }


def _create_native_speculative_config(
    bindings: NativeSpeculativeBindings,
    configuration: dict[str, Any],
) -> Any:
    try:
        spec_type = bindings.speculative_type.from_str(configuration["spec_type"])
        return bindings.spec_config(
            spec_type=spec_type,
            draft_model_path=configuration["draft_model_path"],
            draft_n_max=configuration["draft_n_max"],
            draft_n_min=configuration["draft_n_min"],
            draft_p_min=configuration["draft_p_min"],
            draft_n_gpu_layers=configuration["draft_n_gpu_layers"],
            draft_backend_sampling=configuration["draft_backend_sampling"],
        )
    except Exception as exc:
        if configuration["mtp_provider"] != "off":
            raise BackendError(
                "Native MTP configuration failed. Install the llama-cpp-python fork "
                "with the official SpecConfig API and draft-mtp support. "
                f"Original error: {exc}"
            ) from exc
        raise


def _create_ngram_speculative_config(
    bindings: NativeSpeculativeBindings,
    configuration: dict[str, Any],
) -> Any:
    spec_type_name = f"ngram-map-{configuration['ngram_mode']}"
    try:
        spec_type = bindings.speculative_type.from_str(spec_type_name)
        return bindings.spec_config(
            spec_type=spec_type,
            ngram_size_n=configuration["ngram_size"],
            ngram_size_m=configuration["num_pred_tokens"],
            ngram_min_hits=configuration["ngram_min_hits"],
            ngram_max_entries_per_key=configuration["ngram_max_entries_per_key"],
        )
    except Exception as exc:
        raise BackendError(
            "N-gram speculative decoding requires the official JamePeng "
            "SpecConfig NGRAM_MAP_K/NGRAM_MAP_K4V API."
        ) from exc


def normalize_ngram_speculative(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {"speculative_mode": "off"}
    if not isinstance(value, dict):
        raise InputNormalizationError(
            "ngram_speculative must be a Llama.cpp N-gram Speculative Preset object."
        )

    speculative_mode = value.get("speculative_mode")
    if speculative_mode not in {"off", "ngram"}:
        raise InputNormalizationError(
            "ngram_speculative.speculative_mode must be off or ngram."
        )
    if speculative_mode == "off":
        return {"speculative_mode": "off"}

    integer_ranges = {
        "ngram_size": (1, 8),
        "num_pred_tokens": (1, 32),
        "ngram_min_hits": (1, 16),
        "ngram_max_entries_per_key": (0, 1024),
    }
    normalized: dict[str, Any] = {"speculative_mode": "ngram"}
    for name, (minimum, maximum) in integer_ranges.items():
        candidate = value.get(name)
        if name == "ngram_max_entries_per_key" and candidate is None:
            normalized[name] = None
            continue
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise InputNormalizationError(
                f"ngram_speculative.{name} must be an integer."
            )
        if not minimum <= candidate <= maximum:
            raise InputNormalizationError(
                f"ngram_speculative.{name} must be between {minimum} and {maximum}."
            )
        normalized[name] = candidate

    ngram_mode = value.get("ngram_mode")
    if ngram_mode not in {"k", "k4v"}:
        raise InputNormalizationError("ngram_speculative.ngram_mode must be k or k4v.")
    normalized["ngram_mode"] = ngram_mode
    if normalized["ngram_max_entries_per_key"] == 0:
        normalized["ngram_max_entries_per_key"] = None
    return normalized


def _resolve_file(value: str, *, label: str, required: bool) -> str | None:
    normalized = value.strip()
    if not normalized:
        if required:
            raise InputNormalizationError(f"{label} is required.")
        return None

    path = Path(normalized).expanduser()
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InputNormalizationError(f"{label} does not exist: {normalized}") from exc
    if not path.is_file():
        raise InputNormalizationError(f"{label} is not a file: {normalized}")
    if path.suffix.lower() != ".gguf":
        raise InputNormalizationError(f"{label} must be a GGUF file.")
    return str(path)


def _data_uri(mime_type: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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

    if media.items:
        prompt_part = {"type": "text", "text": prompt}
        content: list[dict[str, Any]] = [] if media_before_prompt else [prompt_part]
        for item in media.items:
            if item.kind == "image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _data_uri(item.mime_type, item.payload),
                        },
                    }
                )
            elif item.kind == "audio":
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(item.payload).decode("ascii"),
                            "format": "wav",
                        },
                    }
                )
            elif item.kind == "video":
                content.append(
                    {
                        "type": "video",
                        "video": {
                            "url": _data_uri(item.mime_type, item.payload),
                        },
                    }
                )
            else:  # pragma: no cover - MediaKind prevents this
                raise InputNormalizationError(
                    f"The llama.cpp multimodal node does not support {item.kind} media."
                )
        if media_before_prompt:
            content.append(prompt_part)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _append_media_to_decision_messages(
    messages: list[dict[str, Any]],
    context: str,
    media: MediaBundle,
    *,
    media_before_prompt: bool = False,
) -> list[dict[str, Any]]:
    media_parts = _build_messages("", "", media)[-1]["content"][1:]
    adapted = [dict(message) for message in messages]
    user_index = next(
        (
            index
            for index in range(len(adapted) - 1, -1, -1)
            if adapted[index].get("role") == "user"
        ),
        None,
    )
    if user_index is None:
        raise BackendError("Decision prefill did not produce a user message.")
    content = adapted[user_index].get("content")
    parts = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else list(content)
        if isinstance(content, list)
        else None
    )
    if parts is None:
        raise BackendError("Decision prefill returned unsupported message content.")

    if context and not media_before_prompt:
        for index, part in enumerate(parts):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str) and context in text:
                before, _, after = text.partition(context)
                replacement = [{"type": "text", "text": before + context}]
                replacement.extend(media_parts)
                if after:
                    replacement.append({"type": "text", "text": after})
                parts[index : index + 1] = replacement
                adapted[user_index]["content"] = parts
                return adapted

    adapted[user_index]["content"] = media_parts + parts
    return adapted


def _with_decision_output_rule(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adapted = [dict(message) for message in messages]
    for message in adapted:
        if message.get("role") == "system":
            content = message.get("content", "")
            if isinstance(content, list):
                message["content"] = content + [
                    {"type": "text", "text": _DECISION_OUTPUT_RULE}
                ]
            else:
                message["content"] = (
                    f"{content}\n\n{_DECISION_OUTPUT_RULE}"
                    if content
                    else _DECISION_OUTPUT_RULE
                )
            return adapted
    return [{"role": "system", "content": _DECISION_OUTPUT_RULE}, *adapted]


class _DecisionMediaPrefill:
    def __init__(
        self,
        llama: Any,
        context: str,
        media: MediaBundle,
        *,
        media_before_prompt: bool = False,
    ):
        self._llama = llama
        self._context = context
        self._media = media
        self._media_before_prompt = media_before_prompt

    def tokenize(
        self, text: bytes, *, add_bos: bool = True, special: bool = True
    ) -> Any:
        return self._llama.tokenize(text, add_bos=add_bos, special=special)

    def create_chat_prefill(self, *, messages: list[dict[str, Any]]) -> Any:
        if self._media.items:
            messages = _append_media_to_decision_messages(
                messages,
                self._context,
                self._media,
                media_before_prompt=self._media_before_prompt,
            )
        return self._llama.create_chat_prefill(
            messages=_with_decision_output_rule(messages)
        )


def _adapt_messages_for_model_template(
    messages: list[dict[str, Any]],
    *,
    handler: str,
    metadata: Any,
) -> list[dict[str, Any]]:
    """Adapt OpenAI media parts when an auto-selected model template requires it."""
    if handler != "auto" or not isinstance(metadata, dict):
        return messages

    architecture = str(metadata.get("general.architecture", "")).strip().lower()
    if architecture.replace("_", "-") != "muse-glimmer":
        return messages

    adapted_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            adapted_messages.append(message)
            continue

        adapted_content: list[Any] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                adapted_content.append(part)
                continue

            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if not isinstance(url, str) or not url:
                adapted_content.append(part)
                continue

            # Muse-Glimmer's embedded template emits <|patch|> only for
            # template-native image parts. GenericMTMDChatHandler accepts this
            # representation and still extracts the same data URI as media.
            adapted_content.append({"type": "image", "image": url})

        adapted_messages.append({**message, "content": adapted_content})

    return adapted_messages


def _create_handler(
    bindings: LlamaCppBindings,
    *,
    handler: str,
    mmproj_path: str | None,
    verbose: bool,
    thinking: bool | None,
    reasoning_strength: str | None,
    image_min_tokens: int | None,
    image_max_tokens: int | None,
    custom_chat_template: str = "",
) -> Any | None:
    if handler == "auto":
        return None
    if handler not in HANDLER_NAMES:
        raise InputNormalizationError(
            f"handler must be one of {', '.join(HANDLER_NAMES)}."
        )
    if mmproj_path is None:
        raise InputNormalizationError(
            f"mmproj_path is required for the {handler} handler."
        )
    handler_class = bindings.handlers.get(handler)
    if handler_class is None:
        class_name = _HANDLER_CLASSES[handler]
        raise BackendError(
            f"The installed llama-cpp-python build does not provide {class_name}. "
            + _fork_install_hint()
        )
    handler_kwargs: dict[str, Any] = {
        "mmproj_path": mmproj_path,
        "verbose": verbose,
    }
    if image_min_tokens is not None:
        handler_kwargs["image_min_tokens"] = image_min_tokens
    if image_max_tokens is not None:
        handler_kwargs["image_max_tokens"] = image_max_tokens
    if handler == "gemma4" and thinking is not None:
        handler_kwargs["enable_thinking"] = thinking
    elif handler == "qwen3_vl" and thinking is not None:
        handler_kwargs["force_reasoning"] = thinking
    elif thinking is not None or reasoning_strength is not None:
        handler_kwargs["extra_template_arguments"] = {
            "enable_thinking": bool(thinking),
            "force_reasoning": bool(thinking),
        }
        if reasoning_strength is not None:
            handler_kwargs["extra_template_arguments"]["reasoning_strength"] = (
                reasoning_strength
            )
    if handler == "generic":
        handler_kwargs["chat_format"] = custom_chat_template or None
    return handler_class(**handler_kwargs)


def _install_text_template_handler(
    bindings: LlamaCppBindings,
    llm: _LlamaPublicAdapter,
    *,
    thinking: bool | None,
    reasoning_strength: str | None,
    custom_chat_template: str = "",
) -> bool:
    if thinking is None and reasoning_strength is None and not custom_chat_template:
        if llm.chat_handler is not None:
            llm.chat_handler = None
        return False
    formatter_class = bindings.jinja_formatter_class
    to_handler = bindings.chat_formatter_to_handler
    metadata = llm.metadata
    template = (
        custom_chat_template
        if custom_chat_template
        else (
            metadata.get("tokenizer.chat_template")
            if isinstance(metadata, dict)
            else None
        )
    )
    if not isinstance(template, str) or not template:
        return False
    if formatter_class is None or not callable(to_handler):
        if (
            "enable_thinking" in template
            or "force_reasoning" in template
            or custom_chat_template
        ):
            raise BackendError(
                "The installed llama-cpp-python fork cannot pass thinking controls or "
                "custom Jinja chat templates to a text-only GGUF chat template. Upgrade the "
                "JamePeng fork to a build that exposes Jinja2ChatFormatter and "
                "chat_formatter_to_chat_completion_handler."
            )
        return False

    token_ids = llm.token_ids()
    special_tokens_map = {
        name: text
        for name, value in token_ids.items()
        if value != -1 and (text := llm.token_text(value))
    }
    stop_token_ids = [
        value
        for value in (token_ids["eos_token"], token_ids["eot_token"])
        if value != -1
    ]
    formatter = formatter_class(
        template=template,
        eos_token=special_tokens_map.get("eos_token", ""),
        bos_token=special_tokens_map.get("bos_token", ""),
        stop_token_ids=stop_token_ids or None,
        special_tokens_map=special_tokens_map,
    )
    template_arguments: dict[str, Any] = {
        "enable_thinking": bool(thinking),
        "force_reasoning": bool(thinking),
    }
    if reasoning_strength is not None:
        template_arguments["reasoning_strength"] = reasoning_strength
    configured_formatter = partial(formatter, **template_arguments)
    llm.chat_handler = to_handler(configured_formatter)
    return True


def _native_model_kwargs(
    native: LlamaCppBindings,
    *,
    model_path: str,
    mmproj_path: str | None,
    has_media: bool,
    handler: str,
    verbose: bool,
    thinking: bool | None,
    effective_reasoning_strength: str | None,
    preserve_thinking: bool,
    custom_chat_template: str,
    n_ctx: int,
    n_batch: int,
    gpu_layers: str,
    main_gpu: int,
    n_threads: int,
    flash_attention: str,
    use_mmap: bool,
    type_k: str,
    type_v: str,
    n_ubatch_override: int | None,
    image_min_tokens_override: int | None,
    image_max_tokens_override: int | None,
    native_configuration: dict[str, Any] | None,
    ngram_configuration: dict[str, Any],
    native_speculative_api: NativeSpeculativeBindings | None,
) -> tuple[dict[str, Any], Any | None]:
    chat_handler = None
    model_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": int(n_ctx),
        "n_batch": int(n_batch),
        "n_gpu_layers": 0 if gpu_layers == "cpu" else gpu_layers,
        "main_gpu": int(main_gpu),
        "n_threads": None if int(n_threads) <= 0 else int(n_threads),
        "flash_attn_type": _FLASH_ATTN_TYPES[flash_attention],
        "use_mmap": bool(use_mmap),
        "type_k": _kv_cache_type_id("type_k", type_k),
        "type_v": _kv_cache_type_id("type_v", type_v),
        "verbose": bool(verbose),
    }
    if n_ubatch_override is not None:
        model_kwargs["n_ubatch"] = n_ubatch_override
    if not has_media:
        if mmproj_path is not None:
            model_kwargs["mmproj_path"] = mmproj_path
        handler_kwargs: dict[str, Any] = {"verbose": bool(verbose)}
        if handler == "qwen35":
            handler_kwargs["preserve_thinking"] = bool(preserve_thinking)
            if effective_reasoning_strength is not None:
                handler_kwargs["reasoning_effort"] = (
                    "xhigh"
                    if effective_reasoning_strength == "high"
                    else effective_reasoning_strength
                )
        if thinking is not None or effective_reasoning_strength is not None:
            handler_kwargs["extra_template_arguments"] = {
                "enable_thinking": bool(thinking),
                "force_reasoning": bool(thinking),
            }
            if effective_reasoning_strength is not None:
                handler_kwargs["extra_template_arguments"]["reasoning_strength"] = (
                    effective_reasoning_strength
                )
        if image_min_tokens_override is not None:
            handler_kwargs["image_min_tokens"] = image_min_tokens_override
        if image_max_tokens_override is not None:
            handler_kwargs["image_max_tokens"] = image_max_tokens_override
        model_kwargs["chat_handler_kwargs"] = handler_kwargs
        if custom_chat_template:
            model_kwargs["chat_format"] = custom_chat_template

    if native_configuration is not None:
        assert native_speculative_api is not None
        model_kwargs["speculative"] = _create_native_speculative_config(
            native_speculative_api,
            native_configuration,
        )
        if native_configuration["mtp_provider"] != "off":
            model_kwargs["n_seq_max"] = 1
    elif ngram_configuration["speculative_mode"] == "ngram":
        assert native_speculative_api is not None
        model_kwargs["speculative"] = _create_ngram_speculative_config(
            native_speculative_api,
            ngram_configuration,
        )
    if has_media:
        chat_handler = _create_handler(
            native,
            handler=handler,
            mmproj_path=mmproj_path,
            verbose=verbose,
            thinking=thinking,
            reasoning_strength=effective_reasoning_strength,
            image_min_tokens=image_min_tokens_override,
            image_max_tokens=image_max_tokens_override,
            custom_chat_template=custom_chat_template,
        )
        model_kwargs["chat_handler"] = chat_handler
    return model_kwargs, chat_handler


def _optional_positive_override(name: str, enabled: bool, value: int) -> int | None:
    if not enabled:
        return None
    normalized = int(value)
    if normalized < 1:
        raise InputNormalizationError(
            f"{name} must be at least 1 when its override is enabled."
        )
    return normalized


def _effective_reasoning_strength(thinking: bool | None, value: str) -> str | None:
    if not thinking:
        return None
    normalized = str(value).strip().lower()
    if normalized not in REASONING_STRENGTHS:
        raise InputNormalizationError(
            f"reasoning_strength must be one of {', '.join(REASONING_STRENGTHS)}."
        )
    return None if normalized == "auto" else normalized


def _effective_reasoning_budget(thinking: bool | None, value: int) -> int:
    if not thinking:
        return 0
    normalized = int(value)
    if normalized < 0 or normalized > _MAX_REASONING_BUDGET:
        raise InputNormalizationError(
            f"reasoning_budget must be between 0 and {_MAX_REASONING_BUDGET}."
        )
    return normalized


def _reasoning_budget_arguments(
    *,
    metadata: Any,
    handler: str,
    reasoning_budget: int,
    custom_chat_template: str = "",
) -> tuple[dict[str, Any], str | None]:
    if reasoning_budget == 0:
        return {}, None

    template = (
        custom_chat_template
        if custom_chat_template
        else (
            metadata.get("tokenizer.chat_template", "")
            if isinstance(metadata, dict)
            else ""
        )
    )
    if handler == "qwen3_vl" or (
        isinstance(template, str) and "<think>" in template and "</think>" in template
    ):
        return {
            "reasoning_budget": reasoning_budget,
            "reasoning_start": "<think>",
            "reasoning_end": "</think>",
            "reasoning_start_in_prompt": True,
        }, "think_tags"
    if handler == "gemma4" or (
        isinstance(template, str)
        and "<|channel>" in template
        and "<channel|>" in template
    ):
        return {
            "reasoning_budget": reasoning_budget,
            "reasoning_start": "<|channel>",
            "reasoning_end": "<channel|>",
            "reasoning_start_in_prompt": False,
        }, "channel_tags"
    raise InputNormalizationError(
        "reasoning_budget is positive, but this model's GGUF chat template does not "
        "expose a supported reasoning format (<think>...</think> or Gemma channel "
        "tags). Set reasoning_budget to 0 or select a compatible model/handler."
    )


def _validate_multimodal_batch_settings(
    *,
    handler: str,
    media: MediaBundle,
    n_ctx: int,
    n_batch: int,
    n_ubatch: int | None,
    image_min_tokens: int | None,
    image_max_tokens: int | None,
) -> None:
    if n_ubatch is not None and n_ubatch > min(n_ctx, n_batch):
        raise InputNormalizationError(
            "n_ubatch cannot exceed n_batch or n_ctx because llama-cpp-python clamps the "
            "physical batch to both values."
        )
    if (
        image_min_tokens is not None
        and image_max_tokens is not None
        and image_min_tokens > image_max_tokens
    ):
        raise InputNormalizationError(
            "image_min_tokens cannot exceed image_max_tokens when both are overridden."
        )
    image_token_limit = (
        image_max_tokens if image_max_tokens is not None else image_min_tokens
    )
    image_token_limit_name = (
        "image_max_tokens" if image_max_tokens is not None else "image_min_tokens"
    )
    if image_token_limit is None or not any(
        item.kind in {"image", "video"} for item in media.items
    ):
        return

    if image_token_limit > n_ctx:
        raise InputNormalizationError(
            f"{image_token_limit_name} cannot exceed n_ctx for an image or video request."
        )
    if image_token_limit > n_batch:
        raise InputNormalizationError(
            f"n_batch must be at least {image_token_limit_name} for an image or video request."
        )
    effective_n_ubatch = n_ubatch
    if effective_n_ubatch is None:
        effective_n_ubatch = min(n_ctx, n_batch, _DEFAULT_N_UBATCH)
    if image_token_limit > effective_n_ubatch and (
        image_token_limit_name != "image_max_tokens" or handler == "gemma4"
    ):
        raise InputNormalizationError(
            f"The effective n_ubatch must be at least {image_token_limit_name} for an image or video "
            "request. Enable the n_ubatch override and raise its value to avoid a native "
            "non-causal attention assertion."
        )


def _extract_response(raw: dict[str, Any]) -> tuple[str, str]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise BackendError(
            "llama-cpp-python returned a response without a choices array."
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise BackendError(
            "llama-cpp-python returned a response without an assistant message."
        )

    content = message.get("content")
    response = "" if content is None else str(content)
    thinking_value = message.get("reasoning_content", message.get("thinking", ""))
    thinking = "" if thinking_value is None else str(thinking_value)

    if not thinking:
        if "</think>" in response:
            reasoning, response = response.split("</think>", 1)
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            thinking = reasoning.strip()
            response = response.lstrip()
        elif response.startswith("<|channel>thought"):
            channel = response[len("<|channel>thought") :].lstrip("\r\n")
            if "<channel|>" in channel:
                reasoning, response = channel.split("<channel|>", 1)
                thinking = reasoning.strip()
                response = response.lstrip()
            else:
                thinking = channel.strip()
                response = ""
    return response, thinking


def _capture_media_diagnostics(
    *,
    llm: _LlamaPublicAdapter,
    fallback_handler: Any,
    media: MediaBundle,
    model_path: str,
    mmproj_path: str | None,
) -> dict[str, Any]:
    """Copy public MTMD capability state while the handler is still alive."""
    active_handler = llm.chat_handler or fallback_handler
    handler_name = (
        type(active_handler).__name__ if active_handler is not None else "none"
    )
    capabilities = (
        _LlamaPublicAdapter.handler_capabilities(active_handler)
        if media.items and active_handler is not None
        else _HandlerCapabilities(False, False, False, False)
    )
    vision_available = capabilities.vision
    audio_available = capabilities.audio
    video_available = capabilities.video
    strict_mtmd_pipeline = capabilities.callable

    manifest = media.manifest()
    requested_image_count = int(manifest["image_count"])
    requested_audio_count = int(manifest["audio_count"])
    requested_video_count = int(manifest["video_count"])
    modalities_available = (
        (requested_image_count == 0 or vision_available)
        and (requested_audio_count == 0 or audio_available)
        and (requested_video_count == 0 or video_available)
    )
    verified = strict_mtmd_pipeline and modalities_available
    evaluated_image_count = requested_image_count if verified else 0
    evaluated_audio_count = requested_audio_count if verified else 0
    evaluated_video_count = requested_video_count if verified else 0
    requested_media_count = (
        requested_image_count + requested_audio_count + requested_video_count
    )
    evaluated_media_count = (
        evaluated_image_count + evaluated_audio_count + evaluated_video_count
    )

    if requested_media_count == 0:
        verification = "no_media"
        all_media_evaluated = True
    elif verified and evaluated_media_count == requested_media_count:
        verification = "mtmd_evaluated"
        all_media_evaluated = True
    else:
        verification = "unavailable"
        all_media_evaluated = False

    return {
        "schema_version": 1,
        "backend": "llama-cpp-python",
        "model": Path(model_path).name,
        "mmproj": Path(mmproj_path).name if mmproj_path else None,
        "handler": handler_name,
        "capabilities": {
            "vision": vision_available,
            "audio": audio_available,
            "video": video_available,
        },
        "requested": manifest,
        "evaluated": {
            "media_count": evaluated_media_count,
            "image_count": evaluated_image_count,
            "audio_count": evaluated_audio_count,
            "video_count": evaluated_video_count,
        },
        "mtmd": {
            "strict_pipeline": strict_mtmd_pipeline,
            "completion_succeeded": True,
            "all_media_evaluated": all_media_evaluated,
            "verification": verification,
        },
    }


def run_chat(
    *,
    model_path: str,
    mmproj_path: str = "",
    handler: str = "auto",
    system: str,
    prompt: str,
    media: MediaBundle,
    n_ctx: int = 8192,
    n_batch: int = 512,
    override_n_ubatch: bool = False,
    n_ubatch: int = _DEFAULT_N_UBATCH,
    gpu_layers: str = "all",
    main_gpu: int = 0,
    n_threads: int = 0,
    flash_attention: str = "auto",
    use_mmap: bool = True,
    type_k: str = "FP16",
    type_v: str = "FP16",
    max_tokens: int = 512,
    thinking: bool | None = False,
    reasoning_strength: str = "auto",
    reasoning_budget: int = 0,
    preserve_thinking: bool = False,
    override_image_min_tokens: bool = False,
    image_min_tokens: int = 1024,
    override_image_max_tokens: bool = False,
    image_max_tokens: int = 1120,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    presence_penalty: float = 0.0,
    repeat_penalty: float = 1.0,
    seed: int = -1,
    stop: str = "",
    verbose: bool = False,
    draft_model_path: str = "",
    spec_type: str | None = None,
    spec_n_max: int = 2,
    spec_n_min: int = 0,
    spec_p_min: float = 0.0,
    mtp_provider: str = "off",
    draft_n_gpu_layers: str | int = "all",
    draft_backend_sampling: bool = True,
    ngram_speculative: dict[str, Any] | None = None,
    bindings: LlamaCppBindings | None = None,
    speculative_api: NativeSpeculativeBindings | None = None,
    custom_chat_template: str = "",
    reuse_kv_cache: bool = False,
    media_before_prompt: bool = False,
) -> LlamaCppResult:
    if not isinstance(reuse_kv_cache, bool):
        raise InputNormalizationError("reuse_kv_cache must be a boolean.")
    if not isinstance(media_before_prompt, bool):
        raise InputNormalizationError("media_before_prompt must be a boolean.")
    effective_reasoning_strength = _effective_reasoning_strength(
        bool(thinking),
        reasoning_strength,
    )
    effective_reasoning_budget = _effective_reasoning_budget(
        bool(thinking),
        reasoning_budget,
    )
    ngram_configuration = normalize_ngram_speculative(ngram_speculative)
    resolved_model = _resolve_file(model_path, label="model_path", required=True)
    has_media = bool(media.items)
    resolved_mmproj = (
        _resolve_file(mmproj_path, label="mmproj_path", required=False)
        if has_media
        else None
    )
    resolved_draft = _resolve_file(
        draft_model_path,
        label="draft_model_path",
        required=False,
    )
    if spec_type is None and resolved_draft is not None:
        raise InputNormalizationError(
            "draft_model_path requires an explicit spec_type; use the official "
            "JamePeng SpecConfig path instead of the removed implicit Experimental API."
        )
    effective_spec_type = "none" if spec_type is None else str(spec_type)
    if flash_attention not in _FLASH_ATTN_TYPES:
        raise InputNormalizationError(
            "flash_attention must be auto, enabled, or disabled."
        )
    if gpu_layers not in {"auto", "all", "cpu"}:
        raise InputNormalizationError("gpu_layers must be auto, all, or cpu.")
    native_configuration = _normalize_native_speculative(
        resolved_draft=resolved_draft,
        spec_type=effective_spec_type,
        spec_n_max=spec_n_max,
        spec_n_min=spec_n_min,
        spec_p_min=spec_p_min,
        mtp_provider=mtp_provider,
        draft_n_gpu_layers=draft_n_gpu_layers,
        draft_backend_sampling=draft_backend_sampling,
        verbose=verbose,
        has_media=has_media,
        gpu_layers=gpu_layers,
        n_ctx=n_ctx,
    )
    if has_media and resolved_mmproj is None:
        raise InputNormalizationError(
            "mmproj_path is required when image, audio, or video media are supplied."
        )
    if (
        native_configuration is not None
        and ngram_configuration["speculative_mode"] == "ngram"
    ):
        raise InputNormalizationError(
            "Native draft GGUF and N-gram speculative decoding cannot be enabled together."
        )
    if has_media and ngram_configuration["speculative_mode"] == "ngram":
        raise InputNormalizationError(
            "N-gram speculative decoding currently supports text-only generation; "
            "disconnect IMAGE, AUDIO, and VIDEO inputs."
        )

    n_ubatch_override = _optional_positive_override(
        "n_ubatch", override_n_ubatch, n_ubatch
    )
    image_min_tokens_override = _optional_positive_override(
        "image_min_tokens", override_image_min_tokens, image_min_tokens
    )
    image_max_tokens_override = _optional_positive_override(
        "image_max_tokens", override_image_max_tokens, image_max_tokens
    )
    _validate_multimodal_batch_settings(
        handler=handler,
        media=media,
        n_ctx=int(n_ctx),
        n_batch=int(n_batch),
        n_ubatch=n_ubatch_override,
        image_min_tokens=image_min_tokens_override,
        image_max_tokens=image_max_tokens_override,
    )

    messages = _build_messages(
        system, prompt, media, media_before_prompt=media_before_prompt
    )
    native = bindings or _import_bindings()
    native_speculative_api = None
    if (
        native_configuration is not None
        or ngram_configuration["speculative_mode"] == "ngram"
    ):
        native_speculative_api = (
            speculative_api or _import_native_speculative_bindings()
        )
    load_seconds = 0.0
    generation_seconds = 0.0
    cleanup_seconds = 0.0
    llm = None
    chat_handler = None
    raw: dict[str, Any] | None = None
    media_diagnostics: dict[str, Any] | None = None
    speculative_stats: dict[str, Any] | None = None
    mtp_n_layer_nextn: int | None = None
    reasoning_budget_format: str | None = None
    execution_error: Exception | None = None
    cleanup_error: Exception | None = None

    with _NATIVE_EXECUTION_LOCK:
        load_started = time.perf_counter()
        try:
            model_kwargs, chat_handler = _native_model_kwargs(
                native,
                model_path=resolved_model,
                mmproj_path=resolved_mmproj,
                has_media=has_media,
                handler=handler,
                verbose=verbose,
                thinking=thinking,
                effective_reasoning_strength=effective_reasoning_strength,
                preserve_thinking=preserve_thinking,
                custom_chat_template=custom_chat_template,
                n_ctx=n_ctx,
                n_batch=n_batch,
                gpu_layers=gpu_layers,
                main_gpu=main_gpu,
                n_threads=n_threads,
                flash_attention=flash_attention,
                use_mmap=use_mmap,
                type_k=type_k,
                type_v=type_v,
                n_ubatch_override=n_ubatch_override,
                image_min_tokens_override=image_min_tokens_override,
                image_max_tokens_override=image_max_tokens_override,
                native_configuration=native_configuration,
                ngram_configuration=ngram_configuration,
                native_speculative_api=native_speculative_api,
            )
            if ngram_configuration["speculative_mode"] == "ngram":
                _LOGGER.info(
                    "N-gram speculative decoding: ngram size=%s, max predicted tokens=%s, "
                    "mode=%s, minimum hits=%s.",
                    ngram_configuration["ngram_size"],
                    ngram_configuration["num_pred_tokens"],
                    ngram_configuration["ngram_mode"],
                    ngram_configuration["ngram_min_hits"],
                )

            try:
                raw_llm = native.llama_class(**model_kwargs)
                llm = (
                    raw_llm
                    if isinstance(raw_llm, _LlamaPublicAdapter)
                    else _LlamaPublicAdapter(raw_llm)
                )
            except Exception as exc:
                if (
                    native_configuration is not None
                    and native_configuration["mtp_provider"] != "off"
                ):
                    raise BackendError(
                        "Native MTP initialization failed. Install the llama-cpp-python "
                        "fork with the official SpecConfig API and draft-mtp support. "
                        f"Original error: {exc}"
                    ) from exc
                raise
            if resolved_mmproj is None:
                _install_text_template_handler(
                    native,
                    llm,
                    thinking=thinking,
                    reasoning_strength=effective_reasoning_strength,
                    custom_chat_template=custom_chat_template,
                )
            if (
                native_configuration is not None
                and native_configuration["mtp_provider"] == "internal"
            ):
                try:
                    mtp_n_layer_nextn = llm.n_layer_nextn()
                except (AttributeError, TypeError) as exc:
                    raise BackendError(
                        "The installed llama-cpp-python fork does not expose "
                        "Llama.n_layer_nextn(); reinstall the matching Native MTP wheel."
                    ) from exc
                if mtp_n_layer_nextn <= 0:
                    raise BackendError(
                        "Selected Qwen 3.5+ target GGUF has no usable embedded NextN/MTP "
                        "layers."
                    )
            load_seconds = time.perf_counter() - load_started
            generation_started = time.perf_counter()
            completion_messages = _adapt_messages_for_model_template(
                messages,
                handler=handler,
                metadata=llm.metadata,
            )
            completion_kwargs: dict[str, Any] = {
                "messages": completion_messages,
                "stream": False,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "min_p": float(min_p),
                # The targeted JamePeng fork follows llama.cpp's `present` spelling.
                # Keep the public/profile name aligned with model cards and translate
                # only at the binding boundary.
                "present_penalty": float(presence_penalty),
                "repeat_penalty": float(repeat_penalty),
                "seed": None if int(seed) < 0 else int(seed),
            }
            if stop:
                completion_kwargs["stop"] = [stop]
            budget_arguments, reasoning_budget_format = _reasoning_budget_arguments(
                metadata=llm.metadata,
                handler=handler,
                reasoning_budget=effective_reasoning_budget,
                custom_chat_template=custom_chat_template,
            )
            completion_kwargs.update(budget_arguments)
            completion = llm.create_chat_completion(
                reuse_kv_cache=reuse_kv_cache,
                **completion_kwargs,
            )
            generation_seconds = time.perf_counter() - generation_started
            if not isinstance(completion, dict):
                raise BackendError(
                    "llama-cpp-python returned a streaming or non-object response unexpectedly."
                )
            raw = completion
            if (
                native_configuration is not None
                or ngram_configuration["speculative_mode"] == "ngram"
            ):
                speculative_stats = _native_speculative_stats(llm)
                try:
                    drafted_tokens = int(
                        speculative_stats.get("drafted_tokens", 0) or 0
                    )
                    accepted_tokens = int(
                        speculative_stats.get("accepted_tokens", 0) or 0
                    )
                    draft_calls = int(speculative_stats.get("draft_calls", 0) or 0)
                except (TypeError, ValueError):
                    drafted_tokens = 0
                    accepted_tokens = 0
                    draft_calls = 0
                acceptance_rate = (
                    accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0
                )
                if draft_calls <= 0 or drafted_tokens <= 0:
                    _LOGGER.warning(
                        "Stateful speculative decoding completed without draft activity; "
                        "draft_calls=%s, drafted_tokens=%s.",
                        draft_calls,
                        drafted_tokens,
                    )
                else:
                    _LOGGER.info(
                        "Stateful speculative decoding (%s): drafted tokens=%s, accepted "
                        "tokens=%s, acceptance rate=%s, mean accepted/call=%s.",
                        (
                            native_configuration["spec_type"]
                            if native_configuration is not None
                            else f"ngram-map-{ngram_configuration['ngram_mode']}"
                        ),
                        drafted_tokens,
                        accepted_tokens,
                        acceptance_rate,
                        speculative_stats.get(
                            "mean_accepted_per_call",
                            speculative_stats.get("mean_accepted_tokens", 0.0),
                        ),
                    )
                    if drafted_tokens >= 100 and acceptance_rate < 0.05:
                        _LOGGER.warning(
                            "Speculative acceptance is below 5%%; the target "
                            "and draft/provider may be incompatible, and target-only "
                            "generation may be faster. drafted_tokens=%s, "
                            "acceptance_rate=%.2f%%.",
                            drafted_tokens,
                            acceptance_rate * 100.0,
                        )
            media_diagnostics = _capture_media_diagnostics(
                llm=llm,
                fallback_handler=chat_handler,
                media=media,
                model_path=resolved_model,
                mmproj_path=resolved_mmproj,
            )
        except (
            Exception
        ) as exc:  # preserve cleanup while presenting a stable node error
            execution_error = exc
        finally:
            cleanup_started = time.perf_counter()
            cleanup_errors: list[Exception] = []
            if llm is not None:
                try:
                    llm.close()
                except (
                    Exception
                ) as exc:  # pragma: no cover - platform-specific native failure
                    cleanup_errors.append(exc)
                    cleanup_errors.extend(_close_resources((chat_handler,)))
            else:
                cleanup_errors.extend(_close_resources((chat_handler,)))
            if cleanup_errors:
                cleanup_error = cleanup_errors[0]
            llm = None
            chat_handler = None
            gc.collect()
            cleanup_seconds = time.perf_counter() - cleanup_started

    if execution_error is not None:
        if isinstance(execution_error, (BackendError, InputNormalizationError)):
            raise execution_error
        raise BackendError(
            f"llama-cpp-python inference failed: {execution_error}"
        ) from execution_error
    if cleanup_error is not None:
        raise BackendError(
            f"llama-cpp-python completed, but the model could not be fully unloaded: {cleanup_error}"
        ) from cleanup_error
    if raw is None:  # pragma: no cover - defensive invariant
        raise BackendError("llama-cpp-python did not return a response.")
    if media_diagnostics is None:  # pragma: no cover - defensive invariant
        raise BackendError("llama-cpp-python did not produce media diagnostics.")

    response, thinking_output = _extract_response(raw)
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    metrics = {
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "cleanup_seconds": cleanup_seconds,
        "total_seconds": load_seconds + generation_seconds + cleanup_seconds,
        "usage": usage,
        "model_unloaded": True,
        "configuration": {
            "thinking": thinking if thinking is None else bool(thinking),
            "reasoning_strength": effective_reasoning_strength or "auto",
            "reasoning_budget": effective_reasoning_budget,
            "reasoning_budget_applied": reasoning_budget_format is not None,
            "reasoning_budget_format": reasoning_budget_format,
            "custom_chat_template": bool(custom_chat_template),
            "n_ctx": int(n_ctx),
            "n_batch": int(n_batch),
            "n_ubatch_override": n_ubatch_override,
            "image_min_tokens_override": image_min_tokens_override,
            "image_max_tokens_override": image_max_tokens_override,
            "presence_penalty": float(presence_penalty),
        },
    }
    if native_configuration is not None:
        metrics["speculative"] = {
            "enabled": True,
            "implementation": native_configuration["spec_type"],
            "draft_model": (
                Path(native_configuration["draft_model_path"]).name
                if native_configuration["draft_model_path"] is not None
                else None
            ),
            "n_max": native_configuration["draft_n_max"],
            "n_min": native_configuration["draft_n_min"],
            "p_min": native_configuration["draft_p_min"],
            "stats": speculative_stats or {},
        }
        if native_configuration["mtp_provider"] != "off":
            choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
            first_choice = (
                choices[0] if choices and isinstance(choices[0], dict) else {}
            )
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            metrics["speculative"].update(
                {
                    "mtp_provider": native_configuration["mtp_provider"],
                    "verbose": native_configuration["verbose"],
                    "n_layer_nextn": mtp_n_layer_nextn,
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": (
                        completion_tokens / generation_seconds
                        if generation_seconds > 0
                        else 0.0
                    ),
                    "finish_reason": first_choice.get("finish_reason"),
                }
            )
    if ngram_configuration["speculative_mode"] == "ngram":
        metrics["ngram_speculative"] = {
            **dict(ngram_configuration),
            "implementation": f"ngram-map-{ngram_configuration['ngram_mode']}",
            "stats": speculative_stats or {},
        }
    media_diagnostics["model_unloaded_after_response"] = True
    return LlamaCppResult(
        response=response,
        thinking=thinking_output,
        raw=raw,
        metrics=metrics,
        media_diagnostics=media_diagnostics,
    )


def run_chat_sequential(
    *,
    media_items: list[MediaBundle],
    prompt_items: list[str] | None = None,
    bindings: LlamaCppBindings | None = None,
    **kwargs: Any,
) -> list[LlamaCppResult]:
    """Run independent completions on one loaded model, then unload it once."""
    if not media_items:
        raise InputNormalizationError(
            "Sequential generation requires at least one input item."
        )
    if prompt_items is not None and len(prompt_items) != len(media_items):
        raise InputNormalizationError(
            "Sequential generation requires exactly one prompt per input item."
        )
    if (
        normalize_ngram_speculative(kwargs.get("ngram_speculative"))["speculative_mode"]
        != "off"
    ):
        raise InputNormalizationError(
            "Sequential generation does not support N-gram speculative decoding because "
            "its history map may carry state between items."
        )

    native = bindings or _import_bindings()
    session = _SequentialLlamaSession(native.llama_class)
    session_bindings = LlamaCppBindings(
        llama_class=session.create,
        handlers=native.handlers,
        jinja_formatter_class=native.jinja_formatter_class,
        chat_formatter_to_handler=native.chat_formatter_to_handler,
    )
    results: list[LlamaCppResult] = []
    cleanup_seconds = 0.0
    execution_error: Exception | None = None
    cleanup_error: Exception | None = None
    try:
        with _NATIVE_EXECUTION_LOCK:
            for index, media in enumerate(media_items):
                item_kwargs = kwargs
                if prompt_items is not None:
                    item_kwargs = {**kwargs, "prompt": prompt_items[index]}
                results.append(
                    run_chat(
                        media=media,
                        bindings=session_bindings,
                        **item_kwargs,
                    )
                )
    except Exception as exc:
        execution_error = exc
    finally:
        cleanup_started = time.perf_counter()
        try:
            session.close()
        except Exception as exc:  # pragma: no cover - platform-specific native failure
            cleanup_error = exc
        cleanup_seconds = time.perf_counter() - cleanup_started

    if execution_error is not None:
        if cleanup_error is not None:
            _LOGGER.warning(
                "Sequential llama.cpp cleanup failed after an execution error: %s",
                cleanup_error,
            )
        raise execution_error
    if cleanup_error is not None:
        raise BackendError(
            "llama-cpp-python completed, but the sequential model could not be fully "
            f"unloaded: {cleanup_error}"
        ) from cleanup_error

    item_count = len(results)
    for index, result in enumerate(results):
        result.metrics.update(
            model_unloaded=True,
            sequential={
                "item_index": index,
                "item_count": item_count,
                "context_reset_before_item": True,
                "model_reused": item_count > 1,
                "model_unloaded_after_sequence": True,
            },
        )
        result.media_diagnostics["model_unloaded_after_response"] = False
        result.media_diagnostics["model_unloaded_after_sequence"] = True
    if results:
        results[-1].metrics["cleanup_seconds"] += cleanup_seconds
        results[-1].metrics["total_seconds"] += cleanup_seconds
    if session.reset_count != item_count:
        raise BackendError(
            "Sequential generation did not reset context exactly once per item."
        )
    return results


__all__ = [
    "HANDLER_NAMES",
    "REASONING_STRENGTHS",
    "LlamaCppBindings",
    "NativeSpeculativeBindings",
    "LlamaCppResult",
    "LlamaCppSession",
    "normalize_ngram_speculative",
    "require_native_speculative",
    "run_chat",
    "run_chat_sequential",
]
