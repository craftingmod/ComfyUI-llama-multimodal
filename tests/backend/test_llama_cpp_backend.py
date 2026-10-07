import base64
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import backend.backends.llama_cpp as llama_cpp_backend
from backend.backends.llama_cpp import (
    LlamaCppBindings,
    LlamaCppSession,
    run_chat,
    run_chat_sequential,
)
from backend.core import (
    BackendError,
    InputNormalizationError,
    normalize_audio,
    normalize_images,
    normalize_media,
    normalize_video,
)
from backend.llama_cpp.llama_cpp_session_cleanup import close_tracked_sessions
from tests.backend.tensor_stub import VideoInputStub, silent_audio, solid_image

NATIVE_EVENTS = []


def test_decision_output_rule_preserves_existing_system_and_messages():
    for content in ("", "User instruction"):
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": "Question"},
        ]
        adapted = llama_cpp_backend._with_decision_output_rule(messages)
        assert adapted[0]["content"].endswith(llama_cpp_backend._DECISION_OUTPUT_RULE)
        assert content in adapted[0]["content"]
        assert adapted[1] == messages[1]
        assert messages[0]["content"] == content
    adapted = llama_cpp_backend._with_decision_output_rule([messages[1]])
    assert adapted[0] == {
        "role": "system",
        "content": llama_cpp_backend._DECISION_OUTPUT_RULE,
    }


class FakeMTMDHandler:
    is_support_vision = True
    is_support_audio = True
    is_support_video = False

    def __call__(self, **_kwargs):
        return object()

    def close(self):
        self.closed = True


class FakeLlama:
    instances = []
    metadata = {}
    response = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    generation_error = None
    close_error = None
    nextn_layers = 0
    speculative_stats = {
        "drafted": 20,
        "accepted_draft_tokens": 9,
        "draft_calls": 4,
        "accept_calls": 3,
        "draft_token_acceptance_rate": 0.45,
    }
    token_values = {}
    token_pieces = {}
    decision_token_ids = {}
    decision_logits = {"A": 0.0, "B": 2.0}
    prefill_error = None

    def __init__(self, **kwargs):
        NATIVE_EVENTS.append("target")
        self.kwargs = kwargs
        self.completion_kwargs = None
        self.completion_kwargs_history = []
        self.prefill_messages = []
        self.prefill_count = 0
        self.reset_count = 0
        self.metadata = dict(type(self).metadata)
        self.last_speculative_stats = dict(type(self).speculative_stats)
        self.closed = False
        self.close_count = 0
        if "chat_handler" in kwargs:
            self.chat_handler = kwargs["chat_handler"]
        elif "mmproj_path" in kwargs:
            self.chat_handler = FakeMTMDHandler()
            self.chat_handler.closed = False
        else:
            self.chat_handler = None
        type(self).instances.append(self)

    def create_chat_completion(self, **kwargs):
        self.completion_kwargs = kwargs
        self.completion_kwargs_history.append(kwargs)
        if type(self).generation_error is not None:
            raise type(self).generation_error
        return type(self).response

    def tokenize(self, value, *, add_bos, special):
        assert add_bos is False
        assert special is False
        target = value.decode("utf-8")
        return [type(self).decision_token_ids.get(target, ord(target))]

    def create_chat_prefill(self, *, messages):
        self.prefill_count += 1
        self.prefill_messages.append(messages)
        if type(self).prefill_error is not None:
            raise type(self).prefill_error
        logits = [0.0] * 128
        for target, score in type(self).decision_logits.items():
            token_id = type(self).decision_token_ids.get(target, ord(target))
            logits[token_id] = score
        return SimpleNamespace(logits=logits)

    def reset(self):
        self.reset_count += 1

    def n_layer_nextn(self):
        return type(self).nextn_layers

    def _token_value(self, name):
        return type(self).token_values.get(name, -1)

    def token_eos(self):
        return self._token_value("eos_token")

    def token_bos(self):
        return self._token_value("bos_token")

    def token_eot(self):
        return self._token_value("eot_token")

    def token_sep(self):
        return self._token_value("sep_token")

    def token_nl(self):
        return self._token_value("nl_token")

    def token_pad(self):
        return self._token_value("pad_token")

    def token_mask(self):
        return self._token_value("mask_token")

    def detokenize(self, tokens, *, special=False):
        assert special is True
        return b"".join(type(self).token_pieces.get(token, b"") for token in tokens)

    def close(self):
        self.close_count += 1
        if type(self).close_error is not None:
            raise type(self).close_error
        self.closed = True
        if self.chat_handler is not None:
            close_handler = getattr(self.chat_handler, "close", None)
            if callable(close_handler):
                close_handler()


class FakeHandler(FakeMTMDHandler):
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        type(self).instances.append(self)

    def close(self):
        self.closed = True


class FakeVideoHandler(FakeHandler):
    is_support_video = True

    def __init__(self, *, chat_format, **kwargs):
        super().__init__(chat_format=chat_format, **kwargs)


class FakeJinjaFormatter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.call_kwargs = None
        type(self).instances.append(self)

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return object()


class FakeSpeculativeType:
    @classmethod
    def from_str(cls, value):
        return value


class FakeSpecConfig:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)


@pytest.fixture(autouse=True)
def reset_fakes():
    NATIVE_EVENTS.clear()
    FakeLlama.instances = []
    FakeLlama.response = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    FakeLlama.generation_error = None
    FakeLlama.close_error = None
    FakeLlama.metadata = {}
    FakeLlama.nextn_layers = 0
    FakeLlama.speculative_stats = {
        "drafted": 20,
        "accepted_draft_tokens": 9,
        "draft_calls": 4,
        "accept_calls": 3,
        "draft_token_acceptance_rate": 0.45,
    }
    FakeLlama.token_values = {}
    FakeLlama.token_pieces = {}
    FakeLlama.decision_token_ids = {}
    FakeLlama.decision_logits = {"A": 0.0, "B": 2.0}
    FakeLlama.prefill_error = None
    FakeHandler.instances = []
    FakeJinjaFormatter.instances = []
    FakeSpecConfig.instances = []


def make_bindings(
    *,
    jinja_formatter_class=None,
    chat_formatter_to_handler=None,
    **handlers,
):
    return LlamaCppBindings(
        llama_class=FakeLlama,
        handlers=handlers,
        jinja_formatter_class=jinja_formatter_class,
        chat_formatter_to_handler=chat_formatter_to_handler,
    )


def make_speculative_api():
    return llama_cpp_backend.NativeSpeculativeBindings(
        spec_config=FakeSpecConfig,
        speculative_type=FakeSpeculativeType,
    )


def gguf_files(tmp_path):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projector")
    return model, mmproj


def test_backend_core_has_no_llama_private_api_or_broad_proxy():
    source = Path(llama_cpp_backend.__file__).read_text(encoding="utf-8")
    forbidden = (
        "llm._model",
        "llm._ctx",
        "llm._sampling_ctx",
        "llm._hybrid_cache_mgr",
        "llm._stack",
        "handler._get_media_items",
        "handler._mtmd_tokenize",
        "handler._process_mtmd_prompt",
        "llm.speculative",
        "selector_top_k",
        "is_dflash2",
        "LlamaNGramMapDecoding",
        "ngram_sync_check_tokens",
        "__getattr__",
        "__setattr__",
    )
    assert [name for name in forbidden if name in source] == []


def test_missing_optional_dependency_points_to_supported_fork(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    with pytest.raises(BackendError) as error:
        llama_cpp_backend._import_bindings()

    message = str(error.value)
    assert "JamePeng's multimodal llama-cpp-python fork" in message
    assert "LLAMA_CPP_PYTHON_VISION_INSTALL.md" in message
    assert "https://github.com/JamePeng/llama-cpp-python/releases/" in message


def test_missing_native_speculative_api_has_actionable_error(monkeypatch):
    llama_cpp_package = ModuleType("llama_cpp")
    llama_cpp_package.__path__ = []
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp_package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_speculative", None)

    with pytest.raises(BackendError) as error:
        llama_cpp_backend._import_native_speculative_bindings()

    message = str(error.value)
    assert "not installed in the Python environment that runs ComfyUI" in message
    assert "SpecConfig/SpeculativeType" in message
    assert "https://github.com/JamePeng/llama-cpp-python/releases/" in message
    assert "CUDA runtime, and native DLLs" in message
    assert "No model was loaded" in message


def test_implicit_draft_model_path_api_is_rejected(tmp_path):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"draft")

    with pytest.raises(InputNormalizationError, match="official JamePeng SpecConfig"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_images(None),
            draft_model_path=str(draft),
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


def test_target_only_path_does_not_import_or_pass_speculative_binding(
    tmp_path, monkeypatch
):
    model, _ = gguf_files(tmp_path)

    def unexpected_import():
        raise AssertionError("target-only generation imported the speculative API")

    monkeypatch.setattr(
        llama_cpp_backend,
        "_import_native_speculative_bindings",
        unexpected_import,
    )
    run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_images(None),
        ngram_speculative={"speculative_mode": "off", "ignored": "value"},
        bindings=make_bindings(),
    )

    assert "draft_model" not in FakeLlama.instances[0].kwargs


def test_text_only_generate_omits_selected_mmproj_but_forwards_thinking(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="gemma4",
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=True,
        reasoning_strength="xhigh",
        bindings=make_bindings(gemma4=FakeHandler),
    )

    assert "mmproj_path" not in FakeLlama.instances[0].kwargs
    assert FakeLlama.instances[0].kwargs["chat_handler_kwargs"] == {
        "verbose": False,
        "extra_template_arguments": {
            "enable_thinking": True,
            "force_reasoning": True,
            "reasoning_strength": "xhigh",
        },
    }
    assert FakeHandler.instances == []
    assert result.media_diagnostics["mmproj"] is None


@pytest.mark.parametrize(
    ("type_k", "type_v", "expected_type_k", "expected_type_v"),
    [
        ("FP16", "FP16", 1, 1),
        ("Q8_0", "Q4_0", 8, 2),
        ("Q4_0", "Q8_0", 2, 8),
    ],
)
def test_kv_cache_types_map_to_native_integer_ids(
    tmp_path, type_k, type_v, expected_type_k, expected_type_v
):
    model, _ = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        type_k=type_k,
        type_v=type_v,
        bindings=make_bindings(),
    )

    assert FakeLlama.instances[0].kwargs["type_k"] == expected_type_k
    assert FakeLlama.instances[0].kwargs["type_v"] == expected_type_v


def test_text_only_auto_reasoning_leaves_template_arguments_untouched(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ enable_thinking }}"}

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=None,
        reasoning_strength="xhigh",
        reasoning_budget=512,
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=lambda formatter: formatter,
        ),
    )

    assert FakeLlama.instances[0].kwargs["chat_handler_kwargs"] == {"verbose": False}
    assert FakeJinjaFormatter.instances == []
    assert result.metrics["configuration"]["thinking"] is None
    assert result.metrics["configuration"]["reasoning_strength"] == "auto"
    assert result.metrics["configuration"]["reasoning_budget"] == 0


def test_text_only_jinja_handler_receives_disabled_thinking_arguments(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ enable_thinking }}"}
    configured_formatters = []

    def to_handler(formatter):
        configured_formatters.append(formatter)
        return formatter

    run_chat(
        model_path=str(model),
        system="",
        prompt="translate this",
        media=normalize_media(),
        thinking=False,
        reasoning_strength="xhigh",
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=to_handler,
        ),
    )

    formatter = FakeJinjaFormatter.instances[0]
    assert formatter.kwargs["template"] == "{{ enable_thinking }}"
    configured_formatters[0](messages=[])
    assert formatter.call_kwargs["enable_thinking"] is False
    assert formatter.call_kwargs["force_reasoning"] is False
    assert "reasoning_strength" not in formatter.call_kwargs


def test_text_template_uses_public_detokenize_for_special_tokens(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ eos_token }}"}
    FakeLlama.token_values = {
        "eos_token": 1,
        "bos_token": 2,
        "eot_token": 3,
    }
    FakeLlama.token_pieces = {
        1: b"<eos>",
        2: b"<bos>",
        3: b"\xe2",
    }

    run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=False,
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=lambda formatter: formatter,
        ),
    )

    formatter = FakeJinjaFormatter.instances[0]
    assert formatter.kwargs["eos_token"] == "<eos>"
    assert formatter.kwargs["bos_token"] == "<bos>"
    assert formatter.kwargs["special_tokens_map"]["eot_token"] == "�"


def test_text_only_thinking_template_requires_configurable_jinja_api(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {
        "tokenizer.chat_template": "{% if enable_thinking %}think{% endif %}"
    }

    with pytest.raises(BackendError, match="cannot pass thinking controls"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="translate this",
            media=normalize_media(),
            thinking=False,
            bindings=make_bindings(),
        )

    assert FakeLlama.instances[0].closed is True


def test_ngram_speculative_uses_official_spec_config(tmp_path):
    model, _ = gguf_files(tmp_path)

    result = run_chat(
        model_path=str(model),
        system="system",
        prompt="repeat a template",
        media=normalize_media(),
        ngram_speculative=llama_cpp_backend.normalize_ngram_speculative(
            {
                "speculative_mode": "ngram",
                "ngram_size": 4,
                "num_pred_tokens": 12,
                "ngram_mode": "k4v",
                "ngram_min_hits": 3,
                "ngram_max_entries_per_key": 0,
            }
        ),
        bindings=make_bindings(),
        speculative_api=make_speculative_api(),
    )

    assert NATIVE_EVENTS == ["target"]
    config = FakeSpecConfig.instances[0]
    assert config.kwargs == {
        "spec_type": "ngram-map-k4v",
        "ngram_size_n": 4,
        "ngram_size_m": 12,
        "ngram_min_hits": 3,
        "ngram_max_entries_per_key": None,
    }
    instance = FakeLlama.instances[0]
    assert instance.kwargs["speculative"] is config
    assert instance.closed is True
    assert result.metrics["ngram_speculative"] == {
        "speculative_mode": "ngram",
        "ngram_size": 4,
        "num_pred_tokens": 12,
        "ngram_min_hits": 3,
        "ngram_max_entries_per_key": None,
        "ngram_mode": "k4v",
        "implementation": "ngram-map-k4v",
        "stats": {
            "drafted": 20,
            "accepted_draft_tokens": 9,
            "draft_calls": 4,
            "accept_calls": 3,
            "draft_token_acceptance_rate": 0.45,
            "drafted_tokens": 20,
            "accepted_tokens": 9,
            "acceptance_rate": 0.45,
            "mean_accepted_tokens": 2.25,
            "mean_accepted_per_call": 2.25,
        },
    }
    assert "speculative" not in result.metrics


def test_ngram_speculative_rejects_multimodal_requests(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_images(solid_image(1, 2, 3, 3, 0.5))

    with pytest.raises(InputNormalizationError, match="text-only"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            handler="auto",
            system="system",
            prompt="repeat a template",
            media=bundle,
            ngram_speculative={
                "speculative_mode": "ngram",
                "ngram_size": 4,
                "num_pred_tokens": 12,
                "ngram_mode": "k4v",
                "ngram_min_hits": 3,
                "ngram_max_entries_per_key": 8,
            },
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeLlama.instances == []


def test_ngram_speculative_generation_failure_closes_target(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.generation_error = RuntimeError("generation failed")

    with pytest.raises(BackendError, match="generation failed"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="repeat",
            media=normalize_images(None),
            ngram_speculative={
                "speculative_mode": "ngram",
                "ngram_size": 3,
                "num_pred_tokens": 10,
                "ngram_mode": "k",
                "ngram_min_hits": 2,
                "ngram_max_entries_per_key": 8,
            },
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeLlama.instances[0].closed is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"ngram_size": 0}, "ngram_size"),
        ({"num_pred_tokens": 33}, "num_pred_tokens"),
        ({"ngram_mode": "other"}, "ngram_mode"),
        ({"ngram_min_hits": 0}, "ngram_min_hits"),
        ({"ngram_max_entries_per_key": 1025}, "ngram_max_entries_per_key"),
    ],
)
def test_ngram_speculative_parameters_are_validated(override, message):
    values = {
        "speculative_mode": "ngram",
        "ngram_size": 3,
        "num_pred_tokens": 10,
        "ngram_mode": "k",
        "ngram_min_hits": 2,
        "ngram_max_entries_per_key": 8,
    }
    values.update(override)

    with pytest.raises(InputNormalizationError, match=message):
        llama_cpp_backend.normalize_ngram_speculative(values)


def test_native_and_ngram_speculative_modes_cannot_be_combined(tmp_path):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"draft")

    with pytest.raises(InputNormalizationError, match="cannot be enabled together"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_images(None),
            draft_model_path=str(draft),
            spec_type="draft-dflash",
            ngram_speculative={
                "speculative_mode": "ngram",
                "ngram_size": 3,
                "num_pred_tokens": 10,
                "ngram_mode": "k",
                "ngram_min_hits": 2,
                "ngram_max_entries_per_key": 8,
            },
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeLlama.instances == []


def test_native_speculative_uses_official_spec_config_and_stats(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    draft = tmp_path / "dflash.gguf"
    draft.write_bytes(b"draft")

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        system="",
        prompt="hello",
        media=normalize_images(None),
        gpu_layers="all",
        draft_model_path=str(draft),
        spec_type="draft-dflash",
        spec_n_max=15,
        spec_n_min=2,
        spec_p_min=0.25,
        draft_n_gpu_layers=0,
        draft_backend_sampling=False,
        bindings=make_bindings(),
        speculative_api=make_speculative_api(),
    )

    assert NATIVE_EVENTS == ["target"]
    config = FakeSpecConfig.instances[0]
    assert config.kwargs == {
        "spec_type": "draft-dflash",
        "draft_model_path": str(draft.resolve()),
        "draft_n_max": 15,
        "draft_n_min": 2,
        "draft_p_min": 0.25,
        "draft_n_gpu_layers": 0,
        "draft_backend_sampling": False,
    }
    target = FakeLlama.instances[0]
    assert target.kwargs["speculative"] is config
    assert "draft_model" not in target.kwargs
    assert "mmproj_path" not in target.kwargs
    assert "chat_handler_kwargs" in target.kwargs
    assert target.closed is True
    assert result.metrics["speculative"]["stats"] == {
        "drafted": 20,
        "accepted_draft_tokens": 9,
        "draft_calls": 4,
        "accept_calls": 3,
        "draft_token_acceptance_rate": 0.45,
        "drafted_tokens": 20,
        "accepted_tokens": 9,
        "acceptance_rate": 0.45,
        "mean_accepted_tokens": 2.25,
        "mean_accepted_per_call": 2.25,
    }
    assert result.metrics["speculative"]["implementation"] == "draft-dflash"
    assert result.metrics["speculative"]["draft_model"] == "dflash.gguf"
    assert result.metrics["speculative"]["n_max"] == 15
    assert result.metrics["speculative"]["n_min"] == 2
    assert result.metrics["speculative"]["p_min"] == 0.25


def test_native_speculative_target_initialization_failure_is_not_silently_disabled(
    tmp_path,
):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "dflash.gguf"
    draft.write_bytes(b"draft")

    class FailingLlama:
        def __init__(self, **_kwargs):
            NATIVE_EVENTS.append("target")
            raise RuntimeError("target init failed")

    with pytest.raises(BackendError, match="target init failed"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_media(),
            draft_model_path=str(draft),
            spec_type="draft-dflash",
            bindings=LlamaCppBindings(llama_class=FailingLlama, handlers={}),
            speculative_api=make_speculative_api(),
        )

    assert NATIVE_EVENTS == ["target"]
    assert FakeSpecConfig.instances


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"spec_type": "unknown"}, "spec_type"),
        ({"spec_n_max": 0}, "spec_n_max"),
        ({"spec_n_max": 2, "spec_n_min": 3}, "spec_n_min"),
        ({"spec_p_min": 1.1}, "spec_p_min"),
    ],
)
def test_native_speculative_parameters_are_validated_before_loading(
    tmp_path, overrides, message
):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"draft")

    native_values = {"spec_type": "draft-dflash", **overrides}
    with pytest.raises(InputNormalizationError, match=message):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_images(None),
            draft_model_path=str(draft),
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
            **native_values,
        )

    assert FakeSpecConfig.instances == []
    assert FakeLlama.instances == []


@pytest.mark.parametrize("spec_type", ["draft-dflash", "draft-dspark"])
def test_native_speculative_requires_a_draft_gguf(tmp_path, spec_type):
    model, _ = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="requires a compatible draft"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_media(),
            spec_type=spec_type,
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeSpecConfig.instances == []
    assert FakeLlama.instances == []


def test_gemma4_external_mtp_forwards_official_config_and_reports_stats(tmp_path):
    model, _ = gguf_files(tmp_path)
    assistant = tmp_path / "gemma4-assistant.gguf"
    assistant.write_bytes(b"assistant")
    FakeLlama.response["choices"][0]["finish_reason"] = "stop"

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        gpu_layers="all",
        draft_model_path=str(assistant),
        spec_type="draft-mtp",
        mtp_provider="external",
        spec_n_max=2,
        spec_n_min=1,
        spec_p_min=0.2,
        verbose=True,
        bindings=make_bindings(),
        speculative_api=make_speculative_api(),
    )

    assert NATIVE_EVENTS == ["target"]
    config = FakeSpecConfig.instances[0]
    assert config.kwargs == {
        "spec_type": "draft-mtp",
        "draft_model_path": str(assistant.resolve()),
        "draft_n_max": 2,
        "draft_n_min": 1,
        "draft_p_min": 0.2,
        "draft_n_gpu_layers": "all",
        "draft_backend_sampling": True,
    }
    target = FakeLlama.instances[0]
    assert target.kwargs["speculative"] is config
    assert target.kwargs["n_seq_max"] == 1
    assert "native_context_reprefill" not in target.kwargs
    assert target.closed is True
    stats = result.metrics["speculative"]["stats"]
    assert stats["drafted_tokens"] == 20
    assert stats["accepted_tokens"] == 9
    assert result.metrics["speculative"]["mtp_provider"] == "external"
    assert result.metrics["speculative"]["verbose"] is True
    assert result.metrics["speculative"]["n_layer_nextn"] is None
    assert result.metrics["speculative"]["completion_tokens"] == 2
    assert result.metrics["speculative"]["finish_reason"] == "stop"
    assert result.metrics["speculative"]["tokens_per_second"] > 0


def test_qwen35_internal_mtp_passes_none_and_validates_embedded_nextn(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.nextn_layers = 2

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        gpu_layers="all",
        spec_type="draft-mtp",
        mtp_provider="internal",
        spec_n_max=2,
        bindings=make_bindings(),
        speculative_api=make_speculative_api(),
    )

    config = FakeSpecConfig.instances[0]
    assert config.kwargs["draft_model_path"] is None
    assert config.kwargs["spec_type"] == "draft-mtp"
    assert FakeLlama.instances[0].kwargs["speculative"] is config
    assert result.metrics["speculative"]["mtp_provider"] == "internal"
    assert result.metrics["speculative"]["draft_model"] is None
    assert result.metrics["speculative"]["n_layer_nextn"] == 2
    assert FakeLlama.instances[0].closed is True


def test_qwen35_internal_mtp_rejects_target_without_nextn_and_unloads(tmp_path):
    model, _ = gguf_files(tmp_path)

    with pytest.raises(BackendError, match="no usable embedded NextN/MTP layers"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_media(),
            gpu_layers="all",
            spec_type="draft-mtp",
            mtp_provider="internal",
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeLlama.instances[0].closed is True
    assert FakeSpecConfig.instances[0].kwargs["spec_type"] == "draft-mtp"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"mtp_provider": "internal", "spec_type": "none"},
            "mtp_provider must be off",
        ),
        (
            {"mtp_provider": "internal", "spec_type": "draft-dflash"},
            "spec_type must be draft-mtp",
        ),
        (
            {"mtp_provider": "off"},
            "draft-mtp requires mtp_provider",
        ),
        (
            {"mtp_provider": "external"},
            "requires a draft GGUF",
        ),
        (
            {"mtp_provider": "internal", "draft_model_path": "{draft}"},
            "leave draft_model unselected",
        ),
        (
            {"mtp_provider": "internal", "gpu_layers": "cpu"},
            "requires gpu_layers=all",
        ),
        (
            {"mtp_provider": "internal", "spec_n_max": 0},
            "spec_n_max",
        ),
        (
            {
                "mtp_provider": "internal",
                "spec_n_max": 2,
                "spec_n_min": 3,
            },
            "spec_n_min",
        ),
        (
            {"mtp_provider": "internal", "spec_p_min": 1.1},
            "spec_p_min",
        ),
    ],
)
def test_mtp_configuration_is_validated_before_loading(tmp_path, kwargs, message):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "assistant.gguf"
    draft.write_bytes(b"draft")
    resolved_kwargs = {
        key: (str(draft) if value == "{draft}" else value)
        for key, value in kwargs.items()
    }

    call_kwargs = {
        "model_path": str(model),
        "system": "",
        "prompt": "hello",
        "media": normalize_media(),
        "gpu_layers": "all",
        "spec_type": "draft-mtp",
        "bindings": make_bindings(),
        "speculative_api": make_speculative_api(),
        **resolved_kwargs,
    }
    with pytest.raises(InputNormalizationError, match=message):
        run_chat(**call_kwargs)

    assert FakeSpecConfig.instances == []
    assert FakeLlama.instances == []


def test_spec_type_none_runs_target_only_even_with_a_stale_draft_selection(tmp_path):
    model, _ = gguf_files(tmp_path)
    draft = tmp_path / "stale-draft.gguf"
    draft.write_bytes(b"draft")

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        draft_model_path=str(draft),
        spec_type="none",
        mtp_provider="off",
        bindings=make_bindings(),
    )

    assert "draft_model" not in FakeLlama.instances[0].kwargs
    assert "speculative" not in result.metrics


@pytest.mark.parametrize(
    ("spec_type", "mtp_provider"),
    [
        ("draft-dflash", "off"),
        ("draft-dspark", "off"),
        ("draft-mtp", "internal"),
    ],
)
def test_native_speculative_rejects_media_before_loading(
    tmp_path, spec_type, mtp_provider
):
    model, mmproj = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="text-only"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            system="",
            prompt="hello",
            media=normalize_images(solid_image(1, 2, 2, 3, 0.5)),
            gpu_layers="all",
            spec_type=spec_type,
            mtp_provider=mtp_provider,
            bindings=make_bindings(),
            speculative_api=make_speculative_api(),
        )

    assert FakeSpecConfig.instances == []
    assert FakeLlama.instances == []


def test_mtp_initialization_failure_does_not_fallback_to_target_only(tmp_path):
    model, _ = gguf_files(tmp_path)

    class FailingLlama:
        def __init__(self, **_kwargs):
            raise RuntimeError("ABI mismatch")

    with pytest.raises(BackendError, match="Native MTP initialization failed"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_media(),
            gpu_layers="all",
            spec_type="draft-mtp",
            mtp_provider="internal",
            bindings=LlamaCppBindings(llama_class=FailingLlama, handlers={}),
            speculative_api=make_speculative_api(),
        )

    assert FakeSpecConfig.instances[0].kwargs["draft_model_path"] is None


def test_run_chat_sends_all_images_once_and_unloads_model(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_images(
        [solid_image(1, 2, 3, 3, 0.1), solid_image(1, 4, 1, 3, 0.9)]
    )

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="system",
        prompt="compare",
        media=bundle,
        gpu_layers="all",
        flash_attention="enabled",
        max_tokens=42,
        presence_penalty=1.5,
        seed=7,
        stop="END",
        bindings=make_bindings(generic=FakeHandler),
    )

    assert len(FakeLlama.instances) == 1
    instance = FakeLlama.instances[0]
    assert instance.kwargs["model_path"] == str(model.resolve())
    assert instance.kwargs["chat_handler"] is FakeHandler.instances[0]
    assert FakeHandler.instances[0].kwargs == {
        "mmproj_path": str(mmproj.resolve()),
        "verbose": False,
        "extra_template_arguments": {
            "enable_thinking": False,
            "force_reasoning": False,
        },
        "chat_format": None,
    }
    assert instance.kwargs["n_gpu_layers"] == "all"
    assert instance.kwargs["flash_attn_type"] == 1
    assert instance.closed is True
    assert instance.completion_kwargs["max_tokens"] == 42
    assert instance.completion_kwargs["present_penalty"] == 1.5
    assert "presence_penalty" not in instance.completion_kwargs
    assert instance.completion_kwargs["seed"] == 7
    assert instance.completion_kwargs["stop"] == ["END"]

    messages = instance.completion_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "system"}
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "compare"}
    data_uris = [part["image_url"]["url"] for part in content[1:]]
    assert len(data_uris) == 2
    assert all(uri.startswith("data:image/png;base64,") for uri in data_uris)
    assert [base64.b64decode(uri.split(",", 1)[1]) for uri in data_uris] == [
        item.payload for item in bundle.items
    ]
    assert result.response == "done"
    assert result.metrics["usage"]["total_tokens"] == 12
    assert result.metrics["model_unloaded"] is True
    assert result.metrics["configuration"]["presence_penalty"] == 1.5
    assert result.media_diagnostics["capabilities"] == {
        "vision": True,
        "audio": True,
        "video": False,
    }
    assert result.media_diagnostics["evaluated"] == {
        "media_count": 2,
        "image_count": 2,
        "audio_count": 0,
        "video_count": 0,
    }
    assert result.media_diagnostics["mtmd"] == {
        "strict_pipeline": True,
        "completion_succeeded": True,
        "all_media_evaluated": True,
        "verification": "mtmd_evaluated",
    }
    assert result.media_diagnostics["model_unloaded_after_response"] is True


def test_auto_adapts_images_for_muse_glimmer_embedded_template(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_images(
        [solid_image(1, 2, 3, 3, 0.1), solid_image(1, 4, 1, 3, 0.9)]
    )
    FakeLlama.metadata = {"general.architecture": "muse-glimmer"}

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        system="",
        prompt="compare",
        media=bundle,
        bindings=make_bindings(),
    )

    content = FakeLlama.instances[0].completion_kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image", "image"]
    assert all(
        part["image"].startswith("data:image/png;base64,") for part in content[1:]
    )


def test_explicit_generic_keeps_openai_image_parts_for_muse_glimmer(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    FakeLlama.metadata = {"general.architecture": "muse-glimmer"}

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 2, 3, 3, 0.5)),
        bindings=make_bindings(generic=FakeHandler),
    )

    content = FakeLlama.instances[0].completion_kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]


def test_run_chat_sends_audio_as_base64_pcm16_wav(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_audio(
        {"waveform": silent_audio(1, 1, 80), "sample_rate": 16_000}
    )

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        system="",
        prompt="transcribe",
        media=bundle,
        bindings=make_bindings(),
    )

    content = FakeLlama.instances[0].completion_kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "input_audio"]
    audio = content[1]["input_audio"]
    assert audio["format"] == "wav"
    assert base64.b64decode(audio["data"]) == bundle.items[0].payload
    assert base64.b64decode(audio["data"]).startswith(b"RIFF")
    assert FakeLlama.instances[0].closed is True


def test_run_chat_sequential_reuses_model_resets_each_audio_and_unloads_once(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundles = [
        normalize_audio(
            {"waveform": silent_audio(1, 1, sample_count), "sample_rate": 16_000}
        )
        for sample_count in (80, 160)
    ]

    results = run_chat_sequential(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        system="",
        prompt="transcribe independently",
        media_items=bundles,
        bindings=make_bindings(),
    )

    assert len(FakeLlama.instances) == 1
    instance = FakeLlama.instances[0]
    assert instance.reset_count == 2
    assert len(instance.completion_kwargs_history) == 2
    assert instance.closed is True
    assert instance.close_count == 1
    payloads = [
        base64.b64decode(call["messages"][-1]["content"][1]["input_audio"]["data"])
        for call in instance.completion_kwargs_history
    ]
    assert payloads == [bundle.items[0].payload for bundle in bundles]
    assert len(results) == 2
    assert all(result.response == "done" for result in results)
    assert all(
        result.metrics["sequential"]["context_reset_before_item"] is True
        for result in results
    )
    assert all(
        result.media_diagnostics["model_unloaded_after_sequence"] is True
        for result in results
    )


def test_retained_session_reuses_model_until_prompt_end_unload(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    session = LlamaCppSession(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        bindings=make_bindings(),
    )
    bundles = [
        normalize_audio(
            {"waveform": silent_audio(1, 1, sample_count), "sample_rate": 16_000}
        )
        for sample_count in (80, 160)
    ]

    first = session.generate(
        system="",
        prompt="first",
        media=bundles[0],
        max_tokens=32,
        seed=-1,
        stop="",
    )
    second = session.generate(
        system="",
        prompt="second",
        media=bundles[1],
        max_tokens=32,
        seed=-1,
        stop="",
    )

    assert len(FakeLlama.instances) == 1
    instance = FakeLlama.instances[0]
    assert instance.reset_count == 2
    assert instance.closed is False
    assert first.metrics["model_unloaded"] is False
    assert second.metrics["session"] == {
        "execution_index": 1,
        "model_reused": True,
        "unload_required": True,
    }
    assert second.media_diagnostics["model_unloaded_after_response"] is False

    close_tracked_sessions()

    assert session.closed is True
    assert instance.closed is True
    assert instance.close_count == 1


def test_retained_session_allows_kv_prefix_reuse_with_media_before_prompt(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    session = LlamaCppSession(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        bindings=make_bindings(),
    )
    media = normalize_images(solid_image(1, 1, 1, 3, 0.5))

    try:
        session.generate(
            system="",
            prompt="first question",
            media=media,
            max_tokens=8,
            seed=-1,
            stop="",
        )
        session.generate(
            system="",
            prompt="second question",
            media=media,
            max_tokens=8,
            seed=-1,
            stop="",
            media_before_prompt=True,
            reuse_kv_cache=True,
        )
        session.generate(
            system="",
            prompt="legacy question",
            media=media,
            max_tokens=8,
            seed=-1,
            stop="",
        )

        instance = FakeLlama.instances[0]
        assert len(FakeLlama.instances) == 1
        assert instance.reset_count == 2
        assert [
            part["type"]
            for part in instance.completion_kwargs_history[0]["messages"][-1]["content"]
        ] == ["text", "image_url"]
        assert [
            part["type"]
            for part in instance.completion_kwargs_history[1]["messages"][-1]["content"]
        ] == ["image_url", "text"]
        assert [
            part["type"]
            for part in instance.completion_kwargs_history[2]["messages"][-1]["content"]
        ] == ["text", "image_url"]
        assert all(
            "reuse_kv_cache" not in kwargs
            for kwargs in instance.completion_kwargs_history
        )
    finally:
        session.close()


def test_retained_session_decides_with_prefill_then_reuses_model_for_generate(tmp_path):
    pytest.importorskip("makoto_decision")
    model, _ = gguf_files(tmp_path)
    session = LlamaCppSession(
        model_path=str(model),
        handler="auto",
        bindings=make_bindings(),
    )

    try:
        first, probabilities = session.decide(
            question="Choose one",
            context="Context: Keep the answer concise.",
            answers=["first answer", "second answer"],
        )
        second, _ = session.decide(
            question="Choose another",
            context="",
            answers=["left", "right"],
        )
        generated = session.generate(
            system="",
            prompt="continue",
            media=normalize_media(),
            max_tokens=8,
            seed=-1,
            stop="",
        )

        assert first == "second answer"
        assert second == "right"
        assert list(probabilities) == ["first answer", "second answer"]
        assert probabilities["second answer"] > probabilities["first answer"]
        assert len(FakeLlama.instances) == 1
        assert generated.metrics["session"] == {
            "execution_index": 2,
            "model_reused": True,
            "unload_required": True,
        }
        instance = FakeLlama.instances[0]
        assert instance.reset_count == 3
        assert instance.prefill_count == 2
        assert (
            "Context: Keep the answer concise."
            in instance.prefill_messages[0][-1]["content"]
        )
        for messages in instance.prefill_messages:
            assert messages[0]["role"] == "system"
            assert "Return exactly one choice label" in messages[0]["content"]
        assert instance.closed is False
    finally:
        session.close()

    assert instance.closed is True


def test_retained_session_decide_allows_media_prefix_reuse(tmp_path):
    pytest.importorskip("makoto_decision")
    model, mmproj = gguf_files(tmp_path)
    session = LlamaCppSession(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        bindings=make_bindings(generic=FakeHandler),
    )
    media = normalize_images(solid_image(1, 1, 1, 3, 0.5))

    try:
        for index, (reuse_kv_cache, media_before_prompt) in enumerate(
            ((False, False), (True, True), (False, True))
        ):
            session.decide(
                question=f"Question {index}",
                context="Shared context",
                answers=["first", "second"],
                media=media,
                reuse_kv_cache=reuse_kv_cache,
                media_before_prompt=media_before_prompt,
            )

        instance = FakeLlama.instances[0]
        assert instance.reset_count == 2
        legacy_parts = instance.prefill_messages[0][-1]["content"]
        assert [part["type"] for part in legacy_parts] == [
            "text",
            "image_url",
            "text",
        ]
        assert legacy_parts[0]["text"] == "Shared context"
        media_first_parts = instance.prefill_messages[1][-1]["content"]
        assert [part["type"] for part in media_first_parts] == [
            "image_url",
            "text",
        ]
        third_parts = instance.prefill_messages[2][-1]["content"]
        assert third_parts[0]["type"] == "image_url"
    finally:
        session.close()


def test_retained_session_rejects_duplicate_choice_tokens_and_missing_prefill(
    tmp_path,
):
    pytest.importorskip("makoto_decision")
    model, _ = gguf_files(tmp_path)
    session = LlamaCppSession(
        model_path=str(model),
        handler="auto",
        bindings=make_bindings(),
    )
    FakeLlama.decision_token_ids = {"A": 10, "B": 10}
    with pytest.raises(BackendError, match="same token ID"):
        session.decide("Choose", "", ["first", "second"])
    assert session.closed is True
    assert FakeLlama.instances[0].close_count == 1

    FakeLlama.decision_token_ids = {}
    FakeLlama.prefill_error = NotImplementedError("prefill is unavailable")
    session = LlamaCppSession(
        model_path=str(model),
        handler="auto",
        bindings=make_bindings(),
    )
    with pytest.raises(BackendError, match="chat handler does not support prefill"):
        session.decide("Choose", "", ["first", "second"])
    assert session.closed is True


def test_run_chat_preserves_image_then_audio_order_in_one_message(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_media(
        images=solid_image(1, 2, 3, 3, 0.5),
        audio={"waveform": silent_audio(1, 2, 160), "sample_rate": 16_000},
    )

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="system",
        prompt="analyze both",
        media=bundle,
        bindings=make_bindings(generic=FakeHandler),
    )

    content = FakeLlama.instances[0].completion_kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "input_audio",
    ]
    assert (
        base64.b64decode(content[2]["input_audio"]["data"]) == bundle.items[1].payload
    )
    assert result.media_diagnostics["requested"]["image_count"] == 1
    assert result.media_diagnostics["requested"]["audio_count"] == 1
    assert result.media_diagnostics["evaluated"]["image_count"] == 1
    assert result.media_diagnostics["evaluated"]["audio_count"] == 1


def test_run_chat_sends_comfy_video_as_internal_video_data_uri(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    bundle = normalize_video(VideoInputStub(b"fake-video-stream"))

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="",
        prompt="describe the video",
        media=bundle,
        bindings=make_bindings(generic=FakeVideoHandler),
    )

    content = FakeLlama.instances[0].completion_kwargs["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "video"]
    video_uri = content[1]["video"]["url"]
    assert video_uri.startswith("data:video/mp4;base64,")
    assert base64.b64decode(video_uri.split(",", 1)[1]) == b"fake-video-stream"
    assert result.media_diagnostics["capabilities"]["video"] is True
    assert result.media_diagnostics["requested"]["video_count"] == 1
    assert result.media_diagnostics["evaluated"]["video_count"] == 1
    assert result.media_diagnostics["mtmd"]["all_media_evaluated"] is True
    assert FakeVideoHandler.instances[0].kwargs["chat_format"] is None


def test_generic_handler_receives_custom_chat_template(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    custom_template = "CUSTOM_TEMPLATE_JINJA"

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="",
        prompt="describe the video",
        media=normalize_video(VideoInputStub(b"fake-video-stream")),
        custom_chat_template=custom_template,
        bindings=make_bindings(generic=FakeVideoHandler),
    )

    assert FakeVideoHandler.instances[0].kwargs["chat_format"] == custom_template


def test_specific_handler_is_created_and_owned_by_llama(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="gemma4",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        bindings=make_bindings(gemma4=FakeHandler),
    )

    handler = FakeHandler.instances[0]
    assert handler.kwargs == {
        "mmproj_path": str(mmproj.resolve()),
        "verbose": False,
        "enable_thinking": False,
    }
    assert FakeLlama.instances[0].kwargs["chat_handler"] is handler
    assert FakeLlama.instances[0].closed is True


def test_qwen3_asr_handler_is_available_for_audio_models(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="qwen3_asr",
        system="",
        prompt="transcribe",
        media=normalize_audio(
            {"waveform": silent_audio(1, 1, 80), "sample_rate": 16_000}
        ),
        bindings=make_bindings(qwen3_asr=FakeHandler),
    )

    handler = FakeHandler.instances[0]
    assert handler.kwargs == {
        "mmproj_path": str(mmproj.resolve()),
        "verbose": False,
        "extra_template_arguments": {
            "enable_thinking": False,
            "force_reasoning": False,
        },
    }
    assert FakeLlama.instances[0].kwargs["chat_handler"] is handler
    assert FakeLlama.instances[0].closed is True


def test_auto_multimodal_handler_passes_verbose_to_llama(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="auto",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        verbose=True,
        bindings=make_bindings(),
    )

    model_kwargs = FakeLlama.instances[0].kwargs
    assert model_kwargs["verbose"] is True
    assert model_kwargs["chat_handler"] is None
    assert FakeHandler.instances == []


def test_thinking_and_multimodal_overrides_reach_specific_handler(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    result = run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="gemma4",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        thinking=True,
        n_batch=1120,
        override_n_ubatch=True,
        n_ubatch=1120,
        override_image_min_tokens=True,
        image_min_tokens=1024,
        override_image_max_tokens=True,
        image_max_tokens=1120,
        bindings=make_bindings(gemma4=FakeHandler),
    )

    assert FakeHandler.instances[0].kwargs == {
        "mmproj_path": str(mmproj.resolve()),
        "verbose": False,
        "enable_thinking": True,
        "image_min_tokens": 1024,
        "image_max_tokens": 1120,
    }
    assert FakeLlama.instances[0].kwargs["n_ubatch"] == 1120
    assert result.metrics["configuration"] == {
        "thinking": True,
        "reasoning_strength": "auto",
        "reasoning_budget": 0,
        "reasoning_budget_applied": False,
        "reasoning_budget_format": None,
        "custom_chat_template": False,
        "n_ctx": 8192,
        "n_batch": 1120,
        "n_ubatch_override": 1120,
        "image_min_tokens_override": 1024,
        "image_max_tokens_override": 1120,
        "presence_penalty": 0.0,
    }


def test_qwen3_vl_thinking_maps_to_force_reasoning(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="qwen3_vl",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        thinking=True,
        bindings=make_bindings(qwen3_vl=FakeHandler),
    )

    assert FakeHandler.instances[0].kwargs["force_reasoning"] is True
    assert "extra_template_arguments" not in FakeHandler.instances[0].kwargs


def test_disabled_thinking_ignores_selected_reasoning_strength(tmp_path):
    model, _ = gguf_files(tmp_path)

    result = run_chat(
        model_path=str(model),
        handler="auto",
        system="",
        prompt="describe",
        media=normalize_media(),
        thinking=False,
        reasoning_strength="xhigh",
        bindings=make_bindings(),
    )

    template_arguments = FakeLlama.instances[0].kwargs["chat_handler_kwargs"][
        "extra_template_arguments"
    ]
    assert template_arguments == {
        "enable_thinking": False,
        "force_reasoning": False,
    }
    assert result.metrics["configuration"]["reasoning_strength"] == "auto"


def test_auto_reasoning_strength_is_not_forwarded_when_thinking_is_enabled(tmp_path):
    model, _ = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=True,
        reasoning_strength="auto",
        bindings=make_bindings(),
    )

    template_arguments = FakeLlama.instances[0].kwargs["chat_handler_kwargs"][
        "extra_template_arguments"
    ]
    assert template_arguments == {
        "enable_thinking": True,
        "force_reasoning": True,
    }


def test_qwen_reasoning_budget_is_forwarded_to_completion(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "<think>{{ messages }}</think>"}

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=True,
        reasoning_budget=256,
        bindings=make_bindings(),
    )

    completion_kwargs = FakeLlama.instances[0].completion_kwargs
    assert completion_kwargs["reasoning_budget"] == 256
    assert completion_kwargs["reasoning_start"] == "<think>"
    assert completion_kwargs["reasoning_end"] == "</think>"
    assert completion_kwargs["reasoning_start_in_prompt"] is True
    assert result.metrics["configuration"]["reasoning_budget"] == 256
    assert result.metrics["configuration"]["reasoning_budget_applied"] is True
    assert result.metrics["configuration"]["reasoning_budget_format"] == "think_tags"


def test_gemma_reasoning_budget_uses_channel_markers(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {
        "tokenizer.chat_template": "<|channel>analysis<channel|>{{ messages }}"
    }

    run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=True,
        reasoning_budget=128,
        bindings=make_bindings(),
    )

    completion_kwargs = FakeLlama.instances[0].completion_kwargs
    assert completion_kwargs["reasoning_budget"] == 128
    assert completion_kwargs["reasoning_start"] == "<|channel>"
    assert completion_kwargs["reasoning_end"] == "<channel|>"
    assert completion_kwargs["reasoning_start_in_prompt"] is False


def test_zero_or_disabled_reasoning_budget_is_not_forwarded(tmp_path):
    model, _ = gguf_files(tmp_path)

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=False,
        reasoning_budget=512,
        bindings=make_bindings(),
    )

    assert "reasoning_budget" not in FakeLlama.instances[0].completion_kwargs
    assert result.metrics["configuration"]["reasoning_budget"] == 0
    assert result.metrics["configuration"]["reasoning_budget_applied"] is False


def test_positive_reasoning_budget_rejects_unknown_template_and_unloads(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ messages }}"}

    with pytest.raises(InputNormalizationError, match="supported reasoning format"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_media(),
            thinking=True,
            reasoning_budget=256,
            bindings=make_bindings(),
        )

    assert FakeLlama.instances[0].closed is True


def test_disabled_overrides_do_not_pass_integer_values(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler="generic",
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        n_ubatch=2048,
        image_min_tokens=2048,
        image_max_tokens=2048,
        bindings=make_bindings(generic=FakeHandler),
    )

    model_kwargs = FakeLlama.instances[0].kwargs
    handler_kwargs = FakeHandler.instances[0].kwargs
    assert "n_ubatch" not in model_kwargs
    assert model_kwargs["chat_handler"] is FakeHandler.instances[0]
    assert "image_min_tokens" not in handler_kwargs
    assert "image_max_tokens" not in handler_kwargs


def test_image_token_override_rejects_unsafe_physical_batch(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="effective n_ubatch"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            handler="gemma4",
            system="",
            prompt="describe",
            media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
            n_batch=1120,
            override_image_max_tokens=True,
            image_max_tokens=1120,
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


@pytest.mark.parametrize("handler", ["auto", "qwen3_vl"])
def test_non_gemma_image_max_tokens_can_exceed_physical_batch(tmp_path, handler):
    model, mmproj = gguf_files(tmp_path)

    run_chat(
        model_path=str(model),
        mmproj_path=str(mmproj),
        handler=handler,
        system="",
        prompt="describe",
        media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
        n_batch=2048,
        override_n_ubatch=True,
        n_ubatch=1024,
        override_image_max_tokens=True,
        image_max_tokens=2048,
        bindings=make_bindings(qwen3_vl=FakeHandler),
    )

    assert FakeLlama.instances[0].kwargs["n_ubatch"] == 1024
    if handler == "qwen3_vl":
        assert FakeHandler.instances[0].kwargs["image_max_tokens"] == 2048


def test_image_min_token_override_rejects_unsafe_physical_batch(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="effective n_ubatch"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            handler="auto",
            system="",
            prompt="ground the objects",
            media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
            n_batch=1024,
            override_image_min_tokens=True,
            image_min_tokens=1024,
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


def test_image_min_tokens_cannot_exceed_explicit_maximum(tmp_path):
    model, mmproj = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="cannot exceed image_max_tokens"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            handler="auto",
            system="",
            prompt="ground the objects",
            media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
            n_batch=2048,
            override_n_ubatch=True,
            n_ubatch=2048,
            override_image_min_tokens=True,
            image_min_tokens=1024,
            override_image_max_tokens=True,
            image_max_tokens=768,
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


def test_generation_failure_still_closes_model(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.generation_error = RuntimeError("CUDA failure")

    with pytest.raises(BackendError, match="CUDA failure"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="hello",
            media=normalize_images(None),
            bindings=make_bindings(),
        )

    assert FakeLlama.instances[0].closed is True


def test_llm_close_failure_still_closes_caller_owned_handler(tmp_path):
    model, mmproj = gguf_files(tmp_path)
    FakeLlama.close_error = RuntimeError("target close failed")

    with pytest.raises(BackendError, match="could not be fully unloaded"):
        run_chat(
            model_path=str(model),
            mmproj_path=str(mmproj),
            handler="gemma4",
            system="",
            prompt="describe",
            media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
            bindings=make_bindings(gemma4=FakeHandler),
        )

    assert FakeHandler.instances[0].closed is True


def test_images_require_mmproj_before_native_import(tmp_path):
    model, _ = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="mmproj_path is required"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="describe",
            media=normalize_images(solid_image(1, 1, 1, 3, 0.5)),
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


def test_audio_requires_mmproj_before_native_import(tmp_path):
    model, _ = gguf_files(tmp_path)

    with pytest.raises(InputNormalizationError, match="mmproj_path is required"):
        run_chat(
            model_path=str(model),
            system="",
            prompt="transcribe",
            media=normalize_audio(
                {"waveform": silent_audio(1, 1, 80), "sample_rate": 16_000}
            ),
            bindings=make_bindings(),
        )

    assert FakeLlama.instances == []


@pytest.mark.parametrize(
    "content",
    [
        "<think>inspect details</think>final answer",
        "inspect details</think>final answer",
    ],
)
def test_reasoning_tags_are_split_from_response(tmp_path, content):
    model, _ = gguf_files(tmp_path)
    FakeLlama.response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
        "usage": {},
    }

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_images(None),
        bindings=make_bindings(),
    )

    assert result.thinking == "inspect details"
    assert result.response == "final answer"


@pytest.mark.parametrize(
    ("content", "expected_thinking", "expected_response"),
    [
        (
            "<|channel>thought\ninspect details\n<channel|>final answer",
            "inspect details",
            "final answer",
        ),
        ("<|channel>thought\npartial reasoning", "partial reasoning", ""),
    ],
)
def test_gemma4_thought_channel_is_split(
    tmp_path, content, expected_thinking, expected_response
):
    model, _ = gguf_files(tmp_path)
    FakeLlama.response = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {},
    }

    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_images(None),
        bindings=make_bindings(),
    )

    assert result.thinking == expected_thinking
    assert result.response == expected_response


def test_model_path_must_be_an_existing_gguf(tmp_path):
    invalid = tmp_path / "model.bin"
    invalid.write_bytes(b"model")

    with pytest.raises(InputNormalizationError, match="must be a GGUF"):
        run_chat(
            model_path=str(invalid),
            system="",
            prompt="hello",
            media=normalize_images(None),
            bindings=make_bindings(),
        )


def test_custom_chat_template_overrides_gguf_metadata_template(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ gguf_default }}"}
    configured_formatters = []

    def to_handler(formatter):
        configured_formatters.append(formatter)
        return formatter

    custom_jinja = "{% for m in messages %}{{ m.content }}{% endfor %}"
    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        custom_chat_template=custom_jinja,
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=to_handler,
        ),
    )

    assert len(FakeJinjaFormatter.instances) == 1
    formatter = FakeJinjaFormatter.instances[0]
    assert formatter.kwargs["template"] == custom_jinja
    assert result.metrics["configuration"]["custom_chat_template"] is True


def test_custom_chat_template_with_thinking_controls(tmp_path):
    model, _ = gguf_files(tmp_path)
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ gguf_default }}"}
    configured_formatters = []

    def to_handler(formatter):
        configured_formatters.append(formatter)
        return formatter

    custom_jinja = "{% if enable_thinking %}think{% endif %}{{ messages }}"
    result = run_chat(
        model_path=str(model),
        system="",
        prompt="hello",
        media=normalize_media(),
        thinking=True,
        reasoning_strength="high",
        custom_chat_template=custom_jinja,
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=to_handler,
        ),
    )

    assert len(FakeJinjaFormatter.instances) == 1
    formatter = FakeJinjaFormatter.instances[0]
    assert formatter.kwargs["template"] == custom_jinja
    configured_formatters[0](messages=[])
    assert formatter.call_kwargs["enable_thinking"] is True
    assert formatter.call_kwargs["force_reasoning"] is True
    assert formatter.call_kwargs["reasoning_strength"] == "high"
    assert result.metrics["configuration"]["custom_chat_template"] is True


def test_retained_session_uses_custom_chat_template(tmp_path):
    model, _ = gguf_files(tmp_path)
    custom_template = "CUSTOM_TEMPLATE_JINJA"
    FakeLlama.metadata = {"tokenizer.chat_template": "{{ gguf_default }}"}
    session = LlamaCppSession(
        model_path=str(model),
        handler="auto",
        custom_chat_template=custom_template,
        bindings=make_bindings(
            jinja_formatter_class=FakeJinjaFormatter,
            chat_formatter_to_handler=lambda formatter: formatter,
        ),
    )

    try:
        result = session.generate(
            system="",
            prompt="test prompt",
            media=normalize_media(),
            max_tokens=8,
            seed=-1,
            stop="",
        )
    finally:
        session.close()

    assert len(FakeJinjaFormatter.instances) == 1
    assert FakeJinjaFormatter.instances[0].kwargs["template"] == custom_template
    assert result.metrics["configuration"]["custom_chat_template"] is True
