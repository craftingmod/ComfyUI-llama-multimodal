# Testing

Install the locked development dependencies:

```bash
bun install --frozen-lockfile
uv sync --locked --group dev
```

Run the same validation sequence as CI:

```bash
bun run fmt:check
bun run lint
bun run typecheck
bun run test:unit
bun run build
bun run build:custom-node
```

`bun run test:frontend` runs Bun frontend tests and `bun run test:backend` runs the Python suite. Generated frontend files live in `dist/`; edit `frontend/` rather than the bundle.

The automated suite covers:

- image/audio/video normalization, list flattening, limits, PNG encoding, and PCM16 WAV encoding;
- Ollama request/response handling through a local mock HTTP server;
- llama.cpp image-only, audio-only, video-only, and mixed-media message construction;
- model-specific and generic thinking arguments, response/thinking extraction, sampling presets, and Gemma 4 runtime presets;
- `n_ubatch` and image-token override validation;
- typed MTMD diagnostics, availability flags, evaluated counts, and payload-free receipts;
- unconditional native cleanup on success and failure through test doubles;
- Compact Media Sequential Generate single-load execution, native-fork `reset()` and non-native memory-clear paths, independent per-item media messages, explicit list outputs, and final unload through test doubles;
- Session Prompt Sequential flat-string validation, one shared media bundle, five parallel outputs, media-first message order, native reset bypass, explicit server `cache_prompt` true/false, legacy request omission, and sequence unload through test doubles;
- Sequential Decide list validation and singleton/exact-count broadcasting, atomic media pairing, context/question index alignment without Cartesian expansion, shared-bundle prompt decisions, four parallel result lists with a scalar session, one final unload, native/server probability semantics, and cache-option forwarding through test doubles;
- Server Decide full-vocabulary pre-sampling requests, cached vocabulary metadata, exclusion of unrelated special-token candidates, complete-choice normalization, and rejection of missing choices or all-zero choice mass through test doubles (new cases added, not executed);
- System One choice/score/noul request mapping, probability ordering and validation, image data URLs, Answer extraction, and 404/501 errors through test doubles;
- native speculative early dependency failure, draft ordering, official `SpecConfig` parameter forwarding, stats capture, target-only isolation, validation, and initialization-failure cleanup through test doubles;
- JamePeng public-boundary guards: no broad sequential attribute proxy or llama.cpp private context/tokenizer/handler access, public `detokenize()` special-token handling, and public MTMD capability flags;
- Model/Hardware/Thinking/Native Profile output typing, published Gemma/Muse/Qwen sampling values, presence-penalty forwarding, Qwen 3.5+ reasoning-mode consistency, Custom Model Profile forwarding, zero-sentinel override behavior, Muse DFlash defaults, unified speculative forwarding, and Compact execution through test doubles;
- normal-node N-gram Preset typing, off-path isolation, lazy SpecConfig API failure, parameter forwarding, 0-to-None conversion, text-only validation, mode separation, public stats capture, and cleanup through test doubles;
- native CLIP Generate Text system templates, IMAGE list flattening, model detection, and Gemma 4 named-parameter compatibility;
- V3 schemas, backend-specific node categories, extension registration, and the thin package entrypoint;
- MiniMax system-prompt preset file selection, validated enum-string override, common/reference base concatenation, and release packaging.

The suite does not install or load a real GGUF, start ComfyUI, exercise native `MTMD_VIDEO` decoding, launch a browser, or contact Ollama. Those integrations remain manual because wheel, GPU backend, model, projector, and chat-template compatibility are environment-specific.

### Internal llama.cpp daemon restart

Use an isolated ComfyUI instance with no active generation; restarting can interrupt requests using its internal daemon. In **Settings → Ollama Image List → llama.cpp Daemon**:

1. With runtime activation off, click **Restart internal daemon**. Confirm the enablement guidance appears and the daemon stays stopped.
2. Enable runtime activation, note the daemon settings file contents, then restart. Confirm the button is disabled while pending, the state reaches `running`, and the success toast appears only after `/health` succeeds.
3. Compare the settings file contents to confirm restart did not save or change daemon settings.
4. In a test instance with a failed or stopped daemon, retry and confirm it can reach `running`; make a stop failure and confirm no replacement process starts.
5. Confirm an external llama.cpp server and **Llama.cpp Connect Session** remain untouched.

Before publishing a llama.cpp build, manually verify in the target ComfyUI environment:

1. Model Profile, Hardware Runtime Profile, Thinking / Reasoning Profile, and Native Speculative Profile appear under `llama_cpp / profile`; N-gram Speculative Config, Generate, and Media Sequential Generate appear under `llama_cpp / compact`; Generate, Generate (Media Sequential), and Generate (Prompt Sequential) appear under `llama_cpp / generate`; Build Question (Prefill), Prefill Decide, Prefill Decide (Media Sequential), and Prefill Decide (Prompt Sequential) appear under `llama_cpp / decision / prefill`; Extract Answer appears under `llama_cpp / decision`; Build Question, Build Noul, System One Decide, System One Decide (Media Sequential), and System One Decide (Prompt Sequential) appear under `llama_cpp / decision / system_one`; Diagnostics and Muse Parser appear under `utils`; the four detailed legacy schemas stay hidden outside developer mode; and four deprecated v0.6.1 IDs remain available for saved workflows;
2. GGUF files from the local and any `extra_model_paths.yaml` `LLM` directories appear in both Combos;
3. the selected main model and projector complete the intended IMAGE, AUDIO, and/or VIDEO request;
4. Media Diagnostics reports the requested capability and evaluated item counts;
5. `metrics_json.model_unloaded` is `true` after a stateless Compact completion or explicit session unload; a retained Session Generate leaves it `false` until the session is unloaded;
6. Compact Media Sequential Generate loads one model for IMAGE, AUDIO, and VIDEO lists, creates one singleton media bundle per item in modality-major order, keeps VIDEO-owned audio inside VIDEO, produces flat `response` and typed `response_seq` lists, and unloads once after the sequence. Confirm the renamed Compact and Session Generate nodes keep their existing saved node IDs and output slot order;
7. Session Generate (Media Sequential) sends one independent request per media item and returns five parallel output lists; Session Generate (Prompt Sequential) sends one request per flat string prompt against the same complete normalized media bundle, returns five parallel output lists, and unloads only after the sequence when requested;
8. Prefill Decide (Media Sequential) pairs each IMAGE/AUDIO/VIDEO item with a shared or exact-count context and typed question, keeps VIDEO-owned audio paired, and makes one text-only decision with no media; Prefill Decide (Prompt Sequential) broadcasts singleton context/question lists or pairs equal-length positions against the fixed media bundle, without Cartesian expansion, returns four result lists plus one session, and unloads once at the end when requested;
9. For Prompt Sequential Generate and Prefill Decide, compare `reuse_kv_cache` enabled and disabled with a native session, then with a server-backed session. Confirm server requests carry explicit `cache_prompt` true/false for the new nodes. Native text-only Decide currently passes through a public prefill handler that resets internally, so reuse cannot be guaranteed there; native media/MTMD reuse remains best effort. Treat any cache hit or speedup as best effort, since the model, handler, matching prefix, and runtime build determine reuse. Repeat on each production hybrid/recurrent checkpoint before relying on reuse;
10. after unloading the session, a second diffusion workflow can reclaim the released VRAM.
11. With a decision model that supports `/v1/systemone`, run choice, score, and noul questions through System One Decide, Media Sequential, and Prompt Sequential using a Runtime Session and a Connect Session; confirm returned probability order, the extracted `value` semantics, per-image versus shared-image request behavior, optional image counts marked `unverified_remote`, explicit failure for Native sessions and unsupported-server errors, and session unload.

For Native Speculative Profile, connect one compatible text target/draft pair to Compact Generate. Confirm that `metrics_json.speculative.stats.draft_calls` and `drafted_tokens` are greater than zero, the completion stops normally, and target/draft VRAM is reclaimed after the response. Confirm IMAGE, AUDIO, and VIDEO are rejected because the current stateful native engines are text-only.

For both MTP paths, select `External MTP` or `Internal MTP` and start with `draft_n_max=2` and `draft_p_min=0.0`. For external MTP, use any target GGUF and a draft GGUF in `draft_model`; use `gpu_layers=all`. For internal MTP, leave `draft_model` unselected and use a target GGUF that contains embedded NextN layers. Confirm enabling the normal `verbose` switch also enables MTP diagnostics. Run text-only prompts long enough to open draft cycles. Confirm `implementation` is `draft-mtp`, the requested `mtp_provider` is reported, `draft_calls` and `drafted_tokens` increase, internal MTP reports `n_layer_nextn > 0`, and target/decoder VRAM is reclaimed. Also confirm external without a draft, internal with a selected draft, internal without NextN layers, and any MTP request with media fail explicitly without target-only fallback.

For normal-node n-gram speculative decoding, run the same repetition-heavy text prompt with Preset mode `off` and `ngram`, keeping model, sampling, seed, and `max_tokens` fixed. Confirm non-empty normal completions, no damaged repetition loop, clean unload, public speculative statistics, and compare `generation_seconds` or tokens per second. Confirm IMAGE, AUDIO, and VIDEO inputs are rejected because the official stateful path is text-only. Do not require token-exact equality.

In the frontend, confirm that N-gram detail widgets are disabled at `off` and restored with their prior values at `ngram` on both N-gram configuration nodes. Confirm Model Profile's six sampling widgets are enabled only for `Custom` and retain their values across profile changes. Hardware Runtime Profile is fixed to `Custom`: `n_batch` and `n_ubatch` are regular inputs, while `gpu_layers` through `use_mmap` are Advanced inputs; confirm `n_ubatch=0` reaches the backend as no override. Compact Generate with a disconnected Hardware Runtime input still uses automatic GPU layer fitting; a connected profile overrides it. On Thinking / Reasoning Profile, confirm effort and token-limit widgets are enabled only for `on`; verify disconnected/`auto` leaves template arguments untouched for ordinary profiles, selects `on` for Qwen 3.5+ Thinking and `off` for Qwen 3.5+ Non-thinking, and rejects an explicitly contradictory mode. Verify `off` sends an explicit disable and `max_reasoning_tokens=0` sends no budget. Confirm `presence_penalty` is editable in Custom and reaches the targeted fork as `present_penalty`. Confirm the node is discoverable by both `thinking` and `reasoning` searches. Confirm Compact Generate sends no image-token override when `image_min_tokens=0` and `image_max_tokens=0`, enables each override for a positive value, and rejects a minimum greater than an explicit maximum. For Qwen-VL grounding, verify `image_min_tokens=1024` reaches the handler with `n_ctx`, `n_batch`, and effective `n_ubatch` all at least 1024. Confirm the Compact N-gram and Native Speculative nodes can each connect to the same Compact Generate `speculative` input. On Native Speculative Profile, confirm all fields are disabled for `Off`; `draft_model` is enabled only for `External MTP`, `DFlash`, `DSpark`, and `Custom`; `custom_spec_type` and `custom_mtp_provider` are enabled only for `Custom`; and `draft_n_max`/`draft_p_min` remain editable for every non-Off preset. On the normal Generate node, connect and disconnect Sampling and Runtime presets and confirm that only their corresponding overridden widgets are disabled, with values preserved across each round trip.

Before publishing native CLIP support, also verify Qwen3-VL or Qwen3.5 with two IMAGE list items and verify Gemma 4 with both a same-resolution batch and, on a ComfyUI build containing PR #15450, two different resolutions.
