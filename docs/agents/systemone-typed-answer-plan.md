# System One typed questions and answers

## Objective and scope

Add llama.cpp `/v1/systemone` support to existing Runtime and Connect session handles. Build choice/score questions with one builder, noul questions with a separate builder, return one custom Answer payload, and extract selected/value/probabilities/result_json. Reuse stdlib HTTP transport and existing session ownership. No new dependency, native decision-model evaluator, automatic fallback, model download, or server launch during this task.

Existing Question socket ID, builders, Decide IDs and output positions remain compatible. New typed questions use the existing Question socket but are accepted only by the new System One node; legacy Decide continues accepting only legacy choice payloads. Existing native/generic LLM token-scoring behavior is unchanged.

## Contracts

- Add `[llama.cpp] Build Question`: `question` STRING, `answer` flat STRING list, `type` COMBO choice|score (default choice). Use `is_input_list=True`, unwrap singleton question/type; preserve answer order. Typed payload `{type, question, answer}`. choice: 2-26 unique nonempty answers (current compatible cap); score: 2-10 unique nonempty ordered levels, lowest first. Existing `{question, answer}` payload is interpreted as choice by System One.
- Add `[llama.cpp] Build Noul`: question STRING only, nonempty; payload `{type: "noul", question}`.
- Add `[llama.cpp] System One Decide` under `llama_cpp/decision/system_one`: session, system/context text, Question socket, optional images, session_unload. Accept Runtime/Connect (`LlamaCppServerSession`, including owned subclass), explicitly reject Native. Outputs: Answer custom socket, metrics_json, media_diagnostics, session. No seed/sampling/reasoning/KV controls unsupported by the API.
- Add `[llama.cpp] System One Decide (Media Sequential)` and `[llama.cpp] System One Decide (Prompt Sequential)` under the same category. Media Sequential makes one request per image with shared or matched context/question values; Prompt Sequential pairs context/question values and reuses the complete image bundle. Both return parallel answer/metrics/diagnostics lists and one session.
- HTTP request uses model=self.model, state=existing `_decision_context` text, questions={"question": {type, instructions, criteria}}. choice criteria map answers to null, score criteria preserve the level list, noul omits criteria. Optional image data URLs reuse existing encoding. No audio/video sockets. No implicit transformation of video into images.
- Answer socket ID `OLLAMA_IMAGE_LIST_LLAMA_CPP_ANSWER`. A plain dict with `type`, `selected`, `value`, ordered `probabilities`, and `result` (the complete per-question API answer, retaining confidence/legend/extra fields). choice selected=returned option and value=its probability; score selected="", value=returned expected index; noul selected="", value=P(true), probabilities=[1-P(true), P(true)]. Choice/score probabilities follow input answer order, never response object iteration order. No response renormalization or rounding.
- Add `[llama.cpp] Extract Answer`: Answer input; selected STRING, value FLOAT, probabilities FLOAT `is_output_list=True`, result_json STRING. Extract exactly one Answer per execution; reject malformed payloads. JSON is the retained complete per-question answer, not metrics or the request envelope. Outputs are copies where needed and do not mutate upstream payloads.
- Validate API response object, answers.question, matching type, valid selected membership, exact probability keys/level keys, finite numeric non-boolean values, bounds [0,1], probability sum within a small numerical tolerance, score range [0,n-1]. Do not silently accept missing keys, mismatched types, or unsupported model/server errors.
- Metrics retain duration, model, usage and operation. Image diagnostics retain requested counts and `unverified_remote`; HTTP success does not prove MTMD evaluation. session_unload uses existing close semantics, including borrowed vs owned process ownership. Closed sessions fail explicitly.

## Sequence and owners

1. Parent: inspect source and official docs; save this plan before implementation.
2. Luna Max implementation agent: add backend request/response path, builders, Answer/extractor, registration, focused contract tests, and concise usage/manual verification documentation. Prefer existing decision module and server module; avoid speculative infrastructure. Update this plan's implementation status. Do not execute tests while another agent is editing.
3. Luna Max independent review/test agent: read plan and inspect contracts; after implementation completes, review all paths and tests, fix concrete issues, and run `bun run test:agent` once at the end. Retry only after an actual fix or a necessary environment workaround. Record exact command/results; never claim fake transport proof is live inference.
4. Parent: inspect final diff/results, resolve substantive findings through agents, open plan and report implementation plus automated/live validation separately.

## Verification and stop rules

- Check builder list handling, defaults/legacy choice compatibility, score limits/order, noul omission, all three extraction results and Float list schema.
- Fake HTTP tests cover model routing, text/image encoding, ordering, malformed/NaN/bool/missing/type-mismatched replies, 404/501 errors without fallback, closed sessions and owned/borrowed close behavior.
- Registration tests cover all new IDs and verify unchanged existing outputs.
- Full test suite: `bun run test:agent` (explicitly authorized by user). Python outside repo scripts uses uv. No full validation command unless needed to address a concrete test failure.
- Real ComfyUI Canvas/list mapping, decision GGUF, GPU, vision evaluation, current installed runtime version remain unverified unless performed separately with available resources. Do not download models or launch services for this task.
- Halt unsupported Native execution with a clear error; never replace old node output sockets in place. Keep changes scoped and preserve any unrelated edits discovered during work.

## Follow-up status (2026-10-08)

- Moved the System One nodes under `llama_cpp/decision/system_one`, renamed the base display label, and added Media Sequential and Prompt Sequential variants. Automated validation was not run for this follow-up.

## Sources

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#typesafe-compatible-api-endpoints
- https://github.com/ggml-org/llama.cpp/pull/29818
- https://docs.comfy.org/custom-nodes/v3_migration

## Status

- Design and implementation complete, including server request/response handling, all four nodes, registration, and usage/manual-testing documentation.
- `bun run test:agent` passed: frontend 23 passed; backend 181 passed, 6 skipped. The first run caught an exact-float expectation in the noul test; after changing it to compare with `1.0 - P(true)`, the final run passed.
- Live ComfyUI, decision-model, GPU, and remote image-evaluation behavior remain unverified.
