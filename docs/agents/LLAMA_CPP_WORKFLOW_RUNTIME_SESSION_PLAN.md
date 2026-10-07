# Workflow 수준 llama.cpp Runtime Session 구현 계획

## 목표

`[llama.cpp] Create Runtime Session`을 추가한다. 워크플로가 이 노드를 실행할 때만 로컬 `llama server`를 시작하고, 기존 `[llama.cpp] Generate`와 `[llama.cpp] Unload Session`을 사용해 생성 후 종료한다. `llama_cpp_python` 설치나 Internal runtime의 상시 활성화는 필요하지 않다.

## 노드와 소유권 계약

- Create Runtime Session은 `LlamaCppMtmd_CreateRuntimeSession` ID를 사용하고 기존 `OLLAMA_IMAGE_LIST_LLAMA_CPP_SESSION` 소켓을 출력한다. 새 ID를 적용하면서 이전 Create Runtime Session ID와의 워크플로 호환 등록은 두지 않는다. 모든 새 파라미터 이름은 `snake_case`로 둔다.
- 실행 파일은 기존 `_resolve_llama_executable()` 경로를 사용한다. `PATH`의 `llama`가 우선이고, 없으면 완료된 Internal runtime 설치본을 사용한다. 둘 다 없으면 모델을 로드하거나 세션을 출력하기 전에 명확한 오류를 낸다. Internal runtime의 `auto_start`, 설정 포트, 실행 중 프로세스에는 손대지 않는다.
- 각 Create Runtime Session 실행은 자신만의 supervisor와 `127.0.0.1` 로컬 포트를 소유한다. 포트는 OS가 할당한 빈 포트를 선택하고 충돌로 시작하지 못하면 오류를 반환하며, 다른 프로세스를 종료하지 않는다. 기존 supervisor의 lifetime pipe, 자식 환경 구성, 종료 대기/강제 종료 방식을 재사용하되 전역 daemon의 `_process`와 `_lock` 상태에는 등록하지 않는다.
- 지정된 단일 GGUF와 선택적 mmproj를 서버 실행 명령에 전달한다. 서버 프로세스가 살아 있고 `/health`가 `ok`일 때만 세션을 출력한다. 시작 실패나 준비 시간 초과 시 자신이 시작한 프로세스를 정리한다. 같은 세션의 Generate 호출은 모델이 로드된 서버를 재사용한다.
- 새 세션 핸들은 기존 서버 생성 어댑터의 요청·응답 처리를 재사용하되 프로세스 소유권을 추가한다. `Unload Session`의 `close()`는 이 핸들의 프로세스를 종료한다. 외부 `Connect Session`의 기존 `/models/unload` 동작은 유지한다. 닫기는 여러 번 호출돼도 안전해야 한다.
- 기존 prompt-end 세션 추적에 새 핸들을 등록한다. 명시적 Unload에 도달하지 못한 실패·중단 경로에서도 종료하고, ComfyUI 프로세스가 비정상 종료하면 lifetime pipe를 감시하는 supervisor가 자식을 종료한다. `Unload Session`의 `timing` 의존성은 루프의 마지막 결과에 연결하도록 문서화한다.

## 입력 및 옵션 변환

새 노드의 입력은 기존 `[llama.cpp] Create Native Session`과 같이 `model_path`, `mmproj_path`, `model_profile`, 선택적 `custom_chat_template`, `hardware_profile`, `reasoning`, `speculative`, `n_ctx`, `image_min_tokens`, `image_max_tokens`, `verbose`로 구성한다. 기존 프로필 정규화와 GGUF 경로 확인은 재사용한다. Python 전용 `require_native_speculative()`와 `LlamaCppSession` 생성은 호출하지 않는다.

`hardware_profile`이 연결되지 않으면 `Automatic Offload` 값을 사용한다. `reasoning`과 `speculative`이 연결되지 않으면 각각 기본 추론 설정과 Off를 사용한다.

| 입력 | 서버 적용 규칙 |
| --- | --- |
| `model_path`, `mmproj_path`, `n_ctx` | 확인된 절대 GGUF 경로를 `--model`, 선택적 `--mmproj`로 전달하고 `n_ctx`를 `--ctx-size`로 전달한다. |
| `image_min_tokens`, `image_max_tokens` | 각각 `0`이면 해당 옵션을 생략한다. 양수면 `--image-min-tokens`, `--image-max-tokens`로 전달한다. 두 값이 모두 양수일 때 최소값이 최대값을 넘으면 오류를 낸다. |
| `model_profile.handler` | 입력과 기존 프로필 구조는 받되 서버에서는 무시한다. |
| `model_profile` 샘플링 값 | `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`를 서버의 대응 설정으로 전달한다. 한 세션에 고정된 프로필 값이며 Generate별 `max_tokens`, `seed`, `stop`은 기존 요청 입력을 따른다. |
| 별도 `custom_chat_template` 입력 | 값이 있으면 임시 Jinja 템플릿 파일을 만들어 `--chat-template-file`로 전달하고, 서버 종료 후 파일을 정리한다. 비어 있으면 모델 메타데이터의 템플릿을 사용한다. |
| `model_profile.recommended_reasoning_mode` | `reasoning`이 없거나 `auto`일 때의 기본 모드로 사용한다. 명시적으로 충돌하는 모드는 기존 `[llama.cpp] Create Native Session`과 같이 오류로 처리한다. |
| `hardware_profile` | `n_batch`→`--batch-size`, 양수 `n_ubatch`→`--ubatch-size`, `gpu_layers`→`--gpu-layers`(`cpu`는 `0`), `main_gpu`→`--main-gpu`, 양수 `n_threads`→`--threads`, `flash_attention`→`--flash-attn`(`enabled`/`disabled`는 `on`/`off`)으로 변환한다. `n_ubatch=0`, `n_threads=0`은 옵션을 생략한다. |
| `hardware_profile.use_mmap` | `true`는 `--load-mode mmap`, `false`는 `--load-mode none`으로 변환한다. |
| `reasoning.reasoning_mode`, `reasoning_effort` | 유효 모드를 `--reasoning`에 전달한다. `reasoning_effort=auto`는 effort 옵션을 생략하고 그 외 값은 `--reasoning-effort`에 전달한다. |
| `reasoning.max_reasoning_tokens` | 양수일 때만 `--reasoning-budget`에 전달한다. `0`이면 **옵션 자체를 생략**한다. 서버 기본 동작을 별도의 무제한 값으로 명시하지 않는다. |
| `reasoning.preserve_thinking` | 선택한 실행 파일이 `--reasoning-preserve`/`--no-reasoning-preserve`를 제공하면 값에 맞게 전달한다. 제공하지 않으면 이 값만 무시한다. |
| `speculative` Off | speculative 옵션을 생략한다. |
| `speculative` Native | `draft-mtp`, `draft-dflash`, `draft-dspark`를 `--spec-type`으로, 필요한 draft GGUF를 `--spec-draft-model`로 전달한다. `draft_n_max`, `draft_p_min`, `draft_n_gpu_layers`, `draft_backend_sampling`은 대응 `--spec-draft-*` 옵션으로 변환한다. External MTP는 draft 모델을 요구하고 Internal MTP는 draft 모델을 전달하지 않는다. |
| `speculative` N-gram | `k`/`k4v`를 각각 `ngram-map-k`/`ngram-map-k4v`로 변환하고 `ngram_size`, `num_pred_tokens`, `ngram_min_hits`를 해당 N-gram의 `size-n`, `size-m`, `min-hits` 옵션에 전달한다. `ngram_max_entries_per_key`는 입력은 받되 무시한다. |
| `verbose` | 서버의 verbose 로깅 옵션에 대응시킨다. |

`reasoning_mode=off`일 때는 effort와 budget을 전달하지 않는다. 프로필별 샘플링 및 reasoning 값은 워크플로 Session에 고정한다. 현재 Generate 노드가 받는 요청별 입력과 서버 응답 소켓은 유지한다. `handler`와 `ngram_max_entries_per_key`처럼 무시하기로 한 필드 외에는 선택한 speculative 모드를 조용히 해제하거나 다른 방식으로 바꾸지 않는다.

## 작업 순서와 담당 위치

1. **실행 경로 — `backend/llama_cpp/llama_cpp_runtime.py`, `backend/llama_cpp/llama_cpp_supervisor.py`:** 실행 파일 탐색, 자식 환경, supervisor 시작과 준비/종료 로직 중 재사용할 부분을 작은 함수로 분리한다. 상시 daemon 설정과 전역 상태의 소유권은 유지한다. 임시 서버는 단일 모델 명령과 독립 포트를 사용한다.
2. **입력 변환 — `backend/nodes/llama_cpp_compact.py` 및 새 서버 Session 코드:** 기존 프로필 정규화와 GGUF 경로 확인을 사용해 위 표의 서버 명령을 만든다. 현재 `build_compact_session_kwargs()`에는 Python native speculative 의존성 확인이 있으므로 새 노드에서 통째로 호출하지 않는다. 명시적으로 무시하는 두 필드만 제외하고 잘못된 경로·상충하는 입력은 시작 전에 오류로 처리한다.
3. **세션 수명 — `backend/backends/llama_cpp_server.py`, `backend/nodes/llama_cpp_session.py`:** 기존 HTTP 생성 어댑터를 쓰는 소유형 핸들과 새 Create 노드를 추가한다. 기존 Generate/Unload의 세션 타입 검사를 새 핸들까지 확장하고 기존 prompt-end 추적을 재사용한다. 외부 Connect의 close 동작은 분리해 보존한다.
4. **등록과 문서 — `backend/extension.py`, `backend/nodes/__init__.py`, `docs/LLAMA_CPP.md`:** 새 노드만 추가 등록하고, PATH/Internal 실행 파일 우선순위, 임시 포트, 프로필 변환과 무시되는 두 필드, Unload/중단 시 종료 순서를 설명한다.
5. **회귀 체크 추가 — 기존 `tests/backend/` 위치:** 실행 파일 부재, PATH 우선, 옵션 변환과 `0` 생략, 두 무시 필드, 템플릿 파일 정리, 시작 실패 정리, 생성 후 명시적 Unload, prompt-end 종료, 외부 Connect 및 상시 daemon 비간섭을 검증하는 최소 체크를 남긴다. 저장소 규칙상 요청받기 전에는 검증을 실행하지 않는다. 요청받으면 `bun run test:agent`를 사용하고, 실제 ComfyUI/모델 실행은 `docs/TESTING.md`에 따라 별도로 확인한다.

## 완료 기준

- 실행 파일이 없으면 새 노드만 명확히 실패하며 서버 프로세스나 세션을 남기지 않는다.
- PATH 또는 Internal 설치본이 있으면 워크플로 실행 중에만 모델이 로드된 로컬 서버가 열리고, 기존 Generate 노드가 이를 사용한다.
- 명시적 Unload, 생성 오류, 프롬프트 종료, ComfyUI 비정상 종료에서 소유 서버가 정리된다. 이미 실행 중인 상시 daemon과 외부 Connect Session은 영향을 받지 않는다.
- 기존 노드 ID·소켓·저장된 워크플로는 그대로 동작한다.

참고: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
