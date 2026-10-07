# Internal llama.cpp Daemon 재시작 버튼 구현 계획

## 목표와 현재 경로

ComfyUI 설정의 **llama.cpp Daemon** 영역에 `Restart internal daemon` 버튼을 추가한다. 버튼은 이 ComfyUI 프로세스가 소유한 supervisor와 `llama server`만 재시작한다. 연결된 외부 llama.cpp 서버와 `Llama.cpp Connect Session`에는 영향을 주지 않는다.

현재 `backend/llama_cpp/llama_cpp_runtime.py`가 `_lock` 아래에서 설정, supervisor 프로세스, 상태를 소유한다. 설정 변경 시 `_stop_locked()` 후 `_start_locked()`를 호출하고, `/ollama_image_list/llama_cpp/runtime`의 GET/POST가 상태 조회와 설정 변경을 처리한다. `frontend/src/llama-cpp-runtime-settings.ts`는 같은 경로로 요청하고 상태/오류를 설정 화면에 반영한다. `backend/llama_cpp/llama_cpp_supervisor.py`가 소유 프로세스 종료를 맡는다. 이 경로를 그대로 재사용한다.

## 동작 계약

1. `auto_start=true`일 때 버튼은 현재 저장된 `ctx_size`, `port`, `model_dir`로 재시작한다. 설정 파일이나 `auto_start` 값은 변경하지 않는다. `running`뿐 아니라 `failed` 또는 `stopped` 상태에서도 재시도를 허용한다.
2. `auto_start=false`이면 재시작하지 않고 버튼에 켜기 안내를 표시한다. 버튼이 비활성 상태라도 백엔드에서 이를 다시 검사한다. 외부 서버나 포트 점유 프로세스를 종료 대상으로 삼지 않는다.
3. 기존 supervisor를 `_stop_locked()`로 완전히 종료한 뒤에만 `_start_locked()`를 호출한다. 종료가 실패하면 시작하지 않고 기존 오류 상태를 반환한다. 시작 성공은 기존 `/health` 검사 후 `running`이 된 경우로 한정한다. 중복 클릭은 한 번의 요청으로 취급한다.
4. 재시작 중 기존 요청/세션은 끊길 수 있다. 버튼 설명에 이 점을 알리고 자동 확인 대화상자는 추가하지 않는다. 성공/실패 결과는 현재 상태 필드와 toast로 보여준다.

## 작업 순서와 담당 위치

1. **백엔드 — `backend/llama_cpp/llama_cpp_runtime.py`:** `restart_runtime()`을 추가해 기존 `_lock`, `_observe_process_exit_locked()`, `_stop_locked()`, `_start_locked()`, `_runtime_status_locked()`를 사용한다. 설정 저장 함수는 호출하지 않는다. 로컬 요청만 허용하는 `POST /ollama_image_list/llama_cpp/runtime/restart`를 등록하고, `auto_start=false`에는 명확한 409 응답을 준다. 실행은 기존 엔드포인트처럼 `asyncio.to_thread()`로 옮긴다. 재시작 결과는 기존 상태 JSON 형식으로 반환한다.
2. **프런트엔드 — `frontend/src/llama-cpp-runtime-settings.ts`:** 현재 설치된 `@comfyorg/comfyui-frontend-types`의 설정 `type`이 custom renderer를 받으므로, 같은 **llama.cpp Daemon** 영역에 네이티브 `<button>`을 반환하는 설정 항목을 추가한다. 요청은 기존 `api.fetchApi`와 `settingUpdateQueue`를 사용해 설정 변경과 순서를 맞춘다. 진행 중 버튼을 비활성화하고 결과를 `updatePathStatus()`와 `showStartWarning()`에 전달한다. `auto_start`가 꺼졌을 때는 버튼 클릭을 막고 켜기 안내를 준다. 별도의 영구 설정 값은 만들지 않는다.
3. **문구/문서 — `frontend/src/llama-cpp-runtime-messages.ts`, `docs/LLAMA_CPP.md`:** 한국어·영어 버튼 결과/안내 문구를 추가하고, 재시작 범위와 실행 중 요청 중단 가능성을 설명한다.
4. **작은 회귀 체크 — 기존 테스트 위치:** 소유 프로세스 재시작, 실패 상태 재시도, `auto_start=false` 거부, 종료 실패 시 신규 시작 금지, 설정 미변경, 로컬 요청 제한을 확인한다. 프런트엔드는 중복 클릭과 상태/오류 반영만 확인한다. 필요한 체크만 추가한다.

## 완료 기준

- 버튼을 누른 뒤 이전 supervisor가 종료되고 새 supervisor의 `/health`가 성공하면 상태가 `running`으로 갱신된다.
- 시작 실패 시 `failed`와 원인이 보이고, 다시 누르면 재시도할 수 있다.
- 종료 실패 또는 `auto_start=false`에서는 새 프로세스를 시작하지 않는다.
- 외부 llama.cpp 서버, 저장된 daemon 설정, 기존 노드 ID와 소켓 계약은 변하지 않는다.
- 구현 시 저장소 규칙에 따라 요청받기 전에는 검증을 실행하지 않는다. 검증 요청을 받으면 `bun run test:agent`를 사용하고, 실제 ComfyUI/브라우저에서의 동작은 `docs/TESTING.md`에 따라 별도로 확인한다.
