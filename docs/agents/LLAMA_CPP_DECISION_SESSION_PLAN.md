# llama.cpp Session 결정 노드 구현 계획

## 목표와 범위

`makoto-decision`으로 지정된 답변 중 하나를 선택하고 각 답변의 점수 기반 확률을 출력한다. 기존 `[llama.cpp] Generate` 및 저장된 워크플로의 입력과 출력은 유지한다. 새 결정 노드는 현재 Session 소켓을 재사용하며, 질문은 `[llama.cpp] Build Question (Prefill)`에서 전용 데이터 소켓으로 공급한다.

**범위 변경:** `[llama.cpp] Create Question`은 Nodes 2.0에서 제대로 작동하지 않아 제거했다.

구현 담당 서브 에이전트는 저장소 `AGENTS.md`에 따라 **GPT-6 Luna / MAX**를 사용한다. 기존 미커밋 변경을 먼저 확인하고 보존한다. Python 실행에는 `uv`를 사용한다. 이 계획은 구현 지시서이며 현재 검증 실행을 승인하지 않는다.

## 확정 계약

| 노드 표시 이름 | 핵심 입력 | 출력 |
| --- | --- | --- |
| `[llama.cpp] Build Question (Prefill)` | `question`(STRING 연결), `answer`(ComfyUI STRING 목록 연결) | 동일한 `question` 소켓 |
| `[llama.cpp] Prefill Decide` | `session`, `system`, `context`, `question`, optional `images`/`audio`/`video`, `video_with_audio`, `seed`, `session_unload` | `selected`, `probabilities_json`, `metrics_json`, `media_diagnostics`, `session` |

- 전용 소켓의 값은 `{"question": str, "answer": list[str]}`이다. 생성 노드는 이 값을 검증하고 `answer`의 순서와 표시 문자열을 보존한다.
- `[llama.cpp] Build Question (Prefill)`의 `answer`는 ComfyUI의 **STRING 타입 목록 출력**을 받는다. Python 리스트 한 개를 일반 STRING 출력에 실어 보내는 것과는 다른 계약이다. `is_input_list=True` 사용 시 `question`도 목록으로 전달되므로 단일 값인지 검사해 꺼낸다. 여러 `question` 값이나 중첩된 `answer` 목록을 조용히 합치지 않는다.
- 공통 검증은 빈 질문, 2개 미만 또는 26개 초과 답변, 빈 답변, 중복 답변을 거부한다. 사용자 입력 파라미터와 반환 키는 `snake_case`로 쓴다.
- 결정 노드는 `Choices.letters(*answer)`와 `Decision(question=..., context=..., choices=...)`와 같은 후보 순서를 사용하고 `selected`에는 A/B 같은 후보 토큰이 아닌 원래 답변 문자열을 반환한다. `probabilities_json`은 입력 순서의 `{답변: 확률}` 객체이다. Native의 값은 후보 raw logits의 softmax이고 Runtime의 값은 grammar 제약 completion의 후보 확률을 재정규화한 것이다. 어느 쪽도 보정된 정확도 확률이라고 표시하지 않는다.
- 현재 설치된 `makoto-decision` 0.1.2의 `LlamaCppEvaluator`는 별도 system role 인자를 받지 않는다. `system`은 명시적으로 구분해 `context` 텍스트 앞에 배치하며, 이를 실제 system role이라고 문서화하지 않는다. `question`과 answer 선택지는 텍스트이고, 선택적 미디어는 context prefill에만 들어간다.
- `[llama.cpp] Create Native Session`은 MTMD content와 기존 `LlamaCppEvaluator` prefill logits를 사용한다. Runtime 및 Connect Session은 context의 MTMD marker와 payload를 `/completion`에 전달한다. 서버 후보 확률은 raw logits가 아니며 후보 집합에 대해 재정규화한다. Connect Session은 필요한 `/apply-template`, `/tokenize`, `/completion` endpoint를 제공하는 서버에서 사용할 수 있다.

## 구현 순서

1. **현재 경로와 직렬화 확인.** `backend/nodes/llama_cpp_session.py`, `backend/backends/llama_cpp.py`의 Session 생성·재사용·잠금·정리 경로와 `backend/extension.py` 등록을 확인한다. 현재 설치된 `makoto-decision`/JamePeng fork의 `create_chat_prefill()` 계약을 확인한다. 연결된 STRING 목록 입력이 prompt JSON에 올바르게 전달되어야 한다.
2. **질문 데이터 계약.** `backend/nodes/`에 `[llama.cpp] Build Question (Prefill)`과 이름이 충돌하지 않는 전용 `io.Custom` 타입을 구현한다. 질문 값 생성·검증을 한 함수로 공유하고, ComfyUI 목록 처리와 단일 질문 검사를 수행한다. 노드를 `backend/nodes/__init__.py`와 `backend/extension.py`에 등록한다.
3. **프런트엔드 확장.** 별도 프런트엔드 확장은 필요하지 않다. 입력 노드 스키마와 ComfyUI STRING 목록 연결을 사용한다.
4. **결정 실행.** Native Session은 기존 모델 인스턴스를 재사용해 `LlamaCppEvaluator.evaluate()`를 호출한다. 최초 작업이 결정일 때도 모델 준비가 가능해야 한다. Runtime 및 Connect Session은 해당 서버에서 `/apply-template`로 프로필 템플릿을 적용하고, 후보 토큰을 확인한 뒤 grammar 제약된 1-token `/completion`을 요청한다. `top_probs`를 의미 선택지에 매핑하고 재정규화한다. 두 경로 모두 기존 Session 핸들과 unload/prompt-end 수명주기를 유지한다. `Generate`의 스키마와 결과는 변경하지 않는다.
5. **등록·문서·체크.** `tests/backend/test_extension_registration.py`의 노드 등록/스키마 기대값과 `docs/LLAMA_CPP.md`의 사용법을 갱신한다. 새 검사에는 목록 입력, 값 오류, 선택된 원래 답변과 확률 순서, Native/Runtime/Connect Session 재사용 및 종료, 서버 응답 파싱과 비호환 endpoint 오류를 포함한다.

## 중단 조건과 검증 경계

- Native Session에서 JamePeng fork의 `create_chat_prefill()` 또는 선택한 핸들러의 `prefill`이 없으면 명확히 실패한다. 일반 채팅 생성과 `logprobs`로 조용히 우회하지 않는다. Runtime Session에서 `/tokenize`, grammar 제약 `/completion`, `post_sampling_probs`를 지원하지 않으면 명확히 실패한다.
- 후보 기호 A–Z가 각각 정확히 한 토큰으로 인코딩되지 않거나 서로 같은 토큰 ID로 매핑되면 실행을 중단하고 오류를 낸다. 입력 순서와 선택지 확률의 대응을 추측하지 않는다.
- ComfyUI API가 연결된 STRING 목록을 올바르게 전달하지 않으면 구현을 멈추고 원인을 보고한다.
- 사용자가 검증을 요청하기 전에는 명령을 실행하지 않는다. 요청받은 테스트는 저장소 규칙대로 `bun run test:agent`를 사용한다. 전체 검증은 마지막에 부모 에이전트 또는 사용자가 한 번 수행한다. 실제 ComfyUI 브라우저, GGUF, GPU, 다중 실행, VRAM 회수는 `docs/TESTING.md`에 따른 별도 수동 확인으로 보고한다.

## 구현 상태

- 구현 완료: 공통 질문 검증, 두 질문 입력 경로, 저장 가능한 2–26 답변 위젯과 `Update inputs`, text/MTMD context 결정, seed 및 선택적 session unload, metrics와 media diagnostics, 노드 등록, `docs/LLAMA_CPP.md` 사용법.
- 자동 검증은 저장소 규칙에 따라 실행하지 않았다. ComfyUI 브라우저, 실제 STRING 목록 소켓, JamePeng GGUF prefill, GPU, prompt-end 정리는 수동 검증 전이다.

참고: [ComfyUI 목록 계약](https://docs.comfy.org/custom-nodes/backend/lists), [ComfyUI 프런트엔드 훅](https://docs.comfy.org/custom-nodes/js/javascript_hooks), [JamePeng prefill API PR](https://github.com/JamePeng/llama-cpp-python/pull/183).
