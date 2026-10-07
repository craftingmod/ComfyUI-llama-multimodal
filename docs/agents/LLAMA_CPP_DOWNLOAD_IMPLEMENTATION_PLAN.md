# llama.cpp b11146 다운로드 구현 계획

## 목표와 현재 상태

- ComfyUI 프로세스의 `PATH`에 `llama`가 없을 때 설정에 `Download llama.cpp` 버튼을 보여준다. 버튼을 누르면 **ComfyUI의 사용자 디렉터리** 아래 `__llama_cpp/artifacts/`에 이 실행 환경에 맞는 [b11146 릴리스](https://github.com/ggml-org/llama.cpp/releases/tag/b11146)를 설치한다. 사용자 홈 디렉터리(`Path.home()`)를 쓰지 않는다.
- 현재 미커밋 작업에는 `backend/llama_cpp/llama_cpp_runtime.py`의 상태 조회, PATH 탐색, 자동 실행, 로컬 전용 HTTP 경계와 `frontend/src/llama-cpp-runtime-settings.ts`의 설정 UI가 이미 있다. **이 파일을 확장**하고, 다른 미커밋 변경을 지우거나 되돌리지 않는다. 기존 노드 및 `settings.json` 계약은 유지한다.
- 설치 후에도 시스템 PATH는 바꾸지 않는다. 실행 파일 탐색 순서를 `PATH`의 `llama` → 검증된 내부 설치본으로 바꿔 기존 자동 실행 코드를 그대로 사용한다. 설치가 끝났고 자동 실행이 켜져 있으면 이전의 “PATH에 없음” 시도 상태를 해제하고 한 번 시작한다.
- 이 문서는 구현 지시다. 현재 작업에서는 코드 구현과 검증을 하지 않는다.

## 고정된 배포 파일 선택

OS와 CPU 아키텍처는 `sys.platform`, `platform.machine()`으로 판별한다. `AMD64/x86_64`는 `x64`, `aarch64/arm64`는 `arm64`로 정규화한다. 가속 백엔드는 **ComfyUI가 사용하는 Python의** `torch.version.hip` → `torch.version.cuda` → CPU 순으로 판별한다. macOS는 PyTorch가 MPS인지 여부와 관계없이 macOS 파일을 쓴다. 다른 Python, 시스템에 따로 설치된 CUDA/ROCm, 브라우저의 OS를 판별 근거로 쓰지 않는다. CUDA인 경우 b11146의 CUDA **13.4** 파일을 선택한다. 런타임/드라이버의 실제 호환성은 설치 전 확정할 수 없으므로 실행 실패는 명확히 표시한다.

| 환경 | llama 배포 파일 | 추가 파일 |
| --- | --- | --- |
| Windows x64 CPU | `llama-b11146-bin-win-cpu-x64.zip` | 없음 |
| Windows arm64 CPU | `llama-b11146-bin-win-cpu-arm64.zip` | 없음 |
| Windows x64 CUDA | `llama-b11146-bin-win-cuda-13.4-x64.zip` | `cudart-llama-bin-win-cuda-13.4-x64.zip` |
| Windows arm64 CUDA | `llama-b11146-bin-win-cuda-13.4-arm64.zip` | `cudart-llama-bin-win-cuda-13.4-arm64.zip` |
| Windows x64 ROCm | `llama-b11146-bin-win-rocm-10.0-x64.zip` | 없음 |
| macOS x64 | `llama-b11146-bin-macos-x64.tar.gz` | 없음 |
| macOS arm64 | `llama-b11146-bin-macos-arm64.tar.gz` | 없음 |
| Linux x64 CPU | `llama-b11146-bin-ubuntu-x64.tar.gz` | 없음 |
| Linux arm64 CPU | `llama-b11146-bin-ubuntu-arm64.tar.gz` | 없음 |
| Linux x64 CUDA | `llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz` | `cudart-llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz` |
| Linux arm64 CUDA | `llama-b11146-bin-ubuntu-cuda-13.4-arm64.tar.gz` | `cudart-llama-b11146-bin-ubuntu-cuda-13.4-arm64.tar.gz` |
| Linux x64 ROCm | `llama-b11146-bin-ubuntu-rocm-10.0-x64.tar.gz` | 없음 |

릴리스에 없는 조합(예: Windows arm64 ROCm, Linux arm64 ROCm)은 **지원하지 않는다고 표시하고 다운로드하지 않는다**. CPU 파일로 조용히 대체하지 않는다. Linux 파일은 Ubuntu 빌드이므로 다른 배포판에서 실행 호환성을 보장하지 않는다. 파일 이름과 SHA-256은 [b11146 릴리스의 공식 attestation](https://github.com/ggml-org/llama.cpp/attestations/49623059)에서 정확히 복사하여 코드의 고정 매핑에 넣는다. 다운로드 URL은 `https://github.com/ggml-org/llama.cpp/releases/download/b11146/<고정 파일명>` 형태로만 만든다. 사용자가 URL/파일명/태그를 API에 전달하지 못하게 한다. 추가 CUDA 파일의 이름은 Windows와 Linux에서 서로 다르다.

## 구현 순서

1. **저장 위치 및 실행 파일 탐색:** `backend/llama_cpp/llama_cpp_runtime.py`의 기존 `_config_path()`가 사용하는 `folder_paths.get_system_user_directory("llama_cpp")`를 공통 기준으로 삼아 `artifacts/`를 만든다. 공식 [ComfyUI `folder_paths.py`](https://github.com/Comfy-Org/ComfyUI/blob/master/folder_paths.py)의 이 API는 `user/__llama_cpp`를 반환한다. 없으면 기존 설정 코드처럼 명시적으로 실패한다. 설치 대상은 `artifacts/b11146/<os>-<arch>-<backend>/`처럼 플랫폼별 고정 하위 디렉터리다. `PATH` 실행 파일을 우선하고, 없으면 **설치 완료 표식과 실제 파일이 모두 있는** 현재 환경의 내부 실행 파일만 선택한다. 두 경로 모두 기존 버전 표시와 자동 실행에 사용한다. PATH 표시에는 “PATH에 없음”과 “내부 설치본 사용 가능”을 구별할 수 있게 상태 필드를 추가한다.
2. **선택 및 다운로드:** 위 표를 반환하는 작은 순수 함수 하나를 만든다. `torch`는 백엔드에서 지연 import하고, import/런타임 조회 실패를 CPU로 추측하지 말고 오류로 표시한다. 로컬 전용 `POST /ollama_image_list/llama_cpp/runtime/download`는 서버가 선택한 파일만 내려받도록 하고 단일 작업만 시작한다. 기존 `_is_local_request()` 검사를 재사용한다. 중복 클릭은 새 작업을 만들지 않는다. 다운로드가 오래 걸려도 요청이 붙잡히지 않게 작업 스레드에서 표준 라이브러리로 스트리밍하고, 기존 `GET .../runtime` 응답에 `download_state` (`idle/downloading/installing/installed/error`), 대상 설명, 진행 바이트와 오류를 넣어 UI가 조회하게 한다. 첫 URL은 위 고정 GitHub 주소이며 리디렉션은 HTTPS의 `github.com` 및 GitHub 릴리스 자산 호스트(`release-assets.githubusercontent.com`, `objects.githubusercontent.com`)만 허용한다.
3. **검증 및 설치:** 각 파일을 임시 디렉터리에 받아 **압축 해제 전에** 공식 SHA-256과 비교한다. 해시가 다르면 즉시 삭제하고 기존 설치본은 건드리지 않는다. ZIP/TAR 멤버의 절대 경로, `..`, 링크/심볼릭 링크, 대상 밖 경로를 거부한다. 필요한 실행 파일과 라이브러리만 같은 설치 디렉터리에 안전하게 배치한다. 압축 내부 레이아웃을 확인하여 `llama.exe`/`llama`의 실제 위치를 기록하고, Unix 실행 권한을 보장한다. CUDA 추가 파일은 해당 라이브러리가 실행 파일에서 로드되는 위치에 둔다. Linux에서 필요한 경우 **자식 프로세스에만** 라이브러리 경로를 전달한다. 두 파일 설치가 모두 끝나고 `llama --version`이 성공한 뒤 완료 표식을 쓴다. 대상 디렉터리가 없을 때만 준비된 디렉터리를 rename으로 게시한다. 이미 있는 설치본은 삭제하거나 덮어쓰지 않는다. 실패 시 임시 파일만 정리한다. 이미 유효한 내부 설치본이면 중복 다운로드하지 않는다.
4. **설정 버튼:** `frontend/src/llama-cpp-runtime-settings.ts`의 기존 카테고리에 `Download llama.cpp`를 추가한다. 설치된 프런트엔드 타입 `SettingInputType`에는 `button`이 없으므로, `SettingParams.type`이 허용하는 `SettingCustomRenderer`로 실제 `<button type="button">`을 렌더링한다. PATH의 `llama`가 없고 내부 설치본도 없으며 지원 환경일 때만 누를 수 있게 한다. 클릭 시 POST 후 기존 상태 GET을 주기적으로 조회하며 진행률/오류를 표시하고, 중복 클릭을 막는다. 설정을 닫거나 다시 열어도 서버 상태가 기준이다. 완료되면 기존 경로·버전·실행 상태를 새 응답으로 갱신한다. 텍스트는 기존 `llama-cpp-runtime-messages.ts`의 영어/한국어 구조에 추가한다.
5. **최소 확인 코드와 문서:** 선택 매핑, PATH 우선순위, 해시 불일치/경로 탈출 거부, 중복 다운로드를 검증하는 작은 백엔드 테스트와 버튼의 성공/실패 상태를 검증하는 작은 프런트엔드 테스트를 추가한다. 실제 네트워크 대신 고정 가짜 아카이브를 사용한다. `docs/LLAMA_CPP.md`에 설치 위치, 지원 조합, PATH 우선순위, CUDA/ROCm 런타임 요구사항, 오류 확인 위치를 적는다. 별도 패키지나 설치 관리자 프레임워크는 추가하지 않는다.

## 완료 조건과 중단 규칙

- PATH에 `llama`가 있으면 다운로드 동작이 시작되지 않고 기존 실행 파일을 계속 쓴다. PATH에 없고 지원 환경이면 버튼으로 b11146을 설치한 직후 상태 경로/버전이 내부 설치본으로 바뀌며, 기존 자동 실행이 이를 사용할 수 있다.
- 다운로드·검증·압축 해제·버전 확인 중 실패하면 기존 설치와 자동 실행 설정은 보존되고, 브라우저에 원인이 보인다. 지원하지 않는 조합은 요청 전과 백엔드 양쪽에서 거부한다. 시스템 PATH는 수정하지 않는다.
- 릴리스 자산 구조가 예상과 다르거나 CUDA 라이브러리 배치가 확인되지 않으면 추측해서 완료 처리하지 말고, 확인된 구조와 막힌 지점을 보고한다. 필요한 ComfyUI 설정/경로 API 변경은 현재 공식 소스로 다시 확인한다.
- 이 저장소 `AGENTS.md`에 따라 **검증 실행은 별도 요청이 있을 때만** 한다. 요청받은 경우 `bun run test:agent`를 사용하고, 전체 검증은 부모 에이전트가 마지막에 한 번 수행한다. 실제 다운로드와 Windows/macOS/Linux GPU 실행은 각 환경에서 별도로 확인하고, 미실행은 미검증으로 보고한다.
