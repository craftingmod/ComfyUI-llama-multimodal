type RuntimeMessages = {
  checking: string
  statusLoadFailure: string
  startFailureSummary: (projectName: string) => string
  startFailureFallback: string
  saveFailureSummary: (projectName: string) => string
  unknownError: string
  restartButton: string
  restarting: string
  restartTooltip: string
  restartRequiresAutoStart: string
  restartSuccessSummary: (projectName: string) => string
  restartSuccessDetail: string
  restartFailureSummary: (projectName: string) => string
  restartFailureFallback: string
  pathAvailable: string
  internalInstallAvailable: string
  internalUpdateAvailable: string
  pathUnavailable: string
  downloadButton: string
  updateButton: string
  updateTooltip: string
  updateAvailable: string
  downloadStarting: string
  downloading: (target: string, progress: string) => string
  installing: (target: string) => string
  installed: (target: string) => string
  downloadFailed: (error: string) => string
  downloadUnsupported: string
  downloadProgressLabel: string
}

const messages: Record<string, RuntimeMessages> = {
  en: {
    checking: "Checking...",
    statusLoadFailure: "Could not load runtime status",
    startFailureSummary: (projectName) => `${projectName} could not start the llama.cpp server`,
    startFailureFallback: "The llama server process is not running.",
    saveFailureSummary: (projectName) => `${projectName} could not save the llama.cpp setting`,
    unknownError: "Unknown error.",
    restartButton: "Restart internal daemon",
    restarting: "Restarting…",
    restartTooltip:
      "Restarting can interrupt active requests. This restarts only the internal daemon owned by this ComfyUI process; external servers and Connect Session are unaffected.",
    restartRequiresAutoStart:
      "Turn on Internal llama.cpp runtime activation before restarting the daemon.",
    restartSuccessSummary: (projectName) =>
      `${projectName} restarted the internal llama.cpp daemon`,
    restartSuccessDetail: "The daemon passed its health check and is running.",
    restartFailureSummary: (projectName) =>
      `${projectName} could not restart the internal llama.cpp daemon`,
    restartFailureFallback: "The daemon did not pass its health check.",
    pathAvailable: "Available on PATH",
    internalInstallAvailable: "Unavailable (installable)",
    internalUpdateAvailable: "Older internal install (update available)",
    pathUnavailable: "Unavailable on both PATH and internal",
    downloadButton: "Download llama.cpp",
    updateButton: "Update llama.cpp",
    updateTooltip:
      "Install the pinned llama.cpp version and restart the internal daemon if enabled. Active requests may be interrupted.",
    updateAvailable: "A pinned llama.cpp update is available.",
    downloadStarting: "Starting download…",
    downloading: (target, progress) => `Downloading ${target} (${progress})…`,
    installing: (target) => `Installing ${target}…`,
    installed: (target) => `Installed ${target}`,
    downloadFailed: (error) => `Download failed: ${error}`,
    downloadUnsupported: "This platform is not supported.",
    downloadProgressLabel: "llama.cpp download progress",
  },
  ko: {
    checking: "확인 중...",
    statusLoadFailure: "런타임 상태를 불러오지 못했습니다",
    startFailureSummary: (projectName) => `${projectName} llama.cpp 서버를 시작하지 못했습니다`,
    startFailureFallback: "llama server 프로세스가 실행 중이 아닙니다.",
    saveFailureSummary: (projectName) => `${projectName} llama.cpp 설정을 저장하지 못했습니다`,
    unknownError: "알 수 없는 오류입니다.",
    restartButton: "내부 daemon 재시작",
    restarting: "재시작 중…",
    restartTooltip:
      "재시작하면 실행 중인 요청이 중단될 수 있습니다. 이 ComfyUI 프로세스가 소유한 내부 daemon만 재시작하며, 외부 서버와 Connect Session에는 영향이 없습니다.",
    restartRequiresAutoStart:
      "daemon을 재시작하려면 Internal llama.cpp runtime activation을 먼저 켜세요.",
    restartSuccessSummary: (projectName) => `${projectName} 내부 llama.cpp daemon을 재시작했습니다`,
    restartSuccessDetail: "헬스 체크를 통과해 daemon이 실행 중입니다.",
    restartFailureSummary: (projectName) =>
      `${projectName} 내부 llama.cpp daemon을 재시작하지 못했습니다`,
    restartFailureFallback: "daemon이 헬스 체크를 통과하지 못했습니다.",
    pathAvailable: "PATH에서 사용 가능",
    internalInstallAvailable: "내부 설치본을 설치해 사용 가능",
    internalUpdateAvailable: "내부 설치 버전이 오래되었습니다 (업데이트 가능)",
    pathUnavailable: "사용 불가",
    downloadButton: "llama.cpp 다운로드",
    updateButton: "llama.cpp 업데이트",
    updateTooltip:
      "고정된 llama.cpp 버전을 설치하고, 활성화된 경우 내부 daemon을 재시작합니다. 실행 중인 요청이 중단될 수 있습니다.",
    updateAvailable: "llama.cpp 고정 버전 업데이트를 사용할 수 있습니다.",
    downloadStarting: "다운로드 시작 중…",
    downloading: (target, progress) => `${target} 다운로드 중 (${progress})…`,
    installing: (target) => `${target} 설치 중…`,
    installed: (target) => `${target} 설치 완료`,
    downloadFailed: (error) => `다운로드 실패: ${error}`,
    downloadUnsupported: "지원하지 않는 플랫폼입니다.",
    downloadProgressLabel: "llama.cpp 다운로드 진행률",
  },
}

export function getRuntimeMessages(locale: string | undefined): RuntimeMessages {
  const language = locale?.replace("_", "-").split("-")[0]?.toLowerCase()
  return messages[language ?? ""] ?? messages.en
}
