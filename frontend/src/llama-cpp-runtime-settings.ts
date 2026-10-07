import type { ComfyApi, ComfyApp } from "@comfyorg/comfyui-frontend-types"

import { PROJECT_NAME } from "./constants.ts"
import { getRuntimeMessages } from "./llama-cpp-runtime-messages.ts"

const RUNTIME_ROUTE = "/ollama_image_list/llama_cpp/runtime"
const RUNTIME_RESTART_ROUTE = `${RUNTIME_ROUTE}/restart`
const RUNTIME_DOWNLOAD_ROUTE = `${RUNTIME_ROUTE}/download`
const SETTINGS_PROJECT_NAME = "llama-multimodal"
const COMFY_LOCALE_SETTING = "Comfy.Locale"
const AUTO_START_SETTING = "OllamaImageList.LlamaCpp.AutoStart"
const RESTART_SETTING = "OllamaImageList.LlamaCpp.Restart"
const SERVICE_STATE_SETTING = "OllamaImageList.LlamaCpp.ServiceState"
const PATH_STATUS_SETTING = "OllamaImageList.LlamaCpp.PathStatus"
const EXECUTABLE_PATH_SETTING = "OllamaImageList.LlamaCpp.ExecutablePath"
const LLAMA_VERSION_SETTING = "OllamaImageList.LlamaCpp.Version"
const CTX_SIZE_SETTING = "OllamaImageList.LlamaCpp.CtxSize"
const PORT_SETTING = "OllamaImageList.LlamaCpp.Port"
const SERVER_LINK_SETTING = "OllamaImageList.LlamaCpp.ServerLink"
const MODEL_DIR_SETTING = "OllamaImageList.LlamaCpp.ModelDir"
const DEFAULT_CTX_SIZE = 16384
const MIN_CTX_SIZE = 512
const MAX_CTX_SIZE = 1048576
const DEFAULT_PORT = 18582
const MIN_PORT = 1024
const MAX_PORT = 65535
const SERVICE_STATES = ["stopped", "starting", "running", "stopping", "failed"] as const
const DOWNLOAD_STATES = ["idle", "downloading", "installing", "installed", "error"] as const
const DOWNLOAD_POLL_INTERVAL_MS = 1000

type RuntimeSettingsUpdate = {
  auto_start?: boolean
  ctx_size?: number
  port?: number
  model_dir?: string
}

type RuntimeStatus = {
  auto_start: boolean
  state: (typeof SERVICE_STATES)[number]
  ctx_size: number
  port: number
  active_port: number | null
  model_dir: string
  model_dirs: string[]
  default_model_dir: string
  llama_available: boolean
  llama_source: "path" | "internal" | null
  llama_executable: string | null
  llama_version: string | null
  running: boolean
  error: string | null
  download_state: (typeof DOWNLOAD_STATES)[number]
  download_target: string | null
  download_supported: boolean
  download_support_error: string | null
  download_bytes_received: number
  download_bytes_total: number | null
  download_error: string | null
  update_available?: boolean
}

const syncingSettings = new Map<string, number>()
let setupRevision = 0
let autoStartRevision = 0
let ctxSizeRevision = 0
let portRevision = 0
let modelDirRevision = 0
let settingUpdateQueue: Promise<void> = Promise.resolve()
let latestStatus: RuntimeStatus | undefined
let restartInProgress = false
let downloadInProgress = false
let downloadStartPending = false
let downloadRequestError: string | null = null
let downloadPollTimer: ReturnType<typeof setTimeout> | undefined
let downloadView:
  | {
      app: ComfyApp
      button: HTMLButtonElement
      updateButton: HTMLButtonElement
      progress: HTMLProgressElement
      message: HTMLElement
    }
  | undefined
let serverLinkView: { link: HTMLAnchorElement; message: HTMLSpanElement } | undefined

async function setSettingFromBackend(app: ComfyApp, id: string, value: unknown): Promise<void> {
  syncingSettings.set(id, (syncingSettings.get(id) ?? 0) + 1)
  try {
    await Promise.resolve(app.extensionManager.setting.set(id, value))
  } finally {
    const pending = (syncingSettings.get(id) ?? 1) - 1
    if (pending > 0) syncingSettings.set(id, pending)
    else syncingSettings.delete(id)
  }
}

async function updatePathStatus(app: ComfyApp, status: RuntimeStatus): Promise<void> {
  latestStatus = status
  const messages = getRuntimeMessages(
    app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
  )
  const settings = app.ui.settings as any
  const modelDirSetting = settings.settingsParamLookup?.[MODEL_DIR_SETTING]
  if (modelDirSetting) {
    modelDirSetting.options = status.model_dirs
  }
  if (app.extensionManager.setting.get<string>(MODEL_DIR_SETTING) !== status.model_dir) {
    await setSettingFromBackend(app, MODEL_DIR_SETTING, status.model_dir)
  }
  app.extensionManager.setting.set(SERVICE_STATE_SETTING, status.state)
  app.extensionManager.setting.set(
    PATH_STATUS_SETTING,
    status.update_available
      ? messages.internalUpdateAvailable
      : status.llama_source === "path"
        ? messages.pathAvailable
        : status.llama_source === "internal"
          ? messages.internalInstallAvailable
          : messages.pathUnavailable,
  )
  app.extensionManager.setting.set(EXECUTABLE_PATH_SETTING, status.llama_executable ?? "—")
  app.extensionManager.setting.set(LLAMA_VERSION_SETTING, status.llama_version ?? "—")
  downloadInProgress = downloadStartPending || isDownloadActive(status.download_state)
  renderDownloadView()
  renderServerLink()
}

function isDownloadActive(state: RuntimeStatus["download_state"]): boolean {
  return state === "downloading" || state === "installing"
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ["KB", "MB", "GB", "TB"]
  let value = bytes
  let unit = -1
  do {
    value /= 1024
    unit += 1
  } while (value >= 1024 && unit < units.length - 1)
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`
}

function renderDownloadView(): void {
  if (!downloadView) return
  const { app, button, updateButton, progress, message } = downloadView
  const status = latestStatus
  const messages = getRuntimeMessages(
    app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
  )
  const state = status?.download_state
  const active = downloadInProgress || (state !== undefined && isDownloadActive(state))
  const installed = state === "installed"
  const updateAvailable = status?.update_available === true
  const target = status?.download_target ?? "llama.cpp"
  progress.setAttribute("aria-label", messages.downloadProgressLabel)

  button.textContent = active
    ? state === "installing"
      ? messages.installing(target)
      : state === "downloading" && status
        ? messages.downloading(
            target,
            status.download_bytes_total && status.download_bytes_total > 0
              ? `${formatBytes(status.download_bytes_received)} / ${formatBytes(status.download_bytes_total)}`
              : formatBytes(status.download_bytes_received),
          )
        : messages.downloadStarting
    : messages.downloadButton
  updateButton.textContent = active ? button.textContent : messages.updateButton
  button.hidden = updateAvailable
  updateButton.hidden = !updateAvailable
  updateButton.title = messages.updateTooltip
  button.disabled =
    !status ||
    status.llama_available ||
    !status.download_supported ||
    (installed && !updateAvailable) ||
    active
  updateButton.disabled = !status || !updateAvailable || !status.download_supported || active

  progress.hidden = state !== "downloading"
  if (state === "downloading" && status) {
    if (status.download_bytes_total && status.download_bytes_total > 0) {
      progress.max = status.download_bytes_total
      progress.value = status.download_bytes_received
    } else {
      progress.removeAttribute("value")
    }
  }

  if (downloadRequestError) {
    message.textContent = messages.downloadFailed(downloadRequestError)
  } else if (state === "downloading" && status) {
    const total = status.download_bytes_total
    const received = formatBytes(status.download_bytes_received)
    message.textContent = messages.downloading(
      target,
      total && total > 0 ? `${received} / ${formatBytes(total)}` : received,
    )
  } else if (state === "installing") {
    message.textContent = messages.installing(target)
  } else if (state === "error") {
    message.textContent = messages.downloadFailed(status?.download_error ?? messages.unknownError)
  } else if (updateAvailable) {
    message.textContent = messages.updateAvailable
  } else if (state === "installed") {
    message.textContent = messages.installed(target)
  } else if (status && !status.llama_available && !status.download_supported) {
    message.textContent = status.download_support_error ?? messages.downloadUnsupported
  } else {
    message.textContent = ""
  }
}

function scheduleDownloadStatusPoll(app: ComfyApp, api: ComfyApi): void {
  if (downloadPollTimer !== undefined) clearTimeout(downloadPollTimer)
  downloadPollTimer = setTimeout(() => {
    downloadPollTimer = undefined
    void refreshDownloadStatus(app, api)
  }, DOWNLOAD_POLL_INTERVAL_MS)
}

async function refreshDownloadStatus(
  app: ComfyApp,
  api: ComfyApi,
  keepRequestError = false,
): Promise<void> {
  const wasActive = downloadInProgress
  try {
    const status = await requestLatestStatus(api)
    if (
      !keepRequestError ||
      isDownloadActive(status.download_state) ||
      status.download_state === "installed" ||
      status.download_state === "error"
    ) {
      downloadRequestError = null
    }
    if (status.download_state !== "idle") downloadStartPending = false
    await updatePathStatus(app, status)
    if (wasActive && status.download_state === "installed") {
      showStartWarning(app, status)
    }
    if (downloadStartPending || isDownloadActive(status.download_state)) {
      scheduleDownloadStatusPoll(app, api)
    }
  } catch (error) {
    downloadInProgress = true
    downloadRequestError =
      error instanceof Error ? error.message : "Could not refresh download status."
    renderDownloadView()
    scheduleDownloadStatusPoll(app, api)
  }
}

async function startDownload(app: ComfyApp, api: ComfyApi): Promise<void> {
  const updateAvailable = latestStatus?.update_available === true
  if (
    downloadInProgress ||
    !latestStatus ||
    (latestStatus.llama_available && !updateAvailable) ||
    (latestStatus.download_state === "installed" && !updateAvailable) ||
    !latestStatus.download_supported
  )
    return
  downloadInProgress = true
  downloadStartPending = true
  downloadRequestError = null
  renderDownloadView()
  try {
    const start = settingUpdateQueue.then(() =>
      api.fetchApi(RUNTIME_DOWNLOAD_ROUTE, { method: "POST" }),
    )
    settingUpdateQueue = start.then(
      () => undefined,
      () => undefined,
    )
    const response = await start
    if (!response.ok) {
      const payload: unknown = await response.json().catch(() => null)
      throw new Error(
        payload && typeof payload === "object" && "error" in payload
          ? String(payload.error)
          : `ComfyUI returned HTTP ${response.status}.`,
      )
    }
    await refreshDownloadStatus(app, api)
  } catch (error) {
    downloadStartPending = false
    downloadRequestError = error instanceof Error ? error.message : "Unknown error."
    await refreshDownloadStatus(app, api, true)
  }
}

function createDownloadControl(app: ComfyApp, api: ComfyApi): HTMLElement {
  const wrapper = document.createElement("div")
  wrapper.style.display = "flex"
  wrapper.style.flexWrap = "wrap"
  wrapper.style.gap = "4px"

  const button = document.createElement("button")
  button.type = "button"
  button.addEventListener("click", () => void startDownload(app, api))
  const updateButton = document.createElement("button")
  updateButton.type = "button"
  updateButton.addEventListener("click", () => void startDownload(app, api))
  const progress = document.createElement("progress")
  progress.setAttribute("aria-label", "llama.cpp download progress")
  progress.hidden = true
  const message = document.createElement("div")
  message.setAttribute("role", "status")
  message.setAttribute("aria-live", "polite")
  wrapper.append(message, progress, button, updateButton)
  downloadView = { app, button, updateButton, progress, message }
  renderDownloadView()
  void refreshDownloadStatus(app, api)
  return wrapper
}

function renderServerLink(): void {
  if (!serverLinkView) return
  const { link, message } = serverLinkView
  const port = latestStatus?.active_port
  if (port === null || port === undefined) {
    link.hidden = true
    link.removeAttribute("href")
    message.hidden = false
    message.textContent = latestStatus
      ? "Internal daemon is not running."
      : "Checking internal daemon status…"
    return
  }

  link.href = `http://127.0.0.1:${port}`
  link.textContent = `127.0.0.1:${port}`
  link.hidden = false
  message.hidden = true
}

function createServerLinkControl(): HTMLElement {
  const wrapper = document.createElement("div")
  const link = document.createElement("a")
  link.target = "_blank"
  link.rel = "noopener noreferrer"
  const message = document.createElement("span")
  wrapper.append(link, message)
  serverLinkView = { link, message }
  renderServerLink()
  return wrapper
}

function showStartWarning(app: ComfyApp, status: RuntimeStatus): void {
  if (isDownloadActive(status.download_state)) return
  if (status.state === "failed" || (status.auto_start && !status.running)) {
    const messages = getRuntimeMessages(
      app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
    )
    app.extensionManager.toast.add({
      severity: "warn",
      summary: messages.startFailureSummary(PROJECT_NAME),
      detail: status.error ?? messages.startFailureFallback,
      life: 7000,
    })
  }
}

async function requestStatus(
  api: ComfyApi,
  update?: RuntimeSettingsUpdate,
  restart = false,
): Promise<RuntimeStatus> {
  const route = restart ? RUNTIME_RESTART_ROUTE : RUNTIME_ROUTE
  const options: RequestInit =
    update === undefined
      ? { method: restart ? "POST" : "GET" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(update),
        }
  const response = await api.fetchApi(route, options)
  const payload: unknown = await response.json()
  if (
    !response.ok ||
    !payload ||
    typeof payload !== "object" ||
    !("auto_start" in payload) ||
    typeof payload.auto_start !== "boolean" ||
    !("state" in payload) ||
    typeof payload.state !== "string" ||
    !SERVICE_STATES.includes(payload.state as RuntimeStatus["state"]) ||
    !("ctx_size" in payload) ||
    typeof payload.ctx_size !== "number" ||
    !Number.isInteger(payload.ctx_size) ||
    payload.ctx_size < MIN_CTX_SIZE ||
    payload.ctx_size > MAX_CTX_SIZE ||
    !("port" in payload) ||
    typeof payload.port !== "number" ||
    !Number.isInteger(payload.port) ||
    payload.port < MIN_PORT ||
    payload.port > MAX_PORT ||
    !("active_port" in payload) ||
    (typeof payload.active_port !== "number" && payload.active_port !== null) ||
    (typeof payload.active_port === "number" &&
      (!Number.isInteger(payload.active_port) ||
        payload.active_port < MIN_PORT ||
        payload.active_port > MAX_PORT)) ||
    !("model_dir" in payload) ||
    typeof payload.model_dir !== "string" ||
    !("model_dirs" in payload) ||
    !Array.isArray(payload.model_dirs) ||
    !payload.model_dirs.every((path) => typeof path === "string") ||
    !("default_model_dir" in payload) ||
    typeof payload.default_model_dir !== "string" ||
    !payload.model_dirs.includes(payload.model_dir) ||
    !payload.model_dirs.includes(payload.default_model_dir) ||
    !("llama_available" in payload) ||
    typeof payload.llama_available !== "boolean" ||
    !("llama_source" in payload) ||
    (payload.llama_source !== "path" &&
      payload.llama_source !== "internal" &&
      payload.llama_source !== null) ||
    !("llama_executable" in payload) ||
    (typeof payload.llama_executable !== "string" && payload.llama_executable !== null) ||
    !("llama_version" in payload) ||
    (typeof payload.llama_version !== "string" && payload.llama_version !== null) ||
    ("update_available" in payload && typeof payload.update_available !== "boolean") ||
    !("running" in payload) ||
    typeof payload.running !== "boolean" ||
    !("error" in payload) ||
    (typeof payload.error !== "string" && payload.error !== null) ||
    !("download_state" in payload) ||
    typeof payload.download_state !== "string" ||
    !DOWNLOAD_STATES.includes(payload.download_state as RuntimeStatus["download_state"]) ||
    !("download_target" in payload) ||
    (typeof payload.download_target !== "string" && payload.download_target !== null) ||
    !("download_supported" in payload) ||
    typeof payload.download_supported !== "boolean" ||
    !("download_support_error" in payload) ||
    (typeof payload.download_support_error !== "string" &&
      payload.download_support_error !== null) ||
    !("download_bytes_received" in payload) ||
    typeof payload.download_bytes_received !== "number" ||
    !Number.isInteger(payload.download_bytes_received) ||
    payload.download_bytes_received < 0 ||
    !("download_bytes_total" in payload) ||
    (typeof payload.download_bytes_total !== "number" && payload.download_bytes_total !== null) ||
    (typeof payload.download_bytes_total === "number" &&
      (!Number.isInteger(payload.download_bytes_total) || payload.download_bytes_total < 0)) ||
    !("download_error" in payload) ||
    (typeof payload.download_error !== "string" && payload.download_error !== null)
  ) {
    const detail =
      payload && typeof payload === "object" && "error" in payload
        ? String(payload.error)
        : `ComfyUI returned HTTP ${response.status}.`
    throw new Error(detail)
  }
  return payload as RuntimeStatus
}

let currentStatusRequest: Promise<RuntimeStatus> | undefined

function requestLatestStatus(api: ComfyApi): Promise<RuntimeStatus> {
  if (currentStatusRequest) return currentStatusRequest
  const request = requestStatus(api).finally(() => {
    if (currentStatusRequest === request) currentStatusRequest = undefined
  })
  currentStatusRequest = request
  return request
}

function createRestartButton(app: ComfyApp, api: ComfyApi): HTMLButtonElement {
  const messages = getRuntimeMessages(
    app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
  )
  const button = document.createElement("button")
  button.type = "button"
  button.textContent = messages.restartButton
  button.title = messages.restartTooltip
  button.setAttribute("aria-label", messages.restartButton)
  button.disabled = restartInProgress
  button.addEventListener("click", () => {
    if (restartInProgress) return
    restartInProgress = true
    button.disabled = true
    button.textContent = messages.restarting

    const restart = settingUpdateQueue.then(async () => {
      if (app.extensionManager.setting.get<boolean>(AUTO_START_SETTING) !== true) {
        app.extensionManager.toast.add({
          severity: "info",
          summary: messages.restartRequiresAutoStart,
          life: 5000,
        })
        return
      }

      const status = await requestStatus(api, undefined, true)
      await updatePathStatus(app, status)
      if (status.state === "running") {
        app.extensionManager.toast.add({
          severity: "success",
          summary: messages.restartSuccessSummary(PROJECT_NAME),
          detail: messages.restartSuccessDetail,
          life: 5000,
        })
      } else {
        showStartWarning(app, status)
      }
    })
    settingUpdateQueue = restart.then(
      () => undefined,
      () => undefined,
    )
    void restart
      .catch((error: unknown) => {
        app.extensionManager.toast.add({
          severity: "error",
          summary: messages.restartFailureSummary(PROJECT_NAME),
          detail: error instanceof Error ? error.message : messages.unknownError,
          life: 7000,
        })
      })
      .finally(() => {
        restartInProgress = false
        button.disabled = false
        button.textContent = messages.restartButton
      })
  })
  return button
}

export function registerLlamaCppRuntimeSettings(app: ComfyApp, api: ComfyApi): void {
  app.registerExtension({
    name: "ollama-image-list.llama-cpp-runtime-settings",
    settings: [
      {
        id: AUTO_START_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "AutoStart"],
        name: "Internal llama.cpp runtime activation",
        tooltip:
          "Starts the PATH llama executable on 127.0.0.1 now and on each ComfyUI startup. Settings changes restart the owned service; closing ComfyUI stops it.",
        type: "boolean",
        defaultValue: false,
        sortOrder: 100,
        async onChange(value, oldValue) {
          if (
            syncingSettings.has(AUTO_START_SETTING) ||
            oldValue === undefined ||
            typeof value !== "boolean"
          )
            return
          const revision = ++autoStartRevision
          setupRevision += 1
          try {
            const update = settingUpdateQueue.then(() => requestStatus(api, { auto_start: value }))
            settingUpdateQueue = update.then(
              () => undefined,
              () => undefined,
            )
            const status = await update
            if (revision !== autoStartRevision) return
            await updatePathStatus(app, status)
            showStartWarning(app, status)
          } catch (error) {
            if (revision !== autoStartRevision) return
            const messages = getRuntimeMessages(
              app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
            )
            app.extensionManager.toast.add({
              severity: "error",
              summary: messages.saveFailureSummary(PROJECT_NAME),
              detail: error instanceof Error ? error.message : messages.unknownError,
              life: 5000,
            })
            if (typeof oldValue === "boolean") {
              try {
                await setSettingFromBackend(app, AUTO_START_SETTING, oldValue)
              } catch {
                // The original setting write has already been reported above.
              }
            }
          }
        },
      },
      {
        id: SERVICE_STATE_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "LifecycleStatus"],
        name: "Internal service state",
        tooltip: "Lifecycle state of the service owned by this ComfyUI process.",
        type: "text",
        attrs: { readonly: true },
        defaultValue: "stopped",
        sortOrder: 90,
        telemetry: { trackChanges: false },
      },
      {
        id: SERVER_LINK_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "ServerLink"],
        name: "Internal server address (Beta)",
        tooltip:
          "Opens the running llama.cpp server in a new tab. This 127.0.0.1 link works from a browser on the same machine as ComfyUI.",
        type: () => createServerLinkControl(),
        defaultValue: null,
        sortOrder: 85,
        telemetry: { trackChanges: false },
      },
      {
        id: "OllamaImageList.LlamaCpp.Download" as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "Install"],
        name: "Install llama.cpp",
        tooltip: "Download the pinned llama.cpp build or update an older internal installation.",
        type: () => createDownloadControl(app, api),
        defaultValue: null,
        sortOrder: 95,
        telemetry: { trackChanges: false },
      },
      {
        id: PATH_STATUS_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "PathStatus"],
        name: "llama executable availability",
        tooltip: "Shows whether llama is available on PATH or provided by the internal install.",
        type: "text",
        attrs: { readonly: true, style: { width: "auto" } },
        defaultValue: getRuntimeMessages(
          app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
        ).checking,
        sortOrder: 80,
        telemetry: { trackChanges: false },
      },
      {
        id: EXECUTABLE_PATH_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "ExecutablePath"],
        name: "llama executable path",
        tooltip: "The resolved executable path when llama is available.",
        type: "text",
        attrs: { readonly: true, style: { width: "26rem" } },
        defaultValue: "—",
        sortOrder: 50,
        telemetry: { trackChanges: false },
      },
      {
        id: LLAMA_VERSION_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "Version"],
        name: "llama version",
        tooltip: "The version reported by llama --version.",
        type: "text",
        attrs: { readonly: true, style: { width: "26rem" } },
        defaultValue: "—",
        sortOrder: 60,
        telemetry: { trackChanges: false },
      },
      {
        id: RESTART_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon", "RestartActor"],
        name: "Restart internal daemon",
        tooltip: getRuntimeMessages(app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING))
          .restartTooltip,
        type: () => createRestartButton(app, api),
        defaultValue: null,
        sortOrder: 30,
        telemetry: { trackChanges: false },
      },
      {
        id: CTX_SIZE_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon Config", "ContextSize"],
        name: "Context size",
        tooltip:
          "Changing this while the internal server is running restarts it with the new context size.",
        type: "number",
        attrs: {
          min: MIN_CTX_SIZE,
          max: MAX_CTX_SIZE,
          step: 1,
          useGrouping: true,
          locale: "en-US",
        },
        defaultValue: DEFAULT_CTX_SIZE,
        sortOrder: 0,
        telemetry: { trackChanges: false },
        async onChange(value, oldValue) {
          if (syncingSettings.has(CTX_SIZE_SETTING) || oldValue === undefined) return
          if (
            typeof value !== "number" ||
            !Number.isInteger(value) ||
            value < MIN_CTX_SIZE ||
            value > MAX_CTX_SIZE
          ) {
            if (typeof oldValue === "number") {
              await setSettingFromBackend(app, CTX_SIZE_SETTING, oldValue)
            }
            return
          }
          const revision = ++ctxSizeRevision
          setupRevision += 1
          try {
            const update = settingUpdateQueue.then(() => requestStatus(api, { ctx_size: value }))
            settingUpdateQueue = update.then(
              () => undefined,
              () => undefined,
            )
            const status = await update
            if (revision !== ctxSizeRevision) return
            await updatePathStatus(app, status)
          } catch (error) {
            if (revision !== ctxSizeRevision) return
            const messages = getRuntimeMessages(
              app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
            )
            app.extensionManager.toast.add({
              severity: "error",
              summary: messages.saveFailureSummary(PROJECT_NAME),
              detail: error instanceof Error ? error.message : messages.unknownError,
              life: 5000,
            })
            if (typeof oldValue === "number") {
              try {
                await setSettingFromBackend(app, CTX_SIZE_SETTING, oldValue)
              } catch {
                // The original setting write has already been reported above.
              }
            }
          }
        },
      },
      {
        id: PORT_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon Config", "Port"],
        name: "Internal server port",
        tooltip: "Changing this while the internal server is running restarts it on the new port.",
        type: "number",
        attrs: {
          min: MIN_PORT,
          max: MAX_PORT,
          step: 1,
          useGrouping: false,
        },
        defaultValue: DEFAULT_PORT,
        sortOrder: -10,
        telemetry: { trackChanges: false },
        async onChange(value, oldValue) {
          if (syncingSettings.has(PORT_SETTING) || oldValue === undefined) return
          if (
            typeof value !== "number" ||
            !Number.isInteger(value) ||
            value < MIN_PORT ||
            value > MAX_PORT
          ) {
            if (typeof oldValue === "number") {
              await setSettingFromBackend(app, PORT_SETTING, oldValue)
            }
            return
          }
          const revision = ++portRevision
          setupRevision += 1
          try {
            const update = settingUpdateQueue.then(() => requestStatus(api, { port: value }))
            settingUpdateQueue = update.then(
              () => undefined,
              () => undefined,
            )
            const status = await update
            if (revision !== portRevision) return
            await updatePathStatus(app, status)
          } catch (error) {
            if (revision !== portRevision) return
            const messages = getRuntimeMessages(
              app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
            )
            app.extensionManager.toast.add({
              severity: "error",
              summary: messages.saveFailureSummary(PROJECT_NAME),
              detail: error instanceof Error ? error.message : messages.unknownError,
              life: 5000,
            })
            if (typeof oldValue === "number") {
              try {
                await setSettingFromBackend(app, PORT_SETTING, oldValue)
              } catch {
                // The original setting write has already been reported above.
              }
            }
          }
        },
      },
      {
        id: MODEL_DIR_SETTING as any,
        category: [SETTINGS_PROJECT_NAME, "llama.cpp Daemon Config", "ModelDir"],
        name: "LLM models directory",
        tooltip:
          "Changing this while the internal server is running restarts it with the new models directory.",
        type: "combo",
        options: [],
        defaultValue: "",
        sortOrder: -20,
        telemetry: { trackChanges: false },
        attrs: {
          style: { width: "20rem" },
        },
        async onChange(value, oldValue) {
          if (syncingSettings.has(MODEL_DIR_SETTING) || oldValue === undefined) return
          if (
            typeof value !== "string" ||
            (latestStatus && !latestStatus.model_dirs.includes(value))
          ) {
            if (typeof oldValue === "string") {
              await setSettingFromBackend(app, MODEL_DIR_SETTING, oldValue)
            }
            return
          }
          const revision = ++modelDirRevision
          setupRevision += 1
          try {
            const update = settingUpdateQueue.then(() => requestStatus(api, { model_dir: value }))
            settingUpdateQueue = update.then(
              () => undefined,
              () => undefined,
            )
            const status = await update
            if (revision !== modelDirRevision) return
            await updatePathStatus(app, status)
          } catch (error) {
            if (revision !== modelDirRevision) return
            const messages = getRuntimeMessages(
              app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
            )
            app.extensionManager.toast.add({
              severity: "error",
              summary: messages.saveFailureSummary(PROJECT_NAME),
              detail: error instanceof Error ? error.message : messages.unknownError,
              life: 5000,
            })
            if (typeof oldValue === "string") {
              try {
                await setSettingFromBackend(app, MODEL_DIR_SETTING, oldValue)
              } catch {
                // The original setting write has already been reported above.
              }
            }
          }
        },
      },
    ],
    async setup() {
      app.ui.settings.addEventListener(`${COMFY_LOCALE_SETTING}.change`, () => {
        if (latestStatus) {
          void updatePathStatus(app, latestStatus).catch(() => undefined)
        } else {
          renderDownloadView()
        }
      })
      try {
        const revision = setupRevision
        const status = await requestLatestStatus(api)
        if (revision !== setupRevision) return
        await updatePathStatus(app, status)
        if (isDownloadActive(status.download_state)) {
          void refreshDownloadStatus(app, api)
        }
        if (app.extensionManager.setting.get<boolean>(AUTO_START_SETTING) !== status.auto_start) {
          await setSettingFromBackend(app, AUTO_START_SETTING, status.auto_start)
        }
        if (app.extensionManager.setting.get<number>(CTX_SIZE_SETTING) !== status.ctx_size) {
          await setSettingFromBackend(app, CTX_SIZE_SETTING, status.ctx_size)
        }
        if (app.extensionManager.setting.get<number>(PORT_SETTING) !== status.port) {
          await setSettingFromBackend(app, PORT_SETTING, status.port)
        }
        showStartWarning(app, status)
      } catch {
        const messages = getRuntimeMessages(
          app.extensionManager.setting.get<string>(COMFY_LOCALE_SETTING),
        )
        app.extensionManager.setting.set(PATH_STATUS_SETTING, messages.statusLoadFailure)
      }
    },
  })
}
