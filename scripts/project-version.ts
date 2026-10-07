import fs from "node:fs/promises"
import Path from "node:path"

import { $ } from "bun"

export async function getProjectVersion(projectDir: string): Promise<string> {
  const version = (await $`uvx uv-dynamic-versioning`.cwd(projectDir).text()).trim()
  if (!version) throw new Error("uv-dynamic-versioning returned an empty project version.")
  return version
}

export function registryVersionSource(version: string): string {
  if (!/^\d+\.\d+\.\d+(?:[.+-][0-9A-Za-z]+)*$/.test(version)) {
    throw new Error(`Expected an X.Y.Z version with an optional suffix, got ${version}.`)
  }
  return `__version__ = "${version}"\n`
}

export function packagedPyproject(source: string, version: string): string {
  registryVersionSource(version)
  const metadata = Bun.TOML.parse(source) as {
    project: { version?: string; dynamic?: string[] }
  }
  const project = metadata.project
  project.version = version
  if (project.dynamic) {
    project.dynamic = project.dynamic.filter((field) => field !== "version")
    if (project.dynamic.length === 0) delete project.dynamic
  }
  const packaged = Bun.TOML.stringify(metadata)
  if (!packaged) throw new Error("Failed to serialize packaged pyproject.toml")
  return packaged
}

export function projectArchiveName(displayName: unknown, version: string): string {
  if (typeof displayName !== "string" || !displayName.trim()) {
    throw new Error("Expected tool.comfy.DisplayName to be a non-empty string in pyproject.toml")
  }
  const name = displayName.trim()
  if (
    /[<>:"/\\|?*]/.test(name) ||
    name.split("").some((character) => character.charCodeAt(0) < 32)
  ) {
    throw new Error("tool.comfy.DisplayName contains invalid filename characters")
  }
  registryVersionSource(version)
  return `${name}-${version}.zip`
}

if (import.meta.main) {
  const projectDir = Path.resolve(import.meta.dir, "../")
  const version = await getProjectVersion(projectDir)
  await fs.writeFile(
    Path.join(projectDir, "backend", "_version.py"),
    registryVersionSource(version),
  )
  console.log(`Generated backend/_version.py for Registry version ${version}.`)
}
