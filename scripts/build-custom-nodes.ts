import fs from "node:fs/promises"
import Path from "node:path"

import { $ } from "bun"
import { zipSync } from "fflate"

import {
  getProjectVersion,
  packagedPyproject,
  projectArchiveName,
  registryVersionSource,
} from "./project-version.ts"

type ComfyConfig = {
  tool?: { comfy?: { DisplayName?: unknown; includes?: unknown } }
}

const projectDir = Path.resolve(import.meta.dir, "../")
const outputDir = Path.join(projectDir, "build")

$.cwd(projectDir)

function normalizeArchivePath(filePath: string): string {
  const normalized = filePath.replaceAll("\\", "/").replace(/^\.\//, "")
  if (normalized.length === 0 || normalized.startsWith("/") || normalized.includes("../")) {
    throw new Error(`Refusing unsafe archive path: ${filePath}`)
  }
  return normalized
}

async function collectDirectoryFiles(directory: string, files: Set<string>): Promise<void> {
  const entries = await fs.readdir(Path.join(projectDir, directory), { withFileTypes: true })
  for (const entry of entries) {
    const relativePath = normalizeArchivePath(Path.posix.join(directory, entry.name))
    const stats = await fs.lstat(Path.join(projectDir, relativePath))
    if (stats.isSymbolicLink()) {
      throw new Error(`Refusing to package symbolic link: ${relativePath}`)
    }
    if (stats.isDirectory()) {
      await collectDirectoryFiles(relativePath, files)
    } else if (stats.isFile()) {
      files.add(relativePath)
    }
  }
}

async function addIncludedPath(includePath: string, files: Set<string>): Promise<void> {
  const relativePath = normalizeArchivePath(includePath)
  const absolutePath = Path.resolve(projectDir, relativePath)
  const relativeToProject = Path.relative(projectDir, absolutePath)
  if (relativeToProject.startsWith("..") || Path.isAbsolute(relativeToProject)) {
    throw new Error(`Included path escapes the project directory: ${includePath}`)
  }

  const stats = await fs.lstat(absolutePath).catch(() => undefined)
  if (!stats) {
    throw new Error(`Included path does not exist: ${includePath}`)
  }
  if (stats.isSymbolicLink()) {
    throw new Error(`Refusing to package symbolic link: ${includePath}`)
  }
  if (stats.isDirectory()) {
    await collectDirectoryFiles(relativePath, files)
  } else if (stats.isFile()) {
    files.add(relativePath)
  }
}

async function gitFiles(): Promise<Set<string>> {
  const tracked = (await $`git ls-files -z`.text())
    .split("\0")
    .filter(Boolean)
    .map(normalizeArchivePath)
  const ignored = new Set<string>()

  if (await Bun.file(Path.join(projectDir, ".comfyignore")).exists()) {
    const output = await $`git ls-files --cached --ignored --exclude-from=.comfyignore -z`.text()
    for (const filePath of output.split("\0").filter(Boolean)) {
      ignored.add(normalizeArchivePath(filePath))
    }
  }

  return new Set(tracked.filter((filePath) => !ignored.has(filePath)))
}

const pyproject = Bun.TOML.parse(
  await fs.readFile(Path.join(projectDir, "pyproject.toml"), "utf8"),
) as ComfyConfig
const version = await getProjectVersion(projectDir)
const archiveName = projectArchiveName(pyproject.tool?.comfy?.DisplayName, version)
await fs.writeFile(Path.join(projectDir, "backend", "_version.py"), registryVersionSource(version))
const includesValue = pyproject.tool?.comfy?.includes ?? []
if (!Array.isArray(includesValue) || includesValue.some((value) => typeof value !== "string")) {
  throw new Error("Expected tool.comfy.includes to be an array of strings in pyproject.toml")
}

await $`bun run build`

const files = await gitFiles()
for (const includePath of includesValue as string[]) {
  await addIncludedPath(includePath, files)
}

const archiveFiles: Record<string, Uint8Array> = {}
for (const relativePath of [...files].sort()) {
  const absolutePath = Path.join(projectDir, relativePath)
  const stats = await fs.lstat(absolutePath).catch(() => undefined)
  if (!stats) {
    throw new Error(`Tracked package file is missing: ${relativePath}`)
  }
  if (!stats.isFile() || stats.isSymbolicLink()) {
    throw new Error(`Package entry must be a regular file: ${relativePath}`)
  }
  archiveFiles[relativePath] = new Uint8Array(await fs.readFile(absolutePath))
}

if (!("__init__.py" in archiveFiles) || !("pyproject.toml" in archiveFiles)) {
  throw new Error("Package must contain __init__.py and pyproject.toml")
}

archiveFiles["pyproject.toml"] = new TextEncoder().encode(
  packagedPyproject(new TextDecoder().decode(archiveFiles["pyproject.toml"]), version),
)

const zip = zipSync(archiveFiles, {
  level: 9,
  mtime: new Date("1980-01-01T00:00:00.000Z"),
})
const outputPath = Path.join(outputDir, archiveName)
await fs.mkdir(outputDir, { recursive: true })
await Bun.write(outputPath, zip)

console.log(`Built ${Path.relative(projectDir, outputPath)} with ${files.size} files.`)
