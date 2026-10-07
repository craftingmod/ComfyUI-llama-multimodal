import { expect, it } from "bun:test"

import { githubReleaseInfo } from "../../scripts/prepare-github-release.ts"
import {
  packagedPyproject,
  projectArchiveName,
  registryVersionSource,
} from "../../scripts/project-version.ts"

const pyprojectSource = '[project]\nname = "example"\n[tool.comfy]\nDisplayName = "Example Node"'

it("freezes only the packaged version while preserving other dynamic metadata", async () => {
  const source = await Bun.file("pyproject.toml").text()
  const packaged = Bun.TOML.parse(packagedPyproject(source, "1.3.4")) as {
    project: { version: string; dynamic?: string[] }
    tool: unknown
  }
  expect(packaged.project.version).toBe("1.3.4")
  expect(packaged.project.dynamic).toBeUndefined()
  expect(packaged.tool).toEqual((Bun.TOML.parse(source) as { tool: unknown }).tool)
  expect(await Bun.file("pyproject.toml").text()).toBe(source)
  const extra = '[project]\nname = "example"\ndynamic = ["version", "description"]'
  expect(
    (
      Bun.TOML.parse(packagedPyproject(extra, "1.3.4")) as {
        project: { dynamic: string[] }
      }
    ).project.dynamic,
  ).toEqual(["description"])
  expect(() => packagedPyproject(source, '1.3.4"\ninvalid')).toThrow("X.Y.Z")
})

it("generates the version assignment read by comfy-cli", () => {
  for (const version of ["1.2.3", "1.2.3.dev1", "0.9.4.post7.dev0+d064875", "1.2.3-rc.1+build.2"]) {
    expect(registryVersionSource(version)).toBe(`__version__ = "${version}"\n`)
    expect(projectArchiveName("Example Node", version)).toBe(`Example Node-${version}.zip`)
    expect(githubReleaseInfo(pyprojectSource, `v${version}`)).toEqual({
      archivePath: `build/Example Node-${version}.zip`,
      projectName: "example",
      version,
    })
  }
  for (const version of ["1.2", "1.2.3.", "1.2.3/extra", '1.2.3"\ninvalid']) {
    expect(() => registryVersionSource(version)).toThrow("X.Y.Z")
    expect(() => githubReleaseInfo(pyprojectSource, `v${version}`)).toThrow("vX.Y.Z")
  }
})

it("rejects missing or unsafe archive display names", () => {
  for (const name of [
    undefined,
    "",
    "../example",
    "example\\node",
    "example:node",
    "example\nnode",
  ]) {
    expect(() => projectArchiveName(name, "1.2.3")).toThrow("tool.comfy.DisplayName")
  }
})
