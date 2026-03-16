import { platform, arch } from "os";
import { createRequire } from "module";
import { join } from "path";

const PLATFORMS = {
  "darwin-arm64": "@substructure.ai/cli-darwin-arm64",
  "darwin-x64": "@substructure.ai/cli-darwin-x64",
  "linux-arm64": "@substructure.ai/cli-linux-arm64",
  "linux-x64": "@substructure.ai/cli-linux-x64",
  "win32-x64": "@substructure.ai/cli-win32-x64",
};

export function resolve() {
  const key = `${platform()}-${arch()}`;
  const pkg = PLATFORMS[key];
  if (!pkg) {
    throw new Error(
      `Unsupported platform: ${key}. Substructure CLI supports: ${Object.keys(PLATFORMS).join(", ")}`
    );
  }
  const require = createRequire(import.meta.url);
  const pkgJson = require.resolve(`${pkg}/package.json`);
  const binName = key.startsWith("win32") ? "substructure.exe" : "substructure";
  return join(pkgJson, "..", "bin", binName);
}
