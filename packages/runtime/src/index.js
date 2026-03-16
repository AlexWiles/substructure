import { platform, arch } from "os";
import { createRequire } from "module";
import { join } from "path";

const PLATFORMS = {
  "darwin-arm64": "@substructure.ai/runtime-darwin-arm64",
  "darwin-x64": "@substructure.ai/runtime-darwin-x64",
  "linux-arm64": "@substructure.ai/runtime-linux-arm64",
  "linux-x64": "@substructure.ai/runtime-linux-x64",
  "win32-x64": "@substructure.ai/runtime-win32-x64",
};

function loadBinding() {
  const key = `${platform()}-${arch()}`;
  const pkg = PLATFORMS[key];
  if (!pkg) {
    throw new Error(
      `Unsupported platform: ${key}. Substructure runtime supports: ${Object.keys(PLATFORMS).join(", ")}`
    );
  }
  const require = createRequire(import.meta.url);
  return require(`${pkg}/substructure-napi.node`);
}

const binding = loadBinding();

export const { JsRuntime } = binding;
