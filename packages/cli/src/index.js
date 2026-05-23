import { platform, arch } from "os";
import { createRequire } from "module";
import { join } from "path";

const PLATFORMS = {
  "darwin-arm64": "@substructure.ai/cli-darwin-arm64",
  "darwin-x64": "@substructure.ai/cli-darwin-x64",
  "linux-arm64": "@substructure.ai/cli-linux-arm64",
  "linux-x64": "@substructure.ai/cli-linux-x64",
};

export function resolve() {
  if (process.env.SUBS_BIN) return process.env.SUBS_BIN;

  const key = `${platform()}-${arch()}`;
  const pkg = PLATFORMS[key];
  if (!pkg) {
    throw new Error(
      `Unsupported platform: ${key}. Supported: ${Object.keys(PLATFORMS).join(", ")}`
    );
  }

  const require = createRequire(import.meta.url);

  try {
    const pkgJson = require.resolve(`${pkg}/package.json`);
    return join(pkgJson, "..", "bin", "subs");
  } catch {
    throw new Error(
      `Could not find package ${pkg}. Make sure it's installed.\n` +
      `If you're developing locally, build the Rust binary with: cargo build -p substructure-core --bin subs`
    );
  }
}
