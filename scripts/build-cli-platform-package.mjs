#!/usr/bin/env node
// Build a publishable @substructure.ai/cli-<platform> package directory.
//
// Usage:
//   node scripts/build-cli-platform-package.mjs \
//     --target <rust-target-triple> \
//     --binary <path-to-built-binary> \
//     --version <version> \
//     --out-dir <output-dir>

import { readFileSync, writeFileSync, mkdirSync, copyFileSync, chmodSync, existsSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..");

const TARGETS = {
  "aarch64-apple-darwin":      { platform: "darwin-arm64", os: "darwin", cpu: "arm64", binName: "substructure" },
  "x86_64-apple-darwin":       { platform: "darwin-x64",   os: "darwin", cpu: "x64",   binName: "substructure" },
  "aarch64-unknown-linux-gnu": { platform: "linux-arm64",  os: "linux",  cpu: "arm64", binName: "substructure", libc: "glibc" },
  "x86_64-unknown-linux-gnu":  { platform: "linux-x64",    os: "linux",  cpu: "x64",   binName: "substructure", libc: "glibc" },
  "x86_64-pc-windows-msvc":    { platform: "win32-x64",    os: "win32",  cpu: "x64",   binName: "substructure.exe" },
};

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const k = argv[i];
    const v = argv[i + 1];
    if (!k?.startsWith("--") || v === undefined) throw new Error(`Bad arg: ${k}`);
    out[k.slice(2)] = v;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
for (const k of ["target", "binary", "version", "out-dir"]) {
  if (!args[k]) {
    console.error(`Missing required --${k}`);
    process.exit(1);
  }
}

const target = args.target;
const meta = TARGETS[target];
if (!meta) {
  console.error(`Unknown target: ${target}. Known: ${Object.keys(TARGETS).join(", ")}`);
  process.exit(1);
}

const sourceBinary = resolve(args.binary);
if (!existsSync(sourceBinary)) {
  console.error(`Binary not found: ${sourceBinary}`);
  process.exit(1);
}

const outDir = resolve(args["out-dir"]);
const binDir = join(outDir, "bin");
mkdirSync(binDir, { recursive: true });

const destBinary = join(binDir, meta.binName);
copyFileSync(sourceBinary, destBinary);
if (meta.os !== "win32") chmodSync(destBinary, 0o755);

const pkg = {
  name: `@substructure.ai/cli-${meta.platform}`,
  version: args.version,
  description: `${meta.platform} binary for @substructure.ai/cli`,
  license: "FSL-1.1-ALv2",
  homepage: "https://github.com/substructureai/substructure#readme",
  repository: {
    type: "git",
    url: "git+https://github.com/substructureai/substructure.git",
    directory: `packages/cli/npm/${meta.platform}`,
  },
  bugs: { url: "https://github.com/substructureai/substructure/issues" },
  publishConfig: { access: "public" },
  os: [meta.os],
  cpu: [meta.cpu],
  ...(meta.libc ? { libc: [meta.libc] } : {}),
  files: ["bin", "LICENSE", "README.md"],
};

writeFileSync(join(outDir, "package.json"), `${JSON.stringify(pkg, null, 2)}\n`);

const licenseSrc = join(repoRoot, "packages", "cli", "LICENSE");
if (existsSync(licenseSrc)) copyFileSync(licenseSrc, join(outDir, "LICENSE"));

writeFileSync(
  join(outDir, "README.md"),
  `# ${pkg.name}\n\nPlatform-specific binary for [@substructure.ai/cli](https://www.npmjs.com/package/@substructure.ai/cli). Do not install directly.\n`,
);

console.log(`Built ${pkg.name} at ${outDir}`);
