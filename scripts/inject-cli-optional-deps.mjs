#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PLATFORMS } from "../packages/cli/src/index.js";

const version = process.argv[2];
if (!version) {
  console.error("Usage: inject-cli-optional-deps.mjs <version>");
  process.exit(1);
}

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const file = join(repoRoot, "packages", "cli", "package.json");
const pkg = JSON.parse(readFileSync(file, "utf8"));

if (pkg.version !== version) {
  console.error(`Version mismatch: package.json is ${pkg.version}, asked for ${version}`);
  process.exit(1);
}

pkg.optionalDependencies = Object.fromEntries(
  Object.values(PLATFORMS)
    .sort()
    .map((name) => [name, version]),
);

writeFileSync(file, `${JSON.stringify(pkg, null, 2)}\n`);
console.log(`Injected optionalDependencies @ ${version}:`);
for (const name of Object.keys(pkg.optionalDependencies)) console.log(`  ${name}`);
