#!/usr/bin/env node
// Overwrite the bin shim with the native binary so `subs` execs it directly,
// skipping Node startup. Best-effort: on any failure the Node shim
// (bin/subs.js) stays in place and still works, just slower.
import { linkSync, copyFileSync, chmodSync, renameSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolve } from "./src/index.js";

// Skip under Yarn (overwriting a package file can corrupt its shared cache) and
// when a binary is supplied out of band.
const isYarn = (process.env.npm_config_user_agent || "").startsWith("yarn");
if (process.env.SUBS_BIN || process.env.SUBS_SKIP_POSTINSTALL || isYarn) process.exit(0);

const here = dirname(fileURLToPath(import.meta.url));
const shim = join(here, "bin", "subs.js");
const tmp = `${shim}.tmp`;

try {
  const bin = resolve();
  if (bin === shim) process.exit(0);
  let linked = false;
  try {
    linkSync(bin, tmp); // hardlink: no duplicate on disk
    linked = true;
  } catch {
    copyFileSync(bin, tmp); // cross-device or unsupported: fall back to a copy
    chmodSync(tmp, 0o755);
  }
  renameSync(tmp, shim); // atomic replace
} catch {
  try {
    unlinkSync(tmp);
  } catch {}
}
