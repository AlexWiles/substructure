#!/usr/bin/env node
import { resolve } from "../src/index.js";
import { execFileSync } from "child_process";

const bin = resolve();
try {
  execFileSync(bin, process.argv.slice(2), { stdio: "inherit" });
} catch (e) {
  if (e.status !== undefined) {
    process.exit(e.status);
  }
  throw e;
}
