#!/usr/bin/env node
// Print the body of one version's section from CHANGELOG.md, for use as
// GitHub Release notes. Excludes the `## [version] - date` heading itself.
// Exits non-zero if the version has no section (so CI can fall back to
// auto-generated notes).
//
// Usage:
//   node scripts/changelog-section.mjs <version>
//   node scripts/changelog-section.mjs 0.1.15

import { readFileSync } from 'node:fs';

const version = process.argv[2];
if (!version) {
  console.error('Usage: changelog-section.mjs <version>');
  process.exit(1);
}

const FILE = new URL('../CHANGELOG.md', import.meta.url);
const lines = readFileSync(FILE, 'utf8').split('\n');

// Match `## [<version>]` allowing an optional ` - date` suffix.
const escaped = version.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const headingRe = new RegExp(`^##\\s*\\[${escaped}\\]`);

const start = lines.findIndex((l) => headingRe.test(l));
if (start === -1) {
  console.error(`No section for version ${version} found in CHANGELOG.md`);
  process.exit(1);
}

let end = lines.length;
for (let i = start + 1; i < lines.length; i++) {
  if (/^##\s/.test(lines[i])) {
    end = i;
    break;
  }
}

const body = lines.slice(start + 1, end).join('\n').trim();
if (!body) {
  console.error(`Section for version ${version} is empty`);
  process.exit(1);
}

process.stdout.write(body + '\n');
