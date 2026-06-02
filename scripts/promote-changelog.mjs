#!/usr/bin/env node
// Promote the `## [Unreleased]` section of CHANGELOG.md to a released version.
//
// Moves everything currently under `[Unreleased]` into a new dated version
// section, resets `[Unreleased]` to empty, and updates the compare-link
// references at the bottom of the file. The GitHub repo URL is derived from the
// existing `[unreleased]` link so there is nothing else to keep in sync.
//
// Usage:
//   node scripts/promote-changelog.mjs <version> <date> [prevVersion]
//   node scripts/promote-changelog.mjs 0.1.15 2026-06-10 0.1.14

import { readFileSync, writeFileSync } from 'node:fs';

const [version, date, prevVersion] = process.argv.slice(2);

if (!version || !date) {
  console.error('Usage: promote-changelog.mjs <version> <date> [prevVersion]');
  process.exit(1);
}

const FILE = new URL('../CHANGELOG.md', import.meta.url);
const src = readFileSync(FILE, 'utf8');
const lines = src.split('\n');

// ── Locate the [Unreleased] heading and its body (up to the next `## ` heading).
const unreleasedIdx = lines.findIndex((l) => /^##\s*\[Unreleased\]/i.test(l));
if (unreleasedIdx === -1) {
  console.error('No `## [Unreleased]` heading found in CHANGELOG.md');
  process.exit(1);
}

let nextHeadingIdx = lines.length;
for (let i = unreleasedIdx + 1; i < lines.length; i++) {
  if (/^##\s/.test(lines[i])) {
    nextHeadingIdx = i;
    break;
  }
}

// Body lines between the heading and the next section, trimmed of blank padding.
const body = lines
  .slice(unreleasedIdx + 1, nextHeadingIdx)
  .join('\n')
  .trim();

if (!body) {
  console.error(
    `Refusing to release: the [Unreleased] section is empty.\n` +
      `Add at least one entry to CHANGELOG.md before cutting v${version}.`,
  );
  process.exit(1);
}

// ── Rebuild the heading region: empty [Unreleased] + new version section.
const newHeading = [
  '## [Unreleased]',
  '',
  `## [${version}] - ${date}`,
  '',
  body,
  '',
];
const headBefore = lines.slice(0, unreleasedIdx);
const tailAfter = lines.slice(nextHeadingIdx);
let out = [...headBefore, ...newHeading, ...tailAfter];

// ── Update the link reference definitions at the bottom of the file.
const linkIdx = out.findIndex((l) => /^\[unreleased\]:/i.test(l));
if (linkIdx === -1) {
  console.error('No `[unreleased]:` link reference found in CHANGELOG.md');
  process.exit(1);
}

// Derive the compare base (e.g. https://github.com/org/repo) from the existing link.
const baseMatch = out[linkIdx].match(/(https?:\/\/\S+?)\/compare\//);
if (!baseMatch) {
  console.error('Could not parse the repo URL from the `[unreleased]:` link.');
  process.exit(1);
}
const base = baseMatch[1];

out[linkIdx] = `[Unreleased]: ${base}/compare/v${version}...HEAD`;
const versionLink = prevVersion
  ? `[${version}]: ${base}/compare/v${prevVersion}...v${version}`
  : `[${version}]: ${base}/releases/tag/v${version}`;
out.splice(linkIdx + 1, 0, versionLink);

writeFileSync(FILE, out.join('\n'));
console.error(`Promoted [Unreleased] -> [${version}] - ${date} in CHANGELOG.md`);
