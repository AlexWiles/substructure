# Releasing

`@substructure.ai/cli` ships from `.github/workflows/release.yml`. A `v*` tag triggers it.

## One-time setup

1. Create an automation-scoped npm token under the `substructure.ai` org.
2. Add it as the `NPM_TOKEN` repository secret (Settings → Secrets and variables → Actions).
3. Publish each platform sub-package through CI first, not by hand. The org must own the name before `optionalDependencies` resolves.
4. Install `cargo-edit`: `cargo install cargo-edit`.

## Changelog

Add bullets under `## [Unreleased]` in `CHANGELOG.md` as you merge. Group them under `### Added` / `### Changed` / `### Fixed`.

`scripts/release.sh` promotes `[Unreleased]` into a dated `[<version>]` section and updates the compare links. It aborts if `[Unreleased]` is empty.

The workflow publishes that section as the GitHub Release body. It falls back to auto-generated notes if the tag has no section.

## Cutting a release

The CLI package and the `substructure-core` crate share a version.

```sh
pnpm run release 0.1.1   # or: patch | minor | major
```

The script requires a clean tree on `main`. It bumps `packages/cli/package.json`, bumps the crate, promotes the changelog, regenerates schemas, runs the tests, then commits and tags.

Push to trigger the workflow:

```sh
git push origin main && git push origin v0.1.1
```

## What the workflow does

1. Builds the CLI binary for `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`, and `aarch64-unknown-linux-gnu` (cross-compiled).
2. Builds an `@substructure.ai/cli-<platform>` package directory per target.
3. Publishes the platform packages.
4. Writes the wrapper's `optionalDependencies`, then publishes `@substructure.ai/cli`.
5. Attaches `subs-<target>.tar.gz` and a `.sha256` to the GitHub Release.

Linux builds use the `ubuntu-22.04` image. Its glibc 2.35 is the floor for anyone who installs the binary.

## Wrapper optionalDependencies

`scripts/inject-cli-optional-deps.mjs` writes them in CI, just before `npm publish`. It reads the names from the `PLATFORMS` map in `packages/cli/src/index.js`, which is what the shim resolves at runtime.

They are not committed. `packages/cli` is a workspace member, so committed entries make every `pnpm install` download the last published binaries and rewrite the lockfile each release.

## Tarballs

`curl -fsSL https://subs.dev/cli.sh | bash` installs the GitHub Release tarballs.

Asset names carry no version, so `releases/latest/download/<name>` resolves without an API call. `SUBS_VERSION` pins a tag.

## Dry runs

Run the workflow from the Actions tab (`workflow_dispatch`). `dry_run` defaults to `true` and builds without publishing. Set it to `false` to publish.

## Adding a target

1. Add the package name to the `PLATFORMS` map in `packages/cli/src/index.js`.
2. Add a matrix entry in `.github/workflows/release.yml`.
3. Add the target metadata to `scripts/build-cli-platform-package.mjs`.
