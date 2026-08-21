# Releasing

The npm package `@substructure.ai/cli` ships from a single tag-triggered GitHub Actions workflow at `.github/workflows/release.yml`.

## One-time setup

1. Create an automation-scoped npm access token under the `substructure.ai` org and add it as the `NPM_TOKEN` repository secret in GitHub (Settings → Secrets and variables → Actions).
2. Make sure each platform sub-package name is reserved by publishing the first version through CI, not by hand. npm requires the org to own them before optionalDependencies will resolve.

## Changelog

Notes live in `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com)
format). As you merge changes, add bullets under the `## [Unreleased]` heading
(grouped under `### Added` / `### Changed` / `### Fixed` / etc.).

At release time `scripts/release.sh` promotes `[Unreleased]` into a dated
`[<version>]` section and updates the compare links — so the release aborts if
`[Unreleased]` is empty. The release workflow then publishes those same notes as
the GitHub Release body (falling back to auto-generated notes if a tag has no
changelog section).

## Cutting a release

The CLI package and the `substructure-core` crate release together at the same version.

```sh
VERSION=0.1.1
npm --prefix packages/cli version "$VERSION" --no-git-tag-version

git commit -am "release $VERSION"
git tag "v$VERSION"
git push origin main --tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

1. Builds the Rust CLI binary for each of:
   - `aarch64-apple-darwin`, `x86_64-apple-darwin`
   - `aarch64-unknown-linux-gnu` (cross-compiled), `x86_64-unknown-linux-gnu`
2. For each target, builds an `@substructure.ai/cli-<platform>` package directory containing the binary.
3. Publishes the CLI platform packages, then `@substructure.ai/cli`.
4. Attaches `subs-<target>.tar.gz` and a `.sha256` for each to the GitHub Release.

Those tarballs are what `curl -fsSL https://subs.dev/cli.sh | bash` installs.
The asset names carry no version so `releases/latest/download/<name>` resolves
without an API call; `SUBS_VERSION` pins a tag instead. Linux builds run on the
`ubuntu-22.04` image on purpose — glibc 2.35 is the floor the binary imposes on
whoever installs it.

## Dry runs

Trigger the workflow manually from the Actions tab (`workflow_dispatch`). By default `dry_run` is `true`, so the build matrix runs without publishing. Set it to `false` to also publish.

## Adding a target

1. Add the `@substructure.ai/cli-<platform>` entry to the `optionalDependencies` of `packages/cli/package.json`.
2. Add a new matrix entry in `.github/workflows/release.yml`.
3. Add the target metadata to `scripts/build-cli-platform-package.mjs`.
