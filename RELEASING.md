# Releasing

All three npm packages (`@substructure.ai/sdk`, `@substructure.ai/runtime`, `@substructure.ai/cli`) ship from a single tag-triggered GitHub Actions workflow at `.github/workflows/release.yml`.

## One-time setup

1. Create an automation-scoped npm access token under the `substructure.ai` org and add it as the `NPM_TOKEN` repository secret in GitHub (Settings → Secrets and variables → Actions).
2. Make sure each platform sub-package name is reserved by publishing the first version through CI, not by hand. npm requires the org to own them before optionalDependencies will resolve.

## Cutting a release

All packages release together at the same version.

```sh
# Bump versions across all package.jsons (one source of truth)
VERSION=0.1.1
for f in packages/*/package.json; do
  npm --prefix "$(dirname "$f")" version "$VERSION" --no-git-tag-version
done

git commit -am "release $VERSION"
git tag "v$VERSION"
git push origin main --tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

1. Builds the Rust CLI binary and the NAPI `.node` for each of:
   - `aarch64-apple-darwin`, `x86_64-apple-darwin`
   - `aarch64-unknown-linux-gnu` (cross-compiled), `x86_64-unknown-linux-gnu`
   - `x86_64-pc-windows-msvc`
2. For each target, builds an `@substructure.ai/cli-<platform>` package directory containing the binary.
3. Publishes everything in order: runtime platform packages → `@substructure.ai/runtime` → CLI platform packages → `@substructure.ai/cli` → `@substructure.ai/sdk`.

## Dry runs

Trigger the workflow manually from the Actions tab (`workflow_dispatch`). By default `dry_run` is `true`, so the build matrix runs without publishing. Set it to `false` to also publish.

## Adding a target

1. Add the triple to `napi.targets` in `packages/runtime/package.json`.
2. Add the corresponding `@substructure.ai/runtime-<platform>` and `@substructure.ai/cli-<platform>` entries to the `optionalDependencies` of `packages/runtime/package.json` and `packages/cli/package.json`.
3. Add a new matrix entry in `.github/workflows/release.yml`.
4. Add the target metadata to `scripts/build-cli-platform-package.mjs`.
