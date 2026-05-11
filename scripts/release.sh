#!/usr/bin/env bash
# Cut a release: bump versions in all package.jsons, commit, tag, push.
# The push of the v<version> tag triggers .github/workflows/release.yml.
#
# Usage:
#   scripts/release.sh <version>
#   scripts/release.sh 0.1.0
#   scripts/release.sh patch    # auto-bump from current version
#   scripts/release.sh minor
#   scripts/release.sh major

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version|patch|minor|major>" >&2
  exit 1
fi

ARG="$1"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Working tree must be clean.
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit or stash first." >&2
  git status --short >&2
  exit 1
fi

# Must be on main.
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
  echo "Refusing to release from branch '$BRANCH' (expected: main)." >&2
  exit 1
fi

git fetch origin --tags

CURRENT=$(node -p "require('./packages/cli/package.json').version")

case "$ARG" in
  patch|minor|major)
    VERSION=$(node -e "
      const [maj, min, pat] = process.argv[1].split('.').map(Number);
      const kind = process.argv[2];
      const bumped = kind === 'major' ? [maj + 1, 0, 0]
        : kind === 'minor' ? [maj, min + 1, 0]
        : [maj, min, pat + 1];
      console.log(bumped.join('.'));
    " "$CURRENT" "$ARG")
    ;;
  *)
    VERSION="$ARG"
    ;;
esac

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid version: $VERSION" >&2
  exit 1
fi

TAG="v$VERSION"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag $TAG already exists." >&2
  exit 1
fi

echo "Bumping $CURRENT -> $VERSION"

for pkg in packages/sdk packages/cli packages/runtime; do
  npm --prefix "$pkg" version "$VERSION" --no-git-tag-version --allow-same-version >/dev/null
done

# Keep optionalDependencies in cli + runtime aligned with the new version.
node -e "
  const fs = require('fs');
  for (const f of ['packages/cli/package.json', 'packages/runtime/package.json']) {
    const j = JSON.parse(fs.readFileSync(f, 'utf8'));
    if (j.optionalDependencies) {
      for (const k of Object.keys(j.optionalDependencies)) {
        if (k.startsWith('@substructure.ai/')) j.optionalDependencies[k] = process.argv[1];
      }
    }
    fs.writeFileSync(f, JSON.stringify(j, null, 2) + '\n');
  }
" "$VERSION"

git add packages/sdk/package.json packages/cli/package.json packages/runtime/package.json
git commit -m "release $TAG"
git tag -a "$TAG" -m "$TAG"

echo
echo "Created commit and tag $TAG. Review with:"
echo "  git show $TAG"
echo
echo "To trigger the release workflow:"
echo "  git push origin main && git push origin $TAG"
