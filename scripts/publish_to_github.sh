#!/usr/bin/env bash
# One-shot: create the GitHub repo, push, and enable Pages so docs/index.html goes live.
# Requires the GitHub CLI (https://cli.github.com) and `gh auth login` done once.
set -euo pipefail

REPO="${1:-transition-timing}"
VIS="${2:-private}"   # pass "public" as the second argument to make it public

cd "$(dirname "$0")/.."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then git init; fi
git add -A
git commit -m "Optimal-stopping model for social transition timing" || true

gh repo create "$REPO" --"$VIS" --source=. --remote=origin --push

# Serve docs/ on GitHub Pages so the interactive chart is live.
gh api -X POST "repos/{owner}/$REPO/pages" \
  -f "source[branch]=main" -f "source[path]=/docs" >/dev/null && echo "Pages enabled." \
  || echo "Pages API call failed; enable it under Settings > Pages (branch: main, folder: /docs)."

OWNER=$(gh api user -q .login)
echo
echo "Repo:  https://github.com/$OWNER/$REPO"
echo "Chart: https://$OWNER.github.io/$REPO/   (takes a minute to build)"
