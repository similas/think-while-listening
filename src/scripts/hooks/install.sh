#!/bin/sh
# Install the secret-scan hooks (pre-commit, pre-merge-commit, pre-push) into
# each repository's OWN hooks directory, as stubs that call this checkout's
# wrappers by absolute path:
#
#   src/scripts/hooks/install.sh REPO [REPO ...]      (`make hooks` runs it)
#
# Not core.hooksPath: a relative hooksPath, or one into a working tree, points
# at nothing after a checkout of an older commit, and git then runs no hook and
# says nothing. A stub whose target is missing refuses instead (exit 2).
set -eu
src=$(cd "$(dirname "$0")" && pwd -P)
for repo in "$@"; do
  if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
    echo "hooks: skip $repo (not a git repository)"
    continue
  fi
  # Unset hooksPath FIRST: while it is set, --git-path hooks resolves to it.
  # One set at another scope (global, system, worktree) would win over the
  # stubs and is not this script's to remove: refuse.
  git -C "$repo" config --local --unset core.hooksPath || true
  if left=$(git -C "$repo" config --get core.hooksPath); then
    echo "hooks: core.hooksPath is still set for $repo ($left), at a scope above the repository; unset it, then rerun" >&2
    exit 2
  fi
  dir=$(git -C "$repo" rev-parse --path-format=absolute --git-path hooks)
  if [ "$(cd "$dir" 2>/dev/null && pwd -P || echo "$dir")" = "$src" ]; then
    echo "hooks: refusing to write stubs over the wrappers in $src" >&2
    exit 2
  fi
  mkdir -p "$dir"
  for h in pre-commit pre-merge-commit pre-push; do
    cat > "$dir/$h" <<EOF
#!/bin/sh
# Installed by \`make hooks\` (src/scripts/hooks/install.sh): the secret scan.
# If the checkout below no longer has the wrapper, refuse rather than skip.
[ -x "$src/$h" ] || { echo "secret-scan hook missing: $src/$h; run make hooks from a checkout that has it. Refusing." >&2; exit 2; }
[ "$src/$h" -ef "\$0" ] && { echo "secret-scan hook: this stub would call itself; refusing" >&2; exit 2; }
exec "$src/$h" "\$@"
EOF
    chmod +x "$dir/$h"
  done
  echo "hooks: $repo -> $src"
done
