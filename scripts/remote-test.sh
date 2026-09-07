#!/usr/bin/env bash
set -euo pipefail

platform=${1:-}
case "$platform" in
  windows|linux|macos) ;;
  *)
    printf 'usage: %s {windows|linux|macos}\n' "${0##*/}" >&2
    exit 64
    ;;
esac

platform_key=${platform^^}
host_var="REMOTE_TEST_${platform_key}_HOST"
path_var="REMOTE_TEST_${platform_key}_PATH"
host=${!host_var:-}
remote_path=${!path_var:-}

if [[ -z "$host" ]]; then
  printf '%s is required; define it in .env\n' "$host_var" >&2
  exit 64
fi
if [[ -z "$remote_path" ]]; then
  printf '%s is required; define it in .env\n' "$path_var" >&2
  exit 64
fi

# Host values are SSH destinations, not argument strings; leading options or whitespace could inject flags.
if [[ "$host" == -* || "$host" == *[$'\t\r\n ']* ]]; then
  printf '%s must be one SSH destination without options or whitespace\n' "$host_var" >&2
  exit 64
fi
# Sync destinations must be absolute project subdirectories. Root, home shorthand, and drive roots are unsafe with --delete.
if [[ "$remote_path" != /* || "$remote_path" == / || "$remote_path" == *[$'\t\r\n']* || "$remote_path" =~ ^/[^/]+/?$ || "$remote_path" =~ ^/[A-Za-z]/?$ || "$remote_path" =~ ^/cygdrive/[A-Za-z]/?$ ]]; then
  printf '%s must be an absolute POSIX project path, not a root or home shorthand\n' "$path_var" >&2
  exit 64
fi

if [[ "${REMOTE_TEST_VALIDATE_ONLY:-0}" == 1 ]]; then
  printf 'remote-test-%s configuration is valid\n' "$platform"
  exit 0
fi

for command in ssh rsync; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf '%s is required locally for remote test tasks\n' "$command" >&2
    exit 69
  fi
done

project_root=${MISE_PROJECT_ROOT:-$(pwd -P)}
if command -v cygpath >/dev/null 2>&1; then
  project_root=$(cygpath -u "$project_root")
fi
if [[ ! -f "$project_root/Cargo.toml" ]]; then
  printf 'MISE_PROJECT_ROOT does not identify the Cargo workspace\n' >&2
  exit 66
fi

# Bash %q preserves spaces and quotes without evaluating destination content in the remote POSIX shell.
printf -v remote_path_q '%q' "$remote_path"
remote_validation="set -eu; destination=$remote_path_q; mkdir -p \"\$destination\"; cd \"\$destination\"; resolved=\$(pwd -P); home=\$(cd \"\$HOME\" && pwd -P); remainder=\${resolved#/}; case \"\$resolved\" in /|\"\$home\"|/[A-Za-z]|/cygdrive/[A-Za-z]) printf '%s\\n' 'remote test destination resolves to an unsafe root' >&2; exit 64;; esac; case \"\$remainder\" in */*) :;; *) printf '%s\\n' 'remote test destination resolves to a top-level directory' >&2; exit 64;; esac"
ssh "$host" "$remote_validation"

if ! rsync \
  --archive \
  --compress \
  --delete \
  --protect-args \
  --include='/.env.example' \
  --exclude='.git/' \
  --exclude='target/' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.log' \
  --exclude='artifacts/' \
  --exclude='.artifacts/' \
  --exclude='logs/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='.zed/' \
  --exclude='.sessions/' \
  --exclude='.omp/' \
  --exclude='.mcp/' \
  -e ssh \
  -- \
  "$project_root/" \
  "$host:$remote_path/"
then
  printf 'source synchronization failed; remote tests were not started\n' >&2
  exit 74
fi
if ! ssh "$host" "test -f $remote_path_q/Cargo.toml"; then
  printf 'source synchronization did not produce a Cargo workspace; remote tests were not started\n' >&2
  exit 74
fi

remote_env="export EZ_GFX_EXAMPLE_HIDDEN=1 RUST_TEST_THREADS=1"
case "$platform" in
  windows)
    tests='cargo test -p ez-gfx-hal -p ez-gfx-backend-vulkan -p ez-gfx-backend-dx12 -p ez-gfx --lib'
    ;;
  linux)
    tests='cargo test -p ez-gfx-hal -p ez-gfx-backend-vulkan -p ez-gfx --lib'
    remote_env+=' VK_LOADER_LAYERS_DISABLE=~implicit~'
    if [[ -n "${REMOTE_TEST_LINUX_SLANG_DIR:-}" ]]; then
      printf -v slang_dir_q '%q' "$REMOTE_TEST_LINUX_SLANG_DIR"
      remote_env+=" SLANG_DIR=$slang_dir_q"
    fi
    ;;
  macos)
    tests='cargo test -p ez-gfx-hal -p ez-gfx-backend-metal -p ez-gfx --lib'
    ;;
esac

remote_command="set -eu; cd $remote_path_q; $remote_env; $tests"
if [[ "$platform" == macos ]]; then
  # Non-interactive macOS SSH omits Homebrew from PATH; a login zsh loads the configured toolchain.
  printf -v remote_command_q '%q' "$remote_command"
  ssh "$host" "zsh -lc $remote_command_q"
else
  ssh "$host" "$remote_command"
fi
