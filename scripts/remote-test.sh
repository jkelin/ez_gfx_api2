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

# Local rsync comes in two flavors: a POSIX (Cygwin/MSYS) rsync that speaks
# POSIX paths and execs `ssh` itself, and a native Windows rsync that needs
# Windows-form paths for both the remote shell and the source tree.
rsync_bin=$(command -v rsync)
native_rsync=0
if [[ -n "${REMOTE_TEST_RSYNC_FLAVOR:-}" ]]; then
  case "$REMOTE_TEST_RSYNC_FLAVOR" in
    native) native_rsync=1 ;;
    posix) native_rsync=0 ;;
    *)
      printf 'REMOTE_TEST_RSYNC_FLAVOR must be "native" or "posix"\n' >&2
      exit 64
      ;;
  esac
elif command -v cygpath >/dev/null 2>&1; then
  case "$(cygpath -u "$rsync_bin")" in
    /usr/bin/*|/bin/*) native_rsync=0 ;;
    *) native_rsync=1 ;;
  esac
fi

protect_args=--protect-args
rsync_destination="$host:$remote_path/"
if [[ "$platform" == macos ]]; then
  protect_args=
  # Old Apple rsync passes the destination through the remote shell without
  # protecting spaces or glob characters.
  printf -v remote_rsync_path_q '%q' "$remote_path/"
  rsync_destination="$host:$remote_rsync_path_q"
fi

ssh_opt=ssh
rsync_source=$project_root/
if [[ "$native_rsync" == 1 ]]; then
  printf -v ssh_opt '"%s"' "$(cygpath -m "$(command -v ssh)")"
  # This rsync build treats drive-letter sources as remote specs. Running
  # from the tree and using ./ preserves the native-tool boundary safely.
  cd "$project_root"
  rsync_source=./
fi

compress_args=--compress
if [[ "$platform" == macos ]]; then
  # Apple’s bundled rsync can terminate a compressed native-rsync stream
  # before all files arrive; correctness matters more than transfer size.
  compress_args=
fi

if ! rsync \
  --archive \
  ${compress_args:+$compress_args} \
  --delete \
  ${protect_args:+$protect_args} \
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
  -e "$ssh_opt" \
  -- \
  "$rsync_source" \
  "$rsync_destination"
then
  printf 'source synchronization failed; remote tests were not started\n' >&2
  exit 74
fi
remote_env="export EZ_GFX_EXAMPLE_HIDDEN=1 RUST_TEST_THREADS=1 PATH=\"\$HOME/.cargo/bin:\$PATH\""
slang_setup=""
metadata_command="set -eu; cd $remote_path_q; $remote_env; cargo metadata --no-deps --format-version 1 >/dev/null"
if [[ "$platform" == macos ]]; then
  printf -v metadata_command_q '%q' "$metadata_command"
  if ! ssh "$host" "zsh -lc $metadata_command_q"; then
    printf 'remote workspace metadata validation failed; remote tests were not started\n' >&2
    exit 74
  fi
elif ! ssh "$host" "$metadata_command"; then
  printf 'remote workspace metadata validation failed; remote tests were not started\n' >&2
  exit 74
fi

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
    if [[ -n "${REMOTE_TEST_LINUX_VK_DRIVER_FILES:-}" ]]; then
      printf -v icd_files_q '%q' "$REMOTE_TEST_LINUX_VK_DRIVER_FILES"
      remote_env+=" VK_ICD_FILENAMES=$icd_files_q"
    fi
    ;;
  macos)
    tests='cargo test -p ez-gfx-hal -p ez-gfx-backend-metal -p ez-gfx --lib'
    if [[ -n "${REMOTE_TEST_MACOS_SLANG_DIR:-}" ]]; then
      printf -v slang_dir_q '%q' "$REMOTE_TEST_MACOS_SLANG_DIR"
      remote_env+=" SLANG_DIR=$slang_dir_q"
    fi
    # Respect complete compiler configuration already present on the host.
    # Otherwise normalize explicit and discovered prefixes for both supported
    # header layouts.
    slang_setup='if [ -z "${SLANG_DIR:-}" ] && [ -n "${VULKAN_SDK:-}" ]; then
        :
      elif [ -z "${SLANG_DIR:-}" ] && [ -n "${SLANG_INCLUDE_DIR:-}" ] && [ -n "${SLANG_LIB_DIR:-}" ]; then
        :
      else
        slang_prefix=${SLANG_DIR:-}
        if [ -z "$slang_prefix" ] && command -v brew >/dev/null 2>&1; then
          slang_prefix=$(brew --prefix slang 2>/dev/null || true)
        fi
        if { [ -z "$slang_prefix" ] || { [ ! -f "$slang_prefix/include/slang.h" ] && [ ! -f "$slang_prefix/include/slang/slang.h" ]; }; } && command -v slangc >/dev/null 2>&1; then
          slangc_bin=$(command -v slangc)
          slang_depth=0
          while [ -L "$slangc_bin" ] && [ "$slang_depth" -lt 10 ]; do
            slang_link=$(readlink "$slangc_bin")
            case "$slang_link" in
            /*) slangc_bin=$slang_link ;;
            *) slangc_bin=$(dirname "$slangc_bin")/$slang_link ;;
            esac
            slang_depth=$((slang_depth + 1))
          done
          slang_prefix=$(dirname "$(dirname "$slangc_bin")")
        fi
        slang_have_lib=0
        if [ -n "$slang_prefix" ] && { [ -f "$slang_prefix/lib/libslang.dylib" ] || [ -f "$slang_prefix/lib/libslang.so" ]; }; then
          slang_have_lib=1
        fi
        if [ "$slang_have_lib" = 1 ] && [ -f "$slang_prefix/include/slang.h" ]; then
          export SLANG_DIR="$slang_prefix"
        elif [ "$slang_have_lib" = 1 ] && [ -f "$slang_prefix/include/slang/slang.h" ]; then
          export SLANG_DIR="$slang_prefix"
          export SLANG_INCLUDE_DIR="$slang_prefix/include/slang"
        else
          printf "%s\n" "Slang toolkit not found on the macOS host; install it (brew install slang) or set REMOTE_TEST_MACOS_SLANG_DIR in .env" >&2
          exit 69
        fi
      fi'
    ;;
esac

remote_command="set -eu; cd $remote_path_q; $remote_env; ${slang_setup:+$slang_setup; }$tests"
if [[ "$platform" == macos ]]; then
  # Non-interactive macOS SSH omits Homebrew from PATH; a login zsh loads the configured toolchain.
  printf -v remote_command_q '%q' "$remote_command"
  ssh "$host" "zsh -lc $remote_command_q"
else
  ssh "$host" "$remote_command"
fi
