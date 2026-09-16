#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  printf 'usage: %s <package> [package ...]\n' "${0##*/}" >&2
  exit 64
fi

timeout_seconds=${REMOTE_TEST_TIMEOUT_SECONDS:-60}
if [[ ! "$timeout_seconds" =~ ^[0-9]+$ || "${#timeout_seconds}" -gt 4 ]] || (( 10#$timeout_seconds < 1 || 10#$timeout_seconds > 3600 )); then
  printf 'REMOTE_TEST_TIMEOUT_SECONDS must be an integer from 1 through 3600\n' >&2
  exit 64
fi
timeout_seconds=$((10#$timeout_seconds))

package_args=()
for package in "$@"; do
  if [[ ! "$package" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'invalid Cargo package name: %s\n' "$package" >&2
    exit 64
  fi
  package_args+=(-p "$package")
done

config_file=$(mktemp .remote-nextest.XXXXXX)
trap 'rm -f -- "$config_file"' EXIT
cat >"$config_file" <<EOF
[profile.remote]
fail-fast = true
test-threads = 1
slow-timeout = { period = "${timeout_seconds}s", terminate-after = 1 }
EOF

printf 'remote nextest: packages=%s timeout=%ss\n' "$#" "$timeout_seconds"
cargo nextest run \
  --config-file "$config_file" \
  --profile remote \
  "${package_args[@]}" \
  --lib
printf 'remote nextest passed: packages=%s\n' "$#"
