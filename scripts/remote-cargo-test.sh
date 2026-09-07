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

for package in "$@"; do
  if [[ ! "$package" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'invalid Cargo package name: %s\n' "$package" >&2
    exit 64
  fi
done

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/remote-cargo-test.XXXXXX")
active_test_pid=
watchdog_pid=

signal_group() {
  local signal=$1
  local pid=$2

  # Monitor mode gives each background job its own process group in Bash,
  # including macOS's bundled Bash. Targeting it also terminates test children.
  kill "-$signal" -- "-$pid" 2>/dev/null || kill "-$signal" "$pid" 2>/dev/null || true
}

stop_active_jobs() {
  if [[ -n "$active_test_pid" ]]; then
    signal_group TERM "$active_test_pid"
  fi
  if [[ -n "$watchdog_pid" ]]; then
    signal_group TERM "$watchdog_pid"
  fi
  if [[ -n "$active_test_pid" || -n "$watchdog_pid" ]]; then
    sleep 1
  fi
  if [[ -n "$active_test_pid" ]]; then
    signal_group KILL "$active_test_pid"
  fi
  if [[ -n "$watchdog_pid" ]]; then
    signal_group KILL "$watchdog_pid"
  fi

  if [[ -n "$active_test_pid" ]]; then
    wait "$active_test_pid" 2>/dev/null || true
  fi
  if [[ -n "$watchdog_pid" ]]; then
    wait "$watchdog_pid" 2>/dev/null || true
  fi

  active_test_pid=
  watchdog_pid=
}

cleanup() {
  stop_active_jobs
  rm -rf -- "$work_dir"
}

on_signal() {
  local status=$1
  trap - HUP INT TERM
  stop_active_jobs
  exit "$status"
}

trap cleanup EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

run_timed_test() {
  local package=$1
  local test_name=$2
  local marker=$work_dir/timed-out
  local status

  rm -f -- "$marker"
  # Monitor mode assigns each background job a process group. Disable it after
  # both launches so Bash does not print job-control notices into test output.
  set -m
  cargo test -p "$package" --lib -- --exact "$test_name" &
  active_test_pid=$!

  (
    sleep "$timeout_seconds"
    : >"$marker"
    signal_group TERM "$active_test_pid"
    sleep 2
    signal_group KILL "$active_test_pid"
  ) &
  watchdog_pid=$!
  set +m

  set +e
  wait "$active_test_pid"
  status=$?
  set -e
  active_test_pid=

  if [[ -f "$marker" ]]; then
    wait "$watchdog_pid" 2>/dev/null || true
    watchdog_pid=
    printf 'remote cargo test timed out after %s seconds: package=%s test=%s\n' \
      "$timeout_seconds" "$package" "$test_name" >&2
    return 124
  fi

  signal_group TERM "$watchdog_pid"
  wait "$watchdog_pid" 2>/dev/null || true
  watchdog_pid=

  if [[ "$status" -ne 0 ]]; then
    printf 'remote cargo test failed with status %s: package=%s test=%s\n' \
      "$status" "$package" "$test_name" >&2
    return "$status"
  fi
}

total_tests=0
for package in "$@"; do
  listing=$work_dir/list
  if ! cargo test -p "$package" --lib -- --list --format terse >"$listing"; then
    printf 'failed to compile or list lib tests: package=%s\n' "$package" >&2
    exit 1
  fi

  tests=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      *': test')
        test_name=${line%: test}
        if [[ -z "$test_name" || "$test_name" == [[:space:]]* || "$test_name" == *[[:space:]] || "$test_name" == *[[:cntrl:]]* ]]; then
          printf 'malformed libtest name from package %s\n' "$package" >&2
          exit 65
        fi
        tests[${#tests[@]}]=$test_name
        ;;
      *': benchmark')
        ;;
      *)
        printf 'malformed cargo test listing for package %s\n' "$package" >&2
        exit 65
        ;;
    esac
  done <"$listing"

  if [[ "${#tests[@]}" -eq 0 ]]; then
    printf 'no lib tests listed for expected package: %s\n' "$package" >&2
    exit 65
  fi

  printf 'remote cargo tests: package=%s tests=%s timeout=%ss\n' \
    "$package" "${#tests[@]}" "$timeout_seconds"
  for test_name in "${tests[@]}"; do
    run_timed_test "$package" "$test_name"
    total_tests=$((total_tests + 1))
  done
done

printf 'remote cargo tests passed: packages=%s tests=%s\n' "$#" "$total_tests"
