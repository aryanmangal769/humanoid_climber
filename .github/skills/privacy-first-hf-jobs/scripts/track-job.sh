#!/bin/sh
set -eu

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 JOB_ID NAME NAMESPACE STATUS PURPOSE [CHECKPOINT]" >&2
  exit 2
fi

job_id="$1"
name="$2"
namespace="$3"
status="$4"
purpose="$5"
checkpoint="${6:-}"
tracker="${HF_JOB_TRACKER:-logs/hf-jobs.tsv}"

mkdir -p "$(dirname "$tracker")"
if [ ! -f "$tracker" ]; then
  printf 'timestamp_utc\tjob_id\tname\tnamespace\tstatus\tpurpose\tcheckpoint\n' > "$tracker"
fi

sanitize() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  "$(sanitize "$job_id")" \
  "$(sanitize "$name")" \
  "$(sanitize "$namespace")" \
  "$(sanitize "$status")" \
  "$(sanitize "$purpose")" \
  "$(sanitize "$checkpoint")" >> "$tracker"
chmod 600 "$tracker"
