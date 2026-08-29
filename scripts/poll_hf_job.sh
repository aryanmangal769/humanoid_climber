#!/bin/zsh

set -u

readonly PROJECT_DIR="/Users/aryan.aryan/Desktop/Hum Climber"
readonly JOB_ID="iteratehack/6a93595a984507d9db4ec17b"
readonly JOB_URL="https://huggingface.co/jobs/iteratehack/6a93595a984507d9db4ec17b"
readonly LOG_FILE="$PROJECT_DIR/logs/job-monitor/hf-job.log"

cd "$PROJECT_DIR" || exit 1

output="$(UV_PROJECT_ENVIRONMENT=.venv-rl /opt/homebrew/bin/uv run hf jobs inspect "$JOB_ID" --json 2>&1)"
exit_code=$?
timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"

if (( exit_code != 0 )); then
  printf '[%s] Poll failed: %s\n' "$timestamp" "$output" >> "$LOG_FILE"
  exit "$exit_code"
fi

stage="$(printf '%s' "$output" | sed -n 's/.*"stage": "\([A-Z]*\)".*/\1/p' | head -n 1)"
message="$(printf '%s' "$output" | sed -n 's/.*"message": "\([^"]*\)".*/\1/p' | head -n 1)"
printf '[%s] %s%s\n' "$timestamp" "${stage:-UNKNOWN}" "${message:+ — $message}" >> "$LOG_FILE"

case "$stage" in
  COMPLETED)
    osascript -e 'display notification "Fine-tuning completed. The checkpoint is ready to evaluate." with title "Hum Climber"' >/dev/null
    ;;
  ERROR|CANCELED|DELETED)
    osascript -e "display notification \"Fine-tuning stopped: $stage. Check the Hugging Face job logs.\" with title \"Hum Climber\"" >/dev/null
    ;;
esac

printf '[%s] %s\n' "$timestamp" "$JOB_URL" >> "$LOG_FILE"
