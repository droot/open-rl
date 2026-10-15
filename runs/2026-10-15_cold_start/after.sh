#!/usr/bin/env bash
# The matrix again on the fixed images (coldstart1-dca985a): first run per
# node pair pulls the new tag (cold image), the next two find it cached.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PATH:$HOME/.local/bin
R=runs/2026-10-15_cold_start
run() { scripts/cold_start_case.sh "$R/$1" "$2" "$3" "$4" > "$R/$1.run.log" 2>&1; echo "$1 DONE=$? $(date -u +%T)" >> "$R/after-progress.txt"; }
run after-A1-0.5b-l4-cold   Qwen/Qwen2.5-0.5B lvw8 2208
run after-B1-0.5b-l4-cached Qwen/Qwen2.5-0.5B lvw8 2208
run after-B2-0.5b-l4-cached Qwen/Qwen2.5-0.5B lvw8 2208
run after-C1-8b-h100-cold   Qwen/Qwen3-8B 1g3a 0853
run after-D1-8b-h100-cached Qwen/Qwen3-8B 1g3a 0853
run after-D2-8b-h100-cached Qwen/Qwen3-8B 1g3a 0853
echo "ALL DONE $(date -u +%T)" >> "$R/after-progress.txt"
