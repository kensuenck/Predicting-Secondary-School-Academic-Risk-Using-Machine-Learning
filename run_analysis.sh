#!/usr/bin/env bash
set -euo pipefail
python src/dissertation_analysis.py \
  --data data/student-mat.csv \
  --output-dir results_local
