#!/usr/bin/env bash
set -euo pipefail

for f in disp-*/forces.dat; do
  # If the last line is empty, delete it in-place
  sed -i '${/^$/d}' "$f"
done

echo "✔ Trailing blank lines removed."