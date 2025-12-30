#!/bin/sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <module_name>"
  exit 1
fi

module="$1"
runs=0

while true; do
  runs=$((runs + 1))

  # python3 -m "scripts.$module"
  mjpython -m "scripts.$module"
  rc=$?

  echo "exit_code=$rc"

  if [ "$rc" -eq 0 ]; then
    echo "Success. Total runs=$runs"
    break
  fi

  sleep 2
done
