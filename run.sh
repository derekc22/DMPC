#!/bin/sh

use_mjpython=0

if [ "$1" = "-q" ]; then
  use_mjpython=1
  shift
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 [-q] <module_name>"
  exit 1
fi

module="$1"
runs=0

while true; do
  runs=$((runs + 1))

  if [ "$use_mjpython" -eq 1 ]; then
    mjpython -m "scripts.$module"
  else
    python3 -m "scripts.$module"
  fi
  rc=$?

  echo "exit_code=$rc"

  if [ "$rc" -eq 0 ]; then
    echo "Success. Total runs=$runs"
    break
  fi

  sleep 2
done
