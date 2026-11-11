#!/bin/sh
# Mirrors mid-sync can serve an empty repomd.xml, which dnf rejects without trying another mirror
for attempt in 1 2 3; do
  dnf "$@" && exit 0
  [ "$attempt" = 3 ] && exit 1
  sleep 60
done
