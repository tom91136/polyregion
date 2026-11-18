#!/bin/sh
# Mirrors mid-sync can serve an empty repomd.xml, which dnf rejects without trying another mirror
attempt=1
while :; do
  dnf "$@" && exit 0
  [ "$attempt" -eq 6 ] && exit 1
  dnf clean expire-cache >/dev/null 2>&1 || true
  delay=$((attempt * 60))
  [ "$delay" -gt 300 ] && delay=300
  echo "dnf failed (attempt $attempt); refreshing metadata and retrying in ${delay}s" >&2
  sleep "$delay"
  attempt=$((attempt + 1))
done
