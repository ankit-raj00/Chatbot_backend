#!/bin/bash
# Upsert KEY=VALUE pairs from a values file into an env file, in place.
# Used by deploy.yml so secrets/vars managed in GitHub can be layered onto
# the .env that's otherwise just copied forward from the previous deploy,
# without hand-editing the box. Existing keys not present in the values
# file are left untouched.
set -e
ENV_FILE="$1"
NEW_VALUES="$2"

while IFS='=' read -r key val; do
  [ -z "$key" ] && continue
  case "$key" in \#*) continue ;; esac
  grep -v "^${key}=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
  mv "$ENV_FILE.tmp" "$ENV_FILE"
  echo "${key}=${val}" >> "$ENV_FILE"
done < "$NEW_VALUES"
