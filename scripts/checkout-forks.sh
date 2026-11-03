#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$repo_root/sources}"
mkdir -p "$destination"
while IFS=$'\t' read -r component repository branch revision; do
    [[ -z "$component" || "$component" == \#* ]] && continue
    if [[ -e "$destination/$component" ]]; then
        printf 'Destination already exists: %s\n' "$destination/$component" >&2
        exit 1
    fi
done < "$repo_root/revisions.tsv"
while IFS=$'\t' read -r component repository branch revision; do
    [[ -z "$component" || "$component" == \#* ]] && continue
    git clone --filter=blob:none --single-branch --branch "$branch" \
        "$repository" "$destination/$component"
    git -C "$destination/$component" checkout --detach "$revision"
    printf '%s: %s\n' "$component" "$revision"
done < "$repo_root/revisions.tsv"
