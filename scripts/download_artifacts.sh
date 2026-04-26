#!/usr/bin/env bash
# Pull artifacts/ from a running RunPod pod down to a local directory.
#
# Run this from your laptop, *before* terminating the pod (otherwise the
# Volume disk is wiped and the artefacts are gone). RunPod gives you the
# host and port in its "Connect → SSH over exposed TCP" panel.
#
# Usage:
#   scripts/download_artifacts.sh <pod_host> <pod_port> [destination]
#
# Default destination: ~/Documents/composable-dllms-artifacts/
#
# Examples:
#   scripts/download_artifacts.sh 213.181.123.45 12345
#   scripts/download_artifacts.sh ssh.runpod.io 22 ~/Documents/run-2026-04-26/

set -euo pipefail

POD_HOST="${1:?usage: $0 <pod_host> <pod_port> [destination]}"
POD_PORT="${2:?usage: $0 <pod_host> <pod_port> [destination]}"
DEST="${3:-$HOME/Documents/composable-dllms-artifacts}"

# Common pod paths to look for (we try the first one that exists).
REMOTE_CANDIDATES=(
    "/workspace/energy-composable-dllms/artifacts"
    "/root/energy-composable-dllms/artifacts"
)

REMOTE_PATH=""
for path in "${REMOTE_CANDIDATES[@]}"; do
    if ssh -p "$POD_PORT" -o StrictHostKeyChecking=accept-new "root@$POD_HOST" "test -d $path"; then
        REMOTE_PATH="$path"
        break
    fi
done

if [ -z "$REMOTE_PATH" ]; then
    echo "Could not find artifacts/ at any of: ${REMOTE_CANDIDATES[*]}" >&2
    echo "Adjust the script if you cloned the repo elsewhere on the pod." >&2
    exit 1
fi

mkdir -p "$DEST"
echo "Pulling $REMOTE_PATH → $DEST/ (rsync over SSH on port $POD_PORT)"

rsync -avz --progress \
    -e "ssh -p $POD_PORT -o StrictHostKeyChecking=accept-new" \
    "root@$POD_HOST:$REMOTE_PATH/" \
    "$DEST/"

echo
echo "Done. Artifacts saved to $DEST"
du -sh "$DEST" 2>/dev/null || true
ls -la "$DEST"
