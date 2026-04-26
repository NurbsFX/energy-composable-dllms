#!/usr/bin/env bash
# Watch a running RunPod pod for pipeline completion, download artefacts
# the moment they are ready, then run Phase 5 (the κ/CKA/MI vs deficit
# plots) locally on the downloaded data. Fire-and-forget: start this in
# a terminal on your laptop and walk away.
#
# Usage:
#   scripts/auto_download_artifacts.sh <pod_host> <pod_port> [destination] [poll_seconds]
#
# Defaults:
#   destination   = ~/Documents/composable-dllms-artifacts
#   poll_seconds  = 300 (5 min)
#
# Ctrl+C at any time downloads whatever is currently on the pod and
# exits without running the local plotting step. Useful if the pipeline
# crashed mid-way and you want to inspect the partial state.

set -euo pipefail

POD_HOST="${1:?usage: $0 <pod_host> <pod_port> [destination] [poll_seconds]}"
POD_PORT="${2:?usage: $0 <pod_host> <pod_port> [destination] [poll_seconds]}"
DEST="${3:-$HOME/Documents/composable-dllms-artifacts}"
INTERVAL="${4:-300}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS=(-p "$POD_PORT" -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

remote() { ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "$@"; }

# Locate the repo on the pod (RunPod puts it in /workspace by default,
# but the user might clone elsewhere).
REPO_PATH=""
for path in "/workspace/energy-composable-dllms" "/root/energy-composable-dllms"; do
    if remote "test -d $path" 2>/dev/null; then
        REPO_PATH="$path"
        break
    fi
done
if [ -z "$REPO_PATH" ]; then
    echo "Repo not found on pod under /workspace or /root. Did you clone it?" >&2
    exit 1
fi
DONE_MARKER="$REPO_PATH/artifacts/PIPELINE_DONE.txt"

do_download() {
    mkdir -p "$DEST"
    echo "Downloading $REPO_PATH/artifacts/ → $DEST/"
    rsync -avz --progress \
        -e "ssh ${SSH_OPTS[*]}" \
        "root@$POD_HOST:$REPO_PATH/artifacts/" \
        "$DEST/"
}

run_local_phase5() {
    echo
    echo "Running Phase 5 (κ/CKA/MI vs deficit plots) on downloaded artefacts..."
    local PY="$REPO_DIR/.venv/bin/python"
    [ -x "$PY" ] || PY="python"
    if "$PY" "$REPO_DIR/scripts/08_final_plots.py" \
        --gram-json "$DEST/gram_matrix.json" \
        --independence-json "$DEST/independence_metrics.json" \
        --js-json "$DEST/joint_satisfaction.json" \
        --out-dir "$DEST/plots"
    then
        echo
        echo "Done. Plots saved to $DEST/plots/"
    else
        echo "(Phase 5 plotting failed — likely missing one of the JSON inputs;"
        echo " inspect $DEST/ and re-run scripts/08_final_plots.py manually.)" >&2
    fi
}

trap 'echo; echo "Interrupted. Pulling current state and exiting."; do_download; exit 0' INT

echo "Watching $POD_HOST:$POD_PORT for $DONE_MARKER (poll every ${INTERVAL}s)."
echo "Ctrl+C to download immediately and skip Phase 5."

ELAPSED=0
while true; do
    if remote "test -f $DONE_MARKER" 2>/dev/null; then
        echo "Pipeline finished (after ~${ELAPSED}s of polling). Downloading..."
        do_download
        run_local_phase5
        echo
        echo "All done. Don't forget to terminate the pod once you have verified $DEST/."
        exit 0
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done
