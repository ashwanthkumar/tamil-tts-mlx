#!/usr/bin/env bash
# Overnight orchestrator: wait for the current 120k MLX run to finish, then resume/extend
# from its last checkpoint toward a higher target so the GPU keeps training while the user sleeps.
# All checkpoints are saved every 2k steps, so stopping anywhere and picking the best is free.
set -u
cd "$(dirname "$0")/.."
RUN_DIR="runs_mlx/tamil_mlx"
TARGET="${1:-500000}"

echo "[overnight] $(date) waiting for current 120k run to finish..."
# Wait until no training process for this run is alive.
while pgrep -f "tamiltts.mlx.train" >/dev/null; do
  sleep 60
done
echo "[overnight] $(date) current run ended."

# The original run used pre-resume code (no state file); seed step from the newest checkpoint
# so the resume continues numbering instead of restarting at 0 (which would clobber checkpoints).
if [ ! -f "$RUN_DIR/latest_state.json" ]; then
  LATEST=$(ls "$RUN_DIR"/ckpt_*.safetensors 2>/dev/null | sort | tail -1)
  STEP=$(echo "$LATEST" | grep -oE '[0-9]+' | tail -1)
  STEP=${STEP:-0}
  echo "{\"step\": $STEP}" > "$RUN_DIR/latest_state.json"
  echo "[overnight] seeded latest_state.json at step $STEP"
fi

echo "[overnight] $(date) resuming -> target $TARGET steps"
exec uv run python -m tamiltts.mlx.train \
  --data data/mlx --out runs_mlx --run tamil_mlx \
  --steps "$TARGET" --batch 16 --layers 4 --d_model 256 --max_frames 1200 \
  --log_every 50 --save_every 2000 \
  --resume "$RUN_DIR"
