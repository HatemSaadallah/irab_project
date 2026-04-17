#!/bin/bash
# Convenience wrapper for common training configs.
#
# Usage:
#     ./scripts/train.sh small              # 5M params, CPU-friendly
#     ./scripts/train.sh medium             # 60M params, GPU required
#     ./scripts/train.sh large              # 300M params, A100 required
#     ./scripts/train.sh medium --resume runs/medium/epoch_5.pt

set -euo pipefail

SIZE=${1:-medium}
shift || true

case "$SIZE" in
    small|medium|large)
        CONFIG="configs/model_${SIZE}.yaml"
        ;;
    *)
        echo "Unknown size: $SIZE. Use small/medium/large."
        exit 1
        ;;
esac

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG"
    exit 1
fi

echo "Training with $CONFIG"
echo "Extra args: $@"
echo

python -m irab_tashkeel.training.cli --config "$CONFIG" "$@"
