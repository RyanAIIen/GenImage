#!/usr/bin/env bash
set -euo pipefail

RAW="/nfs/turbo/umd-anglial/GenImageDetector/raw_dataset"
GEN_GLOB="${1:-*}"   # e.g., "Stable*1.4*" or "Midjourney*"

# pick first existing dir from a list of candidates (globs allowed)
pick_first_dir () {
  for pat in "$@"; do
    for p in $pat; do
      if [ -d "$p" ]; then echo "$p"; return 0; fi
    done
  done
  return 1
}

GEN_DIR=$(find "$RAW" -maxdepth 1 -mindepth 1 -type d -iname "$GEN_GLOB" | head -n1)
if [ -z "${GEN_DIR:-}" ]; then
  echo "No generator directory matching pattern: $GEN_GLOB under $RAW"
  exit 1
fi
echo "Using generator directory: $GEN_DIR"

TR_REAL=$(pick_first_dir \
  "$GEN_DIR/train/nature" \
  "$GEN_DIR/train"/*/nature \
  "$GEN_DIR/train/nature/images" \
  "$GEN_DIR/train"/*/nature/images)
TR_FAKE=$(pick_first_dir \
  "$GEN_DIR/train/ai" \
  "$GEN_DIR/train"/*/ai \
  "$GEN_DIR/train/ai/images" \
  "$GEN_DIR/train"/*/ai/images)
VA_REAL=$(pick_first_dir \
  "$GEN_DIR/val/nature" \
  "$GEN_DIR/val"/*/nature \
  "$GEN_DIR/val/nature/images" \
  "$GEN_DIR/val"/*/nature/images)
VA_FAKE=$(pick_first_dir \
  "$GEN_DIR/val/ai" \
  "$GEN_DIR/val"/*/ai \
  "$GEN_DIR/val/ai/images" \
  "$GEN_DIR/val"/*/ai/images)

for v in TR_REAL TR_FAKE VA_REAL VA_FAKE; do
  if [ -z "${!v:-}" ]; then echo "Could not find path for $v"; exit 2; fi
done

echo "Train/real  -> $TR_REAL"
echo "Train/fake  -> $TR_FAKE"
echo "Val/real    -> $VA_REAL"
echo "Val/fake    -> $VA_FAKE"

# Relink dataset/ to the found paths
rm -rf dataset
mkdir -p dataset/train dataset/val
ln -snf "$TR_REAL" dataset/train/real
ln -snf "$TR_FAKE" dataset/train/fake
ln -snf "$VA_REAL" dataset/val/real
ln -snf "$VA_FAKE" dataset/val/fake

# Quick counts to confirm there are images
count_imgs () { find -L "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.tif' -o -iname '*.bmp' \) | wc -l; }
echo "Counts:"
echo "  train/real: $(count_imgs dataset/train/real)"
echo "  train/fake: $(count_imgs dataset/train/fake)"
echo "  val/real:   $(count_imgs dataset/val/real)"
echo "  val/fake:   $(count_imgs dataset/val/fake)"
