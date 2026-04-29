#!/usr/bin/env bash
set -e

echo "Running realistic benchmark..."

cargo run -- benchmark-native \
  --base demo/base.txt \
  --blob demo/blob.bin \
  --steps 40 \
  --merge-policy adaptive \
  --merge-threshold 1.4 \
  --base-target-tokens 1200 \
  --delta-variance 0.25 \
  --workdir demo/native_benchmark_realistic \
  --input-cost-per-m 5.00 \
  --out demo/native_benchmark_realistic.json

echo ""
echo "Done. Output:"
cat demo/native_benchmark_realistic.json | jq '.savings'
