#!/usr/bin/env bash
# Build + push the pre-baked nanoRT training image.
#
# Target arch is linux/amd64 (the H200 cluster). Best options, in order:
#   1. A linux/amd64 host (a build VM, or the cluster login node): native +
#      fast, no emulation. Plain `docker build` — this script auto-detects it.
#   2. An Apple-silicon Mac: needs a running daemon + buildx for --platform.
#      Works but the amd64 build runs under QEMU (slow: torch/CUDA download +
#      the `import torch` sanity check). OrbStack or Docker Desktop give both.
#
# Usage (image ref must match `image:` in plx_nanort_train.yaml):
#   IMAGE=gabrielmveloso/nanort:cu124-torch2.6 ./docker_build.sh
set -euo pipefail

IMAGE="${IMAGE:-gabrielmveloso/nanort:cu124-torch2.6}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: no reachable Docker daemon."
  echo "  - Mac:   start OrbStack / Docker Desktop (or 'colima start')."
  echo "  - Linux: start dockerd, or run this on a linux/amd64 build host."
  exit 1
fi

arch="$(uname -m)"
if docker buildx version >/dev/null 2>&1; then
  echo ">> buildx available: cross-building linux/amd64 and pushing $IMAGE"
  docker buildx build --platform linux/amd64 -t "$IMAGE" --push .
elif [ "$arch" = "x86_64" ] || [ "$arch" = "amd64" ]; then
  echo ">> native amd64 host, no buildx: plain build + push $IMAGE"
  docker build -t "$IMAGE" .
  docker push "$IMAGE"
else
  echo "ERROR: on '$arch' without buildx — cannot target linux/amd64."
  echo "  Install buildx (OrbStack / Docker Desktop include it) or build on a"
  echo "  linux/amd64 host."
  exit 1
fi

echo
echo "Pushed: $IMAGE"
