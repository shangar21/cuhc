#!/usr/bin/env bash
# Profiles the pairwise distance kernel with Nsight Compute.
#
# Needs GPU performance counters, which this machine restricts to root
# (RmProfilingAdminOnly=1), so run it with sudo:
#   sudo scripts/profile_distance.sh
#
# Writes build/profile/<metric>_d<numFeatures>.ncu-rep (open in ncu-ui) and a
# text version next to it.
set -euo pipefail

cd "$(dirname "$0")/.."
ncu=${NCU:-/usr/local/cuda/bin/ncu}
outDir=build/profile
mkdir -p "$outDir"

# metric numFeatures: small d (write-bound), medium d, large d (compute-bound),
# and the Minkowski case the docs list as unexplained.
cases=("euclidean 8" "euclidean 64" "euclidean 512" "minkowski3 8")

for case in "${cases[@]}"; do
  read -r metric numFeatures <<<"$case"
  name="${metric}_d${numFeatures}"
  echo "== $name"
  "$ncu" --set full --import-source yes \
    --kernel-name regex:pairwiseDistanceKernel --launch-count 1 \
    --force-overwrite --export "$outDir/$name" \
    ./build/bench_distance 16384 "$numFeatures" "$metric" > /dev/null
  "$ncu" --import "$outDir/$name.ncu-rep" --page details --print-units base \
    > "$outDir/$name.txt"
done

# Files created as root should still be readable/editable by the normal user.
if [[ -n "${SUDO_USER:-}" ]]; then
  chown -R "$SUDO_USER" "$outDir"
fi
echo "Reports in $outDir"
