#!/bin/sh
#
# submit.sh -- ARCH-COMP 2026 NLN repeatability entrypoint for PRoTECT.
#
# Builds the Docker image, runs the benchmarks, and extracts results.csv
# (plus per-benchmark JSON figures) from the container into ./results/.
#
# We use `docker cp` instead of a `-v` bind mount because bind-mount
# propagation is not reliable in every environment (the portal's
# rootless / docker-in-docker setup silently drops bind-mount writes).
# `docker cp` reads from the stopped container's filesystem and works
# in every Docker setup, including ours and the portal's.

set -e

cd "$(dirname "$0")"

rm -Rf result results
mkdir results

if [ ! -f mosek.lic ]; then
    echo "[submit.sh] WARNING: mosek.lic not found in $(pwd). The build will"
    echo "[submit.sh]          fall back to the open-source CVXOPT solver."
    echo "[submit.sh]          To use MOSEK, place mosek.lic in this folder"
    echo "[submit.sh]          (it is .gitignored, so it will never be committed)."
    : > mosek.lic
    SOLVER_OVERRIDE="-e PROTECT_SOLVER=cvxopt"
    PLACEHOLDER_LIC=1
    REAL_LIC_TO_PURGE=0
else
    SOLVER_OVERRIDE=""
    PLACEHOLDER_LIC=0
    REAL_LIC_TO_PURGE=1
fi

# Make sure the license is removed from disk no matter how the script
# exits (success, error, ctrl-C). The license is shipped inside the zip
# only so the portal's first build can pick it up; once it has been baked
# into the Docker image we don't want it left lying around in submit/.
purge_license() {
    if [ "$PLACEHOLDER_LIC" = "1" ] || [ "$REAL_LIC_TO_PURGE" = "1" ]; then
        rm -f mosek.lic
    fi
}
trap purge_license EXIT INT TERM

docker build -t protect-arch2026-nln .

if [ "$PLACEHOLDER_LIC" = "1" ]; then
    rm mosek.lic
    PLACEHOLDER_LIC=0
fi

# Use a named, non-removed container so we can `docker cp` the results
# out. `--rm` is incompatible with `docker cp` after exit on some Docker
# versions, so we manually `docker rm` once we've extracted the results.
CONTAINER_NAME=protect-arch2026-nln-run

# Clean up any orphaned container from a previous failed run.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --name "$CONTAINER_NAME" $SOLVER_OVERRIDE protect-arch2026-nln

# Extract everything in the container's result directory (results.csv +
# per-benchmark JSON figures) into the host's ./results/ folder.
docker cp \
    "$CONTAINER_NAME":/PRoTECT/ex/ARCH-COMP/2026-NLN/results/. \
    "$PWD/results/"

# Tidy up: remove the container and image so the host doesn't accumulate
# leftovers per submission.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rmi -f protect-arch2026-nln >/dev/null 2>&1 || true

# Sanity-check: verify the canonical results.csv is present, since the
# portal's verifier checks for exactly that file.
if [ ! -f results/results.csv ]; then
    echo "[submit.sh] ERROR: results/results.csv was not produced!" 1>&2
    exit 1
fi

echo "[submit.sh] Done. Results in $(pwd)/results/"
