#!/usr/bin/env bash
# Run from any directory. All downloads and environments remain in this repo.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$root/inference_workspace/groove2groove"
mkdir -p "$work"
export TMPDIR="$work/tmp"
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$work/mpl"
if [[ ! -x "$work/bin/micromamba" ]]; then
    curl -L --fail https://micro.mamba.pm/api/micromamba/linux-64/2.9.0 \
        -o "$work/micromamba.tar.bz2"
    python3 - "$work" <<'PY'
import pathlib
import sys
import tarfile
work = pathlib.Path(sys.argv[1])
with tarfile.open(work / "micromamba.tar.bz2") as archive:
    archive.extract("bin/micromamba", str(work))
PY
fi
if [[ ! -d "$work/upstream" ]]; then
    git clone https://github.com/cifkao/groove2groove.git "$work/upstream"
    git -C "$work/upstream" checkout fe016926e62973a66f0b49d697374effe0f9e627
fi
if [[ ! -d "$work/museflow" ]]; then
    git clone https://github.com/cifkao/museflow.git "$work/museflow"
    git -C "$work/museflow" checkout e5642fd48687a6d3165e213eaab79d5d78d4c0d5
fi
if [[ ! -x "$work/env/bin/python" ]]; then
    "$work/bin/micromamba" create -y --no-rc -r "$work/mamba" \
        -p "$work/env" -c conda-forge python=3.7.12 pip
fi
"$work/env/bin/python" -m pip install --no-cache-dir setuptools==67.8.0 wheel==0.42.0
"$work/env/bin/python" -m pip install --no-cache-dir \
    -r "$root/tools/groove2groove/requirements.lock.txt"
"$work/env/bin/python" -m pip install --no-cache-dir --no-deps \
    "$work/museflow" "$work/upstream/code"
if [[ ! -d "$work/checkpoints/v01" ]]; then
    curl -L --fail https://groove2groove.telecom-paris.fr/data/checkpoints/v01.zip \
        -o "$work/v01.zip"
    python3 - "$work" <<'PY'
import pathlib
import sys
import zipfile
work = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(work / "v01.zip") as archive:
    archive.extractall(work / "checkpoints")
PY
fi
if [[ ! -f "$work/checkpoints/v01/model.yaml" ]]; then
    cp "$work/upstream/experiments/v01/model.yaml" "$work/checkpoints/v01/model.yaml"
fi
"$work/env/bin/python" -m pip check
