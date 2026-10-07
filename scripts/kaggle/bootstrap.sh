#!/usr/bin/env bash
# =============================================================================
# RALF on Kaggle / Colab -- environment bootstrap
#
# Usage (run from anywhere; it cd's to the repo root):
#   source scripts/kaggle/bootstrap.sh            # env only (fast, repeatable)
#   source scripts/kaggle/bootstrap.sh --install  # env + pip install deps
#
# After this, the repo's own bash scripts work unmodified, e.g.:
#   bash scripts/run_job/eval_inference.sh 0 \
#        ./cache/training_logs/ralf_refinement_cgl refinement cgl
#   bash scripts/run_job/end_to_end.sh 0 ralf cgl refinement
# =============================================================================

# 1) cd to repo root (this file lives at <repo>/scripts/kaggle/bootstrap.sh)
_THIS="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd "$(dirname "$_THIS")/../.." && pwd)"
cd "$REPO_ROOT" || return 1

# 2) Locate the 13 GB `cache/` and symlink read-only subdirs into a writable cache/.
#    On Kaggle, upload `cache/` once as a Kaggle Dataset -> /kaggle/input/<slug>/cache
#    On Colab, put `cache/` on Drive -> /content/drive/MyDrive/RALF/cache
if [ ! -e "cache/dataset" ]; then
    for cand in /kaggle/input/*/cache /kaggle/input/* /content/drive/MyDrive/RALF/cache /content/cache ./cache; do
        if [ -d "$cand/dataset" ] && [ -d "$cand/PRECOMPUTED_WEIGHT_DIR" ]; then
            echo "Linking cache from: $cand"
            mkdir -p cache
            ln -sfn "$cand/dataset"                 cache/dataset
            ln -sfn "$cand/PRECOMPUTED_WEIGHT_DIR"  cache/PRECOMPUTED_WEIGHT_DIR
            [ -d "$cand/eval_gt_features" ] && ln -sfn "$cand/eval_gt_features" cache/eval_gt_features
            [ -f "$cand/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt" ] \
                && ln -sfn "$cand/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt" cache/
            [ -d "$cand/training_logs" ] && ln -sfn "$cand/training_logs" cache/training_logs
            break
        fi
    done
fi

# 3) Optional dependency install. We deliberately DO NOT touch torch/torchvision:
#    Kaggle/Colab already ship a CUDA build of torch 2.x (works with the patched
#    `torch.load(..., weights_only=False)` calls). TensorFlow is not imported by
#    the code, so it is skipped.
if [ "$1" = "--install" ]; then
    echo "Installing RALF runtime dependencies (torch left untouched)..."
    pip install -q \
        "datasets>=2.13.0" "omegaconf>=2.3.0" "hydra-core>=1.3.2" \
        "einops>=0.6.1" "timm>=0.9.5" "rich>=13.5.2" \
        "faiss-cpu>=1.7.4" "prdc>=0.2" "pytorch-fid>=0.3.0" \
        "python-json-logger>=2.0.7" "seaborn>=0.12.2" \
        "opencv-python-headless>=4.8.0.74" "fsspec" \
        "protobuf<=3.20.3" "multiprocess" "scipy<=1.10.1" "pyyaml>=6.0.1"
    # dreamsim is only needed if you rebuild the retrieval backbone from scratch
    # (inference of the shipped checkpoints uses the precomputed retrieval indexes):
    # pip install -q dreamsim
fi

# 4) `poetry` shim: the repo scripts call `poetry run python ...`. Provide a tiny
#    shim so they run against the system interpreter. Both a shell function
#    (for `source`d scripts) and a PATH executable (for `bash sub.sh` subshells).
poetry() { if [ "$1" = "run" ]; then shift; fi; "$@"; }
export -f poetry 2>/dev/null || true

mkdir -p ./.kaggle_bin
cat > ./.kaggle_bin/poetry <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "run" ]; then shift; fi
exec "$@"
EOF
chmod +x ./.kaggle_bin/poetry
export PATH="$PWD/.kaggle_bin:$PATH"

# 5) RALF environment variables (mirror scripts/bin/setup.sh)
export DATA_ROOT="$PWD/cache/dataset"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "RALF env ready."
echo "  REPO_ROOT=$REPO_ROOT"
echo "  DATA_ROOT=$DATA_ROOT"
python - <<'PY'
import sys
try:
    import torch
    print(f"  python={sys.version.split()[0]}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu={torch.cuda.get_device_name(0)}")
except Exception as e:  # pragma: no cover
    print("  torch not importable:", e)
PY
