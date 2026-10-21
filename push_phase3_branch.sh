#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/anay-28/learned-icbsh-mapf.git"
BRANCH_NAME="${1:-phase3-cl-icbsh}"

# Sync any updated files from the extracted archive into the current git repository
if [ ! -d ".git" ]; then
    echo "[1/4] Initializing git repository and fetching main from ${REPO_URL}..."
    git init
    git remote add origin "${REPO_URL}" 2>/dev/null || git remote set-url origin "${REPO_URL}"
    git fetch origin main
    git checkout -b "${BRANCH_NAME}" FETCH_HEAD || git checkout -b "${BRANCH_NAME}"
else
    echo "[1/4] Creating and switching to branch '${BRANCH_NAME}'..."
    git remote set-url origin "${REPO_URL}" 2>/dev/null || git remote add origin "${REPO_URL}"
    git checkout -B "${BRANCH_NAME}"
fi

echo "[2/4] Staging Phase 3 CL-ICBS-h core upgrades, ranker models, and benchmark results..."
git add download_data.py \
        mapf/grid.py \
        mapf/conflicts.py \
        mapf/low_level.py \
        mapf/cardinal.py \
        mapf/conflict_graph.py \
        mapf/features.py \
        mapf/high_level_cbs.py \
        mapf/icbs_h.py \
        mapf/test_features.py \
        mapf/baselines.py \
        mapf/oracle_labeler.py \
        mapf/train_cl_ranker.py \
        mapf/phase3_benchmark.py \
        models/ranker_cl_oracle.json \
        models/linear_cl_oracle.pkl \
        results/phase3_training_summary.json \
        results/phase3_benchmark.csv \
        results/phase3_statistical_summary.md \
        results/PHASE3_FINAL_REPORT.md

echo "[3/4] Committing Phase 3 changes..."
git commit -m "feat(phase3): add CL-ICBS-h 24-feature ranker, hybrid oracle, SOTA baselines, and 25-scen benchmark"

echo "[4/4] Pushing branch '${BRANCH_NAME}' to ${REPO_URL}..."
git push -u origin "${BRANCH_NAME}"

echo "Done! Branch '${BRANCH_NAME}' pushed to https://github.com/anay-28/learned-icbsh-mapf/tree/${BRANCH_NAME}"
