#!/usr/bin/env bash
# Paper figure-data export on the LOCKED config (J2 closure). runA ONLY.
# Same flags as the closure/curves run (commit ed1c200 lineage): figure config
# legacy_pid_numerical + exact box + the locked fix (ε_wrench=3, ε_v=5 mm/s),
# AOCS-in-DS + c_curr on (committed defaults). FRAMES_PER_STEP=0 skips the
# (osmesa) render subprocess. NO crawlbot/ change. Run B = future work (not run).
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
LOCKED="--qp-envelope-exact --aocs_mode legacy_pid_numerical \
  --interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 5e-3 --n-steps 5"

echo "=== RUN A (LOCKED config) ==="; t0=$SECONDS
FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
  $LOCKED --out-dir figA_locked > "$SP/figA_locked.log" 2>&1
echo "  diag rc=$? dt=$((SECONDS-t0))s"

echo "=== POSTPROC A (postproc_F3F4.csv: stance idx + planned Hdot_s + gate metrics) ==="
SSMOM_RUN_DIR=figA_locked MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/postprocess_results_figs.py \
  > "$SP/figA_locked_pp.log" 2>&1
echo "  postproc rc=$?  F3F4=$([ -f results/figA_locked/postproc_F3F4.csv ] && echo YES || echo NO)"

echo "=== EXPORT A → results/j2_figdata/runA_traversal.csv (+ runA_meta.json) ==="
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_figure_data.py \
  --run-dir results/figA_locked --out results/j2_figdata/runA \
  --label "LOCKED config 5-step (figure config: legacy_pid_numerical, exact box, eps=3 + eps_v=5mm/s)" \
  --config '{"qp_envelope_exact": true, "aocs_active_in_interstep": true, "interstep_hw_refresh": true, "aocs_mode": "legacy_pid_numerical", "interstep_settle_alpha_wrench": 3.0, "interstep_settle_epsilon_v": 5e-3, "n_steps": 5}'
echo "  export rc=$?"
echo "=== DONE ==="
