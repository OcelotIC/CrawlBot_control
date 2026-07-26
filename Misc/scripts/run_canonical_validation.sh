#!/usr/bin/env bash
# CANONICAL-config re-validation (J2). Flags = run_figdata.sh COMMON (verbatim)
# + the two chatter-fix flags. NOT the simple config. NO crawlbot/ change.
#   CANON = --ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000
#           --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical
#           --K_omega 50 --n-steps 5 --qp-envelope-exact
# Runs (all on CANON):
#   figA_canon      : + alpha-wrench 3 + epsilon-v 5e-3   (THE fix → runA export)
#   figA_canon_e0   : + alpha-wrench 0                    (POINT 1 baseline, chatter)
#   figA_canon_e1   : + alpha-wrench 1                    (POINT 1 threshold, partial)
#   figA_canon_evb  : + alpha-wrench 3 + epsilon-v 0      (POINT A ε_v "before" = 1mm/s)
set -u
cd "$(dirname "$0")/.."
SP=/tmp/claude-0/-home-user-CrawlBot-control/51ef2e0d-fbab-59f4-a937-ec828e0d7eda/scratchpad
CANON="--ss-two-task --ss-alpha-mom 5000 --alpha-torso-pose 24000 \
  --ss-kp-torso 3 --ss-kd-torso 2.5 --aocs_mode legacy_pid_numerical --K_omega 50 \
  --n-steps 5 --qp-envelope-exact"
run() {  # $1 out-dir, $2.. extra flags
  local out="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $out : CANON $* ==="
  FRAMES_PER_STEP=0 MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/diag_cooperative_arms.py \
    $CANON "$@" --out-dir "$out" > "$SP/${out}.log" 2>&1
  echo "   diag rc=$?"
}
run figA_canon      --interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 5e-3
run figA_canon_e0   --interstep-settle-alpha-wrench 0
run figA_canon_e1   --interstep-settle-alpha-wrench 1
run figA_canon_evb  --interstep-settle-alpha-wrench 3 --interstep-settle-epsilon-v 0
echo "=== POSTPROC + EXPORT the fix run (figA_canon → runA) ==="
SSMOM_RUN_DIR=figA_canon MUJOCO_GL=disabled PYTHONPATH=. python3 Misc/scripts/postprocess_results_figs.py > "$SP/figA_canon_pp.log" 2>&1
echo "  postproc rc=$? F3F4=$([ -f Misc/runs/figA_canon/postproc_F3F4.csv ] && echo YES || echo NO)"
MUJOCO_GL=disabled PYTHONPATH=. python3 scripts/export_figure_data.py \
  --run-dir Misc/runs/figA_canon --out results/j2_figdata/runA \
  --label "CANONICAL config 5-step (run_figdata.sh COMMON + chatter fix eps=3, eps_v=5mm/s)" \
  --config '{"ss_two_task": true, "ss_alpha_mom": 5000, "alpha_torso_pose": 24000, "ss_kp_torso": 3, "ss_kd_torso": 2.5, "K_omega": 50, "aocs_mode": "legacy_pid_numerical", "qp_envelope_exact": true, "aocs_active_in_interstep": true, "interstep_hw_refresh": true, "interstep_settle_alpha_wrench": 3.0, "interstep_settle_epsilon_v": 5e-3, "n_steps": 5}'
echo "  export rc=$?"
echo "=== CANONICAL VALIDATION RUNS DONE ==="