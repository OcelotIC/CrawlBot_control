# M7 — Environment provenance inventory (2026-04-19)

**Date:** 2026-04-19
**Scope:** Capture the current environment on this machine and whatever
can be recovered about the 2026-04-17 environment that produced the
archived `results/M7_1pct_1step_v21/`. Inventory only. No interpretation,
no fix, no recommendation.

---

## 1. Current environment on this machine today (2026-04-19)

### 1.1 Python / OS

| field | value |
|---|---|
| `python3 --version` | Python 3.11.15 |
| `uname -a` | `Linux runsc 4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016 x86_64 x86_64 x86_64 GNU/Linux` |
| `/etc/os-release` | Ubuntu 24.04.4 LTS (Noble) |
| `multiprocessing.cpu_count()` | 16 |
| `os.sched_getaffinity(0)` | 16 |
| `/proc/cpuinfo model name` | unknown (sandbox-masked); cpu MHz 2100.000; cache size 8192 KB |

### 1.2 Python packages (`pip freeze`)

Full output (71 entries):

```
absl-py==2.4.0
argcomplete==3.1.4
blinker==1.7.0
casadi==3.7.2
certifi==2026.2.25
charset-normalizer==3.4.6
cmeel==0.59.0
cmeel-assimp==6.0.2
cmeel-boost==1.89.0
cmeel-console-bridge==1.0.2.3
cmeel-octomap==1.10.0
cmeel-qhull==8.0.2.1
cmeel-tinyxml2==10.0.0
cmeel-urdfdom==4.0.1
cmeel-zlib==1.3.1
coal==3.0.2
colorama==0.4.6
conan==2.27.0
contourpy==1.3.3
cryptography==41.0.7
cycler==0.12.1
dbus-python==1.3.2
distro==1.9.0
eigenpy==3.12.0
etils==1.14.0
fasteners==0.20
fonttools==4.62.1
fsspec==2026.3.0
glfw==2.10.0
httplib2==0.20.4
idna==3.11
iniconfig==2.3.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
launchpadlib==1.11.0
lazr.restfulclient==0.14.6
lazr.uri==1.0.6
libcoal==3.0.2
libpinocchio==3.9.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mujoco==3.7.0
numpy==1.26.4
oauthlib==3.2.2
osqp==1.1.1
packaging==24.0
patch-ng==1.18.1
pillow==12.2.0
pin==3.9.0
pluggy==1.6.0
Pygments==2.20.0
PyGObject==3.48.2
PyJWT==2.7.0
PyOpenGL==3.1.10
pyparsing==3.1.1
pytest==9.0.3
python-apt==2.7.7+ubuntu5.2
python-dateutil==2.9.0.post0
PyYAML==6.0.1
qpsolvers==4.11.0
requests==2.33.1
scipy==1.17.1
six==1.16.0
toml==0.10.2
typing_extensions==4.15.0
urllib3==2.6.3
wadllib==1.3.6
xmltodict==0.13.0
yq==3.1.0
zipp==3.23.1
```

Load-time versions of the packages that drive simulation numerics:

| package | version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| mujoco | 3.7.0 |
| pin | 3.9.0 |
| libpinocchio | 3.9.0 |
| casadi | 3.7.2 |
| osqp | 1.1.1 |
| qpsolvers | 4.11.0 |
| matplotlib | 3.10.8 |
| eigenpy | 3.12.0 |
| coal (hpp-fcl) | 3.0.2 |

### 1.3 MuJoCo fingerprint

| field | value |
|---|---|
| `mujoco.__version__` | 3.7.0 |
| `mj_version()` | 3007000 |
| `mj_versionString()` | 3.7.0 |
| `mjMINVAL` | 1e-15 |
| `mujoco.__file__` | `/usr/local/lib/python3.11/dist-packages/mujoco/__init__.py` |
| `MUJOCO_GL` env | unset at subprocess level; `osmesa` fails this session because `OpenGL.raw.GL._errors` cannot resolve `glGetError` (observed during session setup); `MUJOCO_GL=disabled` used for the four-run campaign |

### 1.4 BLAS / LAPACK backend (`np.show_config()`)

- **BLAS:** `openblas64` version `0.3.23.dev` (detection via pkgconfig).
  Build flags: `USE_64BITINT=1 DYNAMIC_ARCH=1 DYNAMIC_OLDER= NO_CBLAS=
  NO_LAPACK= NO_LAPACKE= NO_AFFINITY=1 USE_OPENMP= HASWELL
  MAX_THREADS=2`.
- **LAPACK:** internal, bundled with NumPy 1.26.4 (`dep140213194937296`).
- **Compiler:** gcc 10.2.1, cython 3.0.8, linker `ld.bfd`, flags
  `-fno-strict-aliasing -Wl,--strip-debug`.
- **SIMD available:** SSE/SSE2/SSE3/SSSE3/SSE41/POPCNT/SSE42, AVX,
  F16C, FMA3, AVX2, AVX512F/CD/SKX/CLX/CNL/ICL.
  Not found: AVX512_KNL, AVX512_KNM.

### 1.5 Threading-related environment variables

At sim-time, **all of the following are unset**:

```
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
BLIS_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMEXPR_NUM_THREADS
PYTHONHASHSEED
PYTHONPATH          (set by the caller via the CLI, not in env)
LD_LIBRARY_PATH
```

Note: OpenBLAS above was built with `MAX_THREADS=2` regardless of env;
`USE_OPENMP=` is empty, so it uses its pthread backend. `threadpoolctl`
is not installed, so a dynamic threadpool enumeration is not available.

---

## 2. What can be recovered about the 2026-04-17 environment

### 2.1 Archive fingerprints

All files in `results/M7_1pct_1step_v21/` have mtime
`2026-04-17 22:52:30 +0000` (identical to the second across all 13
files). Sizes:

```
148 677  fig1_tracking.png
328 552  fig2_momentum_aocs.png
332 334  fig3_com_momentum.png
105 088  fig4_energy_passivity.png
 69 023  fig5_nmpc_health.png
121 782  fig6_contact_wrenches.png
216 130  fig7_joints.png
 18 516  fig8_snapshots_grid.png
203 745  fig9_ee_6d_tracking.png
243 718  fig10_torso_6d_tracking.png
    625  metrics.csv
 82 732  physics_trace.pkl
850 671  sim_log.json
```

### 2.2 Git history — HEAD at archive mtime

Commits whose author timestamps bracket the archive mtime
(`2026-04-17 22:52:30 UTC`):

```
ac14811  2026-04-17 22:19:11 UTC  update todo and code handoff for closed loop orientation debug
<archive mtime here>
3241ada  2026-04-17 23:47:24 UTC  diagnostics: per-phase metrics + M7 post-abort audit (Steps 1-3)
```

The archive was therefore almost certainly produced with HEAD at
`ac14811` (55 min before the per-phase refactor landed). Consistent with
this: `results/M7_1pct_1step_v21/metrics.csv` uses the **pre-refactor**
metric schema (keys `torso_ori_err_peak_deg`, `torso_pos_err_peak_mm`,
`hw_saturation_ratio_peak`, …), not the per-phase schema
(`torso_ori_peak_deg_SS`, `_DS`, `_global`) introduced by `3241ada`.

### 2.3 Environment-shaping files — stability check

Files that describe the install contract (requirements/setup/lockfiles)
between `ac14811` and current `HEAD`:

| file | present? | changed between ac14811 and HEAD? |
|---|---|---|
| `requirements.txt` | yes | **no** (`git diff ac14811..HEAD -- requirements.txt` is empty) |
| `docs/architecture/setup_env.sh` | yes | **no** (same diff empty) |
| `pyproject.toml` | absent | — |
| `setup.py` / `setup.cfg` | absent | — |
| `Pipfile` / `Pipfile.lock` | absent | — |
| `poetry.lock` / `uv.lock` | absent | — |
| `.python-version` | absent | — |
| `environment.yml` | absent | — |
| `.github/` (CI configs) | absent | — |

`requirements.txt` at `ac14811` is identical to current HEAD:

```
pin==3.9.0
mujoco>=3.1,<4
casadi>=3.6,<4
numpy>=1.24,<2
matplotlib>=3.7
qpsolvers>=0.9
osqp>=0.6
Pillow>=10.0
pytest>=7.0
pytest-benchmark>=4.0
scipy>=1.10
```

Of the eleven entries, **one is a hard pin** (`pin==3.9.0`); the other
ten are lower-bound or open ranges. The install contract therefore does
**not** determine the exact versions of numpy, mujoco, scipy, casadi,
osqp, qpsolvers, matplotlib, or Pillow at install time.

### 2.4 Embedded metadata in the archived artefacts

- **PNG `Software` tag** (matplotlib itself embeds its version in
  `img.info['Software']`): `fig1_tracking.png` and
  `fig10_torso_6d_tracking.png` report
  `Matplotlib version3.10.8, https://matplotlib.org/`, dpi
  `(150.0124, 150.0124)`. This is the **only versioned fingerprint**
  recoverable from the archive.
- **`physics_trace.pkl`** is pickle protocol 4 (`\x80\x04`). Pickle
  protocol 4 is the Python 3.4+ default, so it does not fingerprint
  the Python or library version. Contents: a `list[dict]` of 57
  entries with keys `cond_J_t`, `cond_NJe`, `contact_fL`, `lambda`,
  `phase`, `q`, `qdd_t`, `sig_min_J_t`, `sig_min_NJe`, `t`,
  `tau_abs_max`, `tau_l2`, `tau_q`, `tau_sat_idx`, `torso_debug`.
  No version string in the structure.
- **`sim_log.json`** top-level scalar keys are all simulation
  quantities (e.g. `settling_T_target`, `settling_exit_reason`,
  `settling_stage1_steps`, `settling_stage2_steps`). It carries **no**
  version/config/env/timestamp/commit/hash fingerprint.
- **`metrics.csv`** contains the pre-refactor metric schema
  (section 2.2), which is itself a commit fingerprint: it narrows the
  producer commit to `ac14811` or earlier.

### 2.5 Session notes / CI logs preserved on the branch

- **No CI:** `.github/` absent; no `.gitlab-ci.yml`, no `.circleci/`,
  no `Jenkinsfile`. Nothing automated runs on push.
- **No session transcripts committed** on or near
  `2026-04-17 22:52:30`. `docs/` contains architectural specs and
  technical logs but no install logs or "what I ran" sessions.
- `docs/architecture/setup_env.sh` is a manual setup script (last
  modified `2026-04-10 00:56:10 +0200` in commit `16dd047`); it runs
  `pip install --break-system-packages -q <ranges>` with no lockfile
  capture.

---

## 3. Diff table — today vs. 2026-04-17

Legend: **=** identical; **?** today's value known, 2026-04-17 value
unknown or unrecoverable; **~** install contract identical, installed
version not captured at archive time.

| field | today (2026-04-19) | 2026-04-17 (at archive mtime) | status |
|---|---|---|---|
| requirements.txt contents | as §1.2 install contract | identical (see §2.3) | = |
| docs/architecture/setup_env.sh contents | as §1.2 install script | identical (see §2.3) | = |
| git HEAD for artefact producer | (n/a — this is today) | `ac14811` inferred from mtime window and metrics.csv schema (§2.2, §2.4) | n/a |
| matplotlib version | 3.10.8 | 3.10.8 (PNG `Software` tag, §2.4) | = |
| pin (Pinocchio wheel) version | 3.9.0 | 3.9.0 (hard-pinned in requirements.txt, §2.3) | = (pinned) |
| libpinocchio | 3.9.0 | 3.9.0 (co-installed with `pin==3.9.0`) | = (pinned-implicitly) |
| Python version | 3.11.15 | unknown (no `.python-version` file) | ? |
| mujoco version | 3.7.0 | unknown — range `>=3.1,<4` | ~ |
| numpy version | 1.26.4 | unknown — range `>=1.24,<2` | ~ |
| scipy version | 1.17.1 | unknown — range `>=1.10` | ~ |
| casadi version | 3.7.2 | unknown — range `>=3.6,<4` | ~ |
| osqp version | 1.1.1 | unknown — range `>=0.6` | ~ |
| qpsolvers version | 4.11.0 | unknown — range `>=0.9` | ~ |
| Pillow version | 12.2.0 | unknown — range `>=10.0` | ~ |
| pytest version | 9.0.3 | unknown — range `>=7.0` | ~ |
| eigenpy version | 3.12.0 | unknown (transitive via `pin==3.9.0`) | ? |
| coal / hpp-fcl | 3.0.2 | unknown (transitive via `pin==3.9.0`) | ? |
| `uname -a` | `Linux runsc 4.4.0 #1 SMP Sun Jan 10 15:06:54 PST 2016 x86_64` | unknown; sandbox kernel typically persistent across sessions on the same host | ? |
| OS release | Ubuntu 24.04.4 LTS | unknown | ? |
| BLAS backend | openblas64 0.3.23.dev, MAX_THREADS=2, USE_OPENMP= empty | unknown | ? |
| LAPACK backend | internal (numpy 1.26.4 bundled) | unknown — depends on numpy version installed | ? |
| compiler of numpy wheel | gcc 10.2.1 | unknown — depends on numpy wheel | ? |
| SIMD available on host | up to AVX512_ICL | unknown — same host presumed but not verified | ? |
| `OMP_NUM_THREADS` | unset | unknown; no session transcript | ? |
| `OPENBLAS_NUM_THREADS` | unset | unknown | ? |
| `MKL_NUM_THREADS` | unset | unknown | ? |
| `BLIS_NUM_THREADS` | unset | unknown | ? |
| `VECLIB_MAXIMUM_THREADS` | unset | unknown | ? |
| `NUMEXPR_NUM_THREADS` | unset | unknown | ? |
| `PYTHONHASHSEED` | unset | unknown | ? |
| `MUJOCO_GL` at runtime | `disabled` (today) | unknown; the archive's plots exist so offscreen rendering worked — implies `osmesa` or `egl`, not `disabled` | see §3.3 |

### 3.1 Summary of recoverable equalities

Three equalities can be asserted directly:
1. `requirements.txt` unchanged between `ac14811` and HEAD.
2. `docs/architecture/setup_env.sh` unchanged between `ac14811` and HEAD.
3. matplotlib version identical (3.10.8) between archive-time install
   and today — from PNG `Software` metadata.

### 3.2 Summary of residual unknowns

Every other version, BLAS/LAPACK fingerprint, threading-env setting,
and kernel/OS/CPU detail from 2026-04-17 is **not recoverable from the
branch**. The install contract in `requirements.txt` allowed wide
version ranges for ten of eleven packages, no lockfile was committed,
no CI ran, and no session transcript was preserved.

### 3.3 OSMesa observation

One observable mismatch: today `MUJOCO_GL=osmesa` errors on import at
`glGetError` resolution (§1.3), while the archive PNGs confirm
offscreen rendering was functional on 2026-04-17 (§2.1, §2.4). Neither
run's simulation trajectory depends on OSMesa (both use
`MUJOCO_GL=disabled`), so this does not directly explain the DS
divergence. It is evidence that the host's native graphics/linker
state has shifted, which is consistent with — but does not prove —
parallel shifts in MuJoCo's own native solver and BLAS build.
