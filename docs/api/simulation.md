# crawlbot.simulation

Boucle de simulation fermée : configuration, logging, orchestration, plotting.

---

## SimConfig

**Fichier** : `crawlbot/simulation/config.py`

Dataclass centralisant tous les paramètres ajustables du pipeline.

### Paramètres principaux

| Catégorie | Paramètre | Défaut | Description |
|-----------|-----------|--------|-------------|
| **Timing** | `dt_nmpc` | 0.1 s | Période NMPC (10 Hz) |
| | `dt_qp` | 0.01 s | Période QP/MuJoCo (100 Hz) |
| | `t_swing` | 6.0 s | Durée single-support |
| | `t_ds` | 0.5 s | Durée double-support |
| **Actionneurs** | `tau_max` | 20 Nm | Limite couple articulaire |
| **Momentum** | `hw_min/max` | ±5 Nms | Enveloppe roues |
| | `L_max` | 10 Nms | Limite moment cinétique robot |
| | `tau_w_max` | 5 Nm | Limite taux L̇ |
| **AOCS** | `aocs_mode` | `'legacy'` | `'legacy'` / `'H_est'` / `'nmpc_plan'` |
| | `aocs_tau_w_max` | 5 Nm | Couple roue max |
| | `aocs_K_hw` | 2.0 | Gain feedback legacy [1/s] |
| **NMPC** | `nmpc_N` | 8 | Horizon de prédiction |
| | `nmpc_f_max` | 25 N | Limite force contact |
| **QP poids** | `ss_alpha_com` | 200 | Poids CoM en single-support |
| | `ext_alpha_ee` | 10000 | Poids EE en extension |
| **QP gains** | `ss_Kp_com` | 3.0 | Gain proportionnel CoM |
| | `ext_Kp_ee` | 40.0 | Gain proportionnel EE en extension |
| **Docking** | `weld_radius` | 5 mm | Seuil de docking |

---

## SimLog

**Fichier** : `crawlbot/simulation/logging.py`

Collecteur de séries temporelles. Chaque champ est une liste, un élément
par pas NMPC (≈10 Hz).

### Champs

| Groupe | Champs | Type par élément |
|--------|--------|------------------|
| Temps | `t`, `phase`, `step_idx` | float, str, int |
| Torso | `p_torso(3)`, `p_torso_ref(3)`, `e_torso_pos`, `e_torso_ori` | ndarray, float |
| EE | `d_grip_swing`, `d_grip_stance`, `swing_arm` | float, str |
| CoM | `r_com(3)`, `r_com_ref(3)`, `e_com` | ndarray, float |
| Momentum | `L_com(3)`, `L_dot(3)`, `hw(3)` | ndarray |
| RWA | `hw_physical(3)`, `tau_w(3)`, `rw_speed(3)` | ndarray |
| H estimator | `H_rO(3)`, `H_dot_est(3)`, `omega_struct(3)`, `qfrc_constraint_torque(3)` | ndarray |
| Joints | `tau(12)`, `tau_max_joint` | ndarray, float |
| Structure | `struct_pos(3)`, `struct_euler_deg(3)` | ndarray |
| Solveurs | `nmpc_ok`, `qp_ok`, `nmpc_time_ms`, `qp_time_ms` | bool, float |
| Événements | `dock_events` | list of dicts |

### Sérialisation

```python
log.save('results/logs/run.json')
log2 = SimLog.load('results/logs/run.json')
```

---

## SimulationLoop

**Fichier** : `crawlbot/simulation/sim_loop.py`

Orchestrateur principal : MuJoCo + NMPC + QP + AOCS en boucle fermée.

### Construction et setup

```python
sim = SimulationLoop(mjcf_path='...', urdf_path='...', config=SimConfig())
sim.setup(n_steps=3, start_a=2, start_b=2)
log = sim.run(verbose=True)
```

`setup()` initialise dans l'ordre :
1. MuJoCo model + data, détection RWA
2. Lecture ancres → conversion repère structure
3. RobotInterface (Pinocchio)
4. ContactScheduler + plan de traversée
5. SwingPlanner, TorsoPlanner
6. IK configuration initiale → injection dans MuJoCo
7. Activation welds + settling (500 pas)
8. CentroidalNMPC + build (CasADi/IPOPT)
9. 2 × WholeBodyQP (SS et EXT, poids différents)
10. MomentumDisturbanceEstimator

### Méthode `run()` — machine à phases

```
Pour chaque step du GaitPlan :
    DS : stabilisation (torso hold, les deux bras accrochés)
    SS : swing d'un bras (trajectoire EE + torso + NMPC+QP)
    EXT : convergence vers l'ancre cible (QP EE maximal)
         si d < weld_radius → dock → activer weld
    DS final : settling post-dock

Settling final : 20s de DS pour stabilisation
```

### Méthode `_step()` — un pas NMPC+QP

Flowchart d'un appel (exécuté toutes les 0.1s) :

```
┌─── Lecture état MuJoCo ───┐
│  mujoco_to_pinocchio()    │
│  robot.update(q, v) → rs  │
└───────────┬───────────────┘
            ▼
┌─── NMPC (1× par step) ───┐
│  nmpc.solve(rs, refs)     │
│  → rp, vp, lr, hw_dot_ff │
└───────────┬───────────────┘
            ▼
┌─── Boucle QP (10×) ──────────────────────────────┐
│  Pour i = 0..9 :                                  │
│    1. Re-lire MuJoCo → Pinocchio                  │
│    2. Référence torso/EE à t_qp                   │
│    3. qp.solve(...) → τ_q                         │
│    4. clip(τ_q, ±tau_max)                          │
│    5. ctrl[0:12] = τ_q                             │
│                                                    │
│    6. AOCS (si RWA) :                              │
│       hw_phys = I_w · ω_roues                      │
│       ω_s = qvel[3:6]                              │
│                                                    │
│       Mode 'legacy':                               │
│         L̇_est = (L_com - L_com_prev) / dt         │
│         τ_w = -L̇_est - K_hw · hw_error            │
│                                                    │
│       Mode 'H_est':                                │
│         Ḣ = estimator.update(r_com, v_com, L_com)  │
│         τ_w = -Ḣ - K_ω·ω_s - K_h·(hw - hw*)      │
│                                                    │
│       Mode 'nmpc_plan':                            │
│         τ_w = hw_dot_plan - K_hw · hw_error        │
│                                                    │
│       ctrl[12:15] = clip(τ_w, ±tau_w_max)          │
│                                                    │
│    7. mj_step()                                    │
│    8. Mise à jour hw depuis qvel roues             │
└───────────────────────────────────────────────────┘
            ▼
┌─── Logging ───────────────┐
│  Append to SimLog          │
└────────────────────────────┘
```

### Convention de signe AOCS

```
ctrl[12:15] = τ_w   →  couple sur l'arbre de la roue
                        la roue reçoit +τ_w
                        la structure reçoit -τ_w (3ème loi)
                        ḣ_w = τ_w  (la roue accélère)
```

---

## plot_simulation

**Fichier** : `crawlbot/simulation/plotting.py`

Fonction standalone `plot_simulation(log, save_path, cfg) -> fig`.

9 panneaux :
1. Distance EE → ancre (log scale)
2. Avancement torso X
3. Suivi CoM (X + erreur)
4. Moment cinétique L_com (3 axes + norme)
5. Couples articulaires (12 joints + max)
6. Dérive structure (translation)
7. Orientation structure (Euler roll/pitch/yaw)
8. Erreur tracking torso — position
9. Erreur tracking torso — orientation (angle géodésique)

Peut être utilisée offline depuis un JSON sauvegardé, sans MuJoCo :
```python
from crawlbot.simulation import SimLog, plot_simulation
log = SimLog.load('results/logs/run.json')
plot_simulation(log, save_path='fig.png')
```
