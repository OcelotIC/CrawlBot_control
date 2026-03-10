# VISPA — Document de reprise de session
> À fournir en début de conversation pour reprendre le travail à R5.

---

## 1. Contexte du projet

**VISPA** est un robot crawling dual-arm conçu pour la locomotion en microgravité sur structures flottantes. Le principe : le robot se déplace en saisissant alternativement des points d'ancrage sur la structure, sans contact au sol. Le contrôleur est l'objet de la validation pour publication académique.

**Repo :** `https://github.com/OcelotIC/CrawlBot_control.git`
**Clonage local :** `/home/claude/CrawlBot_control/`
**Modèles :** `models/VISPA_crawling_fixed.urdf`, `models/VISPA_crawling.xml`

---

## 2. Architecture du contrôleur

Contrôleur hiérarchique deux étages :

### Stage 1 — CentroidalNMPC (~10 Hz)
- Fichier : `solvers/centroidal_nmpc.py` + `solvers/nmpc_solver.py`
- Planification de la dynamique centroïdale sur un horizon glissant
- Entrées : état CoM, positions de contact, références momentum
- Sorties : forces de contact planifiées `f*`, profil momentum `L*`
- **Hypothèse clé (documentée dans le papier) :** positions de contact supposées constantes dans l'horizon (quasi-static drift hypothesis, voir §7)

### Stage 2 — WholeBodyQP (~100 Hz)
- Fichier : `solvers/wholebody_qp.py` + `solvers/hierarchical_qp.py`
- Dynamique corps complet via Pinocchio
- Jacobians de contact recalculés à chaque pas → robuste à la dérive
- Tâches hiérarchiques : tracking momentum > suivi torso > posture

### Orchestration
- Fichier : `simulation_loop.py` — classe `SimulationLoop`
- Machine d'états : **DS → SS → EXT → DS → ...**
  - `DS` : Double Support (deux bras ancrés)
  - `SS` : Single Support (un bras ancré, l'autre en swing)
  - `EXT` : Extension (repositionnement torso vers prochain ancrage)

---

## 3. Architecture MuJoCo / Pinocchio

Point critique à ne pas oublier :

- **MuJoCo** simule **deux corps flottants** :
  - `qpos[0:7]` → structure (floating base de la structure)
  - `qpos[7:14]` → torso du robot (floating base du robot)
- **Pinocchio** ne voit **que le torso** du robot (`pin_q[0:3] = mj_qpos[7:10]`)
- `rs.r_com` (CoM Pinocchio) et les ancres `read_anchors_from_mujoco()` sont tous les deux en **frame monde** → les bras de levier sont cohérents sans transformation supplémentaire
- `read_anchors_from_mujoco()` lit `mj_data.site_xpos` → positions monde des sites MuJoCo

---

## 4. Inventaire des fichiers clés

| Fichier | Rôle |
|---|---|
| `robot_interface.py` | Wrapper Pinocchio — état cinématique/dynamique |
| `contact_scheduler.py` | `ContactScheduler`, `GaitPlan`, `read_anchors_from_mujoco` |
| `swing_planner.py` | Trajectoires quintic du bras libre |
| `torso_planner.py` | `TorsoPlanner` — refs 6D torso + CoM + frame structure |
| `locomotion_planner.py` | `LocomotionPlanner` — ref CoM linéaire inter-steps |
| `ik.py` | `solve_ik`, `dock_configuration` |
| `simulation_loop.py` | Orchestration boucle fermée + logging |
| `dynamics.py` | Constantes frames : `FRAME_TOOL_A=18`, `FRAME_TOOL_B=32` |
| `solvers/centroidal_nmpc.py` | Formulation OCP CasADi/IPOPT |
| `solvers/nmpc_solver.py` | Interface solver NMPC |
| `solvers/wholebody_qp.py` | QP corps complet |
| `solvers/hierarchical_qp.py` | Stack de tâches hiérarchique |
| `solvers/contact_phase.py` | `ContactConfig`, `ContactPhase` |
| `test_r3_fixes.py` | Suite de tests R3 (11 tests) |
| `test_r4_fixes.py` | Suite de tests R4 (6 tests) |

---

## 5. État d'avancement — Roadmap R1–R7

| Jalon | Description | État | Commit |
|---|---|---|---|
| **R1** | `GaitPlan` n_steps=3 — séquence 7 phases, alternance bras, durées | ✅ COMPLÉTÉ | — |
| **R2** | `SwingPlanner` recalibré sur ancres MuJoCo live (vs nominales) | ✅ COMPLÉTÉ | — |
| **R3** | `TorsoPlanner` live state + frame structure | ✅ COMPLÉTÉ | `a3744e9` |
| **R4** | NMPC contact positions live (r_contact_A/B) | ✅ COMPLÉTÉ | ✅ pushed |
| **R5** | Validation `SimulationLoop` boucle fermée `n_steps=3` | 🔴 **PROCHAIN** | — |
| **R6** | Lancer sim complète → générer logs de locomotion | 🔴 PENDING | — |
| **R7** | `plot_comparison` → figures finales papier | 🔴 PENDING | — |

---

## 6. Détail des fixes R3 et R4 (résumé pour contexte)

### R3 — `_setup_torso_for_step` dans `TorsoPlanner` (commit `a3744e9`)

Trois fixes :
1. **`p_t0` live** : lu depuis `mj_data` via `mujoco_to_pinocchio()` au lieu de la dernière référence planifiée (garantit C⁰ de la trajectoire torso)
2. **Ancres stance live** : `read_anchors_from_mujoco()` dans le calcul IK end (vs ancres nominales du scheduler)
3. **Frame structure** : `TorsoPlanner.add_phase()` et `set_hold()` acceptent `(p_struct, R_struct)` → stockage en frame structure, reconstruction monde via `_struct_to_world()` avec termes Coriolis/centripète. `simulation_loop._step()` extrait `(p_struct, R_struct, v_struct, omega_struct)` depuis `mj_data` à chaque sous-pas.

**Validation :** `test_r3_fixes.py` — **11/11 PASS**

### R4 — NMPC contact positions live (dans `simulation_loop._step()`)

**Problème :** `cc_ss.r_contact_A/B` figés à l'initialisation → bras de levier `(r_Cj − r_com)` biaisé après dérive de structure → plan de forces NMPC corrompu sur tout l'horizon.

**Fix :**
```python
# Dans simulation_loop._step(), avant l'appel nmpc.solve()
mj_a_live, mj_b_live = read_anchors_from_mujoco(self.mj_model, self.mj_data)
cc_nmpc = ContactConfig.from_phase(
    cc_ss.phase,
    mj_a_live[stance_a][:3].copy(),
    mj_b_live[stance_b][:3].copy()
)
# passer cc_nmpc à nmpc.solve() à la place de cc_ss
# le QP continue d'utiliser cc_ss (Jacobians recalculés par Pinocchio)
```

**Erreurs corrigées (monde frame, drift réaliste 6 s) :**
| Scénario | Dérive | Erreur nominale | Erreur live |
|---|---|---|---|
| A — Translation | 8 cm/x + 3 cm/y | **103 mm** | 0.000 mm |
| B — Rotation | 5°/z | **35 mm** | 0.000 mm |
| C — Trans + Rot | 8 cm + 5° | **71 mm** | 0.000 mm |

**Validation :** `test_r4_fixes.py` — **6/6 PASS**, régression R3 **11/11 PASS**

---

## 7. Hypothèse quasi-statique — Décision de design (pour le papier)

La limitation résiduelle du NMPC (positions de contact figées dans l'horizon alors que la structure dérive) est traitée comme **hypothèse déclarée**, pas comme un bug.

**Formulation pour le papier :**

> **Assumption [Quasi-static structure drift]:** The floating structure drifts at low velocity, $\|\dot{p}_s\| \leq v_{s,\max}$, such that contact position variation over the NMPC horizon $T_h$ is negligible w.r.t. the angular momentum envelope: $\|\dot{p}_s\| \cdot \|f_j\| \cdot T_h \ll L_{\max}$.

| Paramètre | Valeur |
|---|---|
| $v_{s,\max}$ | ~2 cm/s |
| $\|f_j\|$ | ~50 N |
| $T_h$ | 0.8 s |
| $\epsilon_L$ (erreur max) | ~0.8 Nm·s |
| $L_{\max}$ | 5 Nm·s |
| **Ratio** | **~16 %** |

Le QP à 100 Hz compense le résidu via régulation de $L_{com}$.

**Future work (phrase pour le papier) :**
> *Extension to a Tube MPC formulation, where the bounded structure drift rate defines the disturbance set, is a natural direction toward guaranteed recursive feasibility under persistent floating-base motion.*

---

## 8. Plan de R5 — Validation boucle fermée n_steps=3

### Objectif
Valider que `SimulationLoop` exécute correctement une séquence de 3 pas de locomotion en boucle fermée dans MuJoCo sans divergence, avec métriques quantitatives.

### Métriques à vérifier
1. **Séquence de phases** : la machine d'états DS→SS→EXT se déroule dans le bon ordre pour les 3 pas
2. **Ancrage** : à chaque fin de phase EXT, l'erreur de docking `||p_tool - p_anchor||` < seuil (ex. 5 mm)
3. **Momentum** : `||L_com||` reste dans l'enveloppe `L_max` tout au long de la simulation
4. **Torques** : `||tau||_inf` reste sous les limites articulaires
5. **Stabilité** : pas de divergence NaN/Inf dans les états

### Approche suggérée
1. Cloner le repo et installer les dépendances (`pip install pin casadi mujoco --break-system-packages -q`)
2. Écrire `test_r5_closed_loop.py` qui instancie `SimulationLoop` avec `n_steps=3` et loggue les métriques ci-dessus
3. Vérifier le logging existant dans `simulation_loop.py` pour s'appuyer dessus
4. Générer des figures : trajectoire CoM, profil momentum, erreurs de docking

### Commande de lancement typique
```bash
PYTHONPATH=/home/claude/CrawlBot_control python3 /home/claude/CrawlBot_control/test_r5_closed_loop.py
```

---

## 9. Dépendances et patterns shell

### Installation
```bash
pip install pin casadi mujoco numpy --break-system-packages -q
```

### Pattern shell fiable
```bash
# Écrire le script sur disque, puis exécuter — NE PAS utiliser les heredocs inline
cat > /home/claude/CrawlBot_control/test_script.py << 'EOF'
# contenu du script
EOF
PYTHONPATH=/home/claude/CrawlBot_control python3 /home/claude/CrawlBot_control/test_script.py
```

### Variables d'environnement utiles
```bash
export MUJOCO_GL=egl  # si pas d'affichage (serveur headless)
```

---

## 10. Instructions pour démarrer la prochaine session

1. Fournir ce fichier `.md` en début de conversation
2. Demander : *"Reprendre le travail à R5 — validation boucle fermée SimulationLoop n_steps=3"*
3. Claude doit :
   - Cloner le repo dans `/home/claude/CrawlBot_control/`
   - Vérifier que R4 est bien commité (sinon réappliquer le fix depuis §6)
   - Écrire et exécuter `test_r5_closed_loop.py`
   - Reporter les métriques et valider ou débugger

