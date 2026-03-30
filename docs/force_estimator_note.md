# Estimateur de couple perturbateur AOCS — Note technique

## 1. Pourquoi `L̇_com` seul est insuffisant

### 1.1 Rappel du problème

L'AOCS actuel applique sur les roues :

```
τ_w = -L̇_com_est - K_hw · hw_error
```

où `L̇_com` est le taux de variation du moment cinétique centroïdal du robot
(moment cinétique autour du CoM mobile du robot).

À 14 % de ratio de masse (structure 500 kg, robot ~71 kg), la structure
tourne de **24°** malgré les roues. La cause : le feedforward `-L̇_com` ne
capture que la composante **spin** du moment cinétique du robot. La composante
**orbitale** — due à la translation du CoM du robot le long de la structure —
n'est pas compensée.

### 1.2 Décomposition du moment cinétique

Le moment cinétique du robot autour d'un point fixe O s'écrit (Varignon) :

```
H_{r/O} = L_com + (r_com - O) × m_r · v_com
           ─────   ────────────────────────────
          spin            orbital
```

- **Spin** (`L_com ≈ 5 Nms`) : bien compensé par `-L̇_com`.
- **Orbital** (`(r_com - O) × m_r · v_com ≈ 20 Nms`) : transite par les
  forces de contact (3ème loi de Newton), **pas par les roues**.

L'AOCS doit compenser la dérivée de `H_{r/O}` totale, pas seulement `L̇_com`.

### 1.3 Le bon signal AOCS

Le couple perturbateur net appliqué par le robot à la structure autour de O
est :

```
τ_dist,O = -Ḣ_{r/O}
```

C'est **ce signal** que les roues doivent rejeter. Pas `L̇_com`, pas les
forces de contact individuelles.

---

## 2. Choix du point O

### 2.1 Vérification dans le dépôt

Le MJCF (`VISPA_crawling_rwa3.xml`, ligne 76–79) définit la structure avec :

```xml
<body name="structure" pos="0 0 -1.8">
  <freejoint name="structure_free"/>
  <inertial pos="0 0 0" mass="7110" .../>
```

`inertial pos="0 0 0"` ⟹ le CoM de la structure **coïncide** avec l'origine
du repère body de la structure.

Dans `mujoco_to_pinocchio()` (simulation_loop.py, ligne 89) :

```python
p_s = mj_qpos[0:3]   # = position de l'origine du repère structure dans le monde
```

MuJoCo stocke la position du repère body dans `qpos`, pas la position du
CoM. Mais puisque `inertial pos="0 0 0"`, les deux coïncident.

**Conséquence** : les quantités Pinocchio (`r_com`, `v_com`, `L_com`)
calculées dans le repère structure sont **déjà relatives à O = CoM structure**.
Aucun offset n'est nécessaire.

### 2.2 Conventions

| Symbole | Définition | Source |
|---------|-----------|--------|
| O | CoM de la structure seule | Origine du repère structure (MJCF `inertial pos="0 0 0"`) |
| `r_com` | Position du CoM robot dans R_s | Pinocchio `data.com[0]` après `mujoco_to_pinocchio()` |
| `v_com` | Vitesse du CoM robot dans R_s | Pinocchio `J_com @ v` |
| `L_com` | Moment centroïdal du robot (autour de son CoM) dans R_s | Pinocchio `data.hg.vector[3:6]` |
| `ω_s` | Vitesse angulaire de la structure dans le monde | `mj_data.qvel[3:6]` |
| `h_w` | Moment cinétique des roues dans R_s | `I_w · mj_data.qvel[6:9]` |

---

## 3. Équations

### 3.1 Moment cinétique total du robot autour de O, dans R_s

```
H_{r/O} = L_com + r_com × (m_r · v_com)
```

où toutes les grandeurs sont dans le repère structure R_s, et `r_com` est
mesuré depuis O (origine de R_s = CoM structure).

### 3.2 Dérivée temporelle

#### Dérivée dans R_s (repère tournant)

```
(dH/dt)_{R_s} = dL_com/dt + r_com × (m_r · a_com) + v_com × (m_r · v_com)
              = dL_com/dt + r_com × (m_r · a_com)
```

puisque `v_com × v_com = 0`.

#### Dérivée inertielle (repère monde)

```
(dH/dt)_I = (dH/dt)_{R_s} + ω_s × H_{r/O}
```

Le terme de transport `ω_s × H_{r/O}` est négligeable tant que `ω_s ≈ 0`
(hypothèse de pointage). Cependant, si la structure a déjà tourné
significativement (comme les 24° observés), ce terme n'est plus négligeable
et doit être inclus pour la stabilité.

### 3.3 Formulation discrète — Voie A (retenue)

**Estimateur par différences finies sur H_{r/O} :**

```
H_k = L_com_k + r_com_k × (m_r · v_com_k)

Ḣ_fd = (H_k - H_{k-1}) / dt

Ḣ_inertiel = Ḣ_fd + ω_s × H_k
```

**Avantages :**
- Utilise uniquement des grandeurs mesurées (`L_com`, `r_com`, `v_com`)
- Pas besoin d'estimer les forces de contact individuelles
- Pas de dépendance au modèle dynamique (H, C) pour l'estimation
- La répartition des forces entre contacts en DS est **invisible** (correct :
  elle ne contribue pas au couple net)

**Inconvénient :**
- Différence finie à 100 Hz → bruit. Nécessite filtrage.

### 3.4 Filtrage

Filtre EMA (Exponential Moving Average) sur `Ḣ_fd` :

```
Ḣ_filtered_k = α · Ḣ_fd_k + (1 - α) · Ḣ_filtered_{k-1}
```

avec `α = dt / (τ_f + dt)` et `τ_f` constante de temps du filtre.

Pour `dt = 0.01 s` et un cutoff à ~10 Hz (`τ_f ≈ 0.016 s`) :
`α ≈ 0.38`.

On peut aussi appliquer le filtre directement sur `H_k` avant la dérivée :

```
H̃_k = α · H_k + (1 - α) · H̃_{k-1}
Ḣ_est = (H̃_k - H̃_{k-1}) / dt
```

Cette seconde forme est préférable car elle filtre le signal **avant**
l'amplification du bruit par la dérivation.

### 3.5 Alternative analytique (si `a_com` disponible)

```
Ḣ_analytical = L̇_com + r_com × (m_r · a_com)
```

où :
- `L̇_com = (L_com_k - L_com_{k-1}) / dt` (FD sur L_com)
- `a_com = J_com · q̈ + J̇_com · q̇`

Le problème : `q̈` n'est pas directement mesuré. On peut l'obtenir via :
- FD sur `v` : `q̈ ≈ (v_new - v_old) / dt` (bruité)
- Dynamique inverse : `q̈ = H^{-1} (B·τ_q + J_c^T·F - C)` (nécessite F)

Mais cela réintroduit soit du bruit (FD), soit la dépendance aux forces de
contact (dynamique inverse). La formulation FD directe sur `H_{r/O}` reste
plus simple et plus robuste.

---

## 4. Loi de commande AOCS proposée

```
τ_w = -Ḣ_est - K_ω · ω_s - K_h · (h_w - h_w*)
       ──────   ─────────   ──────────────────
       FF dist   FB attitude   FB desaturation
```

| Terme | Rôle | Gain suggéré |
|-------|------|--------------|
| `-Ḣ_est` | Rejet du couple perturbateur (spin + orbital) | — |
| `-K_ω · ω_s` | Amortissement de la rotation structure | `K_ω = 50 Nm·s/rad` (à régler) |
| `-K_h · (h_w - h_w*)` | Ramener les roues vers `h_w* = 0` (désaturation) | `K_h = 0.5 s⁻¹` |

### 4.1 Convention de signe

Dans MuJoCo :
- `ctrl[12:15] = τ_w` est le couple appliqué **sur l'arbre de la roue**.
- La roue reçoit `+τ_w`, la structure reçoit `-τ_w` (3ème loi de Newton).
- `ḣ_w = τ_w` (le couple augmente le momentum de la roue).

Conservation du moment cinétique total :

```
L_robot^O + h_w + L_struct^O = 0      (départ au repos)
```

Si l'AOCS fonctionne (`L_struct^O ≈ 0`) :

```
h_w ≈ -H_{r/O}
ḣ_w ≈ -Ḣ_{r/O}
τ_w ≈ -Ḣ_{r/O}
```

Le signe est cohérent avec la loi de commande proposée.

### 4.2 Différence avec l'AOCS actuel

| | AOCS actuel | AOCS proposé |
|---|---|---|
| Feedforward | `-L̇_com` (spin seulement) | `-Ḣ_{r/O}` (spin + orbital) |
| Feedback attitude | Aucun | `-K_ω · ω_s` |
| Feedback roues | `-K_hw · clip_error` | `-K_h · (h_w - h_w*)` |
| Transport | Non | `+ω_s × H_{r/O}` inclus |

Le terme `K_ω · ω_s` est **nouveau** et important : il fournit un
amortissement direct de la rotation de la structure. L'AOCS actuel n'a aucune
rétroaction sur `ω_s`.

### 4.3 Disponibilité de `ω_s`

`ω_s = mj_data.qvel[3:6]` est directement accessible dans la boucle QP.
C'est une mesure, pas une estimation.

**En opérationnel réel** : `ω_s` serait fourni par un gyroscope (IMU/IRU)
embarqué sur la structure, avec une bande passante >> 100 Hz. C'est une
mesure réaliste et disponible sur tout satellite.

---

## 5. Voie B — Validation par wrenches de contact

### 5.1 Formulation

Si les wrenches de contact `λ_c = [f_1, μ_1, f_2, μ_2]` sont connus, on
peut calculer directement :

```
Ḣ_{r/O} = G_O · λ_c
```

avec la matrice de grasp :

```
G_O = [ [r_1-O]× I₃  [r_2-O]× I₃ ]     (3×12)
```

### 5.2 Sources de `λ_c` dans le dépôt

| Source | Fiabilité | Accessibilité |
|--------|-----------|---------------|
| `λ_ref` (NMPC) | Faible — prédiction, pas réalité | `u_opt[:,0]` |
| `λ_qp` (QP) | Moyenne — cohérent avec τ_q mais gap 33× vs MuJoCo | Variables internes QP |
| `dynamics.py` `lambda_c` | Bonne — Pinocchio proximal Lagrangien | Nécessite appel séparé |
| `mj_data.qfrc_constraint` | **Ground truth** — forces de contrainte MuJoCo | `qfrc_constraint[0:6]` pour structure |

### 5.3 Validation croisée

Le test de cohérence principal est :

```
G_O · λ_c ≈ Ḣ_{r/O}
```

Pour le valider, on utilise `mj_data.qfrc_constraint[3:6]` (couple de
contrainte sur la structure autour de O) et on vérifie :

```
qfrc_constraint[3:6] ≈ -Ḣ_{r/O}    (à ω_s ≈ 0)
```

Ce test ne nécessite **aucune reconstruction** de `λ_c` — MuJoCo fournit
directement le couple résultant sur la structure.

### 5.4 Vérification de `dynamics.py`

`VISPAConstrainedDynamics.forward_dynamics()` retourne `lambda_c` via
Pinocchio (`data.lambda_c`). Ce `lambda_c` est en coordonnées locales du
contact (frame LOCAL de Pinocchio), **pas** en frame world-aligned.

Pour utiliser Voie B via dynamics.py, il faudrait :
1. Transformer `lambda_c` en frame world-aligned
2. Reconstruire `G_O · λ_c`

C'est possible mais redondant avec la Voie A pour le signal AOCS.
**Recommandation** : utiliser Voie B uniquement pour la validation, pas pour
la commande.

---

## 6. Risques numériques

### 6.1 Bruit sur les dérivées

| Signal | Bruit typique | Mitigation |
|--------|---------------|------------|
| `L_com` | Faible (Pinocchio intègre) | — |
| `v_com` | Faible (Pinocchio J_com @ v) | — |
| `H_{r/O}` | Faible (combinaison linéaire) | — |
| `Ḣ_fd = ΔH/dt` | **Moyen** (amplification ×100 Hz) | Filtre EMA pré-dérivation |
| `ω_s` | Faible (lecture directe MuJoCo) | — |

Le bruit principal est sur `Ḣ_fd`. Le filtre EMA avec τ_f ≈ 0.016 s
(cutoff ~10 Hz) introduit un retard de ~1 pas (10 ms), acceptable devant
l'échelle de temps du phénomène orbital (~6 s par pas de crawl).

### 6.2 Incohérence MuJoCo / Pinocchio

Le modèle Pinocchio (URDF) ne contient que le robot (torso + 2 bras).
Le modèle MuJoCo (MJCF) contient robot + structure + roues.

Les inerties du robot doivent correspondre exactement entre les deux modèles.
Un mismatch introduirait un biais systématique dans `L_com` et `r_com`.

**Vérification** : comparer `rs.total_mass` (Pinocchio) avec la masse robot
attendue. Le MJCF donne `torso=40 kg` + bras (≈31 kg) ≈ 71 kg.

### 6.3 Définition de O et offset inertiel

Si la masse de la structure change entre les tests (7110 kg → 500 kg),
la géométrie `inertial pos="0 0 0"` reste la même. Le CoM structure reste
à l'origine du repère body. **Pas de risque** sur ce point.

Si en revanche le MJCF était modifié avec un `inertial pos` non nul (ex:
structure asymétrique), il faudrait introduire un offset constant :

```python
O_in_struct_frame = inertial_pos  # from MJCF
r_com_rel_O = r_com - O_in_struct_frame
```

### 6.4 Convention de signe roue / structure

| Quantité | Signe | Vérifié par |
|----------|-------|-------------|
| `ctrl[12:15] > 0` | Couple positif sur la roue | Définition MuJoCo |
| `ḣ_w = τ_w` (couple sur roue) | `h_w` augmente | Physique du moteur |
| Réaction sur structure = `-τ_w` | 3ème loi | Mécanique |
| `H_{r/O} + h_w + L_struct ≈ 0` | Conservation | Test à valider |

### 6.5 Différences entre les trois `λ`

| Source | Signification | Utilisable pour AOCS ? |
|--------|---------------|----------------------|
| `λ_ref` (NMPC `u_opt`) | Décision de planification (forces virtuelles) | **Non** — gap 33× |
| `λ_qp` (QP decision var.) | Wrenches cohérents avec τ_q dans la dynamique rigide | **Non** — MuJoCo contraint différemment |
| `F_contact` (MuJoCo) | Forces réellement appliquées par le solver de contraintes | **Oui** — mais difficile à extraire proprement |

C'est précisément pourquoi la Voie A (estimation directe de `Ḣ_{r/O}` sans
passer par les forces de contact) est préférable pour la commande.

---

## 7. Protocole de validation

### 7.1 Test unitaire de signe (axe unique)

1. Imposer un mouvement du robot selon X (v_com > 0)
2. Calculer `H_{r/O}` → composante Y non nulle (produit vectoriel)
3. Vérifier que `qfrc_constraint[4]` (couple Y sur structure) ≈ `-Ḣ_Y`
4. Vérifier que `τ_w_cmd[1]` a le signe de `-Ḣ_Y`

### 7.2 Comparaison AOCS actuel vs proposé

Exécuter la simulation 3-step à 14 % de ratio de masse avec :
1. AOCS actuel (`-L̇_com` seul)
2. AOCS proposé (`-Ḣ_{r/O}`)

Métriques :
- Rotation maximale de la structure [°]
- Nombre de dockings réussis [/3]
- `||h_w||_max` [Nms]
- Nombre d'infaisabilités NMPC

### 7.3 Vérification de cohérence

À chaque pas de temps, logger et vérifier :

```
ε = ||qfrc_constraint[3:6] + Ḣ_est|| / max(||Ḣ_est||, 0.01)
```

Si `ε > 0.1` de façon persistante, l'estimateur a un problème.

---

## 8. Modifications proposées au dépôt

### 8.1 Nouveaux fichiers

| Fichier | Contenu |
|---------|---------|
| `force_estimator.py` | Classe `MomentumDisturbanceEstimator` |
| `tests/test_force_estimator.py` | Validation de signe et cohérence |

### 8.2 Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `simulation_loop.py` | AOCS section (lignes 892–906) : remplacer `-L̇_com` par `-Ḣ_est` |
| `simulation_loop.py` | SimLog : ajouter `H_rO`, `H_dot_est`, `tau_orbital`, `omega_s` |
| `simulation_loop.py` | `_step()` : extraire `ω_s` pour feedback et logging |

### 8.3 Fichiers NON modifiés

- `centroidal_nmpc.py` — Le NMPC continue d'utiliser sa propre correction
  orbitale indépendamment.
- `wholebody_qp.py` — Interdit de modification.
- `robot_interface.py` — Aucun changement nécessaire.
