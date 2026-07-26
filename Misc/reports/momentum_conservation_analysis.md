# Conservation du moment cinétique et correction orbitale — Note technique

## 1. Problème identifié

Le NMPC centroïdal modélise la dynamique du momentum des roues à réaction
(RWA) comme :

```
ḣ_w = -L̇_com = -Σ_j [(r_Cj - r_com) × f_j + τ_j]
```

Cette formulation calcule le moment des forces de contact autour du **CoM du
robot** (point mobile). Or la conservation du moment cinétique s'exprime autour
d'un **point fixe** dans le repère inertiel. Le terme orbital associé au
mouvement de translation du CoM est absent, ce qui fausse la prédiction de `hw`
sur l'horizon NMPC.

---

## 2. Rappels de mécanique

### 2.1 Moment cinétique et moment centroïdal

Le moment centroïdal n'est **pas** une grandeur physique distincte du moment
cinétique. C'est le moment cinétique évalué au centre de masse du système.
C'est un cas particulier, pas une quantité différente.

Le théorème de transport (Varignon) relie le moment cinétique autour de deux
points :

```
L_O = L_G + m · (r_G - O) × v_G
```

- `L_O` : moment cinétique autour du point O (fixe dans le repère inertiel)
- `L_G` : moment centroïdal (moment cinétique autour du CoM G)
- `m · (r_G - O) × v_G` : **terme orbital** (moment cinétique dû à la
  translation du CoM autour de O)

Le terme orbital est nul si et seulement si `O = G` (le point de réduction est
le CoM) ou `v_G = 0` (le CoM est immobile).

### 2.2 Conservation pour un système isolé

En microgravité, sans forces ni couples externes, le moment cinétique total
du système autour de tout point fixe O est conservé :

```
L_total^O = constante
```

Le système est composé de trois sous-systèmes :
- **Robot** (torso + 2 bras, ~71 kg)
- **Structure** (satellite/poutre, 500 kg)
- **Roues à réaction** (3 × 0.5 kg, montées sur la structure)

Départ au repos → `L_total^O(t=0) = 0`, donc :

```
L_robot^O + L_structure^O + h_w^O = 0     ∀t
```

---

## 3. Hypothèse de commande

L'objectif de l'AOCS est de maintenir la structure immobile en rotation. Si
cet objectif est atteint (`ω_structure ≈ 0`, `v_structure ≈ 0`), alors :

```
L_structure^O ≈ 0
```

Ce qui donne la relation de commande fondamentale :

```
L_robot^O + h_w ≈ 0     ∀t
```

C'est cette relation que le NMPC doit imposer. La contrainte `‖h_w‖ ≤ h_max`
garantit alors que le robot ne génère pas plus de moment cinétique que ce que
les roues peuvent absorber.

**Cette hypothèse justifie la contribution du papier** : le NMPC planifie des
trajectoires qui respectent l'enveloppe de capacité des roues.

---

## 4. Le terme orbital manquant

### 4.1 Développement de L_robot autour d'un point fixe

```
L_robot^O = L_com + m_robot · (r_com - O) × v_com
            ─────   ───────────────────────────────
          centroïdal      terme orbital
```

Le NMPC actuel utilise uniquement `L_com` (moment centroïdal) dans la
dynamique de `hw`. Il manque le terme orbital
`m_robot · (r_com - O) × v_com`.

### 4.2 Dynamique corrigée de hw

En différenciant `L_robot^O + h_w = 0` :

```
ḣ_w = -dL_robot^O/dt
    = -L̇_com - m · d/dt[(r_com - O) × v_com]
    = -L̇_com - m · (r_com - O) × a_com
```

(`v_com × v_com = 0`, et O fixe donc dO/dt = 0)

En utilisant `m · a_com = Σ_j f_j` (Newton) :

```
ḣ_w = -L̇_com - (r_com - O) × Σ_j f_j
```

### 4.3 Forme compacte

En substituant `L̇_com = Σ_j [(r_Cj - r_com) × f_j + τ_j]` :

```
ḣ_w = -Σ_j [(r_Cj - r_com) × f_j + τ_j] - (r_com - O) × Σ_j f_j
     = -Σ_j [(r_Cj - O) × f_j + τ_j]
```

**Résultat : la dynamique corrigée de `hw` est le moment des forces de contact
autour du point fixe O, et non autour du CoM mobile du robot.**

### 4.4 Comparaison

| | Formulation | Point de réduction |
|---|---|---|
| Actuelle | `ḣ_w = -Σ [(r_Cj - r_com) × f_j + τ_j]` | CoM robot (mobile) |
| Corrigée | `ḣ_w = -Σ [(r_Cj - O) × f_j + τ_j]` | Point fixe O |

La dynamique centroïdale `L̇_com` utilisée pour le tracking CoM dans le NMPC
reste **inchangée**. Seule la dynamique de `hw` est corrigée.

---

## 5. Choix du point de référence O

### 5.1 Point fixe exact : CoM du système total

```
r_G = (m_S · r_S + m_robot · r_com + m_rw · r_rw) / M_total
```

Ce point est rigoureusement fixe (système isolé, départ au repos). Mais il
nécessite de connaître `r_S` en position absolue dans le repère inertiel,
ce qui **n'est pas réaliste opérationnellement** :
- pas de GPS précis (ou inexistant hors LEO)
- le star tracker donne l'attitude, pas la position
- la navigation orbitale donne la position du satellite à l'échelle km,
  pas mm

### 5.2 Approximation : CoM de la structure

```
O ≈ r_S    (CoM de la structure)
```

L'erreur de cette approximation :

```
|r_G - r_S| = (m_robot / M_total) · |r_com - r_S| ≈ 0.124 · |Δr|
```

Pour `|Δr| ≈ 1.5 m`, l'erreur est ~18 cm. Acceptable, mais soulève le même
problème : connaître `r_S` en absolu.

### 5.3 Solution retenue : reconstruction depuis la cinématique

**On n'a pas besoin de `r_S` en absolu.** Le terme correctif ne dépend que de
la position **relative** du robot par rapport à la structure :

```
Δr = r_com - r_S
```

Cette quantité est reconstructible depuis des données que le robot possède
nativement :

1. **Ancre de docking active** : le robot sait sur quelle ancre chaque
   gripper est accroché (paramètre discret, connu du contrôleur)
2. **Position de l'ancre dans le repère structure** : constante géométrique
   connue (CAD de la structure)
3. **Position des contacts en repère monde** : `r_Cj` est déjà un paramètre
   du NMPC

La reconstruction :

```
r_S = r_C_stance - R_structure · offset_ancre
```

où `offset_ancre` est la position de l'ancre active dans le repère local de
la structure (table de constantes connues) :

```
anchor_1a: [-2.0,  0.3, 0.025]    anchor_1b: [-2.0, -0.3, 0.025]
anchor_2a: [-1.2,  0.3, 0.025]    anchor_2b: [-1.2, -0.3, 0.025]
anchor_3a: [-0.4,  0.3, 0.025]    anchor_3b: [-0.4, -0.3, 0.025]
anchor_4a: [ 0.4,  0.3, 0.025]    anchor_4b: [ 0.4, -0.3, 0.025]
anchor_5a: [ 1.2,  0.3, 0.025]    anchor_5b: [ 1.2, -0.3, 0.025]
anchor_6a: [ 2.0,  0.3, 0.025]    anchor_6b: [ 2.0, -0.3, 0.025]
```

Si `R_structure ≈ I` (hypothèse de commande — la structure ne tourne pas) :

```
r_S ≈ r_C_stance - offset_ancre
```

---

## 6. Implémentation dans le NMPC

### 6.1 Interface

Avant chaque appel `nmpc.solve()`, `simulation_loop.py` calcule `r_struct`
à partir de l'ancre active et la passe comme paramètre continu (3D) au NMPC.
La logique discrète (quelle ancre → quel offset) reste hors de CasADi.

```python
# simulation_loop.py — _step()
anchor_offset = ANCHOR_OFFSETS[stance_anchor_id]   # lookup table (CAD)
r_struct_est = r_C_stance - anchor_offset           # R_struct ≈ I
nmpc.solve(..., r_struct=r_struct_est)
```

### 6.2 ODE modifiée

Le vecteur de paramètres du NMPC est étendu (+3 composantes pour `r_struct`) :

```
p = [r_ref(3), v_ref(3), r_C1(3), r_C2(3), r_struct(3)]    # 15 au lieu de 12
```

Dans `centroidal_ode()` :

```python
# L̇_com centroïdal — INCHANGÉ (tracking CoM)
L_dot = cross(r_C1 - r_com, f1) + tau1 + cross(r_C2 - r_com, f2) + tau2

# ḣ_w — CORRIGÉ (moment autour de r_struct, point ~fixe)
orbital = cross(r_com - r_struct, f1 + f2)
hw_dot = -L_dot - orbital
# Équivalent à : hw_dot = -(cross(r_C1 - r_struct, f1) + tau1 + cross(r_C2 - r_struct, f2) + tau2)
```

### 6.3 Ce qui ne change pas

- **`wholebody_qp.py`** : pas modifié. Le QP reçoit `hw_current` depuis les
  vitesses de roues physiques et respecte `hw ∈ [hw_min, hw_max]`.
- **L'AOCS physique** dans `_step()` : les roues MuJoCo conservent le
  momentum total par la physique du simulateur. Le feedback
  `τ_w = -L̇_est - K · hw_error` suffit.
- **La dynamique de `L_com`** dans le NMPC : reste autour du CoM robot
  (correct pour le tracking de trajectoire centroïdale).

---

## 7. Estimation de l'impact

### 7.1 Ordre de grandeur du terme orbital manquant

```
correction = (r_com - r_S) × Σ f_j
```

- Bras de levier `|r_com - r_S|` : 0.5–2.0 m (excursion de crawling)
- Forces de contact `|Σ f_j|` : 1–10 N
- Correction instantanée : 0.5–20 Nm

Cumulée sur un pas de crawling (~6 s), la contribution au budget `hw` peut
atteindre **3–120 Nm·s**, largement supérieur à `h_max = 5 Nm·s`.

### 7.2 Conséquence observée

Sans la correction, le NMPC planifie les trajectoires sans comptabiliser le
moment orbital qui s'accumule quand le robot crawle. Avec `hw ∈ [-5, 5]`,
le NMPC devient infaisable car le budget `hw` prédit ne correspond pas à la
réalité physique. Ceci explique :
- Les failures NMPC observées (80/295 dans les premiers tests)
- L'impossibilité de docking au step 2-3 avec `weld_radius = 5 mm`
- La nécessité antérieure de `hw_min/max = ±50` pour masquer le problème

### 7.3 Résultat attendu après correction

Le NMPC prédira correctement l'évolution de `hw` sur l'horizon. Il pourra
planifier des trajectoires faisables avec `hw ∈ [-5, 5] Nm·s` sans devenir
infaisable, ce qui devrait :
- Supprimer les failures NMPC
- Restaurer le docking à 5 mm sur 3+ pas
- Produire les courbes `hw(t) ≤ h_max` démontrées dans le papier

---

## 8. Expression de hw dans le repère inertiel

Les roues sont solidaires de la structure. Leur moment cinétique est
naturellement exprimé dans le repère structure :

```
h_w^local = I_w · ω_roues^local = diag(0.01) · ω_roues
```

Pour la conservation dans le repère inertiel :

```
h_w^world = R_structure · h_w^local
```

Si la structure ne tourne pas (objectif de la commande), `R_structure ≈ I₃`
et la transformation est triviale. Si la structure a tourné significativement,
cette rotation doit être prise en compte dans le NMPC.

---

## 9. Résumé des corrections à apporter

| Composant | Modification | Justification |
|---|---|---|
| `centroidal_nmpc.py` | Ajouter `r_struct` aux paramètres, corriger `ḣ_w` | Terme orbital manquant |
| `simulation_loop.py` | Reconstruire `r_struct` depuis ancre active, passer au NMPC | Interface |
| `simulation_loop.py` | Remettre `weld_radius = 5 mm` | Réalisme physique |
| `wholebody_qp.py` | Aucune | QP utilise hw physique, pas de modèle dynamique de hw |
| AOCS (`_step`) | Aucune | MuJoCo conserve le momentum total ; feedback K·error suffit |
| `L̇_com` dans NMPC | Aucune | Le tracking centroïdal reste autour du CoM robot |
