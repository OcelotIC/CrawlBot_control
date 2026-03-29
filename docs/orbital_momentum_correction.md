# Correction du terme orbital dans la dynamique de hw — NMPC

## Contexte

Le NMPC centroïdal (Stage 1) modélise la dynamique du momentum des roues à
réaction comme :

```
ḣ_w = -L̇_com = -Σ_j [(r_Cj - r_com) × f_j + τ_j]
```

où `L̇_com` est la dérivée du moment cinétique centroïdal du robot, calculé
autour du CoM du robot (point mobile).

**Problème :** cette formulation utilise le moment des forces de contact autour
d'un point mobile (le CoM du robot), alors que la conservation du moment
cinétique s'exprime autour d'un point fixe dans le repère inertiel.

---

## Rappel : moment cinétique vs moment centroïdal

Le moment centroïdal est le moment cinétique évalué au centre de masse du
système. Ce n'est pas une quantité physique différente — c'est un cas
particulier du moment cinétique.

Le théorème de transport (Varignon) relie les deux :

```
L_O = L_G + m · (r_G - O) × v_G
```

où :
- `L_O` = moment cinétique autour du point fixe O
- `L_G` = moment centroïdal (autour du CoM G)
- `m · (r_G - O) × v_G` = **terme orbital**

Le terme orbital représente le moment cinétique dû à la translation du CoM
autour du point de référence. Il est nul si et seulement si O = G ou v_G = 0.

---

## Conservation du moment cinétique total

### Hypothèses
- Microgravité : pas de forces externes
- Système isolé : robot + structure + roues
- État initial au repos : `L_total(t=0) = 0`

### Loi de conservation

Autour d'un point fixe O dans le repère inertiel :

```
L_total^O = L_robot^O + L_structure^O + h_w^O = 0     ∀t
```

### Hypothèse de commande (contribution du papier)

Si le contrôleur AOCS maintient `L_structure ≈ 0` (la structure ne tourne pas),
alors :

```
L_robot^O + h_w^O = 0     ∀t
```

**C'est cette relation que le NMPC doit imposer.**

---

## Développement de L_robot autour du CoM structure

Le moment cinétique du robot autour du point O s'exprime comme :

```
L_robot^O = L_com^robot + m_robot · (r_com - O) × v_com
```

En choisissant **O = r_com_structure** (CoM du satellite/structure) :

```
L_robot^S = L_com + m_robot · (r_com - r_S) × v_com
```

où `r_S = r_com_structure`.

### Choix de O = r_com_structure

Ce choix est motivé par :

1. **Les roues sont solidaires de la structure** — leur moment cinétique `h_w`
   est naturellement exprimé dans le repère structure
2. **Le bras de levier** `Δr = r_com - r_S` est la distance physique entre le
   robot et le satellite, le long de la poutre. C'est cette distance qui crée
   le moment orbital quand le robot crawle.
3. **Praticité** : `r_S` est un état connu de la simulation
   (`mj_data.qpos[0:3]`)

### Approximation vs point fixe exact

Le point fixe exact est le CoM du système total :

```
r_G = (m_S · r_S + m_robot · r_com + m_rw · r_rw) / M_total
```

Avec m_S = 500 kg, m_robot ≈ 71 kg, m_rw = 1.5 kg, M_total ≈ 572.5 kg :

```
r_G ≈ 0.874 · r_S + 0.124 · r_com
```

L'erreur de l'approximation `O ≈ r_S` au lieu de `O = r_G` est :

```
|r_G - r_S| = (m_robot / M_total) · |r_com - r_S| ≈ 0.124 · |Δr|
```

Pour `|Δr| ≈ 1.5 m` (excursion typique du robot), l'erreur est ~18 cm.
Acceptable pour l'horizon NMPC (~1s), et on peut utiliser `r_G` exact si besoin.

---

## Dynamique corrigée de hw

### Dérivation

On différencie la conservation `L_robot^S + h_w = 0` :

```
ḣ_w = -dL_robot^S/dt
    = -L̇_com - m · d/dt[(r_com - r_S) × v_com]
```

Développement de la dérivée temporelle du terme orbital :

```
d/dt[(r_com - r_S) × v_com] = (v_com - v_S) × v_com + (r_com - r_S) × a_com
                              = -v_S × v_com + (r_com - r_S) × a_com
```

En utilisant les équations du mouvement du robot :
`m · a_com = Σ_j f_j`

On obtient :

```
ḣ_w = -L̇_com - (r_com - r_S) × Σ_j f_j + m · v_S × v_com
```

Si la structure est quasi-statique (`v_S ≈ 0`), le dernier terme est
négligeable et :

```
ḣ_w ≈ -L̇_com - (r_com - r_S) × Σ_j f_j
```

### Forme compacte

En substituant `L̇_com = Σ_j [(r_Cj - r_com) × f_j + τ_j]` :

```
ḣ_w = -Σ_j [(r_Cj - r_com) × f_j + τ_j] - (r_com - r_S) × Σ_j f_j
     = -Σ_j [(r_Cj - r_S) × f_j + τ_j]
```

**Résultat :** la dynamique corrigée de `h_w` est simplement le moment des
forces de contact autour du CoM de la structure, et non autour du CoM du robot.

---

## Résumé de la correction dans le NMPC

### ODE actuelle (`centroidal_ode`)

```python
# Moment autour du CoM robot (INCORRECT pour hw)
L_dot = sum((r_Cj - r_com) × f_j + tau_j)
hw_dot = -L_dot
```

### ODE corrigée

```python
# Moment autour d'un point fixe (CoM structure ou CoM total)
L_dot_fixed = sum((r_Cj - r_ref) × f_j + tau_j)
hw_dot = -L_dot_fixed
```

où `r_ref` est :
- soit `r_com_structure` (approximation, erreur ~12%)
- soit `r_G_total` calculé exactement

**Note :** la dynamique de `L_com` (centroïdal) utilisée pour le tracking CoM
dans le NMPC reste inchangée. Seule la dynamique de `hw` est corrigée.

---

## Impact estimé de la correction

Le terme orbital manquant vaut :

```
(r_com - r_S) × Σ f_j
```

Ordre de grandeur :
- Bras de levier `|r_com - r_S|` ≈ 0.5–2.0 m (excursion de crawling)
- Forces de contact `|Σ f_j|` ≈ 1–10 N
- Terme orbital ≈ 0.5–20 Nm (instantané)

Cumulé sur un pas de crawling (~6s) :
- Contribution au budget hw ≈ 3–120 Nm·s ← potentiellement >> h_max = 5 Nm·s

**C'est potentiellement la raison pour laquelle le NMPC devenait infaisable
avec hw_min/max = ±5 Nm·s** : il ne prenait pas en compte le momentum orbital
qui s'accumule quand le robot crawle le long de la structure.

---

## Changement de repère pour h_w

Les roues sont dans le repère structure. Pour la conservation dans le repère
inertiel :

```
h_w^world = R_structure · h_w^local = R_structure · I_w · ω_roues^local
```

Dans le NMPC, `h_w` doit être en repère inertiel. La transformation est :

```
h_w^inertiel(t) = R_S(t) · diag(I_w) · ω_roues(t)
```

Si la structure ne tourne pas (objectif de la commande), `R_S ≈ I₃` et la
transformation est triviale. Mais si elle tourne significativement, il faut
en tenir compte.

---

## Implémentation

### Dans `centroidal_nmpc.py` — `centroidal_ode()`

Ajouter `r_ref` (position du CoM structure) comme paramètre. Modifier la
dynamique de `hw_dot` pour utiliser le moment autour de `r_ref` au lieu de
`r_com`.

### Dans `simulation_loop.py` — `_step()`

Passer `r_com_structure` au NMPC à chaque appel (déjà lu via
`mj_data.qpos[0:3]`).

### Dans le contrôleur AOCS

Le feedforward AOCS utilise aussi `L̇_com` pour estimer le couple à appliquer.
Il doit être corrigé de la même manière :

```python
# Actuel (incorrect)
tau_w = -L_dot_est - K_hw * hw_error

# Corrigé
L_dot_fixed = L_dot_est + m_robot * np.cross(r_com - r_struct, a_com)
tau_w = -L_dot_fixed - K_hw * hw_error
```
