# SOTA plan — dual energy model (fuel + locked body reserve)

Goal: make energy strictly conserved across life stages while letting body size **grow** over
a lifetime and **evolve** across generations, with the fewest parameters possible.

The **only** energy inputs to the whole system remain: **sun** (plant growth) and the
**plant-floor** seeding (anti-extinction). Nothing else creates energy. Everything below only
*moves* energy between pools, or *dissipates* it as heat (movement / attack / trophic
inefficiency), which is the only allowed sink.

---

## 0. The three pools (per individual)

| Pool | Attribute | Nature | Spendable? |
|------|-----------|--------|------------|
| **fuel** | `body.energie` (exists) | metabolic energy: eating fills it, everything is paid from it | yes |
| **reserve** | `body.reserve` (**NEW**) | locked structural mass = the body itself | **no** — only grows, released whole at death |
| **life** | `body.vie` (exists) | health, **NOT energy** — damaged by attacks, regenerated from fuel | n/a (not counted in the energy balance) |

**Size is a read-out of the reserve:** `size = reserve / k`  (equivalently `reserve = k · size`).
Pouring surplus fuel into the reserve makes the animal bigger; the reserve (and thus size) only ever grows,
capped by the `max_size` gene.

**Single source of truth (anti-drift rule).** `reserve` is the ONLY size quantity that mutates. Everything
else that today looks like "size" — `r_collision_box_individu`, `r_eat_box_individu`, `r_attack_box_individu`,
`max_energie_individu`, `max_vie_individu` — becomes a **cached value derived from `reserve`**, recomputed in
exactly one place (`update_size_from_reserve`, §7) and **written nowhere else**. They stay plain attributes
(not Python `@property`) because `r_collision_box_individu` alone is read ~28× in hot loops (quadtree, vision,
attack) — caching avoids per-read call overhead. This keeps `reserve` and the boxes coherent by construction.

`k = energie_par_taille` is the single global constant linking energy to size.

System energy balance (must stay constant up to sun-in / heat-out):
```
Σ plant.energy + Σ perishable.energy + Σ (fuel + reserve over individuals)
   += sun + plant-floor            (inputs)
   -= movement + attack + trophic inefficiency   (heat, allowed)
```

---

## 1. Fixed parameters

New globals in `parameters_V23.py` (4 new + reuse of existing):

| Param | Meaning | Suggested start |
|-------|---------|-----------------|
| `energie_par_taille` (**k**) | energy stored per unit of size; `reserve = k · size`. Sets how "expensive" a body is and how rich a corpse is | `40` → reserve_init ≈ 80 |
| `growth_pace` | **percentage** (fraction in [0,1]) of the surplus fuel moved fuel→reserve **per tick** — a rate, never a fixed amount; growth is therefore asymptotic and slows as the reserve fills | `0.05` (5%/tick) |
| `seuil_croissance` | comfort threshold: growth only happens when `fuel > seuil_croissance · max_energie`. **Must be < `facteur_energie_creer_bb`** so growth can occur before the repro threshold | `0.4` |
| `birth_ratio` | baby size as a fixed fraction of the **parent's current size** (`baby_size = birth_ratio · parent_size`) | `0.3` |

Reused unchanged (still fixed caps, brain/size scale *within* them):
- `max_attack_damage` — cap on damage dealt (hits victim `vie`).
- `max_energie_depensee_attack` — cap on fuel spent attacking.
- `facteur_multiplicatif_perte_vie` — move-cost coefficient.
- `facteur_energie_creer_bb` — fuel level that *triggers* reproduction.
- `facteur_energie_depensee_creer_bb` — repurposed only to size the per-baby budget (see §5).
- `r_collision_box_individu_init`, `max_energie_individu_init`, `max_vie_individu_init` — the baseline size and its derived quantities.

**Genes (heritable, mutated):** only **`max_size`** — the adult-size *cap*. This is the existing size
gene, reinterpreted as a target instead of a fixed value. `birth_ratio` is a **fixed global, not a gene**
(your latest choice), so no new gene is introduced.

---

## 2. What can DECREASE each pool

**fuel (`body.energie`) decreases from:**
- moving — `move()` line ~487 (already scales with size), rotation line ~466.
- attacking — `attack()` line ~764 (**will now also scale with size**, see §6).
- trophallaxy / spitting — `share_ressources()` line ~781.
- growth transfer — fuel → reserve (§4). *Internal move, energy conserved.*
- reproduction — fuel → baby fuel + baby reserve (§5). *Internal move, conserved.*
- starvation — when `fuel ≤ 0`, `vie` erodes (line ~742-743); the *negative fuel* is not "spent", it just
  gates death. Fuel itself never goes below its eaten amount except via the real sinks above.

**reserve (`body.reserve`) decreases from:**
- **only death** — released whole as meat. It never shrinks while alive (you can't eat your own skeleton).

## 3. What can INCREASE each pool

**fuel increases from:**
- eating plant / meat / trophallaxy — `ecosystem…eat block` lines ~779-804 (already conserved, trophic
  inefficiency is the allowed heat sink).

**reserve increases from:**
- **growth transfer only** — surplus fuel moved in each tick, capped at `k · max_size` (§4).
- at **birth**, the baby's reserve is set once from the parent's fuel (§5).

Everything that increases a pool is sourced from another pool or from food (ultimately sun / plant-floor).
Nothing is conjured.

---

## 4. Growth (fuel → reserve), each tick, for alive individuals

Opportunistic, from surplus, never lethal:

```
if body.reserve < k * body.max_size  and  body.energie > seuil_croissance * body.max_energie_individu:
    transfer = growth_pace * body.energie                      # simple: a % of current fuel
    transfer = min(transfer, k * body.max_size - body.reserve) # never overshoot the genetic cap
    body.energie -= transfer          # fuel down
    body.reserve += transfer          # reserve up  (energy conserved: internal move)
    body.update_size_from_reserve()   # recompute size + derived quantities (§7)
```

- Below the comfort line → `surplus ≤ 0` → no transfer → **growth stalls, no death**.
- Asymptotic: the closer to `k·max_size`, the smaller the step → the cap is always *reachable*, just slower
  for a bigger genetic target.
- **When adult size is reached** (`reserve = k·max_size`): transfer stops, surplus fuel simply accumulates
  in `body.energie` (whose ceiling `max_energie_individu` is now large because the body is large) and feeds
  reproduction. This is the intended juvenile-grow → adult-breed transition. **This is good and correct.**

Selection tunes `max_size`: an over-ambitious cap means the animal spends its whole life small, weak, and
sub-fertile → out-competed. No hard max-size parameter needed beyond the gene itself.

---

## 5. Birth — trigger + conserved transfer

**Derived per-baby costs (all paid from parent FUEL; parent reserve is never touched):**
```
baby_size    = birth_ratio * parent_size          # parent_size = parent.reserve / k
baby_reserve = k * baby_size  (= birth_ratio * parent.reserve)
baby_maxE    = max_energie_individu_init * (baby_size / r_collision_box_individu_init)
baby_fuel    = 0.6 * baby_maxE                     # mirrors current init (energie = 0.6·max_energie)
per_baby_cost = baby_reserve + baby_fuel           # taken from parent fuel
```

**Trigger (reproduce) — unchanged spirit, made affordability-aware:**
- Start gestation when `age > age_min_to_childbirth` **and** `fuel ≥ facteur_energie_creer_bb · max_energie`
  (existing gate, `body_V23.py:516`).
- `nbr_bb` = **how many complete babies the parent's spare fuel can pay for, where each baby costs its full
  `per_baby_cost = baby_reserve + baby_fuel`** (both the starter reserve AND the starter fuel are counted).
  The parent must also keep a survival floor for itself:
  ```
  spare_fuel = parent.energie - facteur_energie_creer_bb * parent.max_energie_individu   # fuel above the repro gate
  nbr_bb     = int(spare_fuel // per_baby_cost)                                           # only fully-fundable babies
  ```
  This replaces the current `facteur_energie_depensee_creer_bb` divisor at `body_V23.py:519`, which counted
  only a flat fuel cost and ignored the baby's reserve entirely.
- At delivery: `parent.energie -= nbr_bb * per_baby_cost`; each baby gets `reserve = baby_reserve`,
  `energie = baby_fuel`.
- If `nbr_bb == 0` (can't fully fund even one baby including its reserve) → no birth this cycle. Prevents
  underfunded/starving newborns.

**Conservation:** every joule a baby owns (reserve + fuel) came straight out of the parent's fuel. Parent's
reserve is untouched (reserves only grow). No creation, no parent insta-death (cost is a fraction of surplus
fuel, not the whole body), no starving newborn (starts at 0.6 of its own tank). Baby is *small* and grows up
toward its inherited (mutated) `max_size`.

---

## 6. Movement & attack scale with reserve/size (paid from fuel)

- **Move cost** — already scales via `move_energy_linear(r_collision / r_collision_init)` at `body_V23.py:487`.
  Since `r_collision` now equals the *current* (growing) size, this scales with reserve automatically. **No change.**
- **Attack damage** — already scales with `r_collision / r_collision_init` at `body_V23.py:750`. Tracks
  current size automatically. **No change.**
- **Attack cost** — currently size-independent (`body_V23.py:764`). **Change:** multiply by `(size / r_collision_init)`
  so a bigger body pays more to attack, capped by `max_energie_depensee_attack`.
- Damage (→ victim `vie`) and cost (→ attacker `fuel`) stay **decoupled and unequal** — different currencies,
  intentionally not the same amount.

---

## 7. Death — corpse carries the whole body

Every death lays **one** meat corpse of energy = `reserve + max(0, fuel) + seed_bank`:
- age death → fuel may be large → rich corpse.
- **starvation death → fuel ≈ 0, but reserve is intact → the skeleton is still edible** (this is the whole
  point: predation/scavenging always pays off).
- combat death → attacker lays the corpse at the killing blow.

**One-corpse-per-death guard:** add `body.carcass_dropped` (init `False`). The attacker sets it `True` when it
lays meat; the removal loop lays meat only `if not carcass_dropped`. This prevents the double-drop that would
otherwise happen when a killed individual (vie≤0) is later swept by the removal loop.

Conservation: reserve + leftover fuel + seed_bank all become plant-edible meat → nothing destroyed. (Meat then
rots and decomposes into plants via the already-implemented decomposition — full loop closed.)

---

## 8. How "reserve and fuel mutate"

They **don't** mutate directly — they are **state**, not heritable traits:
- `fuel` and `reserve` are set by birth, then evolve dynamically via eating / growth / spending / death.
- The single **heritable** knob behind them is **`max_size`** (the reserve cap), inherited via `deepcopy` and
  mutated by `mutate_size()`. A bigger `max_size` → the lineage *can* grow bigger reserves, if it can afford
  the fuel to fill them.
- `birth_ratio`, `k`, `growth_pace`, `seuil_croissance` are fixed globals, identical for all individuals.

---

## 9. Exact functions to change

`parameters_V23.py`
- **Add** `energie_par_taille`, `growth_pace`, `seuil_croissance`, `birth_ratio`.

`body_V23.py`
- `__init__` — **add** `self.max_size` (init = `r_collision_box_individu_init`); keep `r_collision_box_individu`
  as the *current* size.
- `initialize_individu` (line ~121) — **add** `self.reserve = k * r_collision_box_individu_init`,
  `self.carcass_dropped = False`; set initial `size = r_collision_box_individu_init`.
- **New** `update_size_from_reserve()` — set `size = reserve/k`; from it recompute `r_collision_box_individu`,
  `r_eat_box_individu`, `r_attack_box_individu`, `max_energie_individu`, `max_vie_individu` **proportionally**
  to their `_init` values. *(Behavior note: this makes the three boxes strict-proportional to size, replacing
  the current additive drift in `mutate_size` — a small, intended coherence change.)*
- `mutate_size` (line ~291) — **change** to mutate the **`max_size` gene** (the cap), not the current radius.
  Clamp `max_size ≥ 0.5`. Remove the direct box/`max_energie`/`max_vie` writes (now derived in `update_size_from_reserve`).
- **New** `grow()` metabolism step (§4) — call it once per tick per alive individual.
- `attack` (line ~764) — **change** cost to `max_energie_depensee_attack * valeur_sortie_brain * (size / r_collision_box_individu_init)`.
- `attack` (line ~757-761) — **change** corpse energy to `reserve + max(0,fuel) + seed_bank`; set `carcass_dropped = True`.
- `process_additional_outputs` / childbirth block (line ~516-526) — **change** trigger to affordability of
  `per_baby_cost` and pass the new per-baby budget to `create_bb`.

`ecosystem_V23_huge_scale.py`
- `create_bb` (line ~611) — **change** to set each baby's `reserve = baby_reserve` and `energie = baby_fuel`,
  and to debit the parent `nbr_bb * per_baby_cost` from fuel (parent reserve untouched).
- Main loop, alive branch (~line 736+) — **add** the `grow()` call each tick.
- Removal loop / death (line ~718-733) — **change** to lay corpse = `reserve + max(0,fuel) + seed_bank` for
  **both** age and starvation death, guarded by `if not body.carcass_dropped`.
- Initial population seeding — ensure `reserve` and `size` are set for t=0 individuals (one-time boundary
  seed, same status as initial plants).

`metrics_V23.py` (optional, later)
- Add mean `reserve`, mean current `size`, mean `max_size` to the aggregate pass so the paper can show the
  juvenile→adult growth and size evolution.

---

## 10. Open decisions before coding

1. Values of `energie_par_taille`, `growth_pace`, `seuil_croissance`, `birth_ratio` (defaults proposed above).
2. Confirm the box-proportionality change in §7 is acceptable (fixes the current additive drift — cleaner, but
   it *is* a behavior change to how eat/attack boxes relate to size).
3. Reserve strictly one-directional (never reclaimed as fuel when starving) — assumed **yes** (realistic,
   makes every corpse worth eating).
