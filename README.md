# Natural Evolution of an Ecosystem

A Python simulation of natural selection in a 2D continuous world. Organisms are controlled by evolving neural networks and compete for energy. Complex behaviors (predation, cooperation, dietary specialization) emerge purely from selection pressure — nothing is hardcoded.

---

## How to Run

**Requirements:** Python 3.9+

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulation
cd "Final version - optimized"
python ecosystem_V23_huge_scale.py
```

**In VS Code:** open `ecosystem_V23_huge_scale.py` and press `F5` (uses the provided `.vscode/launch.json` which sets the correct working directory automatically).

Output (video + population graph) is saved to the `Videos/` folder.

---

## Configuration

All parameters are in `Final version - optimized/parameters_V23.py`.

**World & simulation**

| Parameter | Default | Effect |
|---|---|---|
| `taille_carte` | 1200 | World size in units |
| `size_modification` | 1 | Global scale factor (2 = half resolution, faster; scales hitboxes, speeds, vision) |
| `duree_simulation` | 20,000,000 | Max ticks before auto-stop |
| `factor_recording_video` | 4 | Record 1 frame every N ticks (lower = smoother, slower) |
| `video_scale` | 2 | Video output resolution divisor (2 = half res) |

**Population**

| Parameter | Default | Effect |
|---|---|---|
| `nbr_individus_init` | 100 | Starting individual count |
| `max_individu` | 100,000 | Hard population cap |
| `nbr_plantes_init` | 90 | Starting plant count |
| `nbr_min_plant_init` | 9 | Min plants maintained; new seeds spawn when below this |
| `age_maximum` | 5,000 | Max individual lifespan in ticks |
| `lvl_max_eat_scale` | 6 | Diet spectrum (0 = all omnivores; 6 = full herbivore↔carnivore range) |

**Energy & combat**

| Parameter | Default | Effect |
|---|---|---|
| `max_energie_individu_init` | 150 | Starting max energy |
| `max_vie_individu_init` | 200 | Starting max health |
| `facteur_energie_creer_bb` | 0.61 | Energy % required to reproduce (61% of max) |
| `facteur_energie_depensee_creer_bb` | 0.31 | Energy % consumed per birth (31% of max) |
| `max_attack_damage` | 100 | Max damage per attack hit |
| `compteur_injured` | 20 | Ticks slowed down after being hit |
| `facteur_slowed_down` | 0.2 | Speed multiplier when injured (20% of normal) |

**Brain mutations (per reproduction)**

| Parameter | Default | Effect |
|---|---|---|
| `nbr_mutations_brain` | 5 | Mutation operations per offspring |
| `proba_modifier_poid_init` | 0.75 | Probability of modifying a weight |
| `proba_creer_poid_init` | 0.45 | Probability of creating a new connection |
| `proba_modifier_biais_init` | 0.60 | Probability of modifying a bias |
| `proba_ajouter_neurone_init` | 0.07 | Probability of adding a hidden neuron |
| `proba_supprimer_neurone_init` | 0.07 | Probability of removing a hidden neuron |
| `alpha_init` | 0.2 | Meta-mutation rate (how fast mutation probs themselves evolve) |

Stop anytime with `Ctrl+C` — this is caught gracefully and triggers video/history export.

**To inspect a specific individual:** set `characteristics_ID_individu = <ID>` in `parameters_V23.py` (ID visible in the video), then rerun. Its brain and traits will be displayed.

---

## Energy Loop

The ecosystem runs a near-closed energy loop. Solar energy (`solar_energy` per tick) is injected into plants, which store it and reproduce when they cross an energy threshold — splitting their energy in two. Organisms eat plants or other organisms, transferring that energy into their body reserves. When an organism dies it leaves a meat eatable containing its remaining energy, which other organisms can consume. Trophallaxie (food sharing) moves energy between living organisms without loss. Movement, rotation, attacks, and gestation all drain energy; that energy is simply removed from the system (heat dissipation analog).

The only artificial injections are: the constant solar input, and the minimum-plant seeding rule (when plants drop below `nbr_min_plant_init`, new seeds spawn with `seed_energy` energy from nowhere). Without this floor, a herbivore extinction cascade would permanently collapse the plant population and end the simulation. Everything else — growth, predation, reproduction, decay — conserves energy between entities.

---

## Expected Behaviors

Given enough time (thousands of ticks), the following typically emerge:

- **Herbivores** cluster near plant spawns; **carnivores** intercept them. The `lvl_max_eat_scale` parameter controls how pronounced this split becomes.
- **Predation**: larger organisms deal more damage and tend to specialize as carnivores.
- **Trophallaxie** (food sharing): organisms can emit energy packets for others to eat, enabling cooperation.
- **Sound communication**: organisms can emit and hear sounds with directional localization, allowing alarm calls or coordination.
- **Population cycles**: boom when food is abundant, crash when scarce.
- **Extinction risk**: if `lvl_max_eat_scale` is high and the carnivore population overshoots, prey extinction cascades into predator extinction. The simulation auto-restarts if `start_again_until_alive_pop = True`.

---

## How It Works

### Body & Energy

Each organism has physical traits (size, speed, vision range/angle, hearing range, diet level) that mutate at reproduction. Energy is the universal currency:

- **Gained** by eating plants (herbivores gain more) or meat (carnivores gain more), or via trophallaxie (diet-neutral).
- **Lost** by moving (scales with size and distance), rotating, attacking, and gestating.
- **Health** regenerates from energy. When energy hits 0, health drains instead. Death occurs at 0 health or max age.

Larger organisms are stronger but burn energy faster. Speed, vision, and hearing all evolve independently.

### Brain (Neural Network)

Each organism runs a **sparse recurrent neural network**:

- **Inputs:** normalized energy, vision sectors (distance + entity type encoded as 2D coordinates), optional hearing (directional sound), optional extra features (size awareness, diet sensing).
- **Outputs:** velocity, rotation angle, and optionally: attack intensity, sound emission, food sharing, reproduction trigger.
- **Hidden neurons** use `tanh` or a **hysteresis** activation (a memory unit: latches to +1 or -1 until pushed past the threshold) — the choice itself mutates.
- One forward pass per organism per tick: `outputs = W.T @ neurons + biases`.

### Mutation & Evolution

Reproduction is asexual (cloning + mutation). Both brain and body mutate independently:

- **Brain mutations:** add/remove neurons, add/remove/modify weights and biases, change activation functions. All 11 mutation probabilities are themselves mutable per individual (**meta-evolution**) — organisms can evolve to become more or less exploratory.
- **Body mutations:** ±random changes to size, speed, vision, hearing, diet level. Size scales all hitboxes and energy capacity proportionally.

There is no crossover — lineages are purely vertical. Generation count is tracked per individual.

### Performance Optimizations

The simulation is designed to run at large scale (100k+ individuals):

- **Quadtree spatial index** (pyqtree): vision and collision queries run in O(log n) instead of O(n²).
- **Swap-and-pop lists**: O(1) removal of dead organisms from all entity lists.
- **Vectorized neural forward pass**: numpy matrix multiply for the full network in one operation.
- **Lazy evaluation**: dead/satiated/uninjured organisms skip irrelevant computation paths.
- **Shuffled round-robin**: processing order is periodically shuffled to avoid first-mover bias in reproduction races.
