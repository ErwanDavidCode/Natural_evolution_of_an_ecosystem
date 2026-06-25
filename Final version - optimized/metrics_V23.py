# -*- coding: utf-8 -*-
"""
Centralized, optional observability layer for the ecosystem simulation.

Design goals (deliberately simple and self-contained):
  * ONE entry point to SAVE a sample during the run:   Metrics.metrics_record_step()
  * ONE entry point to RENDER every figure at the end: Metrics.metrics_render()
  * A single master switch in parameters: `enable_metrics`.
  * Everything is namespaced `metrics_* / _metrics_* / METRICS_*` so the whole layer
    is greppable and removable: delete this file + the few `self.metrics.*` call sites
    in the ecosystem and the simulation behaves exactly as before.

Data flow:
  * Every sample we record (a) per-class/per-diet head-counts and (b) cheap aggregates
    over the live population: brain structure, body traits, feature adoption, mutation
    rates. Buffered here, flushed to CSV, reloaded into a single `data` dict at render.
  * The per-individual pickle history is read once, at render time, for the genetics panels.

Outputs (in ../Videos):
  * plot_evolution_entities.png  -> population per class/diet over time (legacy curve)
  * plot_phase_portrait.png      -> predator/prey phase portrait (hero figure)
  * plot_dashboard.png           -> stability + over-grazing + brain/morphology (vs generation)
  * plot_adaptation.png          -> diet composition + population + complexity + evolvability (vs time)
  * plot_brain.png               -> network size + memory + mutation rates + alpha (vs time)
  * plot_body.png                -> sensory ranges + sensorimotor adoption + traits + diet (vs time)
  * a scalar summary printed to stdout
"""

import os
import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt

from parameters_V23 import *


class Metrics:

    # ---- configuration ------------------------------------------------------
    METRICS_CSV_PATH = "./data/plot_evolution.csv"
    METRICS_OUTPUT_DIR = "../Videos"
    METRICS_CV_WINDOW = 50            # samples used for the rolling stability (CV) window
    METRICS_EXTINCTION_RATIO = 0.10   # population below this * mean  -> "near-extinction"
    METRICS_OVERGRAZE_RATIO = 0.05    # plants below this * peak       -> "over-grazed"

    # Per-sample aggregate columns (cheap averages over the live population), grouped.
    METRICS_BRAIN_COLUMNS = ["mean_connections", "mean_alpha", "mean_hidden",
                             "mean_sensors", "mean_actuators", "mean_memory"]
    METRICS_MUTATION_COLUMNS = ["mut_create_weight", "mut_add_neuron",
                                "mut_remove_neuron", "mut_modify_weight"]
    METRICS_BODY_COLUMNS = ["mean_size", "mean_speed", "mean_vision_range",
                            "mean_vision_angle", "mean_rotation", "mean_hearing",
                            "mean_diet", "mean_vision_parts"]
    METRICS_FEATURE_COLUMNS = ["frac_ears", "frac_know_size", "frac_know_diet",
                               "frac_know_energy", "frac_know_age", "frac_attack",
                               "frac_spit", "frac_sound", "frac_birth_ctrl"]

    # Output file names (attributes so metrics_finalize() can rename them per run).
    METRICS_FILE_POPULATION = "plot_evolution_entities.png"   # same name as the legacy plot
    METRICS_FILE_PHASE = "plot_phase_portrait.png"
    METRICS_FILE_DASHBOARD = "plot_dashboard.png"
    METRICS_FILE_ADAPTATION = "plot_adaptation.png"
    METRICS_FILE_BRAIN = "plot_brain.png"
    METRICS_FILE_BODY = "plot_body.png"

    def __init__(self, active=True):
        """`active` is the master switch (enable_metrics AND a full simulation)."""
        self.metrics_enabled = active
        # CSV column layout: temps + entity/diet head-counts + per-sample aggregates.
        self.metrics_class_columns = list(classes.values())                          # [0,1,2,3]
        self.metrics_diet_columns = [len(classes) + d for d in range(lvl_max_eat_scale + 1)]
        self.metrics_count_columns = self.metrics_class_columns + self.metrics_diet_columns
        self.metrics_agg_columns = (self.METRICS_BRAIN_COLUMNS + self.METRICS_MUTATION_COLUMNS
                                    + self.METRICS_BODY_COLUMNS + self.METRICS_FEATURE_COLUMNS)
        # In-memory buffers (flushed to CSV every `saving_rate`, reloaded at render time).
        self.metrics_time = []
        self.metrics_counts = {col: [] for col in self.metrics_count_columns}
        self.metrics_agg = {col: [] for col in self.metrics_agg_columns}
        if self.metrics_enabled:
            self._metrics_init_csv()

    # =========================================================================
    # COLLECTION  (called every tick during the simulation)
    # =========================================================================
    def metrics_record_step(self, temps, counts_per_class, population):
        """Buffer one sample = per-class/diet head-counts + aggregates over the live
        population, and periodically flush to CSV. No-op when metrics are disabled."""
        if not self.metrics_enabled:
            return
        if temps % sampling_rate == 0:
            self.metrics_time.append(temps)
            for col in self.metrics_count_columns:
                self.metrics_counts[col].append(counts_per_class[col])
            agg = self._metrics_compute_aggregates(population)
            for col in self.metrics_agg_columns:
                self.metrics_agg[col].append(agg[col])
        if temps % saving_rate == 0:
            self._metrics_flush_to_csv()

    def _metrics_compute_aggregates(self, population):
        """One pass over the live population -> dict of all aggregate columns (means, and
        fractions for the frac_* features). Returns nan everywhere on an empty population.
        Cost: one O(N) pass per sample (every `sampling_rate` ticks); the only non-trivial
        per-individual ops are count_nonzero(weights) and the memory-neuron count."""
        n = len(population) if population else 0
        if n == 0:
            return {col: float('nan') for col in self.metrics_agg_columns}
        acc = {col: 0.0 for col in self.metrics_agg_columns}
        for individu in population:
            br, bd = individu.brain, individu.body
            # brain structure
            acc["mean_connections"] += int(np.count_nonzero(br.matrice_poids))
            acc["mean_alpha"] += br.alpha
            acc["mean_hidden"] += len(br.valeurs_neurones) - br.nbr_entrees - br.nbr_sorties
            acc["mean_sensors"] += br.nbr_entrees
            acc["mean_actuators"] += br.nbr_sorties
            acc["mean_memory"] += sum(1 for f in br.activation_functions if f is not np.tanh)
            # evolved mutation probabilities
            acc["mut_create_weight"] += br.proba_creer_poid
            acc["mut_add_neuron"] += br.proba_ajouter_neurone
            acc["mut_remove_neuron"] += br.proba_supprimer_neurone
            acc["mut_modify_weight"] += br.proba_modifier_poid
            # body traits
            acc["mean_size"] += bd.r_collision_box_individu
            acc["mean_speed"] += bd.facteur_multiplicatif_deplacement
            acc["mean_vision_range"] += bd.vision_rayon
            acc["mean_vision_angle"] += bd.vision_demi_angle
            acc["mean_rotation"] += bd.max_rotation
            acc["mean_hearing"] += bd.ecoute_rayon
            acc["mean_diet"] += bd.regime
            acc["mean_vision_parts"] += bd.vision_nbr_parts
            # sensorimotor feature presence (-> fractions after the /n below)
            acc["frac_ears"] += "oreille" in bd.liste_entrees_supplementaires
            acc["frac_know_size"] += "know_size" in bd.liste_entrees_supplementaires_par_part
            acc["frac_know_diet"] += "know_diet" in bd.liste_entrees_supplementaires_par_part
            acc["frac_know_energy"] += "know_energy" in bd.liste_entrees_supplementaires_par_part
            acc["frac_know_age"] += "know_age" in bd.liste_entrees_supplementaires_par_part
            acc["frac_attack"] += "attaque" in bd.liste_sorties_supplementaires
            acc["frac_spit"] += "trophallaxy" in bd.liste_sorties_supplementaires
            acc["frac_sound"] += "bouche" in bd.liste_sorties_supplementaires
            acc["frac_birth_ctrl"] += "creer_bb" in bd.liste_sorties_supplementaires
        return {col: acc[col] / n for col in self.metrics_agg_columns}

    # =========================================================================
    # RENDERING  (called once at the end of the simulation)
    # =========================================================================
    def metrics_render(self, history_path=None):
        """Build every figure and print the scalar summary. No-op when disabled."""
        if not self.metrics_enabled:
            return
        self._metrics_flush_to_csv()                       # persist the tail of the buffer
        data = self._metrics_load_csv()
        if data is None or not data["time"]:
            return
        self._metrics_plot_population(data)
        self._metrics_plot_phase_portrait(data)
        self._metrics_plot_dashboard(data, history_path)
        self._metrics_plot_adaptation(data)
        self._metrics_plot_brain(data)
        self._metrics_plot_body(data)
        self._metrics_print_summary(data)

    def metrics_finalize(self, run_id):
        """Save the output figures under a shared '<name>_<run_id>' name, using the same
        simple os.rename the ecosystem uses for the video/history (one set of files per
        worthwhile run). No-op when disabled."""
        if not self.metrics_enabled:
            return
        for fname in (self.METRICS_FILE_POPULATION, self.METRICS_FILE_PHASE,
                      self.METRICS_FILE_DASHBOARD, self.METRICS_FILE_ADAPTATION,
                      self.METRICS_FILE_BRAIN, self.METRICS_FILE_BODY):
            src = os.path.join(self.METRICS_OUTPUT_DIR, fname)
            if os.path.exists(src):
                stem, ext = os.path.splitext(fname)
                os.rename(src, os.path.join(self.METRICS_OUTPUT_DIR, f"{stem}_{run_id}{ext}"))

    # =========================================================================
    # CSV helpers  (one row = temps + count columns + aggregate columns)
    # =========================================================================
    def _metrics_init_csv(self):
        with open(self.METRICS_CSV_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['temps'] + self.metrics_count_columns + self.metrics_agg_columns)

    def _metrics_flush_to_csv(self):
        if not self.metrics_time:
            return
        with open(self.METRICS_CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(self.metrics_time)):
                row = ([self.metrics_time[i]]
                       + [self.metrics_counts[col][i] for col in self.metrics_count_columns]
                       + [self.metrics_agg[col][i] for col in self.metrics_agg_columns])
                writer.writerow(row)
        self.metrics_time.clear()
        for col in self.metrics_count_columns:
            self.metrics_counts[col].clear()
        for col in self.metrics_agg_columns:
            self.metrics_agg[col].clear()

    def _metrics_load_csv(self):
        """Reload the CSV into a single dict: {time:[...], counts:{col:[...]}, agg:{col:[...]}}.
        Returns None if the file is missing/empty."""
        data = {"time": [],
                "counts": {col: [] for col in self.metrics_count_columns},
                "agg": {col: [] for col in self.metrics_agg_columns}}
        n_count = len(self.metrics_count_columns)
        n_total = 1 + n_count + len(self.metrics_agg_columns)
        try:
            with open(self.METRICS_CSV_PATH, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # skip header
                for row in reader:
                    if len(row) < n_total:
                        continue
                    data["time"].append(int(row[0]))
                    for i, col in enumerate(self.metrics_count_columns):
                        data["counts"][col].append(int(row[1 + i]))
                    for j, col in enumerate(self.metrics_agg_columns):
                        data["agg"][col].append(float(row[1 + n_count + j]))
        except (FileNotFoundError, StopIteration):
            return None
        return data

    # =========================================================================
    # FIGURE 1 - population per class/diet over time (the legacy curve, centralized here)
    # =========================================================================
    def _metrics_plot_population(self, data):
        """Figure 1 - number of entities per class/diet over time.
        x = simulation time in ticks (while-loop iterations), sampled every `sampling_rate` ticks.
        y = head-count alive at that tick, one line per series:
              * plant / trophallaxy / meat / individual -> count of that entity type;
              * 'Diet i' -> number of individuals whose diet level == i.
        Source = the per-tick `nbr_par_classes` counts buffered by metrics_record_step()."""
        time_series = data["time"]
        plt.figure(figsize=(10, 6))
        for col in self.metrics_count_columns:
            plt.plot(time_series, data["counts"][col],
                     label=self._metrics_column_label(col),
                     color=self._metrics_bgr_to_rgb(colors.get(col, (128, 128, 128))))
        plt.xlabel('Time [while loop iterations]')
        plt.ylabel('Number of individuals')
        plt.title('Number of individuals per class')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_POPULATION))
        plt.close()

    # =========================================================================
    # FIGURE 2 - predator/prey phase portrait (the hero figure)
    # =========================================================================
    def _metrics_plot_phase_portrait(self, data):
        """Figure 2 - predator/prey phase portrait (one point per recorded sample).
        x = plant count (prey) at that tick.
        y = individual count (predators) at that tick.
        color = time (tick); the RED dot marks the start (first sample).
        Same recorded series as Figure 1, plotted parametrically in time (plants vs
        individuals) instead of against the time axis: a closed loop = sustained
        oscillation, an inward spiral = oscillation damping toward a steady state."""
        time_series = data["time"]
        plant = data["counts"][classes["plant"]]
        individual = data["counts"][classes["individual"]]
        plt.figure(figsize=(7, 6))
        plt.plot(plant, individual, color='lightgray', linewidth=0.6, zorder=1)
        scatter = plt.scatter(plant, individual, c=time_series, cmap='viridis', s=8, zorder=2)
        plt.colorbar(scatter, label='Time [while loop iterations]')
        plt.scatter([plant[0]], [individual[0]], color='red', s=70, zorder=3,
                    edgecolors='black', label='start (t=0)')
        plt.xlabel('Plants (prey)')
        plt.ylabel('Individuals (predators)')
        plt.title('Predator-prey phase portrait')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_PHASE))
        plt.close()

    # =========================================================================
    # FIGURE 3 - dashboard: stability, over-grazing, and genetics-vs-generation
    # =========================================================================
    def _metrics_plot_dashboard(self, data, history_path):
        """Figure 3 - 2x2 dashboard.
        (0,0) Population stability: x = time (ticks); y = rolling coefficient of variation
              (std/mean over a METRICS_CV_WINDOW-sample sliding window) of the individual
              count. Dimensionless; lower = steadier.
        (0,1) Resource availability: x = time (ticks); y = plant count; dashed line =
              over-graze threshold = METRICS_OVERGRAZE_RATIO * peak plant count.
        (1,0) Brain complexification (per-individual history): x = generation (lineage depth,
              parent+1); y = number of non-zero weights in the brain matrix. Scatter = every
              individual that ever lived; black line = MEAN connections per generation.
        (1,1) Morphospace (per-individual history): x = body size (r_collision_box_individu);
              y = speed (facteur_multiplicatif_deplacement); color = diet (0=herbivore ..
              max=carnivore); RED = founders (generation 0). One point per individual that
              ever lived in the run (pooled over all of evolutionary time)."""
        time_series = data["time"]
        individual = np.asarray(data["counts"][classes["individual"]], dtype=float)
        plant = np.asarray(data["counts"][classes["plant"]], dtype=float)
        genetics = self._metrics_load_genetics(history_path)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle('Ecosystem metrics dashboard')

        # (0,0) stability: rolling coefficient of variation of the population
        cv = self._metrics_rolling_cv(individual, self.METRICS_CV_WINDOW)
        axes[0, 0].plot(time_series, cv, color='tab:blue')
        axes[0, 0].set_title('Population stability (rolling CV, lower = steadier)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('CV of individual count')
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) over-grazing: plant availability with the collapse threshold
        axes[0, 1].plot(time_series, plant, color='tab:green', label='plants')
        if plant.size and plant.max() > 0:
            axes[0, 1].axhline(self.METRICS_OVERGRAZE_RATIO * plant.max(),
                               color='tab:red', linestyle='--', label='over-grazed threshold')
        axes[0, 1].set_title('Resource (plant) availability')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Plant count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # (1,0) brain complexification & (1,1) morphospace -- need the pickle history
        if genetics is not None:
            gen, conn = genetics['generation'], genetics['connections']
            axes[1, 0].scatter(gen, conn, s=6, alpha=0.2, color='tab:purple')
            mean_gen, mean_conn = self._metrics_binned_mean(gen, conn)
            axes[1, 0].plot(mean_gen, mean_conn, color='black', linewidth=1.5, label='mean')
            axes[1, 0].set_title('Brain complexification')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Number of connections')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            scatter = axes[1, 1].scatter(genetics['size'], genetics['speed'],
                                         c=genetics['diet'], cmap='RdYlGn_r', s=8, alpha=0.5)
            fig.colorbar(scatter, ax=axes[1, 1], label='diet (0=herbivore .. max=carnivore)')
            founders = genetics['generation'] == 0
            if founders.any():
                axes[1, 1].scatter(genetics['size'][founders], genetics['speed'][founders],
                                   color='red', s=18, edgecolors='black', linewidths=0.3,
                                   zorder=3, label='start (generation 0)')
                axes[1, 1].legend(loc='best')
            axes[1, 1].set_title('Morphospace')
            axes[1, 1].set_xlabel('Body size')
            axes[1, 1].set_ylabel('Speed')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            for ax in (axes[1, 0], axes[1, 1]):
                ax.text(0.5, 0.5, 'history unavailable', ha='center', va='center')
                ax.set_axis_off()

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_DASHBOARD))
        plt.close(fig)

    # =========================================================================
    # FIGURE 4 - adaptation over time (everything on a shared time axis)
    # =========================================================================
    def _metrics_plot_adaptation(self, data):
        """Figure 4 - adaptation over time; x = time (ticks) in every panel.
        (0,0) Diet composition: stacked area of the individual head-count per diet level
              (0=herbivore .. max=carnivore). Shows the trophic succession.
        (0,1) Reference dynamics: individual & plant counts, to align the panels below
              with the boom/bust timeline.
        (1,0) Brain complexity: mean number of non-zero connections over the LIVE population.
        (1,1) Evolvability: mean meta-mutation rate `alpha` over the live population."""
        time_series = data["time"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle('Adaptation over time')

        # (0,0) diet composition (stacked area, same diet colors as the population figure)
        diet_counts = [data["counts"][col] for col in self.metrics_diet_columns]
        diet_colors = [self._metrics_bgr_to_rgb(colors.get(col, (128, 128, 128))) for col in self.metrics_diet_columns]
        diet_labels = [f"Diet {d}" for d in range(lvl_max_eat_scale + 1)]
        if diet_counts:
            axes[0, 0].stackplot(time_series, *diet_counts, colors=diet_colors, labels=diet_labels)
            axes[0, 0].legend(loc='upper right', fontsize=7, ncol=2)
        axes[0, 0].set_title('Diet composition (herbivore -> carnivore)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Individuals')
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) reference dynamics (so the brain panels can be aligned with the crashes)
        axes[0, 1].plot(time_series, data["counts"][classes["individual"]], color='tab:blue', label='individuals')
        axes[0, 1].plot(time_series, data["counts"][classes["plant"]], color='tab:green', label='plants')
        axes[0, 1].set_title('Population (reference timeline)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # (1,0) brain complexity over time (mean connections of the live population)
        axes[1, 0].plot(time_series, data["agg"]["mean_connections"], color='tab:purple')
        axes[1, 0].set_title('Brain complexity over time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Mean connections (live pop.)')
        axes[1, 0].grid(True, alpha=0.3)

        # (1,1) evolvability over time (mean meta-mutation rate alpha)
        axes[1, 1].plot(time_series, data["agg"]["mean_alpha"], color='tab:orange')
        axes[1, 1].set_title('Evolvability over time (meta-mutation rate)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean alpha (live pop.)')
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_ADAPTATION))
        plt.close(fig)

    # =========================================================================
    # FIGURE 5 - brain evolution over time
    # =========================================================================
    def _metrics_plot_brain(self, data):
        """Figure 5 - brain evolution over time; x = time (ticks) in every panel; all
        series are means over the LIVE population at each sample.
        (0,0) Network size: hidden neurons, #sensors (input neurons), #actuators (outputs).
        (0,1) Memory: number of hysteresis (latching/recurrent) neurons per brain.
        (1,0) Mutation rates: the evolved per-individual probabilities of create-weight,
              add-neuron, remove-neuron, modify-weight (what KIND of mutation is selected).
        (1,1) Evolvability: meta-mutation rate alpha (the overall mutability scale)."""
        t = data["time"]
        agg = data["agg"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle('Brain evolution over time')

        axes[0, 0].plot(t, agg["mean_hidden"], label='hidden neurons')
        axes[0, 0].plot(t, agg["mean_sensors"], label='sensors (inputs)')
        axes[0, 0].plot(t, agg["mean_actuators"], label='actuators (outputs)')
        axes[0, 0].set_title('Network size')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Mean count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(t, agg["mean_memory"], color='tab:brown')
        axes[0, 1].set_title('Memory units (hysteresis neurons)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Mean count')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(t, agg["mut_create_weight"], label='create weight')
        axes[1, 0].plot(t, agg["mut_add_neuron"], label='add neuron')
        axes[1, 0].plot(t, agg["mut_remove_neuron"], label='remove neuron')
        axes[1, 0].plot(t, agg["mut_modify_weight"], label='modify weight')
        axes[1, 0].set_title('Mutation rates (evolved probabilities)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Mean probability')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(t, agg["mean_alpha"], color='tab:orange')
        axes[1, 1].set_title('Evolvability (meta-mutation rate alpha)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean alpha')
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_BRAIN))
        plt.close(fig)

    # =========================================================================
    # FIGURE 6 - body & sensorimotor evolution over time
    # =========================================================================
    def _metrics_plot_body(self, data):
        """Figure 6 - body & sensorimotor evolution over time; x = time (ticks); all series
        are means/fractions over the LIVE population at each sample.
        (0,0) Sensory ranges (map units): mean vision range and hearing range (hearing
              starts at 0 and rises as ears are adopted).
        (0,1) Sensorimotor adoption: fraction of the population carrying each evolvable
              sense/actuator (ears, size/diet/energy sense, attack, spit, sound, birth-control).
        (1,0) Body traits relative to their initial value (x initial): size, speed, vision
              angle, rotation, eye-sectors -- all start at 1.0, so >1 = grew, <1 = shrank.
        (1,1) Mean diet level (0=herbivore .. max=carnivore)."""
        t = data["time"]
        agg = data["agg"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle('Body & sensorimotor evolution over time')

        # (0,0) sensory ranges (same map-distance units)
        axes[0, 0].plot(t, agg["mean_vision_range"], label='vision range')
        axes[0, 0].plot(t, agg["mean_hearing"], label='hearing range')
        axes[0, 0].set_title('Sensory ranges')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Mean range (map units)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) sensorimotor adoption (fraction of population)
        adoption = [("frac_ears", "ears"), ("frac_know_size", "size-sense"),
                    ("frac_know_diet", "diet-sense"), ("frac_know_energy", "energy-sense"),
                    ("frac_know_age", "age-sense"), ("frac_attack", "attack"),
                    ("frac_spit", "spit"), ("frac_sound", "sound"),
                    ("frac_birth_ctrl", "birth-control")]
        for col, lbl in adoption:
            axes[0, 1].plot(t, agg[col], label=lbl)
        axes[0, 1].set_title('Sensorimotor adoption')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Fraction of population')
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].legend(fontsize=7, ncol=2)
        axes[0, 1].grid(True, alpha=0.3)

        # (1,0) body traits relative to their initial value
        relative = [("mean_size", r_collision_box_individu_init, 'size'),
                    ("mean_speed", facteur_multiplicatif_deplacement_init, 'speed'),
                    ("mean_vision_angle", vision_demi_angle_init, 'vision angle'),
                    ("mean_rotation", max_rotation_init, 'rotation'),
                    ("mean_vision_parts", vision_nbr_parts_init, 'eye sectors')]
        for col, init_value, lbl in relative:
            series = np.asarray(agg[col]) / init_value if init_value else np.asarray(agg[col])
            axes[1, 0].plot(t, series, label=lbl)
        axes[1, 0].axhline(1.0, color='gray', linewidth=0.6, linestyle=':')
        axes[1, 0].set_title('Body traits (relative to initial value)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('x initial value')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        # (1,1) mean diet
        axes[1, 1].plot(t, agg["mean_diet"], color='tab:red')
        axes[1, 1].set_title('Mean diet (0=herbivore .. max=carnivore)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Mean diet level')
        axes[1, 1].set_ylim(0, lvl_max_eat_scale)
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(self.METRICS_OUTPUT_DIR, self.METRICS_FILE_BODY))
        plt.close(fig)

    # =========================================================================
    # Scalar summary (printed to stdout)
    # =========================================================================
    def _metrics_print_summary(self, data):
        time_series = data["time"]
        individual = np.asarray(data["counts"][classes["individual"]], dtype=float)
        plant = np.asarray(data["counts"][classes["plant"]], dtype=float)
        mean_ind = individual.mean() if individual.size else 0.0
        cv = self._metrics_rolling_cv(individual, self.METRICS_CV_WINDOW)
        mean_cv = np.nanmean(cv) if np.any(~np.isnan(cv)) else float('nan')

        # near-extinction = rising edges where the population enters the low band
        if mean_ind > 0 and individual.size > 1:
            low = individual < self.METRICS_EXTINCTION_RATIO * mean_ind
            near_extinction_events = int(np.sum((~low[:-1]) & (low[1:])))
        else:
            near_extinction_events = 0
        # over-grazing = fraction of samples with plants below the collapse threshold
        if plant.size and plant.max() > 0:
            overgraze_fraction = float(np.mean(plant < self.METRICS_OVERGRAZE_RATIO * plant.max()))
        else:
            overgraze_fraction = 0.0
        # last valid aggregates
        final_conn = self._metrics_last_valid(data["agg"]["mean_connections"])
        final_alpha = self._metrics_last_valid(data["agg"]["mean_alpha"])
        final_diet = self._metrics_last_valid(data["agg"]["mean_diet"])

        print("\n####################### METRICS SUMMARY #######################     - #metrics")
        print(f"Run survival:                {time_series[-1]} ticks")
        print(f"Individuals  peak/mean/min:  {int(individual.max())} / {mean_ind:.1f} / {int(individual.min())}")
        print(f"Stability (mean rolling CV): {mean_cv:.3f}   (lower = steadier)")
        print(f"Near-extinction events:      {near_extinction_events}")
        print(f"Over-grazing time fraction:  {overgraze_fraction:.2%}")
        print(f"Brain  final conn / alpha:   {final_conn:.1f} / {final_alpha:.3f}")
        print(f"Final mean diet (0=herb):    {final_diet:.2f}")
        print("###############################################################\n")

    # =========================================================================
    # small reusable helpers
    # =========================================================================
    def _metrics_column_label(self, col):
        if col < len(classes):
            for name, value in classes.items():
                if value == col:
                    return name
        return f'Diet: {col - len(classes)}'

    @staticmethod
    def _metrics_bgr_to_rgb(bgr):
        if isinstance(bgr, list):     # eatables store [young, old]; take the young color
            bgr = bgr[0]
        return (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

    @staticmethod
    def _metrics_rolling_cv(values, window):
        arr = np.asarray(values, dtype=float)
        cv = np.full(arr.shape, np.nan)
        for i in range(window, arr.size + 1):
            w = arr[i - window:i]
            m = w.mean()
            if m > 0:
                cv[i - 1] = w.std() / m
        return cv

    @staticmethod
    def _metrics_binned_mean(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0:
            return [], []
        unique_x = np.unique(x)
        mean_y = np.array([y[x == ux].mean() for ux in unique_x])
        return unique_x, mean_y

    @staticmethod
    def _metrics_last_valid(values):
        """Last non-nan value of a list, or nan if none."""
        valid = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        return valid[-1] if valid else float('nan')

    def _metrics_load_genetics(self, history_path):
        """Read the per-individual pickle and extract genetics arrays, one entry per
        individual that EVER lived in the run (all that died + those alive at the end).
        Returns None on any failure (optional panels must never break a run).
        Fields: generation, connections (non-zero brain weights), size, speed, diet."""
        if not history_path:
            return None
        try:
            with open(history_path + '.pkl', 'rb') as f:
                history = pickle.load(f)
            generation, connections, size, speed, diet = [], [], [], [], []
            for ind in history.values():
                b, br = ind.body, ind.brain
                generation.append(b.generation)
                connections.append(int(np.count_nonzero(br.matrice_poids)))
                size.append(b.r_collision_box_individu)
                speed.append(b.facteur_multiplicatif_deplacement)
                diet.append(b.regime)
        except Exception as exc:
            print(f"[metrics] genetics panels skipped (history unreadable): {exc}")
            return None
        if not generation:
            return None
        return {'generation': np.asarray(generation), 'connections': np.asarray(connections),
                'size': np.asarray(size), 'speed': np.asarray(speed), 'diet': np.asarray(diet)}
