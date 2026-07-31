# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import cv2
from pyqtree import Index
import time
import networkx as nx
from collections import deque
import sys
import warnings
import pickle
import os
import glob
import signal
from functools import partial
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from pydub.generators import Sine
import moviepy.editor as mp
import math

from parameters_V23 import *
import brain_V23 as brain
import body_V23 as body
import observer_V23 as observer
import eatable_V23 as eatable
import metrics_V23 as metrics

import functions_V23 as functions


warnings.filterwarnings("ignore")

# =============================================================================
# Classes
# =============================================================================

class Obj_individu:
    def __init__(self):
        """Les attributs de individu sont les variables qui seront modifiés par des méthodes de la classe individu && brain et corps. Il y a aussi l'ID de l'individu"""
        self.brain = brain.Brain()
        self.body = body.Body()

        self.ID = None #0 par défaut avant d'être attibué




class Ecosystem:
    def __init__(self):
        """Les attributs de ecosystem sont les variables qui seront modifiés par des méthodes de la classe ecosystem && les variables évolutives qui caractérisent l'écosystème (liste des individus, liste des plantes, quadtree)"""
        self.quadtree = Index(bbox=(0, 0, taille_carte, taille_carte)) #création du quadtree
        self.compteur_mort = 0
        self.compteur_trophallaxie = 0
        self.seed_collected_zoochorie = 0
        self.seed_droped_zoochorie = 0
        
        self.nbr_min_plant = nbr_min_plant_init

        self.historique_path = "./data/historique_individus" #path pour l'historique des individus par défaut

        #self.taille_video = taille_carte
        self.video_scale = video_scale
        self.taille_video = int(taille_carte / self.video_scale)

        self.font = ImageFont.truetype("./Roboto/Roboto-Regular.ttf", max(int(13/(size_modification*self.video_scale)),5)) #font = ImageFont.load_default() #police par défaut et taille non réglable par défaut
        
        self.rr_start = 0 #round-robin start index to avoid always starting with the same individuals between shuffles (if shuffle is not done every turn)

        # necessary for the swap and pop in O(1)
        self.idx_eatable = {}
        self.idx_plantes = {}
        self.idx_perishables = {}

        # Creation of video and history
        if liste_ID_alone_simulation == []: #if traditionnal simulation with all new individuals
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(r'../Videos/video.mp4', fourcc, fps, (self.taille_video, self.taille_video))
            # Initialiser une piste audio vide
            self.audio_out = AudioSegment.silent(duration=0)
            # Initialiser le shelve
            if individu_to_be_used is None:
                self.reinitialize_history()

        else: #if alone simulation
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(r'../Videos/video_alone.mp4', fourcc, fps, (self.taille_video, self.taille_video))
            # Initialiser une piste audio vide
            self.audio_out = AudioSegment.silent(duration=0)

        # Centralized, optional metrics/observability layer (toggle with enable_metrics in parameters)
        self.metrics = metrics.Metrics(active=(enable_metrics and liste_ID_alone_simulation == []))

    def reinitialize_history(self):
        # Supprimer les fichiers existants du shelve
        for ext in ['', '.db', '.bak', '.dat', '.dir', '.pkl']:
            full_path = "./data/historique_individus" + ext
            if os.path.exists(full_path):
                os.remove(full_path)
        with open("./data/historique_individus.pkl", 'wb') as f:
            pickle.dump({}, f)


    def initialisation(self, liste_individus_selectionnes=[], matrice_poids = None, biais_neurones = None):
        """créer les plantes et les individus initiaux
        IN : Si rien n'est passé en argument, on instantie une nouvelle population complete de base.
        IN : Si on passe en argument une liste d'individus, on instantie uniquement ces individus"""
        self.population_etat_fin_simulation = "Population en vie"
        
        #création des eatables
        self.liste_eatable = [] #liste qui rescence les eatables actuellement en vie
        self.liste_plantes = [] #liste qui rescence uniquement les plantes. BUT optimiser le calcul de la mort des plantes
        self.liste_perishables = [] 
        self.idx_eatable.clear()
        self.idx_plantes.clear()
        self.idx_perishables.clear()
        # si idx_individus existe déjà
        if hasattr(self, "idx_individus"):
            self.idx_individus.clear()

        for plante in range(nbr_plantes_init):
            self.add_eatable("plant") 

        #initialisation d'individus pour lancer une simulation seul ou complete dépendant de si "liste_individus_selectionnes" est [] ou non vide
        self.generate_individu_init(liste_individus_selectionnes, matrice_poids = matrice_poids, biais_neurones = biais_neurones)
    

    def add_eatable(self, eatable_type, energy=None, eatable_parent=None, position=None, age=None):
        """Ajoute une plante. eatable_parent: pour faire apparaitre a coté du parent. position: pour faire apparaitre à une position donnée. classe: pour faire apparaitre une plante d'une classe donnée"""
        if eatable_parent is None and position is None: #spawn aléatoire
            # Spawn aléatoire si x et y sont None
            #eatable_parent_position = random.choice(self.liste_plantes)[0].position if self.liste_plantes else [taille_carte/2, taille_carte/5] #on prend une plante au hasard
            eatable_to_add = eatable.Eatable()
            a = random.uniform(1, taille_carte-2)
            b = random.uniform(1, taille_carte-2)
            eatable_to_add.position = [a, b]
            
            # ajouter energy parent : if wind 
            if energy is not None:
                eatable_to_add.energy = energy
            if age is not None:
                eatable_to_add.age = age

            # swap and pop
            entry = (eatable_to_add, eatable_type)  # eatable_type est déjà une string chez toi
            functions.sp_add(self.liste_eatable, self.idx_eatable, entry)
            if eatable_type in ("meat", "trophallaxy"):
                functions.sp_add(self.liste_perishables, self.idx_perishables, entry)

            # Ajout des plantes au quadtree
            bbox = (a - eatable_to_add.r_hit_box_eatable, b - eatable_to_add.r_hit_box_eatable, a + eatable_to_add.r_hit_box_eatable, b + eatable_to_add.r_hit_box_eatable)  # bbox définie comme un point car c'est son centre qui nous intéresse
            self.quadtree.insert(item=entry, bbox=bbox)

        elif position is None: #spawn proche de eatable_parent (si plante fait bb par exemple)
            #spawn proche de plante_parent
            eatable_to_add = eatable.Eatable()
            # Choisir un facteur aléatoire pour la répartition de l'énergie
            variation = random.uniform(0.45, 0.55)
            #energy of the baby plant
            eatable_to_add.energy = eatable_parent.energy * variation
            eatable_parent.energy *= (1 - variation)  #on divise l'energie de la plante parent par 2 car a fait un bb et a partagé son énergie

            #position bb plant : dispersion longue distance (position aléatoire) ou près du parent
            if random.random() < proba_dispersion_plante:
                a = random.uniform(1, taille_carte - 2)
                b = random.uniform(1, taille_carte - 2)
            else:
                a = random.uniform(max(0, eatable_parent.position[0] - range_max_spawn_plant), min(taille_carte, eatable_parent.position[0] + range_max_spawn_plant))
                b = random.uniform(max(0, eatable_parent.position[1] - range_max_spawn_plant), min(taille_carte, eatable_parent.position[1] + range_max_spawn_plant))
            eatable_to_add.position = [a, b]
            
            # swap and pop
            entry = (eatable_to_add, eatable_type)  # eatable_type est déjà une string chez toi
            functions.sp_add(self.liste_eatable, self.idx_eatable, entry)
            if eatable_type in ("meat", "trophallaxy"):
                functions.sp_add(self.liste_perishables, self.idx_perishables, entry)

            # Ajout des plantes au quadtree
            bbox = (a - eatable_to_add.r_hit_box_eatable, b - eatable_to_add.r_hit_box_eatable, a + eatable_to_add.r_hit_box_eatable, b + eatable_to_add.r_hit_box_eatable)  # bbox définie comme un point car c'est son centre qui nous intéresse
            self.quadtree.insert(item=entry, bbox=bbox)

        # else : #spawn à la position donnée (si on tue qqn par exemple)
        #     if eatable_type in ["trophallaxy", "meat"]:
        #         quantity_bouffe_taille_max = int(size // bouffe_taille_max) #si l'individu / trophallaxie est assez gros on lache plusieurs bout de viande pour favoriser la coopération potentielle
        #         taille_bouffe_restante = size % bouffe_taille_max #taille restante
        #         liste_eatable_to_spawn = [bouffe_taille_max for _ in range(quantity_bouffe_taille_max)] #liste des tailles de bouffe à spawn
        #         # On ajoute la taille de bouffe restantes à la liste si il y en a
        #         if taille_bouffe_restante > 0:
        #             liste_eatable_to_spawn.append(taille_bouffe_restante)
        #         #Spawn des eatable
        #         for eatable_taille in liste_eatable_to_spawn:
        #             eatable_to_add = eatable.Eatable()
        #             a = random.uniform(max(0, position[0] - (5*size)), min(taille_carte, position[0] + (5*size)))
        #             b = random.uniform(max(0, position[1] - (5*size)), min(taille_carte, position[1] + (5*size)))
        #             eatable_to_add.position = [a, b]
        #             eatable_to_add.energy = (eatable_taille/size) * energy #on divise l'energie de la bouffe par la taille de la bouffe pour avoir l'energie par unité de taille
        #             eatable_to_add.r_hit_box_eatable = eatable_taille
                    
        #             # swap and pop
        #             entry = (eatable_to_add, eatable_type)  # eatable_type est déjà une string chez toi
        #             functions.sp_add(self.liste_eatable, self.idx_eatable, entry)
        #             if eatable_type in ("meat", "trophallaxy"):
        #                 functions.sp_add(self.liste_perishables, self.idx_perishables, entry)

        #             # Ajout des plantes au quadtree
        #             bbox = (a - eatable_to_add.r_hit_box_eatable, b - eatable_to_add.r_hit_box_eatable, a + eatable_to_add.r_hit_box_eatable, b + eatable_to_add.r_hit_box_eatable)  # bbox définie comme un point car c'est son centre qui nous intéresse
        #             self.quadtree.insert(item=entry, bbox=bbox)
        
        else:  # spawn à une position donnée (meat / trophallaxy)
            eatable_to_add = eatable.Eatable()
            x, y = self.wrap_xy(position[0], position[1]) #evite de faire spawn hors de la map
            eatable_to_add.position = [x, y]
            #eatable_to_add.position = [position[0], position[1]]
            eatable_to_add.energy = max(0.0, 0.0 if energy is None else energy)

            entry = (eatable_to_add, eatable_type)
            functions.sp_add(self.liste_eatable, self.idx_eatable, entry)
            if eatable_type in ("meat", "trophallaxy"):
                functions.sp_add(self.liste_perishables, self.idx_perishables, entry)

            x, y = eatable_to_add.position
            r = eatable_to_add.r_hit_box_eatable
            self.quadtree.insert(item=entry, bbox=(x - r, y - r, x + r, y + r))


        #ajout à la liste des plantes pour la gestion des enfants uniquement
        if eatable_type == "plant":
            functions.sp_add(self.liste_plantes, self.idx_plantes, entry)



    def decompose_perishable(self, eatable):
        """DECOMPOSITION: recycle l'énergie d'un perissable mort de VIEILLESSE (viande, trophallaxie ou
        plante) en plantes, au lieu de la perdre. Chaque plante porte au MAXIMUM energy_per_decomposed_plant:
        on crée autant de plantes "pleines" que possible, puis une derniere plante pour l'énergie restante.
        La somme des bébés plantes vaut exactement energy (conservation). Elles apparaissent dans un petit
        cercle (range_decomposition) autour du défunt (la branche 'position' de add_eatable les replace dans
        la carte et les enregistre comme plantes)."""
        energy = max(0.0, eatable[0].energy)
        if energy <= 0:
            return
        # liste des énergies: des plantes pleines (energy_per_decomposed_plant) + le reste éventuel
        energies = [energy_per_decomposed_plant] * int(energy // energy_per_decomposed_plant)
        reste = energy - sum(energies)
        if reste > 0:
            energies.append(reste)
        x0, y0 = eatable[0].position[0], eatable[0].position[1]
        for energy_plante in energies:
            if len(self.liste_plantes) >= max_plantes: #HARD CAP (god rule, comme max_individu): au plafond on ne fait PAS apparaitre la plante. L'énergie non recyclée est un puits assumé (perf > conservation stricte au plafond)
                break
            x = x0 + random.uniform(-range_decomposition, range_decomposition)
            y = y0 + random.uniform(-range_decomposition, range_decomposition)
            self.add_eatable("plant", energy=energy_plante, position=(x, y))


    def generate_individu_init(self, liste_individus_selectionnes, matrice_poids = None, biais_neurones = None):
        """création des individus"""
        #on créer nos individus au début
        self.liste_individus = [] #liste qui rescence les individus actuellement en vie
        self.idx_individus = {} #for the swap and pop in O(1)

        #initialisation d'individus pour lancer une nouvelle simulation normale
        if liste_individus_selectionnes == [] :
            if matrice_poids is None and biais_neurones is None and individu_to_be_used == None:
                for num_individu in range(nbr_individus_init):
                    individu = Obj_individu() #creation de l'individu, il s'initialise deja avec body et brain
                    body = individu.body # creation de la reference body
                    #body.position = [random.uniform(3*taille_carte/7, 4*taille_carte/7), random.uniform(2*taille_carte/5, 3*taille_carte/5)]
                    body.position = [random.uniform(1, taille_carte-1), random.uniform(1, taille_carte-1)]

                    individu.ID = num_individu #cet ID permet d'identifier chaque individu
                    body.mutate_body_init() #on mute les caractéristiques du body (toutes ont une chance sauf les mutations features : neurones d'entrées et de sorties)
                    body.regime = random.randint(0, lvl_max_eat_scale) #FOUNDERS span herbivore..carnivore so the predator niche exists at t=0 (diversity, not scripted behavior). Babies still INHERIT their parent's regime (create_bb), so this only seeds the initial population
                    body.initialize_individu() #on initialise l'individu apres pour lui donner sa vie et energie de départ ( = f(son seuil max)) si mutations physiques de taille il y a eu
                    individu.brain.mutate_brain()

                    #self.liste_individus.append(individu)
                    functions.sp_add(self.liste_individus, self.idx_individus, individu)
                    bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                    self.quadtree.insert(item=(individu, "individual"), bbox=bbox)
        
            #si on veut lancer une simulation avec des cerveaux spécifiques
            if matrice_poids is not None and biais_neurones is not None and individu_to_be_used == None: 
                for num_individu in range(nbr_individus_init):
                    individu = Obj_individu() #creation de l'individu, il s'initialise deja avec body et brain
                    body = individu.body # creation de la reference body
                    body.position = [random.uniform(0, taille_carte), random.uniform(0, taille_carte)]
                    individu.ID = num_individu #cet ID permet d'identifier chaque individu
                    body.mutate_body_init() #on mute les caractéristiques du body (toutes ont une chance sauf les mutations features : neurones d'entrées et de sorties)
                    body.initialize_individu() #on initialise l'individu apres pour lui donner sa vie et energie de départ ( = f(son seuil max)) si mutations physiques de taille il y a eu

                    individu.brain.matrice_poids = copy.copy(matrice_poids) #on copie le cerveau spécifié
                    individu.brain.biais_neurones = copy.copy(biais_neurones) #on copie le cerveau spécifié
                    individu.brain.valeurs_neurones = np.zeros(len(biais_neurones)) #on initialise les valeurs des neurones à 0
                    individu.brain.mutate_brain()

                    #self.liste_individus.append(individu)
                    functions.sp_add(self.liste_individus, self.idx_individus, individu)
                    bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                    self.quadtree.insert(item=(individu, "individual"), bbox=bbox)
            

        #intitialisation des individus spécifiés pour une simulation seul (liste_individus_selectionnes != [])
        else :
            for i, individu in enumerate(liste_individus_selectionnes):
                body = individu.body # creation de la reference body
                body.initialize_individu() 
                body.position = [random.uniform(0, taille_carte), random.uniform(0, taille_carte)]
                individu.ID = i
                #self.liste_individus.append(individu)
                functions.sp_add(self.liste_individus, self.idx_individus, individu)
                bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                self.quadtree.insert(item=(individu, "individual"), bbox=bbox)
            # If we use a precise individual as a copy for the simulation, we need to reinitialize the history after the individuals are picked up and initialized
            if individu_to_be_used != None:
                self.reinitialize_history() 



    def deplacement_dynamique(self, nvl_position_x, nvl_position_y, individu_obj):
        """Simule une carte continue / ronde. Si on passe une bordure on arrive de l'autre côté de la carte
        delta rerésente de combien on depasse la carte.
        individu_obj est l'individu qui se dépplace"""
        #bordure map
        #bordure gauche
        if nvl_position_x <= 0:
            delta = np.abs(nvl_position_x - 0)
            nvl_position_x = taille_carte  - delta
        #bordure droite
        elif nvl_position_x >= taille_carte:
            delta = np.abs(nvl_position_x - taille_carte)
            nvl_position_x = 0 + delta
        #bordure haute
        if nvl_position_y >= taille_carte:
            delta = np.abs(nvl_position_y - taille_carte)
            nvl_position_y = 0 + delta
        #bordure basse
        elif nvl_position_y <= 0:
            delta = np.abs(nvl_position_y - 0)
            nvl_position_y = taille_carte - delta
        

        #colision
        bbox = (nvl_position_x - individu_obj.body.r_collision_box_individu, nvl_position_y - individu_obj.body.r_collision_box_individu, nvl_position_x + individu_obj.body.r_collision_box_individu, nvl_position_y + individu_obj.body.r_collision_box_individu)
        quadtree_intersected = self.quadtree.intersect(bbox)
        # Parcourir les entités visibles
        for entity in quadtree_intersected:
            entity_type = entity[1]
            if entity_type == "individual":
                ex, ey = (entity[0].body.position[0], entity[0].body.position[1])
                # Exclure l'entité elle-même
                if entity[0] is individu_obj:
                    continue
                distance = np.sqrt((ex - nvl_position_x) ** 2 + (ey - nvl_position_y) ** 2)
                if distance <= individu_obj.body.r_collision_box_individu + entity[0].body.r_collision_box_individu:
                    # colision new pos : on ne bouge pas
                    nvl_position_x, nvl_position_y = individu_obj.body.position[0], individu_obj.body.position[1]

        return nvl_position_x, nvl_position_y



    def handle_ctrl_c(self, simulation_seul_param, liste_individus_a_ajouter_a_history, signum, frame):
            """Gère l'interruption de la simulation par Ctrl+C. Ajoute les individus restants à l'historique et quitte le programme."""
            print("\nCtrl+C detected. Exiting gracefully...")
            # Ajouter les individus restants à l'historique si ce n'est pas une simulation seule : on ne veut pas modifier les stats lors de la simulation seule
            if simulation_seul_param == False:
                self.add_to_history(self.liste_individus + liste_individus_a_ajouter_a_history)
        
            self.metrics.metrics_render(self.historique_path) #centralized metrics figures + summary
            self.out.release() #cloture de la video
            self.export_and_merge_audio() #fusionner audio et video

            tps_final = time.time()
            print("Temps de la simulation : ", tps_final - tps_init)
            # On affiche l'état de la population à la fin de la simulation
            print("Nombre d'enfants crées : ", self.compteur_BB)
            print(self.population_etat_fin_simulation)
            print("Nombre de graines collectées par zoochrorie : ", self.seed_collected_zoochorie)
            print("Nombre de graines déposées par zoochrorie : ", self.seed_droped_zoochorie)
            # Quitter le programme
            sys.exit(0)



    # def get_quadtree_tot(self, individu):
    #     """Renvoie une liste contenant les entités dans le champ de vision de l'individu et un peu plus. Exclu l'individu lui même"""
    #     x, y = individu.body.position[0], individu.body.position[1]
        
    #     rayon_max_hit_box = max(individu.body.r_eat_box_individu, individu.body.r_collision_box_individu, individu.body.r_attack_box_individu, individu.body.ecoute_rayon)

    #     def calculer_points_intermediaires():
    #         angles = np.linspace(individu.body.teta - individu.body.vision_demi_angle, individu.body.teta + individu.body.vision_demi_angle, round(individu.body.vision_demi_angle) + 1) #On couvre tous les degrés/2
    #         points = [(x + individu.body.vision_rayon * np.cos(np.deg2rad(angle)), y + individu.body.vision_rayon * np.sin(np.deg2rad(angle))) for angle in angles]
    #         return points

    #     points_intermediaires = calculer_points_intermediaires()
    #     x_points = [x] + [x - rayon_max_hit_box] + [x + rayon_max_hit_box] + [px for px, _ in points_intermediaires]
    #     y_points = [y] + [y - rayon_max_hit_box] + [y + rayon_max_hit_box] + [py for _, py in points_intermediaires]
    #     min_x, min_y, max_x, max_y = min(x_points), min(y_points), max(x_points), max(y_points)
    #     bbox = (min_x, min_y, max_x, max_y)
    #     quadtree_intersected = self.quadtree.intersect(bbox)

    #     # Exclude the entity itself
    #     quadtree_intersected.remove((individu, "individual"))

    #     return quadtree_intersected, bbox
    



    def get_quadtree_tot(self, individu):
        """BBox minimale qui englobe: individu (rayon_max_hit_box) + cone de vision (secteur)
        IT RETURNS THE INDIVIDUAL ITSELF ! NEED TO SKIP IF IT'S THE INDIVIDUAL ITSELF WHEN CALLING THIS FUNCTION"""
        def _angle_in_interval(a, start, end):
            # intervalle [start, end] sur un cercle (wrap 360)
            if start <= end:
                return start <= a <= end
            else:
                return a >= start or a <= end
            

        body = individu.body
        x, y = body.position[0], body.position[1]

        rmax = max(body.r_eat_box_individu, body.r_collision_box_individu,
                body.r_attack_box_individu, body.ecoute_rayon)  # comme ton code :contentReference[oaicite:5]{index=5}

        R = body.vision_rayon
        teta = body.teta % 360.0
        demi = body.vision_demi_angle

        # Si le champ couvre tout le cercle (ou plus), bbox = disque complet
        if 2.0 * demi >= 360.0:
            min_x = x - max(R, rmax)
            max_x = x + max(R, rmax)
            min_y = y - max(R, rmax)
            max_y = y + max(R, rmax)
            bbox = (min_x, min_y, max_x, max_y)
            out = self.quadtree.intersect(bbox)
            return out, bbox

        start = (teta - demi) % 360.0
        end   = (teta + demi) % 360.0

        # Angles à tester: bords + cardinaux s'ils sont dans l'intervalle
        angles = [start, end]
        for a in (0.0, 90.0, 180.0, 270.0):
            if _angle_in_interval(a, start, end):
                angles.append(a)

        # Points candidats pour la bbox du cone
        xs = [x - rmax, x + rmax, x]  # inclut l'individu
        ys = [y - rmax, y + rmax, y]

        for a in angles:
            rad = math.radians(a)
            xs.append(x + R * math.cos(rad))
            ys.append(y + R * math.sin(rad))

        bbox = (min(xs), min(ys), max(xs), max(ys))
        out = self.quadtree.intersect(bbox)
        return out, bbox



    def get_entities_in_range(self, individu):
        """Renvoie les entités comestibles, attaquables et visibles dans l'ensemble des entités retournées par le quadtree"""
        body = individu.body

        # listes de sortie
        liste_eatable_entity = []
        list_reachable_entity = []
        list_visible_entity = []
        liste_audible_entity = []

        # Constantes locales (évite des lookups + recalc)
        bx, by = body.position[0], body.position[1]

        vision2 = body.vision_rayon * body.vision_rayon
        attack2 = body.r_attack_box_individu * body.r_attack_box_individu
        ecoute2 = body.ecoute_rayon * body.ecoute_rayon

        # stomach_ok = body.energie < facteur_energie_eat * body.max_energie_individu
        # regime = body.regime

        # Cône de vision (calculé une seule fois)
        teta = body.teta % 360.0
        demi = body.vision_demi_angle
        start = (teta - demi) % 360.0
        end   = (teta + demi) % 360.0

        # Interaction avec entités de proximité (quadtree)
        quadtree_intersected, bbox = self.get_quadtree_tot(individu)

        for entity in quadtree_intersected:
            entity_type = entity[1]
            ent = entity[0]
            
            # if individual ITSELF skip
            if entity_type == "individual" and ent is individu:
                continue


            # Position de l'entité
            if entity_type == "individual":
                ex, ey = ent.body.position[0], ent.body.position[1]
            else:
                ex, ey = ent.position[0], ent.position[1]

            dx = ex - bx
            dy = ey - by
            dist2 = dx*dx + dy*dy

            # On calcule angle/distance uniquement si besoin
            angle = None
            distance = None

            # --------------------
            # Eatable (pas besoin d'angle)
            # --------------------
            if entity_type != "individual":
                eat_r = body.r_eat_box_individu + ent.r_hit_box_eatable
                if dist2 <= eat_r * eat_r:
                    # is_eatable = False
                    # if entity_type == "trophallaxy":
                    #     is_eatable = True
                    # elif lvl_max_eat_scale == 0:
                    #     is_eatable = True
                    # else:
                    #     if entity_type == "plant":
                    #         is_eatable = (regime < lvl_max_eat_scale)
                    #     elif entity_type == "meat":
                    #         is_eatable = (regime > 0)

                    # if is_eatable:
                        # liste_eatable_entity.append(entity)

                    liste_eatable_entity.append(entity)

            # --------------------
            # Individu : attaquable / écoutable (angle utile)
            # --------------------
            if entity_type == "individual":
                if dist2 <= attack2:
                    if angle is None:
                        angle = (math.degrees(math.atan2(dy, dx)) % 360.0)
                    if distance is None:
                        distance = math.sqrt(dist2)
                    list_reachable_entity.append([entity, distance, angle])

                if dist2 <= ecoute2 and "bouche" in ent.body.liste_sorties_supplementaires:
                    if angle is None:
                        angle = (math.degrees(math.atan2(dy, dx)) % 360.0)
                    if distance is None:
                        distance = math.sqrt(dist2)
                    liste_audible_entity.append([entity, distance, angle])

            # --------------------
            # Vision (distance² puis cône)
            # --------------------
            if dist2 <= vision2:
                if angle is None:
                    angle = (math.degrees(math.atan2(dy, dx)) % 360.0)

                # Test "in_cone" robuste (wrap 0/360), identique à ton pattern
                if start <= end:
                    in_cone = (start <= angle <= end)
                else:
                    in_cone = (angle >= start or angle <= end)

                if in_cone:
                    if distance is None:
                        distance = math.sqrt(dist2)
                    list_visible_entity.append([entity, distance, angle])

        return liste_eatable_entity, list_reachable_entity, list_visible_entity, liste_audible_entity

  

    def eat_energy_plant(self, x, energy):
        """Énergie assimilée en mangeant une plante. régime x=0 (herbivore) = max d'efficacité; diet_specialization règle la pente, plant_digest_efficiency la richesse de la plante."""
        energy = max(0, energy)
        if lvl_max_eat_scale == 0:
            return energy * plant_digest_efficiency
        return (1 - diet_specialization * x / lvl_max_eat_scale) * energy * plant_digest_efficiency


    def eat_energy_meat(self, x, energy):
        """Énergie assimilée en mangeant de la viande. régime x=max (carnivore) = max d'efficacité; diet_specialization règle la pente, meat_digest_efficiency la richesse de la viande."""
        energy = max(0, energy)
        if lvl_max_eat_scale == 0:
            return energy * meat_digest_efficiency
        return (1 - diet_specialization * (lvl_max_eat_scale - x) / lvl_max_eat_scale) * energy * meat_digest_efficiency


    def can_eat(self, regime, eatable_type):
        # Tout le monde peut manger plante/viande/trophallaxie: le régime ne règle QUE l'efficacité (eat_energy_*), pas l'accès. La spécialisation reste rentable via l'efficacité
        return True



    def create_bb(self, individu, nbr_bb):
        """Create nbr_bb babies from a parent. The parent pays each baby's reserve (its body) + starter fuel
        out of its OWN FUEL (its reserve is never touched) -> energy is conserved. The baby is born small and
        grows over life toward its inherited max_size gene."""
        body = individu.body

        # RESET DES VALEURS
        body.bb_being_created = False #we are done with the BB creation, we can from now on create another one
        body.gestation = 0 #reset the gestation timer
        body.facteur_multiplicatif_deplacement = body.old_facteur_multiplicatif_deplacement #we reset the facteur multiplicatif deplacement to the old one
        body.nbr_bb = 0 #reset the number of BB

        # Per-baby cost from the parent's CURRENT state (fuel changed during gestation), then clamp to what is
        # actually affordable now so the parent's fuel can never go negative (strict conservation).
        baby_reserve, baby_fuel = body.birth_reserve_fuel()
        per_baby_cost = baby_reserve + baby_fuel
        if per_baby_cost <= 0:
            return
        nbr_bb = min(nbr_bb, int(body.energie // per_baby_cost))
        if nbr_bb < 1:
            return
        body.energie -= nbr_bb * per_baby_cost #parent pays from fuel only

        for _ in range(nbr_bb):
            self.compteur_BB += 1

            bb = copy.deepcopy(individu) #copie du parent
            body_bb = bb.body
            brain_bb = bb.brain

            bb.ID = self.ID_BB
            self.ID_BB += 1 #on incrémente l'ID pour le prochain BB
            #modifications génétiques body
            body_bb.mutate_body(brain_bb)
            body_bb.initialize_individu() #reset l'état (age, gestation ...) et donne une taille de base
            # override with the paid-for starter body: reserve and fuel come straight from the parent's fuel
            body_bb.reserve = baby_reserve
            body_bb.energie = baby_fuel
            body_bb.update_size_from_reserve() #boxes/max_energie/max_vie reflect the baby's (small) size
            body_bb.vie = body_bb.max_vie_individu
            #modifications génétiques brain
            brain_bb.mutate_brain()
            brain_bb.mutate_probas()
            brain_bb.mutate_alpha()
            #initialisation des valeurs
            angle_bb = np.deg2rad(random.uniform(0,360))
            body_bb.position[0] += (2*body_bb.r_collision_box_individu + body.r_collision_box_individu + np.ceil(pas_de_temps*1*body.facteur_multiplicatif_deplacement)) * np.cos(angle_bb) #no baby spawn in the parent if it spawns toward him
            body_bb.position[1] += (2*body_bb.r_collision_box_individu + body.r_collision_box_individu + np.ceil(pas_de_temps*1*body.facteur_multiplicatif_deplacement)) * np.sin(angle_bb)
            body_bb.position[0], body_bb.position[1] = self.wrap_xy(body_bb.position[0], body_bb.position[1])
            body_bb.generation = body.generation + 1 #c'est la génération suivante
            # Ajouter dans la liste du quadtree l'individu et à la liste des individus
            #self.liste_individus.append(bb)
            functions.sp_add(self.liste_individus, self.idx_individus, bb)

            bbox = (body_bb.position[0] - body_bb.r_collision_box_individu, body_bb.position[1] - body_bb.r_collision_box_individu, body_bb.position[0] + body_bb.r_collision_box_individu, body_bb.position[1] + body_bb.r_collision_box_individu)
            self.quadtree.insert(item=(bb,"individual"), bbox=bbox)
            

    def size_plant_calculator(self, num_intervals=5):
        """do the calculation for the size of the plant and its energy"""
        size_factors = [(0.8 + 0.6 * i)/size_modification for i in range(num_intervals)]  # Génère les tailles [1, 1, 1.5, ..., 3.5] #SIZE MODIFCATION
        energy_thresholds = [(0.45 + 0.1 * i) for i in range(num_intervals)]  # Génère les seuils [0.45, 0.55, ..., 1.05] facteur multplicatif
        return size_factors, energy_thresholds
    

    def wrap_xy(self, x, y):
        # carte continue : remet toujours dans [0, taille_carte)
        return (x % taille_carte), (y % taille_carte)


    def jeu(self, simulation_seul_param):
        """c'est la simulation du jeu, on balaie tous les individus à chaque fois"""
        
        self.compteur_BB = 0
        self.ID_BB = nbr_individus_init
        liste_individus_a_ajouter_a_history = []
        temps = 0  #unité non définie

        # Définir le nombre d'intervalles et les tailles associées des PLANTES
        #self.size_factors, self.energy_thresholds = self.size_plant_calculator(num_intervals=5)

        signal.signal(signal.SIGINT, partial(self.handle_ctrl_c, simulation_seul_param, liste_individus_a_ajouter_a_history)) # Capture Ctrl+C

        #while temps <= duree_simulation and len(self.liste_individus) :#and time.time() < time.mktime(end_time)> 0:
        while temps < duree_simulation and self.liste_individus:
            #affichage du temps en simulation d'entrainement
            if temps % 50 == 0 and simulation_seul_param == False: #tous les 5 pas de temps
                print(f"temps : {temps}         Nbr individu : {len(self.liste_individus)}") #         Compteur mort : {self.compteur_mort}         Compteur trophallaxie : {self.compteur_trophallaxie}")
            #affichage du temps en simulation seul
            elif temps % 100 == 0 and simulation_seul_param == True: #tous les 5 pas de temps
                print("temps :", temps)
                
            temps += pas_de_temps


            # Réinitialisation dico pour compter nbr individus par classe
            self.nbr_par_classes = {classe: 0 for classe in classes.values()} 
            for regime in range(lvl_max_eat_scale + 1):
                self.nbr_par_classes[regime + len(classes)] = 0


            #on mélange la liste des individus. BUT : ne pas parcourir tjrs dans le même sens la liste : On évite ainsi ques les mêmes individus vont des BB
            if temps % time_to_shuffle == 0:
                random.shuffle(self.liste_individus)
                self.idx_individus = functions.rebuild_index(self.liste_individus) # swap and pop
                self.rr_start = 0  # optionnel : repartir propre

            nombre_individus = len(self.liste_individus)
            if self.liste_individus: #avoid mudulo 0
                self.rr_start = (self.rr_start + 1) % nombre_individus # round-robin start index update to avoid always starting with the same individuals between shuffles (if shuffle is not done every turn)
            to_remove_individu = [] # list to kill individual after the loop (allows not to do a "list()" that copies all individual at each time step)
            # We go over every individual
            for k in range(nombre_individus):
                i = (self.rr_start + k) % nombre_individus
                individu = self.liste_individus[i]


                vivant = True #si l'individu meurt il n'a plus de volonté. On ne calcul donc pas sa vision par exemple. C'est de l'optimisation
                #creation de la reference body & brain
                body = individu.body
                brain = individu.brain


                #meurt si l'individu n'a plus d'energie ou plus de vie
                if body.vie <= 0 or body.age >= age_maximum:
                    vivant = False
                    #self.liste_individus.remove(individu) #on enlève de la liste des individus
                    # Swap and pop
                    to_remove_individu.append(individu)
                    #functions.sp_remove(self.liste_individus, self.idx_individus, individu)
                    # ind_item = (individu, "individual")
                    # bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                    # self.quadtree.remove(item=ind_item, bbox=bbox)
                    if not simulation_seul_param:
                        liste_individus_a_ajouter_a_history.append(individu)

                    # Corpse on ANY death (age OR starvation): reserve (the body) + leftover fuel + banked seeds.
                    # Fuel may be NEGATIVE (starvation over-spends it) -> keep it raw so death doesn't re-create energy;
                    # floor the whole corpse at 0. carcass_dropped guards against a double corpse if killed in combat this tick.
                    if not body.carcass_dropped:
                        body.carcass_dropped = True
                        self.add_eatable("meat", energy=max(0.0, body.reserve + body.energie + body.seed_bank), position=(body.position[0], body.position[1]))

                # only considering alive individual for no useless calculations
                if vivant == True:

                    # Metabolisme de base: cout de rester en vie, avant le test de famine
                    body.basal_metabolism()

                    # Gain life each turn according to it's lvl of energy. Full stomach => fast regeneration !
                    body.vie = min(body.max_vie_individu, body.vie + max(0, body.energie) / body.max_energie_individu) #gain de vie proportionnel à l'energie
                    
                    # Lose life if energy is too low
                    if body.energie <= 0 :
                        body.vie -= abs(body.energie) #on perd  de la vie si on a plus d'energie 
                        body.is_losing_life = True

                    # On parcourt le quadtree pour rechercher les entité à portée
                    liste_eatable_entity, list_reachable_entity, list_visible_entity, liste_audible_entity = self.get_entities_in_range(individu)
                    

                    # INDIVIDU MANGE
                    inf_limit_to_eat = 1e-1
                    for eatable in liste_eatable_entity:
                        
                        entity, type_entity = eatable

                        # ------------------------
                        # ZOOCHORIE (attraper graines au contact)
                        # ------------------------
                        if type_entity == "plant" and random.random() < seed_attach_prob and body.seed_bank < seed_bank_max :
                            # on prélève un petit "paquet de graines" sur la plante (pas d'énergie créée)
                            take = min(seed_packet_energy, max(0.0, entity.energy))
                            if take > 0:
                                body.seed_bank += take
                                entity.energy -= take
                                self.seed_collected_zoochorie += 1

                                # si la plante tombe à 0, on la supprime comme d'habitude
                                if entity.energy <= inf_limit_to_eat + 1e-3:
                                    functions.sp_remove(self.liste_eatable, self.idx_eatable, eatable)
                                    functions.sp_remove(self.liste_plantes, self.idx_plantes, eatable)
                                    x, y = entity.position[0], entity.position[1]
                                    r = entity.r_hit_box_eatable
                                    self.quadtree.remove(item=eatable, bbox=(x - r, y - r, x + r, y + r))

                        # ------------------------
                        # MANGE
                        # ------------------------
                        stomach_room = body.max_energie_individu - body.energie
                        if stomach_room <= inf_limit_to_eat:
                            break  # estomac plein

                        # check if eatable (regime doesn't allow herbivorous to eat meat and vice-vers)
                        if not self.can_eat(body.regime, type_entity):
                            continue

                        # énergie assimilable si on mange tout (linéaire en entity.energy)
                        if type_entity == "plant":
                            gain_full = self.eat_energy_plant(body.regime, entity.energy)
                        elif type_entity == "meat":
                            gain_full = self.eat_energy_meat(body.regime, entity.energy)
                        else:  # trophallaxy
                            gain_full = entity.energy

                        # Peut pas manger des micros miettes
                        if gain_full < inf_limit_to_eat:
                            continue

                        # on assimile au plus ce qu'il reste dans l'estomac
                        gain = stomach_room if stomach_room < gain_full else gain_full
                        body.energie += gain

                        # consommation partielle : fraction brute retirée (linéarité). Car la consommation n'est pas linéaire et dépend du régime
                        entity.energy -= (gain / gain_full) * entity.energy

                        # compteurs
                        if type_entity == "plant":
                            body.compteur_plant_eaten += 1
                        elif type_entity == "meat":
                            body.compteur_meat_eaten += 1

                        # suppression uniquement si entièrement consommé
                        if entity.energy <= inf_limit_to_eat + 1e-3: #j'ajoute 1e-9 pour être sur que ce soit supprimé
                            functions.sp_remove(self.liste_eatable, self.idx_eatable, eatable)
                            if type_entity == "plant":
                                functions.sp_remove(self.liste_plantes, self.idx_plantes, eatable)
                            elif type_entity in ("meat", "trophallaxy"):
                                functions.sp_remove(self.liste_perishables, self.idx_perishables, eatable)

                            x, y = entity.position[0], entity.position[1]
                            r = entity.r_hit_box_eatable
                            self.quadtree.remove(item=eatable, bbox=(x - r, y - r, x + r, y + r))

                    
                    # CREER BEBE
                    # GESTATION CURRENT BABY
                    if body.bb_being_created: 
                        #le temps de gestation avance si bb en cours de création
                        body.gestation += 1 # Temps pour enfanter avance
                    # NEW BABY (one at the time)
                    if body.age > age_min_to_childbirth and body.energie >= facteur_energie_creer_bb*body.max_energie_individu and len(self.liste_individus) < max_individu and not simulation_seul_param and not "creer_bb" in body.liste_sorties_supplementaires and not body.bb_being_created:
                        #lance la gestation seulement si on peut financer au moins un bébé complet (reserve + fuel)
                        nbr_bb = body.compute_nbr_bb()
                        if nbr_bb >= 1:
                            body.bb_being_created = True
                            body.nbr_bb = nbr_bb
                            #we are slower when creating a bb
                            body.old_facteur_multiplicatif_deplacement = body.facteur_multiplicatif_deplacement
                            body.facteur_multiplicatif_deplacement = 0.6/size_modification
                    if body.bb_being_created and body.gestation >= body.duree_gestation and len(self.liste_individus) < max_individu:
                        #create_bb débite le fuel du parent (reserve + fuel de chaque bébé): énergie conservée
                        self.create_bb(individu, body.nbr_bb)
                        


                    # Reset bruit emis à chaque tour
                    body.reset_bruit_emis()
                    # FEED FORWARD
                    liste_entree = body.fonction_vision(list_visible_entity, liste_audible_entity)  
                    liste_sortie = brain.propagation_avant(liste_entree)
                    #everything not related with moving first (the individual do the static thing first and then move, so he does smth related to what he really sees) (attack, trophallaxie ...)
                    body.process_additional_outputs(self, simulation_seul_param, individu, liste_sortie, list_reachable_entity) 

                    #capturing the noise
                    if son_individu != None and son_individu == individu.ID:
                        self.noise_freq, self.noise_intensity = individu.body.get_bruit_entendu()

                    #reset noise earing : it's only for one turn
                    body.reset_bruit_entendu()


                    # INDIVIDU MOVE
                    # calculer la nouvelle position de l'individu
                    nvl_position_x, nvl_position_y, deplacement = body.move(liste_sortie)
                    # gérer les sorties de carte
                    nvl_position = self.deplacement_dynamique(nvl_position_x, nvl_position_y, individu)  
                    # Retirer de la liste du quadtree l'individu
                    bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                    self.quadtree.remove(item=(individu,"individual"), bbox=bbox)
                    # GROWTH: grow here (between the paired remove/insert) so the size change rides the insert
                    # already happening -> quadtree stays consistent, no extra quadtree operation
                    body.grow()
                    # Ajouter de la liste du quadtree l'individu
                    bbox = (nvl_position[0] - body.r_collision_box_individu, nvl_position[1] - body.r_collision_box_individu, nvl_position[0] + body.r_collision_box_individu, nvl_position[1] + body.r_collision_box_individu)
                    self.quadtree.insert(item=(individu,"individual"), bbox=bbox) #on l'ajoute dans la liste du quadtree
                    # Assigner la position finale            
                    body.position = [nvl_position[0], nvl_position[1]]

                    # ZOOCHORIE
                    if body.seed_bank >= seed_energy and random.random() < seed_drop_prob and len(self.liste_plantes) < max_plantes: #len < max_plantes: HARD CAP (god rule, comme max_individu). Au plafond -> pas de graine ET seed_bank conservé (pas de perte), la graine sortira plus tard
                        x = body.position[0] - (2 + r_hit_box_eatable_init + body.r_eat_box_individu) * math.cos(math.radians(body.teta))
                        y = body.position[1] - (2 + r_hit_box_eatable_init + body.r_eat_box_individu) * math.sin(math.radians(body.teta))
                        self.add_eatable("plant", energy=seed_energy, position=(x, y))
                        body.seed_bank -= seed_energy
                        self.seed_droped_zoochorie += 1


                    #vieilli de 1 unité d'age
                    body.age += pas_de_temps
                    #ralentissement tappé
                    if body.compteur_injured > 0:
                        body.compteur_injured -= 1
                    else:
                        body.compteur_injured = 0

                    # Compting the classes
                    self.nbr_par_classes[classes["individual"]] += 1
                    self.nbr_par_classes[body.regime + len(classes)] += 1

            # We kill every individual that are dead this tick of time 
            for individu in to_remove_individu:
                # get the bbox of the individual to delete him from the quadtree
                body = individu.body
                bbox = (body.position[0] - body.r_collision_box_individu, body.position[1] - body.r_collision_box_individu, body.position[0] + body.r_collision_box_individu, body.position[1] + body.r_collision_box_individu)
                # delete him from the list and the quadtree
                functions.sp_remove(self.liste_individus, self.idx_individus, individu)
                self.quadtree.remove(item=(individu,"individual"), bbox=bbox)

            if temps % 1500 == 0:
                self.nbr_min_plant = max(nbr_min_plant_final, self.nbr_min_plant - pas_de_temps)


            # Traitement des plantes : croissance avec l'énergie solaire, reproduction et mort par vieillesse
            taille_liste_plantes = len(self.liste_plantes) 
            if taille_liste_plantes > 0:

                # Calcul de l'énergie solaire distribuée équitablement
                energy_per_plant = min(solar_energy / taille_liste_plantes, gain_max_energy_per_turn)
                
                to_remove_plants = []
                for i in range(taille_liste_plantes):  # Utilisation d'une copie implicite
                    eatable = self.liste_plantes[i]
                    plant = eatable[0]
                    
                    # Croissance et reproduction
                    if len(self.liste_plantes) < max_plantes:
                        plant.energy += energy_per_plant  # Ajout de l'énergie solaire

                        # Reproduction si l'énergie dépasse le seuil
                        if plant.energy >= energy_plant_bb:
                            self.add_eatable("plant", eatable_parent=plant)

                    # Mort par vieillesse
                    if plant.age >= age_plant_max:
                        to_remove_plants.append(eatable)
                    
                    # grow plants
                    plant.grow()

                    # count
                    self.nbr_par_classes[classes[eatable[1]]] += 1 
                
                # deleting (décomposition: l'énergie d'une plante morte de vieillesse est recyclée en plantes)
                for eatable in to_remove_plants:
                    self.decompose_perishable(eatable)
                    self.delete_plant(eatable)
                        

            # Ajouter des plantes si le nombre est inférieur au minimum requis
            while len(self.liste_plantes) < self.nbr_min_plant:
                self.add_eatable("plant")

            
            # Add a plant at a random location on the map (simulate the wind that brings seeds)
            if temps % 500 == 0 and self.liste_plantes:
                plant_to_move = random.choice(self.liste_plantes)
                self.delete_plant(plant_to_move)
                self.add_eatable("plant", energy=plant_to_move[0].energy, age=plant_to_move[0].age)
                

            to_remove_perish = []
            # Processing perishables (meat, trophallaxy) in a single pass
            for perishable in self.liste_perishables:
                if perishable[0].age >= age_eatable_perish:
                    to_remove_perish.append(perishable)
                # grow
                perishable[0].grow()
                # count
                self.nbr_par_classes[classes[perishable[1]]] += 1 
                    
            # deleting
            for perishable in to_remove_perish:
                # décomposition: l'énergie d'une viande/trophallaxie pourrie est recyclée en plantes (conservée)
                self.decompose_perishable(perishable)
                # remove des listes (swap&pop)
                functions.sp_remove(self.liste_perishables, self.idx_perishables, perishable)
                functions.sp_remove(self.liste_eatable, self.idx_eatable, perishable)
                bbox = (
                    perishable[0].position[0] - perishable[0].r_hit_box_eatable, 
                    perishable[0].position[1] - perishable[0].r_hit_box_eatable,
                    perishable[0].position[0] + perishable[0].r_hit_box_eatable, 
                    perishable[0].position[1] + perishable[0].r_hit_box_eatable
                )
                self.quadtree.remove(item=perishable, bbox=bbox)
                del perishable # Suppression de l'objet pour libérer la mémoire



            #Ajout des individus à l'historique des que la taille des morts est suffisament grande (but: ne pas a ouvrir et fermer constamment le fichier)
            if len(liste_individus_a_ajouter_a_history) > nbr_iterations_shelve_update and simulation_seul_param == False:
                self.add_to_history(liste_individus_a_ajouter_a_history)
                liste_individus_a_ajouter_a_history = []


            if temps % factor_recording_video  == 0:
                # # Ajoutez la création de la frame à chaque itération
                img = self.create_image_from_frame()  # Méthode pour créer une image à partir de la frame
                self.out.write(img)
                
                # Suppression explicite de l'image pour libérer de la mémoire
                del img

            # Son
            self.export_sound()  # Méthode pour exporter le son à chaque itération

            # Collecte des données pour le plot (centralized metrics layer; self-gates when disabled)
            self.metrics.metrics_record_step(temps, self.nbr_par_classes, self.liste_individus, self.liste_eatable)


        # Ajout des individus à l'historique
        self.add_to_history(self.liste_individus + liste_individus_a_ajouter_a_history) if simulation_seul_param == False else None

        # Render every metric/figure from the centralized layer (reads the just-completed history)
        self.metrics.metrics_render(self.historique_path)

        # Si le run dépasse time_to_save_video steps, on sauvegarde la vidéo, l'historique et les
        # plots sous un même ID de run unique "_<run_id>". Un jeu de fichiers par run intéressant,
        # donc plusieurs sauvegardes en laissant tourner la nuit (1 sauvegarde par run > seuil).
        if temps > time_to_save_video and not simulation_seul_param:
            run_id = int(time.time())  # ID unique partagé par la vidéo, l'historique et les plots
            # Save Video
            new_video_path = f'../Videos/video_{run_id}.mp4'
            self.out.release()  # Relâcher le fichier actuel
            os.rename(r'../Videos/video.mp4', new_video_path)  # Renommer le fichier
            # Save History
            self.historique_path = f'./data/historique_individus_{run_id}'
            for fpath in glob.glob('./data/historique_individus.*'):
                ext = os.path.splitext(fpath)[1]
                os.rename(fpath, self.historique_path + ext)
            # Save metric figures
            self.metrics.metrics_finalize(run_id)

        self.out.release() #cloture de la video
        self.export_and_merge_audio() #fusionner audio et video
        

        if not self.liste_individus: #Si plus aucun individu en vie
            self.population_etat_fin_simulation = "Popoulation éteinte"

        if simulation_seul_param == False:        
            print("Nombre d'enfants crées : ", self.compteur_BB)


    def delete_plant(self, eatable):
        """ Delete a plant from quadtree, list_plant and list_eatable"""
        plant = eatable[0]
        #self.liste_plantes.remove(eatable)
        #self.liste_eatable.remove(eatable)
        # swap and pop
        functions.sp_remove(self.liste_plantes, self.idx_plantes, eatable)
        functions.sp_remove(self.liste_eatable, self.idx_eatable, eatable)

        bbox = (
            plant.position[0] - plant.r_hit_box_eatable, 
            plant.position[1] - plant.r_hit_box_eatable,
            plant.position[0] + plant.r_hit_box_eatable, 
            plant.position[1] + plant.r_hit_box_eatable
        )
        self.quadtree.remove(item=eatable, bbox=bbox)
        del eatable # Suppression de l'objet pour libérer la mémoire



    
    def export_sound(self):
        """Export the sound to a WAV file"""
        if son_individu != None:
            # Si pas de son
            if self.noise_intensity <= 0:
                # Générer un silence
                duration_ms = int(1000 / fps)  # Durée en ms pour une frame
                silence = AudioSegment.silent(duration=duration_ms)  # Génération d'un segment de silence
                self.audio_out += silence  # Ajouter le silence à la sortie audio
            # Si son
            else:
                freq = np.clip(self.noise_freq, -1, 1) * 4000 + 4400 #freq entre [400, 8400] Hz
                intensity = -36 + (np.clip(self.noise_intensity, 0, 1) * 36) #intensité entre [-36, 0] dB
                duration_ms = int(1000 / fps)  # Durée en ms pour une frame
                sine_wave = Sine(freq).to_audio_segment(duration=duration_ms, volume=intensity)
                self.audio_out += sine_wave
            # Exporter l'audio à chaque itération
            self.audio_out.export('../Videos/output_audio.wav', format="wav")

    def export_and_merge_audio(self):
        # Ajouter le son à la vidéo si demandée
        if son_individu != None:
            self.audio_out.export(f'../Videos/output_audio.wav', format="wav")
            # Fusionner la vidéo et l'audio
            video_clip = mp.VideoFileClip("../Videos/video.mp4")
            audio_clip = mp.AudioFileClip("../Videos/output_audio.wav")
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile("../Videos/output_with_sound.mp4", codec='libx264')



    # Population logging + all plotting now live in the centralized, optional Metrics
    # layer (metrics_V23.py): self.metrics.metrics_record_step() during the run and
    # self.metrics.metrics_render() at the end. Remove that file + the few
    # self.metrics.* call sites to drop the whole observability layer.



    def create_image_from_frame(self):
        """Extrait x et y de chaque entité de la frame en question.
        Frame de la forme : [liste_individus, liste_eatable] de la frame en question"""
        
        def eclaircir_couleur_rgb(r, g, b, facteur=0.5):
            """ Éclaircir une couleur RGB en mélangeant avec du blanc.
            :param r: Valeur rouge (0-255)
            :param g: Valeur verte (0-255)
            :param b: Valeur bleue (0-255)
            :param facteur: Facteur d'éclaircissement (0.0 à 1.0) - 0.0 = couleur originale, 1.0 = blanc
            :return: Nouvelle couleur RGB éclaircie """
            r_blanc = int(r + (255 - r) * facteur)
            g_blanc = int(g + (255 - g) * facteur)
            b_blanc = int(b + (255 - b) * facteur)
            return (r_blanc, g_blanc, b_blanc)
       
        def eclaircir_teinte_gris(facteur=0.5):
            """ Générer une teinte de gris proportionnelle à un facteur donné, en partant de noir (0, 0, 0).
            :param facteur: Facteur d'éclaircissement (0.0 à 1.0) - 0.0 = noir, 1.0 = blanc
            :return: Nouvelle teinte de gris (r, g, b) """
            gris = int(255 * facteur)
            return (gris, gris, gris)
        
        def _lerp_int(a, b, t):
            return int(a + (b - a) * t)

        def _age_color(c_young, c_old, age, age_max):
            """Interpolation de la couleur quand un eatable viellit il se foncit"""
            t = 1.0 if age_max <= 0 else age / age_max
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            return (_lerp_int(c_young[0], c_old[0], t),
                    _lerp_int(c_young[1], c_old[1], t),
                    _lerp_int(c_young[2], c_old[2], t))



        point_color_plante = colors[0]
        point_color_trophallaxie = colors[1]
        point_color_meat = colors[2]
        point_color_attack = (200, 0, 180) # on utilise les couleurs openCV poru pas a voir a faire la conversion


        # Créer une image blanche de la taille de l'image de sortie
        img = Image.new('RGB', (self.taille_video, self.taille_video), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = self.font
        

        # Boucler à travers chaque individu et dessiner un cercle à leur position
        for individu in self.liste_individus:
            # creation de la reference body
            body = individu.body
            teta = body.teta
            x, y = body.position[0], body.position[1]

            # Resize at the scale of the resolution of the video
            sx = x / self.video_scale
            sy = y / self.video_scale
            sr = body.r_collision_box_individu / self.video_scale

            
            point_color_individu = colors[body.regime + 4]
            point_color_text = (0,0,0)
            # Epaisseur augmentée si enfante
            fill_individu = None
            if body.bb_being_created:
                fill_individu = point_color_individu
            # Couleur violette si attaque
            if body.attack_bool:
                point_color_individu = point_color_attack
                fill_individu = point_color_individu
            if body.is_losing_life:
                point_color_individu = eclaircir_couleur_rgb(*point_color_individu, facteur=0.7)
                point_color_text = eclaircir_teinte_gris(facteur=0.7) #facteur de meme valeur que facteur point color individu pour une meilleur visualisation
                fill_individu = point_color_individu if fill_individu is not None else None
            # Vibrer si emet du bruit
            if body.make_noise_bool:
                # Clignotement : alterner l'affichage du cercle
                if body.age % 3 == 0:  # Change de couleur tous les 5 frames
                    draw.ellipse((sx - sr * 2, sy - sr * 2,
                                sx + sr * 2, sy + sr * 2),
                                outline=point_color_individu, width=1)
            # Dessiner INDIVIDU un cercle avec un rayon non entier
            draw.ellipse((sx - sr, sy - sr,
                        sx + sr, sy + sr),
                        outline=point_color_individu, width=1, fill=fill_individu)
            draw.line((sx, sy, sx+(sr+2.5/self.video_scale)*np.cos(np.deg2rad(body.teta)), sy+(sr+2.5/self.video_scale)*np.sin(np.deg2rad(body.teta))), fill=point_color_individu, width=1)  # Tracer la ligne
            # IDENTIFIANT : afficher ID au dessus de l'individu
            if print_ID_individu:
                draw.text((sx + 1/self.video_scale, sy + 1/self.video_scale), str(individu.ID), fill=point_color_text, font=font)

            if affichage_complet:
                liste_eatable_entity, list_reachable_entity, list_visible_entity, liste_audible_entity = self.get_entities_in_range(individu)
                liste_entree = body.fonction_vision(list_visible_entity, liste_audible_entity)

                # resize video resolution
                ser = body.r_eat_box_individu / self.video_scale
                svr = body.vision_rayon / self.video_scale
                sar = body.r_attack_box_individu / self.video_scale
                secr = body.ecoute_rayon / self.video_scale

                # HIT BOX : afficher hitbox
                draw.ellipse((sx - ser, sy - ser,
                            sx + ser, sy + ser),
                            outline=(0,240,0), width=1)
                draw.ellipse((sx - sr, sy - sr,
                            sx + sr, sy + sr),
                            outline=(0, 0, 0), width=1)
                draw.ellipse((sx - svr, sy - svr,
                            sx + svr, sy + svr),
                            outline=(255, 0, 0), width=1)
                draw.ellipse((sx - sar, sy - sar,
                            sx + sar, sy + sar),
                            outline=point_color_individu, width=1)
                draw.ellipse((sx - secr, sy - secr,
                            sx + secr, sy + secr),
                            outline=(0, 0, 0), width=1)

                if son_individu != None and individu.ID == son_individu:
                    # Tracer origine bruit
                    frequence, intensity_x, intensity_y, depreciate_sound = body.liste_bruit_entendu
                    if depreciate_sound > 0: #si il y a un bruit

                        # Calculer la position de la source du bruit avec une rotation inverse
                        teta_rad = np.deg2rad(body.teta)
                        cos = np.cos(-teta_rad)
                        sin = np.sin(-teta_rad)
                        rotation_matrix = np.array([
                            [cos, -sin],
                            [sin, cos]
                        ])
                        source_x, source_y = np.dot(rotation_matrix, np.array([intensity_x, intensity_y]))
                        source_x = body.position[0] + source_x
                        source_y = body.position[1] + source_y
                        # resize video resolution
                        ssource_x = source_x / self.video_scale
                        ssource_y = source_y / self.video_scale
                        # Vérifier que les coordonnées pour l'ellipse sont valides + resize video resolution
                        sx0 = ssource_x - 4/self.video_scale
                        sy0 = ssource_y - 4/self.video_scale
                        sx1 = ssource_x + 4/self.video_scale
                        sy1 = ssource_y + 4/self.video_scale
                    
                        draw.ellipse((sx0, sy0, sx1, sy1), fill=(0, 0, 255))  # Tracer l'ellipse
                        draw.line((sx, sy, ssource_x, ssource_y), fill=(0, 0, 255), width=1)  # Tracer la ligne

                angles = np.linspace(teta - body.vision_demi_angle, teta + body.vision_demi_angle, body.vision_nbr_parts + 1) % 360
                rayon_s = body.vision_rayon / self.video_scale
                points_vision = [(sx + rayon_s*np.cos(np.deg2rad(a)), sy + rayon_s*np.sin(np.deg2rad(a))) for a in angles]

                for p in points_vision:
                    draw.line([(sx, sy), p], fill=(0,0,0), width=1)

                for part_idx, i in enumerate(range(body.nbr_neurones_entrees_supplementaires, len(liste_entree), body.nbr_neurones_par_part)):
                    if part_idx >= len(points_vision): break
                    ma_liste = liste_entree[i:i+body.nbr_neurones_par_part]
                    draw.text(points_vision[part_idx], str([round(float(v), 2) for v in ma_liste]), fill=(0,0,0), font=font)


                # BBOX : afficher la boite bbox du quadtree
                quadtree_intersected, bbox = self.get_quadtree_tot(individu)
                min_x, min_y, max_x, max_y = bbox
                # resize video resolution
                smin_x, smin_y, smax_x, smax_y = min_x / self.video_scale, min_y / self.video_scale, max_x / self.video_scale, max_y / self.video_scale
                draw.rectangle([smin_x, smin_y, smax_x, smax_y], outline=(0, 255, 0), width=1)
            
            #reset color is_losing_life at each turn
            body.is_losing_life = False

        # EATABLE : Boucler à travers chaque eatable et dessiner un cercle à leur position
        for eatable in self.liste_eatable:
            ##point_size_eatable = eatable[0].r_hit_box_eatable
            # # DRAW PLANT function of their size representing the amount of energy it has
            # if eatable[1] == "plant":
            #     # taille VISUELLE = f(énergie), hitbox inchangée
            #     e = eatable[0].energy
            #     for i, threshold in enumerate(self.energy_thresholds):
            #         if e < threshold * energy_plant_bb:
            #             point_size_eatable = self.size_factors[i]
            #             break
            #     else:
            #         point_size_eatable = self.size_factors[-1]
            # # Draw other eatable
            # else:
            #     point_size_eatable = eatable[0].r_hit_box_eatable

            # draw just function of the amount of energy it has
            ent = eatable[0]
            e = ent.energy
            if e < 0.0:
                e = 0.0

            # Rayon VISUEL strictement linéaire en énergie (pas de max)
            point_size_eatable = eatable_draw_min + eatable_draw_scale * e


            # resize video resolution
            spoint_size_eatable = point_size_eatable / self.video_scale
            #color = point_color_plante if eatable[1] == "plant" else point_color_trophallaxie if eatable[1] == "trophallaxy" else point_color_meat
            #color evolves linearily between 2 thresholds
            typ = eatable[1]
            if typ == "plant":
                c0, c1 = colors[0]
                color = _age_color(c0, c1, eatable[0].age, age_plant_max)
            elif typ == "trophallaxy":
                c0, c1 = colors[1]
                color = _age_color(c0, c1, eatable[0].age, age_eatable_perish)
            else:  # meat
                c0, c1 = colors[2]
                color = _age_color(c0, c1, eatable[0].age, age_eatable_perish)

            
            x_p, y_p = eatable[0].position[0], eatable[0].position[1]
            # resize video resolution
            sx_p = x_p / self.video_scale
            sy_p = y_p / self.video_scale
            draw.ellipse((sx_p - spoint_size_eatable, sy_p - spoint_size_eatable, sx_p + spoint_size_eatable, sy_p + spoint_size_eatable), fill=color)

        # conversion en image editable par opencv
        image_cv2 = np.array(img)
        return image_cv2
    

    def alone_simulation(self, liste_ID_individu_selectionnes):
        """lancer la simulation avec un seul individu qui ne peut pas enfanter issu de la simulation avec tous les individus"""
        print('\n')
        try:
            #on récupère l'individu avec l'ID : ID_individu
            with open(fichier_path + '.pkl', 'rb') as f:
                db = pickle.load(f)
                liste_individus_selectionnes = []
                for ID_individu in liste_ID_individu_selectionnes:
                    individu = db.get(str(ID_individu), None)
                    liste_individus_selectionnes.append(individu)
            #initialize the ecosystem with the individual and the normal number of plants
            self.initialisation(liste_individus_selectionnes) 
            print(f"####################### ID = {liste_ID_individu_selectionnes} #######################                - #alone_simulation")
            self.jeu(True)
        except Exception as e:
            print(f"L'ID '{ID_individu}' n'existe pas dans la simulation actuelle (ou autre erreur)       #alone_simulation")
            print(e)
        


    def add_to_history(self, liste_individus):
        """Add the list of individuals to the history file with ID as key."""
        path = self.historique_path + '.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                db = pickle.load(f)
        else:
            db = {}
        for individu in liste_individus:
            db[str(individu.ID)] = individu
        with open(path, 'wb') as f:
            pickle.dump(db, f)



def detail_individu(ID_individu):
    """Affiche les détails de l'individu dont l'ID est passé en argument."""
    with open(fichier_path + '.pkl', 'rb') as f:
        db = pickle.load(f)
    individu_selectionne = db.get(str(ID_individu), None)

    print("\n")
    print(individu_selectionne)
    print(f"####################### ID = {individu_selectionne.ID}      #detail_individu #######################")   
    print("\n")

    print(f"####################### BRAIN #######################")
    
    print("Matrice Poids: ")
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(np.array2string(individu_selectionne.brain.matrice_poids, separator=', ', prefix='', max_line_width=np.inf))
    print("\n")
    print("Valeurs neurones: ")
    print(np.array2string(individu_selectionne.brain.valeurs_neurones, separator=', ', prefix='', max_line_width=np.inf)) #On formate l'affichage de la matrice pour un affichage par ligne de la matrice sur une ligne de la console
    print("\n")
    print("Biais neurones: ")
    print(np.array2string(individu_selectionne.brain.biais_neurones, separator=', ', prefix='', max_line_width=np.inf))
    print("\n")
    print(f"Valeur max poids: {np.max(individu_selectionne.brain.matrice_poids)}            Valeur min poids: {np.min(individu_selectionne.brain.matrice_poids)}")
    print(f"Valeur max biais: {np.max(individu_selectionne.brain.biais_neurones)}            Valeur min biais: {np.min(individu_selectionne.brain.biais_neurones)}")
    print(f"Nbr neurones crées: {len(individu_selectionne.brain.valeurs_neurones) - individu_selectionne.brain.nbr_entrees - individu_selectionne.brain.nbr_sorties}")
    print(f"Nbr connexions crées: {np.count_nonzero(individu_selectionne.brain.matrice_poids)}") 

    print("\n")
    print("Probas brain:")
    print(f"Alpha rate:               {round(individu_selectionne.brain.alpha, 3)}               Initial probas: {alpha_init}")
    print(f"Proba_rien_faire:         {round(individu_selectionne.brain.proba_rien_faire, 3)}               Initial probas: {proba_rien_faire_init}")
    #print(f"Dynamical probas:  {individu_selectionne.brain.proba_modifier_poid_et_biais}        Initial probas: {proba_modifier_poid_et_biais_init}")
    print(f"Proba_modifier_poid:      {round(individu_selectionne.brain.proba_modifier_poid, 3)}               Initial probas: {proba_modifier_poid_init}")
    print(f"proba_modifier_fonction:  {round(individu_selectionne.brain.proba_modifier_fonction, 3)}               Initial probas: {proba_modifier_fonction_init}")
    print(f"Proba_modifier_biais:     {round(individu_selectionne.brain.proba_modifier_biais, 3)}               Initial probas: {proba_modifier_biais_init}")
    print(f"Proba_remplacer_poid:     {round(individu_selectionne.brain.proba_remplacer_poid, 3)}               Initial probas: {proba_remplacer_poid_init}")
    print(f"Proba_remplacer_biais:    {round(individu_selectionne.brain.proba_remplacer_biais, 3)}               Initial probas: {proba_remplacer_biais_init}")
    print(f"Proba_creer_poid:         {round(individu_selectionne.brain.proba_creer_poid, 3)}               Initial probas: {proba_creer_poid_init}")
    print(f"Proba_ajouter_neurone:    {round(individu_selectionne.brain.proba_ajouter_neurone, 3)}               Initial probas: {proba_ajouter_neurone_init}")
    print(f"Proba_supprimer_neurone:  {round(individu_selectionne.brain.proba_supprimer_neurone, 3)}               Initial probas: {proba_supprimer_neurone_init}")
    print(f"Proba_supprimer_poid:     {round(individu_selectionne.brain.proba_supprimer_poid, 3)}               Initial probas: {proba_supprimer_poid_init}")
    print(f"Proba_supprimer_biais:    {round(individu_selectionne.brain.proba_supprimer_biais, 3)}               Initial probas: {proba_supprimer_biais_init}")
    print("\n")

    print(f"####################### BODY #######################")
    print("---------------------------------")
    print(f"Vision rayon:                     {individu_selectionne.body.vision_rayon}")
    print(f"Vision demi angle:                {individu_selectionne.body.vision_demi_angle}")
    print(f"Vision nombre de parts:           {individu_selectionne.body.vision_nbr_parts}")
    print(f"Vision nombre de neurones/parts:  {individu_selectionne.body.nbr_neurones_par_part}")
    print(f"Ecoute rayon:                     {individu_selectionne.body.ecoute_rayon}")
    print(f"Rayon eat box individu:           {individu_selectionne.body.r_eat_box_individu}")
    print(f"Rayon collision box individu:     {individu_selectionne.body.r_collision_box_individu}")
    print(f"Rayon attack box individu:        {individu_selectionne.body.r_attack_box_individu}")
    print(f"Facteur x deplacement:            {individu_selectionne.body.facteur_multiplicatif_deplacement}")
    print(f"Energie:                          {individu_selectionne.body.energie}")
    print(f"Energie max:                      {individu_selectionne.body.max_energie_individu}")
    print(f"Vie:                              {individu_selectionne.body.vie}")
    print(f"Vie max:                          {individu_selectionne.body.max_vie_individu}")
    print(f"Age:                              {individu_selectionne.body.age}")
    print(f"Generation:                       {individu_selectionne.body.generation}")
    print(f"Kills:                            {individu_selectionne.body.compteur_kill}")
    print(f"Trophallaxie:                     {individu_selectionne.body.compteur_trophallaxie}")
    print(f"Plantes mangées:                  {individu_selectionne.body.compteur_plant_eaten}")
    print(f"Classe:                           {individu_selectionne.body.regime}")
    print(f"Taille:                           {individu_selectionne.body.r_collision_box_individu}")
    print(f"Vitesse:                          {individu_selectionne.body.facteur_multiplicatif_deplacement}")

    
    plot_neural_network(individu_selectionne.brain.matrice_poids, individu_selectionne, ID_individu) 

    # except Exception as e:
    #     print(f"L'ID '{ID_individu}' n'existe pas dans la simulation actuelle (ou autre erreur)      #detail_individu")
    #     print(e)
    

def plot_neural_network(matrice_poids, individu_selectionne, ID_individu):
    """
    Trace le graphe d'un réseau de neurones donné par une matrice de poids. Rouge = négatif, vert = positif.
    Plus la couleur est foncée, plus la valeur du poids est élevée.
    """
    nbr_entrees = individu_selectionne.brain.nbr_entrees
    nbr_sorties = individu_selectionne.brain.nbr_sorties
    liste_entrees_supplementaires = individu_selectionne.body.liste_entrees_supplementaires
    liste_sorties_supplementaires = individu_selectionne.body.liste_sorties_supplementaires

    # Création du graphe
    G = nx.DiGraph()

    # Ajouter des neurones au graphe
    for i in range(len(matrice_poids)):
        G.add_node(i)

    # Ajouter des connexions au graphe selon la matrice de poids
    for i in range(len(matrice_poids)):
        for j in range(len(matrice_poids[i])):
            poids = matrice_poids[i][j]
            if poids != 0:
                G.add_edge(i, j, weight=poids)  # Ajout de l'attribut weight pour stocker le poids de la connexion

    # Déterminer les positions des neurones et les couches
    positions = {}
    layers = {}
    node_shapes = {}  # Ajout d'un dictionnaire pour les formes des noeuds

    # Positions pour la couche d'entrée (x=1)
    for i in range(nbr_entrees):
        positions[i] = (1, -i)
        layers[i] = 1
        node_shapes[i] = 'o'  # Rond pour les neurones d'entrée

    # Utiliser une file pour gérer les couches
    queue = deque([(node, 1) for node in range(nbr_entrees)])

    # Parcourir les neurones pour déterminer les couches
    while queue:
        current_node, current_layer = queue.popleft()
        next_layer = current_layer + 1
        for successor in G.successors(current_node):
            if successor not in layers:
                layers[successor] = next_layer
                queue.append((successor, next_layer))

    # Définir les neurones de sortie
    output_nodes = list(range(nbr_entrees, nbr_entrees + nbr_sorties))
    for node in output_nodes:
        layers[node] = -1
        node_shapes[node] = 'o'  # Rond pour les neurones de sortie

    # Ajouter les neurones isolés s'ils existent
    for node in G.nodes():
        if node not in layers:
            layers[node] = max(layers.values()) + 1
            node_shapes[node] = 'o'  # Rond pour les neurones isolés

    # Positionner les neurones par couche avec un décalage horizontal
    max_nodes_per_x = 2
    layer_counts = {layer: 0 for layer in set(layers.values())}
    layer_colors = {}

    for node, layer in layers.items():
        # Décalage vertical pour aligner les neurones sur la même couche
        vertical_offset = -(layer_counts[layer])
        # Calcul du décalage horizontal en fonction du nombre de neurones déjà placés sur la couche
        if layer == -1:
            positions[node] = (max(layers.values()) + 1, -output_nodes.index(node))
        elif layer == 1:
            positions[node] = (1, -node)
        else:
            additional_offset = (layer_counts[layer] % max_nodes_per_x) * 0.1 * ((layer_counts[layer] // max_nodes_per_x) % 2 * 2 - 1)
            positions[node] = (layer + additional_offset, vertical_offset)
        layer_counts[layer] += 1

        # Couleurs des couches
        if layer not in layer_colors:
            layer_colors[layer] = plt.cm.Set3(len(layer_colors) / (max(layers.values()) + 1))  # Couleur unique par couche

        # Déterminer la forme des noeuds en fonction de la fonction d'activation
        if isinstance(individu_selectionne.brain.activation_functions[node], brain.Hysteresis):
            node_shapes[node] = 's'  # Carré pour Hysteresis
        else:
            node_shapes[node] = 'o'  # Rond pour tanh

    # Création des libellés pour les neurones d'entrée
    labels = {}
    index = 0

    # Ajouter des labels pour les entrées supplémentaires
    for feature, count in liste_entrees_supplementaires.items():
        for _ in range(count):  # Répéter pour chaque neurone correspondant
            labels[index] = feature
            index += 1

    # Ajouter un label "vision" pour le reste des neurones d'entrée
    for i in range(index, nbr_entrees):
        labels[i] = "vision"

    # Création des libellés pour les neurones de sortie
    index = 0  # Réinitialiser l'index pour les neurones de sortie

    # D'abord, assigner les étiquettes pour les neurones de sortie prédéfinis
    labels[output_nodes[index]] = "velocity"  # Vélocité à la position 0
    index += 1
    labels[output_nodes[index]] = "angle"      # Angle à la position 1
    index += 1

    # Ajouter des labels pour les sorties supplémentaires
    for feature, count in liste_sorties_supplementaires.items():
        for _ in range(count):  # Répéter pour chaque neurone correspondant
            if index < len(output_nodes):
                labels[output_nodes[index]] = feature
                index += 1
            else:
                break  # Sortir si l'on a déjà utilisé tous les neurones de sortie

    # Récupérer les positions des noeuds
    node_pos = positions

    # Création de la légende pour les couleurs de couches
    legend_labels = []
    handles = []
    for layer, color in sorted(layer_colors.items(), key=lambda x: x[0]):
        if layer == 1:
            label = 'Input'
        elif layer == -1:
            label = 'Output'
        else:
            label = f'Layer {layer}'
        legend_labels.append(label)
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label))

    # Vérifier si le graphe a des arêtes avant de calculer les couleurs et les largeurs des arêtes. Si il en a pas on trace juste les neurones
    if G.number_of_edges() > 0:
        edges = G.edges(data=True)
        weights = np.array([d['weight'] for _, _, d in edges])
        min_weight = np.min(weights)
        max_weight = np.max(weights)

        # Fonction pour calculer la couleur en échelle logarithmique
        def compute_edge_color(weight):
            if weight > 0:
                color_intensity = np.log10(weight + 1) / np.log10(max_weight + 1)  # Logarithme pour les poids positifs
                return plt.cm.Greens(color_intensity)
            elif weight < 0:
                color_intensity = np.log10(-weight + 1) / np.log10(-min_weight + 1)  # Logarithme pour les poids négatifs
                return plt.cm.Reds(color_intensity)
            else:
                return 'black'  # Pour les poids nuls

        edge_colors = [compute_edge_color(d['weight']) for _, _, d in edges]
        edge_widths = [1.5 if abs(d['weight']) > 0 else 0.5 for _, _, d in edges]  # Épaisseur de la flèche
    else:
        edge_colors = []
        edge_widths = []

    # Tracer le graphe
    plt.figure(figsize=(10, 5))
    plt.title(f"Brain of the individual: {ID_individu}")
    
    # Combiner les labels avec les couches pour les neurones intermédiaires
    combined_labels = {node: f"{labels[node]}\nLayer {layers[node]}" if node in labels else f"Neuron {node}\nLayer {layers[node]}" for node in G.nodes()}

    # Tracer les noeuds avec leurs formes respectives et couleurs correctes par couches
    for shape in set(node_shapes.values()):
        nodes_of_shape = [node for node, s in node_shapes.items() if s == shape]
        nx.draw_networkx_nodes(
            G, pos=node_pos, nodelist=nodes_of_shape,
            node_shape=shape, 
            node_color=[layer_colors[layers[node]] for node in nodes_of_shape],
            node_size=700
        )

    # Tracer les arêtes
    nx.draw_networkx_edges(G, pos=node_pos, width=edge_widths, edge_color=edge_colors)

    # Tracer les labels
    nx.draw_networkx_labels(G, pos=node_pos, labels=combined_labels, font_size=12, font_weight='bold')

    # Création de la légende pour les couleurs de couches
    plt.legend(handles=handles, loc='best', title='Layers')
    plt.show()







# =============================================================================
# Main
# =============================================================================

import cProfile

def main(path_population=None):
    #Creation du monde
    ecosystem_obj = Ecosystem()

    #on initialise les plantes et les individus de la génération 0
    #If we load an entire previous population
    if path_population is not None:
        ecosystem_obj.initialisation(liste_individus_selectionnes = functions.get_list_last_individuals(path_population, nbr_individus_init))
    #If we load the same initial brain for all individuals
    elif isinstance(brain_to_be_used, list):
        brain_to_be_used_array = np.array(brain_to_be_used)
        biais_to_be_used_array = np.array(biais_to_be_used)
        ecosystem_obj.initialisation(matrice_poids = brain_to_be_used_array, biais_neurones = biais_to_be_used_array)
    #If we load the same initial individual for all individuals
    elif individu_to_be_used is not None:
        ecosystem_obj.initialisation(liste_individus_selectionnes = functions.get_list_same_individuals(fichier_path, individu_to_be_used, nbr_individus_init))
    #If we start a new simulation
    else:
        ecosystem_obj.initialisation()

    #on lance la simulation avec tous les individus pour qu'ils se reproduisent & apprennent
    global tps_init
    tps_init = time.time()

    # # Sources de temps de la simulation
    # def run_simulation():
    #     ecosystem_obj.jeu(False)
    # # Profilage de la fonction run_simulation()
    # cProfile.runctx('run_simulation()', globals(), locals())

    ecosystem_obj.jeu(False)
    tps_final = time.time()
    print("Nombre de graines collectées par zoochrorie : ", ecosystem_obj.seed_collected_zoochorie)
    print("Nombre de graines déposées par zoochrorie : ", ecosystem_obj.seed_droped_zoochorie)
    print("Temps de la simulation : ", tps_final - tps_init)
    # On affiche l'état de la population à la fin de la simulation
    print(ecosystem_obj.population_etat_fin_simulation)

    return ecosystem_obj


# Si on veut lancer une simulation complète
if characteristics_ID_individu == None and liste_ID_alone_simulation == [] and statistics == False:
    # Re start simulation if population dies 
    if start_again_until_alive_pop == True:
        while start_again_until_alive_pop:
            
            ecosystem_obj = main(path_population=load_population)
            # Vérifier si on doit relancer la simulation
            if ecosystem_obj.population_etat_fin_simulation == "Population en vie":
                start_again_until_alive_pop = False
            else:
                print("\n")
                print("Relance de la simulation car la population s'est éteinte.")
    # Start just one the simulation
    else: 
        ecosystem_obj = main(load_population)

# Si on veut afficher les détails d'un individu
if characteristics_ID_individu != None:
    detail_individu(characteristics_ID_individu)

# Si on veut une simu avec un individu seul
if liste_ID_alone_simulation != []:
    ecosystem_obj = Ecosystem()
    ecosystem_obj.alone_simulation(liste_ID_alone_simulation)

# Si on veut les top des individus les plus interessants
if statistics == True: 
    #statistique observateur
    observer_module = observer.Observer()
    observer_module.stat()



"""



CONSOMMATEUR DE TEMPS : 
- add_to_history
- film generation
- quadtree.intersect
- collision

"""