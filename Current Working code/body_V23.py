import numpy as np
import random
import math

from parameters_V23 import *
import functions_V23 as functions



def vision_normalizer(size_individual, my_size):
    """Normalise entre [-1, 1] la vision en fonction de la taille de l'individu
    La normalisation de la taille de l'individu dépend de ma taille : elle varie beaucoup si la cible est d'une taille similaire à la mienne"""
    return np.arctan(size_individual - my_size) / (np.pi / 2)


def diet_normalizer(diet_individual, my_diet):
    """
    Normalise entre [-1, 1] la différence de régime alimentaire (diet) entre deux individus.
    La normalisation est basée sur l'échelle maximale de régime alimentaire.
    
    :param diet_individual: Le régime de l'individu cible (entier entre 0 et lvl_max_eat_scale).
    :param my_diet: Mon propre régime alimentaire (entier entre 0 et lvl_max_eat_scale).
    :param lvl_max_eat_scale: La valeur maximale sur l'échelle de régime.
    :return: Valeur normalisée entre [-1, 1] indiquant la similarité des régimes alimentaires.
    """
    # if it is not an individual, so has no diet
    if diet_individual == None:
        return -1
    # Calcul de la différence normalisée entre -1 et 1
    difference = diet_individual - my_diet
    normalized_difference = difference / lvl_max_eat_scale if lvl_max_eat_scale != 0 else 0
    # Appliquer arctan pour lisser les valeurs et normaliser entre -1 et 1
    return np.arctan(normalized_difference) / (np.pi / 2)


def move_energy(x):
    """Fonction pour la perte d'énergie en fonction de la taille"""
    return 0.015*(x - 1)**3 + 1 #vaut 1 pour la taille de départ

def move_energy_linear(x):
    """Fonction pour la perte d'énergie en fonction de la taille linéaire"""
    return 0.4*(x - 1) + 1 #vaut 1 pour la taille de départ




class Body:
    def __init__(self):
        """Les attributs de body sont les variables qui seront modifiés par des méthodes de la classe body && les variables évolutives qui caractérisent le corps (enerie, age, génération)
        Les variables "physiques" qui ne sont pas attributs de la classe body sont des variables qui dépendent des lois de la physique"""
        self.vision_rayon = vision_rayon_init
        self.vision_demi_angle = vision_demi_angle_init
        self.r_eat_box_individu = r_eat_box_individu_init
        self.r_collision_box_individu = r_collision_box_individu_init
        self.r_attack_box_individu = r_attack_box_individu_init
        self.facteur_multiplicatif_deplacement = facteur_multiplicatif_deplacement_init
        self.ecoute_rayon = 0 #0 at the beginning because no ears, it speeds up the research in the quadtree

        self.max_energie_individu = max_energie_individu_init
        self.max_vie_individu = max_vie_individu_init
        self.vision_nbr_parts = vision_nbr_parts_init
        self.max_rotation = max_rotation_init

        self.nbr_neurones_entrees_supplementaires = nbr_neurones_entrees_supplementaires_init
        self.nbr_neurones_sorties_supplementaires = nbr_neurones_sorties_supplementaires_init

        self.liste_entrees_supplementaires = liste_entrees_supplementaires_init
        self.liste_entrees_supplementaires_par_part = liste_entrees_supplementaires_par_part_init
        self.liste_sorties_supplementaires = liste_sorties_supplementaires_init

        self.liste_entrees_supplementaires_possibles = liste_entrees_supplementaires_possibles_init
        self.liste_entrees_supplementaires_possibles_par_part = liste_entrees_supplementaires_possibles_par_part_init
        self.liste_sorties_supplementaires_possibles = liste_sorties_supplementaires_possibles_init

        self.nbr_neurones_par_part = nbr_neurones_par_part_init

        self.know_size = False
        self.know_diet = False

        #initialize
        self.generation = 0 #to modify if is a child
        self.position = [0,0] #to modify if is a child
        self.regime = lvl_max_eat_scale//2 # Spawning as a balanced individual. Must be in [0, lvl_max_eat_scale]
        self.initialize_individu()

        # If we initialize with ears
        if "oreille" in self.liste_entrees_supplementaires:
            self.ecoute_rayon = ecoute_rayon_init


    def initialize_individu(self):
        """Initialize the individual with the initial values of the attributes of the body. Used for relaunching the simulation with them"""
        self.energie = self.max_energie_individu * 0.6 #pour ne pas creer de bb tout de suite
        self.vie = self.max_vie_individu
        self.age = random.randint(0, 500)
        self.teta = random.uniform(0,360)
        self.duree_gestation = random.uniform(30,100) #time to create a baby
        self.gestation = 0
        self.bb_being_created = False
        self.is_losing_life = False
        self.old_facteur_multiplicatif_deplacement = self.facteur_multiplicatif_deplacement
        self.nbr_bb = 0
        
        self.reset_bruit_entendu() #Normalisation entre -1 et +1 tanh.  liste des bruits entendus. Sert pour l'audio et pour les valeurs par défauts si on entend rien
        self.reset_bruit_emis() #sert pour le bruit emis par défaut (0 si on emet pas de bruit comme ca 'depreciate_distance_sound' renvoit 0 et il n'y a donc pas de max

        self.attack_bool = False
        self.make_noise_bool = False

        self.compteur_kill = 0
        self.compteur_trophallaxie = 0
        self.compteur_plant_eaten = 0
        self.compteur_meat_eaten = 0



    def mutate_regime(self):
        """Augmente ou diminue aléatoirement la classe. Autant de chance de monter ou diminuer."""
        augmentation = random.choice([1, -1])
        self.regime += augmentation
        # Limit conditions A MODIF AVEC REGIMES ALIM PARAMETRES
        if self.regime < 0:
            self.regime = 0
        elif self.regime > lvl_max_eat_scale:
            self.regime = lvl_max_eat_scale


    def mutate_nbr_part(self, brain):
        """Augmente ou diminue aléatoirement le nombre de part de vision"""
        augmentation = random.choice([1, -1]) #on peut ajouter ou enlever une SEULE PART part
        # Limit conditions
        if self.vision_nbr_parts + augmentation < 1:
            augmentation = 0
        # On update le nombre de parts
        self.vision_nbr_parts += augmentation
        # Update the number of input neurons
        brain.update_nbr_part_entrees(augmentation, self.nbr_neurones_par_part)


    def do_nothing(self):
        pass


    
    def mutate_neuron_feature(self, brain):
        """Ajoute un neurone d'entrée ou de sortie au réseau de neurone : ajoute une feature"""
        #dictionnaire des actions possibles
        actions = {
            lambda: self.mutate_nbr_part(brain): 0.6, #bigger as it is adding and removing as the same function ...
            
            lambda: self.add_neuron_entry(brain): 1,
            lambda: self.add_know_something(brain): 0.5,
            lambda: self.add_neuron_exit(brain): 1,

            lambda: self.remove_neuron_entry(brain): 0.8, #removing neurons should be less frequent for leting a chance to the evolution to be assimilated
            lambda: self.remove_know_something(brain): 0.4,
            lambda: self.remove_neuron_exit(brain): 0.8 
        }
        
        # Tirage probabiliste de l'action à effectuer
        methods = list(actions.keys())
        probabilities = list(actions.values())
        selected_method = random.choices(methods, probabilities, k=1)[0]
        # Appeler la méthode sélectionnée
        selected_method()


    def add_neuron_entry(self, brain):
        """Ajoute un neurone d'entrée"""
        #Y a t -il des ajouts possibles ?
        if self.liste_entrees_supplementaires_possibles == {}:
            return
        #si il y a toujours de ajouts possibles
        cles = list(self.liste_entrees_supplementaires_possibles.keys())
        neurone = random.choices(cles)[0]
        nbr_neurone_a_ajouter = self.liste_entrees_supplementaires_possibles[neurone]

        # Add the additional input neurons in the brain
        brain.update_nbr_supplementaire_entrees(augmentation=nbr_neurone_a_ajouter)
        #on insert au début des entrées la nouvelle. On rappelle que la vision sont en dernier 
        self.liste_entrees_supplementaires = {neurone: nbr_neurone_a_ajouter, **self.liste_entrees_supplementaires} #On ajoute au début
        self.liste_entrees_supplementaires_possibles.pop(neurone)
        # Update the number of additional input neurons
        self.nbr_neurones_entrees_supplementaires += nbr_neurone_a_ajouter

        # If ears, the ears range is no more nul
        if neurone == "oreille":
            self.ecoute_rayon = ecoute_rayon_init


    def remove_neuron_entry(self, brain):
        """Retire un neurone d'entrée"""
        # Y a-t-il des retraits possibles ?
        if not self.liste_entrees_supplementaires:
            return
        # Si il y a toujours des retraits possibles
        neurone = random.choice(list(self.liste_entrees_supplementaires.keys()))
        nbr_neurone_a_retirer = self.liste_entrees_supplementaires[neurone]
        # Calcul de la somme des valeurs avant le neurone sélectionné
        keys = list(self.liste_entrees_supplementaires.keys())
        index = keys.index(neurone)
        sum_values_before_index = sum(list(self.liste_entrees_supplementaires.values())[:index])
        # Remove the additional input neurons in the brain
        brain.update_nbr_supplementaire_entrees(augmentation=-nbr_neurone_a_retirer, position=sum_values_before_index)

        #on retire l'entrée
        self.liste_entrees_supplementaires.pop(neurone)
        self.liste_entrees_supplementaires_possibles[neurone] = nbr_neurone_a_retirer
        # Update the number of additional input neurons
        self.nbr_neurones_entrees_supplementaires -= nbr_neurone_a_retirer

        # If ears, the ears range is nul
        if neurone == "oreille":
            self.ecoute_rayon = 0



    def add_neuron_exit(self, brain):
        """Ajoute un neurone d'entrée"""
        #Y a t -il des ajouts possibles ?
        if self.liste_sorties_supplementaires_possibles == {}:
            return
        #si il y a toujours de ajouts possibles
        cles = list(self.liste_sorties_supplementaires_possibles.keys())
        neurone = random.choices(cles)[0]
        nbr_neurone_a_ajouter = self.liste_sorties_supplementaires_possibles[neurone]

        # Add the additional input neurons in the brain
        brain.update_nbr_supplementaire_sorties(nbr_neurone_a_ajouter)
        #on insert au début des entrées la nouvelle. On rappelle que la vision sont en dernier 
        self.liste_sorties_supplementaires[neurone] = nbr_neurone_a_ajouter #On ajoute à la fin
        self.liste_sorties_supplementaires_possibles.pop(neurone)
        # Update the number of additional input neurons
        self.nbr_neurones_sorties_supplementaires += nbr_neurone_a_ajouter

    
    def remove_neuron_exit(self, brain):
        """Retire un neurone de sortie"""
        #Y a t -il des retraits possibles ?
        if self.liste_sorties_supplementaires == {}:
            return
        # Si il y a toujours des retraits possibles
        neurone = random.choice(list(self.liste_sorties_supplementaires.keys()))
        nbr_neurone_a_retirer = self.liste_sorties_supplementaires[neurone]
        # Calcul de la somme des valeurs avant le neurone sélectionné
        keys = list(self.liste_sorties_supplementaires.keys())
        index = keys.index(neurone)
        sum_values_before_index = sum(list(self.liste_sorties_supplementaires.values())[:index])
        # Remove the additional input neurons in the brain
        brain.update_nbr_supplementaire_sorties(augmentation=-nbr_neurone_a_retirer, position=sum_values_before_index)
        #on retire l'entrée
        self.liste_sorties_supplementaires.pop(neurone)
        self.liste_sorties_supplementaires_possibles[neurone] = nbr_neurone_a_retirer
        # Update the number of additional input neurons
        self.nbr_neurones_sorties_supplementaires -= nbr_neurone_a_retirer


    def mutate_size(self):
        """Modifie la taille du corps de l'individu en gardant les meme proportions pour eat, hit et collision box"""
        variation = random.uniform(-1, 1)
        r_collision_box_individu_old = self.r_collision_box_individu
        # Update the size of the body
        if self.r_collision_box_individu + variation <= 0.5: #on ne peut pas avoir un rayon négatif (round pour ne pas que cv2 dessine un individu de rayon nul)
            variation = 0
        self.r_eat_box_individu += variation
        self.r_collision_box_individu += variation
        self.r_attack_box_individu += variation
        # Update the lvl of energy and amount of life. The bigger an individual is the more energy and life it has
        rapport = self.r_collision_box_individu/r_collision_box_individu_old # new/old
        self.max_vie_individu *= rapport
        self.max_energie_individu *= rapport


    def mutate_speed(self):
        """Modifie la vitesse de l'individu"""
        variation = random.uniform(-1, 1)
        # Update the speed of the body
        if self.facteur_multiplicatif_deplacement + variation <= 0.1: #on ne peut pas avoir une vitesse négative
            variation = 0
        self.facteur_multiplicatif_deplacement += variation
    

    def mutate_vision(self):
        """Ajoute un neurone d'entrée ou de sortie au réseau de neurone : ajoute une feature"""
        #dictionnaire des actions possibles
        actions = {
            self.mutate_vision_rayon: 1,
            self.mutate_vision_angle: 1
        }
        
        # Tirage probabiliste de l'action à effectuer
        methods = list(actions.keys())
        probabilities = list(actions.values())
        selected_method = random.choices(methods, probabilities, k=1)[0]
        # Appeler la méthode sélectionnée
        selected_method()


    def mutate_vision_rayon(self):
        """Modifie la vision de l'individu"""
        variation = random.uniform(-8, 8)
        # Update the speed of the body
        if self.vision_rayon + variation <= 0.1: #on ne peut pas avoir un rayon nulle ou négatif
            variation = 0
        self.vision_rayon += variation


    def mutate_vision_angle(self):
        """Modifie la vision de l'individu"""
        variation = random.uniform(-5, 5)
        # Update the speed of the body
        if self.vision_demi_angle + variation <= 0.1: #on ne peut pas avoir un angle nulle ou négatif
            variation = 0
        self.vision_demi_angle += variation


    def mutate_rotation(self):
        """Modifie la rotation maximum d'un coup (son agilité) de l'individu en degré"""
        variation = random.uniform(-5, 5)
        # Update the speed of the body
        if self.max_rotation + variation <= 0.1: #on ne peut pas avoir une rotation max nulle ou négative
            variation = 0
        elif self.max_rotation + variation > 180:
            variation = 180
        self.max_rotation += variation


    def mutate_ears(self):
        """Modifie la portée d'écoute de l'individu"""
        variation = random.uniform(-10, 10)
        # Update the speed of the body
        if self.ecoute_rayon + variation <= 0.1: #on ne peut pas avoir un rayon nulle ou négatif
            variation = 0
        self.ecoute_rayon += variation



    def add_know_something(self, brain):
        """Ajoute la possibilité a un individu de connaitre la diet relative de l'individu en face de lui par rapport à lui-meme"""
        #Y a t -il des ajouts possibles ?
        if self.liste_entrees_supplementaires_possibles_par_part == {}:
            return
        #si il y a toujours de ajouts possibles
        cles = list(self.liste_entrees_supplementaires_possibles_par_part.keys())
        neurone = random.choices(cles)[0]
        nbr_neurone_a_ajouter = self.liste_entrees_supplementaires_possibles_par_part[neurone]
        # Add the additional input neurons in the brain
        brain.update_know_something(self.vision_nbr_parts, self.nbr_neurones_par_part, self.nbr_neurones_entrees_supplementaires, augmentation=nbr_neurone_a_ajouter)
        #on insert au début des entrées la nouvelle. On rappelle que la vision sont en dernier 
        self.liste_entrees_supplementaires_par_part = {neurone: nbr_neurone_a_ajouter, **self.liste_entrees_supplementaires_par_part} #On ajoute au début
        self.liste_entrees_supplementaires_possibles_par_part.pop(neurone)
        # Update the number of additional input neurons
        self.nbr_neurones_par_part += nbr_neurone_a_ajouter



    def remove_know_something(self, brain):
        """Retire les neurones d'entrées dans chaque part de vision qui apportent des connaissances de l'individu qu'on voit"""
        #Y a t -il des retraits possibles ?
        if self.liste_entrees_supplementaires_par_part == {}:
            return
        #si il y a toujours de retraits possibles
        neurone = random.choice(list(self.liste_entrees_supplementaires_par_part.keys()))
        nbr_neurone_a_retirer = self.liste_entrees_supplementaires_par_part[neurone]
        index = list(self.liste_entrees_supplementaires_par_part.keys()).index(neurone)
        # Remove the additional input neurons in the brain
        brain.update_know_something(self.vision_nbr_parts, self.nbr_neurones_par_part, self.nbr_neurones_entrees_supplementaires, augmentation=-nbr_neurone_a_retirer, position=index)
        #on retire l'entrée
        self.liste_entrees_supplementaires_par_part.pop(neurone)
        self.liste_entrees_supplementaires_possibles_par_part[neurone] = nbr_neurone_a_retirer
        # Update the number of additional input neurons
        self.nbr_neurones_par_part -= nbr_neurone_a_retirer



    def mutate_body(self, brain):
        """mutation du corps"""
        #on peut effectuer une mutation de classe
        for n in range(nbr_modifications_body):
            #dictionnaire des actions possibles
            actions = {
                lambda: self.mutate_neuron_feature(brain): proba_modifier_neurone_feature,
                self.mutate_regime: proba_modifier_regime,
                self.mutate_size: proba_modifier_size,
                self.mutate_speed: proba_modifier_speed,
                self.mutate_vision: proba_modifier_vision,
                self.mutate_rotation: proba_modifier_rotation,
                self.mutate_ears: proba_modifier_ears,
                self.do_nothing: proba_rien_faire_body
            }
            
            # Tirage probabiliste de l'action à effectuer
            methods = list(actions.keys())
            probabilities = list(actions.values())
            selected_method = random.choices(methods, probabilities, k=1)[0]
            # Appeler la méthode sélectionnée
            selected_method()


    def mutate_body_init(self):
        """mutation du corps. UTILISEE UNIQUEMENT LORS DE L'INITIALISATION DES INDIVIDUS POUR LA PREMIERE FOIS"""
        #on peut effectuer une mutation de classe
        for n in range(nbr_modifications_body_init):
            #dictionnaire des actions possibles
            actions = {
                self.mutate_regime: proba_modifier_regime,
                self.mutate_size: proba_modifier_size,
                self.mutate_speed: proba_modifier_speed,
                self.mutate_vision: proba_modifier_vision,
                self.mutate_rotation: proba_modifier_rotation,
                self.do_nothing: proba_rien_faire_body
            }
            
            # Tirage probabiliste de l'action à effectuer
            methods = list(actions.keys())
            probabilities = list(actions.values())
            selected_method = random.choices(methods, probabilities, k=1)[0]
            # Appeler la méthode sélectionnée
            selected_method()


    def move(self, sortie_brain):
        """déplacement du corps
        Ne vérifie pas les sorties de la carte"""
        #angle et normalisation
        teta = sortie_brain[1]
        #teta = teta*(self.max_rotation*2) - (self.max_rotation)  #on normalise la sortie de la sigmoid entre [-max_rotation; +max_rotation] -----------------  FONCTION ACTIVATION
        teta = teta*self.max_rotation  #on normalise la sortie de la sigtanh moid entre [-max_rotation; +max_rotation] -----------------  FONCTION ACTIVATION
        self.teta += teta #en degré par défaut
        #perte energie rotation
        self.energie -= np.abs(teta * facteur_multiplicatif_perte_vie/10)
        
        #radian car en argument d'une fonction trigo
        nouvel_angle = self.teta * np.pi/180
        
        #deplacement et velocité
        velocite = sortie_brain[0]
        deplacement = velocite*pas_de_temps*self.facteur_multiplicatif_deplacement
        deplacement_x = deplacement*np.cos(nouvel_angle)
        deplacement_y = deplacement*np.sin(nouvel_angle)

        #bordures de la carte, la carte est ronde/continue
        position_x, position_y = self.position[0], self.position[1]
        nvl_position_x = position_x + deplacement_x
        nvl_position_y = position_y + deplacement_y
        # perte de la vie en fonction du deplacement: ca correspond à sa faim/soif
        self.energie -= np.abs(deplacement * facteur_multiplicatif_perte_vie * move_energy_linear(self.r_collision_box_individu/r_collision_box_individu_init)) #plus l'individu est gros plus il perd de l'energie. /r_collision_box_individu_init pour que ce soit un facteur 1 au départ. move_energy renvoit 1 en 1

        return [nvl_position_x, nvl_position_y, deplacement]


    def process_additional_outputs(self, ecosystem_obj, simulation_seul_param, individu, sortie_brain, list_reachable_entity):
        """Process the additional outputs of the brain. liste_audible_entity = entité qui peuvent entendre mon son"""
        self.attack_bool = False # Reset the attack boolean
        self.make_noise_bool = False # Reset the make noise boolean

        # Ajouter les neurones supplémentaires
        index_sortie_brain = nbr_neurones_de_base_sortie
        for neurone, nbr_neurones_correspondant in self.liste_sorties_supplementaires.items():

            # Récupérer les valeurs de sortie correspondantes de sortie_brain
            valeurs_sortie_brain = sortie_brain[index_sortie_brain:index_sortie_brain + nbr_neurones_correspondant]

            # Incrémenter l'index pour la prochaine sortie
            index_sortie_brain += nbr_neurones_correspondant

            # Prendre une décision en fonction de la sortie et de ses valeurs
            if neurone == "attaque" and valeurs_sortie_brain > seuil_attaque and self.age > age_min_to_attack and self.energie > 0:
                self.attack(ecosystem_obj, valeurs_sortie_brain, list_reachable_entity)
            elif neurone == "trophallaxy" and valeurs_sortie_brain > seuil_trophallaxie and self.energie > 0:
                self.share_ressources(ecosystem_obj, valeurs_sortie_brain)
            elif neurone == "bouche" and valeurs_sortie_brain[1] > seuil_bruit: #valeurs_sortie_brain[0] = freq        &&          valeurs_sortie_brain[1] = intensité
                self.make_noise(valeurs_sortie_brain)
            elif neurone == "creer_bb" and valeurs_sortie_brain > seuil_creer_bb: 
                if self.age > age_min_to_childbirth and self.energie >= facteur_energie_creer_bb*self.max_energie_individu and len(ecosystem_obj.liste_individus) < max_individu and not simulation_seul_param and not self.bb_being_created:
                    #lance le processus pour creer un bébé si il peut (il faut un certain age et energie pour procréer et il ne doit pas y avoir trop d'individus)
                    self.bb_being_created = True
                    self.nbr_bb = 1 + int((self.energie - facteur_energie_creer_bb*self.max_energie_individu) // (facteur_energie_depensee_creer_bb*self.max_energie_individu)) #Nbr max of possible baby to born
                    #we are slower when creating a bb
                    self.old_facteur_multiplicatif_deplacement = self.facteur_multiplicatif_deplacement 
                    self.facteur_multiplicatif_deplacement = 0.6/size_modification
                if self.bb_being_created and self.gestation >= self.duree_gestation and len(ecosystem_obj.liste_individus) < max_individu:
                    #creer effectivement le bébé si le temps de gestation est fini
                    self.energie /= (self.nbr_bb + 1) #on divise l'energie par le nombre de bébé + 1 (car le parent)
                    ecosystem_obj.create_bb(individu, self.nbr_bb, self.energie)


    def make_noise(self, valeurs_sortie_brain):
        """Fait du bruit : on indique a travers notre attribut bruit_emis, qu'on fait du bruit"""
        self.make_noise_bool = True
        # Get the intensity and frequency of the sound
        frequence, intensite = valeurs_sortie_brain[0], valeurs_sortie_brain[1]
        self.bruit_emis = [frequence, intensite]


    def reset_bruit_emis(self):
        """Reset le bruit emis par défaut à chaque tour (avant appel de 'process_additional_outputs')"""
        self.bruit_emis = [0, 0]


    def trigo_ear(self, depreciate_distance, angle_bruit):
        """calcul trigo de position origine son"""
        angle_bruit_rad = np.deg2rad(angle_bruit)
        teta_rad = np.deg2rad(self.teta)
        
        # Calculer les coordonnées de l'émetteur dans le repère global
        x_emetteur = depreciate_distance * np.cos(angle_bruit_rad)
        y_emetteur = depreciate_distance * np.sin(angle_bruit_rad)

        # Créer la matrice de rotation pour transformer du repère global au repère local de l'écouteur
        cos = np.cos(teta_rad)
        sin = np.sin(teta_rad)
        rotation_matrix = np.array([
            [cos, -sin],
            [sin,  cos]
        ])
        # Transformer les coordonnées de l'émetteur dans le repère local de l'écouteur
        emetteur_position_local = np.dot(rotation_matrix, np.array([x_emetteur, y_emetteur]))
        return emetteur_position_local

    def ear(self, depreciate_sound, depreciate_distance, frequence, angle_bruit):
        """Ecoute le bruit"""
        intensity_x, intensity_y = self.trigo_ear(depreciate_distance, angle_bruit) 
        self.liste_bruit_entendu = [frequence, intensity_x, intensity_y, depreciate_sound]


    def reset_bruit_entendu(self):
        """Reset le bruit entendu si aucun nouveau bruit n'est entendu"""
        self.liste_bruit_entendu = [-1, -1, -1, -1]


    def get_bruit_entendu(self):
        """Get le bruit entendu. Cette méthode permet d'encapsuler les valeurs du bruit entendu dans cette méthode, de rendre le code plus modulable si je veux faire des tests dedans ou que je change la forme de liste_bruit_entendu"""
        return self.liste_bruit_entendu[0], self.liste_bruit_entendu[3]


    def fonction_vision(self, list_visible_entity, list_audible_entity):
        """ENTREE : rien de spécial, SORTIE : [vision_rayon - distance, one hot encoding type d'entité] par part de vision"""
        
        # Initialisation de la vision
        nbr_entrees = self.vision_nbr_parts*self.nbr_neurones_par_part + self.nbr_neurones_entrees_supplementaires
        vision = - np.ones(nbr_entrees) # Vaut -1 par défaut (normalisation entre -1 et +1) +neurones_supplementaires pour ma classe par exemple
        
        # Initialisation des distances minimales
        min_distances = np.full(self.vision_nbr_parts, np.inf)
        
        # Calcul des angles des bords des parts de camembert
        angles_parts = np.linspace(self.teta - self.vision_demi_angle, self.teta + self.vision_demi_angle, self.vision_nbr_parts + 1) % 360 # On compte dans le sens trigo les angles et en degrés dans un repere y vers le haut classique


        # Initialiser les neurones supplémentaires
        index_vision = 0
        for neurone, nbr_neurones_correspondant in self.liste_entrees_supplementaires.items(): #neurons in liste_entrees_supplementaires are in the same order that the ones of the entrance layer of the brain

            # Initialiser les valeurs de vision correspondantes
            if neurone == "age":
                vision[index_vision] = 2 * self.age / age_maximum - 1  # normalisation entre [-1, 1] tanh
            elif neurone == "vie":
                vision[index_vision] = 2 * self.vie / self.max_vie_individu - 1  # normalisation entre [-1, 1] tanh
            elif neurone == "energie":
                vision[index_vision] = 2 * self.energie / self.max_energie_individu - 1  # normalisation entre [-1, 1] tanh
            elif neurone == "regime":
                if lvl_max_eat_scale == 0:
                    vision[index_vision] = 0  # Valeur neutre car il n'y a qu'un seul régime
                else:
                    vision[index_vision] = 2 * self.regime / lvl_max_eat_scale - 1
            elif neurone == "oreille":
                spec_max_bruit = None  # Initialisation de la liste de bruit max
                max_depreciate_sound = 0 # Initialisation du max, si aucun bruit emis, on a freq, intensité = 0,0 par défaut
                # Parcourir chaque élément dans liste_audible_entity
                for entity, distance, angle in list_audible_entity:
                    # Calculer depreciate_sound et depreciate_distance pour l'élément actuel
                    depreciate_sound, depreciate_distance = functions.depreciate_distance_sound(entity[0].body.bruit_emis[1], distance, self.ecoute_rayon)
                    # Comparer avec le depreciate_sound maximum actuel
                    if depreciate_sound > max_depreciate_sound:
                        max_depreciate_sound = depreciate_sound
                        spec_max_bruit = [depreciate_sound, depreciate_distance, entity[0].body.bruit_emis[0], angle] #pour le calcul de ear dans vision
                # Récupérer l'objet.body.bruit_emis de l'entité avec le depreciate_sound maximal
                if spec_max_bruit:
                    depreciate_sound, depreciate_distance, frequence, angle = spec_max_bruit
                    self.ear(depreciate_sound, depreciate_distance, frequence, angle)
                for i in range(nbr_neurones_correspondant):
                    vision[index_vision + 0] = self.liste_bruit_entendu[0] #frequence
                    vision[index_vision + 1] = self.liste_bruit_entendu[1] #intensité axe vision
                    vision[index_vision + 2] = self.liste_bruit_entendu[2] #intensité gauche axe vision
            elif neurone == "is_giving_birth":
                vision[index_vision] = 1 if self.bb_being_created else -1  # normalisation entre [-1 = FAUX, 1 = TRUE] tanh
            elif neurone == "is_stomach_full":
                vision[index_vision] = 1 if self.energie >= self.max_energie_individu * facteur_energie_eat else -1 # normalisation entre [-1 = FAUX, 1 = TRUE] tanh

            # Incrémenter l'index pour les prochaines entrées
            index_vision += nbr_neurones_correspondant



        # get the visible entities
        for entity, distance, angle in list_visible_entity:
            part_index = self.determine_part_index(angle, angles_parts)

            # Type numétirique (Pas de One hot encoding de l'entité
            if part_index != -1 and distance < min_distances[part_index]:
                entity_type = entity[1]
                min_distances[part_index] = distance
                # Distance normalisée pour la vision
                dist_value = 2*(self.vision_rayon - distance)/self.vision_rayon - 1 # Normalisation dans [-1, 1] : part lin fontion tanh
                #dist_value = 4*(self.vision_rayon - distance)/self.vision_rayon - 2 # Normalisation dans [-2, 2] : part lin fontion tanh  -----------------  FONCTION ACTIVATION
                vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part] = dist_value
                # taille_value = vision_normalizer(entity[0].body.r_collision_box_individu if entity_type == "individual" else entity[0].r_hit_box_eatable, self.r_collision_box_individu) #vision_normalizer se décale en fonction de ma taille (toujours 0 pour un indvidu de ma taille et négatif pour les plus petits et positif pour les plus grands)
                # vision[self.nbr_neurones_entrees_supplementaires + part_index * nbr_neurones_par_part + 1] = taille_value
                # Binary encoding of the entity classe
                entity_classe = classes[entity_type]
                binary_encoding = functions.classe_to_binary(entity_classe, nbr_classes)
                for i, bit in enumerate(binary_encoding):
                    vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part + i + 1] = bit
                # If it has additional input to describe the entitie it sees
                if self.liste_entrees_supplementaires_par_part:
                    index_vision = 0
                    for neurone, nbr_neurones_correspondant in self.liste_entrees_supplementaires_par_part.items():
                        if neurone == "know_size":
                            taille_value = vision_normalizer(entity[0].body.r_collision_box_individu if entity_type == "individual" else entity[0].r_hit_box_eatable, self.r_collision_box_individu) #vision_normalizer se décale en fonction de ma taille (toujours 0 pour un indvidu de ma taille et négatif pour les plus petits et positif pour les plus grands)
                            vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part + i + 1 + index_vision] = taille_value
                        elif neurone == "know_diet":
                            diet_value = diet_normalizer(entity[0].body.regime if entity_type == "individual" else None, self.regime)
                            vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part + i + 1 + index_vision] = diet_value
                        
                        index_vision += nbr_neurones_correspondant

                # # If has the capability to know the size of the individual in front of him
                # if self.know_size:
                #     taille_value = vision_normalizer(entity[0].body.r_collision_box_individu if entity_type == "individual" else entity[0].r_hit_box_eatable, self.r_collision_box_individu) #vision_normalizer se décale en fonction de ma taille (toujours 0 pour un indvidu de ma taille et négatif pour les plus petits et positif pour les plus grands)
                #     vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part + i + 2] = taille_value
                # if self.know_diet:
                #     diet_value = diet_normalizer(entity[0].body.regime if entity_type == "individual" else None, self.regime)
                #     vision[self.nbr_neurones_entrees_supplementaires + part_index * self.nbr_neurones_par_part + i + 2] = diet_value



        #print("vision", vision)# if self.know_diet else None
        return vision


    
    def determine_part_index(self, angle, angles_parts):
        """# Fonction pour déterminer l'index de la part de vision où se trouve un angle"""
        for i in range(len(angles_parts) - 1):
            if angles_parts[i] <= angle < angles_parts[i + 1] or (angles_parts[i] > angles_parts[i + 1] and (angle >= angles_parts[i] or angle < angles_parts[i + 1])):
                return i
        return -1 #-1 par défaut


    def attack(self, ecosystem_obj, valeur_sortie_brain, list_reachable_entity):
        """Attaquer un individu dans la hit box de l'attaquant et vu par l'attaquant
        IN : "list_reachable_entity" is a list of individuals in range"""
        self.attack_bool = True

        # Initialisation
        teta = self.teta
        closest_entity = None
        min_angle_diff = float('inf')

        # Fonction pour calculer la différence d'angle minimale
        def angle_diff(angle1, angle2):
            diff = abs(angle1 - angle2) % 360
            return min(diff, 360 - diff)

        

        # Calcul des angles des bords des parts de vision
        angles_parts = np.linspace(teta - self.vision_demi_angle, teta + self.vision_demi_angle, self.vision_nbr_parts + 1) % 360 # On compte dans le sens trigo les angles et en degrés dans un repere y vers le haut classique

        # Parcourir les entités visibles
        for entity, distance, angle in list_reachable_entity:

            part_index = self.determine_part_index(angle, angles_parts)

            # Vérifier si l'entité est visible dans la partie de vision
            if part_index != -1:
                # Calculer la différence d'angle par rapport au centre de la vision
                angle_diff_value = angle_diff(teta, angle)
                if angle_diff_value < min_angle_diff:
                    min_angle_diff = angle_diff_value
                    closest_entity = entity

        # Si une entité cible a été trouvée, appliquer les dégâts
        if closest_entity:
            # Vérifier que c'est bien notre attaque qui a tué l'entité
            was_dead = True if closest_entity[0].body.vie <= 0 else False
            # Appliquer les dégâts
            damage = max_attack_damage * valeur_sortie_brain * self.r_collision_box_individu/r_collision_box_individu_init #LINEAIRE : plus l'attaquant est gros plus il inflige de damage. /r_collision_box_individu_init pour que ce soit un facteur  au départ
            closest_entity[0].body.vie -= damage #plus l'attaqué est gros plus il a de vie
            # For the visualisation
            closest_entity[0].body.is_losing_life = True

            if closest_entity[0].body.vie <= 0 and not was_dead:
                closest_entity_body = closest_entity[0].body
                # lay meat on the ground when die*
                ecosystem_obj.add_eatable("meat", energy=closest_entity_body.energie, size=closest_entity_body.r_collision_box_individu, position=(closest_entity[0].body.position[0], closest_entity[0].body.position[1])) #mangeable en fonction de la diet
                self.compteur_kill += 1

        self.energie -= max_energie_depensee_attack * valeur_sortie_brain # Coût de l'attaque


    def share_ressources(self, ecosystem_obj, valeurs_sortie_brain):
        """Share ressources means the individual loose energy to create a split in front of him. The others can eat it"""
        # Calculer l'énergie maximale que l'individu peut cracher
        max_energy_output = valeurs_sortie_brain * self.max_energie_individu
        # Déterminer la quantité d'énergie que l'individu peut réellement cracher
        energy = min(self.energie, max_energy_output)
        # Calculer la taille de l'objet en fonction de l'énergie disponible
        size_trophallaxie = self.r_collision_box_individu * energy / self.max_energie_individu

        # Soustraire l'énergie utilisée pour cracher l'objet de l'énergie disponible de l'individu
        self.energie -= energy
        x_drop, y_drop = self.position[0] - (2 + size_trophallaxie + self.r_collision_box_individu + np.ceil(pas_de_temps*1*self.facteur_multiplicatif_deplacement)) * np.cos(np.deg2rad(self.teta)), self.position[1] - (2 + size_trophallaxie + self.r_collision_box_individu + np.ceil(pas_de_temps*1*self.facteur_multiplicatif_deplacement)) * np.sin(np.deg2rad(self.teta))
        ecosystem_obj.add_eatable("trophallaxy", energy=energy, size=size_trophallaxie, position=(x_drop, y_drop)) #mangeable par tout le monde
        self.compteur_trophallaxie += 1

