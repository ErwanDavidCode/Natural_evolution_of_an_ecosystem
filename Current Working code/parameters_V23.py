import numpy as np
import random
import time

import functions_V23 as functions


# Pour que ca marche il faut l'arborescence de fichier : 
# Un fichier "on s'en fout du nom" dans lequel il y a un dossier "data" et un dossier "Roboto" (police d'écriture pour les ID) et nos .py
# Un fichier "Videos" dans lequel seront sauvegardées les vidéos




[]
True, False, None

start_again_until_alive_pop = True                                                # True or False. Relancer la simulation jusqu'à ce qu'il y ait une population vivante
statistics = False                                                                # False, True. A LANCER APRES LACEMENT DE LA SIMULATION AVEC TOUT LE MONDE. Affiche les ID des individus aux comportements les plus intéressants. Doit etre False pour lancer simulation complete
affichage_complet = False                                                         # True or False. Afficher les détails de chaque individu (rayon vision etc ...). Que pour simulation_seule

fichier_path = './data/historique_individus'                                      # Chemin pour: characteristics_ID_individu / liste_ID_alone_simulation / statistics. Doit etre de la forme "historique_individus_1721293772"
characteristics_ID_individu = None                                      # None or Int. A LANCER APRES LACEMENT DE LA SIMULATION AVEC TOUT LE MONDE. Lancer la simulation complete si characteristics_ID_individu = None  /  afficher détail individu si characteristics_ID_individu = un ID (int)
son_individu = None                                                               # None or Int. On ne spécifie qu'un individu a la fois, on se place dans ses oreilles. Ne peut etre fais qu"avant de lancer la simulation avec tout le monde.      
liste_ID_alone_simulation = []                                                    # List(Int) or []. A LANCER APRES LACEMENT DE LA SIMULATION AVEC TOUT LE MONDE. Lancer la simulation avec la liste des individu d'ID : ID_alone_simulation / ne pas relancer de simulation si None. Doit etre [] pour lancer simulation complete

#If we want to initialize every individual with a specific brain - Both variables have to be filled
brain_to_be_used = None                                                           # None or List[List[float]]. MUST BE WITHOUT ADDITIONNAL BRAIN FEATURE NEURONS (like a normal initial brain). None to initialize with random brains. List of brains to initialize the individuals with. Must be of the same size as the number of individuals
biais_to_be_used = None                                                           # None or List[float]. MUST BE WITHOUT ADDITIONNAL BRAIN FEATURE NEURONS (like a normal initial brain). None to initialize with random biais. List of biais to initialize the individuals with. Must be of the same size as the number of individuals
#If we want to initialize every individual as the same individual
individu_to_be_used = None                                                        # None or Int. Uses the "fichier_path" to select the individual. None to initialize with random individuals. Int to initialize the individuals as all the individu_to_be_used.
#If we want to start a simulation with an entire population
load_population = None                                                            # None ou path. METTRE UN PATH VA INITIALISER LES "nbr_individus_init" DERNIERS INDIVIDUS DE CE FICHIER. Doit etre None pour lancer simulation complete normale.



#PARAMETRES
time_to_save_video = 10000 #Si le temps attend cette valeur, on sauvegarde la vidéo et l'historique dans un fichier avec un nom unique. BUT : avoir une save des vidéos et historiques intéressants
duree_simulation = 20000000 #Can do a Ctrl+C to stop the simulation and still have the video and the history of individuals
end_time = time.strptime("24 Jun 2024 14:30:00", "%d %b %Y %H:%M:%S") #lancer la simulation jusquà ce temps
pas_de_temps = 1
nbr_iterations_shelve_update = 1500
sampling_rate = 20 #temps échantillonage des données
saving_rate = 500 #temps sauvegarde données dans csv et vider buffers
factor_recording_video = 2 # On enregesi tre une image tous les "factor_recording_video" pas de temps. Plus factor_recording_video est grand, moins on enregistre d'images. 1 pour enregistrer toutes les images
fps = 30 / factor_recording_video #fps de la vidéo

size_modification = 1 #Divise par 2 la taille de tout : revient a faire un zoom out. J'ai mis ca car limite de carte 4000*4000 avec open cv. Le size_multiplication permet de simuler des cartes plus grandes. #SIZE MODIFCATION



# #initialisation - big simu -------------------------------------------------------------------------------
# #size_modification should be 2
# taille_carte = 4000
# max_individu = 100000
# nbr_individus_init = 300
# max_plantes = 100000
# nbr_plantes_init = 425*size_modification
# nbr_min_plant_init = 20*size_modification #nombre de nouvelle graine quand plus de plante du tout


#initialisation - normal simu -------------------------------------------------------------------------------
#size_modification should be 2
taille_carte = 1200
max_individu = 100000
nbr_individus_init = 100
max_plantes = 100000
nbr_plantes_init = 120*size_modification
nbr_min_plant_init = 5*size_modification #nombre de nouvelle graine quand plus de plante du tout



#paramètres plantes
age_plant_max = 4000 #age pour qu'une plante meurt
energy_plant_bb = 60/size_modification #energy pour que la plante fasse un bébé et DIVISEE PAR 2 son énergie. C'est donc l'energie max apportée a un individu quand il mange une plante par conservation de l'énergie #SIZE MODIFCATION
age_eatable_perish = 250 #age pour qu'un eatble pourisse et disparaisse
range_max_spawn_plant = 100/size_modification #range max pour qu'une plante créer un bébé autour d'elle #SIZE MODIFCATION
bouffe_taille_max = 2/size_modification #taille max d'un steak ou trophallaxie sans etre dé-doublé lors de son spawn #SIZE MODIFCATION
#Attention, en plus des plant min, une seed de plante apparait tous les 2000 itérations

#Energie
solar_energy = np.sqrt(taille_carte)/11 #Energie solaire par pas de temps, sert a faire pousser les plantes
gain_max_energy_per_turn = 0.1/size_modification #max gain of energy per turn for a plant
print(f"BB_PLANTE tous les : {np.ceil(energy_plant_bb/(solar_energy/nbr_plantes_init)/2)} turns pour NBR_PLANT = {nbr_plantes_init}")
print(f"Min duration to create BB_PLANT : {np.ceil(energy_plant_bb/gain_max_energy_per_turn/2)} turns")
#paramètres individus
age_maximum = 2500
# Les individus commeneent avec une énergie/2 et une vie de 100
max_energie_individu_init = 150/size_modification #SIZE MODIFCATION
max_vie_individu_init = 100/size_modification #SIZE MODIFCATION
age_min_to_attack = 50 
age_min_to_childbirth = 50
facteur_energie_eat = 0.9 # EN % de l'energie tot. Au dela de ce %tage d'energie, l'individu ne mange plus. Symbolise un estomac de taille fini. Pourquoi pas 100 ? Car l'individu n'a que trs rarement 100% de son energie quand il mange car il perd de l'energie en bougeant (notamment en bougeant vers un eatable)

#create bb
facteur_energie_creer_bb = 0.61 # EN % de l'energie tot. Required energy to create a baby, doit etre supérieur a energie_init. Plus un individu est gros plus il a besoin d"energie pour creer BB.
facteur_energie_depensee_creer_bb = 0.31 # EN % de l'energie tot. Energie perdu quand on enfante, 1 enfants max 
seuil_creer_bb = 0 #seuil de sortie neurone pour creer un bb si il a le neurone de sortie "creer_bb"
#eat trophallaxie
seuil_trophallaxie = 0.01 #seuil pour trophallaxie
#Attack individu
max_attack_damage = 200/size_modification #SIZE MODIFCATION
max_energie_depensee_attack = 0.2/size_modification # ex: 0.1 => lot of pple were constantly attacking #SIZE MODIFCATION
seuil_attaque = 0.01 #seuil pour attaquer
#moving
facteur_multiplicatif_perte_vie = 0.01/size_modification #plus on augmente, plus ils perdent de la vie. 0.11 marche bien
facteur_multiplicatif_deplacement_init = 2/size_modification #SIZE MODIFCATION
#hit box
r_collision_box_individu_init = 2/size_modification #range to collide with other individuals. Cannot mute, just depends on the size of the individual. C'est ce qui permet d'infliger +/- de dégats #SIZE MODIFCATION
r_eat_box_individu_init = r_collision_box_individu_init * 3 #range to eat
r_attack_box_individu_init = r_collision_box_individu_init * 10 #range to attack

r_hit_box_eatable_init = 1 #size of the plant. Peut avoir des tailles différentes pour les steak et trophallaxie fonction de la taille de l'individu. Doit etre de la meme taille que r_hit_box_individu a l'initialisation ? #SIZE MODIFCATION

#vision
vision_rayon_init = 800/size_modification #ATTENTION : il faut que ce soit supérieur à (r_hit_box_individu+r_hit_box_plante) #SIZE MODIFCATION
vision_demi_angle_init = 20 #degrés
max_rotation_init = 30 #degrés du demi angle

#bruit
ecoute_rayon_init = 200/size_modification #parametre physique qui dépend du milieu on va dire #SIZE MODIFCATION
seuil_bruit = 0 #doit etre supérieure à 0


# Avoir des régime est complexe car on commence avec un régime omnivore qui nous %2 les gains de tout type de nouriture.
lvl_max_eat_scale = 4 #Vaut 0: tout le monde mange viande ou plante avec la meme energie et vie. Ou vaut >0: si on veut plus de diversité. ATTENTION : dans le cas >0, les extremes ne peuvent manger que plantes ou viandes. Spawn en //2 (au milieu)

# Couleurs graphe + simulation #BGR in OpenCV. Uniquement la couleurs des classes
colors = {
    0: (0, 255, 0),       #classe 0 plante
    1: (38, 103, 167),          #classe 1 trophallaxie
    2: (0, 0, 240),         #classe 2 meat
    3: (255, 0, 0),        #classe 3 individu
    4: (0, 110, 24),      #classe 4 individu de régime 0
    5: (91, 160, 123),      #classe 5 individu de régime 1
    6: (142, 142, 0),       #classe 6 individu de régime 2
    7: (0, 0, 0),       #classe 7 individu de régime 3
    8: (0, 180, 240),       #classe 8 individu de régime 4
    9: (0, 127, 255),     #classe 9 individu de régime 5
    10: (0, 7, 126),          #classe 10 individu de régime 6
}

#proba body
nbr_modifications_body_init = 8
nbr_modifications_body = 2 #less physical mutations than brain mutations for the brain to have the time to adapt to the new body

proba_modifier_neurone_feature = 0.04 #Equi chance d'ajouter n'importe quel neurone feature
proba_modifier_regime = 0.05 if lvl_max_eat_scale !=0 else 0 #Equi chance d'ajouter + ou -1 à au régime alimentaire
proba_modifier_size = 0.05 #augmente autant eat, collision et attack box
proba_modifier_speed = 0.05 #augmente facteur_multiplicatif_deplacement
proba_modifier_vision = 0.04 #Equi chance de modifier angle ou rayon vision. Chance plus faible car moins de désavantages à avoir une vision plus grande
proba_modifier_rotation = 0.05 #Equi chance de modifier la rotation max d'un coup
proba_modifier_ears = 0.04 #Equi chance de modifier le rayon d'écoute

proba_rien_faire_body = 0.2

# Définitions des différents types d'entités présentes dans la simulation
classes = { 
    "plant": 0, #plante
    "trophallaxy": 1, #crachat
    "meat": 2, #steak
    "individual": 3, #individu
}


# One hot encoding
vision_nbr_parts_init = 2

liste_entrees_supplementaires_init = {} #doit etre complementaire a liste_entrees_supplementaires_possibles_init
liste_entrees_supplementaires_par_part_init = {}
liste_sorties_supplementaires_init = {} #doit etre complementaire a liste_sorties_supplementaires_possibles_init

nbr_neurones_entrees_supplementaires_init = sum(liste_entrees_supplementaires_init.values()) #pas besoin d'indiquer notre classe dans un cas simple  ou tous les individus sont de la classe individu
nbr_neurones_entrees_supplementaires_par_part_init = sum(liste_entrees_supplementaires_par_part_init.values()) #pas besoin d'indiquer notre classe dans un cas simple  ou tous les individus sont de la classe individu
nbr_neurones_sorties_supplementaires_init = sum(liste_sorties_supplementaires_init.values())

nbr_classes = len(classes)
nbr_neurones_par_part_classe = functions.neurones_par_part_vision(nbr_classes)
nbr_neurones_par_part_init = nbr_neurones_par_part_classe + 1 + nbr_neurones_entrees_supplementaires_par_part_init #+1 car on ajoute la distance


nbr_entrees_init = vision_nbr_parts_init*nbr_neurones_par_part_init + nbr_neurones_entrees_supplementaires_init #entree
nbr_neurones_de_base_sortie = 2 #(2) velocité et angle
nbr_sorties_init = nbr_neurones_de_base_sortie + nbr_neurones_sorties_supplementaires_init #(2) velocité et angle + d'éventuels autres sorties

#Liste des neurones ajoutables, les renseigner ici pour avoir le nbr de modifs possible, mais aussi coder les méthodes asocié dans "body" vision ou "process_additional_outputs"
#Pas besoin de voir sa taille car vision_normalizer se décale en fonction de ma taille (toujours 0 pour un indvidu de ma taille et négatif pour les plus petits et positif pour les plus grands)
liste_entrees_supplementaires_possibles_init = {"energie" : 1, "regime" : 1, "is_giving_birth" : 1, "is_stomach_full" : 1, "oreille" : 3, "vie" : 1} #liste des entrees possiblement ajouter à l'avenir. FORMAT : {"nom_entree_supplementaire" : nbr de neurones correspondant}
liste_entrees_supplementaires_possibles_par_part_init = {"know_size" : 1, "know_diet" : 1} #Pour l'instant : ne marche qu'avec des neurones de taille 1 ???????????????????????????????????????
liste_sorties_supplementaires_possibles_init = {"attaque" : 1, "trophallaxy" : 1, "bouche" : 2, "creer_bb" : 1} #liste des sorties possiblement ajouter à l'avenir. FORMAT : {"nom_entree_supplementaire" : nbr de neurones correspondant}



#proba cerveau
nbr_connexions_init = 2 #DOIT ETRE <= NBR SORTIE INIT (=2). nombre de connexions initiales cerveaux par neurones d'entrée
nbr_connexions_hidden = 2 #DOIT ETRE <= NBR SORTIE INIT (=2). nombre de connexions cerveaux pour chaque nv neuronnes feature ajouté (en entrée ou sortie) vers les hidden
nbr_modifications_brain_init = 8
nbr_mutations_brain = 5

variation_mutation_poids = 0.1
variation_mutation_biais = 0.05
# La somme des proba ne doit pas forcément être égale à 1
proba_rien_faire_init = 0.01
#proba_modifier_poid_et_biais_init = 0.4 #Sert que si mutate_weights_and_biais est décommenté
proba_modifier_poid_init = 0.75
proba_modifier_biais_init = 0.6
proba_modifier_fonction_init = 0.02
proba_remplacer_poid_init = 0.08
proba_remplacer_biais_init = 0.05
proba_creer_poid_init = 0.45
proba_ajouter_neurone_init = 0.07
proba_supprimer_neurone_init = 0.07
proba_supprimer_poid_init = 0.05
proba_supprimer_biais_init = 0.05
#meta proba
nbr_modifications_proba = 5 #every probas mutation are equiprobable
alpha_init = 0.3 #brain proba mutation rate, the bigger it is, the most the mutation proba can change. It is the same rate for all the mutation proba. This alpha is mutated by the beta
beta = 0.1 #DO NOT MUTATE. alpha mutation rate, the bigger it is, the most alpha can change, so the more adaptable the individual can become. Should be small. Is fixed