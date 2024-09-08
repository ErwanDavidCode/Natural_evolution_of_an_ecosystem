import math
import numpy as np
import random
import colorsys
import shelve


def neurones_par_part_vision(nbr_classes):
    """Calcule le nombre de neurones par part de vision en fonction du nombre de classes. ENCODAGE BINAIRE"""
    if nbr_classes < 2:
        raise ValueError("Le nombre de classes doit être au moins 2.")
    
    return math.ceil(math.log2(nbr_classes))


def classe_to_binary(classe, nbr_classes):
    """Convertit une classe en encodage binaire en fonction du nombre de neurones par part de vision. BUT : que ca fasse moins de neurones en entrées que du one hot encoding"""
    if nbr_classes < 2:
        raise ValueError("Le nombre de classes doit être au moins 2.")

    neurones = neurones_par_part_vision(nbr_classes)
    return [(classe >> i) & 1 for i in reversed(range(neurones))]


# nbr_classes = 2
# classe = 1
# print(classe_to_binary(classe, nbr_classes)) 


def create_color_scale(lvl_max_eat_scale):
    # Créer une échelle de couleurs du bleu cyan au rose rouge
    colors = []
    for i in range(lvl_max_eat_scale + 1):
        # Calculer le ratio de progression
        if lvl_max_eat_scale == 0:
            ratio = 0
        else:
            ratio = i / lvl_max_eat_scale
        
        # Interpoler entre bleu cyan (0, 255, 255) et rose rouge (255, 0, 255)
        blue = int(255 - (255 - 255) * ratio)
        green = int(255 - (255 - 0) * ratio)
        red = int(0 + (255 - 0) * ratio)
        
        # Ajouter la couleur à la liste (format BGR pour OpenCV)
        colors.append((blue, green, red))
    
    return colors



def depreciate_distance_sound(intensite, distance, bruit_rayon):
    """Déprécie l'intensité du son en fonction de la distance. LINAIRE DECROISSANTE EN FONCTION DE LA DISTANCE. Intensité entre 0 et +1 si seuil est à 0."""
    depreciate_sound = intensite * (1 - distance / bruit_rayon) #decroissance linéaire du son en fonction de la distance
    depreciate_distance = intensite * distance #distance emetteur - recepteur fonction de l'intensité du son
    return depreciate_sound, depreciate_distance



def get_list_last_individuals(fichier_path, nbr_elem):
    """retourne la liste des obj individus dans le fichier"""
    with shelve.open(fichier_path) as db:
        keys = list(db.keys())
        num_objects_to_return = min(nbr_elem, len(keys))
        last_keys = keys[-num_objects_to_return:]
        last_objects = [db[key] for key in last_keys]
    return last_objects


import copy

def get_list_same_individuals(fichier_path, individu_to_be_used, nbr_elem):
    """Retourne une liste contenant 'nbr_elem' fois une copie profonde de l'objet individu_to_be_used dans le fichier"""
    with shelve.open(fichier_path) as db:
        individu_selectionne = db.get(str(individu_to_be_used), None)
        if individu_selectionne is None:
            raise ValueError(f"L'individu '{individu_to_be_used}' n'existe pas dans le fichier.")
        return [copy.deepcopy(individu_selectionne) for _ in range(nbr_elem)]

