import numpy as np
import random

from parameters_V23 import *


class Eatable:
    """Classe qui regrouppe les eatables (plantes, trophallaxie et steak)"""
    __slots__ = ['position', 'r_hit_box_eatable', 'age', 'energy']
    def __init__(self):
        self.position = [0,0]
        self.r_hit_box_eatable = r_hit_box_eatable_init #r_hit_box_eatable_init = 2, size of a default plant 

        self.age = random.randint(0, 200) #pour pas qu'elles ne fassent tous des bb plantes en meme temps. Vaut cette valeur quand créer nv plante
        self.energy = energy_plant_bb/2 + random.uniform(0, energy_plant_bb/4) #Energy of the eatable that will be passed when eaten. Vaut PAS cette valeur quand créer nv plante

    def grow(self):
        self.age = self.age + 1


