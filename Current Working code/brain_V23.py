import numpy as np
import random

from parameters_V23 import *




# =============================================================================
# Fonctions activations
# =============================================================================

def SIG(x, k=1): #sigmoid non translaté . (non Translaté de 0.5 = en 0). elle vaut donc 0.5 en 0  -----------------  FONCTION ACTIVATION
    return (1/(1+np.exp(-k * x)))


#tanh(de numpy)


def fonction_tanh_pallier(x, k=100):
    return np.tanh(x / 2) * (SIG(x - 0.1, k) + SIG(-x - 0.1, k))







class Hysteresis:
    def __init__(self, threshold_up=1, threshold_down=-1):
        """Classe pour la fonction d'activation à hystérésis avec mémoire d'état.
        Seuils définissent les points de commutation.
        """
        self.state = 0  # État initial, entre -1 et 1
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down

    def __call__(self, x):
        """Activation avec effet d'hystérésis classique.
        Retourne 1 si l'entrée dépasse le seuil supérieur, -1 si elle est en dessous du seuil inférieur,
        et conserve l'état sinon.
        """
        if x > self.threshold_up:
            self.state = 1
        elif x < self.threshold_down:
            self.state = -1
        # Conserve l'état si x est entre les seuils
        return self.state






class Brain:
    def __init__(self):
        """Les attributs de brain sont les variables qui seront modifiés par des méthodes de la classe brain && les variables évolutives qui caractérisent le brain (valeurs_neurones, biais_neurones, matrice_poids)."""
        
        self.nbr_entrees = nbr_entrees_init
        self.nbr_sorties = nbr_sorties_init

        #création des matrices des neurones
        self.valeurs_neurones = np.zeros(self.nbr_entrees + self.nbr_sorties)
        self.activation_functions = [np.tanh] * (self.nbr_entrees + self.nbr_sorties) #fonction d'activation pour chaque neurone
        self.biais_neurones = np.zeros(self.nbr_entrees + self.nbr_sorties)
        
        #création de la matrice des poids
        self.matrice_poids = np.zeros((self.nbr_entrees + self.nbr_sorties, self.nbr_entrees + self.nbr_sorties))
        
        self.connect_entry_to_output(0, self.nbr_entrees - 1) #connecter les neurones d'entrée aux neurones de sortie
        #pas de biais pour les neurones d'entrées. Ils représentent les valeurs d'entrées litéralement
        for neurone in range(self.nbr_entrees, self.nbr_entrees + self.nbr_sorties) :
            self.biais_neurones[neurone] = random.uniform(-0.1, 0.1) #close to 0 for the initialization

        # La somme des proba doit être égale à 1
        self.proba_rien_faire = proba_rien_faire_init
        #self.proba_modifier_poid_et_biais = proba_modifier_poid_et_biais_init #Sert que si mutate_weights_and_biais est décommenté
        self.proba_modifier_poid = proba_modifier_poid_init
        self.proba_modifier_biais = proba_modifier_biais_init
        self.proba_modifier_fonction = proba_modifier_fonction_init
        self.proba_remplacer_poid = proba_remplacer_poid_init
        self.proba_remplacer_biais = proba_remplacer_biais_init
        self.proba_creer_poid = proba_creer_poid_init
        self.proba_ajouter_neurone = proba_ajouter_neurone_init
        self.proba_supprimer_neurone = proba_supprimer_neurone_init
        self.proba_supprimer_poid = proba_supprimer_poid_init
        self.proba_supprimer_biais = proba_supprimer_biais_init

        self.alpha = alpha_init


    def connect_entry_to_output(self, ind_min, ind_max):
        """On connecte tous les neurones entre [ind_min, ind_max] aux neurones de sortie"""
        # Connecter tous les neurones d'entrée aux neurones de sortie initialement
        for neurone_entree in range(ind_min, ind_max+1):
            # Choisir des indices de neurones de sortie sans répétition
            sorties_connectees = np.random.choice(range(self.nbr_entrees, self.nbr_entrees + self.nbr_sorties), nbr_connexions_init, replace=False)
            for sortie in sorties_connectees:
                self.matrice_poids[neurone_entree][sortie] = np.random.uniform(-1, 1)
    

    def connect_entry_to_hidden(self, ind_min, ind_max):
        """
        Connecte les neurones entre [ind_min, ind_max] soit aux neurones cachés soit aux neurones de sortie. """
        # Définir les indices des neurones cachés et de sortie
        hidden_indices = range(self.nbr_entrees + self.nbr_sorties, len(self.valeurs_neurones))
        output_indices = range(self.nbr_entrees, self.nbr_entrees + self.nbr_sorties)
        
        # Créer une liste de tous les neurones disponibles (cachés + sorties)
        neurons_available = list(hidden_indices) + list(output_indices)

        # Connecter chaque neurone de la plage [ind_min, ind_max] à des neurones aléatoires
        for neurone in range(ind_min, ind_max + 1):
            hidden_connectees = np.random.choice(neurons_available, nbr_connexions_hidden, replace=False)
            for hidden in hidden_connectees:
                # Assigner un poids aléatoire à la connexion
                self.matrice_poids[neurone][hidden] = np.random.uniform(-1, 1)


    def connect_hidden_to_output(self, ind_min, ind_max):
        """
        Connecte les neurones d'entrée ou hidden aux nouveaux neurones de sortie"""
        # Indices des neurones d'entrée et intermédiaires
        input_indices = range(0, self.nbr_entrees)
        hidden_indices = range(self.nbr_entrees + self.nbr_sorties, len(self.valeurs_neurones))
        
        # Liste des neurones disponibles pour les connexions
        neurons_available = list(input_indices) + list(hidden_indices)

        # Connexion des neurones de sortie aux neurones disponibles
        for neurone in range(ind_min, ind_max + 1):
            for _ in range(nbr_connexions_hidden):
                connected = np.random.choice(neurons_available)
                self.matrice_poids[connected][neurone] = np.random.uniform(-1, 1)



    def propagation_avant(self, liste_entree):
        """
        Propagation avant avec fonctions d'activation spécifiques pour chaque neurone.
        Utilise des opérations vectorielles pour optimiser les calculs.
        """
        # Mise à jour des neurones d'entrée avec les valeurs d'entrée
        self.valeurs_neurones[:self.nbr_entrees] = liste_entree
        # Calcul des entrées des neurones cachés et de sortie avec un produit matriciel
        outputs = np.dot(self.matrice_poids.T, self.valeurs_neurones) + self.biais_neurones

        # Application des fonctions d'activation spécifiques à chaque neurone
        # Utilisation d'une approche plus rapide avec numpy et des fonctions d'activation vectorisées
        activation_results = np.array([
            func(outputs[i]) for i, func in enumerate(self.activation_functions)
        ])

        # Mise à jour des valeurs des neurones
        self.valeurs_neurones = activation_results

        # Extraction et retour des valeurs des neurones de sortie
        return self.valeurs_neurones[self.nbr_entrees:self.nbr_entrees + self.nbr_sorties]



    def update_nbr_part_entrees(self, augmentation, nbr_neurones_par_part):
        """Ajoute ou retire des entrées au cerveau : dépend de la mutation physique 'mutate_nbr_part' """
        #on ajoute pas un nuerone mais nbr_neurone_par_part car on rajoute une part de vision
        if augmentation > 0:
            # Ajouter des lignes et des colonnes dans matrice_poids
            for _ in range(nbr_neurones_par_part):
                self.matrice_poids = np.insert(self.matrice_poids, self.nbr_entrees, 0, axis=0)  # Ajouter une ligne
                self.matrice_poids = np.insert(self.matrice_poids, self.nbr_entrees, 0, axis=1)  # Ajouter une colonne

            # Ajouter des éléments nuls dans valeurs_neurones et biais_neurones
            self.valeurs_neurones = np.insert(self.valeurs_neurones, self.nbr_entrees, [-1] * nbr_neurones_par_part)
            self.biais_neurones = np.insert(self.biais_neurones, self.nbr_entrees, [0] * nbr_neurones_par_part)
            self.activation_functions = np.insert(self.activation_functions, self.nbr_entrees, [np.tanh] * nbr_neurones_par_part)

            # Mettre à jour le nombre d'entrées
            self.nbr_entrees += nbr_neurones_par_part # A mettre avant connect_neuron_to_output car il utilise nbr_entrees   
            # COnnect neux neurons to hidden as the nbr of mutation is fixed
            self.connect_entry_to_hidden(self.nbr_entrees - nbr_neurones_par_part, self.nbr_entrees - 1)     

        elif augmentation < 0 and self.nbr_entrees >= nbr_neurones_par_part:
            # Supprimer des lignes et des colonnes dans matrice_poids
            for i in range(nbr_neurones_par_part):
                self.matrice_poids = np.delete(self.matrice_poids, self.nbr_entrees - i - 1, axis=0)  # Supprimer la ligne
                self.matrice_poids = np.delete(self.matrice_poids, self.nbr_entrees - i - 1, axis=1)  # Supprimer la colonne
            
                # Supprimer des éléments dans valeurs_neurones et biais_neurones
                self.valeurs_neurones = np.delete(self.valeurs_neurones, self.nbr_entrees - i - 1)
                self.biais_neurones = np.delete(self.biais_neurones, self.nbr_entrees - i - 1)
                self.activation_functions = np.delete(self.activation_functions, self.nbr_entrees - i - 1)

            # Mettre à jour le nombre d'entrées
            self.nbr_entrees -= nbr_neurones_par_part


    def update_know_something(self, vision_nbr_parts, nbr_neurones_par_part, nbr_neurones_entrees_supplementaires, augmentation=1, position=0):
        """Ajoute ou retire des entrées au cerveau au tout début : neurones supplémentaires.
        This function is called once we add the "know_size" feature neuron at the entrance net (not when we add a new "vision part" -> this is the purpose of an other function)"""
        if augmentation > 0:  # Ajouter des neurones supplémentaires
            # Ajouter des lignes et des colonnes dans matrice_poids
            indices = [nbr_neurones_entrees_supplementaires + (i + 1) * nbr_neurones_par_part for i in range(vision_nbr_parts)]
            for indice in reversed(indices):
                # Ajouter des lignes et des colonnes dans matrice_poids
                for _ in range(augmentation):
                    self.matrice_poids = np.insert(self.matrice_poids, indice, 0, axis=0)  # Ajouter une ligne
                    self.matrice_poids = np.insert(self.matrice_poids, indice, 0, axis=1)  # Ajouter une colonne
                # Ajouter des éléments nuls dans valeurs_neurones et biais_neurones
                self.valeurs_neurones = np.insert(self.valeurs_neurones, indice, [-1] * augmentation)
                self.biais_neurones = np.insert(self.biais_neurones, indice, [0] * augmentation)
                self.activation_functions = np.insert(self.activation_functions, indice, [np.tanh] * augmentation)

                # Mettre à jour le nombre d'entrées
                self.nbr_entrees += augmentation #est appelé a chaque part
                self.connect_entry_to_hidden(indice, indice + augmentation - 1)


        elif augmentation < 0:  # Ajouter des neurones supplémentaires
            # Ajouter des lignes et des colonnes dans matrice_poids
            indices = [nbr_neurones_entrees_supplementaires + (i + 1) * nbr_neurones_par_part -1 - position * augmentation for i in range(vision_nbr_parts)]
            for indice in reversed(indices):
                # Supprimer des lignes et des colonnes dans matrice_poids
                for _ in range(-augmentation):
                    self.matrice_poids = np.delete(self.matrice_poids, indice, axis=0)
                    self.matrice_poids = np.delete(self.matrice_poids, indice, axis=1)
                # Supprimer des éléments dans valeurs_neurones et biais_neurones
                self.valeurs_neurones = np.delete(self.valeurs_neurones, indice)
                self.biais_neurones = np.delete(self.biais_neurones, indice)
                self.activation_functions = np.delete(self.activation_functions, indice)

            # Mettre à jour le nombre d'entrées
            self.nbr_entrees += augmentation * vision_nbr_parts







    def update_nbr_supplementaire_entrees(self, augmentation=1, position=0):
        """Ajoute ou retire des entrées au cerveau au tout début : neurones supplémentaires.
        This function is called once we add a feature neuron at the entrance net"""

        if augmentation > 0:
            # Ajouter des lignes et des colonnes dans matrice_poids
            for _ in range(augmentation):
                self.matrice_poids = np.insert(self.matrice_poids, position, 0, axis=0)  # Ajouter une ligne
                self.matrice_poids = np.insert(self.matrice_poids, position, 0, axis=1)  # Ajouter une colonne
            # Ajouter des éléments nuls dans valeurs_neurones et biais_neurones
            self.valeurs_neurones = np.insert(self.valeurs_neurones, position, [-1] * augmentation)
            self.biais_neurones = np.insert(self.biais_neurones, position, [0] * augmentation)
            self.activation_functions = np.insert(self.activation_functions, position, [np.tanh] * augmentation)

            # Mettre à jour le nombre d'entrées
            self.nbr_entrees += augmentation
            # COnnect neux neurons to hidden as the nbr of mutation is fixed
            self.connect_entry_to_hidden(0, augmentation-1)
        
        if augmentation < 0:
            # Supprimer des lignes et des colonnes dans matrice_poids
            for _ in range(-augmentation):
                self.matrice_poids = np.delete(self.matrice_poids, position, axis=0)
                self.matrice_poids = np.delete(self.matrice_poids, position, axis=1)
                # Supprimer des éléments dans valeurs_neurones et biais_neurones
                self.valeurs_neurones = np.delete(self.valeurs_neurones, position)
                self.biais_neurones = np.delete(self.biais_neurones, position)
                self.activation_functions = np.delete(self.activation_functions, position)

            # Mettre à jour le nombre d'entrées
            self.nbr_entrees += augmentation


    def update_nbr_supplementaire_sorties(self, augmentation=1, position=0):
        """Ajoute ou retire des sorties au cerveau apres les sorties de bases et avant les neurones des couches intermédiaires : neurones supplémentaires' """
        #on ajoute pas un nuerone mais nbr_neurone_par_part car on rajoute une part de vision
        if augmentation > 0:
            # Ajouter des lignes et des colonnes dans matrice_poids
            for _ in range(augmentation):
                self.matrice_poids = np.insert(self.matrice_poids, self.nbr_entrees + self.nbr_sorties, 0, axis=0)  # Ajouter une ligne
                self.matrice_poids = np.insert(self.matrice_poids, self.nbr_entrees + self.nbr_sorties, 0, axis=1)  # Ajouter une colonne

            # Ajouter des éléments nuls dans valeurs_neurones et biais_neurones
            self.valeurs_neurones = np.insert(self.valeurs_neurones, self.nbr_entrees + self.nbr_sorties, [0] * augmentation)
            self.biais_neurones = np.insert(self.biais_neurones, self.nbr_entrees + self.nbr_sorties, [0] * augmentation)
            self.activation_functions = np.insert(self.activation_functions, self.nbr_entrees + self.nbr_sorties, [np.tanh] * augmentation)

            # Mettre à jour le nombre de sorties
            self.nbr_sorties += augmentation
            # COnnect neux neurons to hidden as the nbr of mutation is fixed
            self.connect_hidden_to_output(self.nbr_entrees + self.nbr_sorties - augmentation, self.nbr_entrees + self.nbr_sorties - 1)
           
        elif augmentation < 0:
            # Supprimer des lignes et des colonnes dans matrice_poids
            for _ in range(-augmentation):
                self.matrice_poids = np.delete(self.matrice_poids, self.nbr_entrees + nbr_neurones_de_base_sortie + position, axis=0)  # Supprimer la ligne. 
                self.matrice_poids = np.delete(self.matrice_poids, self.nbr_entrees + nbr_neurones_de_base_sortie + position, axis=1)  # Supprimer la colonne
            
                # Supprimer des éléments dans valeurs_neurones et biais_neurones
                self.valeurs_neurones = np.delete(self.valeurs_neurones, self.nbr_entrees + nbr_neurones_de_base_sortie + position)
                self.biais_neurones = np.delete(self.biais_neurones, self.nbr_entrees + nbr_neurones_de_base_sortie + position)
                self.activation_functions = np.delete(self.activation_functions, self.nbr_entrees + nbr_neurones_de_base_sortie + position)

            # Mettre à jour le nombre de sorties
            self.nbr_sorties += augmentation



    # Assurez-vous que cette partie est ajoutée dans votre classe Brain
    def mutate_activation_function(self):
        """Mutation pour changer une fonction d'activation aléatoire en hystérésis."""
        # if there is no hidden neurons
        if self.nbr_entrees + self.nbr_sorties >= len(self.activation_functions):
            return

        # Effectively change the activation function to the other
        neuron_index = random.randint(self.nbr_entrees + self.nbr_sorties, len(self.activation_functions) - 1)
        # Mutation : soit on remplace tanh par Hysteresis, soit on remplace Hysteresis par tanh
        if self.activation_functions[neuron_index] == np.tanh:
            self.activation_functions[neuron_index] = Hysteresis()
        elif isinstance(self.activation_functions[neuron_index], Hysteresis):
            self.activation_functions[neuron_index] = np.tanh



    def mutate_weights(self):
        """mute un poid alétoirement
        Mute suivant une distribution uniforme entre -variation_mutation et +variation_mutation"""
        #on choisi une lisaison non nulle (donc qui existe)
        liste = [np.nonzero(self.matrice_poids)[0], np.nonzero(self.matrice_poids)[1]] # != 0

        #si il existe des poids à muter
        if len(liste[0]) > 0:
            indice = random.randint(0,len(liste[0])-1)
            ligne , colonne = liste[0][indice] , liste[1][indice]
            
            variation = random.uniform(-variation_mutation_poids,variation_mutation_poids)
            self.matrice_poids[ligne][colonne] += variation
    

    def mutate_biais(self):
        """mute un biais alétoirement
        Mute suivant une distribution uniforme entre -variation_mutation et +variation_mutation"""

        indice = random.randint(self.nbr_entrees, len(self.biais_neurones)-1)

        variation = random.uniform(-variation_mutation_biais,variation_mutation_biais)
        self.biais_neurones[indice] += variation
    

    def delete_biais(self):
        """Met un biais aléatoirement à zéro.
        Sélectionne un biais parmi les neurones de sortie ou cachés."""
        
        # Choisir un indice aléatoire parmi les neurones de sortie et cachés
        indice = random.randint(self.nbr_entrees, len(self.biais_neurones) - 1)
        
        # Mettre le biais sélectionné à zéro
        self.biais_neurones[indice] = 0

    
    def replace_bias(self):
        """Remplace un biais aléatoirement.
        Sélectionne un biais parmi les neurones de sortie ou cachés."""
        
        # Choisir un indice aléatoire parmi les neurones de sortie et cachés
        indice = random.randint(self.nbr_entrees, len(self.biais_neurones) - 1)
        
        # Remplacer le biais sélectionné par une nouvelle valeur aléatoire
        self.biais_neurones[indice] = random.uniform(-0.1, 0.1)



    def mutate_weights_and_biais(self):
        self.mutate_weights()
        self.mutate_biais()

        
    def replace_weights(self):
        """remplace un poid alétoirement/on ajuste les valeurs en fonction du nombre d'entrée (classique dans la littérature)
        """
        #on choisi une lisaison non nulle (donc qui existe)
        liste = [np.nonzero(self.matrice_poids)[0], np.nonzero(self.matrice_poids)[1]] # != 0
        
        #si il existe des poids à muter
        if len(liste[0]) > 0:
            indice = random.randint(0,len(liste[0])-1)
            ligne , colonne = liste[0][indice] , liste[1][indice]

            #ajout dans la matrice
            self.matrice_poids[ligne][colonne] = random.uniform(-1, 1)


    def delete_weight(self):
        """Supprime un poid alétoirement
        """
        #on choisi une lisaison non nulle (donc qui existe)
        liste = [np.nonzero(self.matrice_poids)[0], np.nonzero(self.matrice_poids)[1]] # != 0
        
        #si il existe des poids à muter
        if len(liste[0]) > 0:
            indice = random.randint(0,len(liste[0])-1)
            ligne , colonne = liste[0][indice] , liste[1][indice]

            #ajout dans la matrice
            self.matrice_poids[ligne][colonne] = 0


    def add_weights(self):
        """Ajoute deux neurones non reliés aléatoirement et les relis avec un poid aléatoire dans [-1,1]"""
        #on choisi une lisaison nulle (donc qui n'existe pas)
        
        # # Définition du masque pour exclure une partie spécifique (carré 2 par 2 au milieu)
        # masque = np.ones_like(matrice, dtype=bool)
        # masque[2:4, 2:4] = False
        # # Recherche des indices non nuls en utilisant le masque
        # indices = np.argwhere(matrice * masque)
        
        liste = np.argwhere(self.matrice_poids[:, self.nbr_entrees:] == 0.0) # == 0 sans l'entrée
        liste[:,-1] += self.nbr_entrees # indice prennent en compte l'entrée pour pouvoir être utilisés facilement
        
        #si tous les neurones ne sont pas déja bien reliés
        if len(liste) > 0:
            indice = random.randint(0,len(liste)-1)
            ligne , colonne = liste[indice][0] , liste[indice][1]
            
            #ajout dans la matrice
            self.matrice_poids[ligne][colonne] = random.uniform(-1, 1)
            
            
    def add_neuron(self):
        """Ajoute un neurone dans une connexion existante. Reporte le poids sur la 2nd partie et met 1 sur la première"""
        
        #on choisi une lisaison non nulle (donc qui existe)
        liste = [np.nonzero(self.matrice_poids)[0], np.nonzero(self.matrice_poids)[1]] # != 0
       
        #si il existe des poids à utiliser pour créer des neurones
        if len(liste[0]) > 0:
            indice = random.randint(0,len(liste[0])-1)
            ligne , colonne = liste[0][indice] , liste[1][indice] 
            
            #ajout d'une nouvelle ligne et colonne de 0
            self.matrice_poids = np.pad(self.matrice_poids, ((0, 1), (0, 1)), mode='constant')
            
            #mise à jour de la matrice des poids
            valeur_ancienne_liaison = self.matrice_poids[ligne][colonne]
            self.matrice_poids[ligne][colonne] = 0   #ancienne liaison n'existe plus             
            
            #mettre à jour les poids : 1 avant le neurone et ancienne valeur après
            self.matrice_poids[ligne][-1] = 1 #mettre 1 avant le neurone
            self.matrice_poids[-1][colonne] = valeur_ancienne_liaison #mettre ancienne valeur après le neurone
            
            #ajouter neurone dans liste_neurones, biais et fonctions
            self.valeurs_neurones = np.append(self.valeurs_neurones, 0.0)
            self.biais_neurones = np.append(self.biais_neurones, random.random()*2-1)
            self.activation_functions = np.append(self.activation_functions, np.tanh)


    def delete_neuron(self):
        """Supprime un neurone aléatoirement et toutes ses connexions"""
        # On choisit un neurone intermédiaire aléatoirement
        if self.nbr_entrees + self.nbr_sorties >= len(self.valeurs_neurones):
            return  # Aucun neurone intermédiaire à supprimer

        # Sélectionner un neurone intermédiaire aléatoire
        neurone_a_supprimer = random.randint(self.nbr_entrees + self.nbr_sorties, len(self.valeurs_neurones) - 1)
        # Supprimer la ligne et la colonne correspondantes dans la matrice des poids
        self.matrice_poids = np.delete(self.matrice_poids, neurone_a_supprimer, axis=0)  # Supprimer la ligne
        self.matrice_poids = np.delete(self.matrice_poids, neurone_a_supprimer, axis=1)  # Supprimer la colonne

        # Supprimer la valeur du neurone et le biais correspondant
        self.valeurs_neurones = np.delete(self.valeurs_neurones, neurone_a_supprimer)
        self.biais_neurones = np.delete(self.biais_neurones, neurone_a_supprimer)
        self.activation_functions = np.delete(self.activation_functions, neurone_a_supprimer)



    def do_nothing(self):
        pass

        
    def mutate_brain(self):
        """Applique des mutations aléatoire sur le cerveau pour le nouveau-né
        IN : les différentes proba => la somme doit faire 1"""

        #on peut effectuer plusieurs mutations
        for n in range(0, nbr_mutations_brain):
            #dictionnaire des actions possibles
            actions = {
                #self.mutate_weights_and_biais: proba_modifier_poid_et_biais,
                self.mutate_weights: self.proba_modifier_poid,
                self.mutate_biais: self.proba_modifier_biais,
                self.mutate_activation_function: self.proba_modifier_fonction,
                self.replace_weights: self.proba_remplacer_poid,
                self.replace_bias: self.proba_remplacer_biais,
                self.add_weights: self.proba_creer_poid,
                self.add_neuron: self.proba_ajouter_neurone,
                self.delete_neuron: self.proba_supprimer_neurone,
                self.delete_weight: self.proba_supprimer_poid,
                self.delete_biais: self.proba_supprimer_biais,

                self.do_nothing: self.proba_rien_faire
            }
            
            # Tirage probabiliste de l'action à effectuer
            methods = list(actions.keys())
            probabilities = list(actions.values())
            selected_method = random.choices(methods, probabilities, k=1)[0]
            # Appeler la méthode sélectionnée
            selected_method()


    def mutate_weights_proba(self):
        self.proba_modifier_poid *= 1 + (random.uniform(-self.alpha, self.alpha))

    
    def mutate_biais_proba(self):
        self.proba_modifier_biais *= 1 + (random.uniform(-self.alpha, self.alpha))

    
    def mutate_function_proba(self):
        self.proba_modifier_fonction *= 1 + (random.uniform(-self.alpha, self.alpha))
    

    def replace_weights_proba(self):
        self.proba_remplacer_poid *= 1 + (random.uniform(-self.alpha, self.alpha))
    

    def replace_biais_proba(self):
        self.proba_remplacer_biais *= 1 + (random.uniform(-self.alpha, self.alpha))
        

    def add_weights_proba(self):
        self.proba_creer_poid *= 1 + (random.uniform(-self.alpha, self.alpha))


    def add_neuron_proba(self):
        self.proba_ajouter_neurone *= 1 + (random.uniform(-self.alpha, self.alpha))


    def delete_neuron_proba(self):
        self.proba_supprimer_neurone *= 1 + (random.uniform(-self.alpha, self.alpha))


    def delete_weight_proba(self):
        self.proba_supprimer_poid *= 1 + (random.uniform(-self.alpha, self.alpha))


    def delete_biais_proba(self):
        self.proba_supprimer_biais *= 1 + (random.uniform(-self.alpha, self.alpha))


    def do_nothing_proba(self):
        self.proba_rien_faire *= 1 + (random.uniform(-self.alpha, self.alpha))


    def mutate_probas(self):
        """Mutate the probas themselves (#meta mutation) according to a parameter alpha representing the speed of adaptability of the brain"""

        for n in range(0, nbr_modifications_proba):
            #dictionnaire des actions possibles, equiprobable pour l'instant
            actions = {
                self.mutate_weights_proba: 1,
                self.mutate_biais_proba: 1,
                self.mutate_function_proba: 1,
                self.replace_weights_proba: 1,
                self.replace_biais_proba: 1,
                self.add_weights_proba: 1,
                self.add_neuron_proba: 1,
                self.delete_neuron_proba: 1,
                self.delete_weight_proba: 1,
                self.delete_biais_proba: 1,
                self.do_nothing_proba: 1
            }
            
            # Tirage probabiliste de l'action à effectuer
            methods = list(actions.keys())
            probabilities = list(actions.values())
            selected_method = random.choices(methods, probabilities, k=1)[0]
            # Appeler la méthode sélectionnée
            selected_method()


    def mutate_alpha_proba(self):
        """Mutate alpha itself. The goal is for some individuals to be more adaptable than others"""
        self.alpha *= 1 + (random.uniform(-beta, beta)) #should vary slowly


    def mutate_alpha(self):
        """Mutate the alpha (#meta meta mutation) according to a parameter beta representing how much the alpha (adaptability) can change. It is fixed."""

        #dictionnaire des actions possibles, equiprobable pour l'instant
        actions = {
            self.mutate_alpha_proba: 0.7,
            self.do_nothing: 0.3
        }
        
        # Tirage probabiliste de l'action à effectuer
        methods = list(actions.keys())
        probabilities = list(actions.values())
        selected_method = random.choices(methods, probabilities, k=1)[0]
        # Appeler la méthode sélectionnée
        selected_method()



