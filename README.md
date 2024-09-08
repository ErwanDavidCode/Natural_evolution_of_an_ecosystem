# Natural evolution of an ecosystem
Ce projet python de plusieurs mois est une simulation de l'évolution naturelle permettant de voir emerger des comportements complexes. Les individus sont pilotés par des réseaux neuronaux artificiels et évoluent par mutations génétiques. Ils doivent apprendre à développer des stratégies individuelles et de groupe pour survivre dans cet environnement.

# Installation
Il y a quelques librairies à installer pour que la simulation puisse se lancer, rien de très volumineux.
- Installer les librairies Python
```sh
pip install -r requirements.txt
```

# Configuration de l'algorithme
Les paramètres utilisés pour lancer la simulation peuvent être modifiés dans le main du fichier `parameters_V23.py`. Ce fichier regrouppe tous les paramètres nécessaire à la simulation.

Ce projet contients énormément de paramètres. Je ne liste ici que les plus importants qui, je pense, sont les premiers à modifier si des modifications doivent être faites :


## Lancer une simulation
Les paramètres se modifient dans `parameters_V23.py`. La simulation se lance en lancant `ecosystem_V23_huge_scale.py`.
Le premier bloc de code avant "PARAMETRES" n'est pas à modifier pour lancer une simulation.
Le "end_time" n'est pas à modifier tans que la ligne suivante ne remplace pas le while: 
```python
#while temps <= duree_simulation and len(self.liste_individus) and time.time() < time.mktime(end_time)> 0:
```
Ainsi, pour arreter la simulation,  
Peuvent êtremodifiés sans risques :
- faire Ctrl + C pour arreter le processus, l'évenement sera capturé et l'arret génerera l'historique et la video
- spécifier le "duree_simulation" voulu
- avoir la condition "time.time() < time.mktime(end_time)> 0" dans le while et spécifier la bonne date.

```python
taille_carte = 1200 #spécifie la taille de la carte. Les individus font à l'origine 2 unités de long

nbr_individus_init = 100 #spécifie le nombre d'individu au lancement de la simulation

nbr_plantes_init = 120*size_modification #Ne modifier que la valeur numérique du nombre (ici 120). Spécifie le nombre de plantes au lancement de la simulation
nbr_min_plant_init = 5*size_modification #Ne modifier que la valeur numérique du nombre (ici 5). Spécifie le nombre de plantes minimum avant replante de graine à un endroit aléatoire de la simulation


lvl_max_eat_scale = 4 #Spécifie le nombre de diet différentes dans la simulation. Par exemple, 1 veut dire qu'il y a que des individus omnivores, 2 qu'il y a des carnivores et des herbivores ...

# Pour les suivants, il est possible d'enlever un élément d'un des dictionnaires "liste_entrees_supplementaires_possibles_init", "liste_entrees_supplementaires_possibles_par_part_init" ou "liste_sorties_supplementaires_possibles_init" et de l'ajouter dans le dictionnaire correspondant "init". Attention, il faut bien enlever la clés et la valeur et les mettre tels quels dans le dictionnaire init correspondant. Ca permet d'ajouter des features initiales aux individus.
liste_entrees_supplementaires_init = {} #doit etre complementaire a liste_entrees_supplementaires_possibles_init
liste_entrees_supplementaires_par_part_init = {}
liste_sorties_supplementaires_init = {}
liste_entrees_supplementaires_possibles_init = {"energie" : 1, "regime" : 1, "is_giving_birth" : 1, "is_stomach_full" : 1, "oreille" : 3, "vie" : 1}
liste_entrees_supplementaires_possibles_par_part_init = {"know_size" : 1, "know_diet" : 1}
liste_sorties_supplementaires_possibles_init = {"attaque" : 1, "trophallaxy" : 1, "bouche" : 2, "creer_bb" : 1}

```

## Visualiser les résultats
Dans le dossier Videos se trouvera la/les videos et le graphique issu de la simulation. 
Dans le dossier "Current Working code / data" se trouvera l'historique des individus issu de la simulation.

Il est possible d'afficher le cerveau et caractéristique d'un individu via son ID.
Pour ce faire, il faut : 
- noter son ID via la video
- inscrire le chemin de la video dans "fichier_path" au début du fichier  `parameters_V23.py`
- inscrire l'ID de l'individu dans "characteristics_ID_individu"
- relancer la simulation.
A ce moment là une image et des infomations consoles apparaitront. 

A noter : la simulation se relance à chaque echec si "start_again_until_alive_pop" est "True". 
Alors, une video et un historique est sauvegardé en copue dans le dossier Videos et data respectivement si la simulation dépasse "time_to_save_video" itérations.


