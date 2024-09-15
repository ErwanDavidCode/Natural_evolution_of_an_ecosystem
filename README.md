# Natural evolution of an ecosystem
This python project of several months is a simulation of natural evolution allowing to see complex behaviors emerge. Individuals are driven by artificial neural networks and evolve by genetic mutations. They must learn to develop individual and group strategies to survive in this environment.

# Installation
There are a few libraries to install so that the simulation can start, nothing very large.
- Install the Python libraries
```sh
pip install -r requirements.txt
```

# Algorithm configuration
The parameters used to start the simulation can be modified in the main of the `parameters_V23.py` file. This file groups together all the parameters necessary for the simulation.

This project contains a lot of parameters. I only list here the most important ones which, I think, are the first to modify if modifications must be made:

## Launch a simulation
The parameters are modified in `parameters_V23.py`. The simulation is launched by launching `ecosystem_V23_huge_scale.py`.
The first block of code before "PARAMETERS" does not need to be modified to launch a simulation.
The "end_time" does not need to be modified as long as the following line does not replace the while:
```python
#while temps <= duree_simulation and len(self.liste_individus) and time.time() < time.mktime(end_time)> 0:
```
Thus, to stop the simulation,
Can be modified without risk:
- do Ctrl + C to stop the process, the event will be captured and the stop will generate the history and the video
- specify the desired "duration_simulation"
- have the condition "time.time() < time.mktime(end_time)> 0" in the while and specify the correct date.

```python
taille_carte = 1200 #specifies the size of the map. Individuals are originally 2 units long

nbr_individus_init = 100 #specifies the number of individuals when launching the simulation

nbr_plantes_init = 120*size_modification #Only change the numeric value of the number (here 120). Specifies the number of plants when launching the simulation
nbr_min_plant_init = 5*size_modification #Only change the numeric value of the number (here 5). Specifies the minimum number of plants before replanting seeds at a random location in the simulation


lvl_max_eat_scale = 4 #Specifies the number of different diets in the simulation. For example, 1 means that there are only omnivores, 2 that there are carnivores and herbivores...

# For the following, it is possible to remove an element from one of the dictionaries "liste_entrees_supplementaires_possibles_init", "liste_entrees_supplementaires_possibles_par_part_init" or "liste_sorties_supplementaires_possibles_init" and add it to the corresponding dictionary "init". Be careful, you must remove the key and the value and put them as is in the corresponding init dictionary. This allows you to add initial features to individuals.
liste_entrees_supplementaires_init = {} #must be complementary to list_of_additional_possible_entries_init
liste_entrees_supplementaires_par_part_init = {}
liste_sorties_supplementaires_init = {}
liste_entrees_supplementaires_possibles_init = {"energie" : 1, "regime" : 1, "is_giving_birth" : 1, "is_stomach_full" : 1, "oreille" : 3, "vie" : 1}
liste_entrees_supplementaires_possibles_par_part_init = {"know_size" : 1, "know_diet" : 1}
liste_sorties_supplementaires_possibles_init = {"attaque" : 1, "trophallaxy" : 1, "bouche" : 2, "creer_bb" : 1}

```
## View the results
In the Videos folder will be the video(s) and the graph from the simulation.

In the "Current Working code / data" folder will be the history of the individuals from the simulation.

It is possible to display the brain and characteristics of an individual via its ID.
To do this, you must:
- note its ID via the video
- enter the path of the video in "fichier_path" at the beginning of the `parameters_V23.py` file
- enter the ID of the individual in "characteristics_ID_individu"
- restart the simulation.
At this point an image and console information will appear.

Note: the simulation restarts at each failure if "start_again_until_alive_pop" is "True".
So, a video and a history are saved in copy in the Videos and data folder respectively if the simulation exceeds "time_to_save_video" iterations.

