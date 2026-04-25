import shelve
import heapq

from parameters_V23 import *



class Observer:
    def __init__(self):
        with shelve.open(fichier_path) as db:
            self.historique_individus_a_observer = dict(db)  # Convert to dict for faster iteration


    def get_top_individuals(self, attribute, top_n=3):
        """Get the top n individuals based on the given attribute. If the attribute is not found, default to 0.
        It uses a queue to maintain order and keep the top n individuals."""
        heap = []
        for individu_id, individu in self.historique_individus_a_observer.items():
            attribute_value = getattr(individu.body, attribute)  # Default to 0 if attribute not found
            if len(heap) < top_n:
                heapq.heappush(heap, (attribute_value, individu_id))
            else:
                heapq.heappushpop(heap, (attribute_value, individu_id))
        
        return sorted(heap, reverse=True, key=lambda x: x[0])
    
    
    def get_bottom_individuals(self, attribute, top_n=3):
        """Get the bottom n individuals based on the given attribute. If the attribute is not found, default to 0.
        It uses a queue to maintain order and keep the bottom n individuals."""
        heap = []
        for individu_id, individu in self.historique_individus_a_observer.items():
            attribute_value = getattr(individu.body, attribute, 0)  # Default to 0 if attribute not found
            if len(heap) < top_n:
                heapq.heappush(heap, (-attribute_value, individu_id))  # Invert value to use max heap as min heap
            else:
                heapq.heappushpop(heap, (-attribute_value, individu_id))
        
        # Invert values back to positive for correct display
        return [( -count, individu_id ) for count, individu_id in sorted(heap, reverse=False, key=lambda x: -x[0])]



    def count_diet_types(self):
        diet_counts = {}
        for individu in self.historique_individus_a_observer.values():
            regime = individu.body.regime
            if regime not in diet_counts:
                diet_counts[regime] = 0
            diet_counts[regime] += 1
        return diet_counts
    

    def stat(self, top_n=3):

        self.best_killers = self.get_top_individuals('compteur_kill', top_n)
        self.best_trophallaxieurs = self.get_top_individuals('compteur_trophallaxie', top_n)
        self.best_plant_eaters = self.get_top_individuals('compteur_plant_eaten', top_n)
        self.best_meat_eaters = self.get_top_individuals('compteur_meat_eaten', top_n)
        self.nbr_part_vision = self.get_top_individuals('vision_nbr_parts', top_n)
        
        self.biggest_individuals = self.get_top_individuals('r_collision_box_individu', top_n)
        self.fastest_individuals = self.get_top_individuals('facteur_multiplicatif_deplacement', top_n)
        self.largest_vision_field = self.get_top_individuals('vision_rayon', top_n)
        self.largest_vision_angle = self.get_top_individuals('vision_demi_angle', top_n)
        self.largest_ears_field = self.get_top_individuals('ecoute_rayon', top_n)

        self.smallest_individuals = self.get_bottom_individuals('r_collision_box_individu', top_n)
        self.slowest_individuals = self.get_bottom_individuals('facteur_multiplicatif_deplacement', top_n)
        self.smallest_vision_field = self.get_bottom_individuals('vision_rayon', top_n)
        self.smallest_vision_angle = self.get_bottom_individuals('vision_demi_angle', top_n)
        self.smallest_ears_field = self.get_bottom_individuals('ecoute_rayon', top_n)

        diet_counts = self.count_diet_types()

        try:
            print("\n")
            print(f"####################### STATISTIQUES GLOBALES #######################     - #observer ")
            print(f"Top bigger ID:               {max(self.historique_individus_a_observer.keys(), key=int)}")
            print(f"Top killers:                 {',             '.join([f'ID: {individu_id}     Count:  {count}' for count, individu_id in self.best_killers])}")
            print(f"Top trophallaxieurs:         {',             '.join([f'ID: {individu_id}     Count:  {count}' for count, individu_id in self.best_trophallaxieurs])}")
            print(f"Top plant eaters:            {',             '.join([f'ID: {individu_id}     Count:  {count}' for count, individu_id in self.best_plant_eaters])}")
            print(f"Top meat eaters:             {',             '.join([f'ID: {individu_id}     Count:  {count}' for count, individu_id in self.best_meat_eaters])}")
            print(f"Most part vision:            {',             '.join([f'ID: {individu_id}     Count:  {count}' for count, individu_id in self.nbr_part_vision])}")
            print("\n")
            print(f"Biggest individuals:         {',             '.join([f'ID: {individu_id}     Size:   {round(count,2)}' for count, individu_id in self.biggest_individuals])}")
            print(f"Fastest individuals:         {',             '.join([f'ID: {individu_id}     Speed:  {round(count,2)}' for count, individu_id in self.fastest_individuals])}")
            print(f"Largest vision field:        {',             '.join([f'ID: {individu_id}     Radius: {round(count,2)}' for count, individu_id in self.largest_vision_field])}")
            print(f"Largest vision angle:        {',             '.join([f'ID: {individu_id}     Angle:  {round(count,2)}' for count, individu_id in self.largest_vision_angle])}")
            print(f"Largest ears field:          {',             '.join([f'ID: {individu_id}     Radius: {round(count,2)}' for count, individu_id in self.largest_ears_field])}")
            print("\n")
            print(f"Smallest  individuals:       {',             '.join([f'ID: {individu_id}     Size:   {round(count,2)}' for count, individu_id in self.smallest_individuals])}")
            print(f"Slowest  individuals:        {',             '.join([f'ID: {individu_id}     Speed:  {round(count,2)}' for count, individu_id in self.slowest_individuals])}")
            print(f"Smallest  vision field:      {',             '.join([f'ID: {individu_id}     Radius: {round(count,2)}' for count, individu_id in self.smallest_vision_field])}")
            print(f"Smallest  vision angle:      {',             '.join([f'ID: {individu_id}     Angle:  {round(count,2)}' for count, individu_id in self.smallest_vision_angle])}")
            print(f"Smallest  ears field:        {',             '.join([f'ID: {individu_id}     Angle:  {round(count,2)}' for count, individu_id in self.smallest_ears_field])}")
            
            print("\nDistribution des régimes alimentaires:")
            for regime, count in sorted(diet_counts.items()):
                print(f"Régime {regime}: {count} individu(s)")

        except:
            print(f"Ce fichier: '{fichier_path}' n'existe pas")


