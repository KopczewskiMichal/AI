import numpy as np
import pygad
# A
items = [
    {"name": "porcelana", "value": 250, "weight": 3},
    {"name": "obraz-pejzaż", "value": 300, "weight": 7},
    {"name": "obraz-portret", "value": 200, "weight": 6},
    {"name": "zegar", "value": 100, "weight": 7},
    {"name": "laptop", "value": 500, "weight": 5},
    {"name": "lampka nocna", "value": 70, "weight": 6},
    {"name": "radio", "value": 40, "weight": 2},
    {"name": "srebrne sztućce", "value": 100, "weight": 1},
    {"name": "figura z brązu", "value": 300, "weight": 10},
    {"name": "skórzana torebka", "value": 280, "weight": 3},
    {"name": "odkurzacz", "value": 300, "weight": 15}
]

# B
max_weight = 25

def fitness_func(model, solution, solution_idx):
    fitness = np.sum(solution * [item['value'] for item in items])
    weight = np.sum(solution * [item['weight'] for item in items])
    
    if weight > max_weight:
        return 0 # odrzucamy chromosomy ze zbyt wysoką wagą
    else:
        return fitness

fitness_function = fitness_func


gene_space = [0, 1]


sol_per_pop = 20
num_genes = len(items)

num_parents_mating = 10
num_generations = 100
keep_parents = 2


mutation_percent_genes = 10

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    parent_selection_type="sss",
    mutation_percent_genes=mutation_percent_genes,
    num_genes=num_genes,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    keep_parents=keep_parents,
    crossover_type="single_point", # ilośc punktów skrzyżowania
    mutation_type="random"
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

selected_items = [items[i] for i in range(len(solution)) if solution[i] == 1]
selected_names = [item['name'] for item in selected_items]
total_value = sum(item['value'] for item in selected_items)
total_weight = sum(item['weight'] for item in selected_items)

print("Najlepsze przedmioty: ", selected_names)
print("Ich wartość: ", total_value)
print("Waga: ", total_weight)

ga_instance.plot_fitness()



# Parameters of the best solution : [0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0.]
# Fitness value of the best solution = 1630.0
# Najlepsze przedmioty:  ['obraz-pejzaż', 'obraz-portret', 'laptop', 'srebrne sztućce', 'porcelana', 'skórzana torebka']
# Ich wartość:  1630
# Waga:  25
