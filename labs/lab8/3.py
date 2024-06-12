import math
import numpy as np
import pygad
import matplotlib.pyplot as plt

def endurance(x, y, z, u, v, w):
 return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)


gene_space = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return endurance(x, y, z, u, v, w)

sol_per_pop = 10
num_genes = 6

initial_population = np.random.rand(sol_per_pop, num_genes)

ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=4,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=6,  # ilość dostępnych metali na wejściu
                       gene_type=float,
                       gene_space=gene_space,
                       initial_population=initial_population,
                       mutation_percent_genes=17)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution =", solution_fitness)
print("Parameters of the best solution : ", solution)

fitness_values = ga_instance.best_solutions_fitness
plt.plot(fitness_values)
plt.xlabel('Generacja')
plt.ylabel('Fitness')
plt.grid()
plt.savefig("plot3.png")


# Fitness value of the best solution = 2.8414709848078967
# Parameters of the best solution :  [0. 0. 1. 1. 0. 1.]