import matplotlib.pyplot as plt
import random

from aco import AntColony



def generate_random_coords(n, min_value=0, max_value=100):
    return [(random.randint(min_value, max_value), random.randint(min_value, max_value)) for _ in range(n)]


COORDS = tuple(generate_random_coords(10))


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2, 
                    pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
                    iterations=300)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


# plt.show()
plt.savefig("wykres3.png")

# 200 losowych node, 300 iteracji (wykres 1)
# 5 losowych node, 300 iteracji (wykres 2)
# 10 losowych node, 300 iteracji wykres 3